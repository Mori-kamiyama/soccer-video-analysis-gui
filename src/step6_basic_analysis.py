"""
Step 6: 基本分析を行う。

出力:
  outputs/player_distances.csv
  outputs/team_centroids.csv
  outputs/trajectory.png
  outputs/heatmap.png
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pipeline_common import PITCH_LENGTH_M, PITCH_WIDTH_M, distance, read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="移動距離・軌跡・ヒートマップ・チーム重心を出す")
    parser.add_argument("--input-csv", default="outputs/player_positions.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--pitch-length", type=float, default=PITCH_LENGTH_M)
    parser.add_argument("--pitch-width", type=float, default=PITCH_WIDTH_M)
    parser.add_argument("--image-width", type=int, default=1400)
    return parser.parse_args()


def to_pixel(x: float, y: float, pitch_length: float, pitch_width: float, image_width: int) -> tuple[int, int]:
    margin = 50
    image_height = int(image_width * pitch_width / pitch_length)
    px = margin + int(x / pitch_length * image_width)
    py = margin + int(y / pitch_width * image_height)
    return px, py


def pitch_canvas(pitch_length: float, pitch_width: float, image_width: int) -> np.ndarray:
    margin = 50
    image_height = int(image_width * pitch_width / pitch_length)
    canvas = np.full((image_height + margin * 2, image_width + margin * 2, 3), 245, dtype=np.uint8)
    top_left = (margin, margin)
    bottom_right = (margin + image_width, margin + image_height)
    cv2.rectangle(canvas, top_left, bottom_right, (40, 150, 60), thickness=-1)
    cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), thickness=2)
    center_x = margin + image_width // 2
    cv2.line(canvas, (center_x, margin), (center_x, margin + image_height), (255, 255, 255), 2)
    cv2.circle(canvas, (center_x, margin + image_height // 2), int(image_height * 0.14), (255, 255, 255), 2)
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    rows = read_csv(Path(args.input_csv))
    # 範囲外（in_pitch=0）の発散点は可視化・分析から除外する。列が無い場合は全採用。
    rows = [row for row in rows if row.get("in_pitch", "1") != "0"]
    rows.sort(key=lambda row: (int(row["player_id"]), float(row["time"])))

    by_player: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_frame_team: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_player[row["player_id"]].append(row)
        by_frame_team[(row["frame"], row.get("team", ""))].append(row)

    distance_rows = []
    for player_id, player_rows in by_player.items():
        total = 0.0
        max_speed = 0.0
        for previous, current in zip(player_rows, player_rows[1:]):
            previous_point = (float(previous["x_pitch"]), float(previous["y_pitch"]))
            current_point = (float(current["x_pitch"]), float(current["y_pitch"]))
            step = distance(previous_point, current_point)
            dt = max(1e-6, float(current["time"]) - float(previous["time"]))
            total += step
            max_speed = max(max_speed, step / dt)
        distance_rows.append(
            {
                "player_id": player_id,
                "team": player_rows[0].get("team", ""),
                "samples": len(player_rows),
                "distance_m": round(total, 2),
                "max_speed_mps": round(max_speed, 2),
            }
        )

    centroid_rows = []
    for (frame, team), group in sorted(by_frame_team.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        xs = [float(row["x_pitch"]) for row in group]
        ys = [float(row["y_pitch"]) for row in group]
        centroid_rows.append(
            {
                "frame": frame,
                "time": group[0]["time"],
                "team": team,
                "players": len(group),
                "centroid_x": round(sum(xs) / len(xs), 3),
                "centroid_y": round(sum(ys) / len(ys), 3),
            }
        )

    write_csv(output_dir / "player_distances.csv", distance_rows, ["player_id", "team", "samples", "distance_m", "max_speed_mps"])
    write_csv(output_dir / "team_centroids.csv", centroid_rows, ["frame", "time", "team", "players", "centroid_x", "centroid_y"])

    trajectory = pitch_canvas(args.pitch_length, args.pitch_width, args.image_width)
    colors = [(30, 80, 240), (240, 80, 30), (160, 30, 180), (30, 180, 180), (60, 60, 60)]
    for player_index, (player_id, player_rows) in enumerate(by_player.items()):
        color = colors[player_index % len(colors)]
        points = [
            to_pixel(float(row["x_pitch"]), float(row["y_pitch"]), args.pitch_length, args.pitch_width, args.image_width)
            for row in player_rows
        ]
        for a, b in zip(points, points[1:]):
            cv2.line(trajectory, a, b, color, 2)
        if points:
            cv2.circle(trajectory, points[-1], 4, color, -1)
            cv2.putText(trajectory, player_id, points[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(output_dir / "trajectory.png"), trajectory)

    heat_base = pitch_canvas(args.pitch_length, args.pitch_width, args.image_width)
    heat = np.zeros(heat_base.shape[:2], dtype=np.float32)
    for row in rows:
        px, py = to_pixel(float(row["x_pitch"]), float(row["y_pitch"]), args.pitch_length, args.pitch_width, args.image_width)
        if 0 <= py < heat.shape[0] and 0 <= px < heat.shape[1]:
            heat[py, px] += 1.0
    kernel_size = max(31, int(args.image_width * 0.035) | 1)
    heat = cv2.GaussianBlur(heat, (kernel_size, kernel_size), 0)
    if heat.max() > 0:
        heat = heat / heat.max()
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (heat > 0.03).astype(np.uint8)[:, :, None]
    heatmap = np.where(mask, cv2.addWeighted(heat_base, 0.55, heat_color, 0.45, 0), heat_base)
    cv2.imwrite(str(output_dir / "heatmap.png"), heatmap)

    print(f"保存: {output_dir / 'player_distances.csv'}")
    print(f"保存: {output_dir / 'team_centroids.csv'}")
    print(f"保存: {output_dir / 'trajectory.png'}")
    print(f"保存: {output_dir / 'heatmap.png'}")


if __name__ == "__main__":
    main()
