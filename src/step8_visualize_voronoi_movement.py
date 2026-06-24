"""
Step 8: ボロノイ図と移動量グラフを作成する。

出力:
  outputs/all_analysis/voronoi.png
  outputs/all_analysis/movement_graph.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi

from pipeline_common import PITCH_LENGTH_M, PITCH_WIDTH_M

matplotlib.rcParams["font.family"] = "Hiragino Sans"
matplotlib.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


MARGIN = 50


def to_pixel(x: float, y: float, pitch_length: float, pitch_width: float, image_width: int) -> tuple[int, int]:
    image_height = int(image_width * pitch_width / pitch_length)
    px = MARGIN + int(x / pitch_length * image_width)
    py = MARGIN + int(y / pitch_width * image_height)
    return px, py


def pitch_canvas(pitch_length: float, pitch_width: float, image_width: int) -> np.ndarray:
    image_height = int(image_width * pitch_width / pitch_length)
    canvas = np.full((image_height + MARGIN * 2, image_width + MARGIN * 2, 3), 245, dtype=np.uint8)
    tl = (MARGIN, MARGIN)
    br = (MARGIN + image_width, MARGIN + image_height)
    cv2.rectangle(canvas, tl, br, (40, 150, 60), thickness=-1)
    cv2.rectangle(canvas, tl, br, (255, 255, 255), thickness=2)
    cx = MARGIN + image_width // 2
    cv2.line(canvas, (cx, MARGIN), (cx, MARGIN + image_height), (255, 255, 255), 2)
    cv2.circle(canvas, (cx, MARGIN + image_height // 2), int(image_height * 0.14), (255, 255, 255), 2)
    return canvas


def create_voronoi_diagram(
    positions_csv: Path,
    pitch_length: float,
    pitch_width: float,
    image_width: int,
) -> np.ndarray:
    """各選手の位置からボロノイ図（支配領域）を描く。"""
    rows = read_csv(positions_csv)
    if not rows:
        return pitch_canvas(pitch_length, pitch_width, image_width)

    # in_pitch==1 のみ
    rows = [r for r in rows if r.get("in_pitch", "1") != "0"]

    # 選手数が最も多いフレームを選ぶ
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[r["frame"]].append(r)

    best_frame = max(by_frame, key=lambda f: len(by_frame[f]))
    frame_rows = by_frame[best_frame]

    if len(frame_rows) < 3:
        return pitch_canvas(pitch_length, pitch_width, image_width)

    image_height = int(image_width * pitch_width / pitch_length)
    canvas = pitch_canvas(pitch_length, pitch_width, image_width)

    # 選手のピッチ座標を取得
    points = []
    teams = []
    pids = []
    for r in frame_rows:
        xp, yp = float(r["x_pitch"]), float(r["y_pitch"])
        if 0 <= xp <= pitch_length and 0 <= yp <= pitch_width:
            points.append((xp, yp))
            teams.append(r.get("team", "B"))
            pids.append(r.get("player_id", ""))

    if len(points) < 3:
        return canvas

    pts = np.array(points)

    # ピクセルごとに最近接選手を見つけてカラーマップ
    h, w = image_height, image_width
    yy, xx = np.mgrid[0:h, 0:w]
    # ピクセル→ピッチ座標
    pitch_x = xx.astype(np.float64) / w * pitch_length
    pitch_y = yy.astype(np.float64) / h * pitch_width

    # 距離計算して最近接選手を選ぶ
    nearest = np.zeros((h, w), dtype=np.int32)
    min_dist = np.full((h, w), np.inf)
    for i, (px, py) in enumerate(points):
        d = (pitch_x - px) ** 2 + (pitch_y - py) ** 2
        mask = d < min_dist
        min_dist[mask] = d[mask]
        nearest[mask] = i

    # チーム色: 半透明でピッチ上に重ねる
    palette_b = np.array([180, 100, 60], dtype=np.uint8)   # 青系 (BGR)
    palette_a = np.array([60, 100, 220], dtype=np.uint8)   # 赤系 (BGR)
    # 選手ごとに微妙に色を変える
    np.random.seed(42)
    player_colors = []
    for i, t in enumerate(teams):
        base = palette_a if t == "A" else palette_b
        jitter = np.random.randint(-30, 30, 3).astype(np.int16)
        c = np.clip(base.astype(np.int16) + jitter, 0, 255).astype(np.uint8)
        player_colors.append(c)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(player_colors):
        mask = nearest == i
        overlay[mask] = c

    # ピッチ領域だけに重ねる
    pitch_region = canvas[MARGIN:MARGIN + h, MARGIN:MARGIN + w]
    blended = cv2.addWeighted(pitch_region, 0.45, overlay, 0.55, 0)
    canvas[MARGIN:MARGIN + h, MARGIN:MARGIN + w] = blended

    # ボロノイ境界線を描く
    mirror_pts = np.copy(pts)
    # 境界用に4辺のミラーポイントを追加
    mirror_extra = []
    for px, py in points:
        mirror_extra.append((-px, py))
        mirror_extra.append((2 * pitch_length - px, py))
        mirror_extra.append((px, -py))
        mirror_extra.append((px, 2 * pitch_width - py))

    all_pts = np.vstack([pts, np.array(mirror_extra)])
    try:
        vor = Voronoi(all_pts)
        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue
            v0 = vor.vertices[ridge[0]]
            v1 = vor.vertices[ridge[1]]
            # ピッチ範囲内の辺だけ
            if (v0[0] < -5 or v0[0] > pitch_length + 5 or
                v1[0] < -5 or v1[0] > pitch_length + 5):
                continue
            px0 = MARGIN + int(v0[0] / pitch_length * image_width)
            py0 = MARGIN + int(v0[1] / pitch_width * image_height)
            px1 = MARGIN + int(v1[0] / pitch_length * image_width)
            py1 = MARGIN + int(v1[1] / pitch_width * image_height)
            cv2.line(canvas, (px0, py0), (px1, py1), (255, 255, 255), 1)
    except Exception:
        pass

    # 選手位置をマーク
    for i, (xp, yp) in enumerate(points):
        px, py = to_pixel(xp, yp, pitch_length, pitch_width, image_width)
        cv2.circle(canvas, (px, py), 7, (255, 255, 255), -1)
        cv2.circle(canvas, (px, py), 7, (0, 0, 0), 2)
        label = pids[i] if len(pids[i]) <= 3 else ""
        if label:
            cv2.putText(canvas, label, (px - 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # フレーム情報
    cv2.putText(canvas, f"Frame {best_frame} ({len(frame_rows)} players)",
                (MARGIN + 10, MARGIN - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    return canvas


def create_movement_graph(distances_csv: Path, top_n: int = 20) -> None:
    """上位選手の移動量を横棒グラフで描画し、PNGに保存。"""
    distances = read_csv(distances_csv)
    if not distances:
        return

    # samples >= 10 かつ distance > 0 の選手だけ
    valid = []
    for row in distances:
        try:
            d = float(row.get("distance_m", 0))
            s = int(row.get("samples", 0))
            speed = float(row.get("max_speed_mps", 0))
            if s >= 10 and d > 0:
                valid.append({
                    "player_id": row["player_id"],
                    "team": row.get("team", "?"),
                    "distance_m": d,
                    "samples": s,
                    "max_speed_mps": speed,
                })
        except ValueError:
            pass

    valid.sort(key=lambda x: x["distance_m"], reverse=True)
    top = valid[:top_n]
    top.reverse()  # 一番多い選手が上に来るように反転

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.4)))
    fig.patch.set_facecolor("#f6f7f4")
    ax.set_facecolor("#f6f7f4")

    labels = [f"#{p['player_id']}" for p in top]
    values = [p["distance_m"] for p in top]
    colors = ["#e74c3c" if p["team"] == "A" else "#2980b9" for p in top]

    bars = ax.barh(labels, values, color=colors, edgecolor="#20251f", linewidth=0.5, height=0.7)

    for bar, p in zip(bars, top):
        w = bar.get_width()
        ax.text(w + 3, bar.get_y() + bar.get_height() / 2,
                f"{p['distance_m']:.0f}m  (max {p['max_speed_mps']:.1f}m/s)",
                ha="left", va="center", fontsize=9, color="#333")

    ax.set_xlabel("移動距離 (m)", fontsize=12)
    ax.set_title(f"選手別移動量 TOP{top_n}", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    max_val = max(values) if values else 100
    ax.set_xlim(0, max_val * 1.25)

    plt.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ボロノイ図と移動量グラフを作成")
    parser.add_argument("--positions-csv", default="outputs/player_positions_all.csv")
    parser.add_argument("--distances-csv", default="outputs/all_analysis/player_distances.csv")
    parser.add_argument("--output-dir", default="outputs/all_analysis")
    parser.add_argument("--pitch-length", type=float, default=PITCH_LENGTH_M)
    parser.add_argument("--pitch-width", type=float, default=PITCH_WIDTH_M)
    parser.add_argument("--image-width", type=int, default=1400)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("ボロノイ図を作成中...")
    voronoi_img = create_voronoi_diagram(
        Path(args.positions_csv),
        args.pitch_length,
        args.pitch_width,
        args.image_width,
    )
    cv2.imwrite(str(output_dir / "voronoi.png"), voronoi_img)
    print(f"保存: {output_dir / 'voronoi.png'}")

    print("移動量グラフを作成中...")
    fig = create_movement_graph(Path(args.distances_csv), top_n=args.top_n)
    if fig:
        fig.savefig(str(output_dir / "movement_graph.png"), dpi=150, bbox_inches="tight",
                    facecolor="#f6f7f4")
        plt.close(fig)
        print(f"保存: {output_dir / 'movement_graph.png'}")
    else:
        print("移動量データなし")


if __name__ == "__main__":
    main()
