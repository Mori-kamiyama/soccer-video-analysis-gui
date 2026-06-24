"""
Step 4: 検出結果に仮の選手IDを付ける。

前フレームの近い検出と対応付ける、軽量なSORT風の最近傍トラッカー。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import distance, read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="検出CSVから仮player_id付きCSVを作る")
    parser.add_argument("--input-csv", default="outputs/detections_clean.csv")
    parser.add_argument("--output-csv", default="outputs/tracks.csv")
    parser.add_argument("--max-distance", type=float, default=90.0)
    parser.add_argument("--max-missed", type=int, default=4)
    return parser.parse_args()


def detection_point(row: dict[str, str]) -> tuple[float, float]:
    return (float(row["foot_x"]), float(row["foot_y"]))


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.input_csv))
    rows.sort(key=lambda row: (int(row["frame"]), int(row["detection_id"])))

    tracks: dict[int, dict[str, object]] = {}
    next_id = 1
    output_rows = []
    frame_values = sorted({int(row["frame"]) for row in rows})

    for frame in frame_values:
        frame_rows = [row for row in rows if int(row["frame"]) == frame]
        assigned_tracks: set[int] = set()
        for row in frame_rows:
            point = detection_point(row)
            best_id = None
            best_distance = args.max_distance
            for track_id, track in tracks.items():
                if track_id in assigned_tracks:
                    continue
                if frame - int(track["last_frame"]) > args.max_missed + 1:
                    continue
                candidate_distance = distance(point, track["point"])
                if candidate_distance < best_distance:
                    best_id = track_id
                    best_distance = candidate_distance
            if best_id is None:
                best_id = next_id
                next_id += 1
            assigned_tracks.add(best_id)
            tracks[best_id] = {"point": point, "last_frame": frame}
            row = dict(row)
            row["player_id"] = best_id
            row["track_distance"] = round(best_distance, 2) if best_distance < args.max_distance else ""
            output_rows.append(row)

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    write_csv(Path(args.output_csv), output_rows, fieldnames)
    print(f"保存: {args.output_csv} ({len(output_rows)} rows, players={next_id - 1})")


if __name__ == "__main__":
    main()
