"""
Step 3: 検出結果CSVを後続処理向けに整形する。

Step 2の出力を読み、座標の破綻を除外して frame/time 順に並べる。
team_hint が空の場合は画面の上下で仮に A/B を割り当てる。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import read_csv, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="検出CSVを整形して保存する")
    parser.add_argument("--input-csv", default="outputs/detections.csv")
    parser.add_argument("--output-csv", default="outputs/detections_clean.csv")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.input_csv))
    clean_rows = []
    for row in rows:
        x1, y1, x2, y2 = map(float, (row["x1"], row["y1"], row["x2"], row["y2"]))
        confidence = float(row["confidence"])
        if confidence < args.min_confidence:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        row = dict(row)
        if not row.get("team_hint"):
            row["team_hint"] = "A" if float(row["center_y"]) < 540 else "B"
        clean_rows.append(row)

    clean_rows.sort(key=lambda item: (int(item["frame"]), int(item["detection_id"])))
    fieldnames = list(clean_rows[0].keys()) if clean_rows else list(rows[0].keys()) if rows else []
    write_csv(Path(args.output_csv), clean_rows, fieldnames)
    print(f"保存: {args.output_csv} ({len(clean_rows)} rows)")


if __name__ == "__main__":
    main()
