"""
Step 5: 画像座標をピッチ座標に変換する。

まずテンプレートを作る:
  uv run python step5_transform_to_pitch.py --write-template

pitch_points.json の image 点を、画像上の左上/右上/右下/左下の順で埋めてから実行する。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pipeline_common import PITCH_LENGTH_M, PITCH_WIDTH_M, read_csv, read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="選手の足元画像座標をピッチ座標に変換する")
    parser.add_argument("--input-csv", default="outputs/tracks.csv")
    parser.add_argument("--output-csv", default="outputs/player_positions.csv")
    parser.add_argument("--points", default="pitch_points.json")
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--pitch-length", type=float, default=PITCH_LENGTH_M)
    parser.add_argument("--pitch-width", type=float, default=PITCH_WIDTH_M)
    return parser.parse_args()


def write_template(path: Path, pitch_length: float, pitch_width: float) -> None:
    template = {
        "note": "imageは画像上のピッチ四隅を左上,右上,右下,左下の順で入れる",
        "image": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
        "pitch": [[0, 0], [pitch_length, 0], [pitch_length, pitch_width], [0, pitch_width]],
    }
    write_json(path, template)
    print(f"テンプレート作成: {path}")


def main() -> None:
    args = parse_args()
    points_path = Path(args.points)
    if args.write_template:
        write_template(points_path, args.pitch_length, args.pitch_width)
        return

    rows = read_csv(Path(args.input_csv))
    points = read_json(points_path)
    src = np.array(points["image"], dtype=np.float32)
    dst = np.array(points["pitch"], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)

    # ピッチ範囲（少し余裕を持たせる）。範囲外は消失線より上などで発散した点。
    margin = 5.0
    x_min, x_max = -margin, args.pitch_length + margin
    y_min, y_max = -margin, args.pitch_width + margin

    output_rows = []
    for row in rows:
        foot = np.array([[[float(row["foot_x"]), float(row["foot_y"])]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(foot, matrix)[0][0]
        x_pitch = float(transformed[0])
        y_pitch = float(transformed[1])
        in_pitch = 1 if (x_min <= x_pitch <= x_max and y_min <= y_pitch <= y_max) else 0
        row = dict(row)
        row["x_pitch"] = round(x_pitch, 3)
        row["y_pitch"] = round(y_pitch, 3)
        row["in_pitch"] = in_pitch  # 0=台形の外/消失線より上で発散した点（可視化から除外する）
        row["team"] = row.get("team_hint", "")
        output_rows.append(row)

    fieldnames = list(output_rows[0].keys()) if output_rows else []
    write_csv(Path(args.output_csv), output_rows, fieldnames)
    print(f"保存: {args.output_csv} ({len(output_rows)} rows)")


if __name__ == "__main__":
    main()
