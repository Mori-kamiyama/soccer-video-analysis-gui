"""
Colab(SAM2)の出力zipを解凍したフォルダを取り込み、GUIで開けるセグメントにする。

Colab出力: <dir>/frames/00000.jpg... , tracks.csv(pitch空), meta.json
ここで: 足元をピッチ座標(m)へ変換し、フレームを f_00000.jpg にリネーム配置、
        outputs/segments/<name>/ を作る。

使い方:
  .venv/bin/python ingest_sam2.py --name seg_sam_105 --dir ~/Downloads/seg_sam_105
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

POINTS = Path("pitch_points.json")


def load_homography():
    if not POINTS.exists():
        return None
    pts = json.loads(POINTS.read_text())
    src = np.array(pts["image"], dtype=np.float32)
    dst = np.array(pts["pitch"], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_pitch(matrix, fx, fy):
    if matrix is None:
        return "", "", 0
    p = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype=np.float32), matrix)[0][0]
    x, y = float(p[0]), float(p[1])
    return round(x, 3), round(y, 3), (1 if -5 <= x <= 110 and -5 <= y <= 73 else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True, help="解凍したColab出力フォルダ")
    args = ap.parse_args()

    src = Path(args.dir).expanduser()
    out = Path("outputs/segments") / args.name
    (out / "frames").mkdir(parents=True, exist_ok=True)

    # メタ
    meta = json.loads((src / "meta.json").read_text())
    meta["name"] = args.name
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # フレームを f_00000.jpg にリネームコピー
    n = 0
    for p in sorted((src / "frames").glob("*.jpg")):
        idx = int(p.stem)
        shutil.copy(p, out / "frames" / f"f_{idx:05d}.jpg")
        n += 1

    # tracks.csv にピッチ座標を付与
    matrix = load_homography()
    rows = []
    with (src / "tracks.csv").open() as f:
        for r in csv.DictReader(f):
            fx, fy = float(r["foot_x"]), float(r["foot_y"])
            xp, yp, inp = to_pitch(matrix, fx, fy)
            r["x_pitch"], r["y_pitch"], r["in_pitch"] = xp, yp, inp
            rows.append(r)
    fields = ["frame", "track_id", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
              "x_pitch", "y_pitch", "in_pitch", "conf"]
    with (out / "tracks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"取り込み完了: {out}  フレーム{n}枚  トラック{len(rows)}行")
    print(f"GUIで開く:  ./run_gui.sh --name {args.name}")


if __name__ == "__main__":
    main()
