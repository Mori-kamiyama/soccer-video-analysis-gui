"""
旧 frames/(2fps・0秒開始) の frame番号で打たれた ball_keyframes.csv を、
frames_ball/(24fps・105秒開始) の frame番号へ変換する。

座標(image_x,image_y)はどちらも1920系なので不変。frame番号だけ変換:
  時刻 t = old_frame / OLD_FPS                  (旧は0秒開始)
  new_frame = round((t - BALL_START) * BALL_FPS)  (frames_ballの通し番号)
→ OLD_FPS=2, BALL_FPS=24, BALL_START=105 のとき new = old*12 - 2520

ball_positions.csv も frames_ball の密度(24fps)で補間し直して再生成する。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

OLD_FPS = 2.0
BALL_FPS = 24.0
BALL_START = 105.0
N_BALL_FRAMES = 2880  # frames_ball の枚数

KEYFRAMES_CSV = Path("outputs/ball_keyframes.csv")
POSITIONS_CSV = Path("outputs/ball_positions.csv")
POINTS_PATH = Path("pitch_points.json")


def old_to_new(old_frame: int) -> int:
    t = old_frame / OLD_FPS
    return round((t - BALL_START) * BALL_FPS)


def load_homography():
    if not POINTS_PATH.exists():
        return None
    import cv2
    import numpy as np
    d = json.loads(POINTS_PATH.read_text())
    img = np.array(d["image"], dtype="float32")
    pitch = np.array(d["pitch"], dtype="float32")
    return cv2.getPerspectiveTransform(img, pitch)


def to_pitch(H, x, y):
    if H is None:
        return None, None
    import cv2
    import numpy as np
    out = cv2.perspectiveTransform(np.array([[[x, y]]], dtype="float32"), H)[0][0]
    return round(float(out[0]), 2), round(float(out[1]), 2)


def main():
    rows = list(csv.DictReader(KEYFRAMES_CSV.open()))
    keys: dict[int, tuple] = {}
    skipped = []
    for r in rows:
        new_f = old_to_new(int(r["frame"]))
        if not (0 <= new_f < N_BALL_FRAMES):
            skipped.append((r["frame"], new_f))
            continue
        if r["state"] == "none":
            keys[new_f] = ("none",)
        else:
            keys[new_f] = ("pos", float(r["image_x"]), float(r["image_y"]))

    # keyframes 書き戻し
    with KEYFRAMES_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "state", "image_x", "image_y"])
        for fr in sorted(keys):
            k = keys[fr]
            if k[0] == "none":
                w.writerow([fr, "none", "", ""])
            else:
                w.writerow([fr, "pos", round(k[1], 1), round(k[2], 1)])

    # positions 再生成（frames_ball の全フレームを線形補間）
    H = load_homography()
    sorted_keys = sorted(keys)

    def ball_at(fr):
        k = keys.get(fr)
        if k is not None:
            return None if k[0] == "none" else (k[1], k[2], "key")
        prev = next((f for f in reversed(sorted_keys) if f < fr), None)
        nxt = next((f for f in sorted_keys if f > fr), None)
        if prev is None or nxt is None:
            return None
        kp, kn = keys[prev], keys[nxt]
        if kp[0] != "pos" or kn[0] != "pos":
            return None
        s = (fr - prev) / (nxt - prev)
        return (kp[1] + (kn[1] - kp[1]) * s, kp[2] + (kn[2] - kp[2]) * s, "interp")

    pos_rows = []
    for fr in range(N_BALL_FRAMES):
        res = ball_at(fr)
        if res is None:
            continue
        x, y, kind = res
        px, py = to_pitch(H, x, y)
        pos_rows.append([fr, round(x, 1), round(y, 1),
                         "" if px is None else px, "" if py is None else py,
                         1 if kind == "key" else 0])
    with POSITIONS_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "image_x", "image_y", "pitch_x", "pitch_y", "is_key"])
        w.writerows(pos_rows)

    print(f"変換完了: キーフレーム {len(keys)}個（新frame範囲 "
          f"{min(keys) if keys else '-'}〜{max(keys) if keys else '-'}）")
    print(f"  positions: {len(pos_rows)}フレーム展開")
    if skipped:
        print(f"  範囲外でスキップ: {skipped}")


if __name__ == "__main__":
    main()
