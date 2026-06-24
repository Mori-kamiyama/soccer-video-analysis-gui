"""
ボックスが1つも無い（手作業で飛ばした）フレームだけを YOLO で埋める。

注釈済み（緑/黒の選手ボックスがある）フレームには一切触れない。
空フレームにだけ YOLO(COCO person) 検出を追加し、kNN色モデルで緑/黒を自動判定して
box_corrections.csv に 'add' 行として追記する。

使い方:
  .venv/bin/python fill_empty_frames.py --start 212 --end 1311 --skip 955-1010
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

import photo_annotator as pa
import visualize_offside as v
import batch_redetect as br


def parse_skips(skip_args):
    out = []
    for s in skip_args or []:
        a, _, b = s.partition("-")
        out.append((int(a), int(b or a)))
    return out


def in_skip(n, skips):
    return any(a <= n <= b for a, b in skips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=212)
    ap.add_argument("--end", type=int, default=1311)
    ap.add_argument("--skip", action="append", default=["955-1010"])
    ap.add_argument("--conf", type=float, default=pa.YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=pa.YOLO_IMGSZ)
    args = ap.parse_args()
    skips = parse_skips(args.skip)

    boxes = v.reconstruct_boxes()

    # 空フレーム（緑/黒の選手ボックスが0）を抽出
    empty = []
    for n in range(args.start, args.end + 1):
        if in_skip(n, skips):
            continue
        if not (pa.FRAMES_DIR / f"frame_{n:06d}.jpg").exists():
            continue
        fr = n - 1
        nb = sum(1 for b in boxes.get(fr, [])
                 if not b["deleted"] and b["g"] in (1, 2))
        if nb == 0:
            empty.append(n)

    if not empty:
        print("空フレームはありません。")
        return
    print(f"埋める空フレーム: {len(empty)} 枚  {empty}", flush=True)

    # 既存の box_corrections を全保持。追加キーは既存の最小キーより下から振る。
    existing_rows = []
    min_key = 0
    if pa.CORRECTIONS_CSV.exists():
        with pa.CORRECTIONS_CSV.open() as f:
            for r in csv.DictReader(f):
                existing_rows.append(r)
                try:
                    min_key = min(min_key, int(r["box_key"]))
                except (ValueError, KeyError):
                    pass
    added_seq = min_key - 1

    from ultralytics import YOLO
    model = YOLO(str(pa.YOLO_MODEL_PATH))
    knn = br.load_knn_model()
    threshold, green_high = br.load_calibration()
    print("色判定:", "kNNモデル" if knn else f"緑比率しきい値 {threshold:.3f}", flush=True)

    new_rows = []
    total = 0
    gc = {1: 0, 2: 0, 3: 0}
    for i, n in enumerate(empty):
        fr = n - 1
        path = pa.FRAMES_DIR / f"frame_{n:06d}.jpg"
        pil = Image.open(path)
        results = model.predict(source=str(path), classes=[0],
                                conf=args.conf, imgsz=args.imgsz, verbose=False)
        for res in results:
            if res.boxes is None:
                continue
            for xyxy in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (float(c) for c in xyxy[:4])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crop = br.torso_crop(pil, x1, y1, x2, y2)
                g = 3 if crop is None else br.predict_group(crop, knn, threshold, green_high)
                gc[g] = gc.get(g, 0) + 1
                key = added_seq
                added_seq -= 1
                new_rows.append({"frame": fr, "box_key": key, "action": "add",
                                 "x1": round(x1, 1), "y1": round(y1, 1),
                                 "x2": round(x2, 1), "y2": round(y2, 1),
                                 "orig_player_id": -1, "group": g})
                total += 1
        if (i + 1) % 10 == 0 or i + 1 == len(empty):
            print(f"  {i+1}/{len(empty)} フレーム  累計検出{total}", flush=True)

    fields = ["frame", "box_key", "action", "x1", "y1", "x2", "y2",
              "orig_player_id", "group"]
    with pa.CORRECTIONS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing_rows:
            w.writerow({k: r.get(k, "") for k in fields})
        w.writerows(new_rows)

    print(f"\n完了: {len(empty)} 空フレームに {total} ボックス追加 "
          f"(緑{gc.get(1,0)} 黒{gc.get(2,0)} 未検出{gc.get(3,0)})", flush=True)
    print(f"  {pa.CORRECTIONS_CSV} に追記（既存 {len(existing_rows)} 行 + 新規 {len(new_rows)} 行）",
          flush=True)


if __name__ == "__main__":
    main()
