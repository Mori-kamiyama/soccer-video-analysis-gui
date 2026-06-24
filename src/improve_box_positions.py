"""
中間フレームの補間ボックス（速く動く選手でズレがち）を、実YOLO検出にスナップして
位置を実選手に合わせる。キーフレーム（j%6==0＝手作業/正確）は触らない。

box_corrections.csv の該当ボックスの座標だけ差し替える（チーム・キーは保持）。
ユーザーの調整(geom)は build_player_ids 側で後段適用されるので上書きされない。

使い方:
  .venv/bin/python improve_box_positions.py --factor 6 --tol 55
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

import photo_annotator as pa
import batch_redetect as br

CORR = Path("outputs/box_corrections.csv")
BACKUP = Path("outputs/box_corrections.before_snap.csv")
FRAMES_DIR = pa.FRAMES_DIR
BASE_W = 1920.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=6, help="キーフレーム間隔（12/2fps=6）")
    ap.add_argument("--tol", type=float, default=55.0, help="スナップ許容(1920基準px)")
    ap.add_argument("--conf", type=float, default=pa.YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    sample = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    if not sample:
        raise SystemExit(f"{FRAMES_DIR} にフレームがありません")
    fw = Image.open(sample[0]).size[0]
    to_base = BASE_W / fw

    rows = list(csv.DictReader(CORR.open()))
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame"])].append(r)

    inter_frames = sorted(f for f in by_frame if f % args.factor != 0)
    print(f"中間フレーム {len(inter_frames)} 枚をスナップ（許容{args.tol:.0f}px・1920基準）", flush=True)

    from ultralytics import YOLO
    model = YOLO(str(pa.YOLO_MODEL_PATH))

    tol2 = args.tol ** 2
    snapped = 0
    processed = 0
    for fr in inter_frames:
        img_no = fr + 1
        path = FRAMES_DIR / f"frame_{img_no:06d}.jpg"
        if not path.exists():
            continue
        res = model.predict(source=str(path), classes=[0], conf=args.conf,
                            imgsz=args.imgsz, verbose=False)
        dets = []  # (foot_x, foot_y, (x1,y1,x2,y2)) 1920基準
        for rr in res:
            if rr.boxes is None:
                continue
            for xyxy in rr.boxes.xyxy.cpu().numpy():
                X1, Y1, X2, Y2 = (float(v) * to_base for v in xyxy[:4])
                if X2 - X1 < 2 or Y2 - Y1 < 2:
                    continue
                dets.append(((X1 + X2) / 2, Y2, (X1, Y1, X2, Y2)))

        boxes = by_frame[fr]
        # box足元 ↔ det足元 を近い順に1:1スナップ
        pairs = []
        for bi, r in enumerate(boxes):
            bx = (float(r["x1"]) + float(r["x2"])) / 2
            by = float(r["y2"])
            for di, (dx, dy, _) in enumerate(dets):
                d = (bx - dx) ** 2 + (by - dy) ** 2
                if d <= tol2:
                    pairs.append((d, bi, di))
        pairs.sort()
        ub, ud = set(), set()
        for d, bi, di in pairs:
            if bi in ub or di in ud:
                continue
            ub.add(bi); ud.add(di)
            x1, y1, x2, y2 = dets[di][2]
            r = boxes[bi]
            r["x1"], r["y1"] = round(x1, 1), round(y1, 1)
            r["x2"], r["y2"] = round(x2, 1), round(y2, 1)
            snapped += 1
        processed += 1
        if processed % 200 == 0 or processed == len(inter_frames):
            print(f"  {processed}/{len(inter_frames)} 枚  累計スナップ{snapped}", flush=True)

    shutil.copy(CORR, BACKUP)
    fields = ["frame", "box_key", "action", "x1", "y1", "x2", "y2", "orig_player_id", "group"]
    with CORR.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in fields} for r in rows)
    print(f"\n完了: {snapped} ボックスを実検出にスナップ。バックアップ {BACKUP}", flush=True)
    print("  build_player_ids.py を実行して players_hi.csv に反映してください。", flush=True)


if __name__ == "__main__":
    main()
