"""
中間フレームの過検出（審判・観客・重複など、トラックに紐づかないボックス）を落とす。

考え方: キーフレーム(新frame j%6==0)は手作業由来＝そのまま温存。
中間フレームは、tracks_hi.csv（追跡された選手の足元）に紐づくボックスだけ残し、
どのトラックの足元からも遠いボックス（＝余剰検出）を捨てる。
→ 各中間フレームの人数が「追跡されている選手数(約22)」に収まる。

YOLO再実行は不要。box_corrections.csv を直接クリーニングする（バックアップ付き）。

使い方:
  .venv/bin/python clean_overdetect.py --tol 3.0
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

CORR = Path("outputs/box_corrections.csv")
TRACKS = Path("outputs/tracks_hi.csv")
BACKUP = Path("outputs/box_corrections.before_clean.csv")


def foot(r):
    return ((float(r["x1"]) + float(r["x2"])) / 2, float(r["y2"]))


def overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = (float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"]))
    bx1, by1, bx2, by2 = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / min(aa, bb) if min(aa, bb) > 0 else 0.0


def nms(boxes, thresh):
    """重なり(min面積比)がthreshを超える重複を除去。大きいボックス優先で残す。"""
    order = sorted(boxes, key=lambda r: -(
        (float(r["x2"]) - float(r["x1"])) * (float(r["y2"]) - float(r["y1"]))))
    keep = []
    for r in order:
        if all(overlap_ratio(r, k) <= thresh for k in keep):
            keep.append(r)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=3.0, help="足元一致の許容px(1920基準)")
    ap.add_argument("--factor", type=int, default=6, help="キーフレーム間隔")
    ap.add_argument("--iou", type=float, default=0.5, help="中間の重複除去しきい値(min面積比)")
    args = ap.parse_args()

    # トラック足元（中間フレーム判定用）
    track_feet = defaultdict(list)
    with TRACKS.open() as f:
        for r in csv.DictReader(f):
            track_feet[int(r["frame"])].append((float(r["x_img"]), float(r["y_img"])))

    rows = list(csv.DictReader(CORR.open()))
    fields = ["frame", "box_key", "action", "x1", "y1", "x2", "y2", "orig_player_id", "group"]

    # 1) トラック非紐付きの余剰検出を落とす（中間のみ）。キーフレームは全温存。
    tol2 = args.tol ** 2
    by_frame = defaultdict(list)
    for r in rows:
        by_frame[int(r["frame"])].append(r)

    kept, dropped_extra, dropped_dup = [], 0, 0
    for fr, bs in by_frame.items():
        if fr % args.factor == 0:        # キーフレームは温存
            kept.extend(bs); continue
        feet = track_feet.get(fr, [])
        track_backed = []
        for r in bs:
            fx, fy = foot(r)
            if any((fx - a) ** 2 + (fy - b) ** 2 <= tol2 for a, b in feet):
                track_backed.append(r)
            else:
                dropped_extra += 1
        # 2) 重なり合う重複（断片トラック/検出+補間の二重）をNMSで除去
        deduped = nms(track_backed, args.iou)
        dropped_dup += len(track_backed) - len(deduped)
        kept.extend(deduped)
    dropped = dropped_extra + dropped_dup

    # 分布レポート
    def dist(rs):
        per = defaultdict(int)
        for r in rs:
            per[int(r["frame"])] += 1
        v = list(per.values())
        over = sum(1 for x in v if x > 22)
        return statistics.mean(v), min(v), max(v), over

    b = dist(rows); a = dist(kept)
    print(f"前: 平均{b[0]:.1f} 最小{b[1]} 最大{b[2]}  23人以上{b[3]}フレーム")
    print(f"後: 平均{a[0]:.1f} 最小{a[1]} 最大{a[2]}  23人以上{a[3]}フレーム")
    print(f"落とした: 余剰(非紐付き){dropped_extra} + 重複(NMS){dropped_dup} = {dropped}")

    shutil.copy(CORR, BACKUP)
    with CORR.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    print(f"保存: {CORR}（バックアップ {BACKUP}）")


if __name__ == "__main__":
    main()
