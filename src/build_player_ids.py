"""
高fpsデータに player_id を付ける（安定IDの土台）。

入力: outputs/box_corrections.csv（ボックス・1920基準・group）
      outputs/tracks_hi.csv（frame,track_id,team,x_img,y_img,...）= 移行時の追跡(139本)
処理:
  - 各フレームで box と tracks_hi の足元を 1:1 排他で最近傍結合 → track_id 付与（未結合は新規ID）
  - player_id = track_id（初期）。merge_map_hi.csv があればそれで統合IDに置換（エディターで再開）
出力: outputs/players_hi.csv
      frame, player_id, track_id, team, x1,y1,x2,y2, foot_x, foot_y, x_pitch, y_pitch

~22人への統合・スワップ修正は専用エディター（player_merge_hi / player_id_editor）で行う。

使い方:
  .venv/bin/python build_player_ids.py
"""

from __future__ import annotations

import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

CORR = Path("outputs/box_corrections.csv")
TRACKS = Path("outputs/tracks_hi.csv")
POINTS = Path("pitch_points.json")
OUT = Path("outputs/players_hi.csv")
MERGE_MAP = Path("outputs/merge_map_hi.csv")   # 鳥瞰エディター(player_merge_hi)の出力
EDITS_JSON = Path("outputs/id_edits.json")     # 画像エディター(player_id_editor)の出力
TOL = 3.0  # 足元結合の許容px(1920基準)


def load_homography():
    import cv2
    import numpy as np
    d = json.loads(POINTS.read_text())
    return cv2.getPerspectiveTransform(
        __import__("numpy").array(d["image"], dtype="float32"),
        __import__("numpy").array(d["pitch"], dtype="float32"))


def foot_to_pitch(H, fx, fy):
    import cv2
    import numpy as np
    out = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype="float32"), H)[0][0]
    return float(out[0]), float(out[1])


def main():
    H = load_homography()

    # tracks_hi: frame -> [(track_id, fx, fy)]
    tfeet = defaultdict(list)
    max_tid = -1
    for r in csv.DictReader(TRACKS.open()):
        tid = int(r["track_id"])
        tfeet[int(r["frame"])].append((tid, float(r["x_img"]), float(r["y_img"])))
        max_tid = max(max_tid, tid)
    next_new = max_tid + 1

    # box をフレームごとに読み、tracks_hi と 1:1 排他で最近傍結合
    boxes_by_frame = defaultdict(list)
    for r in csv.DictReader(CORR.open()):
        boxes_by_frame[int(r["frame"])].append(r)

    boxes = []
    tol2 = TOL ** 2
    for fr, brows in boxes_by_frame.items():
        feet = tfeet.get(fr, [])
        # 候補ペア(距離, box_i, track_j) を作り、近い順に排他割当
        pairs = []
        for bi, r in enumerate(brows):
            x1, x2, y2 = float(r["x1"]), float(r["x2"]), float(r["y2"])
            fx, fy = (x1 + x2) / 2, y2
            for tj, (tid, cx, cy) in enumerate(feet):
                d = (fx - cx) ** 2 + (fy - cy) ** 2
                if d <= tol2:
                    pairs.append((d, bi, tj))
        pairs.sort()
        used_b, used_t = {}, set()
        for d, bi, tj in pairs:
            if bi in used_b or tj in used_t:
                continue
            used_b[bi] = feet[tj][0]
            used_t.add(tj)
        for bi, r in enumerate(brows):
            tid = used_b.get(bi)
            if tid is None:           # 未結合（余剰検出など）→ 新規ID
                tid = next_new; next_new += 1
            x1, y1, x2, y2 = (float(r["x1"]), float(r["y1"]),
                              float(r["x2"]), float(r["y2"]))
            fx, fy = (x1 + x2) / 2, y2
            px, py = foot_to_pitch(H, fx, fy)
            boxes.append({"frame": fr, "track_id": tid, "team": int(r["group"]),
                          "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "foot_x": round(fx, 1), "foot_y": round(fy, 1),
                          "x_pitch": round(px, 2), "y_pitch": round(py, 2)})

    # 極端に短いトラック（1〜2フレームの単発＝余剰検出/ノイズ）は捨てる
    MIN_FRAMES = 3
    tcount = defaultdict(int)
    for b in boxes:
        tcount[b["track_id"]] += 1
    before = len(boxes)
    boxes = [b for b in boxes if tcount[b["track_id"]] >= MIN_FRAMES]

    # 手追加ボックス（id_edits.json の added・負のtrack_id）を取り込む（短トラック除去の対象外）
    if EDITS_JSON.exists():
        try:
            ad = json.loads(EDITS_JSON.read_text()).get("added", [])
        except Exception:
            ad = []
        for a in ad:
            x1, y1, x2, y2 = float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])
            fx, fy = (x1 + x2) / 2, y2
            px, py = foot_to_pitch(H, fx, fy)
            boxes.append({"frame": int(a["frame"]), "track_id": int(a["track_id"]),
                          "team": int(a["team"]), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "foot_x": round(fx, 1), "foot_y": round(fy, 1),
                          "x_pitch": round(px, 2), "y_pitch": round(py, 2)})
        if ad:
            print(f"手追加ボックス {len(ad)} 件を取り込み", flush=True)

    track_ids = sorted({b["track_id"] for b in boxes})
    print(f"結合後トラック数: {len(track_ids)}（短トラック除去で {before-len([b for b in boxes if b['track_id']>=0])} ボックス削除）",
          flush=True)

    # 編集の適用（カット＋結合＋移動/リサイズ＋名前）。sub_id=(track_id, 直前のカット境界)。
    cuts = defaultdict(list)
    pairs = []
    geom = {}
    names = {}
    if EDITS_JSON.exists():
        d = json.loads(EDITS_JSON.read_text())
        for k, v in d.get("cuts", {}).items():
            cuts[int(k)] = sorted(int(x) for x in v)
        for a, b in d.get("merges", []):
            pairs.append((tuple(a), tuple(b)))
        for k, v in d.get("geom", {}).items():
            fr, tid = k.split("_")
            geom[(int(fr), int(tid))] = tuple(v)
        names = dict(d.get("names", {}))
        print(f"id_edits.json 適用: カット{sum(len(v) for v in cuts.values())} / "
              f"結合{len(pairs)} / 調整{len(geom)} / 名前{len(names)}", flush=True)
        # 移動/リサイズを反映（足元・ピッチ座標を再計算）
        for b in boxes:
            ov = geom.get((b["frame"], b["track_id"]))
            if ov:
                b["x1"], b["y1"], b["x2"], b["y2"] = ov
                fx, fy = (b["x1"] + b["x2"]) / 2, b["y2"]
                px, py = foot_to_pitch(H, fx, fy)
                b["foot_x"], b["foot_y"] = round(fx, 1), round(fy, 1)
                b["x_pitch"], b["y_pitch"] = round(px, 2), round(py, 2)

    def sub_id(tid, fr):
        cs = cuts.get(tid)
        if not cs:
            return (tid, 0)
        i = bisect.bisect_right(cs, fr)
        return (tid, cs[i - 1] if i > 0 else 0)

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in pairs:
        union(a, b)
    # 鳥瞰エディター(merge_map_hi.csv)の track 単位統合も併用（カット無し→境界0）
    if MERGE_MAP.exists():
        for r in csv.DictReader(MERGE_MAP.open()):
            old, new = int(r["old_id"]), int(r["new_id"])
            if old != new:
                union((old, 0), (new, 0))
        print("merge_map_hi も併用", flush=True)

    # 名前を現在の代表(root)へ付け替える（結合で代表が変わるため）
    fixed_names = {}
    for key, nm in names.items():
        a, b = key.split("_")
        r = find((int(a), int(b)))
        fixed_names.setdefault(f"{r[0]}_{r[1]}", nm)
    names = fixed_names

    # player_id = canonical を フレーム数多い順に 1..N
    canon_size = defaultdict(int)
    for b in boxes:
        canon_size[find(sub_id(b["track_id"], b["frame"]))] += 1
    pid_num = {c: i + 1 for i, c in enumerate(sorted(canon_size, key=lambda c: -canon_size[c]))}

    def pid_of_box(b):
        return pid_num[find(sub_id(b["track_id"], b["frame"]))]

    print(f"player 数: {len(pid_num)}（エディターで~22に統合）", flush=True)

    fields = ["frame", "player_id", "track_id", "team", "team_hint", "name", "x1", "y1", "x2", "y2",
              "foot_x", "foot_y", "x_pitch", "y_pitch"]
    team_hint = {1: "緑", 2: "黒"}

    def name_of(b):
        c = find(sub_id(b["track_id"], b["frame"]))
        return names.get(f"{c[0]}_{c[1]}", "")

    boxes.sort(key=lambda b: (b["frame"], pid_of_box(b)))
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for b in boxes:
            row = {"player_id": pid_of_box(b), "name": name_of(b),
                   "team_hint": team_hint.get(b["team"], "?"), **b}
            w.writerow({k: row[k] for k in fields})
    print(f"保存: {OUT}（{len(boxes)}行）", flush=True)

    # 同一player_idが同時に複数（要修正の目安）
    per = defaultdict(lambda: defaultdict(int))
    for b in boxes:
        per[b["frame"]][pid_of_box(b)] += 1
    dup = sum(1 for d in per.values() if any(c >= 2 for c in d.values()))
    print(f"同一player_idが同時に複数のフレーム: {dup}", flush=True)


if __name__ == "__main__":
    main()
