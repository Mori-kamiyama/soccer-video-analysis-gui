"""
チーム伝播スクリプト

211〜342フレームで確認済みのグループからジャージ色を抽出し、
全690グループにチームA/Bを伝播させる。

処理の流れ:
  1. 確認済みフレームの各グループから画像でジャージ色を抽出
  2. K-Means(k=2) でA/Bに分類
  3. 同じグループIDが他フレームに出てもチームは同じ（即座）
  4. 未登場グループは Lab 最近傍で割り当て

起動:
  .venv/bin/python propagate_teams.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

INPUT_CSV   = Path("outputs/player_positions_all.csv")
CORR_CSV    = Path("outputs/box_corrections.csv")
MERGE_MAP   = Path("outputs/merge_map.csv")
OUTPUT_CSV  = Path("outputs/team_assignments.csv")
FRAMES_DIR  = Path("frames")

CONFIRM_LO  = 210   # 確認済みフレーム範囲 (frame番号 = 画像番号-1)
CONFIRM_HI  = 275   # 画像276まで

JERSEY_TOP    = 0.10
JERSEY_BOTTOM = 0.60
SAMPLE_MAX    = 20
MIN_PIX       = 8

# 緑ジャージ判定（HSV）: 芝より彩度・明度が高い濃い緑
GREEN_H_LO, GREEN_H_HI = 35, 85
GREEN_S_MIN = 60
GREEN_V_MIN = 60

# 緑比率がこれ以上 → 緑チーム
GREEN_THRESH = 0.10   # 10%


# ---- 緑比率計算 ----

def green_ratio(bgr_crop: np.ndarray) -> float | None:
    """クロップ内の緑ピクセル比率を返す。ピクセル不足は None。"""
    pixels = bgr_crop.reshape(-1, 3)
    if len(pixels) < MIN_PIX:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    is_green = (h >= GREEN_H_LO) & (h <= GREEN_H_HI) & (s >= GREEN_S_MIN) & (v >= GREEN_V_MIN)
    return float(is_green.sum()) / len(pixels)


def crop_jersey(img: np.ndarray, x1: float, y1: float, x2: float, y2: float):
    h = y2 - y1
    jy1, jy2 = int(y1 + h * JERSEY_TOP), int(y1 + h * JERSEY_BOTTOM)
    ix1, ix2 = max(0, int(x1)), int(x2)
    if jy2 <= jy1 or ix2 <= ix1:
        return None
    crop = img[jy1:jy2, ix1:ix2]
    return crop if crop.size > 0 else None


# ---- データロード ----

def load_positions() -> dict[int, list[dict]]:
    """frame -> list of row (with group applied from merge_map)"""
    merge: dict[int, int] = {}
    if MERGE_MAP.exists():
        with MERGE_MAP.open() as f:
            for r in csv.DictReader(f):
                merge[int(r["old_id"])] = int(r["new_id"])

    by_frame: dict[int, list[dict]] = defaultdict(list)
    with INPUT_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                fr = int(r["frame"])
                pid = int(r["player_id"])
            except (ValueError, KeyError):
                continue
            r = dict(r)
            r["group"] = merge.get(pid, pid)
            by_frame[fr].append(r)
    return by_frame


def load_corrections() -> tuple[dict[tuple, dict], dict[tuple, str]]:
    """
    Returns:
        geom_overrides: (frame, group) -> {x1,y1,x2,y2}  (add/move で確定した座標)
        deleted:        (frame, box_key) -> True
    """
    geom: dict[tuple, dict] = {}   # (frame, group) -> latest box
    deleted: set[tuple] = set()
    if not CORR_CSV.exists():
        return geom, deleted
    with CORR_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                fr = int(r["frame"])
                key = int(r["box_key"])
                g = int(r["group"])
            except (ValueError, KeyError):
                continue
            action = r["action"]
            if action == "delete":
                deleted.add((fr, key))
            elif action in ("add", "move") and r["x1"]:
                geom[(fr, g)] = {
                    "x1": float(r["x1"]), "y1": float(r["y1"]),
                    "x2": float(r["x2"]), "y2": float(r["y2"]),
                }
    return geom, deleted


# ---- メイン処理 ----

def extract_green_ratios_for_groups(
    target_groups: set[int],
    by_frame: dict[int, list[dict]],
    geom_overrides: dict,
    frame_lo: int = None,
    frame_hi: int = None,
) -> dict[int, float | None]:
    """
    target_groups のジャージ緑比率を抽出する。
    frame_lo/frame_hi を指定するとその範囲のフレームのみ使う。
    """
    # 各グループの (frame, x1,y1,x2,y2) リストを収集
    group_frames: dict[int, list[tuple]] = defaultdict(list)
    for fr, rows in by_frame.items():
        if frame_lo is not None and fr < frame_lo:
            continue
        if frame_hi is not None and fr > frame_hi:
            continue
        seen = set()
        for r in rows:
            g = r["group"]
            if g not in target_groups or g in seen:
                continue
            coords = geom_overrides.get((fr, g))
            if coords:
                x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
            else:
                try:
                    x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
                except (ValueError, KeyError):
                    continue
            group_frames[g].append((fr, x1, y1, x2, y2))
            seen.add(g)
    for (fr, g), coords in geom_overrides.items():
        if g in target_groups:
            if frame_lo is not None and fr < frame_lo: continue
            if frame_hi is not None and fr > frame_hi: continue
            group_frames[g].append((fr, coords["x1"], coords["y1"], coords["x2"], coords["y2"]))

    ratio_map: dict[int, float | None] = {}
    groups = sorted(target_groups)
    for i, g in enumerate(groups):
        print(f"\r  緑比率抽出 {i+1}/{len(groups)}", end="", flush=True)
        entries = group_frames.get(g, [])
        step = max(1, len(entries) // SAMPLE_MAX)
        ratios = []
        for (fr, x1, y1, x2, y2) in entries[::step][:SAMPLE_MAX]:
            img = cv2.imread(str(FRAMES_DIR / f"frame_{fr + 1:06d}.jpg"))
            if img is None:
                continue
            c = crop_jersey(img, x1, y1, x2, y2)
            if c is None:
                continue
            rv = green_ratio(c)
            if rv is not None:
                ratios.append(rv)
        ratio_map[g] = float(np.mean(ratios)) if ratios else None
    print()
    return ratio_map


def classify_by_green(ratio_map: dict[int, float | None]) -> dict[int, str]:
    """緑比率 >= GREEN_THRESH → A(緑チーム)、それ以外 → B(黒/ダークチーム)。"""
    assignments: dict[int, str] = {}
    for g, ratio in ratio_map.items():
        if ratio is None:
            assignments[g] = "?"
        elif ratio >= GREEN_THRESH:
            assignments[g] = "A"
        else:
            assignments[g] = "B"
    return assignments


def propagate_to_all(
    known: dict[int, str],
    all_groups: set[int],
    ratio_map_known: dict[int, float | None],
    by_frame: dict[int, list[dict]],
) -> dict[int, str]:
    """
    - 確認済みグループは known から直接コピー（同じグループID = 同じ選手 = 同じチーム）
    - 未知グループはジャージ色の最近傍マッチングで割り当て
      ★フレームを1回だけ読むバッチ処理で高速化
    """
    result = dict(known)
    unknown = {g for g in all_groups if g not in known}
    if not unknown:
        return result

    # 未知グループのサンプリング計画：グループごとに均等に SAMPLE_MAX フレームを選ぶ
    group_rows: dict[int, list[dict]] = defaultdict(list)
    for rows in by_frame.values():
        for r in rows:
            if r["group"] in unknown:
                group_rows[r["group"]].append(r)

    # frame -> [(group, x1,y1,x2,y2)] のバッチ
    frame_tasks: dict[int, list[tuple]] = defaultdict(list)
    for g, rows in group_rows.items():
        step = max(1, len(rows) // SAMPLE_MAX)
        for r in rows[::step][:SAMPLE_MAX]:
            try:
                frame_tasks[int(r["frame"])].append(
                    (g, float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
            except (ValueError, KeyError):
                continue

    # フレームを1回ずつ読んで全グループを一気に処理
    print(f"未知グループ {len(unknown)} 個 / {len(frame_tasks)} フレームを一括処理中...")
    group_labs: dict[int, list[np.ndarray]] = defaultdict(list)
    frames_sorted = sorted(frame_tasks.keys())
    total = len(frames_sorted)
    for i, fr in enumerate(frames_sorted):
        if (i + 1) % 50 == 0 or i == total - 1:
            print(f"\r  フレーム {i+1}/{total}", end="", flush=True)
        fname = f"frame_{fr + 1:06d}.jpg"
        img = cv2.imread(str(FRAMES_DIR / fname))
        if img is None:
            continue
        for (g, x1, y1, x2, y2) in frame_tasks[fr]:
            c = crop_jersey(img, x1, y1, x2, y2)
            if c is None:
                continue
            r = green_ratio(c)
            if r is not None:
                group_labs[g].append([r])  # [緑比率] として格納
    print()

    # グループごとに緑比率を集計してチーム決定
    for g in unknown:
        labs = group_labs.get(g)
        if labs:
            # group_labs に緑比率を float として格納（後でリファクタ可）
            avg_ratio = float(np.mean([l[0] for l in labs]))  # [0]に緑比率を入れる
            result[g] = "A" if avg_ratio >= GREEN_THRESH else "B"
        else:
            result[g] = "?"
    return result


def save_assignments(assignments: dict[int, str], ratio_map: dict[int, float | None] = {}) -> None:
    with OUTPUT_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "team", "green_ratio"])
        for pid in sorted(assignments):
            ratio = ratio_map.get(pid)
            team = assignments[pid]
            w.writerow([pid, team, f"{ratio:.4f}" if isinstance(ratio, float) else ""])
    print(f"保存: {OUTPUT_CSV}")


def load_known_teams() -> dict[int, str]:
    """team_assignments.csv から既知のチームラベルを読む。"""
    result = {}
    if OUTPUT_CSV.exists():
        with OUTPUT_CSV.open() as f:
            for r in csv.DictReader(f):
                try:
                    result[int(r["player_id"])] = r["team"]
                except (ValueError, KeyError):
                    pass
    return result


def learn_threshold(
    ratio_map: dict[int, float | None],
    known_teams: dict[int, str],
) -> float:
    """211-276の既知ラベルと緑比率から最適しきい値を学習する。"""
    a_ratios = [ratio_map[g] for g in ratio_map
                if known_teams.get(g) == "A" and ratio_map[g] is not None]
    b_ratios = [ratio_map[g] for g in ratio_map
                if known_teams.get(g) == "B" and ratio_map[g] is not None]

    print(f"\n  チームA(緑): {len(a_ratios)}グループ  "
          f"緑比率 mean={np.mean(a_ratios):.3f} min={min(a_ratios):.3f}" if a_ratios else
          f"\n  チームA: データなし")
    print(f"  チームB(黒): {len(b_ratios)}グループ  "
          f"緑比率 mean={np.mean(b_ratios):.3f} max={max(b_ratios):.3f}" if b_ratios else
          f"  チームB: データなし")

    if not a_ratios or not b_ratios:
        print(f"  → 学習データ不足。固定しきい値 {GREEN_THRESH} を使用")
        return GREEN_THRESH

    # A の最小値と B の最大値の中間をしきい値にする
    a_min = min(a_ratios)
    b_max = max(b_ratios)
    if a_min > b_max:
        thresh = (a_min + b_max) / 2
    else:
        # 重複がある場合は F1スコアを最大化するしきい値を探す
        candidates = sorted(set(a_ratios + b_ratios))
        best_thresh, best_f1 = GREEN_THRESH, 0.0
        for t in candidates:
            tp = sum(1 for r in a_ratios if r >= t)
            fp = sum(1 for r in b_ratios if r >= t)
            fn = sum(1 for r in a_ratios if r < t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        thresh = best_thresh
        print(f"  ※ 重複あり: F1最大化しきい値={thresh:.3f} (F1={best_f1:.3f})")

    # 訓練データへの適合率を確認
    correct = sum(1 for g in ratio_map if ratio_map[g] is not None and known_teams.get(g) in ("A", "B") and
                  ((ratio_map[g] >= thresh) == (known_teams[g] == "A")))
    total = sum(1 for g in ratio_map if ratio_map[g] is not None and known_teams.get(g) in ("A", "B"))
    print(f"  → 学習しきい値 = {thresh:.3f}  訓練精度 {correct}/{total} ({100*correct//total if total else 0}%)")
    return thresh


def main() -> None:
    print("データ読み込み中...")
    by_frame = load_positions()
    geom_overrides, deleted = load_corrections()
    all_groups = {r["group"] for rows in by_frame.values() for r in rows}
    print(f"全グループ数: {len(all_groups)}")

    # Step1: 211-276 の既知チームラベルを読む（正解ラベル）
    known_teams = load_known_teams()
    labeled_groups = {r["group"] for fr in range(CONFIRM_LO, CONFIRM_HI + 1)
                      for r in by_frame.get(fr, [])}
    labeled = {g: known_teams[g] for g in labeled_groups
               if known_teams.get(g) in ("A", "B")}
    a_labeled = sum(1 for v in labeled.values() if v == "A")
    b_labeled = sum(1 for v in labeled.values() if v == "B")
    print(f"教師データ: {len(labeled)}グループ (A={a_labeled} B={b_labeled})")

    # Step2: 教師グループの緑比率を全フレームから抽出（フレーム制限なし）
    print(f"教師グループ({len(labeled)})の緑比率を抽出中...")
    ratio_map = extract_green_ratios_for_groups(
        set(labeled.keys()), by_frame, geom_overrides
    )

    # Step3: 既知ラベルからしきい値を学習
    thresh = learn_threshold(ratio_map, labeled)

    # Step4: 未知グループをバッチ処理
    unknown = all_groups - set(labeled.keys())
    print(f"\n未知グループ {len(unknown)} 個を一括処理中...")
    unknown_ratios = extract_green_ratios_for_groups(
        unknown, by_frame, geom_overrides
    )
    ratio_map.update(unknown_ratios)

    # Step5: 全グループのチームを決定（教師はラベルそのまま固定）
    global GREEN_THRESH
    GREEN_THRESH = thresh
    all_assignments = dict(labeled)
    for g in unknown:
        ratio = ratio_map.get(g)
        if ratio is None:
            all_assignments[g] = "?"
        else:
            all_assignments[g] = "A" if ratio >= thresh else "B"

    a2 = sum(1 for v in all_assignments.values() if v == "A")
    b2 = sum(1 for v in all_assignments.values() if v == "B")
    q2 = sum(1 for v in all_assignments.values() if v == "?")
    print(f"\n全グループ: 緑(A)={a2}  黒(B)={b2}  ?={q2}")

    save_assignments(all_assignments, ratio_map)
    print("完了。team_classifier.py で確認・修正できます。")


if __name__ == "__main__":
    main()
