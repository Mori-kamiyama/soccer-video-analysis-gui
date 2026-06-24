"""
G1/G2 チーム自動分類 (frames 211-263)

アルゴリズム:
  1. 選手バウンディングボックスの「胴体帯(上10%〜65%)」に限定
  2. 砂グラウンド色・黒画素を除外し「Gチャンネル優位」ピクセルの比率を計算
  3. player_id ごとに全フレームの mean を集計
  4. K-means(k=2) で G1（緑ビブス）/ G2（黒）に分類

起動:
  cd /Users/yuta/Downloads/サッカー
  python3 team_classifier_g1g2.py

出力:
  outputs/team_g1g2_assignments.csv  ... player_id → G1/G2
  outputs/team_g1g2_positions.csv    ... player_positions_all に team_g1g2 列追加
  outputs/team_g1g2_verify/          ... フレーム別検証画像（緑=G1 赤=G2）
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ======== 設定 ========
FRAMES_DIR  = Path("frames")
POS_CSV     = Path("outputs/player_positions_all.csv")
OUT_ASSIGN  = Path("outputs/team_g1g2_assignments.csv")
OUT_POS     = Path("outputs/team_g1g2_positions.csv")
OUT_VERIFY  = Path("outputs/team_g1g2_verify")

FRAME_START = 211
FRAME_END   = 263

# 胴体帯（ビブスが写る範囲）
TORSO_TOP    = 0.10   # バウンディングボックス上端からの割合
TORSO_BOTTOM = 0.65   # 同下端からの割合

# 砂グラウンド（暖色: R が高め）除外
BG_R_G_DIFF =  5   # R > G - BG_R_G_DIFF → 背景
BG_R_B_DIFF = 15   # R > B + BG_R_B_DIFF → 背景

# 緑ビブス判定
DARK_THRESH = 80    # RGB 全チャンネルがこれ未満 → 黒画素
GREEN_G_R   =  5   # G > R + GREEN_G_R → 緑候補
GREEN_G_B   = -5   # G > B + GREEN_G_B → 緑候補

# 検証画像フレーム
VERIFY_FRAMES = [211, 225, 240, 255, 263]
COLORS = {"G1": (0, 210, 70), "G2": (30, 30, 220), "?": (140, 140, 140)}

IMG_CACHE_SIZE = 12


# ======== 胴体帯の緑比率を計算 ========

def calc_green_ratio(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    h = y2 - y1
    if h < 8 or (x2 - x1) < 4:
        return 0.0
    ty1 = y1 + int(h * TORSO_TOP)
    ty2 = y1 + int(h * TORSO_BOTTOM)
    torso = img[ty1:ty2, x1:x2]
    n = torso.shape[0] * torso.shape[1]
    if n < 6:
        return 0.0
    B = torso[:, :, 0].astype(np.float32)
    G = torso[:, :, 1].astype(np.float32)
    R = torso[:, :, 2].astype(np.float32)
    bg    = (R > G - BG_R_G_DIFF) & (R > B + BG_R_B_DIFF)
    dark  = (B < DARK_THRESH) & (G < DARK_THRESH) & (R < DARK_THRESH)
    green = (G > R + GREEN_G_R) & (G > B + GREEN_G_B) & ~bg & ~dark
    return float(green.sum()) / n


# ======== メイン処理 ========

def main() -> None:
    print("=" * 52)
    print("  G1/G2 チーム自動分類")
    print(f"  対象フレーム: {FRAME_START} 〜 {FRAME_END}")
    print("=" * 52)

    if not POS_CSV.exists():
        sys.exit(f"[ERROR] {POS_CSV} が見つかりません")

    df_all = pd.read_csv(POS_CSV)
    target = df_all[
        (df_all["frame"] >= FRAME_START) & (df_all["frame"] <= FRAME_END)
    ].copy()
    print(f"\n対象: {len(target):,} 行  /  ユニーク player_id: {target['player_id'].nunique()}")

    # ---- Step 1: 各検出で green_ratio を計算 ----
    print("\n[Step 1] 胴体帯の green_ratio を計算...")
    records: list[dict] = []
    img_cache: dict[str, np.ndarray | None] = {}

    for i, (_, row) in enumerate(target.iterrows()):
        fname = str(row["filename"])
        if fname not in img_cache:
            p = FRAMES_DIR / fname
            img_cache[fname] = cv2.imread(str(p)) if p.exists() else None
            if len(img_cache) > IMG_CACHE_SIZE:
                del img_cache[next(iter(img_cache))]
        img = img_cache[fname]
        if img is None:
            continue
        gr = calc_green_ratio(
            img,
            int(row["x1"]), int(row["y1"]),
            int(row["x2"]), int(row["y2"]),
        )
        records.append({
            "player_id"  : int(row["player_id"]),
            "frame"      : int(row["frame"]),
            "green_ratio": gr,
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1:4d}/{len(target)} ...", end="\r", flush=True)

    print(f"  {len(records):,} 件 完了          ")
    res_df = pd.DataFrame(records)

    # ---- Step 2: player_id ごとに集計 ----
    print("\n[Step 2] player_id ごとに集計...")
    agg = (
        res_df.groupby("player_id")["green_ratio"]
        .agg(green_mean="mean", green_max="max", n_frames="count")
        .reset_index()
    )

    # ---- Step 3: K-means(k=2) で分類 ----
    print("\n[Step 3] K-means(k=2) でチーム分類...")
    X = agg["green_mean"].values.reshape(-1, 1).astype(np.float32)

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
        centers = km.cluster_centers_.flatten()
    except ImportError:
        c0, c1 = float(X.min()), float(X.max())
        for _ in range(200):
            labels = np.abs(X.flatten() - c1) < np.abs(X.flatten() - c0)
            c0n = float(X[~labels].mean()) if (~labels).any() else c0
            c1n = float(X[ labels].mean()) if  labels.any()  else c1
            if abs(c0n - c0) < 1e-7 and abs(c1n - c1) < 1e-7:
                break
            c0, c1 = c0n, c1n
        centers = np.array([c0, c1])

    km_threshold = float(centers.mean())

    # Otsu 法 (1-D): 分散比が最大になる閾値を探す
    vals = np.sort(agg["green_mean"].values)
    best_t, best_var = km_threshold, -1.0
    for t in vals[1:-1]:   # 両端は除く
        lo = vals[vals < t]
        hi = vals[vals >= t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        w0, w1 = len(lo) / len(vals), len(hi) / len(vals)
        var_b = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var_b > best_var:
            best_var = var_b
            best_t = float(t)

    # K-means と Otsu が近ければ平均を採用
    if abs(km_threshold - best_t) < 0.04:
        threshold = (km_threshold + best_t) / 2
        method = "K-means+Otsu avg"
    else:
        # 乖離が大きい場合は、値が低い方を選ぶ（G2→G1 の誤分類を減らす）
        threshold = min(km_threshold, best_t)
        method = f"Otsu(k={threshold:.3f}) vs km({km_threshold:.3f}) → lower"

    print(f"  K-means 中心: {centers.min():.4f}  /  {centers.max():.4f}")
    print(f"  K-means 閾値: {km_threshold:.4f}")
    print(f"  Otsu   閾値: {best_t:.4f}")
    print(f"  採用閾値    : {threshold:.4f}  ({method})")

    agg["team"] = agg["green_mean"].apply(lambda v: "G1" if v >= threshold else "G2")

    g1 = (agg["team"] == "G1").sum()
    g2 = (agg["team"] == "G2").sum()
    print(f"\n  ✅ G1（緑ビブス）: {g1} player_id")
    print(f"  ✅ G2（黒）      : {g2} player_id")

    # ---- Step 4: CSV 保存 ----
    print("\n[Step 4] CSV 保存...")
    OUT_ASSIGN.parent.mkdir(exist_ok=True)
    out_cols = ["player_id", "team", "green_mean", "green_max", "n_frames"]
    agg[out_cols].sort_values("green_mean", ascending=False).to_csv(OUT_ASSIGN, index=False)
    print(f"  {OUT_ASSIGN}")

    team_map = dict(zip(agg["player_id"], agg["team"]))
    df_all["team_g1g2"] = df_all["player_id"].map(team_map).fillna("?")
    df_all.to_csv(OUT_POS, index=False)
    print(f"  {OUT_POS}")

    # ---- Step 5: 検証画像 ----
    print("\n[Step 5] 検証画像を生成...")
    OUT_VERIFY.mkdir(exist_ok=True)
    available = set(target["frame"].unique())
    verify_list = [f for f in VERIFY_FRAMES if f in available]

    for fnum in verify_list:
        frame_rows = df_all[df_all["frame"] == fnum]
        if frame_rows.empty:
            continue
        fname = frame_rows.iloc[0]["filename"]
        p = FRAMES_DIR / fname
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        vis = img.copy()
        for _, row in frame_rows.iterrows():
            x1, y1 = int(row["x1"]), int(row["y1"])
            x2, y2 = int(row["x2"]), int(row["y2"])
            team = str(row.get("team_g1g2", "?"))
            if team == "nan":
                team = "?"
            color = COLORS.get(team, COLORS["?"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            pid = int(row["player_id"])
            label = f"{team}:{pid}"
            cv2.putText(vis, label, (x1, max(y1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        cv2.rectangle(vis, (8, 8), (185, 52), (20, 20, 20), -1)
        cv2.putText(vis, "G1: green bib", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["G1"], 2)
        cv2.putText(vis, "G2: black",     (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["G2"], 2)
        out_p = OUT_VERIFY / f"verify_frame{fnum:04d}.jpg"
        cv2.imwrite(str(out_p), vis)
        print(f"  {out_p}")

    # ---- 結果サマリー ----
    print("\n" + "=" * 52)
    print("  分類結果サマリー（green_mean 降順）")
    print("=" * 52)
    print(f"採用閾値 = {threshold:.4f}  ({method})")
    print(f"G1 player_ids: {sorted(agg[agg['team']=='G1']['player_id'].tolist())}")
    print(f"G2 player_ids: {sorted(agg[agg['team']=='G2']['player_id'].tolist())}")
    print()
    print(agg[out_cols].sort_values("green_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
