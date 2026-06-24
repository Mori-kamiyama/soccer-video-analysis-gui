"""
2fps の手作業アノテーションを、高fps(12fps)の新フレームへ動きで移行する。

座標系の方針: 保存座標はすべて 1920 基準（pitch_points.json・選手座標・ボールと統一）。
新フレームが 2560 等でも、YOLO検出ボックスは (1920/frame_w) 倍して 1920 基準に換算して保存する。
→ box_corrections / pitch_points / homography はスケール不要でそのまま使える。

フレーム対応（フル動画12fps抽出を前提）:
  新frame j ↔ t = j/12,  旧frame i_old ↔ t = 0.5·i_old  →  j = 6·i_old（factor=6）
  キーフレーム: j % 6 == 0（対応旧frame = j//6）。間の5枚を補間で埋める。

処理:
  - キーフレーム: 旧の手作業ボックスをそのまま新frame j=6·i_old に置く（1920基準のまま）。
  - 中間5枚: 新実画像を YOLO 検出 → foot→pitch → 速度補間トラックに最近傍マッチして
            チーム割当（手作業由来トラックのチームを優先）。未マッチ検出は kNN色で採用、
            未マッチトラックは速度補間ボックスを出力（選手を消さない）。

出力:
  outputs/box_corrections_hi.csv   … 全行 add（1920基準）
  outputs/tracks_hi.csv            … 安定track_id付き（解析・統合用）
  outputs/frames_metadata_hi.json  … 12fps メタデータ

使い方:
  .venv/bin/python migrate_annotations.py --frames-dir frames_hi \
      --start-old-img 212 --end-old-img 1311 --factor 6
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import photo_annotator as pa
import visualize_offside as vo
import batch_redetect as br

OUT_CORR = Path("outputs/box_corrections_hi.csv")
OUT_TRACKS = Path("outputs/tracks_hi.csv")
OUT_META = Path("outputs/frames_metadata_hi.json")

BASE_W = 1920.0
MATCH_THRESH_M = 4.0   # 検出↔トラック prior の最近傍マッチ閾値(m)


def foot_to_pitch(H, fx, fy):
    """foot(1920基準) → ピッチ座標(m)。"""
    import cv2
    pt = np.array([[[fx, fy]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)[0][0]
    return float(out[0]), float(out[1])


def interp_track_pos(track, fr):
    """トラックの平滑化済み位置から、frame fr の位置(X,Y,box[1920基準])を補間して返す。
    fr が pos の範囲外なら端で外挿せず None。"""
    pos = track["pos"]
    if fr in pos:
        return pos[fr]
    fs = sorted(pos)
    prev = next((f for f in reversed(fs) if f < fr), None)
    nxt = next((f for f in fs if f > fr), None)
    if prev is None or nxt is None:
        return None
    Xa, Ya, ba = pos[prev]
    Xb, Yb, bb = pos[nxt]
    s = (fr - prev) / (nxt - prev)
    box = tuple(ba[i] + (bb[i] - ba[i]) * s for i in range(4))
    return (Xa + (Xb - Xa) * s, Ya + (Yb - Ya) * s, box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="frames_hi")
    ap.add_argument("--start-old-img", type=int, default=212)
    ap.add_argument("--end-old-img", type=int, default=1311)
    ap.add_argument("--factor", type=int, default=6, help="新fps/旧fps（12/2=6）")
    ap.add_argument("--conf", type=float, default=pa.YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=2560)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    factor = args.factor

    # 新フレームの解像度（1枚目から）→ 1920基準への換算係数
    sample = sorted(frames_dir.glob("frame_*.jpg"))
    if not sample:
        raise SystemExit(f"{frames_dir}/ にフレームがありません。先に Step A の抽出を。")
    fw, fh = Image.open(sample[0]).size
    to_base = BASE_W / fw   # 2560 → 0.75（2560基準px×0.75 = 1920基準px）
    n_hi = len(sample)
    print(f"新フレーム: {n_hi}枚 {fw}x{fh}  1920換算係数={to_base:.4f}", flush=True)

    # 旧アノテーション復元（1920基準）と homography
    old_boxes = vo.reconstruct_boxes()
    H = vo.load_homography()

    old_i_lo = args.start_old_img - 1   # 211
    old_i_hi = args.end_old_img - 1     # 1310
    old_frames = list(range(old_i_lo, old_i_hi + 1))

    # 旧キーフレームから選手(緑/黒)を取り出し → トラッキング（速度予測）
    raw_by_old = {fr: vo.raw_players(old_boxes.get(fr, []), H) for fr in old_frames}
    tracks = vo.build_tracks(old_frames, raw_by_old)
    # box→tid 逆引きは平滑化前の生ボックスで作る（旧手作業ボックスと一致させるため）
    box_to_tid = defaultdict(dict)  # old_fr -> {(round座標): tid}
    for tid, t in tracks.items():
        for fr, (X, Y, box) in t["pos"].items():
            box_to_tid[fr][(round(box[0], 1), round(box[1], 1),
                            round(box[2], 1), round(box[3], 1))] = tid
    vo.fill_and_smooth(tracks)  # この後 pos は平滑化済み（priors 用）
    print(f"トラック数: {len(tracks)}（速度予測トラッキング）", flush=True)

    # 旧frame → そのキーフレームの選手ボックス（1920基準, group）
    # 中間フレーム割当用に YOLO を遅延ロード
    from ultralytics import YOLO
    model = YOLO(str(pa.YOLO_MODEL_PATH))
    knn = br.load_knn_model()
    threshold, green_high = br.load_calibration()
    print("色判定:", "kNNモデル" if knn else f"緑比率しきい値{threshold:.3f}", flush=True)

    corr_rows = []     # box_corrections_hi 行
    track_rows = []    # tracks_hi 行
    added_seq = -1

    def emit(jframe, x1, y1, x2, y2, group, tid=None):
        nonlocal added_seq
        key = added_seq
        added_seq -= 1
        corr_rows.append({"frame": jframe, "box_key": key, "action": "add",
                          "x1": round(x1, 1), "y1": round(y1, 1),
                          "x2": round(x2, 1), "y2": round(y2, 1),
                          "orig_player_id": -1, "group": group})
        if tid is not None:
            fx, fy = (x1 + x2) / 2.0, y2
            X, Y = foot_to_pitch(H, fx, fy)
            track_rows.append({"frame": jframe, "track_id": tid, "team": group,
                               "x_img": round(fx, 1), "y_img": round(fy, 1),
                               "x_pitch": round(X, 2), "y_pitch": round(Y, 2)})

    n_inter_det = 0
    n_inter_interp = 0
    processed = 0
    total_pairs = len(old_frames) - 1

    for oi in range(len(old_frames)):
        i_old = old_frames[oi]
        jkey = i_old * factor  # キーフレームの新frame index

        # --- キーフレーム: 旧手作業ボックスをそのまま ---
        for b in old_boxes.get(i_old, []):
            if b["deleted"] or b["g"] not in (1, 2):
                continue
            tid = box_to_tid.get(i_old, {}).get(
                (round(b["x1"], 1), round(b["y1"], 1), round(b["x2"], 1), round(b["y2"], 1)))
            emit(jkey, b["x1"], b["y1"], b["x2"], b["y2"], b["g"], tid)

        # --- 中間フレーム（次の旧frameとの間 factor-1 枚）---
        if oi + 1 >= len(old_frames):
            continue
        i_next = old_frames[oi + 1]
        if i_next != i_old + 1:
            continue  # 連続でない（スキップ跨ぎ）→中間は埋めない

        for k in range(1, factor):
            jmid = jkey + k
            sub_fr = i_old + k / factor   # 旧frame小数（補間用）
            img_no = jmid + 1
            path = frames_dir / f"frame_{img_no:06d}.jpg"
            if not path.exists():
                continue
            pil = Image.open(path)

            # トラックの prior（この小数frameでの位置, 1920基準box）
            priors = []
            for tid, t in tracks.items():
                r = interp_track_pos(t, sub_fr)
                if r is None:
                    continue
                X, Y, box = r
                priors.append({"tid": tid, "team": t["team"], "X": X, "Y": Y, "box": box})

            # YOLO検出（新実画像）→ 1920基準box, foot→pitch, 暫定team
            res = model.predict(source=str(path), classes=[0],
                                conf=args.conf, imgsz=args.imgsz, verbose=False)
            dets = []
            for rr in res:
                if rr.boxes is None:
                    continue
                for xyxy in rr.boxes.xyxy.cpu().numpy():
                    X1, Y1, X2, Y2 = (float(v) for v in xyxy[:4])
                    if X2 - X1 < 2 or Y2 - Y1 < 2:
                        continue
                    # 1920基準へ換算
                    bx1, by1, bx2, by2 = X1*to_base, Y1*to_base, X2*to_base, Y2*to_base
                    fx, fy = (bx1 + bx2) / 2.0, by2
                    px, py = foot_to_pitch(H, fx, fy)
                    crop = br.torso_crop(pil, X1, Y1, X2, Y2)  # cropは実画像座標
                    g = 3 if crop is None else br.predict_group(crop, knn, threshold, green_high)
                    dets.append({"box": (bx1, by1, bx2, by2), "X": px, "Y": py,
                                 "knn": g, "used": False})

            # prior↔det 同チーム優先・最近傍マッチ
            pairs = []
            for pi, pr in enumerate(priors):
                for di, d in enumerate(dets):
                    dist = ((pr["X"] - d["X"]) ** 2 + (pr["Y"] - d["Y"]) ** 2) ** 0.5
                    if dist <= MATCH_THRESH_M:
                        # 同チームを優先（距離に小ペナルティ）
                        cost = dist + (0.0 if d["knn"] == pr["team"] else 1.5)
                        pairs.append((cost, pi, di))
            pairs.sort()
            used_p, used_d = set(), set()
            for cost, pi, di in pairs:
                if pi in used_p or di in used_d:
                    continue
                used_p.add(pi); used_d.add(di)
                pr = priors[pi]; d = dets[di]
                d["used"] = True
                bx1, by1, bx2, by2 = d["box"]
                emit(jmid, bx1, by1, bx2, by2, pr["team"], pr["tid"])  # 実box＋手作業由来チーム
                n_inter_det += 1
            # 未マッチ検出 → kNN色で採用（キーフレームに無い余剰選手）
            for d in dets:
                if d["used"] or d["knn"] not in (1, 2):
                    continue
                bx1, by1, bx2, by2 = d["box"]
                emit(jmid, bx1, by1, bx2, by2, d["knn"], None)
            # 未マッチトラック → 補間ボックスで補完（消さない）
            for pi, pr in enumerate(priors):
                if pi in used_p:
                    continue
                bx1, by1, bx2, by2 = pr["box"]
                emit(jmid, bx1, by1, bx2, by2, pr["team"], pr["tid"])
                n_inter_interp += 1

        processed += 1
        if processed % 50 == 0 or processed == total_pairs:
            print(f"  {processed}/{total_pairs} 区間  "
                  f"(中間: 検出割当{n_inter_det} / 補間補完{n_inter_interp})", flush=True)

    # 書き出し
    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "box_key", "action", "x1", "y1", "x2", "y2", "orig_player_id", "group"]
    with OUT_CORR.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(corr_rows)
    with OUT_TRACKS.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "track_id", "team",
                                          "x_img", "y_img", "x_pitch", "y_pitch"])
        w.writeheader()
        w.writerows(track_rows)
    # メタデータ（12fps）
    meta = [{"frame_idx": j, "timestamp": round(j / (2 * factor), 3),
             "filename": f"frame_{j+1:06d}.jpg"} for j in range(n_hi)]
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"\n完了:", flush=True)
    print(f"  {OUT_CORR}  {len(corr_rows)}行（全add・1920基準）", flush=True)
    print(f"  {OUT_TRACKS}  {len(track_rows)}行（track_id付き）", flush=True)
    print(f"  {OUT_META}  {len(meta)}フレーム分(12fps)", flush=True)
    print(f"  中間フレーム: 実検出割当 {n_inter_det} / 補間補完 {n_inter_interp}", flush=True)


if __name__ == "__main__":
    main()
