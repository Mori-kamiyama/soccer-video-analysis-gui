"""
指定フレーム以降を YOLO(COCO person) で再検出し、色を自動判定して
box_corrections.csv に書き出すバッチ処理。

photo_annotator.py の「このFをYOLO再検出」「自動色判定」をフレーム範囲に対して一括実行する。
非破壊: 元データ(player_positions_all.csv)は読むだけ。結果は box_corrections.csv に保存。
指定フレーム未満の既存の手修正はそのまま温存する。

使い方:
  .venv/bin/python batch_redetect.py --start-image 379
  .venv/bin/python batch_redetect.py --start-image 379 --end-image 1312
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


def load_knn_model():
    if not pa.COLOR_MODEL_JSON.exists():
        return None
    data = json.loads(pa.COLOR_MODEL_JSON.read_text())
    if data.get("type") != "knn_color_v1":
        return None
    feats = np.asarray(data.get("features", []), dtype=np.float32)
    labels = np.asarray(data.get("labels", []), dtype=np.int32)
    if feats.ndim != 2 or len(feats) == 0 or len(feats) != len(labels):
        return None
    mean = np.asarray(data.get("mean", np.zeros(feats.shape[1])), dtype=np.float32)
    std = np.asarray(data.get("std", np.ones(feats.shape[1])), dtype=np.float32)
    std[std < 1e-6] = 1.0
    return {
        "z_train": (feats - mean) / std,
        "labels": labels,
        "mean": mean,
        "std": std,
        "k": max(1, min(int(data.get("k", 7)), len(labels))),
    }


def load_calibration():
    """kNNが使えないときのフォールバック用（緑比率しきい値）。"""
    threshold = pa.AUTO_GREEN_THRESH
    green_high_is_a = True
    if pa.COLOR_CALIB_JSON.exists():
        try:
            d = json.loads(pa.COLOR_CALIB_JSON.read_text())
            threshold = float(d.get("threshold", threshold))
            green_high_is_a = bool(d.get("green_high_is_a", True))
        except Exception:
            pass
    return threshold, green_high_is_a


def predict_group(crop: Image.Image, model, threshold: float, green_high_is_a: bool) -> int:
    """胴体cropからチームグループ(1=緑/2=黒/3=不明)を返す。"""
    feature = pa.color_feature_from_crop(crop)
    if feature is not None and model is not None:
        x = np.asarray(feature, dtype=np.float32)
        if x.shape[0] == model["z_train"].shape[1]:
            z = (x - model["mean"]) / model["std"]
            dist = np.linalg.norm(model["z_train"] - z, axis=1)
            idx = np.argsort(dist)[: model["k"]]
            votes = {1: 0.0, 2: 0.0}
            for i in idx:
                votes[int(model["labels"][i])] += 1.0 / (float(dist[i]) + 1e-6)
            return 1 if votes[1] >= votes[2] else 2
    # フォールバック: 緑比率しきい値
    if hasattr(crop, "get_flattened_data"):
        pixels = list(crop.get_flattened_data())
    else:
        pixels = list(crop.getdata())
    ratio = pa.green_ratio_from_rgb_pixels(pixels)
    if ratio is None:
        return 3
    if green_high_is_a:
        return 1 if ratio >= threshold else 2
    return 1 if ratio < threshold else 2


def torso_crop(pil: Image.Image, x1, y1, x2, y2):
    """photo_annotator._box_torso_crop と同じ胴体切り出し。"""
    ix1 = max(0, int(min(x1, x2)))
    ix2 = int(max(x1, x2))
    iy1 = max(0, int(min(y1, y2)))
    iy2 = int(max(y1, y2))
    w, h_img = pil.size
    ix2 = min(w, ix2)
    iy2 = min(h_img, iy2)
    h = iy2 - iy1
    if h < 8 or ix2 - ix1 < 4:
        return None
    ty1 = iy1 + int(h * pa.TORSO_TOP)
    ty2 = iy1 + int(h * pa.TORSO_BOTTOM)
    return pil.crop((ix1, ty1, ix2, ty2)).convert("RGB")


def load_original_detection_ids() -> dict[int, list[tuple[int, int]]]:
    """frame -> [(detection_id, group), ...]  元検出（削除マーク用）。"""
    by_frame: dict[int, list[tuple[int, int]]] = defaultdict(list)
    with pa.DET_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                frame = int(r["frame"])
                key = int(r["detection_id"])
            except (ValueError, KeyError):
                continue
            group = pa.group_from_team_hint(r.get("team_hint", r.get("team", "")))
            by_frame[frame].append((key, group))
    return by_frame


def load_existing_corrections_below(start_frame: int) -> list[dict]:
    """start_frame 未満の既存修正行をそのまま温存して返す。"""
    kept = []
    if not pa.CORRECTIONS_CSV.exists():
        return kept
    with pa.CORRECTIONS_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                if int(r["frame"]) < start_frame:
                    kept.append(r)
            except (ValueError, KeyError):
                continue
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-image", type=int, default=379,
                    help="この画像番号(frame_XXXXXX.jpg)以降を再検出（1始まり）")
    ap.add_argument("--end-image", type=int, default=None,
                    help="終了画像番号（省略時は最後まで）")
    ap.add_argument("--conf", type=float, default=pa.YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=pa.YOLO_IMGSZ)
    args = ap.parse_args()

    start_frame = args.start_image - 1  # 画像番号 → frame index

    # 対象フレーム（画像ファイルが存在するもの）
    frame_files = []
    for p in sorted(pa.FRAMES_DIR.glob("frame_*.jpg")):
        try:
            num = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if num < args.start_image:
            continue
        if args.end_image is not None and num > args.end_image:
            continue
        frame_files.append((num - 1, p))  # (frame index, path)

    if not frame_files:
        raise SystemExit("対象フレームがありません")

    print(f"対象: {len(frame_files)} フレーム "
          f"(画像 {frame_files[0][0]+1} 〜 {frame_files[-1][0]+1})")

    from ultralytics import YOLO
    model_yolo = YOLO(str(pa.YOLO_MODEL_PATH))
    knn = load_knn_model()
    threshold, green_high_is_a = load_calibration()
    print("色判定:", "kNNモデル" if knn else f"緑比率しきい値 {threshold:.3f}")

    orig_dets = load_original_detection_ids()
    kept_rows = load_existing_corrections_below(start_frame)
    print(f"温存する既存修正(frame<{start_frame}): {len(kept_rows)} 行")

    # 追加ボックスの key は負の連番（既存修正と衝突しないよう -1 から下げる）
    added_seq = -1
    new_rows: list[dict] = []
    total_boxes = 0
    group_count = {1: 0, 2: 0, 3: 0}

    for i, (frame, path) in enumerate(frame_files):
        pil = Image.open(path)
        results = model_yolo.predict(source=str(path), classes=[0],
                                     conf=args.conf, imgsz=args.imgsz, verbose=False)
        dets = []
        for res in results:
            if res.boxes is None:
                continue
            for xyxy in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    dets.append((x1, y1, x2, y2))

        # 元検出を削除マーク
        for det_id, group in orig_dets.get(frame, []):
            new_rows.append({"frame": frame, "box_key": det_id, "action": "delete",
                             "x1": "", "y1": "", "x2": "", "y2": "",
                             "orig_player_id": det_id, "group": group})

        # 新規検出を追加＋色判定
        for x1, y1, x2, y2 in dets:
            crop = torso_crop(pil, x1, y1, x2, y2)
            group = 3 if crop is None else predict_group(crop, knn, threshold, green_high_is_a)
            group_count[group] += 1
            key = added_seq
            added_seq -= 1
            new_rows.append({"frame": frame, "box_key": key, "action": "add",
                             "x1": round(x1, 1), "y1": round(y1, 1),
                             "x2": round(x2, 1), "y2": round(y2, 1),
                             "orig_player_id": -1, "group": group})
            total_boxes += 1

        if (i + 1) % 50 == 0 or i + 1 == len(frame_files):
            print(f"  {i+1}/{len(frame_files)} フレーム処理  "
                  f"(累計 検出{total_boxes} / 緑{group_count[1]} 黒{group_count[2]} 不明{group_count[3]})")

    # 書き出し（温存行 + 新規行）
    fields = ["frame", "box_key", "action", "x1", "y1", "x2", "y2", "orig_player_id", "group"]
    with pa.CORRECTIONS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in kept_rows:
            w.writerow({k: r.get(k, "") for k in fields})
        w.writerows(new_rows)

    print("\n完了。")
    print(f"  検出ボックス: {total_boxes} 個 "
          f"(緑{group_count[1]} / 黒{group_count[2]} / 不明{group_count[3]})")
    print(f"  {pa.CORRECTIONS_CSV} に保存（既存 {len(kept_rows)} 行 + 新規 {len(new_rows)} 行）")
    print("  元データ player_positions_all.csv は変更していません。")
    print("  photo_annotator.py を起動すると反映されています。")


if __name__ == "__main__":
    main()
