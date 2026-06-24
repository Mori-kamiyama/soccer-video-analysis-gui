"""
YOLO 充填率 + 色存在 + パースサイズチェック＆自動スナップ修正

【ボックスの実在スコア（3指標 max を採用後、サイズで掛け算）】
  1. YOLO 近傍スコア : 最近傍 YOLO 検出との中心距離 ≤ snap_dist → 1.0
  2. 色存在スコア   : ボックス周辺の緑/暗色ピクセル率 ≥ color_thresh → 1.0
  サイズスコア     : y位置から期待される高さの ratio が許容範囲内 → 1.0（乗数）

【修正内容（--fix 指定時）】
  YOLO 一致ボックス  → YOLO 座標にスナップ（チーム維持）
  色のみ確認ボックス → 座標そのまま保持
  どちらも無し       → 削除（ファントム）
  --add-missing 時   → 既存ボックスが無い YOLO 検出を追加

起動例:
  # サイズモデル構築
  .venv/bin/python src/snap_to_yolo.py --build-size-model

  # 全データスキャン（修正なし）
  .venv/bin/python src/snap_to_yolo.py --size-model outputs/size_model.json

  # 全データ修正実行
  .venv/bin/python src/snap_to_yolo.py \\
      --size-model outputs/size_model.json \\
      --fill-thresh 0.8 --valid-thresh 0.5 --add-missing --fix
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

import photo_annotator as pa

CORRECTIONS_CSV = Path("outputs/box_corrections.csv")
PLAYERS_HI_CSV  = Path("outputs/players_hi.csv")
SIZE_MODEL_PATH = Path("outputs/size_model.json")
BACKUP_SUFFIX   = ".snap_backup"
YOLO_BATCH      = 16   # バッチ推論枚数


def detect_coord_scale() -> float:
    """box_corrections.csv は 1920 基準座標で保存されている。
    現フレーム画像幅 / 1920 を掛けると YOLO の実ピクセル座標になる。"""
    files = sorted(pa.FRAMES_DIR.glob("frame_*.jpg"))
    if files:
        try:
            w, _ = Image.open(files[0]).size
            return w / 1920.0
        except Exception:
            pass
    return 1.0


# ---------------------------------------------------------------------------
# サイズモデル
# ---------------------------------------------------------------------------

def build_size_model(
    min_ratio: float = 0.5,
    max_ratio: float = 1.5,
    save_path: Path = SIZE_MODEL_PATH,
) -> dict:
    frame_counts: dict[int, int] = defaultdict(int)
    with PLAYERS_HI_CSV.open() as f:
        for row in csv.DictReader(f):
            frame_counts[int(row["frame"])] += 1
    good_frames = {f for f, n in frame_counts.items() if n == 22}
    print(f"22人ちょうどのフレーム: {len(good_frames)} 件")

    ys, hs = [], []
    with CORRECTIONS_CSV.open() as f:
        for row in csv.DictReader(f):
            if int(row["frame"]) not in good_frames:
                continue
            try:
                y1, y2 = float(row["y1"]), float(row["y2"])
                h = y2 - y1
                yc = (y1 + y2) / 2
                if h > 0:
                    ys.append(yc); hs.append(h)
            except ValueError:
                continue

    ys_arr, hs_arr = np.array(ys), np.array(hs)
    coef = np.polyfit(ys_arr, hs_arr, 1)
    slope, intercept = float(coef[0]), float(coef[1])
    predicted = np.polyval(coef, ys_arr)
    r2 = 1 - np.var(hs_arr - predicted) / np.var(hs_arr)
    ratio = hs_arr / predicted

    model = {
        "slope": slope, "intercept": intercept,
        "min_ratio": min_ratio, "max_ratio": max_ratio,
        "n_samples": len(ys), "r2": float(r2),
        "ratio_p5": float(np.percentile(ratio, 5)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        "y_range": [float(ys_arr.min()), float(ys_arr.max())],
        "h_mean": float(hs_arr.mean()),
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(model, indent=2, ensure_ascii=False))
    print(f"フィット: height = {slope:.4f} * y + {intercept:.2f}  R²={r2:.4f}")
    print(f"ratio (5th〜95th): {model['ratio_p5']:.2f}〜{model['ratio_p95']:.2f}  "
          f"許容: [{min_ratio}, {max_ratio}]")
    print(f"サンプル数: {len(ys)}  保存: {save_path}")
    return model


def load_size_model(path: Path) -> dict:
    return json.loads(path.read_text())


def _size_score(y_center: float, height: float, model: dict) -> float:
    expected = model["slope"] * y_center + model["intercept"]
    if expected <= 0:
        return 1.0
    ratio = height / expected
    lo, hi = model["min_ratio"], model["max_ratio"]
    if lo <= ratio <= hi:
        return 1.0
    return max(0.0, ratio / lo) if ratio < lo else max(0.0, hi / ratio)


# ---------------------------------------------------------------------------
# 色スコア（numpy vectorized）
# ---------------------------------------------------------------------------

def _color_score(pil: Image.Image, x1, y1, x2, y2, expand_px: int) -> float:
    ix1 = max(0, int(min(x1, x2)) - expand_px)
    ix2 = min(pil.width, int(max(x1, x2)) + expand_px)
    iy1 = max(0, int(min(y1, y2)))
    iy2 = min(pil.height, int(max(y1, y2)))
    h = iy2 - iy1
    if h < 4 or ix2 - ix1 < 4:
        return 0.0
    ty1 = iy1 + int(h * pa.TORSO_TOP)
    ty2 = iy1 + int(h * pa.TORSO_BOTTOM)
    if ty2 <= ty1:
        ty2 = iy2
    arr = np.array(pil.crop((ix1, ty1, ix2, ty2)).convert("RGB"),
                   dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(maxc > 1e-6, delta / maxc, 0.0)
        h_arr = np.zeros_like(maxc)
        safe = delta > 1e-6
        mg = safe & (maxc == g)
        mb = safe & (maxc == b)
        mr = safe & (maxc == r) & ~mg & ~mb
        h_arr[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6 / 6.0
        h_arr[mg] = ((b[mg] - r[mg]) / delta[mg] + 2) / 6.0
        h_arr[mb] = ((r[mb] - g[mb]) / delta[mb] + 4) / 6.0
    is_green = (h_arr > 0.17) & (h_arr < 0.42) & (s > 0.25) & (maxc > 0.20)
    is_dark  = maxc < 0.28
    total = h_arr.size
    return float((is_green | is_dark).sum()) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# チーム色判定
# ---------------------------------------------------------------------------

def _predict_group(crop: Image.Image, knn, threshold: float,
                   green_high_is_a: bool) -> int:
    feature = pa.color_feature_from_crop(crop)
    if feature is not None and knn is not None:
        x = np.asarray(feature, dtype=np.float32)
        if x.shape[0] == knn["z_train"].shape[1]:
            z = (x - knn["mean"]) / knn["std"]
            dist = np.linalg.norm(knn["z_train"] - z, axis=1)
            idx = np.argsort(dist)[: knn["k"]]
            votes = {1: 0.0, 2: 0.0}
            for i in idx:
                votes[int(knn["labels"][i])] += 1.0 / (float(dist[i]) + 1e-6)
            return 1 if votes[1] >= votes[2] else 2
    pixels = list(crop.getdata())
    ratio = pa.green_ratio_from_rgb_pixels(pixels)
    if ratio is None:
        return 3
    return (1 if ratio >= threshold else 2) if green_high_is_a else (1 if ratio < threshold else 2)


def _load_knn():
    if not pa.COLOR_MODEL_JSON.exists():
        return None
    data = json.loads(pa.COLOR_MODEL_JSON.read_text())
    if data.get("type") != "knn_color_v1":
        return None
    feats  = np.asarray(data.get("features", []), dtype=np.float32)
    labels = np.asarray(data.get("labels", []), dtype=np.int32)
    if feats.ndim != 2 or len(feats) == 0 or len(feats) != len(labels):
        return None
    mean = np.asarray(data.get("mean", np.zeros(feats.shape[1])), dtype=np.float32)
    std  = np.asarray(data.get("std",  np.ones(feats.shape[1])), dtype=np.float32)
    std[std < 1e-6] = 1.0
    return {"z_train": (feats - mean) / std, "labels": labels, "mean": mean, "std": std,
            "k": max(1, min(int(data.get("k", 7)), len(labels)))}


# ---------------------------------------------------------------------------
# CSV ロード / セーブ
# ---------------------------------------------------------------------------

def _load_corrections() -> dict[int, list[dict]]:
    fm: dict[int, list[dict]] = defaultdict(list)
    with CORRECTIONS_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fm[int(row["frame"])].append(row)
    return fm


def _save_corrections(frame_map: dict[int, list[dict]]) -> None:
    fieldnames = ["frame", "box_key", "action", "x1", "y1", "x2", "y2",
                  "orig_player_id", "group"]
    backup = CORRECTIONS_CSV.with_suffix(CORRECTIONS_CSV.suffix + BACKUP_SUFFIX)
    shutil.copy2(CORRECTIONS_CSV, backup)
    print(f"バックアップ: {backup}")
    with CORRECTIONS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for frame in sorted(frame_map):
            for row in frame_map[frame]:
                w.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"保存: {CORRECTIONS_CSV}")


# ---------------------------------------------------------------------------
# バッチ YOLO 推論
# ---------------------------------------------------------------------------

def _batch_yolo(yolo_model, frames: list[int], conf: float, imgsz: int,
                batch: int = YOLO_BATCH) -> dict[int, list[tuple]]:
    """frames → {frame: [(x1,y1,x2,y2), ...]} をバッチ処理で返す。"""
    pairs = [(f, pa.FRAMES_DIR / f"frame_{f+1:06d}.jpg") for f in frames]
    pairs = [(f, p) for f, p in pairs if p.exists()]
    result: dict[int, list[tuple]] = {f: [] for f in frames}

    for i in range(0, len(pairs), batch):
        chunk = pairs[i: i + batch]
        paths = [str(p) for _, p in chunk]
        ress = yolo_model.predict(source=paths, classes=[0],
                                  conf=conf, imgsz=imgsz, verbose=False)
        for (fr, _), res in zip(chunk, ress):
            dets = []
            if res.boxes is not None:
                for xyxy in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
                    if x2 - x1 >= 2 and y2 - y1 >= 2:
                        dets.append((x1, y1, x2, y2))
            result[fr] = dets
    return result


# ---------------------------------------------------------------------------
# スコアリング
# ---------------------------------------------------------------------------

def _center_dist(b1: tuple, b2: tuple) -> float:
    return (((b1[0]+b1[2])/2 - (b2[0]+b2[2])/2)**2 +
            ((b1[1]+b1[3])/2 - (b2[1]+b2[3])/2)**2) ** 0.5


def _score_boxes(
    boxes: list[dict],
    yolo_dets: list[tuple],
    pil: Image.Image | None,
    snap_dist: float,
    color_thresh: float,
    expand_px: int,
    size_model: dict | None,
) -> tuple[list[float], dict[int, int]]:
    if not boxes:
        return [], {}

    coords = [(float(b["x1"]), float(b["y1"]),
               float(b["x2"]), float(b["y2"])) for b in boxes]

    # YOLO greedy マッチング
    yolo_match: dict[int, int] = {}
    yolo_used: set[int] = set()
    if yolo_dets:
        pairs = sorted((_center_dist(bc, yc), bi, yi)
                       for bi, bc in enumerate(coords)
                       for yi, yc in enumerate(yolo_dets))
        for d, bi, yi in pairs:
            if d > snap_dist:
                break
            if bi not in yolo_match and yi not in yolo_used:
                yolo_match[bi] = yi
                yolo_used.add(yi)

    scores: list[float] = []
    for bi, bc in enumerate(coords):
        x1, y1, x2, y2 = bc
        yc = (y1 + y2) / 2
        h  = y2 - y1
        y_sc = 1.0 if bi in yolo_match else 0.0
        c_sc = 0.0
        if pil is not None:
            cs = _color_score(pil, x1, y1, x2, y2, expand_px)
            c_sc = min(1.0, cs / max(color_thresh, 1e-6))
        sz_sc = _size_score(yc, h, size_model) if size_model else 1.0
        scores.append(max(y_sc, c_sc) * sz_sc)

    return scores, yolo_match


# ---------------------------------------------------------------------------
# フレーム修正
# ---------------------------------------------------------------------------

def _fix_frame(
    boxes: list[dict], yolo_dets: list[tuple],
    pil: Image.Image | None,
    snap_dist: float, color_thresh: float, expand_px: int,
    size_model: dict | None, valid_thresh: float,
    add_missing: bool, min_box_key: int,
    knn, threshold: float, green_high_is_a: bool, frame: int,
) -> tuple[list[dict], int, dict]:
    stats = {"snapped": 0, "color_kept": 0, "removed": 0, "added": 0}
    scores, yolo_match = _score_boxes(boxes, yolo_dets, pil,
                                      snap_dist, color_thresh, expand_px, size_model)
    yolo_used = set(yolo_match.values())
    new_rows: list[dict] = []

    for bi, box in enumerate(boxes):
        if scores[bi] < valid_thresh:
            stats["removed"] += 1
        elif bi in yolo_match:
            x1, y1, x2, y2 = yolo_dets[yolo_match[bi]]
            new_rows.append({**box, "x1": f"{x1:.1f}", "y1": f"{y1:.1f}",
                             "x2": f"{x2:.1f}", "y2": f"{y2:.1f}"})
            stats["snapped"] += 1
        else:
            new_rows.append(box)
            stats["color_kept"] += 1

    if add_missing:
        img_path = pa.FRAMES_DIR / f"frame_{frame+1:06d}.jpg"
        pil_add = pil or (Image.open(img_path) if img_path.exists() else None)
        for yi, (x1, y1, x2, y2) in enumerate(yolo_dets):
            if yi in yolo_used:
                continue
            group = 3
            if pil_add:
                h = y2 - y1
                crop = pil_add.crop((int(x1), int(y1 + h*pa.TORSO_TOP),
                                     int(x2), int(y1 + h*pa.TORSO_BOTTOM))).convert("RGB")
                if crop.width > 0 and crop.height > 0:
                    group = _predict_group(crop, knn, threshold, green_high_is_a)
            new_rows.append({
                "frame": str(frame), "box_key": str(min_box_key), "action": "add",
                "x1": f"{x1:.1f}", "y1": f"{y1:.1f}",
                "x2": f"{x2:.1f}", "y2": f"{y2:.1f}",
                "orig_player_id": "-1", "group": str(group),
            })
            min_box_key -= 1
            stats["added"] += 1

    return new_rows, min_box_key, stats


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-size-model", action="store_true",
                    help="サイズモデルを構築して保存")
    ap.add_argument("--size-min-ratio", type=float, default=0.5)
    ap.add_argument("--size-max-ratio", type=float, default=1.5)

    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end",   type=int, default=None)
    ap.add_argument("--fill-thresh",  type=float, default=0.8,
                    help="この充填率未満のフレームを修正対象に（デフォルト: 0.8）")
    ap.add_argument("--valid-thresh", type=float, default=0.5,
                    help="このスコア未満のボックスを削除（デフォルト: 0.5）")
    ap.add_argument("--snap-dist",    type=float, default=30.0)
    ap.add_argument("--color-thresh", type=float, default=0.10)
    ap.add_argument("--expand-px",    type=int,   default=10)
    ap.add_argument("--size-model",   type=Path,  default=None)
    ap.add_argument("--conf",   type=float, default=pa.YOLO_CONF)
    ap.add_argument("--imgsz",  type=int,   default=pa.YOLO_IMGSZ)
    ap.add_argument("--batch",  type=int,   default=YOLO_BATCH,
                    help=f"YOLO バッチサイズ（デフォルト: {YOLO_BATCH}）")
    ap.add_argument("--report-csv", type=Path, default=None,
                    help="スキャン結果を全件CSVに保存")
    ap.add_argument("--no-yolo", action="store_true",
                    help="YOLO を実行せず色+サイズチェックのみ（高速）")
    ap.add_argument("--add-missing", action="store_true",
                    help="YOLO のみ検出のボックスも追加")
    ap.add_argument("--fix", action="store_true",
                    help="実際に box_corrections.csv を修正")
    args = ap.parse_args()

    if args.build_size_model:
        build_size_model(args.size_min_ratio, args.size_max_ratio)
        return

    size_model: dict | None = None
    if args.size_model:
        size_model = load_size_model(args.size_model)
        print(f"サイズモデル: height = {size_model['slope']:.4f}*y "
              f"+ {size_model['intercept']:.2f}  R²={size_model['r2']:.4f}  "
              f"ratio許容[{size_model['min_ratio']}, {size_model['max_ratio']}]")

    frame_map = _load_corrections()
    all_frames = sorted(frame_map.keys())
    target_frames = [
        f for f in all_frames
        if (args.start is None or f >= args.start)
        and (args.end is None or f <= args.end)
    ]
    print(f"対象フレーム: {len(target_frames)} 件  "
          f"({target_frames[0]}〜{target_frames[-1]})")
    print(f"fill_thresh={args.fill_thresh}  valid_thresh={args.valid_thresh}  "
          f"snap_dist={args.snap_dist}px  color_thresh={args.color_thresh}  "
          f"expand={args.expand_px}px  batch={args.batch}")

    yolo_model = None
    if not args.no_yolo:
        from ultralytics import YOLO as UltraYOLO
        yolo_model = UltraYOLO(str(pa.YOLO_MODEL_PATH))

    knn = None
    threshold, green_high_is_a = pa.AUTO_GREEN_THRESH, True
    if args.add_missing:
        knn = _load_knn()
        if pa.COLOR_CALIB_JSON.exists():
            try:
                d = json.loads(pa.COLOR_CALIB_JSON.read_text())
                threshold = float(d.get("threshold", threshold))
                green_high_is_a = bool(d.get("green_high_is_a", True))
            except Exception:
                pass
        print("チーム色判定:", "kNN" if knn else f"緑比率 thr={threshold:.3f}")

    # --- YOLO 推論（--no-yolo 時はスキップ）---
    REPORT_EVERY = 500
    all_yolo: dict[int, list[tuple]] = {f: [] for f in target_frames}
    if args.no_yolo:
        print("\n--no-yolo: YOLO 推論をスキップ（色+サイズのみ）")
    else:
        print("\n--- YOLO バッチ推論中 ---")
        for i in range(0, len(target_frames), args.batch):
            chunk = target_frames[i: i + args.batch]
            all_yolo.update(_batch_yolo(yolo_model, chunk, args.conf, args.imgsz, args.batch))
            done = min(i + args.batch, len(target_frames))
            if done % REPORT_EVERY < args.batch or done == len(target_frames):
                print(f"  {done}/{len(target_frames)} frames", flush=True)

    # --- 画像を並列ロードしながらスコアリング ---
    print("\n--- スコアリング中 ---")

    def _load_pil(frame: int) -> Image.Image | None:
        p = pa.FRAMES_DIR / f"frame_{frame+1:06d}.jpg"
        return Image.open(p) if p.exists() else None

    low_fill: list[tuple] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_load_pil, f): f for f in target_frames}
        pil_cache: dict[int, Image.Image | None] = {}
        for fut in futures:
            pil_cache[futures[fut]] = fut.result()

    for i, frame in enumerate(target_frames):
        if (i + 1) % REPORT_EVERY == 0 or i == len(target_frames) - 1:
            print(f"  {i+1}/{len(target_frames)} frame={frame}", flush=True)
        boxes     = frame_map[frame]
        yolo_dets = all_yolo.get(frame, [])
        pil       = pil_cache.get(frame)
        scores, yolo_match = _score_boxes(
            boxes, yolo_dets, pil,
            args.snap_dist, args.color_thresh, args.expand_px, size_model,
        )
        fill = sum(scores) / len(scores) if scores else 1.0
        if fill < args.fill_thresh:
            yolo_cnt   = len(yolo_match)
            color_only = sum(1 for bi, s in enumerate(scores)
                             if s >= args.valid_thresh and bi not in yolo_match)
            phantom    = sum(1 for s in scores if s < args.valid_thresh)
            low_fill.append((frame, fill, len(boxes), len(yolo_dets),
                             yolo_cnt, color_only, phantom))

    # --- レポート ---
    print(f"\n充填率 < {args.fill_thresh} のフレーム: {len(low_fill)} 件")
    total_phantom = sum(r[6] for r in low_fill)
    total_missing = sum(max(0, 22 - (r[2] - r[6])) for r in low_fill)
    print(f"削除予定（幽霊）: {total_phantom} ボックス合計")
    print(f"不足（add-missing で補完予定）: {total_missing} ボックス合計")
    if low_fill:
        print(f"\n{'frame':>7}  {'充填率':>6}  {'既存':>5}  {'YOLO':>5}  "
              f"{'YOLOm':>5}  {'色のみ':>6}  {'幽霊':>5}")
        for row in low_fill[:60]:
            frame, fill, nb, ny, ym, co, ph = row
            print(f"{frame:>7}  {fill:>6.2f}  {nb:>5}  {ny:>5}  "
                  f"{ym:>5}  {co:>6}  {ph:>5}")
        if len(low_fill) > 60:
            print(f"  … 残り {len(low_fill)-60} 件（省略）")

    if args.report_csv and low_fill:
        import csv as _csv
        with args.report_csv.open("w", newline="", encoding="utf-8") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["frame", "fill", "n_boxes", "n_yolo",
                         "yolo_match", "color_only", "phantom", "remaining"])
            for frame, fill, nb, ny, ym, co, ph in low_fill:
                _w.writerow([frame, f"{fill:.4f}", nb, ny, ym, co, ph, nb - ph])
        print(f"レポート保存: {args.report_csv}  ({len(low_fill)} 件)")

    if not args.fix:
        print("\n--fix を付けると修正を実行します。")
        return
    if not low_fill:
        print("修正対象フレームなし。終了。")
        return

    print(f"\n--- 修正実行 ({len(low_fill)} フレーム) ---")
    min_key = min(
        int(r["box_key"])
        for rows in frame_map.values()
        for r in rows if r["box_key"]
    ) - 1

    total_stats = {"snapped": 0, "color_kept": 0, "removed": 0, "added": 0}
    for frame, *_ in low_fill:
        boxes     = frame_map[frame]
        yolo_dets = all_yolo.get(frame, [])
        pil       = pil_cache.get(frame)
        new_rows, min_key, stats = _fix_frame(
            boxes, yolo_dets, pil,
            args.snap_dist, args.color_thresh, args.expand_px, size_model,
            args.valid_thresh, args.add_missing,
            min_key, knn, threshold, green_high_is_a, frame,
        )
        frame_map[frame] = new_rows
        for k in total_stats:
            total_stats[k] += stats[k]

    print(f"スナップ:   {total_stats['snapped']} ボックス")
    print(f"色のみ保持: {total_stats['color_kept']} ボックス")
    print(f"削除:       {total_stats['removed']} ボックス")
    if args.add_missing:
        print(f"追加:       {total_stats['added']} ボックス")

    _save_corrections(frame_map)
    print("完了。")


if __name__ == "__main__":
    main()
