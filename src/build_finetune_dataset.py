"""
アノテーション(box_corrections.csv)からYOLO finetune用データセットを構築する。

各フレームの「正解選手ボックス」を再構成:
  元検出(player_positions_all) − delete + move補正 + add（手動追加）
identity(group)は検出には不要なので無視。単一クラス "player"。

出力:
  dataset/images/{train,val}/*.jpg   （フレーム画像のコピー）
  dataset/labels/{train,val}/*.txt   （YOLO形式 class cx cy w h, 正規化）
  dataset/data.yaml
"""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

FRAMES = Path("frames")
POS_CSV = Path("outputs/player_positions_all.csv")
CORR_CSV = Path("outputs/box_corrections.csv")
DATASET = Path("dataset")
W, H = 1920, 1080
VAL_RATIO = 0.18  # 末尾の連続ブロックをvalにする（時間的に近い漏れを避ける）


def build_boxes_per_frame() -> dict[int, list[tuple[float, float, float, float]]]:
    # 元検出: frame -> {detection_id: (x1,y1,x2,y2)}
    orig: dict[int, dict[int, tuple]] = defaultdict(dict)
    with POS_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                fr, did = int(r["frame"]), int(r["detection_id"])
                box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            except (ValueError, KeyError):
                continue
            orig[fr][did] = box

    deleted: set[tuple[int, int]] = set()
    moved: dict[tuple[int, int], tuple] = {}
    added: dict[int, list[tuple]] = defaultdict(list)
    with CORR_CSV.open() as f:
        for r in csv.DictReader(f):
            fr, key, action = int(r["frame"]), int(r["box_key"]), r["action"]
            if action == "delete":
                deleted.add((fr, key))
            elif action == "move" and r["x1"]:
                moved[(fr, key)] = (float(r["x1"]), float(r["y1"]),
                                    float(r["x2"]), float(r["y2"]))
            elif action == "add" and r["x1"]:
                added[fr].append((float(r["x1"]), float(r["y1"]),
                                  float(r["x2"]), float(r["y2"])))

    # 訂正があったフレームのみを教師に使う（確認済みフレーム）
    corrected_frames = {int(r["frame"]) for r in csv.DictReader(CORR_CSV.open())}

    result: dict[int, list[tuple]] = {}
    for fr in sorted(corrected_frames):
        boxes = []
        for did, box in orig.get(fr, {}).items():
            if (fr, did) in deleted:
                continue
            boxes.append(moved.get((fr, did), box))
        boxes.extend(added.get(fr, []))
        if boxes:
            result[fr] = boxes
    return result


def to_yolo_line(box: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((max(0, x1), min(W, x2)))
    y1, y2 = sorted((max(0, y1), min(H, y2)))
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    boxes_per_frame = build_boxes_per_frame()
    frames = sorted(boxes_per_frame.keys())
    n_val = max(1, int(len(frames) * VAL_RATIO))
    val_frames = set(frames[-n_val:])      # 末尾ブロック
    print(f"教師フレーム: {len(frames)}枚  (train={len(frames)-n_val}, val={n_val})")

    if DATASET.exists():
        shutil.rmtree(DATASET)
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DATASET / sub).mkdir(parents=True)

    total_boxes = 0
    for fr in frames:
        split = "val" if fr in val_frames else "train"
        src = FRAMES / f"frame_{fr+1:06d}.jpg"
        stem = f"frame_{fr+1:06d}"
        shutil.copy(src, DATASET / "images" / split / f"{stem}.jpg")
        lines = [to_yolo_line(b) for b in boxes_per_frame[fr]]
        total_boxes += len(lines)
        (DATASET / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    (DATASET / "data.yaml").write_text(
        f"path: {DATASET.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: player\n"
    )
    print(f"総ボックス: {total_boxes}")
    print(f"データセット: {DATASET}/  (data.yaml 生成済み)")


if __name__ == "__main__":
    main()
