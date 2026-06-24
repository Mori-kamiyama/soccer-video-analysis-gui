"""
GUI用のセグメント処理: 動画の一部区間を
  1. 高画質フレーム(1920幅)で抽出
  2. YOLO + ByteTrack で全フレーム追跡（処理は軽量imgsz）
  3. 足元をピッチ座標(m)へ変換
し、outputs/segments/<name>/ に保存する。

座標系は 1920x1080 に統一（pitch_points.json と一致）。

使い方:
  .venv/bin/python process_segment.py --name seg1 --start-sec 120 --seconds 20 --fps 6
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "緑黒1試合目.MOV"
OUT_W, OUT_H = 1920, 1080
POINTS = Path("pitch_points.json")


def load_homography():
    if not POINTS.exists():
        return None
    pts = json.loads(POINTS.read_text())
    src = np.array(pts["image"], dtype=np.float32)
    dst = np.array(pts["pitch"], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def to_pitch(matrix, fx, fy):
    if matrix is None:
        return "", "", 0
    p = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype=np.float32), matrix)[0][0]
    x, y = float(p[0]), float(p[1])
    in_pitch = 1 if (-5 <= x <= 110 and -5 <= y <= 73) else 0
    return round(x, 3), round(y, 3), in_pitch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="seg1")
    p.add_argument("--video", default=VIDEO)
    p.add_argument("--start-sec", type=float, default=120.0)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--fps", type=float, default=24.0, help="追跡FPS。動きが速いので密(24=実質全フレーム)が最良")
    p.add_argument("--model", default="models/yolo11m.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--classes", default="0", help="0=person(COCO)。soccer時は1,2")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--device", default="mps")
    return p.parse_args()


def main():
    args = parse_args()
    classes = [int(c) for c in args.classes.split(",")]
    outdir = Path("outputs/segments") / args.name
    frames_dir = outdir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(args.start_sec * src_fps)
    stride = max(1, round(src_fps / args.fps))
    n_src = int(args.seconds * src_fps)
    matrix = load_homography()
    print(f"src_fps={src_fps:.2f} stride={stride} → 実効{src_fps/stride:.1f}fps  区間{args.seconds}s")

    model = YOLO(args.model)
    rows = []
    idx = 0  # 出力フレーム番号（連番）
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 開始位置に一度だけシーク
    for i in range(n_src):
        if i % stride != 0:
            cap.grab()  # デコードだけスキップ（高速）
            continue
        ret, frame = cap.read()
        if not ret:
            break
        # 検出は元解像度(4K)で行い小さい選手の漏れを防ぐ。表示用は1920に縮小して保存。
        sx = OUT_W / frame.shape[1]
        sy = OUT_H / frame.shape[0]
        disp = cv2.resize(frame, (OUT_W, OUT_H))
        cv2.imwrite(str(frames_dir / f"f_{idx:05d}.jpg"), disp, [cv2.IMWRITE_JPEG_QUALITY, 92])

        res = model.track(frame, persist=True, classes=classes, tracker=args.tracker,
                          imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            tids = res.boxes.id.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (bx1, by1, bx2, by2), tid, cf in zip(xyxy, tids, confs):
                # 4K座標 → 1920座標へスケール
                x1, y1, x2, y2 = bx1 * sx, by1 * sy, bx2 * sx, by2 * sy
                fx, fy = (x1 + x2) / 2, y2
                xp, yp, inp = to_pitch(matrix, fx, fy)
                rows.append({
                    "frame": idx, "track_id": int(tid),
                    "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                    "foot_x": round(float(fx), 1), "foot_y": round(float(fy), 1),
                    "x_pitch": xp, "y_pitch": yp, "in_pitch": inp,
                    "conf": round(float(cf), 3),
                })
        idx += 1
        if idx % 10 == 0:
            print(f"\r  {idx} フレーム処理", end="", flush=True)
    print()
    cap.release()

    fields = ["frame", "track_id", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
              "x_pitch", "y_pitch", "in_pitch", "conf"]
    with (outdir / "tracks.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    meta = {"name": args.name, "video": args.video, "start_sec": args.start_sec,
            "seconds": args.seconds, "eff_fps": src_fps / stride, "n_frames": idx,
            "width": OUT_W, "height": OUT_H}
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    n_ids = len({r["track_id"] for r in rows})
    print(f"保存: {outdir}/  フレーム{idx}枚  トラック行{len(rows)}  ユニークID{n_ids}")


if __name__ == "__main__":
    main()
