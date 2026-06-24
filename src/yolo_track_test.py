"""
YOLO + ByteTrack 追跡テスト

元動画(4K)の一部区間だけを処理し、
  - 選手にIDを振った確認用動画 (outputs/yolo_track_preview.mp4)
  - トラックCSV            (outputs/yolo_tracks_test.csv)
を出力する。Moondream APIは使わず、ローカルYOLO(MPS)で全フレーム検出する。

使い方:
  .venv/bin/python yolo_track_test.py --start-sec 120 --seconds 10
"""

from __future__ import annotations

import argparse
import csv
import colorsys
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO = "緑黒1試合目.MOV"
OUT_VIDEO = Path("outputs/yolo_track_preview.mp4")
OUT_CSV = Path("outputs/yolo_tracks_test.csv")


def id_color(tid: int) -> tuple[int, int, int]:
    """トラックIDから安定したBGR色を生成。"""
    h = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", default=VIDEO)
    p.add_argument("--start-sec", type=float, default=120.0, help="開始秒")
    p.add_argument("--seconds", type=float, default=10.0, help="処理する秒数")
    p.add_argument("--model", default="models/yolo11m.pt", help="YOLOモデル(n/s/m/l/x)")
    p.add_argument("--classes", default="0", help="検出クラス(カンマ区切り)。COCO=0(person), soccer=1,2(GK,player)")
    p.add_argument("--out-suffix", default="", help="出力ファイル名の接尾辞")
    p.add_argument("--imgsz", type=int, default=1280, help="推論解像度")
    p.add_argument("--conf", type=float, default=0.25, help="検出信頼度しきい値")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml / botsort.yaml")
    p.add_argument("--device", default="mps")
    p.add_argument("--preview-width", type=int, default=1920, help="出力動画の横幅")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_VIDEO.parent.mkdir(exist_ok=True)
    classes = [int(c) for c in args.classes.split(",")]
    out_video = OUT_VIDEO.with_name(f"yolo_track_preview{args.out_suffix}.mp4")
    out_csv = OUT_CSV.with_name(f"yolo_tracks_test{args.out_suffix}.csv")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(args.start_sec * fps)
    num_frames = int(args.seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"動画FPS={fps:.2f}  開始frame={start_frame}  処理枚数={num_frames}")

    model = YOLO(args.model)

    writer = None
    rows: list[dict] = []
    ids_seen: set[int] = set()

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(
            frame,
            persist=True,
            classes=classes,       # 指定クラスのみ
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        r = results[0]
        vis = frame.copy()

        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            tids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, cf in zip(xyxy, tids, confs):
                ids_seen.add(int(tid))
                rows.append({
                    "frame": start_frame + i, "track_id": int(tid),
                    "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1), "y2": round(float(y2), 1),
                    "conf": round(float(cf), 3),
                })
                color = id_color(int(tid))
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                cv2.putText(vis, f"{tid}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # プレビュー動画へ書き出し（縮小）
        scale = args.preview_width / vis.shape[1]
        out = cv2.resize(vis, (args.preview_width, int(vis.shape[0] * scale)))
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(str(out_video),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(out)
        if (i + 1) % 20 == 0:
            print(f"\r  {i+1}/{num_frames}  累計ID数={len(ids_seen)}", end="", flush=True)

    print()
    if writer:
        writer.release()
    cap.release()

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "track_id", "x1", "y1", "x2", "y2", "conf"])
        w.writeheader()
        w.writerows(rows)

    print(f"出力動画: {out_video}")
    print(f"トラックCSV: {out_csv}  ({len(rows)}行)")
    print(f"ユニークID数: {len(ids_seen)}  （選手は実質22人前後が理想）")


if __name__ == "__main__":
    main()
