"""
チーム色の小型kNNモデルを学習する。

手修正ラベルが偏っている場合でも使えるよう、既定では指定フレーム範囲の
胴体crop色特徴を2クラスタに分け、緑比率が高いクラスタをA(緑)として保存する。

例:
  uv run python train_team_color_model.py --frame-start 0 --frame-end 50 --mode cluster
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from photo_annotator import Annotator, COLOR_MODEL_JSON, MAX_TRAIN_SAMPLES_PER_TEAM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="チーム色分類モデルを学習する")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=50, help="この値未満のフレームまで使う")
    p.add_argument("--mode", choices=["cluster", "manual"], default="cluster")
    p.add_argument("--output", default=str(COLOR_MODEL_JSON))
    return p.parse_args()


def make_hidden_app() -> Annotator:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    app = Annotator(root)
    app._hidden_root = root
    return app


def collect_cluster_samples(app: Annotator, frame_start: int, frame_end: int):
    image_cache: dict[int, Image.Image] = {}
    features: list[list[float]] = []
    ratios: list[float] = []
    frames: list[int] = []
    for frame in sorted(app.boxes_by_frame):
        if frame < frame_start or frame >= frame_end:
            continue
        for box in app.boxes_by_frame[frame]:
            if box.deleted:
                continue
            feature = app._box_color_feature(box, image_cache)
            ratio = app._box_green_ratio(box, image_cache)
            if feature is None or ratio is None:
                continue
            features.append(feature)
            ratios.append(ratio)
            frames.append(frame)
    return features, ratios, frames


def cluster_labels(features: list[list[float]], ratios: list[float]) -> tuple[list[int], dict]:
    data = np.asarray(features, dtype=np.float32)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-6] = 1.0
    z = (data - mean) / std
    compactness, labels, centers = cv2.kmeans(
        z,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.001),
        20,
        cv2.KMEANS_PP_CENTERS,
    )
    raw = labels.reshape(-1)
    ratio_by_cluster = []
    for c in (0, 1):
        vals = [ratios[i] for i in range(len(ratios)) if raw[i] == c]
        ratio_by_cluster.append(float(np.mean(vals)) if vals else -1.0)
    green_cluster = 0 if ratio_by_cluster[0] >= ratio_by_cluster[1] else 1
    groups = [1 if int(c) == green_cluster else 2 for c in raw]
    meta = {
        "compactness": float(compactness),
        "cluster_green_ratio_mean": ratio_by_cluster,
        "green_cluster": green_cluster,
    }
    return groups, meta


def balanced_subset(features: list[list[float]], labels: list[int], max_per_team: int):
    selected_features = []
    selected_labels = []
    for group in (1, 2):
        idxs = [i for i, label in enumerate(labels) if label == group]
        if len(idxs) > max_per_team:
            step = (len(idxs) - 1) / (max_per_team - 1)
            idxs = [idxs[round(i * step)] for i in range(max_per_team)]
        for i in idxs:
            selected_features.append(features[i])
            selected_labels.append(labels[i])
    return selected_features, selected_labels


def main() -> None:
    args = parse_args()
    app = make_hidden_app()
    try:
        if args.mode == "manual":
            samples = [
                (feature, group)
                for feature, _ratio, group in app._manual_training_samples()
                if args.frame_start <= 0 or True
            ]
            features = [feature for feature, _ in samples]
            labels = [group for _, group in samples]
            meta = {"source": "manual"}
        else:
            features, ratios, frames = collect_cluster_samples(app, args.frame_start, args.frame_end)
            if len(features) < 20:
                raise SystemExit(f"学習サンプルが少なすぎます: {len(features)}")
            labels, meta = cluster_labels(features, ratios)
            meta.update({
                "source": "cluster",
                "frame_start": args.frame_start,
                "frame_end": args.frame_end,
                "raw_samples": len(features),
                "frames": sorted(set(frames)),
            })

        features, labels = balanced_subset(features, labels, MAX_TRAIN_SAMPLES_PER_TEAM)
        accuracy = app._loo_knn_accuracy(features, labels)
        data = np.asarray(features, dtype=np.float32)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-6] = 1.0
        model = {
            "type": "knn_color_v1",
            "features": features,
            "labels": labels,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "k": min(7, len(labels)),
            "accuracy": accuracy,
            "samples": len(labels),
            "meta": meta,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(model, ensure_ascii=False, indent=2))
        print(f"保存: {output}")
        print(f"samples={len(labels)} A={labels.count(1)} B={labels.count(2)} loo_acc={accuracy:.3f}")
        print(f"meta={meta}")
    finally:
        getattr(app, "_hidden_root").destroy()


if __name__ == "__main__":
    main()
