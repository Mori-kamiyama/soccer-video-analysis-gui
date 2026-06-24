"""
Step 2-3: フレームから選手候補を検出し、CSVに保存する。

MOONDREAM_API_KEYを設定している場合はAPIを呼ぶ。API形式が変わっても試せるよう、
endpoint/prompt/object名は引数で差し替え可能にしている。

APIキーなしで後続工程を動かしたい場合:
  uv run python step2_detect_players.py --mock-grid --limit 20
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_common import frame_number_from_filename, read_json, write_csv, write_json


DEFAULT_ENDPOINT = "https://api.moondream.ai/v1/detect"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def image_to_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def call_moondream(
    image_path: Path,
    endpoint: str,
    api_key: str,
    object_name: str,
    prompt: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "image_url": image_to_data_url(image_path),
        "object": object_name,
    }
    if prompt:
        payload["prompt"] = prompt
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "X-Moondream-Auth": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "soccer-analysis-pipeline/0.1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Moondream API error {error.code}: {body}") from error


def normalize_boxes(response: dict[str, Any], image_width: int, image_height: int) -> list[dict[str, float]]:
    raw_boxes = (
        response.get("objects")
        or response.get("detections")
        or response.get("boxes")
        or response.get("results")
        or []
    )
    boxes = []
    for item in raw_boxes:
        bbox = item.get("bbox") or item.get("box") or item.get("bounding_box") or item
        if isinstance(bbox, dict):
            x1 = bbox.get("x1", bbox.get("left", bbox.get("xmin", bbox.get("x_min"))))
            y1 = bbox.get("y1", bbox.get("top", bbox.get("ymin", bbox.get("y_min"))))
            x2 = bbox.get("x2", bbox.get("right", bbox.get("xmax", bbox.get("x_max"))))
            y2 = bbox.get("y2", bbox.get("bottom", bbox.get("ymax", bbox.get("y_max"))))
        else:
            x1, y1, x2, y2 = bbox[:4]
        if x1 is None or y1 is None or x2 is None or y2 is None:
            continue
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        if max(x1, y1, x2, y2) <= 1.5:
            x1 *= image_width
            x2 *= image_width
            y1 *= image_height
            y2 *= image_height
        boxes.append(
            {
                "x1": max(0.0, min(x1, image_width)),
                "y1": max(0.0, min(y1, image_height)),
                "x2": max(0.0, min(x2, image_width)),
                "y2": max(0.0, min(y2, image_height)),
                "confidence": float(item.get("confidence", item.get("score", 1.0))),
            }
        )
    return boxes


def mock_boxes(image_width: int, image_height: int) -> list[dict[str, float]]:
    boxes = []
    xs = [0.18, 0.28, 0.38, 0.48, 0.58, 0.68, 0.78, 0.24, 0.42, 0.62, 0.76]
    for team_offset, y_base in enumerate((0.38, 0.62)):
        for index, x_ratio in enumerate(xs):
            y_ratio = y_base + ((index % 3) - 1) * 0.035
            cx = x_ratio * image_width
            cy = y_ratio * image_height
            w = image_width * 0.018
            h = image_height * 0.055
            boxes.append(
                {
                    "x1": cx - w / 2,
                    "y1": cy - h,
                    "x2": cx + w / 2,
                    "y2": cy,
                    "confidence": 0.5,
                    "team_hint": "A" if team_offset == 0 else "B",
                }
            )
    return boxes


def image_size(path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"画像を読めません: {path}")
    height, width = image.shape[:2]
    return width, height


def detect_one_frame(
    frame_order: int,
    item: dict[str, Any],
    frames_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    image_path = frames_dir / item["filename"]
    width, height = image_size(image_path)
    if args.mock_grid:
        boxes = mock_boxes(width, height)
        raw_response = {"mock": True, "boxes": boxes}
    else:
        raw_response = call_moondream(
            image_path, args.endpoint, args.api_key, args.object, args.prompt, args.timeout
        )
        boxes = normalize_boxes(raw_response, width, height)

    rows = []
    for detection_index, box in enumerate(boxes):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        rows.append(
            {
                "frame": frame_order,
                "source_frame": frame_number_from_filename(item["filename"]),
                "time": item["timestamp"],
                "filename": item["filename"],
                "detection_id": detection_index,
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "center_x": round((x1 + x2) / 2.0, 2),
                "center_y": round((y1 + y2) / 2.0, 2),
                "foot_x": round((x1 + x2) / 2.0, 2),
                "foot_y": round(y2, 2),
                "confidence": round(float(box.get("confidence", 1.0)), 3),
                "team_hint": box.get("team_hint", ""),
            }
        )

    raw_result = {"filename": item["filename"], "response": raw_response}
    return frame_order, rows, raw_result


def parse_args() -> argparse.Namespace:
    load_env_file(Path(".env"))
    parser = argparse.ArgumentParser(description="Moondream APIで選手候補を検出してCSV保存する")
    parser.add_argument("--frames-dir", default="frames")
    parser.add_argument("--metadata", default=None, help="未指定なら frames-dir/frames_metadata.json")
    parser.add_argument("--output-csv", default="outputs/detections.csv")
    parser.add_argument("--raw-json", default="outputs/detections_raw.json")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key", default=os.getenv("MOONDREAM_API_KEY"))
    parser.add_argument("--object", default="person")
    parser.add_argument("--prompt", default="all visible soccer players")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1, help="API並列数。レート制限が出たら下げる")
    parser.add_argument("--mock-grid", action="store_true", help="APIなしで22人の仮検出を出す")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)
    metadata_path = Path(args.metadata) if args.metadata else frames_dir / "frames_metadata.json"
    metadata = read_json(metadata_path)
    if args.limit is not None:
        metadata = metadata[: args.limit]

    if not args.mock_grid and not args.api_key:
        raise RuntimeError("MOONDREAM_API_KEYがありません。APIなしで試すなら --mock-grid を付けてください。")

    completed: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(detect_one_frame, frame_order, item, frames_dir, args)
            for frame_order, item in enumerate(metadata)
        ]
        for done_count, future in enumerate(as_completed(futures), start=1):
            frame_order, frame_rows, raw_result = future.result()
            completed[frame_order] = (frame_rows, raw_result)
            print(
                f"{done_count}/{len(metadata)} {raw_result['filename']}: {len(frame_rows)} detections",
                flush=True,
            )

    rows = []
    raw_results = []
    for frame_order in sorted(completed):
        frame_rows, raw_result = completed[frame_order]
        rows.extend(frame_rows)
        raw_results.append(raw_result)

    fieldnames = [
        "frame",
        "source_frame",
        "time",
        "filename",
        "detection_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "foot_x",
        "foot_y",
        "confidence",
        "team_hint",
    ]
    write_csv(Path(args.output_csv), rows, fieldnames)
    write_json(Path(args.raw_json), raw_results)
    print(f"保存: {args.output_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
