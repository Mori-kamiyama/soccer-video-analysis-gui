"""
box_corrections.csv を反映した、IDなしの検出CSVを作る。

用途:
  1. photo_annotator.py でボックス位置・チーム色を修正
  2. uv run python build_corrected_detections.py
  3. uv run python step5_transform_to_pitch.py --input-csv outputs/detections_corrected.csv --output-csv outputs/player_positions_corrected.csv

出力は frame ごとの「選手ボックス + 足元位置 + team_hint」だけを持つ。
player_id や長時間の個人識別には依存しない。
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


INPUT_CSV = Path("outputs/detections_all_clean.csv")
FALLBACK_INPUT_CSV = Path("outputs/player_positions_all.csv")
CORRECTIONS_CSV = Path("outputs/box_corrections.csv")
OUTPUT_CSV = Path("outputs/detections_corrected.csv")

GROUP_TO_TEAM = {1: "A", 2: "B", 3: "?"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="手動補正済みの検出CSVを作る")
    p.add_argument("--input-csv", default=None,
                   help="元検出CSV。未指定なら detections_all_clean.csv、なければ player_positions_all.csv")
    p.add_argument("--corrections-csv", default=str(CORRECTIONS_CSV))
    p.add_argument("--output-csv", default=str(OUTPUT_CSV))
    return p.parse_args()


def choose_input(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    if INPUT_CSV.exists():
        return INPUT_CSV
    return FALLBACK_INPUT_CSV


def team_from_group(group: str) -> str:
    try:
        return GROUP_TO_TEAM.get(int(group), "?")
    except ValueError:
        return "?"


def recompute_geometry(row: dict[str, str]) -> None:
    x1, y1, x2, y2 = map(float, (row["x1"], row["y1"], row["x2"], row["y2"]))
    row["center_x"] = round((x1 + x2) / 2.0, 2)
    row["center_y"] = round((y1 + y2) / 2.0, 2)
    row["foot_x"] = round((x1 + x2) / 2.0, 2)
    row["foot_y"] = round(y2, 2)


def load_rows(input_csv: Path) -> tuple[dict[int, dict[int, dict[str, str]]], dict[int, dict[str, str]]]:
    by_frame: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
    frame_meta: dict[int, dict[str, str]] = {}
    with input_csv.open() as f:
        for r in csv.DictReader(f):
            try:
                fr = int(r["frame"])
                did = int(r["detection_id"])
            except (ValueError, KeyError):
                continue
            row = dict(r)
            by_frame[fr][did] = row
            frame_meta.setdefault(fr, row)
    return by_frame, frame_meta


def apply_corrections(
    by_frame: dict[int, dict[int, dict[str, str]]],
    frame_meta: dict[int, dict[str, str]],
    corrections_csv: Path,
) -> None:
    if not corrections_csv.exists():
        return
    with corrections_csv.open() as f:
        for r in csv.DictReader(f):
            try:
                fr = int(r["frame"])
                key = int(r["box_key"])
            except (ValueError, KeyError):
                continue
            action = r.get("action", "")
            if action == "delete":
                by_frame.get(fr, {}).pop(key, None)
                continue

            if action == "add":
                meta = frame_meta.get(fr, {})
                row = {
                    "frame": str(fr),
                    "source_frame": meta.get("source_frame", str(fr + 1)),
                    "time": meta.get("time", ""),
                    "filename": meta.get("filename", f"frame_{fr + 1:06d}.jpg"),
                    "detection_id": str(key),
                    "x1": r["x1"],
                    "y1": r["y1"],
                    "x2": r["x2"],
                    "y2": r["y2"],
                    "confidence": "1.0",
                    "team_hint": team_from_group(r.get("group", "")),
                }
                recompute_geometry(row)
                by_frame[fr][key] = row
                frame_meta.setdefault(fr, row)
                continue

            row = by_frame.get(fr, {}).get(key)
            if row is None:
                continue
            if action == "reassign":
                row["team_hint"] = team_from_group(r.get("group", ""))
            elif action == "move" and r.get("x1"):
                row["x1"], row["y1"], row["x2"], row["y2"] = r["x1"], r["y1"], r["x2"], r["y2"]
                if r.get("group"):
                    row["team_hint"] = team_from_group(r["group"])
                recompute_geometry(row)


def main() -> None:
    args = parse_args()
    input_csv = choose_input(args.input_csv)
    by_frame, frame_meta = load_rows(input_csv)
    apply_corrections(by_frame, frame_meta, Path(args.corrections_csv))

    rows = []
    for fr in sorted(by_frame):
        for did in sorted(by_frame[fr]):
            row = dict(by_frame[fr][did])
            if not row.get("team_hint"):
                row["team_hint"] = "?"
            rows.append(row)

    fields = [
        "frame", "source_frame", "time", "filename", "detection_id",
        "x1", "y1", "x2", "y2",
        "center_x", "center_y", "foot_x", "foot_y",
        "confidence", "team_hint",
    ]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"保存: {output_csv} ({len(rows)} rows, input={input_csv})")


if __name__ == "__main__":
    main()
