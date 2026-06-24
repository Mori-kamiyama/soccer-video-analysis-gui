"""
OpenRouterのVLMで統合候補パネルを下書き判定する。

使い方:
  uv run python openrouter_review_candidates.py --limit 13

APIキー:
  OPENROUTER_API_KEY または OPENROUTER_KEY を環境変数から読む。
  見つからない場合は zsh -lic 経由で ~/.zshrc 側の環境変数も探す。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


CSV_PATH = Path("outputs/vlm_merge_review/candidates.csv")
BASE_DIR = Path("outputs/vlm_merge_review")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"
YES_VALUES = {"same", "yes", "likely same"}
NO_VALUES = {"different", "no", "not same"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--base-dir", type=Path, default=BASE_DIR)
    p.add_argument("--model", default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL))
    p.add_argument("--limit", type=int, default=0, help="0なら空欄を全部処理")
    p.add_argument("--sleep", type=float, default=0.4)
    p.add_argument("--overwrite", action="store_true", help="既存decisionも再判定する")
    return p.parse_args()


def load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if key:
        return key
    try:
        out = subprocess.check_output(
            [
                "zsh",
                "-lic",
                "printf %s \"${OPENROUTER_API_KEY:-${OPENROUTER_KEY:-}}\"",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        out = ""
    key = out.strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY / OPENROUTER_KEY が見つかりません")
    return key


def image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def build_prompt(row: dict[str, str]) -> str:
    return (
        "You are reviewing a soccer player tracking merge candidate. "
        "The image is a 6-panel sequence: top row is player A before disappearing, "
        "bottom row is player B after reappearing. "
        "The red cross marks the tracked foot position; use it to identify which nearby player is the target. "
        "Do not rely on bounding boxes if any are visible. "
        "Decide whether A and B are likely the same real player.\n\n"
        "Use the visual evidence plus these tracking metrics:\n"
        f"- team: {row.get('team')}\n"
        f"- gap_frames: {row.get('gap')}\n"
        f"- end_distance_m: {row.get('end_dist_m')}\n"
        f"- predicted_distance_m: {row.get('pred_dist_m')}\n"
        f"- required_speed_mps: {row.get('req_speed_mps')}\n\n"
        "Return strict JSON only, with keys:\n"
        '{"decision":"same|different|uncertain","confidence":0.0,"reason":"short reason"}'
    )


def call_openrouter(key: str, model: str, image_path: Path, prompt: str) -> dict[str, object]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 220,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "soccer-id-merge-review",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail}") from e
    text = data["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"decision": "uncertain", "confidence": 0.0, "reason": str(text)[:180]}


def normalize_decision(value: object) -> str:
    v = str(value or "").strip().lower()
    if v in YES_VALUES:
        return "yes"
    if v in NO_VALUES:
        return "no"
    return "uncertain"


def save_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    key = load_key()
    with args.csv.open() as f:
        rows = list(csv.DictReader(f))
    processed = 0
    for row in rows:
        if not args.overwrite and (row.get("decision") or "").strip():
            continue
        if args.limit and processed >= args.limit:
            break
        image_path = args.base_dir / row["panel"]
        try:
            result = call_openrouter(key, args.model, image_path, build_prompt(row))
        except RuntimeError as e:
            print(str(e))
            print("OpenRouter判定を中断しました。既存の回答はそのままです。")
            break
        decision = normalize_decision(result.get("decision"))
        conf = result.get("confidence", "")
        reason = str(result.get("reason", "")).replace("\n", " ").strip()
        row["decision"] = decision
        row["note"] = f"vlm:{args.model} confidence={conf} {reason}".strip()
        processed += 1
        print(f"{processed}: rank {row.get('rank')} U{row.get('a_id')} -> U{row.get('b_id')} = {decision} ({conf})")
        save_rows(args.csv, rows)
        time.sleep(args.sleep)
    print(f"processed: {processed}")
    print(f"saved: {args.csv}")


if __name__ == "__main__":
    main()
