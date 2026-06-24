"""
ブラウザでVLM統合候補を、はい/いいえ/不明で確認するローカルサーバ。

使い方:
  uv run python review_vlm_merge_server.py
  ブラウザで http://127.0.0.1:8765 を開く

キー操作:
  y / Enter : はい
  n         : いいえ
  u / Space : 不明
  ArrowLeft : 前へ
  ArrowRight: 次へ
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


CSV_PATH = Path("outputs/vlm_merge_review/candidates.csv")
BASE_DIR = Path("outputs/vlm_merge_review")


HTML = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ID統合候補レビュー</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif; background: #f6f7f8; color: #111; }
    header { display: flex; align-items: center; gap: 12px; padding: 12px 18px; background: #fff; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 2; }
    #title { font-weight: 700; font-size: 17px; flex: 1; }
    main { max-width: 1180px; margin: 14px auto 28px; padding: 0 14px; }
    #panelWrap { background: #fff; border: 1px solid #ddd; padding: 10px; overflow: auto; height: calc(100vh - 205px); }
    #panel { width: 100%; display: block; object-fit: contain; transform-origin: top left; }
    .controls { display: flex; gap: 10px; align-items: center; margin-top: 12px; }
    button { border: 0; padding: 11px 18px; border-radius: 7px; font-size: 16px; font-weight: 700; cursor: pointer; }
    .yes { background: #20a852; color: white; }
    .no { background: #e3342f; color: white; }
    .uncertain { background: #777; color: white; }
    .nav { background: #e9ecef; color: #111; }
    #note { flex: 1; min-width: 220px; font-size: 15px; padding: 10px; border: 1px solid #bbb; border-radius: 6px; }
    #status { color: #333; font-size: 14px; }
    select { font-size: 15px; padding: 7px; }
    .tool { background: #e9ecef; color: #111; padding: 8px 12px; font-size: 14px; }
    #zoomLabel { min-width: 54px; text-align: center; font-variant-numeric: tabular-nums; }
  </style>
</head>
<body>
  <header>
    <div id="title">読み込み中...</div>
    <label>候補 <select id="jump"></select></label>
    <button class="tool" onclick="zoomBy(-0.2)">−</button>
    <div id="zoomLabel">100%</div>
    <button class="tool" onclick="zoomBy(0.2)">＋</button>
    <button class="tool" onclick="fitZoom()">全体</button>
    <button class="tool" onclick="actualZoom()">等倍</button>
    <button class="tool" onclick="openPanel()">画像</button>
    <div id="status"></div>
  </header>
  <main>
    <div id="panelWrap"><img id="panel" alt="candidate panel"></div>
    <div class="controls">
      <button class="nav" onclick="move(-1)">← 前</button>
      <button class="yes" onclick="decide('yes')">はい (Y)</button>
      <button class="no" onclick="decide('no')">いいえ (N)</button>
      <button class="uncertain" onclick="decide('uncertain')">不明 (U)</button>
      <button class="nav" onclick="move(1)">次 →</button>
      <input id="note" placeholder="メモ">
    </div>
  </main>
<script>
let rows = [];
let idx = 0;
let zoom = 1;

async function loadRows() {
  const res = await fetch('/api/rows');
  rows = await res.json();
  const first = rows.findIndex(r => !(r.decision || '').trim());
  idx = first >= 0 ? first : 0;
  const jump = document.getElementById('jump');
  jump.innerHTML = '';
  rows.forEach((r, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `${i + 1}: U${r.a_id}->U${r.b_id}`;
    jump.appendChild(opt);
  });
  jump.onchange = () => { saveNote(); idx = Number(jump.value); render(); };
  render();
}

function render() {
  if (!rows.length) return;
  const r = rows[idx];
  document.getElementById('jump').value = idx;
  document.getElementById('panel').src = `/panel/${encodeURIComponent(r.panel)}?v=${Date.now()}`;
  applyZoom();
  document.getElementById('note').value = r.note || '';
  document.getElementById('title').textContent =
    `${idx + 1}/${rows.length}  U${r.a_id} -> U${r.b_id}  team=${r.team}  score=${r.score}  decision=${r.decision || '-'}`;
  const answered = rows.filter(x => (x.decision || '').trim()).length;
  const yes = rows.filter(x => ['yes', 'same', 'merge'].includes((x.decision || '').trim().toLowerCase())).length;
  document.getElementById('status').textContent = `回答済み ${answered}/${rows.length} / はい ${yes}`;
}

function applyZoom() {
  const panel = document.getElementById('panel');
  panel.style.width = `${Math.round(zoom * 100)}%`;
  document.getElementById('zoomLabel').textContent = `${Math.round(zoom * 100)}%`;
}

function zoomBy(delta) {
  zoom = Math.max(0.5, Math.min(4, zoom + delta));
  applyZoom();
}

function fitZoom() {
  zoom = 1;
  applyZoom();
  document.getElementById('panelWrap').scrollTo({left: 0, top: 0});
}

function actualZoom() {
  zoom = 2;
  applyZoom();
}

function openPanel() {
  if (!rows.length) return;
  window.open(`/panel/${encodeURIComponent(rows[idx].panel)}`, '_blank');
}

async function saveNote() {
  if (!rows.length) return;
  const note = document.getElementById('note').value;
  rows[idx].note = note;
  await fetch('/api/decision', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({rank: rows[idx].rank, note})
  });
}

async function decide(decision) {
  rows[idx].decision = decision;
  rows[idx].note = document.getElementById('note').value;
  await fetch('/api/decision', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({rank: rows[idx].rank, decision, note: rows[idx].note})
  });
  move(1, false);
}

function move(delta, save = true) {
  if (save) saveNote();
  idx = Math.max(0, Math.min(rows.length - 1, idx + delta));
  render();
}

document.addEventListener('keydown', e => {
  if (document.activeElement && document.activeElement.id === 'note') return;
  if (e.key === 'y' || e.key === 'Y' || e.key === 'Enter') decide('yes');
  if (e.key === 'n' || e.key === 'N') decide('no');
  if (e.key === 'u' || e.key === 'U' || e.key === ' ') decide('uncertain');
  if (e.key === 'ArrowLeft') move(-1);
  if (e.key === 'ArrowRight') move(1);
  if (e.key === '+' || e.key === '=') zoomBy(0.2);
  if (e.key === '-' || e.key === '_') zoomBy(-0.2);
  if (e.key === '0') fitZoom();
  if (e.key === '1') actualZoom();
});

loadRows();
</script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    csv_path: Path = CSV_PATH
    base_dir: Path = BASE_DIR

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _rows(self) -> list[dict[str, str]]:
        with self.csv_path.open() as f:
            return list(csv.DictReader(f))

    def _save_rows(self, rows: list[dict[str, str]]) -> None:
        with self.csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/rows":
            self._send(
                HTTPStatus.OK,
                json.dumps(self._rows(), ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
            )
            return
        if parsed.path.startswith("/panel/"):
            rel = unquote(parsed.path.removeprefix("/panel/"))
            path = (self.base_dir / rel).resolve()
            base = self.base_dir.resolve()
            if not str(path).startswith(str(base)) or not path.exists():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, path.read_bytes(), ctype)
            return
        self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/decision":
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size).decode("utf-8"))
        rank = str(payload.get("rank", ""))
        rows = self._rows()
        changed = False
        for row in rows:
            if row.get("rank") == rank:
                if "decision" in payload:
                    row["decision"] = str(payload.get("decision") or "")
                if "note" in payload:
                    row["note"] = str(payload.get("note") or "")
                changed = True
                break
        if changed:
            self._save_rows(rows)
            self._send(HTTPStatus.OK, b'{"ok":true}', "application/json")
        else:
            self._send(HTTPStatus.NOT_FOUND, b'{"ok":false}', "application/json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--base-dir", type=Path, default=BASE_DIR)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"{args.csv} がありません。先に vlm_merge_candidates.py を実行してください。")
    ReviewHandler.csv_path = args.csv
    ReviewHandler.base_dir = args.base_dir
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"review server: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
