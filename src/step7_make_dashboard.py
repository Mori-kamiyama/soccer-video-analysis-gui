"""
Step 7: 分析結果をブラウザで見られるHTMLにまとめる。
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return rows[:limit] if limit is not None else rows


def table_html(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<p class='empty'>データなし</p>"
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def metric_card(label: str, value: str) -> str:
    return f"<section class='metric'><div>{html.escape(label)}</div><strong>{html.escape(value)}</strong></section>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析結果HTMLを作る")
    parser.add_argument("--analysis-dir", default="outputs/all_analysis")
    parser.add_argument("--detections-csv", default="outputs/detections_all.csv")
    parser.add_argument("--tracks-csv", default="outputs/tracks_all.csv")
    parser.add_argument("--positions-csv", default="outputs/player_positions_all.csv")
    parser.add_argument("--output-html", default="outputs/dashboard.html")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    output_html = Path(args.output_html)
    detections = read_rows(Path(args.detections_csv))
    tracks = read_rows(Path(args.tracks_csv))
    positions = read_rows(Path(args.positions_csv))
    distances = read_rows(analysis_dir / "player_distances.csv")
    centroids = read_rows(analysis_dir / "team_centroids.csv", limit=30)

    unique_frames = len({row["frame"] for row in detections}) if detections else 0
    unique_players = len({row["player_id"] for row in tracks}) if tracks else 0
    total_distance = 0.0
    for row in distances:
        try:
            total_distance += float(row.get("distance_m", 0))
        except ValueError:
            pass

    metrics = "".join(
        [
            metric_card("検出フレーム数", str(unique_frames)),
            metric_card("検出数", str(len(detections))),
            metric_card("仮Player ID数", str(unique_players)),
            metric_card("推定総移動距離", f"{total_distance:.1f} m"),
        ]
    )

    html_text = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>サッカー動画分析ダッシュボード</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f4;
      color: #20251f;
    }}
    header {{
      padding: 28px 32px 18px;
      background: #173b21;
      color: white;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    main {{ padding: 24px 32px 40px; max-width: 1600px; margin: 0 auto; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .metric {{
      background: white;
      border: 1px solid #dfe5da;
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric div {{ color: #66715f; font-size: 13px; }}
    .metric strong {{ display: block; font-size: 24px; margin-top: 5px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .grid.full {{
      grid-template-columns: 1fr;
    }}
    section.panel {{
      background: white;
      border: 1px solid #dfe5da;
      border-radius: 8px;
      padding: 16px;
      overflow: auto;
    }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    img {{ width: 100%; height: auto; border-radius: 6px; background: #edf0e9; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e7ebe3; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ position: sticky; top: 0; background: #f8faf6; }}
    .note {{ color: #dce8d8; max-width: 800px; line-height: 1.5; }}
    .empty {{ color: #75806e; }}
  </style>
</head>
<body>
  <header>
    <h1>サッカー動画分析ダッシュボード</h1>
    <div class="note">Moondream検出、仮IDトラッキング、ピッチ座標化、基本分析の出力をまとめています。pitch_points.json が未調整の場合、鳥瞰図は全画面をピッチとして仮変換したものです。</div>
  </header>
  <main>
    <div class="metrics">{metrics}</div>
    <div class="grid">
      <section class="panel">
        <h2>鳥瞰図・軌跡</h2>
        <img src="{html.escape(str((analysis_dir / "trajectory.png").relative_to(output_html.parent)))}" alt="trajectory">
      </section>
      <section class="panel">
        <h2>ヒートマップ</h2>
        <img src="{html.escape(str((analysis_dir / "heatmap.png").relative_to(output_html.parent)))}" alt="heatmap">
      </section>
      <section class="panel">
        <h2>ボロノイ図（チーム領域）</h2>
        <img src="{html.escape(str((analysis_dir / "voronoi.png").relative_to(output_html.parent)))}" alt="voronoi">
      </section>
    </div>
    <div class="grid full">
      <section class="panel">
        <h2>選手ごとの移動量グラフ</h2>
        <img src="{html.escape(str((analysis_dir / "movement_graph.png").relative_to(output_html.parent)))}" alt="movement_graph">
      </section>
    </div>
    <div class="grid">
      <section class="panel">
        <h2>選手ごとの移動量データ</h2>
        {table_html(distances)}
      </section>
      <section class="panel">
        <h2>チーム重心サンプル</h2>
        {table_html(centroids)}
      </section>
    </div>
  </main>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    print(f"保存: {output_html}")


if __name__ == "__main__":
    main()
