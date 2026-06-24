"""
選手ID統合エディター（高fps版・鳥瞰軌跡）

build_player_ids.py が作った players_hi.csv の track_id（断片トラック）を、
鳥瞰図で軌跡を見ながら ~22人へ統合する。既存 player_merge_gui を流用し、
統合単位を track_id にして merge_map_hi.csv を書き出す。

ワークフロー:
  1) このGUIで自動統合→手動で微修正→保存（merge_map_hi.csv）
  2) .venv/bin/python build_player_ids.py で players_hi.csv に反映
  3) visualize_offside / 画像エディターが統合後IDを使う

起動:
  ./run_merge_hi.sh
"""

from __future__ import annotations

import csv
from pathlib import Path

import tkinter as tk

import player_merge_gui as pm

# 入出力を高fps版に差し替え
pm.INPUT_CSV = Path("outputs/players_hi.csv")
pm.MERGE_MAP_CSV = Path("outputs/merge_map_hi.csv")
pm.OUTPUT_CSV = Path("outputs/players_hi_merged.csv")


class MergeHi(pm.MergeApp):
    def _load(self) -> None:
        """players_hi.csv を track_id を統合単位として読み込む。"""
        if not pm.INPUT_CSV.exists():
            raise SystemExit(f"{pm.INPUT_CSV} がありません。先に build_player_ids.py を実行。")
        with pm.INPUT_CSV.open() as f:
            for row in csv.DictReader(f):
                try:
                    tid = int(row["track_id"])
                    frame = int(row["frame"])
                    x = float(row["x_pitch"])
                    y = float(row["y_pitch"])
                except (ValueError, KeyError):
                    continue
                tr = self.tracks.get(tid)
                if tr is None:
                    tr = pm.Track(tid)
                    self.tracks[tid] = tr
                if frame not in tr.pts:
                    tr.frames.append(frame)
                    tr.pts[frame] = (x, y)
                tr.team_counter[row.get("team_hint", "")] += 1
        for tr in self.tracks.values():
            tr.frames.sort()


def main() -> None:
    root = tk.Tk()
    MergeHi(root)
    root.mainloop()


if __name__ == "__main__":
    main()
