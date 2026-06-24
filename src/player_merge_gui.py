"""
選手ID統合GUI（ステップ2）

トラッカーが断片化して 690 個ものIDを生んでいるのを、本来の ~22 人へ統合する。

機能:
  - 自動統合提案: 時間的に途切れず空間的に近い断片を連結（しきい値調整可）
  - 手動統合 / 解除: リストで複数選択して統合、または解除
  - 鳥瞰図で確認: 選択したIDの軌跡を色分け表示。時間重複（＝別人の可能性）を警告
  - 保存: merge_map.csv と統合後CSVを書き出す

起動:
  .venv/bin/python player_merge_gui.py
"""

from __future__ import annotations

import csv
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox, ttk

INPUT_CSV = Path("outputs/player_positions_all.csv")
MERGE_MAP_CSV = Path("outputs/merge_map.csv")
OUTPUT_CSV = Path("outputs/player_positions_merged.csv")

PITCH_L = 105.0
PITCH_W = 68.0
TARGET_PLAYERS = 22

# 鳥瞰図キャンバス
CANVAS_W = 1000
CANVAS_H = int(CANVAS_W * PITCH_W / PITCH_L)
PAD = 30

SELECT_COLORS = [
    "#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
    "#ff2d55", "#5ac8fa", "#ffcc00", "#4cd964", "#5856d6",
]


class UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # 小さいID側を親にしておく（見た目が安定する）
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb

    def reset(self, x: int) -> None:
        self.parent[x] = x


class Track:
    """1つの元IDの軌跡情報。"""

    __slots__ = ("id", "frames", "pts", "team_counter")

    def __init__(self, tid: int) -> None:
        self.id = tid
        self.frames: list[int] = []
        self.pts: dict[int, tuple[float, float]] = {}
        self.team_counter: defaultdict[str, int] = defaultdict(int)

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    @property
    def start_pt(self) -> tuple[float, float]:
        return self.pts[self.frames[0]]

    @property
    def end_pt(self) -> tuple[float, float]:
        return self.pts[self.frames[-1]]

    @property
    def team(self) -> str:
        if not self.team_counter:
            return ""
        return max(self.team_counter, key=self.team_counter.get)


class MergeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("選手ID統合 (ステップ2)")

        self.tracks: dict[int, Track] = {}
        self._load()
        self.uf = UnionFind(list(self.tracks.keys()))
        resumed = self._load_merge_map()

        self._build_ui()
        self._refresh_list()
        if resumed:
            self.info.config(
                text=f"前回の統合状態を merge_map.csv から復元しました（{resumed}件の対応）。",
                fg="#007aff")

    # ---- データ ----
    def _load(self) -> None:
        if not INPUT_CSV.exists():
            raise SystemExit(f"{INPUT_CSV} が見つかりません")
        with INPUT_CSV.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tid = int(row["player_id"])
                    frame = int(row["frame"])
                    x = float(row["x_pitch"])
                    y = float(row["y_pitch"])
                except (ValueError, KeyError):
                    continue
                tr = self.tracks.get(tid)
                if tr is None:
                    tr = Track(tid)
                    self.tracks[tid] = tr
                if frame not in tr.pts:
                    tr.frames.append(frame)
                    tr.pts[frame] = (x, y)
                tr.team_counter[row.get("team_hint", "")] += 1
        for tr in self.tracks.values():
            tr.frames.sort()

    def _load_merge_map(self) -> int:
        """保存済み merge_map.csv があれば統合状態を復元する。返り値は対応行数。"""
        if not MERGE_MAP_CSV.exists():
            return 0
        by_new: dict[int, list[int]] = defaultdict(list)
        try:
            with MERGE_MAP_CSV.open() as f:
                for row in csv.DictReader(f):
                    old = int(row["old_id"])
                    new = int(row["new_id"])
                    if old in self.tracks:
                        by_new[new].append(old)
        except Exception:
            return 0
        count = 0
        for members in by_new.values():
            base = members[0]
            for other in members[1:]:
                self.uf.union(base, other)
            count += len(members)
        return count

    # ---- グループ集計 ----
    def groups(self) -> dict[int, list[int]]:
        g: dict[int, list[int]] = defaultdict(list)
        for tid in self.tracks:
            g[self.uf.find(tid)].append(tid)
        return g

    def group_frames(self, members: list[int]) -> list[int]:
        s: set[int] = set()
        for m in members:
            s.update(self.tracks[m].frames)
        return sorted(s)

    # ---- UI ----
    def _build_ui(self) -> None:
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.count_label = tk.Label(bar, text="", font=("Helvetica", 15, "bold"))
        self.count_label.pack(side=tk.LEFT)

        tk.Button(bar, text="選択を統合", command=self._merge_selected,
                  bg="#007aff", fg="white", font=("Helvetica", 12, "bold")).pack(side=tk.RIGHT)
        tk.Button(bar, text="選択を解除(分割)", command=self._unmerge_selected).pack(side=tk.RIGHT, padx=4)
        tk.Button(bar, text="保存", command=self._save,
                  bg="#34c759", fg="white", font=("Helvetica", 12, "bold")).pack(side=tk.RIGHT, padx=4)

        # 自動統合パネル
        auto = tk.Frame(self.root)
        auto.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        tk.Label(auto, text="自動統合  最大フレーム間隔:").pack(side=tk.LEFT)
        self.gap_var = tk.IntVar(value=30)
        tk.Spinbox(auto, from_=1, to=300, width=5, textvariable=self.gap_var).pack(side=tk.LEFT, padx=4)
        tk.Label(auto, text="最大距離(m):").pack(side=tk.LEFT)
        self.dist_var = tk.DoubleVar(value=8.0)
        tk.Spinbox(auto, from_=0.5, to=50, increment=0.5, width=5, textvariable=self.dist_var).pack(side=tk.LEFT, padx=4)
        tk.Label(auto, text="チーム一致のみ:").pack(side=tk.LEFT)
        self.same_team_var = tk.BooleanVar(value=True)
        tk.Checkbutton(auto, variable=self.same_team_var).pack(side=tk.LEFT)
        tk.Label(auto, text="  近接重複も統合(m):").pack(side=tk.LEFT)
        self.dup_var = tk.DoubleVar(value=2.0)
        tk.Spinbox(auto, from_=0.0, to=10, increment=0.5, width=5, textvariable=self.dup_var).pack(side=tk.LEFT, padx=4)
        tk.Button(auto, text="自動統合を実行", command=self._auto_merge,
                  bg="#ff9500", fg="white").pack(side=tk.LEFT, padx=8)
        tk.Button(auto, text="全リセット", command=self._reset_all).pack(side=tk.LEFT)

        main = tk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # 左: リスト
        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y)
        cols = ("id", "frags", "nframes", "first", "last", "team")
        headers = {"id": "ID", "frags": "断片数", "nframes": "フレーム数",
                   "first": "開始F", "last": "終了F", "team": "チーム"}
        widths = {"id": 60, "frags": 60, "nframes": 80, "first": 70, "last": 70, "team": 60}
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=28,
                                 selectmode="extended")
        for c in cols:
            self.tree.heading(c, text=headers[c], command=lambda cc=c: self._sort_by(cc))
            self.tree.column(c, width=widths[c], anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.Y)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y)
        self.tree.config(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self._draw())

        # 右: 鳥瞰図
        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self.info = tk.Label(right, text="IDを選ぶと軌跡を表示します", font=("Helvetica", 12),
                             anchor="w", justify=tk.LEFT, wraplength=CANVAS_W)
        self.info.pack(side=tk.TOP, fill=tk.X)
        self.canvas = tk.Canvas(right, width=CANVAS_W + 2 * PAD, height=CANVAS_H + 2 * PAD,
                                bg="#2e7d32", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, pady=4)

        self._sort_col = "nframes"
        self._sort_desc = True

    # ---- 座標変換 ----
    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        cx = PAD + x / PITCH_L * CANVAS_W
        cy = PAD + y / PITCH_W * CANVAS_H
        return cx, cy

    def _draw_pitch(self) -> None:
        self.canvas.create_rectangle(PAD, PAD, PAD + CANVAS_W, PAD + CANVAS_H,
                                     outline="white", width=2)
        midx = PAD + CANVAS_W / 2
        self.canvas.create_line(midx, PAD, midx, PAD + CANVAS_H, fill="white")
        r = 9.15 / PITCH_L * CANVAS_W
        cy = PAD + CANVAS_H / 2
        self.canvas.create_oval(midx - r, cy - r, midx + r, cy + r, outline="white")

    # ---- 描画 ----
    def _draw(self) -> None:
        self.canvas.delete("all")
        self._draw_pitch()

        selected_groups = [int(self.tree.item(i, "values")[0]) for i in self.tree.selection()]
        sel_set = set(selected_groups)
        grp = self.groups()

        # 非選択: 薄く
        for canon, members in grp.items():
            if canon in sel_set:
                continue
            for m in members:
                self._draw_track(m, "#1b5e20", width=1)

        # 選択: 色分け
        for idx, canon in enumerate(selected_groups):
            color = SELECT_COLORS[idx % len(SELECT_COLORS)]
            for m in grp.get(canon, []):
                self._draw_track(m, color, width=2, markers=True)

        self._update_info(selected_groups, grp)

    def _draw_track(self, tid: int, color: str, width: int, markers: bool = False) -> None:
        tr = self.tracks[tid]
        pts = [self._to_canvas(*tr.pts[f]) for f in tr.frames]
        if len(pts) >= 2:
            flat = [c for p in pts for c in p]
            self.canvas.create_line(*flat, fill=color, width=width)
        elif pts:
            x, y = pts[0]
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
        if markers and pts:
            sx, sy = pts[0]
            ex, ey = pts[-1]
            self.canvas.create_oval(sx - 5, sy - 5, sx + 5, sy + 5, fill="white", outline=color, width=2)
            self.canvas.create_rectangle(ex - 5, ey - 5, ex + 5, ey + 5, fill=color, outline="white")

    def _update_info(self, selected: list[int], grp: dict[int, list[int]]) -> None:
        if not selected:
            self.info.config(text="IDを選ぶと軌跡を表示します（○=開始, ■=終了）", fg="black")
            return
        lines = []
        # 時間重複チェック（2つ以上選択時）
        if len(selected) >= 2:
            frame_sets = {c: set(self.group_frames(grp[c])) for c in selected}
            overlap = False
            warn = False
            pair_msg = []
            sel_sorted = sorted(selected, key=lambda c: min(frame_sets[c]))
            for i in range(len(sel_sorted)):
                for j in range(i + 1, len(sel_sorted)):
                    a, b = sel_sorted[i], sel_sorted[j]
                    ov = frame_sets[a] & frame_sets[b]
                    if ov:
                        overlap = True
                        # 重複フレームでの平均距離 → 近ければ二重検出、遠ければ別人
                        mean_d = self._overlap_distance(grp[a], grp[b], ov)
                        if mean_d <= 3.0:
                            pair_msg.append(f"ID{a}↔ID{b}: {len(ov)}F重複/平均{mean_d:.1f}m(二重検出)")
                        else:
                            warn = True
                            pair_msg.append(f"ID{a}↔ID{b}: {len(ov)}F重複/平均{mean_d:.1f}m(別人?)")
            if overlap:
                if warn:
                    lines.append("⚠ 時間重複かつ距離大 → 別人の可能性（統合注意）  " + " / ".join(pair_msg))
                else:
                    lines.append("◎ 時間重複だが近接 → 同一選手の二重検出（統合推奨）  " + " / ".join(pair_msg))
            else:
                # 連続具合（隣接ペアの時間ギャップ・距離）
                gaps = []
                for i in range(len(sel_sorted) - 1):
                    a = grp[sel_sorted[i]]
                    b = grp[sel_sorted[i + 1]]
                    a_last = max(a, key=lambda m: self.tracks[m].last)
                    b_first = min(b, key=lambda m: self.tracks[m].first)
                    ta = self.tracks[a_last]
                    tb = self.tracks[b_first]
                    gap = tb.first - ta.last
                    d = ((ta.end_pt[0] - tb.start_pt[0]) ** 2 + (ta.end_pt[1] - tb.start_pt[1]) ** 2) ** 0.5
                    gaps.append(f"間隔{gap}F/距離{d:.1f}m")
                lines.append("✓ 時間重複なし → 統合候補  " + " ".join(gaps))
        for c in selected:
            members = grp[c]
            frames = self.group_frames(members)
            lines.append(f"ID{c}: 断片{len(members)} / {len(frames)}F "
                         f"(F{frames[0]}–F{frames[-1]}) team={self.tracks[c].team}")
        self.info.config(text="\n".join(lines),
                         fg="#c62828" if any("⚠" in l for l in lines) else "black")

    def _overlap_distance(self, members_a: list[int], members_b: list[int],
                          overlap_frames: set[int]) -> float:
        """2グループが重複するフレームでの平均距離。"""
        pa: dict[int, tuple[float, float]] = {}
        for m in members_a:
            pa.update(self.tracks[m].pts)
        pb: dict[int, tuple[float, float]] = {}
        for m in members_b:
            pb.update(self.tracks[m].pts)
        total = 0.0
        n = 0
        for f in overlap_frames:
            if f in pa and f in pb:
                ax, ay = pa[f]
                bx, by = pb[f]
                total += ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                n += 1
        return total / n if n else 999.0

    # ---- リスト ----
    def _refresh_list(self) -> None:
        sel = set(self.tree.selection())
        self.tree.delete(*self.tree.get_children())
        grp = self.groups()
        rows = []
        for canon, members in grp.items():
            frames = self.group_frames(members)
            rows.append((canon, len(members), len(frames), frames[0], frames[-1],
                         self.tracks[canon].team))
        key_index = {"id": 0, "frags": 1, "nframes": 2, "first": 3, "last": 4, "team": 5}
        rows.sort(key=lambda r: r[key_index[self._sort_col]], reverse=self._sort_desc)
        for r in rows:
            iid = str(r[0])
            self.tree.insert("", tk.END, iid=iid, values=r)
            if iid in sel:
                self.tree.selection_add(iid)
        n = len(grp)
        diff = n - TARGET_PLAYERS
        self.count_label.config(
            text=f"現在のID数: {n}  (目標 {TARGET_PLAYERS} / 差 {diff:+d})",
            fg="#34c759" if n <= TARGET_PLAYERS + 4 else "#c62828")
        self._draw()

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            self._sort_desc = True
        self._refresh_list()

    # ---- 操作 ----
    def _merge_selected(self) -> None:
        sel = [int(self.tree.item(i, "values")[0]) for i in self.tree.selection()]
        if len(sel) < 2:
            messagebox.showinfo("統合", "2つ以上のIDを選択してください")
            return
        base = sel[0]
        for other in sel[1:]:
            self.uf.union(base, other)
        self._refresh_list()
        # 統合先を選択状態に
        canon = self.uf.find(base)
        if self.tree.exists(str(canon)):
            self.tree.selection_set(str(canon))
            self.tree.see(str(canon))

    def _unmerge_selected(self) -> None:
        sel = [int(self.tree.item(i, "values")[0]) for i in self.tree.selection()]
        grp = self.groups()
        for canon in sel:
            for m in grp.get(canon, []):
                self.uf.reset(m)
        self._refresh_list()

    def _reset_all(self) -> None:
        if not messagebox.askyesno("確認", "すべての統合を取り消しますか？"):
            return
        for tid in self.tracks:
            self.uf.reset(tid)
        self._refresh_list()

    def _auto_merge(self) -> None:
        max_gap = self.gap_var.get()
        max_dist = self.dist_var.get()
        same_team = self.same_team_var.get()
        dup_dist = self.dup_var.get()

        dup_merges = 0
        if dup_dist > 0:
            dup_merges = self._merge_duplicates(dup_dist, min_overlap=3)

        ids = sorted(self.tracks.keys(), key=lambda t: self.tracks[t].first)
        used_as_next: set[int] = set()
        merges = 0
        # 各断片の末尾に、最も自然に続く断片を貪欲に連結
        for a in ids:
            ta = self.tracks[a]
            ra = self.uf.find(a)
            best = None
            best_cost = None
            for b in ids:
                if b == a or b in used_as_next:
                    continue
                tb = self.tracks[b]
                if self.uf.find(b) == ra:
                    continue
                gap = tb.first - ta.last
                if gap < 1 or gap > max_gap:
                    continue
                if same_team and ta.team and tb.team and ta.team != tb.team:
                    continue
                d = ((ta.end_pt[0] - tb.start_pt[0]) ** 2 + (ta.end_pt[1] - tb.start_pt[1]) ** 2) ** 0.5
                if d > max_dist:
                    continue
                cost = d + gap * 0.1
                if best_cost is None or cost < best_cost:
                    best, best_cost = b, cost
            if best is not None:
                self.uf.union(a, best)
                used_as_next.add(best)
                merges += 1
        self._refresh_list()
        messagebox.showinfo("自動統合",
                            f"近接重複の統合: {dup_merges} 組\n"
                            f"時系列の連結: {merges} 件\n\n"
                            f"現在のID数: {len(self.groups())}")

    def _merge_duplicates(self, dup_dist: float, min_overlap: int) -> int:
        """同一フレームで近接し続ける断片（＝同一選手の二重検出）を統合する。"""
        # frame -> [(id, x, y)]
        per_frame: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        for tr in self.tracks.values():
            for f in tr.frames:
                x, y = tr.pts[f]
                per_frame[f].append((tr.id, x, y))
        # ペアごとに重複フレームでの距離を集計
        pair_sum: dict[tuple[int, int], float] = defaultdict(float)
        pair_cnt: dict[tuple[int, int], int] = defaultdict(int)
        for items in per_frame.values():
            n = len(items)
            for i in range(n):
                ai, ax, ay = items[i]
                for j in range(i + 1, n):
                    bi, bx, by = items[j]
                    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                    if d < dup_dist:
                        key = (ai, bi) if ai < bi else (bi, ai)
                        pair_sum[key] += d
                        pair_cnt[key] += 1
        merged = 0
        for (a, b), cnt in pair_cnt.items():
            if cnt >= min_overlap and pair_sum[(a, b)] / cnt < dup_dist:
                if self.uf.find(a) != self.uf.find(b):
                    self.uf.union(a, b)
                    merged += 1
        return merged

    def _save(self) -> None:
        grp = self.groups()
        # canonical id を 1..N に振り直す（フレーム数が多い順）
        ordered = sorted(grp.items(), key=lambda kv: -len(self.group_frames(kv[1])))
        new_id_of: dict[int, int] = {}
        with MERGE_MAP_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["old_id", "new_id"])
            for new_id, (canon, members) in enumerate(ordered, start=1):
                for m in members:
                    new_id_of[m] = new_id
                    w.writerow([m, new_id])

        # 統合後の player_positions を書き出す。
        # 元IDは orig_player_id として必ず残す → このCSV単体から完全に復元できる。
        with INPUT_CSV.open() as f:
            reader = csv.DictReader(f)
            base_fields = reader.fieldnames or []
            fieldnames = list(base_fields)
            if "orig_player_id" not in fieldnames:
                idx = fieldnames.index("player_id") if "player_id" in fieldnames else len(fieldnames)
                fieldnames.insert(idx, "orig_player_id")
            out_rows = []
            for row in reader:
                try:
                    old = int(row["player_id"])
                except ValueError:
                    continue
                row["orig_player_id"] = old
                row["player_id"] = new_id_of.get(old, old)
                out_rows.append(row)
        with OUTPUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)

        messagebox.showinfo("保存完了",
                            f"統合後ID数: {len(grp)}\n\n"
                            f"・{MERGE_MAP_CSV}（old_id→new_id の対応表）\n"
                            f"・{OUTPUT_CSV}（orig_player_id 列に元IDを保持）\n\n"
                            "元データ player_positions_all.csv は変更していません。\n"
                            "次回起動時は merge_map.csv から作業を自動再開します。")


def main() -> None:
    root = tk.Tk()
    MergeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
