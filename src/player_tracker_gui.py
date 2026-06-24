"""
選手トラッキング割当・訂正GUI（ロスター方式）

process_segment.py が作ったセグメント（高画質フレーム + YOLO/ByteTrackトラック）を読み、
  - 選手名簿（名前 + 緑/黒チーム）を保存しておき（outputs/roster.json, 全セグメント共通）
  - トラックをクリック → 名簿から選んでワンクリック割当（断片の統合も同じ操作）
  - 「いま消えている選手」を名簿の上に出す（再出現した断片を素早く再割当できる）
  - リジェクト（選手でない/誤検出）も1クリック
  - 入れ替わりは「このコマ以降だけ」チェックで前方訂正
  - 結果を移動データCSV（画面座標+ピッチ座標m）として書き出し

非破壊: 割当は outputs/segments/<name>/assignments.json、名簿は outputs/roster.json。

起動:
  ./run_gui.sh --name seg_bt24
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

MAX_W, MAX_H = 1480, 740
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 12.0, 1.3

ROSTER_PATH = Path("outputs/roster.json")
REJECT = "__reject__"

TEAM_COLOR = {"green": "#19d219", "black": "#000000", "other": "#ffd400"}
TEAM_LABEL = {"green": "緑", "black": "黒", "other": "他"}
UNASSIGNED_COLOR = "#ffffff"
REJECT_COLOR = "#7a2020"


def halo_of(hex_color: str) -> str:
    """暗い色には白、明るい色には黒の縁取り色を返す（背景で埋もれない用）。"""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return "#ffffff" if brightness < 110 else "#000000"


class Box:
    __slots__ = ("frame", "tid", "x1", "y1", "x2", "y2",
                 "foot_x", "foot_y", "x_pitch", "y_pitch", "in_pitch")

    def __init__(self, frame, tid, x1, y1, x2, y2, fx, fy, xp, yp, inp):
        self.frame, self.tid = frame, tid
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.foot_x, self.foot_y = fx, fy
        self.x_pitch, self.y_pitch, self.in_pitch = xp, yp, inp

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class TrackerGUI:
    def __init__(self, root, name):
        self.root = root
        self.seg = Path("outputs/segments") / name
        self.name = name
        self.assign_path = self.seg / "assignments.json"
        root.title(f"選手トラッキング割当 — {name}")

        self.meta = json.loads((self.seg / "meta.json").read_text())
        self.n_frames = self.meta["n_frames"]
        self.eff_fps = self.meta.get("eff_fps", 24.0)

        self.boxes_by_frame: dict[int, list[Box]] = defaultdict(list)
        self.tid_frames: dict[int, set[int]] = defaultdict(set)
        self._load_tracks()

        # 名簿: [{"name","team"}]、名前→team
        self.roster: list[dict] = []
        # 割当: base[tid]=name, overrides=[{frame,tid,name}]、name は選手名 or REJECT
        self.base: dict[int, str] = {}
        self.overrides: list[dict] = []
        self._load_roster()
        self._load_assignments()

        self.idx = 0
        self.selected_tid: int | None = None
        self.forward_var = None
        self._presence: list[set[str]] | None = None  # frame -> 表示中の選手名集合（キャッシュ）

        self.scale = self.base_scale = 1.0
        self.zoom = 1.0
        self.img_w = self.img_h = 0
        self.tk_img = None
        self._pil = None

        self._build_ui()
        self._show()

    # ---- 読み込み ----
    def _load_tracks(self):
        with (self.seg / "tracks.csv").open() as f:
            for r in csv.DictReader(f):
                fr, tid = int(r["frame"]), int(r["track_id"])
                xp = float(r["x_pitch"]) if r["x_pitch"] != "" else None
                yp = float(r["y_pitch"]) if r["y_pitch"] != "" else None
                self.boxes_by_frame[fr].append(Box(
                    fr, tid, float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                    float(r["foot_x"]), float(r["foot_y"]), xp, yp, int(r["in_pitch"])))
                self.tid_frames[tid].add(fr)

    def _load_roster(self):
        if ROSTER_PATH.exists():
            self.roster = json.loads(ROSTER_PATH.read_text()).get("players", [])

    def _save_roster(self):
        ROSTER_PATH.write_text(json.dumps({"players": self.roster}, ensure_ascii=False, indent=2))

    def _load_assignments(self):
        if self.assign_path.exists():
            data = json.loads(self.assign_path.read_text())
            self.base = {int(k): v for k, v in data.get("base", {}).items()}
            self.overrides = data.get("overrides", [])

    def _save_assignments(self):
        data = {"base": {str(k): v for k, v in self.base.items()}, "overrides": self.overrides}
        self.assign_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        self._save_roster()
        messagebox.showinfo("保存", f"名簿と割当を保存しました。")

    # ---- 識別の解決 ----
    def team_of(self, name: str) -> str:
        for p in self.roster:
            if p["name"] == name:
                return p["team"]
        return "other"

    def resolve(self, frame: int, tid: int) -> str | None:
        """(frame,tid)の選手名を返す。overrideが優先（frame以下で最新）、なければbase。"""
        best = self.base.get(tid)
        best_frame = -1 if best is not None else -2
        for ov in self.overrides:
            if ov["tid"] == tid and ov["frame"] <= frame and ov["frame"] > best_frame:
                best = ov["name"]
                best_frame = ov["frame"]
        return best

    def color_of(self, name: str | None) -> str:
        if name is None:
            return UNASSIGNED_COLOR
        if name == REJECT:
            return REJECT_COLOR
        return TEAM_COLOR.get(self.team_of(name), "#ffd400")

    # ---- presence キャッシュ ----
    def _invalidate_presence(self):
        self._presence = None

    def _build_presence(self):
        pres = [set() for _ in range(self.n_frames)]
        for fr in range(self.n_frames):
            for b in self.boxes_by_frame.get(fr, []):
                nm = self.resolve(fr, b.tid)
                if nm and nm != REJECT:
                    pres[fr].add(nm)
        self._presence = pres

    def presence(self) -> list[set[str]]:
        if self._presence is None:
            self._build_presence()
        return self._presence

    def last_seen(self, name: str, before: int) -> int:
        """before以下で name が最後に表示されたフレーム。無ければ -1。"""
        pres = self.presence()
        for f in range(min(before, self.n_frames - 1), -1, -1):
            if name in pres[f]:
                return f
        return -1

    # ---- UI ----
    def _build_ui(self):
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Button(bar, text="◀", command=self._prev, width=3).pack(side=tk.LEFT)
        tk.Button(bar, text="▶", command=self._next, width=3).pack(side=tk.LEFT, padx=(2, 8))
        self.play_btn = tk.Button(bar, text="再生", command=self._toggle_play, width=5)
        self.play_btn.pack(side=tk.LEFT)
        self._playing = False
        tk.Button(bar, text="－", command=lambda: self._zoom_by(1/ZOOM_STEP)).pack(side=tk.LEFT, padx=(12, 0))
        self.zoom_label = tk.Label(bar, text="100%", width=5)
        self.zoom_label.pack(side=tk.LEFT)
        tk.Button(bar, text="＋", command=lambda: self._zoom_by(ZOOM_STEP)).pack(side=tk.LEFT)
        tk.Button(bar, text="全体", command=self._zoom_fit).pack(side=tk.LEFT, padx=(2, 12))
        self.info = tk.Label(bar, text="", font=("Helvetica", 11))
        self.info.pack(side=tk.LEFT)
        tk.Button(bar, text="保存", command=self._save_assignments, bg="#34c759", fg="white",
                  font=("Helvetica", 11, "bold")).pack(side=tk.RIGHT)
        tk.Button(bar, text="CSV書き出し", command=self._export, bg="#007aff", fg="white",
                  font=("Helvetica", 11, "bold")).pack(side=tk.RIGHT, padx=6)

        self.slider = tk.Scale(self.root, from_=0, to=self.n_frames - 1, orient=tk.HORIZONTAL,
                               showvalue=False, command=self._on_slider)
        self.slider.pack(side=tk.TOP, fill=tk.X, padx=8)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        panel = tk.Frame(body, width=300, relief=tk.GROOVE, bd=1)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=4)
        panel.pack_propagate(False)
        self._build_panel(panel)

        cwrap = tk.Frame(body)
        cwrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.canvas = tk.Canvas(cwrap, bg="black", highlightthickness=0, cursor="hand2",
                                width=MAX_W, height=MAX_H)
        hsb = tk.Scrollbar(cwrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vsb = tk.Scrollbar(cwrap, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Control-MouseWheel>", lambda e: self._zoom_by(ZOOM_STEP if e.delta > 0 else 1/ZOOM_STEP))
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("r", lambda e: self._reject())

    def _build_panel(self, p):
        tk.Label(p, text="選択中のトラック", font=("Helvetica", 12, "bold")).pack(pady=(6, 0))
        self.sel_label = tk.Label(p, text="（ボックスをクリック）", fg="#555", wraplength=280)
        self.sel_label.pack(pady=2)

        self.forward_var = tk.BooleanVar(value=False)
        tk.Checkbutton(p, text="このコマ以降だけ変更（入れ替わり訂正）",
                       variable=self.forward_var, font=("Helvetica", 9)).pack()

        tk.Label(p, text="名簿（クリックで割当）", font=("Helvetica", 11, "bold")).pack(pady=(6, 0))
        tk.Label(p, text="↑消えている選手が上。★=今表示中",
                 fg="#888", font=("Helvetica", 9)).pack()
        rf = tk.Frame(p); rf.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.roster_box = tk.Listbox(rf, font=("Helvetica", 11), activestyle="none")
        rsb = tk.Scrollbar(rf, command=self.roster_box.yview)
        self.roster_box.config(yscrollcommand=rsb.set)
        rsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.roster_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.roster_box.bind("<ButtonRelease-1>", self._on_roster_click)
        self.roster_box.bind("<Double-Button-1>", self._on_roster_edit)
        self.roster_box.bind("<Button-2>", self._on_roster_edit)
        self.roster_box.bind("<Button-3>", self._on_roster_edit)

        bf = tk.Frame(p); bf.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(bf, text="リジェクト(r)", command=self._reject,
                  bg="#a33", fg="white").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(bf, text="割当解除", command=self._unassign).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        ttk.Separator(p).pack(fill=tk.X, pady=6)
        tk.Label(p, text="補正ショートカット", font=("Helvetica", 11, "bold")).pack()
        tk.Label(p, text="左クリックで基準IDを選択 → 別IDを右クリックで、このコマ以降の担当選手をスワップ",
                 fg="#666", font=("Helvetica", 9), wraplength=280, justify=tk.LEFT).pack(padx=8)
        sf = tk.Frame(p); sf.pack(fill=tk.X, padx=8, pady=2)
        tk.Button(sf, text="次の問題", command=lambda: self._jump_issue(1),
                  bg="#ff9500", fg="white").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(sf, text="前の問題", command=lambda: self._jump_issue(-1)).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.issue_label = tk.Label(p, text="", fg="#c62828", font=("Helvetica", 9),
                                    wraplength=280, justify=tk.LEFT)
        self.issue_label.pack(fill=tk.X, padx=8, pady=(0, 2))

        ttk.Separator(p).pack(fill=tk.X, pady=6)
        tk.Label(p, text="（名簿をダブルクリック/右クリックで名前変更）",
                 fg="#888", font=("Helvetica", 9)).pack()
        af = tk.Frame(p); af.pack(fill=tk.X, padx=8)
        tk.Button(af, text="＋追加", command=self._add_player).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(af, text="名前/チーム変更", command=self._edit_selected_player).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(p, text="緑11・黒11 を自動生成", command=self._seed_roster).pack(fill=tk.X, padx=8, pady=2)

    # ---- 表示 ----
    def _frame_boxes(self):
        return self.boxes_by_frame.get(self.idx, [])

    def _show(self):
        path = self.seg / "frames" / f"f_{self.idx:05d}.jpg"
        if not path.exists():
            self.info.config(text=f"画像なし: {path.name}")
            return
        self._pil = Image.open(path)
        self.img_w, self.img_h = self._pil.size
        self.base_scale = min(MAX_W / self.img_w, MAX_H / self.img_h, 1.0)
        self._render()

    def _render(self):
        self.scale = self.base_scale * self.zoom
        disp = self._pil.resize((max(1, int(self.img_w * self.scale)),
                                 max(1, int(self.img_h * self.scale))))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.config(scrollregion=(0, 0, disp.width, disp.height))
        self.zoom_label.config(text=f"{int(self.zoom*100)}%")
        self._redraw()

    def _sx(self, x): return x * self.scale
    def _sy(self, y): return y * self.scale

    def _redraw(self):
        self.canvas.delete("all")
        if self.tk_img:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        assigned = 0
        for b in self._frame_boxes():
            name = self.resolve(self.idx, b.tid)
            if name == REJECT:
                continue
            if name:
                assigned += 1
            color = self.color_of(name)
            halo = halo_of(color)
            x1, y1, x2, y2 = self._sx(b.x1), self._sy(b.y1), self._sx(b.x2), self._sy(b.y2)
            w = 3 if b.tid == self.selected_tid else 2
            # 縁取り（ハロー）を一回り大きく描いてからチーム色を上書き → 背景で埋もれない
            self.canvas.create_rectangle(x1-1, y1-1, x2+1, y2+1, outline=halo, width=w+2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=w)
            label = name if name else f"#{b.tid}"
            self.canvas.create_text(x1+1, y1 - 6, text=label, fill=halo, anchor=tk.W,
                                    font=("Helvetica", 11, "bold"))  # 影
            self.canvas.create_text(x1, y1 - 7, text=label, fill=color, anchor=tk.W,
                                    font=("Helvetica", 11, "bold"))
            fx, fy = self._sx(b.foot_x), self._sy(b.foot_y)
            self.canvas.create_oval(fx-3, fy-3, fx+3, fy+3, fill=color, outline=halo)
        t = self.idx / self.eff_fps
        self.info.config(text=f"フレーム {self.idx+1}/{self.n_frames}  t={t:.2f}s  "
                              f"検出{len(self._frame_boxes())}  割当済{assigned}")
        self._refresh_issue_label()
        self._refresh_roster()

    # ---- 名簿リスト（消えてる選手を上に） ----
    def _refresh_roster(self):
        pres_now = self.presence()[self.idx] if self.n_frames else set()
        # 各選手の (表示中?, 最後に見えたコマ) を計算し並べ替え
        entries = []
        for p in self.roster:
            nm = p["name"]
            visible = nm in pres_now
            ls = self.last_seen(nm, self.idx - 1)  # 今より前で最後に見えたコマ
            entries.append((nm, p["team"], visible, ls))
        # 消えている選手を上（最近消えた順）、表示中は下
        entries.sort(key=lambda e: (e[2], -(e[3])))  # visible=Falseが先, ls大きい順
        self.roster_box.delete(0, tk.END)
        self._roster_names = []
        for nm, team, visible, ls in entries:
            mark = "★" if visible else "　"
            if not visible and ls >= 0:
                gap = self.idx - ls
                note = f"  ({gap}コマ前に消失)"
            elif not visible:
                note = "  (未登場)"
            else:
                note = ""
            self.roster_box.insert(tk.END, f"{mark}[{TEAM_LABEL[team]}] {nm}{note}")
            self.roster_box.itemconfig(tk.END, foreground=TEAM_COLOR[team])
            self._roster_names.append(nm)

    # ---- 操作 ----
    def _find_box(self, cx, cy):
        x, y = cx / self.scale, cy / self.scale
        hit, best = None, None
        for b in self._frame_boxes():
            if b.contains(x, y):
                area = (b.x2-b.x1)*(b.y2-b.y1)
                if best is None or area < best:
                    hit, best = b, area
        return hit

    def _on_click(self, e):
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        b = self._find_box(cx, cy)
        if b is None:
            return
        self.selected_tid = b.tid
        name = self.resolve(self.idx, b.tid)
        cur = name if name else "未割当"
        self.sel_label.config(text=f"トラック #{b.tid}\n現在: {cur}", fg="#000")
        self._redraw()

    def _on_right_click(self, e):
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        b = self._find_box(cx, cy)
        if b is None:
            return
        if self.selected_tid is None:
            self.selected_tid = b.tid
            name = self.resolve(self.idx, b.tid)
            cur = name if name else "未割当"
            self.sel_label.config(text=f"トラック #{b.tid}\n現在: {cur}", fg="#000")
            self._redraw()
            return
        if b.tid == self.selected_tid:
            return
        self._swap_forward(self.selected_tid, b.tid)

    def _swap_forward(self, tid_a: int, tid_b: int):
        name_a = self.resolve(self.idx, tid_a)
        name_b = self.resolve(self.idx, tid_b)
        if not name_a or not name_b:
            messagebox.showinfo(
                "スワップ不可",
                "両方のトラックに選手が割り当たっている時だけスワップできます。\n"
                "未割当を含む場合は、名簿クリックで直接割り当ててください。")
            return
        if name_a == REJECT or name_b == REJECT:
            messagebox.showinfo("スワップ不可", "リジェクト済みIDは先に割当解除してください。")
            return
        self.overrides.append({"frame": self.idx, "tid": tid_a, "name": name_b})
        self.overrides.append({"frame": self.idx, "tid": tid_b, "name": name_a})
        self.selected_tid = tid_b
        self._invalidate_presence()
        self.sel_label.config(
            text=f"このコマ以降をスワップしました\n#{tid_a}: {name_a} → {name_b}\n#{tid_b}: {name_b} → {name_a}",
            fg="#007aff")
        self._redraw()

    def _apply(self, name: str | None):
        """選択トラックに name を割当（None=解除）。forward_varならこのコマ以降のみ。"""
        if self.selected_tid is None:
            messagebox.showinfo("未選択", "先に画面でトラック（ボックス）をクリックしてください。")
            return
        tid = self.selected_tid
        if self.forward_var.get() and name is not None:
            self.overrides.append({"frame": self.idx, "tid": tid, "name": name})
        else:
            if name is None:
                self.base.pop(tid, None)
                self.overrides = [o for o in self.overrides if o["tid"] != tid]
            else:
                self.base[tid] = name
                self.overrides = [o for o in self.overrides if o["tid"] != tid]
        self._invalidate_presence()
        self._redraw()

    def _on_roster_click(self, e):
        sel = self.roster_box.curselection()
        if not sel or sel[0] >= len(self._roster_names):
            return
        if self.selected_tid is None:
            return  # トラック未選択なら割当しない（選手行の選択のみ＝編集用）
        self._apply(self._roster_names[sel[0]])

    def _on_roster_edit(self, e):
        idx = self.roster_box.nearest(e.y)
        if 0 <= idx < len(self._roster_names):
            self.roster_box.selection_clear(0, tk.END)
            self.roster_box.selection_set(idx)
            self._edit_player(self._roster_names[idx])
        return "break"  # 通常のクリック割当を抑制

    def _edit_selected_player(self):
        sel = self.roster_box.curselection()
        if sel and sel[0] < len(self._roster_names):
            self._edit_player(self._roster_names[sel[0]])
        else:
            messagebox.showinfo("未選択", "名簿の選手を選んでから押してください。")

    def _edit_player(self, old_name: str):
        cur_team = self.team_of(old_name)
        win = tk.Toplevel(self.root); win.title("選手を編集")
        win.transient(self.root); win.grab_set()
        tk.Label(win, text="名前").pack(padx=20, pady=(10, 0))
        nv = tk.StringVar(value=old_name)
        ent = tk.Entry(win, textvariable=nv, font=("Helvetica", 12)); ent.pack(padx=20)
        ent.focus_set(); ent.select_range(0, tk.END)
        tk.Label(win, text="チーム").pack(padx=20, pady=(8, 0))
        tv = tk.StringVar(value=cur_team)
        for key in ("green", "black", "other"):
            tk.Radiobutton(win, text=TEAM_LABEL[key], variable=tv, value=key,
                           fg=TEAM_COLOR[key], font=("Helvetica", 12, "bold")).pack(anchor="w", padx=20)

        def save():
            new_name = nv.get().strip()
            if not new_name:
                return
            if new_name != old_name and any(pp["name"] == new_name for pp in self.roster):
                messagebox.showinfo("重複", "同名の選手が既にいます。", parent=win); return
            for pp in self.roster:
                if pp["name"] == old_name:
                    pp["name"] = new_name; pp["team"] = tv.get()
            if new_name != old_name:
                # 割当データも追従
                for tid, nm in list(self.base.items()):
                    if nm == old_name:
                        self.base[tid] = new_name
                for ov in self.overrides:
                    if ov["name"] == old_name:
                        ov["name"] = new_name
            self._save_roster(); self._invalidate_presence(); win.destroy(); self._redraw()

        bf = tk.Frame(win); bf.pack(pady=10)
        tk.Button(bf, text="保存", command=save, bg="#34c759", fg="white",
                  font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="キャンセル", command=win.destroy).pack(side=tk.LEFT, padx=4)
        ent.bind("<Return>", lambda e: save())

    def _reject(self):
        self._apply(REJECT)

    def _unassign(self):
        self._apply(None)

    def _add_player(self):
        name = simpledialog.askstring("選手を追加", "選手名（例: 緑7, 田中）:", parent=self.root)
        if not name:
            return
        if any(p["name"] == name for p in self.roster):
            messagebox.showinfo("重複", "同名の選手が既にいます。"); return
        win = tk.Toplevel(self.root); win.title("チーム選択")
        tk.Label(win, text=f"{name} のチーム").pack(padx=20, pady=8)
        tv = tk.StringVar(value="green")
        for key in ("green", "black", "other"):
            tk.Radiobutton(win, text=TEAM_LABEL[key], variable=tv, value=key,
                           fg=TEAM_COLOR[key], font=("Helvetica", 12, "bold")).pack(anchor="w", padx=20)
        def ok():
            self.roster.append({"name": name, "team": tv.get()})
            self._save_roster(); win.destroy(); self._redraw()
        tk.Button(win, text="追加", command=ok, bg="#34c759", fg="white").pack(pady=8)

    def _seed_roster(self):
        if self.roster and not messagebox.askyesno("確認", "既存の名簿に緑11・黒11を追加しますか？"):
            return
        exist = {p["name"] for p in self.roster}
        for team in ("green", "black"):
            for i in range(1, 12):
                nm = f"{TEAM_LABEL[team]}{i}"
                if nm not in exist:
                    self.roster.append({"name": nm, "team": team})
        self._save_roster(); self._redraw()

    # ---- 問題フレーム検出 ----
    def _frame_issues(self, frame: int) -> list[str]:
        counts: defaultdict[str, int] = defaultdict(int)
        unassigned = 0
        for b in self.boxes_by_frame.get(frame, []):
            name = self.resolve(frame, b.tid)
            if name is None:
                unassigned += 1
            elif name != REJECT:
                counts[name] += 1
        dup = [name for name, n in counts.items() if n >= 2]
        issues = []
        if dup:
            issues.append("同じ選手が同時に複数表示: " + ", ".join(dup[:4]))
        if unassigned:
            issues.append(f"未割当ID: {unassigned}件")
        return issues

    def _refresh_issue_label(self):
        issues = self._frame_issues(self.idx)
        if issues:
            self.issue_label.config(text=" / ".join(issues), fg="#c62828")
        else:
            self.issue_label.config(text="このフレームの重複・未割当はなし", fg="#555")

    def _jump_issue(self, direction: int):
        if self.n_frames <= 0:
            return
        start = self.idx
        for step in range(1, self.n_frames + 1):
            fr = (start + direction * step) % self.n_frames
            if self._frame_issues(fr):
                self.idx = fr
                self.slider.set(self.idx)
                self._show()
                return
        messagebox.showinfo("問題フレーム", "重複割当・未割当のあるフレームは見つかりませんでした。")

    # ---- ナビ ----
    def _prev(self):
        if self.idx > 0:
            self.idx -= 1; self.slider.set(self.idx); self._show()

    def _next(self):
        if self.idx < self.n_frames - 1:
            self.idx += 1; self.slider.set(self.idx); self._show()

    def _on_slider(self, v):
        i = int(v)
        if i != self.idx:
            self.idx = i; self._show()

    def _toggle_play(self):
        self._playing = not self._playing
        self.play_btn.config(text="停止" if self._playing else "再生")
        if self._playing:
            self._play_step()

    def _play_step(self):
        if not self._playing:
            return
        if self.idx < self.n_frames - 1:
            self._next()
            self.root.after(int(1000/self.eff_fps), self._play_step)
        else:
            self._playing = False; self.play_btn.config(text="再生")

    def _zoom_by(self, f):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * f)); self._render()

    def _zoom_fit(self):
        self.zoom = 1.0; self.canvas.xview_moveto(0); self.canvas.yview_moveto(0); self._render()

    # ---- 書き出し ----
    def _export(self):
        out = self.seg / "movement_data.csv"
        rows = []
        for fr in range(self.n_frames):
            t = fr / self.eff_fps
            for b in self.boxes_by_frame.get(fr, []):
                name = self.resolve(fr, b.tid)
                if name is None or name == REJECT:
                    continue
                rows.append({
                    "frame": fr, "time": round(t, 3),
                    "team": TEAM_LABEL[self.team_of(name)], "player": name,
                    "screen_x": b.foot_x, "screen_y": b.foot_y,
                    "pitch_x": b.x_pitch, "pitch_y": b.y_pitch, "in_pitch": b.in_pitch,
                    "track_id": b.tid,
                })
        fields = ["frame", "time", "team", "player", "screen_x", "screen_y",
                  "pitch_x", "pitch_y", "in_pitch", "track_id"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        n_players = len({r["player"] for r in rows})
        messagebox.showinfo("書き出し完了",
                            f"{out}\n{len(rows)}行  選手{n_players}人ぶんの移動データを書き出しました。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="seg_bt24")
    args = ap.parse_args()
    root = tk.Tk()
    TrackerGUI(root, args.name)
    root.mainloop()


if __name__ == "__main__":
    main()
