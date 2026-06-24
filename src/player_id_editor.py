"""
選手IDエディター（画像ベース・結合/カット）

players_hi.csv のボックスを高fpsフレーム上に player_id で色分け表示し、
実際の映像を見ながら ID を直す。
  - ボックスをクリック → そのトラックを選択（全出現を強調、開始/終了へジャンプ可）
  - 結合: 選択中に別ボックスをクリック → そのトラックを選択トラックへ統合
  - カット: 選択トラックを現フレームで分割（別人にまたがる時）
  - 「次の結合候補」: あるトラックの終了が別トラックの開始に時間・距離で近い箇所へジャンプ

保存: outputs/id_edits.json（cuts と merges）。順序非依存（カットは境界フレームで表現）。
反映: .venv/bin/python build_player_ids.py   （id_edits.json を適用して players_hi.csv 更新）

起動:
  ./run_id_gui.sh
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

FRAMES_DIR = Path("frames")
PLAYERS_CSV = Path("outputs/players_hi.csv")
EDITS_JSON = Path("outputs/id_edits.json")
BASE_W = 1920.0
FACTOR = 6   # キーフレーム間隔（12fps / 2fps）。j%FACTOR==0 がキーフレーム
MAX_W, MAX_H = 1500, 820
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 12.0, 1.3

PALETTE = ["#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de", "#ff2d55",
           "#5ac8fa", "#ffcc00", "#4cd964", "#5856d6", "#e67e22", "#1abc9c",
           "#e84393", "#00b894", "#0984e3", "#fdcb6e", "#6c5ce7", "#d63031",
           "#00cec9", "#fab1a0", "#74b9ff", "#a29bfe"]


def pid_color(pid: int) -> str:
    return PALETTE[pid % len(PALETTE)]


class Box:
    __slots__ = ("frame", "track_id", "team", "x1", "y1", "x2", "y2")

    def __init__(self, frame, tid, team, x1, y1, x2, y2):
        self.frame, self.track_id, self.team = frame, tid, team
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class IDEditor:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("選手IDエディター（結合/カット）")

        self.frame_w = self._frame_width()
        self.coord_scale = self.frame_w / BASE_W

        self.boxes_by_frame: dict[int, list[Box]] = defaultdict(list)
        self._load_players()
        self.frames = sorted(self.boxes_by_frame)
        if not self.frames:
            raise SystemExit(f"{PLAYERS_CSV} にデータがありません")
        self._all_frames = list(self.frames)   # 全フレーム（ナビ切替の元）

        # 編集: cuts[track_id]=[境界frame], merges=union-find(sub_id),
        #       geom[(frame,track_id)]=(x1,y1,x2,y2)=移動/リサイズ, names[canon_str]=名前
        self.cuts: dict[int, list[int]] = defaultdict(list)
        self.merge_parent: dict[tuple, tuple] = {}
        self.geom: dict[tuple, tuple] = {}
        self.names: dict[str, str] = {}
        self.frame_labels: dict[int, str] = {}   # frame -> 注記（例: ボール場外）。問題判定から除外
        self._load_edits()

        self.idx = 0
        self.selected_pid: int | None = None
        self.mode = "select"      # select / adjust / add
        self._drag = None         # ("move",dx,dy) / ("resize",corner) / ("create",)
        self._yolo = None         # YOLO遅延ロード（1コマ位置合わせ用）
        self._added_seq = -1      # 手追加ボックスの track_id（負の連番）
        self._pid_assign = {}     # canonical -> 表示player_id（安定・振り直さない）
        self._next_pid = 1
        self.scale = self.base_scale = 1.0
        self.zoom = 1.0
        self.img_w = self.img_h = 0
        self.tk_img = None
        self._pil = None

        self._init_styles()
        self._build_ui()
        self._recompute_pids()
        self._show()

    # ---- データ ----
    def _frame_width(self) -> float:
        fs = sorted(FRAMES_DIR.glob("frame_*.jpg"))
        if fs:
            try:
                return float(Image.open(fs[0]).size[0])
            except Exception:
                pass
        return BASE_W

    def _load_players(self):
        with PLAYERS_CSV.open() as f:
            for r in csv.DictReader(f):
                fr = int(r["frame"])
                self.boxes_by_frame[fr].append(Box(
                    fr, int(r["track_id"]), int(r["team"]),
                    float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))

    def _load_edits(self):
        self._added_ids = set()
        if not EDITS_JSON.exists():
            return
        try:
            d = json.loads(EDITS_JSON.read_text())
            for k, v in d.get("cuts", {}).items():
                self.cuts[int(k)] = sorted(int(x) for x in v)
            for a, b in d.get("merges", []):
                self._union(tuple(a), tuple(b))
            for k, v in d.get("geom", {}).items():
                fr, tid = k.split("_")
                self.geom[(int(fr), int(tid))] = tuple(v)
            self.names = dict(d.get("names", {}))
            # 手追加ボックスを復元
            for a in d.get("added", []):
                tid = int(a["track_id"])
                self._added_ids.add(tid)
                self._added_seq = min(self._added_seq, tid - 1)
                self.boxes_by_frame[int(a["frame"])].append(Box(
                    int(a["frame"]), tid, int(a["team"]),
                    float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])))
            self.frame_labels = {int(k): v for k, v in d.get("frame_labels", {}).items()}
            self._fix_names()   # 読み込んだ結合に合わせて名前を現代表へ
        except Exception:
            pass

    def _save(self):
        # merges を代表ペアの列として保存
        merges = []
        for sub in list(self.merge_parent):
            root = self._find(sub)
            if root != sub:
                merges.append([list(sub), list(root)])
        added = []
        for fr, bs in self.boxes_by_frame.items():
            for b in bs:
                if b.track_id in self._added_ids:
                    g = self.eff_geom(b)
                    added.append({"frame": fr, "track_id": b.track_id, "team": b.team,
                                  "x1": round(g[0], 1), "y1": round(g[1], 1),
                                  "x2": round(g[2], 1), "y2": round(g[3], 1)})
        EDITS_JSON.parent.mkdir(parents=True, exist_ok=True)
        EDITS_JSON.write_text(json.dumps(
            {"cuts": {str(k): v for k, v in self.cuts.items() if v},
             "merges": merges,
             "geom": {f"{fr}_{tid}": list(v) for (fr, tid), v in self.geom.items()},
             "names": self.names,
             "added": added,
             "frame_labels": {str(k): v for k, v in self.frame_labels.items()}},
            ensure_ascii=False, indent=2))
        n = len(set(self._display_pid_map().values()))
        messagebox.showinfo("保存", f"{EDITS_JSON} に保存しました。\n"
                            f"現在のID数: {n}\n\n"
                            "build_player_ids.py を実行すると players_hi.csv に反映されます。")

    # ---- sub_id（track_id, 直前のカット境界）----
    def _sub_id(self, box: Box) -> tuple:
        cs = self.cuts.get(box.track_id)
        if not cs:
            return (box.track_id, 0)
        i = bisect.bisect_right(cs, box.frame)
        return (box.track_id, cs[i - 1] if i > 0 else 0)

    # union-find over sub_id
    def _find(self, x):
        self.merge_parent.setdefault(x, x)
        while self.merge_parent[x] != x:
            self.merge_parent[x] = self.merge_parent[self.merge_parent[x]]
            x = self.merge_parent[x]
        return x

    def _union(self, a, b):
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.merge_parent[rb] = ra

    # ---- display player_id（1..N, フレーム数多い順）----
    def _recompute_pids(self):
        canon_frames = defaultdict(int)
        for fr, bs in self.boxes_by_frame.items():
            for b in bs:
                canon_frames[self._find(self._sub_id(b))] += 1
        # 番号は安定: 初出のcanonicalにだけ次番号を割当（既存は不変＝振り直さない）。
        # 結合で消えたcanonicalの番号は欠番になる（他の選手の番号は動かない）。
        for c in sorted(canon_frames, key=lambda c: -canon_frames[c]):
            if c not in self._pid_assign:
                self._pid_assign[c] = self._next_pid
                self._next_pid += 1
        self._pid_of_canon = {c: self._pid_assign[c] for c in canon_frames}
        self._n_ids = len(canon_frames)

    def _display_pid_map(self):
        return self._pid_of_canon

    def box_pid(self, b: Box) -> int:
        return self._pid_of_canon[self._find(self._sub_id(b))]

    def canon_str(self, b: Box) -> str:
        c = self._find(self._sub_id(b))
        return f"{c[0]}_{c[1]}"

    def box_name(self, b: Box) -> str:
        return self.names.get(self.canon_str(b), "")

    def eff_geom(self, b: Box):
        """移動/リサイズの上書きがあればそれを、無ければ元のボックスを返す。"""
        return self.geom.get((b.frame, b.track_id), (b.x1, b.y1, b.x2, b.y2))

    # ---- UI ----
    def _init_styles(self):
        self._st = ttk.Style()
        try:
            self._st.theme_use("clam")
        except tk.TclError:
            pass
        self._stc = set()

    def _cbtn(self, parent, text, cmd, bg, fg="white"):
        name = f"e{abs(hash((bg, fg))) % 1_000_000}.TButton"
        if name not in self._stc:
            h = bg.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            dark = f"#{int(r*.85):02x}{int(g*.85):02x}{int(b*.85):02x}"
            self._st.configure(name, background=bg, foreground=fg, font=("Helvetica", 11))
            self._st.map(name, background=[("active", dark), ("pressed", dark)])
            self._stc.add(name)
        return ttk.Button(parent, text=text, command=cmd, style=name)

    def _build_ui(self):
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Button(bar, text="◀", command=self._prev, width=3).pack(side=tk.LEFT)
        tk.Button(bar, text="▶", command=self._next, width=3).pack(side=tk.LEFT, padx=(2, 8))
        self.kf_only = tk.BooleanVar(value=False)
        tk.Checkbutton(bar, text="キーフレームのみ", variable=self.kf_only,
                       command=self._toggle_kf, font=("Helvetica", 11, "bold"),
                       fg="#007aff").pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(bar, text="画像へ:").pack(side=tk.LEFT)
        self.jump_var = tk.IntVar(value=self.frames[0] + 1)
        je = tk.Spinbox(bar, from_=1, to=self.frames[-1] + 1, width=7, textvariable=self.jump_var)
        je.pack(side=tk.LEFT)
        je.bind("<Return>", lambda e: self._jump())
        tk.Button(bar, text="移動", command=self._jump).pack(side=tk.LEFT, padx=(2, 12))
        tk.Button(bar, text="－", command=lambda: self._zoom_by(1/ZOOM_STEP)).pack(side=tk.LEFT)
        self.zoom_label = tk.Label(bar, text="100%", width=5)
        self.zoom_label.pack(side=tk.LEFT)
        tk.Button(bar, text="＋", command=lambda: self._zoom_by(ZOOM_STEP)).pack(side=tk.LEFT)
        tk.Button(bar, text="全体", command=self._zoom_fit).pack(side=tk.LEFT, padx=(2, 12))
        self._cbtn(bar, "保存", self._save, "#34c759").pack(side=tk.RIGHT)

        # モード
        modebar = tk.Frame(self.root)
        modebar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(modebar, text="モード:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="select")
        for label, val in [("選択/結合", "select"), ("調整(移動/リサイズ)", "adjust"), ("追加", "add")]:
            tk.Radiobutton(modebar, text=label, variable=self.mode_var, value=val,
                           indicatoron=False, padx=10, pady=3,
                           command=self._on_mode).pack(side=tk.LEFT, padx=2)
        tk.Label(modebar, text="追加色:").pack(side=tk.LEFT, padx=(8, 0))
        self.add_team = tk.IntVar(value=1)
        tk.Radiobutton(modebar, text="緑", variable=self.add_team, value=1, fg="#20a020").pack(side=tk.LEFT)
        tk.Radiobutton(modebar, text="黒", variable=self.add_team, value=2).pack(side=tk.LEFT)
        tk.Label(modebar, text="追加先ID(空=新規):").pack(side=tk.LEFT, padx=(8, 0))
        self.add_target = tk.IntVar(value=0)
        tk.Spinbox(modebar, from_=0, to=99999, width=6, textvariable=self.add_target).pack(side=tk.LEFT)
        self._cbtn(modebar, "名前を付ける/変更", self._name_selected, "#5856d6").pack(side=tk.LEFT, padx=8)
        self._cbtn(modebar, "このボックスを元に戻す", self._reset_geom, "#8e8e93").pack(side=tk.LEFT, padx=2)
        self._cbtn(modebar, "このコマをYOLO位置合わせ", self._snap_frame, "#00a896").pack(side=tk.LEFT, padx=2)

        # ID打ち込みで結合（フレームをまたいでOK）
        mb = tk.Frame(self.root)
        mb.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(mb, text="ID打ち込み結合:").pack(side=tk.LEFT)
        self.merge_a = tk.IntVar(); self.merge_b = tk.IntVar()
        tk.Spinbox(mb, from_=1, to=99999, width=6, textvariable=self.merge_a).pack(side=tk.LEFT, padx=2)
        tk.Label(mb, text="→").pack(side=tk.LEFT)
        tk.Spinbox(mb, from_=1, to=99999, width=6, textvariable=self.merge_b).pack(side=tk.LEFT, padx=2)
        self._cbtn(mb, "結合(a→b)", self._merge_by_id, "#007aff").pack(side=tk.LEFT, padx=4)
        tk.Label(mb, text="（選択中ボックスのIDは左に自動入力）", fg="#888").pack(side=tk.LEFT, padx=6)

        op = tk.Frame(self.root)
        op.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        self._cbtn(op, "結合: 選択へ取り込み", self._begin_merge, "#007aff").pack(side=tk.LEFT, padx=2)
        self._cbtn(op, "カット(このコマで分割)", self._cut, "#ff9500").pack(side=tk.LEFT, padx=2)
        self._cbtn(op, "選択解除", self._clear_sel, "#8e8e93").pack(side=tk.LEFT, padx=2)
        ttk.Separator(op, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        tk.Button(op, text="◀選択の開始", command=lambda: self._jump_track_edge(-1)).pack(side=tk.LEFT, padx=2)
        tk.Button(op, text="選択の終了▶", command=lambda: self._jump_track_edge(1)).pack(side=tk.LEFT, padx=2)
        self._cbtn(op, "次の結合候補", self._next_candidate, "#af52de").pack(side=tk.LEFT, padx=8)

        # 人数変化ジャンプ（選手が消えた/現れたコマへ）
        cc = tk.Frame(self.root)
        cc.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(cc, text="人数変化へ移動:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self._cbtn(cc, "◀ 前の減少", lambda: self._jump_count_change(-1, "dec"), "#d63031").pack(side=tk.LEFT, padx=2)
        self._cbtn(cc, "減った ▶", lambda: self._jump_count_change(1, "dec"), "#d63031").pack(side=tk.LEFT, padx=2)
        ttk.Separator(cc, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        self._cbtn(cc, "◀ 前の増加", lambda: self._jump_count_change(-1, "inc"), "#0984e3").pack(side=tk.LEFT, padx=2)
        self._cbtn(cc, "増えた ▶", lambda: self._jump_count_change(1, "inc"), "#0984e3").pack(side=tk.LEFT, padx=2)
        ttk.Separator(cc, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        self._cbtn(cc, "◀ 変化", lambda: self._jump_count_change(-1, "any"), "#636e72").pack(side=tk.LEFT, padx=2)
        self._cbtn(cc, "変化 ▶", lambda: self._jump_count_change(1, "any"), "#636e72").pack(side=tk.LEFT, padx=2)

        # 問題ジャンプ（人数過不足/重複/ワープ/ID入替）＋ コマのラベル
        pb = tk.Frame(self.root)
        pb.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(pb, text="問題へ移動:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self._cbtn(pb, "◀ 前の問題", lambda: self._jump_problem(-1), "#c0392b").pack(side=tk.LEFT, padx=2)
        self._cbtn(pb, "次の問題 ▶", lambda: self._jump_problem(1), "#c0392b").pack(side=tk.LEFT, padx=2)
        tk.Label(pb, text="（重複/22人外/ワープ/ID入替。ラベル付きコマは除外）", fg="#888").pack(side=tk.LEFT, padx=6)
        ttk.Separator(pb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        self._cbtn(pb, "このコマにラベル", self._label_frame, "#e67e22").pack(side=tk.LEFT, padx=2)
        for txt in ("ボール場外", "プレー外", "OK"):
            tk.Button(pb, text=txt, command=lambda t=txt: self._label_frame(t)).pack(side=tk.LEFT, padx=1)
        self._cbtn(pb, "ラベル消去", lambda: self._label_frame(""), "#8e8e93").pack(side=tk.LEFT, padx=4)

        self.status = tk.Label(self.root, text="", anchor="w", font=("Helvetica", 12))
        self.status.pack(side=tk.TOP, fill=tk.X, padx=8)
        self.help = tk.Label(self.root, text="ボックスをクリックで選択 → 「結合」後に別ボックスをクリックで取り込み。"
                             "カットは選択トラックをこのコマで分割。", anchor="w", fg="#555")
        self.help.pack(side=tk.TOP, fill=tk.X, padx=8)

        self.slider = tk.Scale(self.root, from_=0, to=len(self.frames) - 1, orient=tk.HORIZONTAL,
                               showvalue=False, command=self._on_slider)
        self.slider.pack(side=tk.TOP, fill=tk.X, padx=8)

        cw = tk.Frame(self.root)
        cw.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.canvas = tk.Canvas(cw, bg="black", highlightthickness=0, cursor="hand2",
                                width=MAX_W, height=MAX_H)
        hsb = tk.Scrollbar(cw, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vsb = tk.Scrollbar(cw, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Control-MouseWheel>", lambda e: self._zoom_by(ZOOM_STEP if e.delta > 0 else 1/ZOOM_STEP))
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("c", lambda e: self._cut())
        self.root.bind("m", lambda e: self._begin_merge())

        self._merge_mode = False

    # ---- 表示 ----
    def _cur(self):
        return self.frames[self.idx]

    def _show(self):
        fr = self._cur()
        p = FRAMES_DIR / f"frame_{fr+1:06d}.jpg"
        if not p.exists():
            self.status.config(text=f"画像なし {p.name}")
            return
        self._pil = Image.open(p)
        self.img_w, self.img_h = self._pil.size
        self.base_scale = min(MAX_W/self.img_w, MAX_H/self.img_h, 1.0)
        self._render()

    def _render(self):
        self.scale = self.base_scale * self.zoom
        disp = self._pil.resize((max(1, int(self.img_w*self.scale)), max(1, int(self.img_h*self.scale))))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.config(scrollregion=(0, 0, disp.width, disp.height))
        self.zoom_label.config(text=f"{int(self.zoom*100)}%")
        self._redraw()

    def _s(self, v):  # 1920基準 → 表示px
        return v * self.coord_scale * self.scale

    def _redraw(self):
        self.canvas.delete("all")
        if self.tk_img:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        ndup = defaultdict(int)
        adjust = (self.mode_var.get() == "adjust") if hasattr(self, "mode_var") else False
        for b in self.boxes_by_frame.get(self._cur(), []):
            pid = self.box_pid(b)
            ndup[pid] += 1
            col = pid_color(pid)
            sel = (pid == self.selected_pid)
            g = self.eff_geom(b)
            x1, y1, x2, y2 = self._s(g[0]), self._s(g[1]), self._s(g[2]), self._s(g[3])
            w = 4 if sel else 2
            self.canvas.create_rectangle(x1-1, y1-1, x2+1, y2+1, outline="black", width=w+2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=col, width=w)
            name = self.box_name(b)
            tag = (name + " " if name else "") + f"{pid}" + ("●" if b.team == 1 else "■")
            self.canvas.create_text(x1+2, y1-8, text=tag, fill="black", anchor=tk.W,
                                    font=("Helvetica", 12, "bold"))
            self.canvas.create_text(x1+1, y1-9, text=tag, fill=col, anchor=tk.W,
                                    font=("Helvetica", 12, "bold"))
            # 調整モードの選択ボックスにリサイズハンドル
            if adjust and sel:
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    self.canvas.create_rectangle(hx-5, hy-5, hx+5, hy+5, fill="white", outline=col, width=2)
        fr = self._cur()
        sel = f"  選択ID={self.selected_pid}" if self.selected_pid else ""
        mm = "  【結合モード: 取り込むボックスをクリック】" if self._merge_mode else ""
        label = self.frame_labels.get(fr)
        lab = f"  🏷{label}" if label else ""
        probs = self._frame_problems(fr)
        warn = ("  ⚠" + " / ".join(probs)) if probs else ""
        self.status.config(
            text=f"画像{fr+1}  ({self.idx+1}/{len(self.frames)})  ID数{self._n_ids}（目標~22）"
                 f"  人数{len(self.boxes_by_frame.get(fr,[]))}{sel}{mm}{lab}{warn}",
            fg="#c0392b" if (probs and not label) else ("#1e7e34" if label else "#000"))

    # ---- 当たり判定 ----
    def _find_box(self, cx, cy):
        x, y = cx/(self.coord_scale*self.scale), cy/(self.coord_scale*self.scale)
        hit, best = None, None
        for b in self.boxes_by_frame.get(self._cur(), []):
            g = self.eff_geom(b)
            if g[0] <= x <= g[2] and g[1] <= y <= g[3]:
                area = (g[2]-g[0])*(g[3]-g[1])
                if best is None or area < best:
                    hit, best = b, area
        return hit

    def _on_mode(self):
        self.mode = self.mode_var.get()
        helps = {
            "adjust": "調整: 選択ボックスをドラッグで移動 / 四隅の白ハンドルでリサイズ。「元に戻す」で解除",
            "add": "追加: 空き領域をドラッグで新規ボックス（追加色を選んでおく）。減った選手を補える",
            "select": "選択/結合: クリックで選択 → 「結合」後に別ボックスで取り込み。c=カット。"
                      "離れたコマ同士はID打ち込み結合が便利",
        }
        self.help.config(text=helps.get(self.mode, ""))
        self._redraw()

    def _on_press(self, e):
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        ix, iy = cx/(self.coord_scale*self.scale), cy/(self.coord_scale*self.scale)
        mode = self.mode_var.get()

        if mode == "add":
            # 新規ボックスを作りドラッグで広げる（新規track_id）
            tid = self._added_seq
            self._added_seq -= 1
            self._added_ids.add(tid)
            b = Box(self._cur(), tid, self.add_team.get(), ix, iy, ix+1, iy+1)
            self.boxes_by_frame[self._cur()].append(b)
            self._adj_box = b
            self._drag = ("create",)
            self._recompute_pids()
            self.selected_pid = self.box_pid(b)
            self._redraw()
            return

        b = self._find_box(cx, cy)
        if mode == "adjust":
            if b is None:
                return
            self.selected_pid = self.box_pid(b)
            self.merge_a.set(self.selected_pid)
            self._adj_box = b
            g = list(self.eff_geom(b))
            corner = self._corner_near(g, ix, iy)
            self._drag = ("resize", corner) if corner else ("move", ix-g[0], iy-g[1])
            self._redraw()
            return
        # select / merge
        if b is None:
            return
        pid = self.box_pid(b)
        if self._merge_mode and self.selected_pid is not None and pid != self.selected_pid:
            self._do_merge(self.selected_pid, b)
            self._merge_mode = False
        else:
            self.selected_pid = pid
            self.merge_a.set(pid)
        self._redraw()

    def _corner_near(self, g, x, y):
        tol = 10/(self.coord_scale*self.scale)
        for name, (cx, cy) in {"nw": (g[0], g[1]), "ne": (g[2], g[1]),
                               "sw": (g[0], g[3]), "se": (g[2], g[3])}.items():
            if abs(x-cx) <= tol and abs(y-cy) <= tol:
                return name
        return None

    def _on_motion(self, e):
        if not self._drag:
            return
        b = self._adj_box
        ix = max(0, min(BASE_W, self.canvas.canvasx(e.x)/(self.coord_scale*self.scale)))
        iy = max(0, min(self.img_h/self.coord_scale, self.canvas.canvasy(e.y)/(self.coord_scale*self.scale)))
        kind = self._drag[0]
        if kind == "create":          # 追加: 直接ボックスを広げる
            b.x2, b.y2 = ix, iy
            self._redraw()
            return
        g = list(self.eff_geom(b))
        if kind == "move":
            _, dx, dy = self._drag
            w, h = g[2]-g[0], g[3]-g[1]
            g[0], g[1] = ix-dx, iy-dy
            g[2], g[3] = g[0]+w, g[1]+h
        elif kind == "resize":
            c = self._drag[1]
            if "n" in c: g[1] = iy
            if "s" in c: g[3] = iy
            if "w" in c: g[0] = ix
            if "e" in c: g[2] = ix
        self.geom[(b.frame, b.track_id)] = tuple(g)
        self._redraw()

    def _on_release(self, e):
        if not self._drag:
            return
        b = self._adj_box
        kind = self._drag[0]
        self._drag = None
        if kind == "create":
            if b.x1 > b.x2: b.x1, b.x2 = b.x2, b.x1
            if b.y1 > b.y2: b.y1, b.y2 = b.y2, b.y1
            if b.x2-b.x1 < 3 or b.y2-b.y1 < 3:   # 極小は破棄
                self.boxes_by_frame[b.frame].remove(b)
                self._added_ids.discard(b.track_id)
                self.selected_pid = None
                self._recompute_pids()
                self._redraw()
                return
            # 追加先IDが指定されていれば、その既存選手へ割り当てる（新番号を作らない）
            tgt = 0
            try:
                tgt = int(self.add_target.get())
            except (ValueError, tk.TclError):
                tgt = 0
            self._recompute_pids()
            if tgt > 0:
                tc = self._canon_of_pid(tgt)
                newc = self._find(self._sub_id(b))
                if tc is not None and tc != newc:
                    self._union(tc, newc)
                    self._merge_names(tc, newc)
                    self._recompute_pids()
                    self.selected_pid = tgt
                else:
                    messagebox.showinfo("追加先", f"ID {tgt} が見つからないので新規IDで追加しました。")
            self._redraw()
        else:
            g = list(self.geom.get((b.frame, b.track_id), self.eff_geom(b)))
            if g[0] > g[2]: g[0], g[2] = g[2], g[0]
            if g[1] > g[3]: g[1], g[3] = g[3], g[1]
            self.geom[(b.frame, b.track_id)] = tuple(g)

    def _snap_frame(self):
        """このコマだけ YOLO 検出して、各ボックスを最寄り検出にスナップ（geom上書き）。"""
        import photo_annotator as pa
        fr = self._cur()
        path = FRAMES_DIR / f"frame_{fr+1:06d}.jpg"
        if not path.exists():
            return
        if self._yolo is None:
            self.status.config(text="YOLOモデル読み込み中…"); self.root.update_idletasks()
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(str(pa.YOLO_MODEL_PATH))
            except Exception as ex:
                messagebox.showerror("YOLO", str(ex)); return
        self.status.config(text="このコマを検出中…"); self.root.update_idletasks()
        to_base = BASE_W / self.frame_w
        TOL = 110.0   # スナップ許容(1920基準px)。ズレが大きめでも拾えるよう広め
        res = self._yolo.predict(source=str(path), classes=[0], conf=0.2,
                                 imgsz=1920, verbose=False)
        dets = []
        for rr in res:
            if rr.boxes is None:
                continue
            for x in rr.boxes.xyxy.cpu().numpy():
                X1, Y1, X2, Y2 = (float(v)*to_base for v in x[:4])
                if X2-X1 >= 2 and Y2-Y1 >= 2:
                    dets.append(((X1+X2)/2, Y2, (X1, Y1, X2, Y2)))
        boxes = self.boxes_by_frame.get(fr, [])
        pairs = []
        for bi, b in enumerate(boxes):
            g = self.eff_geom(b); bx, by = (g[0]+g[2])/2, g[3]
            for di, (dx, dy, _) in enumerate(dets):
                d = (bx-dx)**2 + (by-dy)**2
                if d <= TOL**2:
                    pairs.append((d, bi, di))
        pairs.sort()
        ub, ud = set(), set(); n = 0
        for d, bi, di in pairs:
            if bi in ub or di in ud:
                continue
            ub.add(bi); ud.add(di)
            b = boxes[bi]
            self.geom[(b.frame, b.track_id)] = dets[di][2]
            n += 1
        self._redraw()
        miss = len(boxes) - n
        self.status.config(
            text=f"YOLO位置合わせ: {n}/{len(boxes)} スナップ（検出{len(dets)}・未スナップ{miss}"
                 f"{'＝隠れ/未検出は調整モードで' if miss else ''}）", fg="#007aff")

    def _merge_by_id(self):
        try:
            a = int(self.merge_a.get()); b = int(self.merge_b.get())
        except (ValueError, tk.TclError):
            return
        if a == b:
            return
        ca, cb = self._canon_of_pid(a), self._canon_of_pid(b)
        if ca is None or cb is None:
            messagebox.showinfo("結合", f"ID {a} または {b} が見つかりません。")
            return
        self._union(cb, ca)   # a を b へ取り込み（b側を残す）
        self._merge_names(cb, ca)   # 名前は b 優先で引き継ぎ
        self._recompute_pids()
        self.selected_pid = self._pid_of_canon[self._find(cb)]
        self._redraw()
        self.status.config(text=f"ID {a} を {b} に結合しました", fg="#007aff")

    def _reset_geom(self):
        if self.selected_pid is None:
            return
        fr = self._cur()
        for b in self.boxes_by_frame.get(fr, []):
            if self.box_pid(b) == self.selected_pid:
                self.geom.pop((b.frame, b.track_id), None)
        self._redraw()

    def _name_selected(self):
        if self.selected_pid is None:
            messagebox.showinfo("名前", "先にボックスをクリックで選択してください。")
            return
        from tkinter import simpledialog
        canon = self._canon_of_pid(self.selected_pid)
        key = f"{canon[0]}_{canon[1]}"
        cur = self.names.get(key, "")
        name = simpledialog.askstring("名前", f"ID {self.selected_pid} の選手名:", initialvalue=cur, parent=self.root)
        if name is None:
            return
        if name.strip():
            self.names[key] = name.strip()
        else:
            self.names.pop(key, None)
        self._redraw()

    # ---- 編集操作 ----
    def _canon_of_pid(self, pid):
        for c, p in self._pid_of_canon.items():
            if p == pid:
                return c
        return None

    def _begin_merge(self):
        if self.selected_pid is None:
            messagebox.showinfo("結合", "先に取り込み先のボックスをクリックで選択してください。")
            return
        self._merge_mode = True
        self._redraw()

    def _merge_names(self, tgt_canon, src_canon):
        """結合後、名前を新しい代表へ引き継ぐ（取り込み先優先・取り込まれた側を掃除）。"""
        tk = f"{tgt_canon[0]}_{tgt_canon[1]}"
        sk = f"{src_canon[0]}_{src_canon[1]}"
        name = self.names.get(tk) or self.names.get(sk)
        self.names.pop(tk, None)
        self.names.pop(sk, None)
        root = self._find(tgt_canon)
        if name:
            self.names[f"{root[0]}_{root[1]}"] = name

    def _fix_names(self):
        """全ての名前を現在の代表(root)に付け替える（読み込み後の保険）。"""
        new = {}
        for key, nm in self.names.items():
            a, b = key.split("_")
            r = self._find((int(a), int(b)))
            new.setdefault(f"{r[0]}_{r[1]}", nm)
        self.names = new

    def _do_merge(self, target_pid, src_box: Box):
        tgt_canon = self._canon_of_pid(target_pid)
        src_canon = self._find(self._sub_id(src_box))
        if tgt_canon is None or tgt_canon == src_canon:
            return
        self._union(tgt_canon, src_canon)
        self._merge_names(tgt_canon, src_canon)
        self._recompute_pids()
        self.selected_pid = self._pid_of_canon[self._find(tgt_canon)]

    def _cut(self):
        if self.selected_pid is None:
            messagebox.showinfo("カット", "先にトラック（ボックス）を選択してください。")
            return
        fr = self._cur()
        # 選択IDのうち、現フレームに居るボックスの track_id をこのコマで分割
        targets = [b for b in self.boxes_by_frame.get(fr, []) if self.box_pid(b) == self.selected_pid]
        if not targets:
            messagebox.showinfo("カット", "このコマに選択トラックのボックスがありません。")
            return
        for b in targets:
            cs = self.cuts[b.track_id]
            if fr not in cs:
                bisect.insort(cs, fr)
        self._recompute_pids()
        self.selected_pid = None
        self._redraw()
        self.status.config(text=f"画像{fr+1} でカットしました（このコマ以降は別IDに分割）", fg="#007aff")

    def _clear_sel(self):
        self.selected_pid = None
        self._merge_mode = False
        self._redraw()

    # ---- ナビ ----
    def _selected_frames(self):
        if self.selected_pid is None:
            return []
        frs = [fr for fr, bs in self.boxes_by_frame.items()
               if any(self.box_pid(b) == self.selected_pid for b in bs)]
        return sorted(frs)

    def _jump_track_edge(self, direction):
        frs = self._selected_frames()
        if not frs:
            return
        target = frs[-1] if direction > 0 else frs[0]
        self._goto_frame(target)

    def _next_candidate(self):
        """選択トラックの終了に時間・距離で近い別トラックの開始へジャンプ（結合候補）。"""
        if self.selected_pid is None:
            messagebox.showinfo("結合候補", "先にトラックを選択してください。")
            return
        frs = self._selected_frames()
        if not frs:
            return
        last = frs[-1]
        # 選択トラックの終了位置
        endb = next((b for b in self.boxes_by_frame.get(last, []) if self.box_pid(b) == self.selected_pid), None)
        if endb is None:
            return
        ex, ey = (endb.x1+endb.x2)/2, endb.y2
        team = endb.team
        best = None
        for fr in range(last+1, min(last+120, self.frames[-1])+1):
            for b in self.boxes_by_frame.get(fr, []):
                pid = self.box_pid(b)
                if pid == self.selected_pid or b.team != team:
                    continue
                # そのトラックの開始フレームか（直前フレームに同pidが居ない）
                prev = self.boxes_by_frame.get(fr-1, [])
                if any(self.box_pid(pb) == pid for pb in prev):
                    continue
                bx, by = (b.x1+b.x2)/2, b.y2
                d = ((ex-bx)**2 + (ey-by)**2) ** 0.5 / self.coord_scale
                cost = d + (fr-last)*2
                if best is None or cost < best[0]:
                    best = (cost, fr)
        if best is None:
            messagebox.showinfo("結合候補", "近い結合候補は見つかりませんでした。")
            return
        self._goto_frame(best[1])
        self.status.config(text=f"結合候補: 画像{best[1]+1} 付近。映像で同一選手か確認して「結合」", fg="#af52de")

    def _jump_count_change(self, direction, kind):
        """人数（ボックス数）が前のコマから変化した場面へ移動。kind: dec/inc/any。"""
        n = len(self.frames)
        def cnt(i):
            return len(self.boxes_by_frame.get(self.frames[i], []))
        j = self.idx + direction
        while 0 <= j < n:
            prev = cnt(j - 1) if j - 1 >= 0 else cnt(j)
            c = cnt(j)
            hit = ((kind == "dec" and c < prev) or
                   (kind == "inc" and c > prev) or
                   (kind == "any" and c != prev))
            if hit:
                self._goto_idx(j)
                sign = "減" if c < prev else "増"
                self.status.config(
                    text=f"画像{self.frames[j]+1}: 人数 {prev}→{c}（{sign}）", fg="#007aff")
                return
            j += direction
        messagebox.showinfo("人数変化", "この方向に該当する人数変化はありませんでした。")

    def _toggle_kf(self):
        """キーフレームのみ / 全フレーム を切り替え（現在位置に最も近いコマへ）。"""
        cur = self._cur()
        if self.kf_only.get():
            self.frames = [f for f in self._all_frames if f % FACTOR == 0]
        else:
            self.frames = list(self._all_frames)
        if not self.frames:
            self.frames = list(self._all_frames)
            self.kf_only.set(False)
        self.idx = min(range(len(self.frames)), key=lambda i: abs(self.frames[i] - cur))
        self.slider.config(to=len(self.frames) - 1)
        self.slider.set(self.idx)
        self._show()
        n = len(self.frames)
        self.status.config(text=("キーフレームのみ表示" if self.kf_only.get() else "全フレーム表示")
                           + f"（{n}コマ）", fg="#007aff")

    def _goto_idx(self, i):
        self.idx = max(0, min(len(self.frames) - 1, i))
        self.slider.set(self.idx)
        self._show()

    # ---- コマのラベル ----
    def _label_frame(self, text=None):
        fr = self._cur()
        if text is None:
            from tkinter import simpledialog
            text = simpledialog.askstring("ラベル", f"画像{fr+1} の注記（例: ボール場外）:",
                                          initialvalue=self.frame_labels.get(fr, ""), parent=self.root)
            if text is None:
                return
        if text.strip():
            self.frame_labels[fr] = text.strip()
        else:
            self.frame_labels.pop(fr, None)
        self._redraw()

    # ---- 問題検出 ----
    def _frame_problems(self, fr):
        """このコマの問題リストを返す。ラベル付きは空（除外）。"""
        if fr in self.frame_labels:
            return []
        boxes = self.boxes_by_frame.get(fr, [])
        pids = {}
        for b in boxes:
            pids.setdefault(self.box_pid(b), []).append(b)
        probs = []
        dups = [p for p, bs in pids.items() if len(bs) >= 2]
        if dups:
            probs.append(f"ID重複 {dups}")
        n = len(boxes)
        if n != 22:
            probs.append(f"{'不足' if n < 22 else '超過'}{n}人")
        # 前コマ比較: ワープ / ID入替（消えて新ID出現）
        i = self._all_frames.index(fr) if fr in self._all_frames else -1
        if i > 0:
            pf = self._all_frames[i - 1]
            prev = {}
            for b in self.boxes_by_frame.get(pf, []):
                g = self.eff_geom(b)
                prev.setdefault(self.box_pid(b), ((g[0]+g[2])/2, g[3]))
            cur_ids = set(pids)
            prev_ids = set(prev)
            # ワープ: 同IDが前コマから大きく移動
            warps = []
            for p, bs in pids.items():
                if p in prev:
                    g = self.eff_geom(bs[0]); cx, cy = (g[0]+g[2])/2, g[3]
                    px, py = prev[p]
                    if ((cx-px)**2 + (cy-py)**2) ** 0.5 > 130:   # 1920基準px
                        warps.append(p)
            if warps:
                probs.append(f"IDワープ {warps}")
            gone = prev_ids - cur_ids
            new = cur_ids - prev_ids
            if gone and new:
                probs.append(f"ID入替(消{sorted(gone)}→出{sorted(new)})")
        return probs

    def _jump_problem(self, direction):
        n = len(self.frames)
        j = self.idx + direction
        while 0 <= j < n:
            probs = self._frame_problems(self.frames[j])
            if probs:
                self._goto_idx(j)
                self.status.config(text=f"画像{self.frames[j]+1} 問題: " + " / ".join(probs),
                                   fg="#c0392b")
                return
            j += direction
        messagebox.showinfo("問題", "この方向に未対応の問題は見つかりませんでした。")

    def _goto_frame(self, fr):
        if fr in self.boxes_by_frame:
            self.idx = self.frames.index(fr)
        else:
            self.idx = bisect.bisect_left(self.frames, fr)
            self.idx = max(0, min(len(self.frames)-1, self.idx))
        self.slider.set(self.idx)
        self._show()

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1; self.slider.set(self.idx); self._show()

    def _next(self):
        if self.idx < len(self.frames)-1:
            self.idx += 1; self.slider.set(self.idx); self._show()

    def _on_slider(self, v):
        i = int(v)
        if i != self.idx:
            self.idx = i; self._show()

    def _jump(self):
        try:
            t = int(self.jump_var.get())-1
        except (ValueError, tk.TclError):
            return
        self._goto_frame(t)

    def _zoom_by(self, f):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom*f)); self._render()

    def _zoom_fit(self):
        self.zoom = 1.0; self.canvas.xview_moveto(0); self.canvas.yview_moveto(0); self._render()


def main():
    argparse.ArgumentParser().parse_args()
    root = tk.Tk()
    IDEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
