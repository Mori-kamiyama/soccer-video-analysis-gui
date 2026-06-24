"""
写真アノテーションGUI（検出ボックスの確認・追加・チーム色修正）

実際のフレーム画像に Moondream の検出ボックスを重ねて表示し、
  - 見逃された選手のボックスを追加（ドラッグで矩形を描く）
  - ボックスを「チーム色」に割り当てる
  - 余分なボックスを削除
できる。さらにキャリブレーション枠（pitch_points.json）を重ねて、
位置が下すぎる等の補正ズレを写真上で確認できる。

すべて非破壊: 元データは読むだけ。編集は outputs/box_corrections.csv に保存する。

起動:
  .venv/bin/python photo_annotator.py
"""

from __future__ import annotations

import colorsys
import csv
import json
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

FRAMES_DIR = Path("frames")
DET_CSV = Path("outputs/player_positions_all.csv")
MERGE_MAP_CSV = Path("outputs/merge_map.csv")
POINTS_PATH = Path("pitch_points.json")
CORRECTIONS_CSV = Path("outputs/box_corrections.csv")
TRIM_JSON = Path("outputs/trim_settings.json")
COLOR_CALIB_JSON = Path("outputs/team_color_calibration.json")
COLOR_MODEL_JSON = Path("outputs/team_color_knn_model.json")

MAX_W = 1500
MAX_H = 820
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 8.0, 1.25

TORSO_TOP = 0.10
TORSO_BOTTOM = 0.65
AUTO_GREEN_THRESH = 0.06
MIN_COLOR_PIXELS = 8
MAX_TRAIN_SAMPLES_PER_TEAM = 1000

# YOLO再検出（COCO person）設定
YOLO_MODEL_PATH = Path("models/yolo11m.pt")
YOLO_CONF = 0.25
YOLO_IMGSZ = 1920

DUP_COLORS = [
    "#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
    "#ff2d55", "#5ac8fa", "#ffcc00", "#4cd964", "#5856d6",
]

TEAM_GROUPS = {
    1: ("A", "緑", "#20d050"),
    2: ("B", "黒", "#111111"),
    3: ("?", "検出できなかった", "#ffcc00"),
    4: ("他", "その他", "#9b59b6"),
}
# ボールは専用ツール(ball_annotator.py)で扱うため、ここには置かない。

# 選手としてカウントするグループ（人数22/11-11チェックの対象）。
# その他はカウント外。group3(検出できなかった)は警告対象として残す。
PLAYER_GROUPS = (1, 2)
NON_PLAYER_GROUPS = (4,)  # その他: 警告を出さない正式カテゴリ
MAX_GROUP = max(TEAM_GROUPS)


def group_color(gid: int) -> str:
    """チームグループIDから表示色を返す。"""
    return TEAM_GROUPS.get(gid, TEAM_GROUPS[3])[2]


def group_label(gid: int) -> str:
    code, label, _ = TEAM_GROUPS.get(gid, TEAM_GROUPS[3])
    return f"{code}:{label}"


def group_from_team_hint(team_hint: str) -> int:
    hint = (team_hint or "").strip().upper()
    if hint in ("A", "GREEN", "緑"):
        return 1
    if hint in ("B", "BLACK", "黒"):
        return 2
    return 3


def green_ratio_from_rgb_pixels(pixels: list[tuple[int, int, int]]) -> float | None:
    if len(pixels) < MIN_COLOR_PIXELS:
        return None
    green = 0
    for r, g, b in pixels:
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        hue = h * 360.0
        # 砂/芝背景や黒つぶれを避けつつ、緑ビブスを拾う。
        if 65 <= hue <= 165 and s >= 0.25 and v >= 0.18 and g > r + 5 and g > b - 5:
            green += 1
    return green / len(pixels)


def color_feature_from_crop(crop: Image.Image) -> list[float] | None:
    """胴体cropから、緑比率だけに頼らない小さな色特徴を作る。"""
    arr = np.asarray(crop.convert("RGB"), dtype=np.uint8)
    if arr.size == 0 or arr.shape[0] * arr.shape[1] < MIN_COLOR_PIXELS:
        return None

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    rgb_f = arr.astype(np.float32) / 255.0
    hsv_f = hsv.astype(np.float32)
    lab_f = lab.astype(np.float32) / 255.0

    flat_rgb = rgb_f.reshape(-1, 3)
    flat_hsv = hsv.reshape(-1, 3)
    flat_lab = lab_f.reshape(-1, 3)

    h_hist = np.histogram(flat_hsv[:, 0], bins=12, range=(0, 180), density=False)[0].astype(np.float32)
    s_hist = np.histogram(flat_hsv[:, 1], bins=6, range=(0, 256), density=False)[0].astype(np.float32)
    v_hist = np.histogram(flat_hsv[:, 2], bins=6, range=(0, 256), density=False)[0].astype(np.float32)
    for hist in (h_hist, s_hist, v_hist):
        total = float(hist.sum())
        if total > 0:
            hist /= total

    r, g, b = flat_rgb[:, 0], flat_rgb[:, 1], flat_rgb[:, 2]
    dark_ratio = float(((r < 0.25) & (g < 0.25) & (b < 0.25)).mean())
    bright_ratio = float(((r > 0.75) | (g > 0.75) | (b > 0.75)).mean())
    green_ratio = green_ratio_from_rgb_pixels([tuple(map(int, px)) for px in arr.reshape(-1, 3)]) or 0.0

    feature = []
    feature.extend(h_hist.tolist())
    feature.extend(s_hist.tolist())
    feature.extend(v_hist.tolist())
    feature.extend(flat_rgb.mean(axis=0).tolist())
    feature.extend(flat_rgb.std(axis=0).tolist())
    feature.extend(flat_lab.mean(axis=0).tolist())
    feature.extend(flat_lab.std(axis=0).tolist())
    feature.extend([
        green_ratio,
        dark_ratio,
        bright_ratio,
        float((g - r).mean()),
        float((g - b).mean()),
        float((g / (r + b + 1e-4)).mean()),
    ])
    return [float(v) for v in feature]


class Box:
    __slots__ = ("frame", "key", "x1", "y1", "x2", "y2", "orig_pid",
                 "group", "init_group", "init_geom", "added", "deleted")

    def __init__(self, frame, key, x1, y1, x2, y2, orig_pid, group, added=False):
        self.frame = frame
        self.key = key  # 元検出は detection_id(int>=0)、追加は負の連番
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.orig_pid = orig_pid
        self.group = group
        self.init_group = group  # 初期グループ（再割当の検出用）
        self.init_geom = (x1, y1, x2, y2)  # 初期位置（移動検出用）
        self.added = added
        self.deleted = False

    @property
    def foot(self):
        return ((self.x1 + self.x2) / 2, self.y2)

    def moved(self) -> bool:
        return tuple(round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2)) != \
            tuple(round(v, 1) for v in self.init_geom)

    def norm(self) -> None:
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

    def contains(self, x, y) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def corner_near(self, x, y, tol):
        corners = {"nw": (self.x1, self.y1), "ne": (self.x2, self.y1),
                   "sw": (self.x1, self.y2), "se": (self.x2, self.y2)}
        for name, (cx, cy) in corners.items():
            if abs(x - cx) <= tol and abs(y - cy) <= tol:
                return name
        return None


class Annotator:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ボックス修正（位置・チーム色）")

        self.boxes_by_frame: dict[int, list[Box]] = defaultdict(list)
        self.frames_with_det: list[int] = []
        self._added_seq = -1
        self.current_group = 1
        self.selected: Box | None = None
        self.mode_var: tk.StringVar | None = None  # _build_ui で初期化（group/move/add/delete）
        self.auto_color_var: tk.BooleanVar | None = None
        self.auto_threshold = AUTO_GREEN_THRESH
        self.green_high_is_a = True
        self.color_model: dict | None = None
        self.calib_info = ""
        self._yolo_model = None  # YOLO再検出の遅延ロード用キャッシュ

        # 画像表示
        self.scale = 1.0
        self.base_scale = 1.0
        self.zoom = 1.0
        self.img_w = self.img_h = 0
        self.tk_img: ImageTk.PhotoImage | None = None

        # ドラッグ用（("create",) / ("move", dx, dy) / ("resize", corner)）
        self._drag = None

        # 保存座標は1920基準。表示フレームが2560等なら frame_w/1920 倍して内部はネイティブpxで扱う。
        self.coord_scale = self._detect_coord_scale()
        self.pitch_quad = self._load_pitch_quad()
        self._load_detections()
        self._load_corrections()
        self._load_color_calibration()

        self.all_frames = sorted(self.boxes_by_frame.keys())
        if not self.all_frames:
            raise SystemExit("検出データが見つかりません")

        # トリム設定をロード（なければ全範囲）
        trim = self._load_trim()
        self.trim_start = trim["start"]  # 画像番号（1始まり）
        self.trim_end = trim["end"]

        # frame_list = トリム後の表示リスト（データは消えない）
        self.frame_list = self._calc_frame_list()
        self.idx = max(0, min(len(self.frame_list) - 1, trim.get("last_idx", 0)))

        self._build_ui()
        self.slider.set(self.idx)
        self._show()

    # ---- トリム ----
    def _load_trim(self) -> dict:
        if TRIM_JSON.exists():
            try:
                return json.loads(TRIM_JSON.read_text())
            except Exception:
                pass
        return {"start": 1, "end": 999999, "last_idx": 0}

    def _save_trim(self) -> None:
        TRIM_JSON.write_text(json.dumps(
            {"start": self.trim_start, "end": self.trim_end, "last_idx": self.idx},
            indent=2))

    def _calc_frame_list(self) -> list[int]:
        """all_frames をトリム範囲で絞った表示リストを返す。データは消えない。"""
        lo = self.trim_start - 1  # 画像番号→frame番号
        hi = self.trim_end - 1
        return [f for f in self.all_frames if lo <= f <= hi]

    def _apply_trim(self) -> None:
        """トリム変更を表示に反映する。"""
        old_frame = self.frame_list[self.idx] if self.frame_list else None
        self.frame_list = self._calc_frame_list()
        if not self.frame_list:
            messagebox.showwarning("トリム", "その範囲に検出データがありません。")
            return
        # できるだけ元のフレームに近い位置を保持
        if old_frame is not None:
            self.idx = next(
                (i for i, f in enumerate(self.frame_list) if f >= old_frame), 0)
        else:
            self.idx = 0
        self.slider.config(to=len(self.frame_list) - 1)
        self.slider.set(self.idx)
        self._update_trim_labels()
        self._save_trim()
        self._show()

    def _update_trim_labels(self) -> None:
        total = len(self.all_frames)
        shown = len(self.frame_list)
        s = f"画像{self.trim_start}" if self.trim_start > 1 else "先頭"
        e = f"画像{self.trim_end}" if self.trim_end < 999999 else "末尾"
        self.trim_info.config(
            text=f"表示範囲: {s} 〜 {e}  ({shown}/{total} フレーム)")

    # ---- 読み込み ----
    def _load_pitch_quad(self):
        if POINTS_PATH.exists():
            try:
                quad = json.loads(POINTS_PATH.read_text()).get("image")
                if quad:
                    # pitch_points.json は1920基準 → 表示フレームpxへスケール
                    cs = self.coord_scale
                    return [[x * cs, y * cs] for x, y in quad]
                return quad
            except Exception:
                return None
        return None

    def _detect_coord_scale(self) -> float:
        """保存座標(1920基準)→表示フレームpx の倍率。フレーム幅/1920。"""
        files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
        if files:
            try:
                w, _ = Image.open(files[0]).size
                return w / 1920.0
            except Exception:
                pass
        return 1.0

    def _load_detections(self) -> None:
        cs = self.coord_scale
        with DET_CSV.open() as f:
            for r in csv.DictReader(f):
                try:
                    frame = int(r["frame"])
                    key = int(r["detection_id"])
                    x1, y1, x2, y2 = (float(r["x1"]) * cs, float(r["y1"]) * cs,
                                      float(r["x2"]) * cs, float(r["y2"]) * cs)
                    pid = int(r["player_id"])
                except (ValueError, KeyError):
                    continue
                group = group_from_team_hint(r.get("team_hint", r.get("team", "")))
                self.boxes_by_frame[frame].append(
                    Box(frame, key, x1, y1, x2, y2, pid, group))

    def _load_corrections(self) -> None:
        """前回の編集（追加/再割当/削除）を復元する。"""
        if not CORRECTIONS_CSV.exists():
            return
        cs = self.coord_scale
        index = {(b.frame, b.key): b for bs in self.boxes_by_frame.values() for b in bs}
        with CORRECTIONS_CSV.open() as f:
            for r in csv.DictReader(f):
                frame = int(r["frame"])
                key = int(r["box_key"])
                action = r["action"]
                if action == "add":
                    b = Box(frame, key, float(r["x1"]) * cs, float(r["y1"]) * cs,
                            float(r["x2"]) * cs, float(r["y2"]) * cs,
                            int(r["orig_player_id"]), int(r["group"]), added=True)
                    self.boxes_by_frame[frame].append(b)
                    index[(frame, key)] = b
                    self._added_seq = min(self._added_seq, key - 1)
                else:
                    b = index.get((frame, key))
                    if b is None:
                        continue
                    if action == "delete":
                        b.deleted = True
                    elif action == "reassign":
                        b.group = int(r["group"])
                    elif action == "move":
                        b.x1, b.y1 = float(r["x1"]) * cs, float(r["y1"]) * cs
                        b.x2, b.y2 = float(r["x2"]) * cs, float(r["y2"]) * cs

    # ---- 色付きボタン（macOS対応） ----
    def _init_button_styles(self) -> None:
        """macOSのネイティブtk.Buttonは bg= を無視して白いままになる。
        ttkのclamテーマなら背景色が反映されるので、色付きボタンはこちらで作る。"""
        self._btn_style = ttk.Style()
        try:
            self._btn_style.theme_use("clam")
        except tk.TclError:
            pass
        self._btn_style_cache: set[str] = set()

    def _color_button(self, parent, text, command, bg, fg="white", bold=False):
        """背景色つきのボタンを返す（macOSでも色が出る）。"""
        font = ("Helvetica", 11, "bold") if bold else ("Helvetica", 11)
        name = f"c{abs(hash((bg, fg, bold))) % 1_000_000}.TButton"
        if name not in self._btn_style_cache:
            darker = self._darken(bg)
            self._btn_style.configure(name, background=bg, foreground=fg,
                                      font=font, borderwidth=1, focuscolor=bg)
            self._btn_style.map(name,
                                background=[("active", darker), ("pressed", darker)],
                                foreground=[("disabled", "#cccccc")])
            self._btn_style_cache.add(name)
        return ttk.Button(parent, text=text, command=command, style=name)

    @staticmethod
    def _darken(hex_color: str, factor: float = 0.85) -> str:
        h = hex_color.lstrip("#")
        if len(h) == 3:  # #abc → #aabbcc に展開
            h = "".join(c * 2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"

    # ---- UI ----
    def _build_ui(self) -> None:
        self._init_button_styles()
        # モード選択バー
        modebar = tk.Frame(self.root)
        modebar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 0))
        tk.Label(modebar, text="モード:", font=("Helvetica", 12, "bold")).pack(side=tk.LEFT)
        self.mode_var = tk.StringVar(value="group")
        for label, val in [("① 色", "group"), ("② 移動/リサイズ", "move"),
                           ("③ 追加", "add"), ("④ 削除", "delete"),
                           ("⑤ 重複確認", "dup")]:
            tk.Radiobutton(modebar, text=label, variable=self.mode_var, value=val,
                           indicatoron=False, padx=10, pady=4,
                           command=self._on_mode_change).pack(side=tk.LEFT, padx=2)
        tk.Label(modebar, text="  重複しきい値(%):").pack(side=tk.LEFT, padx=(12, 0))
        self.dup_threshold_var = tk.IntVar(value=30)
        tk.Spinbox(modebar, from_=1, to=100, width=4, textvariable=self.dup_threshold_var,
                   command=self._redraw).pack(side=tk.LEFT, padx=2)

        # トリムバー
        trimbar = tk.Frame(self.root, relief=tk.GROOVE, bd=1)
        trimbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 0))
        tk.Label(trimbar, text="トリム:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        self.trim_start_var = tk.IntVar(value=self.trim_start)
        self.trim_end_var = tk.IntVar(value=self.trim_end if self.trim_end < 999999 else 999999)
        tk.Label(trimbar, text="開始画像:").pack(side=tk.LEFT)
        ts = tk.Spinbox(trimbar, from_=1, to=999999, width=7, textvariable=self.trim_start_var)
        ts.pack(side=tk.LEFT, padx=2)
        ts.bind("<Return>", lambda e: self._set_trim_from_fields())
        tk.Button(trimbar, text="現在を開始に",
                  command=self._set_start_here).pack(side=tk.LEFT, padx=2)
        tk.Label(trimbar, text="  終了画像:").pack(side=tk.LEFT)
        te = tk.Spinbox(trimbar, from_=1, to=999999, width=7, textvariable=self.trim_end_var)
        te.pack(side=tk.LEFT, padx=2)
        te.bind("<Return>", lambda e: self._set_trim_from_fields())
        tk.Button(trimbar, text="現在を終了に",
                  command=self._set_end_here).pack(side=tk.LEFT, padx=2)
        self._color_button(trimbar, "適用", self._set_trim_from_fields,
                           "#007aff").pack(side=tk.LEFT, padx=4)
        tk.Button(trimbar, text="全範囲に戻す",
                  command=self._reset_trim).pack(side=tk.LEFT, padx=2)
        self.trim_info = tk.Label(trimbar, text="", fg="#555", font=("Helvetica", 10))
        self.trim_info.pack(side=tk.LEFT, padx=8)

        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Button(bar, text="◀ 前", command=self._prev).pack(side=tk.LEFT)
        tk.Button(bar, text="次 ▶", command=self._next).pack(side=tk.LEFT, padx=(4, 12))

        # 画像番号ジャンプ
        tk.Label(bar, text="画像へ移動:").pack(side=tk.LEFT)
        self.jump_var = tk.IntVar(value=self.trim_start)
        je = tk.Spinbox(bar, from_=1, to=len(self.frame_list) + 10000, width=6,
                        textvariable=self.jump_var)
        je.pack(side=tk.LEFT)
        je.bind("<Return>", lambda e: self._jump_to_image())
        tk.Button(bar, text="移動", command=self._jump_to_image).pack(side=tk.LEFT, padx=(2, 12))

        # ズーム
        tk.Button(bar, text="－", command=lambda: self._zoom_by(1 / ZOOM_STEP)).pack(side=tk.LEFT)
        self.zoom_label = tk.Label(bar, text="100%", width=5)
        self.zoom_label.pack(side=tk.LEFT)
        tk.Button(bar, text="＋", command=lambda: self._zoom_by(ZOOM_STEP)).pack(side=tk.LEFT)
        tk.Button(bar, text="全体", command=self._zoom_fit).pack(side=tk.LEFT, padx=(2, 0))

        # 色操作は数が多いので2行に分ける（1行に詰めると画面右端からはみ出して押せなくなる）
        # 1行目: 色の選択と頻用操作（入替・警告ジャンプ）
        colorbar = tk.Frame(self.root)
        colorbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 2))

        tk.Label(colorbar, text="色操作:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(colorbar, text="現在の色:").pack(side=tk.LEFT, padx=(8, 0))
        self.group_var = tk.IntVar(value=self.current_group)
        self.group_swatch = tk.Label(colorbar, text=group_label(self.current_group),
                                     bg=group_color(self.current_group),
                                     fg="white", width=8)
        self.group_swatch.pack(side=tk.LEFT, padx=4)
        sp = tk.Spinbox(colorbar, from_=1, to=MAX_GROUP, width=3, textvariable=self.group_var,
                        command=self._on_group_change)
        sp.pack(side=tk.LEFT)
        sp.bind("<Return>", lambda e: self._on_group_change())
        self._color_button(colorbar, "緑(A)", lambda: self._set_group(1),
                           group_color(1), fg="white", bold=True).pack(side=tk.LEFT, padx=(6, 2))
        self._color_button(colorbar, "黒(B)", lambda: self._set_group(2),
                           group_color(2), fg="white", bold=True).pack(side=tk.LEFT, padx=2)
        self._color_button(colorbar, "未検出(?)", lambda: self._set_group(3),
                           group_color(3), fg="black", bold=True).pack(side=tk.LEFT, padx=2)
        self._color_button(colorbar, "その他", lambda: self._set_group(4),
                           group_color(4), fg="white", bold=True).pack(side=tk.LEFT, padx=2)
        self.auto_color_var = tk.BooleanVar(value=True)
        tk.Checkbutton(colorbar, text="追加/不明を自動色判定", variable=self.auto_color_var).pack(side=tk.LEFT, padx=8)

        # 2行目: 自動色判定・補正・学習（重い処理）
        colorbar2 = tk.Frame(self.root)
        colorbar2.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(colorbar2, text="自動処理:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self._color_button(colorbar2, "このフレームを自動色判定",
                           self._auto_color_current_frame,
                           "#00a896").pack(side=tk.LEFT, padx=2)
        self._color_button(colorbar2, "このF 11/11補正",
                           self._balance_current_frame_colors,
                           "#00a896").pack(side=tk.LEFT, padx=2)
        self._color_button(colorbar2, "このF ボックス再生成",
                           self._regenerate_current_frame_boxes,
                           "#007aff").pack(side=tk.LEFT, padx=2)
        self._color_button(colorbar2, "このFをYOLO再検出",
                           self._redetect_current_frame_yolo,
                           "#ff3b30").pack(side=tk.LEFT, padx=2)
        ttk.Separator(colorbar2, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        tk.Label(colorbar2, text="色判定の学習:").pack(side=tk.LEFT)
        self._color_button(colorbar2, "注釈から色判定を学習",
                           self._learn_color_from_annotations,
                           "#5856d6").pack(side=tk.LEFT, padx=2)
        self._color_button(colorbar2, "表示範囲で学習",
                           self._learn_color_from_visible_range,
                           "#5856d6").pack(side=tk.LEFT, padx=2)

        # 3行目: フレーム単位の補助操作（入替・警告ジャンプ）
        colorbar3 = tk.Frame(self.root)
        colorbar3.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(colorbar3, text="補助:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self._color_button(colorbar3, "このF 緑/黒入替",
                           self._swap_current_frame_colors,
                           "#8e8e93").pack(side=tk.LEFT, padx=2)
        ttk.Separator(colorbar3, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        self._color_button(colorbar3, "◀ 前の警告", lambda: self._jump_warning(-1),
                           "#ff9500").pack(side=tk.LEFT, padx=2)
        self._color_button(colorbar3, "次の警告 ▶", lambda: self._jump_warning(1),
                           "#ff9500").pack(side=tk.LEFT, padx=2)

        self.show_pitch = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="キャリブ枠を表示", variable=self.show_pitch,
                       command=self._show).pack(side=tk.LEFT, padx=10)
        self.show_foot = tk.BooleanVar(value=True)
        tk.Checkbutton(bar, text="足元点", variable=self.show_foot,
                       command=self._show).pack(side=tk.LEFT)

        self._color_button(bar, "保存", self._save, "#34c759",
                           bold=True).pack(side=tk.RIGHT)
        self._color_button(bar, "選択を削除", self._delete_selected,
                           "#a33").pack(side=tk.RIGHT, padx=4)

        self.status = tk.Label(self.root, text="", anchor="w", font=("Helvetica", 12),
                               justify=tk.LEFT)
        self.status.pack(side=tk.TOP, fill=tk.X, padx=8)
        self.warning_label = tk.Label(self.root, text="", anchor="w", font=("Helvetica", 12, "bold"),
                                      justify=tk.LEFT)
        self.warning_label.pack(side=tk.TOP, fill=tk.X, padx=8)

        self.slider = tk.Scale(self.root, from_=0, to=max(0, len(self.frame_list) - 1),
                               orient=tk.HORIZONTAL, showvalue=False, command=self._on_slider)
        self.slider.pack(side=tk.TOP, fill=tk.X, padx=8)
        self._update_trim_labels()

        # スクロール可能なキャンバス（ズーム時にパンできる）
        cwrap = tk.Frame(self.root)
        cwrap.pack(side=tk.TOP, padx=8, pady=6, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cwrap, bg="black", highlightthickness=0, cursor="crosshair",
                                width=MAX_W, height=MAX_H)
        hsb = tk.Scrollbar(cwrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vsb = tk.Scrollbar(cwrap, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # ホイール: 縦スクロール / Shift+横スクロール / Ctrl+ズーム
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_wheel_h)
        self.canvas.bind("<Control-MouseWheel>", self._on_wheel_zoom)
        # 中ボタンドラッグでパン
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<Delete>", lambda e: self._delete_selected())
        self.root.bind("<BackSpace>", lambda e: self._delete_selected())
        self.root.bind("1", lambda e: self._set_mode("group"))
        self.root.bind("2", lambda e: self._set_mode("move"))
        self.root.bind("3", lambda e: self._set_mode("add"))
        self.root.bind("4", lambda e: self._set_mode("delete"))
        self.root.bind("a", lambda e: self._set_group(1))
        self.root.bind("b", lambda e: self._set_group(2))
        self.root.bind("u", lambda e: self._set_group(3))
        self.root.bind("o", lambda e: self._set_group(4))  # その他
        self.root.bind("x", lambda e: self._swap_current_frame_colors())
        self.root.bind("j", lambda e: self._jump_warning(1))
        self.root.bind("J", lambda e: self._jump_warning(-1))
        self.root.bind("5", lambda e: self._set_mode("dup"))

        self.help_label = tk.Label(self.root, text="", anchor="w", fg="#555", justify=tk.LEFT)
        self.help_label.pack(side=tk.TOP, fill=tk.X, padx=8)
        self._on_mode_change()

    def _set_mode(self, mode: str) -> None:
        self.mode_var.set(mode)
        self._on_mode_change()

    def _on_mode_change(self) -> None:
        mode = self.mode_var.get()
        helps = {
            "group": "① 色: A=緑/B=黒/U=未検出/O=その他 で色を選び、ボックスをクリックして変更。Xで緑/黒入替、Jで次の警告、Shift+Jで前の警告（ボールは専用ツール）",
            "move": "② 移動/リサイズ: ボックス内をドラッグで移動 / 四隅の白ハンドルをドラッグでリサイズ",
            "add": "③ 追加: 空き領域をドラッグして新規ボックス（現在のチーム色）",
            "delete": "④ 削除: ボックスをクリックで削除（元検出は復元可能／追加は完全削除）",
            "dup": "⑤ 重複確認: 重なりあうボックスを同色でグループ表示。×N = N個重複。しきい値で「最小ボックスのN%以上重なり」を調整",
        }
        cursors = {"group": "hand2", "move": "fleur", "add": "tcross", "delete": "X_cursor", "dup": "arrow"}
        self.help_label.config(text=helps[mode])
        self.canvas.config(cursor=cursors.get(mode, "crosshair"))
        self._redraw()

    # ---- 表示 ----
    def _frame(self) -> int:
        return self.frame_list[self.idx]

    def _show(self) -> None:
        frame = self._frame()
        img_path = FRAMES_DIR / f"frame_{frame + 1:06d}.jpg"
        if not img_path.exists():
            self.status.config(text=f"画像が無い: {img_path}")
            return
        self._pil = Image.open(img_path)
        self.img_w, self.img_h = self._pil.size
        self.base_scale = min(MAX_W / self.img_w, MAX_H / self.img_h, 1.0)
        self._render()

    def _render(self) -> None:
        """現在のズームで画像を描き直す（フレームの再読み込みはしない）。"""
        self.scale = self.base_scale * self.zoom
        disp = self._pil.resize((max(1, int(self.img_w * self.scale)),
                                 max(1, int(self.img_h * self.scale))))
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.config(scrollregion=(0, 0, disp.width, disp.height))
        if hasattr(self, "zoom_label"):
            self.zoom_label.config(text=f"{int(self.zoom * 100)}%")
        self._redraw()

    def _sx(self, x): return x * self.scale
    def _sy(self, y): return y * self.scale

    def _redraw(self) -> None:
        if self.mode_var is not None and self.mode_var.get() == "dup":
            self._redraw_dup_mode()
            return
        self.canvas.delete("all")
        if self.tk_img is not None:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # キャリブレーション枠
        if self.show_pitch.get() and self.pitch_quad and len(self.pitch_quad) == 4:
            pts = [(self._sx(x), self._sy(y)) for x, y in self.pitch_quad]
            flat = [c for p in pts for c in p]
            self.canvas.create_polygon(*flat, outline="#00e5ff", fill="", width=2, dash=(6, 3))
            labels = ["奥左", "奥右", "手前右", "手前左"]
            for (px, py), lb in zip(pts, labels):
                self.canvas.create_text(px, py - 8, text=lb, fill="#00e5ff",
                                        font=("Helvetica", 11, "bold"))

        boxes = [b for b in self.boxes_by_frame[self._frame()] if not b.deleted]
        if self.auto_color_var is not None and self.auto_color_var.get():
            for b in boxes:
                if b.group == 3:
                    self._auto_color_box(b)
        for b in boxes:
            color = group_color(b.group)
            halo = "white" if b.group in (2, 4) else "black"  # 黒・紫(その他)は白縁取り
            x1, y1, x2, y2 = self._sx(b.x1), self._sy(b.y1), self._sx(b.x2), self._sy(b.y2)
            w = 4 if b is self.selected else 2
            self.canvas.create_rectangle(x1 - 1, y1 - 1, x2 + 1, y2 + 1, outline=halo, width=w + 2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=w)
            tag = group_label(b.group) + ("＋" if b.added else "")
            self.canvas.create_text(x1 + 3, y1 - 7, text=tag, fill=halo, anchor=tk.W,
                                    font=("Helvetica", 11, "bold"))
            self.canvas.create_text(x1 + 2, y1 - 8, text=tag, fill=color, anchor=tk.W,
                                    font=("Helvetica", 11, "bold"))
            if self.show_foot.get():
                fx, fy = b.foot
                fx, fy = self._sx(fx), self._sy(fy)
                self.canvas.create_oval(fx - 3, fy - 3, fx + 3, fy + 3, fill=color, outline="white")
            # 移動モードの選択ボックスにはリサイズハンドル
            if b is self.selected and self.mode_var is not None and self.mode_var.get() == "move":
                for hx, hy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                    self.canvas.create_rectangle(hx - 5, hy - 5, hx + 5, hy + 5,
                                                 fill="white", outline=color, width=2)
        self._update_status(len(boxes))

    def _update_status(self, nbox: int) -> None:
        frame = self._frame()
        groups = {b.group for b in self.boxes_by_frame[frame] if not b.deleted}
        team_counts = {group_label(g): 0 for g in sorted(groups)}
        for b in self.boxes_by_frame[frame]:
            if not b.deleted:
                team_counts[group_label(b.group)] = team_counts.get(group_label(b.group), 0) + 1
        count_text = " / ".join(f"{k}={v}" for k, v in team_counts.items())
        sel = f"  選択中: {group_label(self.selected.group)}" if self.selected else ""
        calib = f"  色判定:{self.calib_info}" if self.calib_info else ""
        self.status.config(
            text=f"フレーム {frame} (画像 frame_{frame+1:06d}.jpg)  "
                 f"{self.idx+1}/{len(self.frame_list)}  —  "
                 f"ボックス {nbox}個  {count_text}{sel}{calib}")
        warnings = self._frame_warnings(frame)
        if warnings:
            self.warning_label.config(text="警告: " + " / ".join(warnings), fg="#c62828")
            self.status.config(fg="#c62828")
        else:
            self.warning_label.config(text="人数チェックOK: 選手22人、A/B同数、未検出なし", fg="#2e7d32")
            self.status.config(fg="black")

    def _frame_warnings(self, frame: int) -> list[str]:
        boxes = [b for b in self.boxes_by_frame[frame] if not b.deleted]
        a = sum(1 for b in boxes if b.group == 1)
        bcnt = sum(1 for b in boxes if b.group == 2)
        unknown = sum(1 for b in boxes if b.group == 3)
        # ボール・その他は選手数に含めない
        players = a + bcnt + unknown
        warnings = []
        if players != 22:
            warnings.append(f"選手が22人ではない({players}人)")
        if a != bcnt:
            warnings.append(f"A/Bが同数ではない(A={a}, B={bcnt})")
        if unknown:
            warnings.append(f"未検出色が残っている({unknown}個)")
        return warnings

    # ---- チーム色操作 ----
    def _on_group_change(self) -> None:
        self.current_group = max(1, min(MAX_GROUP, int(self.group_var.get())))
        self.group_var.set(self.current_group)
        # 未検出(黄)は黒文字、それ以外は白文字
        fg = "black" if self.current_group == 3 else "white"
        self.group_swatch.config(bg=group_color(self.current_group), fg=fg,
                                 text=group_label(self.current_group))

    def _set_group(self, group: int) -> None:
        self.current_group = max(1, min(MAX_GROUP, group))
        self.group_var.set(self.current_group)
        fg = "black" if self.current_group == 3 else "white"
        self.group_swatch.config(bg=group_color(self.current_group), fg=fg,
                                 text=group_label(self.current_group))

    def _load_color_calibration(self) -> None:
        if COLOR_MODEL_JSON.exists():
            try:
                data = json.loads(COLOR_MODEL_JSON.read_text())
                if data.get("type") == "knn_color_v1":
                    self.color_model = data
                    acc = data.get("accuracy")
                    n = data.get("samples")
                    if isinstance(acc, (int, float)) and isinstance(n, int):
                        self.calib_info = f"kNN色モデル 正解率{acc:.0%} n={n}"
                    else:
                        self.calib_info = "kNN色モデル"
                    return
            except Exception:
                self.color_model = None

        if not COLOR_CALIB_JSON.exists():
            self.calib_info = f"固定しきい値 {self.auto_threshold:.3f}"
            return
        try:
            data = json.loads(COLOR_CALIB_JSON.read_text())
            self.auto_threshold = float(data.get("threshold", AUTO_GREEN_THRESH))
            self.green_high_is_a = bool(data.get("green_high_is_a", True))
            acc = data.get("accuracy")
            n = data.get("samples")
            if isinstance(acc, (int, float)) and isinstance(n, int):
                self.calib_info = f"学習済み 閾値{self.auto_threshold:.3f} 正解率{acc:.0%} n={n}"
            else:
                self.calib_info = f"学習済み 閾値{self.auto_threshold:.3f}"
        except Exception:
            self.auto_threshold = AUTO_GREEN_THRESH
            self.green_high_is_a = True
            self.calib_info = f"固定しきい値 {self.auto_threshold:.3f}"

    def _save_color_calibration(self, accuracy: float, samples: int) -> None:
        COLOR_CALIB_JSON.write_text(json.dumps({
            "threshold": self.auto_threshold,
            "green_high_is_a": self.green_high_is_a,
            "accuracy": accuracy,
            "samples": samples,
        }, ensure_ascii=False, indent=2))
        self.calib_info = f"学習済み 閾値{self.auto_threshold:.3f} 正解率{accuracy:.0%} n={samples}"

    def _save_color_model(self, features: list[list[float]], labels: list[int], accuracy: float) -> None:
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
        }
        COLOR_MODEL_JSON.write_text(json.dumps(model, ensure_ascii=False, indent=2))
        self.color_model = model
        self.calib_info = f"kNN色モデル 正解率{accuracy:.0%} n={len(labels)}"

    def _predict_group_from_ratio(self, ratio: float) -> int:
        if self.green_high_is_a:
            return 1 if ratio >= self.auto_threshold else 2
        return 1 if ratio < self.auto_threshold else 2

    def _predict_group_from_feature(self, feature: list[float]) -> int | None:
        score = self._score_group_a_from_feature(feature)
        if score is None:
            return None
        return 1 if score >= 0 else 2

    def _score_group_a_from_feature(self, feature: list[float]) -> float | None:
        model = self.color_model
        if not model:
            return None
        features = np.asarray(model.get("features", []), dtype=np.float32)
        labels = np.asarray(model.get("labels", []), dtype=np.int32)
        if features.ndim != 2 or len(features) == 0 or len(features) != len(labels):
            return None
        x = np.asarray(feature, dtype=np.float32)
        mean = np.asarray(model.get("mean", np.zeros(features.shape[1])), dtype=np.float32)
        std = np.asarray(model.get("std", np.ones(features.shape[1])), dtype=np.float32)
        if x.shape[0] != features.shape[1]:
            return None
        z_train = (features - mean) / std
        z = (x - mean) / std
        dist = np.linalg.norm(z_train - z, axis=1)
        k = max(1, min(int(model.get("k", 7)), len(labels)))
        idx = np.argsort(dist)[:k]
        votes = {1: 0.0, 2: 0.0}
        for i in idx:
            votes[int(labels[i])] += 1.0 / (float(dist[i]) + 1e-6)
        return votes[1] - votes[2]

    def _score_group_a(self, box: Box) -> float | None:
        feature = self._box_color_feature(box)
        if feature is not None:
            score = self._score_group_a_from_feature(feature)
            if score is not None:
                return score
        ratio = self._box_green_ratio(box)
        if ratio is None:
            return None
        return (ratio - self.auto_threshold) if self.green_high_is_a else (self.auto_threshold - ratio)

    def _swap_box_color(self, box: Box) -> None:
        if box.group == 1:
            box.group = 2
        elif box.group == 2:
            box.group = 1

    def _swap_current_frame_colors(self) -> None:
        changed = 0
        for b in self.boxes_by_frame[self._frame()]:
            if b.deleted:
                continue
            before = b.group
            self._swap_box_color(b)
            if b.group != before:
                changed += 1
        self._redraw()
        self.warning_label.config(text=f"このフレームの緑/黒を入れ替えました（{changed}個）", fg="#007aff")

    def _swap_all_colors(self) -> None:
        if not messagebox.askyesno("全フレーム入替", "全フレームの緑(A)と黒(B)を入れ替えますか？"):
            return
        changed = 0
        for bs in self.boxes_by_frame.values():
            for b in bs:
                if b.deleted:
                    continue
                before = b.group
                self._swap_box_color(b)
                if b.group != before:
                    changed += 1
        self._redraw()
        messagebox.showinfo("全フレーム入替", f"緑/黒を入れ替えました（{changed}個）。保存すると反映されます。")

    # ---- 自動色判定 ----
    def _box_torso_crop(self, box: Box, image_cache: dict[int, Image.Image] | None = None) -> Image.Image | None:
        x1 = max(0, int(min(box.x1, box.x2)))
        x2 = int(max(box.x1, box.x2))
        y1 = max(0, int(min(box.y1, box.y2)))
        y2 = int(max(box.y1, box.y2))
        if image_cache is not None:
            pil = image_cache.get(box.frame)
            if pil is None:
                img_path = FRAMES_DIR / f"frame_{box.frame + 1:06d}.jpg"
                if not img_path.exists():
                    return None
                pil = Image.open(img_path)
                image_cache[box.frame] = pil
        elif hasattr(self, "_pil") and box.frame == self._frame():
            pil = self._pil
        else:
            img_path = FRAMES_DIR / f"frame_{box.frame + 1:06d}.jpg"
            if not img_path.exists():
                return None
            pil = Image.open(img_path)
        width, height = pil.size
        x2 = min(width, x2)
        y2 = min(height, y2)
        h = y2 - y1
        if h < 8 or x2 - x1 < 4:
            return None
        ty1 = y1 + int(h * TORSO_TOP)
        ty2 = y1 + int(h * TORSO_BOTTOM)
        return pil.crop((x1, ty1, x2, ty2)).convert("RGB")

    def _box_green_ratio(self, box: Box, image_cache: dict[int, Image.Image] | None = None) -> float | None:
        crop = self._box_torso_crop(box, image_cache)
        if crop is None:
            return None
        if hasattr(crop, "get_flattened_data"):
            pixels = list(crop.get_flattened_data())
        else:
            pixels = list(crop.getdata())
        return green_ratio_from_rgb_pixels(pixels)

    def _box_color_feature(self, box: Box, image_cache: dict[int, Image.Image] | None = None) -> list[float] | None:
        crop = self._box_torso_crop(box, image_cache)
        if crop is None:
            return None
        return color_feature_from_crop(crop)

    def _auto_color_box(self, box: Box) -> bool:
        feature = self._box_color_feature(box)
        if feature is not None:
            predicted = self._predict_group_from_feature(feature)
            if predicted in (1, 2):
                box.group = predicted
                return True
        ratio = self._box_green_ratio(box)
        if ratio is None:
            box.group = 3
            return False
        box.group = self._predict_group_from_ratio(ratio)
        return True

    def _manual_training_samples(
        self,
        frame_lo: int | None = None,
        frame_hi: int | None = None,
    ) -> list[tuple[list[float], float, int]]:
        samples = []
        image_cache: dict[int, Image.Image] = {}
        for bs in self.boxes_by_frame.values():
            for b in bs:
                if frame_lo is not None and b.frame < frame_lo:
                    continue
                if frame_hi is not None and b.frame > frame_hi:
                    continue
                if b.deleted or b.group not in (1, 2):
                    continue
                # 色分類の教師には、色ラベルを手で触った可能性が高いものだけを使う。
                # 位置だけ動かしたボックスは、色ラベルが未確認のまま混ざりやすい。
                manually_touched = b.added or b.group != b.init_group
                if not manually_touched:
                    continue
                feature = self._box_color_feature(b, image_cache)
                ratio = self._box_green_ratio(b, image_cache)
                if feature is not None and ratio is not None:
                    samples.append((feature, ratio, b.group))
        return samples

    def _learn_color_from_annotations(self) -> None:
        self._learn_color_from_samples(self._manual_training_samples(), "全注釈")

    def _learn_color_from_visible_range(self) -> None:
        if not self.frame_list:
            return
        frame_lo, frame_hi = min(self.frame_list), max(self.frame_list)
        self._learn_color_from_samples(
            self._manual_training_samples(frame_lo, frame_hi),
            f"表示範囲 F{frame_lo}〜F{frame_hi}",
        )

    def _learn_color_from_samples(self, samples: list[tuple[list[float], float, int]], label: str) -> None:
        raw_count = len(samples)
        samples = self._balanced_training_subset(samples, MAX_TRAIN_SAMPLES_PER_TEAM)
        labels = {g for _, _, g in samples}
        if len(samples) < 12 or labels != {1, 2}:
            messagebox.showinfo(
                "教師データ不足",
                f"{label} に、緑(A)と黒(B)の両方を含む手修正ボックスが少ないです。\n"
                "目安として、両チーム合わせて20個以上を保存してから学習してください。")
            return

        features = [feature for feature, _, _ in samples]
        groups = [group for _, _, group in samples]
        knn_accuracy = self._loo_knn_accuracy(features, groups)
        self._save_color_model(features, groups, knn_accuracy)

        ratios = sorted(set(ratio for _, ratio, _ in samples))
        if len(ratios) == 1:
            self._redraw()
            messagebox.showinfo(
                "色判定を学習しました",
                f"範囲: {label}\n"
                f"教師ボックス: {len(samples)}個（候補 {raw_count}個から抽出）\n"
                f"kNNモデルの簡易正解率: {knn_accuracy:.1%}\n\n"
                "緑比率の補助しきい値は更新できませんでしたが、以後はkNNモデルを優先します。")
            return
        candidates = [(ratios[i] + ratios[i + 1]) / 2 for i in range(len(ratios) - 1)]
        candidates = [min(ratios) - 1e-6] + candidates + [max(ratios) + 1e-6]

        best = None
        for direction in (True, False):
            for threshold in candidates:
                ok = 0
                for _, ratio, group in samples:
                    pred = 1 if (ratio >= threshold if direction else ratio < threshold) else 2
                    if pred == group:
                        ok += 1
                accuracy = ok / len(samples)
                if best is None or accuracy > best[0]:
                    best = (accuracy, threshold, direction)

        assert best is not None
        accuracy, threshold, direction = best
        self.auto_threshold = threshold
        self.green_high_is_a = direction
        self._save_color_calibration(accuracy, len(samples))
        self._save_color_model(features, groups, knn_accuracy)
        self._redraw()
        messagebox.showinfo(
            "色判定を学習しました",
            f"範囲: {label}\n"
            f"教師ボックス: {len(samples)}個（候補 {raw_count}個から抽出）\n"
            f"kNNモデルの簡易正解率: {knn_accuracy:.1%}\n"
            f"補助しきい値: {threshold:.4f} / 正解率 {accuracy:.1%}\n\n"
            "以後の自動色判定はkNNモデルを優先します。")

    def _balanced_training_subset(
        self,
        samples: list[tuple[list[float], float, int]],
        max_per_team: int,
    ) -> list[tuple[list[float], float, int]]:
        result = []
        for group in (1, 2):
            group_samples = [s for s in samples if s[2] == group]
            if len(group_samples) <= max_per_team:
                result.extend(group_samples)
                continue
            step = (len(group_samples) - 1) / (max_per_team - 1)
            result.extend(group_samples[round(i * step)] for i in range(max_per_team))
        return result

    def _loo_knn_accuracy(self, features: list[list[float]], labels: list[int]) -> float:
        data = np.asarray(features, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int32)
        if len(y) < 3:
            return 0.0
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std < 1e-6] = 1.0
        z = (data - mean) / std
        correct = 0
        for i in range(len(y)):
            dist = np.linalg.norm(z - z[i], axis=1)
            dist[i] = np.inf
            k = max(1, min(7, len(y) - 1))
            idx = np.argsort(dist)[:k]
            votes = {1: 0.0, 2: 0.0}
            for j in idx:
                votes[int(y[j])] += 1.0 / (float(dist[j]) + 1e-6)
            pred = 1 if votes[1] >= votes[2] else 2
            if pred == int(y[i]):
                correct += 1
        return correct / len(y)

    def _auto_color_current_frame(self) -> None:
        changed = 0
        for b in self.boxes_by_frame[self._frame()]:
            if b.deleted:
                continue
            before = b.group
            if self._auto_color_box(b) and b.group != before:
                changed += 1
        self._redraw()
        messagebox.showinfo("自動色判定", f"このフレームの色を自動判定しました（変更 {changed} 個）。")

    def _balance_frame_colors(self, frame: int) -> bool:
        boxes = [b for b in self.boxes_by_frame.get(frame, []) if not b.deleted]
        if len(boxes) != 22:
            return False
        scored = []
        for b in boxes:
            score = self._score_group_a(b)
            if score is None:
                return False
            scored.append((score, b))
        scored.sort(key=lambda item: item[0], reverse=True)
        top_a = {id(b) for _, b in scored[:11]}
        for _, b in scored:
            b.group = 1 if id(b) in top_a else 2
        return True

    def _balance_current_frame_colors(self) -> None:
        if self._balance_frame_colors(self._frame()):
            self._redraw()
            self.warning_label.config(text="このフレームを11/11に補正しました", fg="#007aff")
        else:
            messagebox.showinfo("11/11補正", "このフレームは22個ボックスではないか、色スコアを出せないボックスがあります。")

    def _regenerate_current_frame_boxes(self) -> None:
        frame = self._frame()
        if not messagebox.askyesno(
            "このフレームを再生成",
            "このフレームの手追加・削除・移動・色変更をリセットして、元検出ボックスに戻しますか？"):
            return
        reset_count = 0
        kept = []
        for b in self.boxes_by_frame[frame]:
            if b.added:
                continue
            b.x1, b.y1, b.x2, b.y2 = b.init_geom
            b.group = b.init_group
            b.deleted = False
            kept.append(b)
            reset_count += 1
        self.boxes_by_frame[frame] = kept
        self.selected = None
        self._redraw()
        self.warning_label.config(
            text=f"このフレームのボックスを元検出から再生成しました（{reset_count}個）",
            fg="#007aff")

    def _load_yolo_model(self):
        """YOLOモデルを遅延ロードしてキャッシュする。"""
        if getattr(self, "_yolo_model", None) is not None:
            return self._yolo_model
        if not YOLO_MODEL_PATH.exists():
            messagebox.showerror(
                "モデルが無い",
                f"{YOLO_MODEL_PATH} が見つかりません。\n"
                "process_segment.py 等で使う yolo11m.pt を models/ フォルダに置いてください。")
            return None
        try:
            from ultralytics import YOLO
        except Exception as e:
            messagebox.showerror("ultralytics 未導入",
                                 f"ultralytics を import できませんでした:\n{e}")
            return None
        self._yolo_model = YOLO(str(YOLO_MODEL_PATH))
        return self._yolo_model

    def _redetect_current_frame_yolo(self) -> None:
        """現フレーム画像をYOLO(COCO person)で検出し直し、ボックスを置き換える。"""
        frame = self._frame()
        img_path = FRAMES_DIR / f"frame_{frame + 1:06d}.jpg"
        if not img_path.exists():
            messagebox.showerror("画像が無い", f"{img_path} が見つかりません。")
            return
        if not messagebox.askyesno(
            "このFをYOLO再検出",
            "このフレームのボックスをすべて破棄して、YOLO(COCO person)で検出し直しますか？\n"
            "（元検出は削除扱い・新規検出は追加ボックスとして保存されます）"):
            return

        model = self._load_yolo_model()
        if model is None:
            return

        self.warning_label.config(text="YOLOで検出中…", fg="#007aff")
        self.root.update_idletasks()
        try:
            results = model.predict(
                source=str(img_path), classes=[0],
                conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False)
        except Exception as e:
            messagebox.showerror("YOLO検出エラー", str(e))
            self.warning_label.config(text="YOLO検出に失敗しました", fg="#c62828")
            return

        dets = []
        for res in results:
            if res.boxes is None:
                continue
            for xyxy in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    dets.append((x1, y1, x2, y2))

        # 既存ボックスを破棄: 元検出は削除扱い、手追加は完全に消す
        kept = []
        for b in self.boxes_by_frame[frame]:
            if b.added:
                continue  # 追加ボックスは捨てる
            b.deleted = True
            kept.append(b)  # 元検出は削除マークだけ残す（復元可能性のため）
        # 新規検出を追加ボックスとして登録
        for x1, y1, x2, y2 in dets:
            key = self._added_seq
            self._added_seq -= 1
            b = Box(frame, key, x1, y1, x2, y2, -1, self.current_group, added=True)
            if self.auto_color_var is not None and self.auto_color_var.get():
                self._auto_color_box(b)
            kept.append(b)
        self.boxes_by_frame[frame] = kept
        self.selected = None
        self._redraw()
        self.warning_label.config(
            text=f"YOLOで再検出しました（{len(dets)}個）。保存で box_corrections.csv に反映されます。",
            fg="#007aff")

    def _auto_color_all_unknown(self) -> None:
        changed = 0
        for bs in self.boxes_by_frame.values():
            for b in bs:
                if b.deleted or b.group != 3:
                    continue
                before = b.group
                if self._auto_color_box(b) and b.group != before:
                    changed += 1
        self._redraw()
        messagebox.showinfo("自動色判定", f"全フレームの不明色を自動判定しました（変更 {changed} 個）。")

    def _balance_all_frame_colors(self) -> None:
        fixed = 0
        skipped = 0
        for fr in self.frame_list:
            if self._balance_frame_colors(fr):
                fixed += 1
            else:
                skipped += 1
        self._redraw()
        messagebox.showinfo("11/11補正", f"表示範囲の {fixed} フレームを11/11に補正しました。\nスキップ: {skipped} フレーム")

    def _overlap_ratio(self, a: Box, b: Box) -> float:
        """2つのボックスの重複率（小さい方のボックス面積に対する交差面積の割合）。"""
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        if area_a <= 0 or area_b <= 0:
            return 0.0
        return inter / min(area_a, area_b)

    def _redraw_dup_mode(self) -> None:
        """重複確認モード専用の描画。同じグループ（重なりあう塊）を同色で表示。"""
        self.canvas.delete("all")
        if self.tk_img is not None:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        if self.show_pitch.get() and self.pitch_quad and len(self.pitch_quad) == 4:
            pts = [(self._sx(x), self._sy(y)) for x, y in self.pitch_quad]
            flat = [c for p in pts for c in p]
            self.canvas.create_polygon(*flat, outline="#00e5ff", fill="", width=2, dash=(6, 3))

        threshold = self.dup_threshold_var.get() / 100.0
        boxes = [b for b in self.boxes_by_frame[self._frame()] if not b.deleted]
        n = len(boxes)

        # ペアごとの重複チェック
        overlapping_with: list[list[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if self._overlap_ratio(boxes[i], boxes[j]) >= threshold:
                    overlapping_with[i].append(j)
                    overlapping_with[j].append(i)

        # Union-Find でグループ化
        parent = list(range(n))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for i in range(n):
            for j in overlapping_with[i]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

        # 重複ありグループのルートに色を割り当て
        group_color_idx: dict[int, int] = {}
        color_counter = 0
        for i in range(n):
            if overlapping_with[i]:
                root = find(i)
                if root not in group_color_idx:
                    group_color_idx[root] = color_counter
                    color_counter += 1

        # 重複領域をハッチング（stipple）で描画
        drawn_pairs: set[tuple[int, int]] = set()
        for i in range(n):
            for j in overlapping_with[i]:
                pair = (min(i, j), max(i, j))
                if pair in drawn_pairs:
                    continue
                drawn_pairs.add(pair)
                a, b = boxes[i], boxes[j]
                root = find(i)
                color = DUP_COLORS[group_color_idx[root] % len(DUP_COLORS)]
                ix1 = self._sx(max(a.x1, b.x1))
                iy1 = self._sy(max(a.y1, b.y1))
                ix2 = self._sx(min(a.x2, b.x2))
                iy2 = self._sy(min(a.y2, b.y2))
                self.canvas.create_rectangle(ix1, iy1, ix2, iy2,
                                             fill=color, stipple="gray50", outline="")

        # ボックスを描画
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = self._sx(b.x1), self._sy(b.y1), self._sx(b.x2), self._sy(b.y2)
            if overlapping_with[i]:
                root = find(i)
                color = DUP_COLORS[group_color_idx[root] % len(DUP_COLORS)]
                width = 3
                count = len(overlapping_with[i]) + 1
                label = f"×{count}"
            else:
                color = "#444444"
                width = 1
                label = ""
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
            if label:
                self.canvas.create_text(x1 + 2, y1 - 7, text=label, fill="black", anchor=tk.W,
                                        font=("Helvetica", 11, "bold"))
                self.canvas.create_text(x1 + 1, y1 - 8, text=label, fill=color, anchor=tk.W,
                                        font=("Helvetica", 11, "bold"))

        n_dup = sum(1 for o in overlapping_with if o)
        n_groups = len(group_color_idx)
        self._update_status(len(boxes))
        if n_dup:
            self.warning_label.config(
                text=f"重複: {n_dup}個のボックスが{n_groups}グループで重なっています"
                     f"（しきい値: {self.dup_threshold_var.get()}%）",
                fg="#c62828")
        else:
            self.warning_label.config(
                text=f"重複なし（しきい値: {self.dup_threshold_var.get()}%）",
                fg="#2e7d32")

    def _jump_warning(self, direction: int) -> None:
        if not self.frame_list:
            return
        start = self.idx
        for step in range(1, len(self.frame_list) + 1):
            i = (start + direction * step) % len(self.frame_list)
            if self._frame_warnings(self.frame_list[i]):
                self.idx = i
                self.selected = None
                self.slider.set(i)
                self._show()
                return
        messagebox.showinfo("警告フレーム", "人数・チーム数の警告があるフレームは見つかりませんでした。")

    # ---- マウス ----
    def _find_box(self, cx, cy) -> Box | None:
        x = cx / self.scale
        y = cy / self.scale
        hit = None
        best_area = None
        for b in self.boxes_by_frame[self._frame()]:
            if b.deleted:
                continue
            if b.x1 <= x <= b.x2 and b.y1 <= y <= b.y2:
                area = (b.x2 - b.x1) * (b.y2 - b.y1)
                if best_area is None or area < best_area:  # 小さい方を優先
                    hit, best_area = b, area
        return hit

    def _on_press(self, e) -> None:
        mode = self.mode_var.get()
        if mode == "dup":
            return
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        ix, iy = cx / self.scale, cy / self.scale

        if mode == "add":
            # 新規ボックスを作り、ドラッグで広げる
            key = self._added_seq
            self._added_seq -= 1
            b = Box(self._frame(), key, ix, iy, ix, iy, -1, self.current_group, added=True)
            self.boxes_by_frame[self._frame()].append(b)
            self.selected = b
            self._drag = ("create",)
            self._redraw()
            return

        box = self._find_box(cx, cy)

        if mode == "group":
            if box is not None:
                box.group = self.current_group
                self.selected = box
            self._redraw()
            return

        if mode == "delete":
            if box is not None:
                if box.added:
                    self.boxes_by_frame[box.frame].remove(box)
                else:
                    box.deleted = True
                self.selected = None
            self._redraw()
            return

        if mode == "move":
            self.selected = box
            if box is not None:
                corner = box.corner_near(ix, iy, 12 / self.scale)
                if corner:
                    self._drag = ("resize", corner)
                else:
                    self._drag = ("move", ix - box.x1, iy - box.y1)
            else:
                self._drag = None
            self._redraw()

    def _on_motion(self, e) -> None:
        if not self._drag or self.selected is None:
            return
        ix = max(0, min(self.img_w, self.canvas.canvasx(e.x) / self.scale))
        iy = max(0, min(self.img_h, self.canvas.canvasy(e.y) / self.scale))
        b = self.selected
        kind = self._drag[0]
        if kind == "create":
            b.x2, b.y2 = ix, iy
        elif kind == "move":
            _, dx, dy = self._drag
            w, h = b.x2 - b.x1, b.y2 - b.y1
            b.x1, b.y1 = ix - dx, iy - dy
            b.x2, b.y2 = b.x1 + w, b.y1 + h
        elif kind == "resize":
            corner = self._drag[1]
            if "n" in corner:
                b.y1 = iy
            if "s" in corner:
                b.y2 = iy
            if "w" in corner:
                b.x1 = ix
            if "e" in corner:
                b.x2 = ix
        self._redraw()

    def _on_release(self, e) -> None:
        if self._drag and self.selected is not None:
            b = self.selected
            b.norm()
            # 極端に小さい新規ボックスは破棄
            if b.added and (b.x2 - b.x1 < 3 or b.y2 - b.y1 < 3):
                if b in self.boxes_by_frame[b.frame]:
                    self.boxes_by_frame[b.frame].remove(b)
                self.selected = None
            elif self.auto_color_var is not None and self.auto_color_var.get():
                self._auto_color_box(b)
        self._drag = None
        self._redraw()

    def _delete_selected(self) -> None:
        if self.selected is None:
            return
        if self.selected.added:
            self.boxes_by_frame[self.selected.frame].remove(self.selected)
        else:
            self.selected.deleted = True
        self.selected = None
        self._redraw()

    # ---- ナビ ----
    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self.selected = None
            self.slider.set(self.idx)
            self._show()

    def _next(self):
        if self.idx < len(self.frame_list) - 1:
            self.idx += 1
            self.selected = None
            self.slider.set(self.idx)
            self._show()

    def _on_slider(self, v):
        i = int(v)
        if i != self.idx:
            self.idx = i
            self.selected = None
            self._show()

    def _set_start_here(self) -> None:
        """現在表示中の画像を開始点に設定。"""
        img_no = self.frame_list[self.idx] + 1  # frame番号→画像番号
        self.trim_start_var.set(img_no)
        self.trim_start = img_no
        self._apply_trim()

    def _set_end_here(self) -> None:
        """現在表示中の画像を終了点に設定。"""
        img_no = self.frame_list[self.idx] + 1
        self.trim_end_var.set(img_no)
        self.trim_end = img_no
        self._apply_trim()

    def _set_trim_from_fields(self) -> None:
        """入力欄の値をトリムに適用。"""
        try:
            s = int(self.trim_start_var.get())
            e = int(self.trim_end_var.get())
        except (ValueError, tk.TclError):
            return
        if s > e:
            s, e = e, s
        self.trim_start, self.trim_end = s, e
        self._apply_trim()

    def _reset_trim(self) -> None:
        """全範囲を表示に戻す。"""
        self.trim_start, self.trim_end = 1, 999999
        self.trim_start_var.set(1)
        self.trim_end_var.set(999999)
        self._apply_trim()

    def _jump_to_image(self) -> None:
        """画像番号（frame_XXXXXX.jpg の XXXXXX）へ移動。検出のある最も近いフレームへ。"""
        try:
            target_frame = int(self.jump_var.get()) - 1  # 画像番号→frame番号
        except (ValueError, tk.TclError):
            return
        i = next((k for k, f in enumerate(self.frame_list) if f >= target_frame), None)
        if i is None:
            i = len(self.frame_list) - 1
        self.idx = i
        self.selected = None
        self.slider.set(i)
        self._show()

    # ---- ズーム / パン ----
    def _zoom_by(self, factor: float) -> None:
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        self._render()

    def _zoom_fit(self) -> None:
        self.zoom = 1.0
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._render()

    def _on_wheel(self, e):
        self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")

    def _on_wheel_h(self, e):
        self.canvas.xview_scroll(-1 if e.delta > 0 else 1, "units")

    def _on_wheel_zoom(self, e):
        self._zoom_by(ZOOM_STEP if e.delta > 0 else 1 / ZOOM_STEP)

    # ---- 保存 ----
    def _save(self) -> None:
        cs = self.coord_scale
        def sx(v):  # 内部ネイティブpx → 1920基準で保存
            return round(v / cs, 1)
        rows = []
        for frame, bs in self.boxes_by_frame.items():
            for b in bs:
                if b.added:
                    if b.deleted:
                        continue  # 追加して消したものは記録しない
                    rows.append({"frame": frame, "box_key": b.key, "action": "add",
                                 "x1": sx(b.x1), "y1": sx(b.y1),
                                 "x2": sx(b.x2), "y2": sx(b.y2),
                                 "orig_player_id": b.orig_pid, "group": b.group})
                elif b.deleted:
                    rows.append({"frame": frame, "box_key": b.key, "action": "delete",
                                 "x1": "", "y1": "", "x2": "", "y2": "",
                                 "orig_player_id": b.orig_pid, "group": b.group})
                else:
                    if b.group != b.init_group:
                        rows.append({"frame": frame, "box_key": b.key, "action": "reassign",
                                     "x1": "", "y1": "", "x2": "", "y2": "",
                                     "orig_player_id": b.orig_pid, "group": b.group})
                    if b.moved():
                        rows.append({"frame": frame, "box_key": b.key, "action": "move",
                                     "x1": sx(b.x1), "y1": sx(b.y1),
                                     "x2": sx(b.x2), "y2": sx(b.y2),
                                     "orig_player_id": b.orig_pid, "group": b.group})
        fields = ["frame", "box_key", "action", "x1", "y1", "x2", "y2", "orig_player_id", "group"]
        with CORRECTIONS_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        messagebox.showinfo("保存完了",
                            f"{CORRECTIONS_CSV} に {len(rows)} 件の編集を保存しました。\n\n"
                            "元データは変更していません。次回起動時に復元されます。")


def main() -> None:
    root = tk.Tk()
    Annotator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
