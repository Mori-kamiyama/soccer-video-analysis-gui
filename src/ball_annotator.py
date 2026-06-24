"""
ボール・アノテーションGUI

フレーム画像を表示し、ボールの位置をクリックで打つ。
ボールは速くて毎フレーム打つのは大変なので「キーフレーム方式」:
  - 数フレームおきにボール位置をクリック（キーフレーム）
  - 間のフレームは自動で線形補間（中割り）して表示・保存
  - 見えない/枠外のフレームは「ボールなし」を打つと、そこで補間が途切れる

非破壊。保存先:
  outputs/ball_keyframes.csv   … 打ったキーフレームだけ（編集の元データ）
  outputs/ball_positions.csv   … 全フレームに展開した位置（補間込み・可視化用）

座標系: クリック位置は内部・保存とも 1920x1080 基準に正規化する
（pitch_points.json・選手座標と統一）。表示フレームが 2560 等の高解像度でも、
ball_keyframes.csv / ball_positions.csv の image_x,image_y は常に1920系。

起動:
  ./run_ball_gui.sh
  ./run_ball_gui.sh --start 212 --end 1311
  ./run_ball_gui.sh --frames-dir frames_ball   # 高解像度フレームを使う
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

FRAMES_DIR = Path("frames")
POINTS_PATH = Path("pitch_points.json")
KEYFRAMES_CSV = Path("outputs/ball_keyframes.csv")
POSITIONS_CSV = Path("outputs/ball_positions.csv")

BASE_W = 1920.0  # 保存座標の基準幅（pitch_points.json・選手座標と統一）

MAX_W, MAX_H = 1500, 820
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 1.0, 12.0, 1.3
BIG_STEP = 10  # Shift+矢印でまとめて進む/戻るフレーム数
BALL_COLOR = "#ffd400"
KEY_COLOR = "#ff7a00"
NONE_COLOR = "#888888"


class BallAnnotator:
    def __init__(self, root: tk.Tk, start: int, end: int, frames_dir: str = "frames"):
        self.root = root
        root.title("ボール・アノテーション")

        self.frames_dir = Path(frames_dir)
        self.meta_path = self.frames_dir / "frames_metadata.json"
        self.frame_files = sorted(self.frames_dir.glob("frame_*.jpg"))
        if not self.frame_files:
            raise SystemExit(f"{self.frames_dir}/ に frame_*.jpg が見つかりません")
        # 画像番号→frame index（frame_XXXXXX.jpg の XXXXXX-1）
        self.n_total = len(self.frame_files)
        self.start_img = max(1, start)
        self.end_img = min(self.n_total, end) if end > 0 else self.n_total
        self.frames = [n - 1 for n in range(self.start_img, self.end_img + 1)
                       if (self.frames_dir / f"frame_{n:06d}.jpg").exists()]
        if not self.frames:
            raise SystemExit("指定範囲に画像がありません")

        self.eff_fps = self._load_fps()

        # キーフレーム: frame -> ('pos', x, y)  or ('none',)
        self.keys: dict[int, tuple] = {}
        self._load_keyframes()

        self.H = self._load_homography()

        self.pos = 0  # frames リスト内の位置
        self.scale = self.base_scale = 1.0
        self.zoom = 1.0
        self.img_w = self.img_h = 0
        self.to_disp = 1.0  # 1920系 → 表示画像px の倍率（img_w/BASE_W）
        self.tk_img = None
        self._pil = None
        self._playing = False

        self._init_button_styles()
        self._build_ui()
        self._show()

    # ---- 起動補助 ----
    def _load_fps(self) -> float:
        try:
            md = json.loads(self.meta_path.read_text())
            if len(md) >= 2:
                dt = md[1]["timestamp"] - md[0]["timestamp"]
                if dt > 0:
                    return 1.0 / dt
        except Exception:
            pass
        return 2.0

    def _load_homography(self):
        if not POINTS_PATH.exists():
            return None
        try:
            import cv2
            import numpy as np
            d = json.loads(POINTS_PATH.read_text())
            img = np.array(d["image"], dtype="float32")
            pitch = np.array(d["pitch"], dtype="float32")
            return cv2.getPerspectiveTransform(img, pitch)
        except Exception:
            return None

    def _to_pitch(self, x, y):
        if self.H is None:
            return None, None
        import cv2
        import numpy as np
        pt = np.array([[[x, y]]], dtype="float32")
        out = cv2.perspectiveTransform(pt, self.H)[0][0]
        return round(float(out[0]), 2), round(float(out[1]), 2)

    # ---- 色付きボタン（macOS対応） ----
    def _init_button_styles(self):
        self._btn_style = ttk.Style()
        try:
            self._btn_style.theme_use("clam")
        except tk.TclError:
            pass
        self._btn_cache = set()

    def _cbtn(self, parent, text, command, bg, fg="white", bold=False):
        font = ("Helvetica", 11, "bold") if bold else ("Helvetica", 11)
        name = f"b{abs(hash((bg, fg, bold))) % 1_000_000}.TButton"
        if name not in self._btn_cache:
            h = bg.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            dark = f"#{int(r*.85):02x}{int(g*.85):02x}{int(b*.85):02x}"
            self._btn_style.configure(name, background=bg, foreground=fg, font=font)
            self._btn_style.map(name, background=[("active", dark), ("pressed", dark)])
            self._btn_cache.add(name)
        return ttk.Button(parent, text=text, command=command, style=name)

    # ---- データ ----
    def _load_keyframes(self):
        if not KEYFRAMES_CSV.exists():
            return
        try:
            with KEYFRAMES_CSV.open() as f:
                for r in csv.DictReader(f):
                    fr = int(r["frame"])
                    if r["state"] == "none":
                        self.keys[fr] = ("none",)
                    else:
                        self.keys[fr] = ("pos", float(r["image_x"]), float(r["image_y"]))
        except Exception:
            pass

    def _save(self):
        KEYFRAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
        # キーフレーム
        with KEYFRAMES_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "state", "image_x", "image_y"])
            for fr in sorted(self.keys):
                k = self.keys[fr]
                if k[0] == "none":
                    w.writerow([fr, "none", "", ""])
                else:
                    w.writerow([fr, "pos", round(k[1], 1), round(k[2], 1)])
        # 全フレーム展開（補間込み）
        rows = []
        for fr in self.frames:
            res = self._ball_at(fr)
            if res is None:
                continue
            x, y, kind = res
            px, py = self._to_pitch(x, y)
            rows.append([fr, round(x, 1), round(y, 1),
                         "" if px is None else px, "" if py is None else py,
                         1 if kind == "key" else 0])
        with POSITIONS_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "image_x", "image_y", "pitch_x", "pitch_y", "is_key"])
            w.writerows(rows)
        messagebox.showinfo("保存",
                            f"キーフレーム {len(self.keys)}個 を保存しました。\n"
                            f"・{KEYFRAMES_CSV}\n・{POSITIONS_CSV}（補間込み {len(rows)}フレーム）")

    # ---- ボール位置の解決（補間） ----
    def _sorted_keys(self):
        return sorted(self.keys)

    def _ball_at(self, fr):
        """(x, y, kind)  kind in ('key','interp')。無ければ None。"""
        k = self.keys.get(fr)
        if k is not None:
            if k[0] == "none":
                return None
            return k[1], k[2], "key"
        # 前後の 'pos' キーフレームで補間（間に 'none' があれば不可）
        ks = self._sorted_keys()
        prev = next((f for f in reversed(ks) if f < fr), None)
        nxt = next((f for f in ks if f > fr), None)
        if prev is None or nxt is None:
            return None
        kp, kn = self.keys[prev], self.keys[nxt]
        if kp[0] != "pos" or kn[0] != "pos":
            return None
        s = (fr - prev) / (nxt - prev)
        x = kp[1] + (kn[1] - kp[1]) * s
        y = kp[2] + (kn[2] - kp[2]) * s
        return x, y, "interp"

    # ---- UI ----
    def _build_ui(self):
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)
        tk.Button(bar, text="◀", command=self._prev, width=3).pack(side=tk.LEFT)
        tk.Button(bar, text="▶", command=self._next, width=3).pack(side=tk.LEFT, padx=(2, 8))
        self.play_btn = self._cbtn(bar, "再生", self._toggle_play, "#34c759")
        self.play_btn.pack(side=tk.LEFT)
        tk.Label(bar, text="  画像へ:").pack(side=tk.LEFT)
        self.jump_var = tk.IntVar(value=self.start_img)
        je = tk.Spinbox(bar, from_=1, to=self.n_total, width=7, textvariable=self.jump_var)
        je.pack(side=tk.LEFT)
        je.bind("<Return>", lambda e: self._jump())
        tk.Button(bar, text="移動", command=self._jump).pack(side=tk.LEFT, padx=(2, 12))
        tk.Button(bar, text="－", command=lambda: self._zoom_by(1/ZOOM_STEP)).pack(side=tk.LEFT)
        self.zoom_label = tk.Label(bar, text="100%", width=5)
        self.zoom_label.pack(side=tk.LEFT)
        tk.Button(bar, text="＋", command=lambda: self._zoom_by(ZOOM_STEP)).pack(side=tk.LEFT)
        tk.Button(bar, text="全体", command=self._zoom_fit).pack(side=tk.LEFT, padx=(2, 0))
        self._cbtn(bar, "保存", self._save, "#34c759", bold=True).pack(side=tk.RIGHT)

        # 操作バー
        ops = tk.Frame(self.root)
        ops.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(ops, text="操作:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self._cbtn(ops, "◀ 前のキー", lambda: self._jump_key(-1), "#007aff").pack(side=tk.LEFT, padx=2)
        self._cbtn(ops, "次のキー ▶", lambda: self._jump_key(1), "#007aff").pack(side=tk.LEFT, padx=2)
        ttk.Separator(ops, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        self._cbtn(ops, "ボールなし(n)", self._set_none, "#8e8e93").pack(side=tk.LEFT, padx=2)
        self._cbtn(ops, "このフレーム消去(d)", self._clear_frame, "#a33").pack(side=tk.LEFT, padx=2)
        tk.Label(ops, text="  ← 画像をクリックでボール位置を打つ（キーフレーム）",
                 fg="#555").pack(side=tk.LEFT, padx=8)

        self.slider = tk.Scale(self.root, from_=0, to=len(self.frames) - 1,
                               orient=tk.HORIZONTAL, showvalue=False, command=self._on_slider)
        self.slider.pack(side=tk.TOP, fill=tk.X, padx=8)

        self.status = tk.Label(self.root, text="", anchor="w", font=("Helvetica", 12))
        self.status.pack(side=tk.TOP, fill=tk.X, padx=8)

        cwrap = tk.Frame(self.root)
        cwrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.canvas = tk.Canvas(cwrap, bg="black", highlightthickness=0, cursor="tcross",
                                width=MAX_W, height=MAX_H)
        hsb = tk.Scrollbar(cwrap, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vsb = tk.Scrollbar(cwrap, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.canvas.bind("<MouseWheel>", lambda e: self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Shift-MouseWheel>", lambda e: self.canvas.xview_scroll(-1 if e.delta > 0 else 1, "units"))
        self.canvas.bind("<Control-MouseWheel>", lambda e: self._zoom_by(ZOOM_STEP if e.delta > 0 else 1/ZOOM_STEP))
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<Shift-Left>", lambda e: self._prev(BIG_STEP))
        self.root.bind("<Shift-Right>", lambda e: self._next(BIG_STEP))
        self.root.bind("n", lambda e: self._set_none())
        self.root.bind("d", lambda e: self._clear_frame())
        self.root.bind("<space>", lambda e: self._toggle_play())

    # ---- 表示 ----
    def _cur_frame(self):
        return self.frames[self.pos]

    def _show(self):
        fr = self._cur_frame()
        path = self.frames_dir / f"frame_{fr + 1:06d}.jpg"
        if not path.exists():
            self.status.config(text=f"画像なし: {path.name}")
            return
        self._pil = Image.open(path)
        self.img_w, self.img_h = self._pil.size
        self.to_disp = self.img_w / BASE_W  # 1920系 → この表示画像px
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

    # 引数は1920系座標。表示画像px(×to_disp)を経てキャンバスpx(×scale)へ。
    def _sx(self, x): return x * self.to_disp * self.scale
    def _sy(self, y): return y * self.to_disp * self.scale

    def _redraw(self):
        self.canvas.delete("all")
        if self.tk_img:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        fr = self._cur_frame()
        k = self.keys.get(fr)
        res = self._ball_at(fr)
        if k is not None and k[0] == "none":
            note = "ボールなし（補間の区切り）"
        elif res is not None:
            x, y, kind = res
            sx, sy = self._sx(x), self._sy(y)
            col = KEY_COLOR if kind == "key" else BALL_COLOR
            r = 9
            self.canvas.create_oval(sx-r, sy-r, sx+r, sy+r, outline="black", width=4)
            self.canvas.create_oval(sx-r, sy-r, sx+r, sy+r, outline=col, width=2)
            self.canvas.create_line(sx-r-5, sy, sx+r+5, sy, fill=col)
            self.canvas.create_line(sx, sy-r-5, sx, sy+r+5, fill=col)
            note = "キーフレーム ●" if kind == "key" else "補間（中割り）○"
        else:
            note = "未設定"

        nkeys = sum(1 for v in self.keys.values() if v[0] == "pos")
        nnone = sum(1 for v in self.keys.values() if v[0] == "none")
        t = fr / self.eff_fps
        self.status.config(
            text=f"画像 frame_{fr+1:06d}.jpg  ({self.pos+1}/{len(self.frames)})  "
                 f"t={t:.1f}s  —  {note}   ［キー{nkeys} / なし{nnone}］")

    # ---- 操作 ----
    def _on_click(self, e):
        cx, cy = self.canvas.canvasx(e.x), self.canvas.canvasy(e.y)
        dx, dy = cx / self.scale, cy / self.scale  # 表示画像px
        if not (0 <= dx <= self.img_w and 0 <= dy <= self.img_h):
            return
        x, y = dx / self.to_disp, dy / self.to_disp  # 1920系へ正規化して保存
        self.keys[self._cur_frame()] = ("pos", round(x, 1), round(y, 1))
        self._redraw()

    def _set_none(self):
        self.keys[self._cur_frame()] = ("none",)
        self._redraw()

    def _clear_frame(self):
        self.keys.pop(self._cur_frame(), None)
        self._redraw()

    def _jump_key(self, direction):
        ks = self._sorted_keys()
        cur = self._cur_frame()
        if direction > 0:
            nxt = next((f for f in ks if f > cur), None)
        else:
            nxt = next((f for f in reversed(ks) if f < cur), None)
        if nxt is None:
            return
        if nxt in self.frames:
            self.pos = self.frames.index(nxt)
            self.slider.set(self.pos)
            self._show()

    # ---- ナビ ----
    def _prev(self, step=1):
        if self.pos > 0:
            self.pos = max(0, self.pos - step); self.slider.set(self.pos); self._show()

    def _next(self, step=1):
        if self.pos < len(self.frames) - 1:
            self.pos = min(len(self.frames) - 1, self.pos + step)
            self.slider.set(self.pos); self._show()

    def _on_slider(self, v):
        i = int(v)
        if i != self.pos:
            self.pos = i; self._show()

    def _jump(self):
        try:
            target = int(self.jump_var.get()) - 1
        except (ValueError, tk.TclError):
            return
        i = next((k for k, f in enumerate(self.frames) if f >= target), None)
        if i is None:
            i = len(self.frames) - 1
        self.pos = i; self.slider.set(i); self._show()

    def _toggle_play(self):
        self._playing = not self._playing
        self.play_btn.config(text="停止" if self._playing else "再生")
        if self._playing:
            self._play_step()

    def _play_step(self):
        if not self._playing:
            return
        if self.pos < len(self.frames) - 1:
            self._next()
            self.root.after(int(1000 / self.eff_fps), self._play_step)
        else:
            self._playing = False; self.play_btn.config(text="再生")

    def _zoom_by(self, f):
        self.zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * f)); self._render()

    def _zoom_fit(self):
        self.zoom = 1.0; self.canvas.xview_moveto(0); self.canvas.yview_moveto(0); self._render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1, help="開始画像番号")
    ap.add_argument("--end", type=int, default=0, help="終了画像番号(0=最後まで)")
    ap.add_argument("--frames-dir", default="frames",
                    help="フレーム画像フォルダ。高解像度なら frames_ball など")
    args = ap.parse_args()
    root = tk.Tk()
    BallAnnotator(root, args.start, args.end, args.frames_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
