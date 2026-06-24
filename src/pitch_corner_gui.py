"""
ピッチ四隅指定GUI

フレーム画像を表示し、ピッチの四隅を
  左上 → 右上 → 右下 → 左下
の順にクリックして指定する。保存すると pitch_points.json の image 座標を更新する。

起動:
  .venv/bin/python pitch_corner_gui.py
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

FRAMES_DIR = Path("frames")
POINTS_PATH = Path("pitch_points.json")
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

CORNER_LABELS = ["左上", "右上", "右下", "左下"]
CORNER_COLORS = ["#ff3b30", "#34c759", "#007aff", "#ff9500"]
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720
MARGIN = 50  # 画像の外側もクリックできるようにする余白(px)


class CornerPicker:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ピッチ四隅指定")

        self.frame_files = sorted(FRAMES_DIR.glob("frame_*.jpg"))
        if not self.frame_files:
            raise SystemExit("frames/ に frame_*.jpg が見つかりません")

        self.frame_index = 0
        self.points: list[tuple[float, float]] = []  # 元画像座標
        self.scale = 1.0
        self.img_w = 0
        self.img_h = 0
        self.tk_img: ImageTk.PhotoImage | None = None

        self._load_existing_points()
        self._build_ui()
        self._show_frame()

    # ---- データ ----
    def _load_existing_points(self) -> None:
        if POINTS_PATH.exists():
            try:
                data = json.loads(POINTS_PATH.read_text())
                pts = data.get("image", [])
                # 画像全体(0,0..1920,1080)のデフォルトなら未指定扱いにする
                default = [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]
                if pts and pts != default and len(pts) == 4:
                    self.points = [(float(x), float(y)) for x, y in pts]
            except Exception:
                pass

    # ---- UI ----
    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.status = tk.Label(top, text="", font=("Helvetica", 14), anchor="w")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btns = tk.Frame(self.root)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        tk.Button(btns, text="◀ 前", command=self._prev_frame).pack(side=tk.LEFT)
        tk.Button(btns, text="次 ▶", command=self._next_frame).pack(side=tk.LEFT, padx=(4, 16))
        tk.Button(btns, text="1つ戻す (Undo)", command=self._undo).pack(side=tk.LEFT)
        tk.Button(btns, text="全消去 (Reset)", command=self._reset).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="保存", command=self._save, bg="#34c759", fg="white",
                  font=("Helvetica", 13, "bold")).pack(side=tk.RIGHT)

        self.frame_slider = tk.Scale(
            self.root, from_=0, to=len(self.frame_files) - 1,
            orient=tk.HORIZONTAL, command=self._on_slider, showvalue=False,
        )
        self.frame_slider.pack(side=tk.TOP, fill=tk.X, padx=8)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self._on_click)

        self.root.bind("<Left>", lambda e: self._prev_frame())
        self.root.bind("<Right>", lambda e: self._next_frame())
        self.root.bind("<u>", lambda e: self._undo())
        self.root.bind("<Escape>", lambda e: self._reset())

    def _show_frame(self) -> None:
        path = self.frame_files[self.frame_index]
        img = Image.open(path)
        self.img_w, self.img_h = img.size
        self.scale = min(MAX_DISPLAY_W / self.img_w, MAX_DISPLAY_H / self.img_h, 1.0)
        disp = img.resize((int(self.img_w * self.scale), int(self.img_h * self.scale)))
        self.disp_w, self.disp_h = disp.width, disp.height
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp.width + 2 * MARGIN, height=disp.height + 2 * MARGIN)
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.tk_img is not None:
            self.canvas.create_image(MARGIN, MARGIN, anchor=tk.NW, image=self.tk_img)
        # 画像の外枠（フレーム境界）を点線で示す
        self.canvas.create_rectangle(MARGIN, MARGIN, MARGIN + self.disp_w, MARGIN + self.disp_h,
                                     outline="#888888", dash=(4, 4))

        # 多角形（4点揃ったら閉じる）
        if len(self.points) >= 2:
            scaled = [(x * self.scale + MARGIN, y * self.scale + MARGIN) for x, y in self.points]
            flat = [c for p in scaled for c in p]
            if len(self.points) == 4:
                self.canvas.create_polygon(*flat, outline="#ffffff", fill="", width=2)
            else:
                self.canvas.create_line(*flat, fill="#ffffff", width=2)

        # 各点
        for i, (x, y) in enumerate(self.points):
            sx, sy = x * self.scale + MARGIN, y * self.scale + MARGIN
            r = 6
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r,
                                    fill=CORNER_COLORS[i], outline="white", width=2)
            self.canvas.create_text(sx + 10, sy - 10, text=CORNER_LABELS[i],
                                    fill=CORNER_COLORS[i], font=("Helvetica", 13, "bold"),
                                    anchor=tk.W)
        self._update_status()

    def _update_status(self) -> None:
        fname = self.frame_files[self.frame_index].name
        n = len(self.points)
        if n < 4:
            nxt = CORNER_LABELS[n]
            msg = f"[{fname}]  {self.frame_index + 1}/{len(self.frame_files)}  —  次にクリック: 【{nxt}】 ({n}/4)"
        else:
            msg = f"[{fname}]  {self.frame_index + 1}/{len(self.frame_files)}  —  4点指定済み。保存できます ✓"
        self.status.config(text=msg)

    # ---- 操作 ----
    def _on_click(self, event: tk.Event) -> None:
        if len(self.points) >= 4:
            return
        # 余白ぶんを引いてから元解像度へ。画像外（負値や範囲外）も許可する
        ox = (event.x - MARGIN) / self.scale
        oy = (event.y - MARGIN) / self.scale
        self.points.append((round(ox, 1), round(oy, 1)))
        self._redraw()

    def _undo(self) -> None:
        if self.points:
            self.points.pop()
            self._redraw()

    def _reset(self) -> None:
        self.points = []
        self._redraw()

    def _prev_frame(self) -> None:
        if self.frame_index > 0:
            self.frame_index -= 1
            self.frame_slider.set(self.frame_index)
            self._show_frame()

    def _next_frame(self) -> None:
        if self.frame_index < len(self.frame_files) - 1:
            self.frame_index += 1
            self.frame_slider.set(self.frame_index)
            self._show_frame()

    def _on_slider(self, value: str) -> None:
        idx = int(value)
        if idx != self.frame_index:
            self.frame_index = idx
            self._show_frame()

    def _save(self) -> None:
        if len(self.points) != 4:
            messagebox.showwarning("未完了", "四隅を4点すべて指定してください")
            return
        data = {
            "note": "imageは画像上のピッチ四隅を左上,右上,右下,左下の順で入れる",
            "image": [[x, y] for x, y in self.points],
            "pitch": [[0, 0], [PITCH_LENGTH_M, 0],
                      [PITCH_LENGTH_M, PITCH_WIDTH_M], [0, PITCH_WIDTH_M]],
        }
        POINTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        messagebox.showinfo("保存完了", f"{POINTS_PATH} を更新しました。\n\n"
                            "次に step5 以降を再実行してください。")


def main() -> None:
    root = tk.Tk()
    CornerPicker(root)
    root.mainloop()


if __name__ == "__main__":
    main()
