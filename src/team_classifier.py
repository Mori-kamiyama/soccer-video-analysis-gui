"""
チーム自動分類GUI

ジャージ色を画像から抽出してK-Meansで2チームに分類する。
修正前提設計: 結果を目で確認して間違いをドラッグ or クリックで直せる。

起動:
  .venv/bin/python team_classifier.py

出力:
  outputs/team_assignments.csv  (player_id -> team A/B)
"""

from __future__ import annotations

import colorsys
import csv
import tkinter as tk
from collections import defaultdict
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---- 設定 ----
INPUT_CSV = Path("outputs/player_positions_all.csv")
FRAMES_DIR = Path("frames")
OUTPUT_CSV = Path("outputs/team_assignments.csv")

JERSEY_TOP    = 0.10   # ボックス高さのうちジャージ抽出開始（頭を除く）
JERSEY_BOTTOM = 0.55   # 胴体中心まで（足・芝を除く）
SAMPLE_FRAMES = 15     # 1選手あたり何フレームからサンプリングするか
MIN_PIXELS    = 8      # サンプル色として採用する最小ピクセル数

# 「ピッチの平均色」をこの色から遠い方を選ぶ（Lab空間）
# cv2 Lab uint8: L=118, a=114, b=145 ≒ 中程度の緑（芝）
PITCH_LAB = np.array([118, 114, 145], dtype=np.float32)

TEAM_COLORS = {"A": "#e63946", "B": "#457b9d"}  # 表示用チーム色


# ---- ユーティリティ ----

def dominant_jersey_lab(bgr_crop: np.ndarray) -> np.ndarray | None:
    """クロップ内でミニK-Means(k=3)を走らせ、ピッチ色から最も遠いクラスタをジャージ色として返す。
    緑ジャージも黒ジャージも消さずに抽出できる。"""
    pixels = bgr_crop.reshape(-1, 3)
    if len(pixels) < MIN_PIXELS:
        return None
    lab_pixels = cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_BGR2LAB)[0].astype(np.float32)
    k = min(3, len(pixels))
    _, labels, centers = cv2.kmeans(
        lab_pixels, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
        3, cv2.KMEANS_RANDOM_CENTERS,
    )
    labels = labels.flatten()
    # 各クラスタの「ピッチ色との距離 × 所属ピクセル数」でスコアリング
    best_center = None
    best_score = -1.0
    for ci in range(k):
        mask = labels == ci
        count = mask.sum()
        if count == 0:
            continue
        dist = float(np.linalg.norm(centers[ci] - PITCH_LAB))
        score = dist * count
        if score > best_score:
            best_score = score
            best_center = centers[ci]
    return best_center  # shape (3,) in Lab float32 (uint8 scale)


def lab_to_rgb_hex(lab: np.ndarray) -> str:
    """cv2 LAB（uint8スケール: L 0-255, a 0-255, b 0-255）→ RGB hex文字列。"""
    lab_img = np.clip(lab, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    bgr = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    b, g, r = int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    return f"#{r:02x}{g:02x}{b:02x}"


# ---- サンプリング ----

def sample_jersey_colors(
    rows_by_pid: dict[int, list[dict]],
) -> dict[int, np.ndarray | None]:
    """選手IDごとにジャージ色（L*a*b平均）を抽出する。"""
    result: dict[int, np.ndarray | None] = {}
    total = len(rows_by_pid)
    for i, (pid, rows) in enumerate(rows_by_pid.items()):
        print(f"\r  サンプリング中... {i+1}/{total}", end="", flush=True)
        # 均等にフレームを選ぶ
        step = max(1, len(rows) // SAMPLE_FRAMES)
        samples = rows[::step][:SAMPLE_FRAMES]
        all_pixels = []
        for row in samples:
            fname = row["filename"]
            img_path = FRAMES_DIR / fname
            if not img_path.exists():
                continue
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                x1 = int(float(row["x1"]))
                y1 = int(float(row["y1"]))
                x2 = int(float(row["x2"]))
                y2 = int(float(row["y2"]))
                h = y2 - y1
                jy1 = y1 + int(h * JERSEY_TOP)
                jy2 = y1 + int(h * JERSEY_BOTTOM)
                if jy2 <= jy1 or x2 <= x1:
                    continue
                crop = img[jy1:jy2, x1:x2]
                if crop.size == 0:
                    continue
                lab = dominant_jersey_lab(crop)
                if lab is not None:
                    all_pixels.append(lab.reshape(1, 3))
            except Exception:
                continue
        if all_pixels:
            merged = np.concatenate(all_pixels, axis=0)  # shape (N, 3)
            result[pid] = merged.mean(axis=0)
        else:
            result[pid] = None
    print()
    return result


# ---- クラスタリング ----

def cluster_teams(
    color_map: dict[int, np.ndarray | None],
) -> dict[int, str]:
    """L*a*b色でK-means(k=2)して A/B を返す。色が取れなかった選手は"?"。"""
    valid_pids = [pid for pid, lab in color_map.items() if lab is not None]
    if len(valid_pids) < 2:
        return {pid: "?" for pid in color_map}

    data = np.array([color_map[pid] for pid in valid_pids], dtype=np.float32)
    _, labels, centers = cv2.kmeans(
        data, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10, cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.flatten()

    # クラスタ0/1をA/Bに割り当て（どちらがAかはL値=明度で決める: 暗い方がA）
    mean_l0 = data[labels == 0, 0].mean() if (labels == 0).any() else 999
    mean_l1 = data[labels == 1, 0].mean() if (labels == 1).any() else 999
    cluster_to_team = {0: "A", 1: "B"} if mean_l0 <= mean_l1 else {0: "B", 1: "A"}

    assignments: dict[int, str] = {}
    for pid, lab in color_map.items():
        if lab is None:
            assignments[pid] = "?"
        else:
            idx = valid_pids.index(pid)
            assignments[pid] = cluster_to_team[labels[idx]]
    return assignments


# ---- GUI ----

class TeamClassifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("チーム自動分類")

        self.rows_by_pid: dict[int, list[dict]] = defaultdict(list)
        self.color_map: dict[int, np.ndarray | None] = {}
        self.assignments: dict[int, str] = {}
        self._drag_pid: int | None = None
        self._drag_label: tk.Label | None = None

        self._load_existing_assignments()
        self._load_rows()
        self._build_ui()

    def _load_rows(self) -> None:
        with INPUT_CSV.open() as f:
            for r in csv.DictReader(f):
                try:
                    pid = int(r["player_id"])
                except (ValueError, KeyError):
                    continue
                self.rows_by_pid[pid].append(r)

    def _load_existing_assignments(self) -> None:
        if OUTPUT_CSV.exists():
            with OUTPUT_CSV.open() as f:
                for r in csv.DictReader(f):
                    try:
                        self.assignments[int(r["player_id"])] = r["team"]
                    except (ValueError, KeyError):
                        pass

    # ---- UI ----
    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(top, text="① ジャージ色を抽出 & 自動分類",
                  command=self._run_auto, bg="#ff9500", fg="white",
                  font=("Helvetica", 13, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="全員をAに", command=lambda: self._set_all("A")).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="全員をBに", command=lambda: self._set_all("B")).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="A↔B 入れ替え", command=self._swap_teams).pack(side=tk.LEFT, padx=8)
        tk.Button(top, text="保存", command=self._save,
                  bg="#34c759", fg="white", font=("Helvetica", 13, "bold")).pack(side=tk.RIGHT)

        self.count_label = tk.Label(top, text="", font=("Helvetica", 12))
        self.count_label.pack(side=tk.RIGHT, padx=12)

        # 説明
        tk.Label(self.root,
                 text="ドラッグでA↔B移動 / 右クリックメニューで変更 / 色の四角＝ジャージ色",
                 fg="#555", font=("Helvetica", 11)).pack(side=tk.TOP, fill=tk.X, padx=8)

        # 2列フレーム
        cols = tk.Frame(self.root)
        cols.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.panels: dict[str, tk.Frame] = {}
        self.list_frames: dict[str, tk.Frame] = {}
        for team in ("A", "B", "?"):
            col = tk.Frame(cols, relief=tk.GROOVE, bd=2)
            col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            color = TEAM_COLORS.get(team, "#888")
            header = tk.Label(col, text=f"チーム {team}", bg=color, fg="white",
                              font=("Helvetica", 14, "bold"), pady=6)
            header.pack(fill=tk.X)
            self.panels[team] = col

            canvas = tk.Canvas(col, bg="#f5f5f5", highlightthickness=0)
            sb = tk.Scrollbar(col, orient=tk.VERTICAL, command=canvas.yview)
            canvas.config(yscrollcommand=sb.set)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            inner = tk.Frame(canvas, bg="#f5f5f5")
            canvas.create_window((0, 0), window=inner, anchor="nw")
            inner.bind("<Configure>", lambda e, c=canvas: c.config(scrollregion=c.bbox("all")))
            self.list_frames[team] = inner

        if self.assignments:
            self._render_lists()

    def _run_auto(self) -> None:
        pids = sorted(self.rows_by_pid.keys())
        n = len(pids)
        if messagebox.askyesno("確認",
                               f"{n}選手のジャージ色を抽出します（数分かかる場合あり）。\n続けますか？"):
            self.count_label.config(text="抽出中...")
            self.root.update()
            self.color_map = sample_jersey_colors(self.rows_by_pid)
            self.assignments = cluster_teams(self.color_map)
            self._render_lists()
            self.count_label.config(
                text=f"A:{sum(1 for v in self.assignments.values() if v=='A')} "
                     f"B:{sum(1 for v in self.assignments.values() if v=='B')} "
                     f"?:{sum(1 for v in self.assignments.values() if v=='?')}")

    def _render_lists(self) -> None:
        for team, frame in self.list_frames.items():
            for w in frame.winfo_children():
                w.destroy()

        by_team: dict[str, list[int]] = defaultdict(list)
        for pid, team in self.assignments.items():
            by_team[team].append(pid)

        for team, pids in by_team.items():
            frame = self.list_frames.get(team)
            if frame is None:
                continue
            for pid in sorted(pids):
                self._make_player_row(frame, pid, team)

        self._update_count()

    def _make_player_row(self, parent: tk.Frame, pid: int, team: str) -> tk.Label:
        row = tk.Frame(parent, bg="#f5f5f5", pady=1)
        row.pack(fill=tk.X, padx=2)

        # ジャージ色スウォッチ
        lab = self.color_map.get(pid)
        swatch_color = lab_to_rgb_hex(lab) if lab is not None else "#cccccc"
        swatch = tk.Label(row, bg=swatch_color, width=3, relief=tk.RAISED)
        swatch.pack(side=tk.LEFT, padx=(2, 4))

        lbl = tk.Label(row, text=f"ID {pid:4d}", bg="#f5f5f5",
                       font=("Courier", 11), cursor="hand2")
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 右クリックメニュー
        menu = tk.Menu(self.root, tearoff=0)
        for t in ("A", "B", "?"):
            if t != team:
                menu.add_command(label=f"チーム {t} に移動",
                                 command=lambda p=pid, tg=t: self._move_pid(p, tg))
        lbl.bind("<Button-2>", lambda e, m=menu: m.post(e.x_root, e.y_root))
        lbl.bind("<Button-3>", lambda e, m=menu: m.post(e.x_root, e.y_root))

        # ドラッグ
        lbl.bind("<ButtonPress-1>", lambda e, p=pid: self._drag_start(e, p))
        lbl.bind("<B1-Motion>", self._drag_motion)
        lbl.bind("<ButtonRelease-1>", self._drag_end)
        swatch.bind("<ButtonPress-1>", lambda e, p=pid: self._drag_start(e, p))
        swatch.bind("<B1-Motion>", self._drag_motion)
        swatch.bind("<ButtonRelease-1>", self._drag_end)

        return lbl

    def _move_pid(self, pid: int, target_team: str) -> None:
        self.assignments[pid] = target_team
        self._render_lists()

    def _set_all(self, team: str) -> None:
        for pid in self.assignments:
            self.assignments[pid] = team
        self._render_lists()

    def _swap_teams(self) -> None:
        swap = {"A": "B", "B": "A", "?": "?"}
        self.assignments = {pid: swap[t] for pid, t in self.assignments.items()}
        self._render_lists()

    def _update_count(self) -> None:
        a = sum(1 for v in self.assignments.values() if v == "A")
        b = sum(1 for v in self.assignments.values() if v == "B")
        q = sum(1 for v in self.assignments.values() if v == "?")
        self.count_label.config(text=f"A:{a}  B:{b}  ?:{q}")

    # ---- ドラッグ ----
    def _drag_start(self, e: tk.Event, pid: int) -> None:
        self._drag_pid = pid
        self._drag_win = tk.Toplevel(self.root)
        self._drag_win.overrideredirect(True)
        self._drag_win.attributes("-alpha", 0.7)
        tk.Label(self._drag_win, text=f" ID {pid} ", bg="#333", fg="white",
                 font=("Courier", 12, "bold"), padx=6).pack()
        self._drag_win.geometry(f"+{e.x_root+10}+{e.y_root+10}")

    def _drag_motion(self, e: tk.Event) -> None:
        if hasattr(self, "_drag_win") and self._drag_win:
            self._drag_win.geometry(f"+{e.x_root+10}+{e.y_root+10}")

    def _drag_end(self, e: tk.Event) -> None:
        if hasattr(self, "_drag_win") and self._drag_win:
            self._drag_win.destroy()
            self._drag_win = None
        if self._drag_pid is None:
            return
        pid = self._drag_pid
        self._drag_pid = None
        # どのパネルの上で離したかを判定
        wx, wy = e.x_root, e.y_root
        for team, panel in self.panels.items():
            px = panel.winfo_rootx()
            py = panel.winfo_rooty()
            pw = panel.winfo_width()
            ph = panel.winfo_height()
            if px <= wx <= px + pw and py <= wy <= py + ph:
                if self.assignments.get(pid) != team:
                    self._move_pid(pid, team)
                return

    # ---- 保存 ----
    def _save(self) -> None:
        if not self.assignments:
            messagebox.showwarning("未分類", "先に「ジャージ色を抽出 & 自動分類」を実行してください。")
            return
        with OUTPUT_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["player_id", "team",
                        "lab_L", "lab_a", "lab_b", "jersey_hex"])
            for pid in sorted(self.assignments):
                lab = self.color_map.get(pid)
                if lab is not None:
                    hex_c = lab_to_rgb_hex(lab)
                    w.writerow([pid, self.assignments[pid],
                                round(float(lab[0]), 1),
                                round(float(lab[1]), 1),
                                round(float(lab[2]), 1), hex_c])
                else:
                    w.writerow([pid, self.assignments[pid], "", "", "", ""])
        a = sum(1 for v in self.assignments.values() if v == "A")
        b = sum(1 for v in self.assignments.values() if v == "B")
        messagebox.showinfo("保存完了",
                            f"チームA: {a}選手 / チームB: {b}選手\n\n"
                            f"→ {OUTPUT_CSV}\n\n"
                            "この結果を photo_annotator.py の「チーム表示」や\n"
                            "step6 のヒートマップに反映できます。")


def main() -> None:
    root = tk.Tk()
    root.geometry("1100x750")
    TeamClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
