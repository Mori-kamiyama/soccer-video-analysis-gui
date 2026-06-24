"""
VLM統合候補パネルを、はい/いいえ/不明で確認するGUI。

使い方:
  uv run python review_vlm_merge_panels.py

キー操作:
  y / Enter : はい
  n         : いいえ
  u / Space : 不明
  Left      : 前へ
  Right     : 次へ
"""

from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


CSV_PATH = Path("outputs/vlm_merge_review/candidates.csv")
BASE_DIR = Path("outputs/vlm_merge_review")


class ReviewApp:
    def __init__(self, root: tk.Tk, csv_path: Path, base_dir: Path) -> None:
        self.root = root
        self.csv_path = csv_path
        self.base_dir = base_dir
        self.rows = self._load_rows()
        self.idx = self._first_unanswered()
        self.photo: ImageTk.PhotoImage | None = None

        self.root.title("ID統合候補レビュー")
        self._build_ui()
        self._bind_keys()
        self._show()

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            raise SystemExit(f"{self.csv_path} がありません。先に vlm_merge_candidates.py を実行してください。")
        with self.csv_path.open() as f:
            return list(csv.DictReader(f))

    def _save_rows(self) -> None:
        if not self.rows:
            return
        fieldnames = list(self.rows[0].keys())
        with self.csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)

    def _first_unanswered(self) -> int:
        for i, row in enumerate(self.rows):
            if not (row.get("decision") or "").strip():
                return i
        return 0

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        self.title = tk.Label(top, text="", font=("Helvetica", 15, "bold"), anchor="w")
        self.title.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.jump_var = tk.IntVar(value=self.idx + 1)
        tk.Label(top, text="候補:").pack(side=tk.LEFT)
        jump = tk.Spinbox(top, from_=1, to=max(1, len(self.rows)), width=5,
                          textvariable=self.jump_var, command=self._jump)
        jump.pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="移動", command=self._jump).pack(side=tk.LEFT)

        self.image_label = tk.Label(self.root, bg="#f4f4f4")
        self.image_label.pack(side=tk.TOP, padx=10, pady=6)

        note_frame = tk.Frame(self.root)
        note_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        tk.Label(note_frame, text="メモ:").pack(side=tk.LEFT)
        self.note_var = tk.StringVar()
        note_entry = ttk.Entry(note_frame, textvariable=self.note_var)
        note_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        note_entry.bind("<FocusOut>", lambda _e: self._save_note())

        buttons = tk.Frame(self.root)
        buttons.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)
        tk.Button(buttons, text="← 前", command=self._prev, width=8).pack(side=tk.LEFT)
        tk.Button(buttons, text="はい (Y)", command=lambda: self._decide("yes"),
                  bg="#34c759", fg="white", font=("Helvetica", 14, "bold"),
                  width=12).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="いいえ (N)", command=lambda: self._decide("no"),
                  bg="#ff3b30", fg="white", font=("Helvetica", 14, "bold"),
                  width=12).pack(side=tk.LEFT)
        tk.Button(buttons, text="不明 (U)", command=lambda: self._decide("uncertain"),
                  bg="#8e8e93", fg="white", font=("Helvetica", 14, "bold"),
                  width=12).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="次 →", command=self._next, width=8).pack(side=tk.LEFT)
        tk.Button(buttons, text="保存して終了", command=self._quit,
                  bg="#007aff", fg="white").pack(side=tk.RIGHT)

        self.status = tk.Label(self.root, text="", anchor="w")
        self.status.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 8))

    def _bind_keys(self) -> None:
        self.root.bind("<y>", lambda _e: self._decide("yes"))
        self.root.bind("<Return>", lambda _e: self._decide("yes"))
        self.root.bind("<n>", lambda _e: self._decide("no"))
        self.root.bind("<u>", lambda _e: self._decide("uncertain"))
        self.root.bind("<space>", lambda _e: self._decide("uncertain"))
        self.root.bind("<Left>", lambda _e: self._prev())
        self.root.bind("<Right>", lambda _e: self._next())

    def _show(self) -> None:
        if not self.rows:
            messagebox.showinfo("レビュー", "候補がありません")
            return
        row = self.rows[self.idx]
        self.jump_var.set(self.idx + 1)
        panel = self.base_dir / row["panel"]
        img = Image.open(panel).convert("RGB")
        max_w, max_h = 1180, 820
        img.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        decision = row.get("decision", "")
        title = (
            f"{self.idx + 1}/{len(self.rows)}  "
            f"U{row['a_id']} -> U{row['b_id']}  "
            f"team={row['team']}  score={row['score']}  decision={decision or '-'}"
        )
        self.title.config(text=title)
        self.note_var.set(row.get("note", ""))
        answered = sum(1 for r in self.rows if (r.get("decision") or "").strip())
        yes = sum(1 for r in self.rows if (r.get("decision") or "").strip().lower() in {"yes", "same", "merge"})
        self.status.config(text=f"回答済み {answered}/{len(self.rows)} / はい {yes}")

    def _save_note(self) -> None:
        if self.rows:
            self.rows[self.idx]["note"] = self.note_var.get()
            self._save_rows()

    def _decide(self, decision: str) -> None:
        self.rows[self.idx]["decision"] = decision
        self.rows[self.idx]["note"] = self.note_var.get()
        self._save_rows()
        self._next()

    def _next(self) -> None:
        self._save_note()
        self.idx = min(len(self.rows) - 1, self.idx + 1)
        self._show()

    def _prev(self) -> None:
        self._save_note()
        self.idx = max(0, self.idx - 1)
        self._show()

    def _jump(self) -> None:
        self._save_note()
        self.idx = max(0, min(len(self.rows) - 1, self.jump_var.get() - 1))
        self._show()

    def _quit(self) -> None:
        self._save_note()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ReviewApp(root, CSV_PATH, BASE_DIR)
    root.mainloop()


if __name__ == "__main__":
    main()
