"""
VLM/人間レビュー済みの統合候補を merge_map_hi.csv に反映する。

入力:
  outputs/vlm_merge_review/candidates.csv

`decision` 列が yes / y / same / merge / 1 の行だけ採用する。
既存の merge_map_hi.csv は .bak を作ってから、全 track_id -> new_id の形で書き直す。

使い方:
  uv run python apply_vlm_merge_decisions.py
  uv run python build_player_ids.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path


PLAYERS_CSV = Path("outputs/players_hi.csv")
CANDIDATES_CSV = Path("outputs/vlm_merge_review/candidates.csv")
MERGE_MAP_CSV = Path("outputs/merge_map_hi.csv")
YES_VALUES = {"yes", "y", "same", "merge", "1", "true", "ok"}


class UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--players-csv", type=Path, default=PLAYERS_CSV)
    p.add_argument("--candidates", type=Path, default=CANDIDATES_CSV)
    p.add_argument("--merge-map", type=Path, default=MERGE_MAP_CSV)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_members(value: str) -> list[int]:
    out = []
    for part in (value or "").replace(",", " ").split():
        try:
            out.append(int(part))
        except ValueError:
            pass
    return out


def load_track_ids(players_csv: Path) -> tuple[list[int], dict[int, int]]:
    counts: dict[int, int] = defaultdict(int)
    with players_csv.open() as f:
        for r in csv.DictReader(f):
            try:
                counts[int(r["track_id"])] += 1
            except (KeyError, ValueError):
                continue
    return sorted(counts), counts


def load_existing(uf: UnionFind, merge_map: Path) -> int:
    if not merge_map.exists():
        return 0
    n = 0
    with merge_map.open() as f:
        for r in csv.DictReader(f):
            try:
                uf.union(int(r["old_id"]), int(r["new_id"]))
                n += 1
            except (KeyError, ValueError):
                continue
    return n


def apply_decisions(uf: UnionFind, candidates: Path) -> int:
    accepted = 0
    with candidates.open() as f:
        for r in csv.DictReader(f):
            decision = (r.get("decision") or "").strip().lower()
            if decision not in YES_VALUES:
                continue
            a_members = parse_members(r.get("a_members", ""))
            b_members = parse_members(r.get("b_members", ""))
            if not a_members or not b_members:
                continue
            base = a_members[0]
            for tid in a_members[1:] + b_members:
                uf.union(base, tid)
            accepted += 1
    return accepted


def write_merge_map(path: Path, track_ids: list[int], frame_counts: dict[int, int],
                    uf: UnionFind, dry_run: bool) -> None:
    groups: dict[int, list[int]] = defaultdict(list)
    for tid in track_ids:
        groups[uf.find(tid)].append(tid)

    ordered = sorted(
        groups.values(),
        key=lambda members: -sum(frame_counts.get(t, 0) for t in members),
    )
    rows = []
    for new_id, members in enumerate(ordered, start=1):
        for old_id in sorted(members):
            rows.append((old_id, new_id))

    if dry_run:
        print(f"dry-run: would write {len(rows)} rows to {path}")
        return

    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"backup: {backup}")
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_id", "new_id"])
        w.writerows(rows)
    print(f"saved: {path} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    track_ids, frame_counts = load_track_ids(args.players_csv)
    uf = UnionFind(track_ids)
    existing = load_existing(uf, args.merge_map)
    accepted = apply_decisions(uf, args.candidates)
    print(f"existing merge rows: {existing}")
    print(f"accepted decisions: {accepted}")
    write_merge_map(args.merge_map, track_ids, frame_counts, uf, args.dry_run)


if __name__ == "__main__":
    main()
