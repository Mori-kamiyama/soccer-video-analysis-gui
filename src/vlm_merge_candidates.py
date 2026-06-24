"""
鳥瞰座標と前後フレーム画像から、ID統合候補を作る。

主な出力:
  outputs/vlm_merge_review/candidates.csv
  outputs/vlm_merge_review/panels/cand_XXXX_a_B.png

使い方:
  uv run python vlm_merge_candidates.py --limit 80

このスクリプトは自動で merge_map_hi.csv を書き換えない。
まず候補と確認画像を作り、VLMまたは人間が yes/no を付けるための土台にする。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


PLAYERS_CSV = Path("outputs/players_hi.csv")
MERGE_MAP_CSV = Path("outputs/merge_map_hi.csv")
FRAMES_DIR = Path("frames")
OUT_DIR = Path("outputs/vlm_merge_review")
PITCH_L = 105.0
PITCH_W = 68.0
BASE_W = 1920.0
BASE_H = 1080.0


@dataclass(frozen=True)
class Obs:
    frame: int
    track_id: int
    team_hint: str
    x1: float
    y1: float
    x2: float
    y2: float
    mark_x: float
    mark_y: float
    x_pitch: float
    y_pitch: float


@dataclass
class Unit:
    id: int
    members: list[int]
    rows: list[Obs]
    frames: set[int]
    team: str

    @property
    def first(self) -> int:
        return self.rows[0].frame

    @property
    def last(self) -> int:
        return self.rows[-1].frame

    @property
    def start(self) -> Obs:
        return self.rows[0]

    @property
    def end(self) -> Obs:
        return self.rows[-1]


@dataclass(frozen=True)
class Candidate:
    a_id: int
    b_id: int
    score: float
    gap: int
    end_dist_m: float
    pred_dist_m: float
    req_speed_mps: float
    turn_cost: float
    team: str
    panel: str


class UnionFind:
    def __init__(self, ids: Iterable[int]) -> None:
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
    p.add_argument("--merge-map", type=Path, default=MERGE_MAP_CSV)
    p.add_argument("--tracks-csv", type=Path, default=Path("outputs/tracks_hi.csv"))
    p.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--max-gap", type=int, default=72, help="候補にする最大フレーム間隔")
    p.add_argument("--max-dist", type=float, default=14.0, help="端点間の最大距離(m)")
    p.add_argument("--max-pred-dist", type=float, default=10.0, help="予測位置からの最大距離(m)")
    p.add_argument("--max-speed", type=float, default=9.0, help="必要速度の上限(m/s)")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--sample-stride", type=int, default=6, help="確認画像に使う前後フレーム間隔")
    p.add_argument("--base-width", type=float, default=BASE_W, help="players_hi.csv の画像座標基準幅")
    p.add_argument("--base-height", type=float, default=BASE_H, help="players_hi.csv の画像座標基準高")
    p.add_argument("--crop-pad", type=int, default=180, help="実フレームpxでの切り出し余白")
    p.add_argument("--draw-box", action="store_true", help="ズレ確認用にboxも描く")
    p.add_argument("--reset-decisions", action="store_true", help="既存のdecision/noteを引き継がない")
    p.add_argument("--allow-team-mismatch", action="store_true")
    return p.parse_args()


def load_track_points(path: Path) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """(frame, track_id) -> (x_img, y_img, x_pitch, y_pitch)。座標は1920基準。"""
    points: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    if not path.exists():
        return points
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                points[(int(r["frame"]), int(r["track_id"]))] = (
                    float(r["x_img"]),
                    float(r["y_img"]),
                    float(r["x_pitch"]),
                    float(r["y_pitch"]),
                )
            except (KeyError, ValueError):
                continue
    return points


def load_rows(path: Path, track_points: dict[tuple[int, int], tuple[float, float, float, float]]) -> list[Obs]:
    rows: list[Obs] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                frame = int(r["frame"])
                tid = int(r["track_id"])
                x1 = float(r["x1"])
                y1 = float(r["y1"])
                x2 = float(r["x2"])
                y2 = float(r["y2"])
                track_pt = track_points.get((frame, tid))
                if track_pt:
                    mark_x, mark_y, xp, yp = track_pt
                else:
                    mark_x = float(r.get("foot_x", (x1 + x2) / 2))
                    mark_y = float(r.get("foot_y", y2))
                    xp = float(r["x_pitch"])
                    yp = float(r["y_pitch"])
                rows.append(Obs(
                    frame=frame,
                    track_id=tid,
                    team_hint=r.get("team_hint", ""),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    mark_x=mark_x,
                    mark_y=mark_y,
                    x_pitch=xp,
                    y_pitch=yp,
                ))
            except (KeyError, ValueError):
                continue
    return rows


def load_units(rows: list[Obs], merge_map: Path) -> dict[int, Unit]:
    track_ids = sorted({r.track_id for r in rows})
    uf = UnionFind(track_ids)
    if merge_map.exists():
        with merge_map.open() as f:
            for r in csv.DictReader(f):
                try:
                    old = int(r["old_id"])
                    new = int(r["new_id"])
                except (KeyError, ValueError):
                    continue
                uf.union(old, new)

    by_unit: dict[int, list[Obs]] = defaultdict(list)
    members: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        uid = uf.find(r.track_id)
        by_unit[uid].append(r)
        members[uid].add(r.track_id)

    units: dict[int, Unit] = {}
    for uid, unit_rows in by_unit.items():
        unit_rows.sort(key=lambda r: (r.frame, r.track_id))
        team_counts = Counter(r.team_hint for r in unit_rows if r.team_hint and r.team_hint != "?")
        team = team_counts.most_common(1)[0][0] if team_counts else ""
        units[uid] = Unit(
            id=uid,
            members=sorted(members[uid]),
            rows=unit_rows,
            frames={r.frame for r in unit_rows},
            team=team,
        )
    return units


def point(r: Obs) -> tuple[float, float]:
    return r.x_pitch, r.y_pitch


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def velocity(rows: list[Obs], from_end: bool, lookback: int = 8) -> tuple[float, float]:
    if len(rows) < 2:
        return 0.0, 0.0
    ordered = rows[-lookback:] if from_end else rows[:lookback]
    if len(ordered) < 2:
        return 0.0, 0.0
    a, b = ordered[0], ordered[-1]
    dt = max(1, b.frame - a.frame)
    return (b.x_pitch - a.x_pitch) / dt, (b.y_pitch - a.y_pitch) / dt


def candidate_score(a: Unit, b: Unit, fps: float) -> tuple[float, float, float, float, float]:
    gap = b.first - a.last
    end_dist = dist(point(a.end), point(b.start))
    va = velocity(a.rows, from_end=True)
    vb = velocity(b.rows, from_end=False)
    pred = (a.end.x_pitch + va[0] * gap, a.end.y_pitch + va[1] * gap)
    pred = (min(PITCH_L, max(0.0, pred[0])), min(PITCH_W, max(0.0, pred[1])))
    pred_dist = dist(pred, point(b.start))
    req_speed = end_dist / max(gap / fps, 1 / fps)
    turn_cost = math.hypot(va[0] - vb[0], va[1] - vb[1]) * fps
    score = pred_dist + end_dist * 0.35 + gap * 0.04 + turn_cost * 0.18
    return score, end_dist, pred_dist, req_speed, turn_cost


def generate_candidates(units: dict[int, Unit], args: argparse.Namespace) -> list[Candidate]:
    ids = sorted(units, key=lambda uid: units[uid].first)
    candidates: list[Candidate] = []
    for a_id in ids:
        a = units[a_id]
        for b_id in ids:
            if a_id == b_id:
                continue
            b = units[b_id]
            gap = b.first - a.last
            if gap < 1 or gap > args.max_gap:
                continue
            if a.frames & b.frames:
                continue
            if not args.allow_team_mismatch and a.team and b.team and a.team != b.team:
                continue
            score, end_dist, pred_dist, req_speed, turn_cost = candidate_score(a, b, args.fps)
            if end_dist > args.max_dist or pred_dist > args.max_pred_dist or req_speed > args.max_speed:
                continue
            team = a.team if a.team == b.team else f"{a.team}/{b.team}".strip("/")
            candidates.append(Candidate(
                a_id=a_id,
                b_id=b_id,
                score=score,
                gap=gap,
                end_dist_m=end_dist,
                pred_dist_m=pred_dist,
                req_speed_mps=req_speed,
                turn_cost=turn_cost,
                team=team,
                panel="",
            ))
    candidates.sort(key=lambda c: c.score)
    return candidates[:args.limit]


def frame_path(frames_dir: Path, frame: int) -> Path:
    return frames_dir / f"frame_{frame:06d}.jpg"


def nearest_obs(unit: Unit, target_frame: int) -> Obs:
    return min(unit.rows, key=lambda r: abs(r.frame - target_frame))


def scaled_box(obs: Obs, img: Image.Image, base_w: float, base_h: float) -> tuple[float, float, float, float]:
    """players_hi.csv の基準座標を、実フレーム解像度のpxへ変換する。"""
    sx = img.width / base_w
    sy = img.height / base_h
    return obs.x1 * sx, obs.y1 * sy, obs.x2 * sx, obs.y2 * sy


def make_tile(args: argparse.Namespace, obs: Obs, label: str,
              size: tuple[int, int] = (300, 260)) -> Image.Image:
    img_path = frame_path(args.frames_dir, obs.frame)
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
    else:
        img = Image.new("RGB", (int(args.base_width), int(args.base_height)), "black")
    x1b, y1b, x2b, y2b = scaled_box(obs, img, args.base_width, args.base_height)
    sx0 = img.width / args.base_width
    sy0 = img.height / args.base_height
    mx = obs.mark_x * sx0
    my = obs.mark_y * sy0
    pad = args.crop_pad
    # 足元点は選手の下側なので、上方向を広く見る。
    x1 = max(0, int(mx - pad))
    y1 = max(0, int(my - pad * 1.45))
    x2 = min(img.width, int(mx + pad))
    y2 = min(img.height, int(my + pad * 0.55))
    crop = img.crop((x1, y1, x2, y2))
    crop.thumbnail((size[0], size[1] - 34))
    tile = Image.new("RGB", size, (245, 245, 245))
    tx = (size[0] - crop.width) // 2
    ty = 30 + (size[1] - 34 - crop.height) // 2
    tile.paste(crop, (tx, ty))
    draw = ImageDraw.Draw(tile)
    sx = crop.width / max(1, x2 - x1)
    sy = crop.height / max(1, y2 - y1)
    cmx = tx + (mx - x1) * sx
    cmy = ty + (my - y1) * sy
    if args.draw_box:
        bx1 = tx + (x1b - x1) * sx
        by1 = ty + (y1b - y1) * sy
        bx2 = tx + (x2b - x1) * sx
        by2 = ty + (y2b - y1) * sy
        draw.rectangle((bx1, by1, bx2, by2), outline=(255, 149, 0), width=2)
    # 赤十字は tracks_hi の足元点。boxがズレても位置候補を見られる。
    r = 8
    draw.line((cmx - 20, cmy, cmx + 20, cmy), fill=(255, 59, 48), width=4)
    draw.line((cmx, cmy - 20, cmx, cmy + 20), fill=(255, 59, 48), width=4)
    draw.ellipse((cmx - r, cmy - r, cmx + r, cmy + r), outline=(255, 255, 255), width=3)
    draw.text((8, 8), label, fill=(0, 0, 0))
    return tile


def make_panel(cand: Candidate, units: dict[int, Unit], args: argparse.Namespace, out_path: Path) -> None:
    a = units[cand.a_id]
    b = units[cand.b_id]
    stride = args.sample_stride
    refs = [
        (a, a.last - 2 * stride, "A end-2"),
        (a, a.last - stride, "A end-1"),
        (a, a.last, "A end"),
        (b, b.first, "B start"),
        (b, b.first + stride, "B start+1"),
        (b, b.first + 2 * stride, "B start+2"),
    ]
    tiles = []
    for unit, target, label in refs:
        obs = nearest_obs(unit, target)
        full_label = f"{label}  F{obs.frame}  {unit.team}  U{unit.id}"
        tiles.append(make_tile(args, obs, full_label))
    w, h = 3 * 300, 2 * 260 + 54
    panel = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    title = (
        f"candidate U{cand.a_id} -> U{cand.b_id}  "
        f"score={cand.score:.2f} gap={cand.gap}F "
        f"end={cand.end_dist_m:.1f}m pred={cand.pred_dist_m:.1f}m "
        f"speed={cand.req_speed_mps:.1f}m/s team={cand.team}"
    )
    draw.text((10, 8), title, fill=(0, 0, 0))
    for i, tile in enumerate(tiles):
        x = (i % 3) * 300
        y = 44 + (i // 3) * 260
        panel.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)


def write_outputs(candidates: list[Candidate], units: dict[int, Unit], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = args.out_dir / "panels"
    previous: dict[tuple[str, str], dict[str, str]] = {}
    existing_csv = args.out_dir / "candidates.csv"
    if existing_csv.exists() and not args.reset_decisions:
        try:
            with existing_csv.open() as f:
                for row in csv.DictReader(f):
                    previous[(row.get("a_id", ""), row.get("b_id", ""))] = row
        except Exception:
            previous = {}
    rows = []
    prompt_rows = []
    for i, cand in enumerate(candidates, start=1):
        panel_rel = Path("panels") / f"cand_{i:04d}_u{cand.a_id}_u{cand.b_id}.png"
        panel_path = args.out_dir / panel_rel
        make_panel(cand, units, args, panel_path)
        a = units[cand.a_id]
        b = units[cand.b_id]
        prev = previous.get((str(cand.a_id), str(cand.b_id)), {})
        rows.append({
            "rank": i,
            "decision": prev.get("decision", ""),
            "a_id": cand.a_id,
            "b_id": cand.b_id,
            "a_members": " ".join(map(str, a.members)),
            "b_members": " ".join(map(str, b.members)),
            "team": cand.team,
            "score": f"{cand.score:.3f}",
            "gap": cand.gap,
            "end_dist_m": f"{cand.end_dist_m:.3f}",
            "pred_dist_m": f"{cand.pred_dist_m:.3f}",
            "req_speed_mps": f"{cand.req_speed_mps:.3f}",
            "turn_cost": f"{cand.turn_cost:.3f}",
            "a_last": a.last,
            "b_first": b.first,
            "panel": panel_rel.as_posix(),
            "note": prev.get("note", ""),
        })
        prompt_rows.append({
            "rank": i,
            "a_id": cand.a_id,
            "b_id": cand.b_id,
            "image": str((args.out_dir / panel_rel).resolve()),
            "prompt": (
                "You are reviewing a soccer player tracking merge candidate. "
                "The panel shows the same candidate before disappearance (A) and after reappearance (B). "
                "The red cross marks the tracked foot position; it is more reliable than any bounding box. "
                "Decide whether A and B are likely the same player. Consider uniform color, body/pose, "
                "motion continuity, nearby lookalike players, and whether the transition is physically plausible. "
                "Return one of: same, different, uncertain. Then give a short reason."
            ),
            "metrics": {
                "team": cand.team,
                "gap_frames": cand.gap,
                "end_distance_m": round(cand.end_dist_m, 3),
                "predicted_distance_m": round(cand.pred_dist_m, 3),
                "required_speed_mps": round(cand.req_speed_mps, 3),
            },
        })
    fields = [
        "rank", "decision", "a_id", "b_id", "a_members", "b_members", "team",
        "score", "gap", "end_dist_m", "pred_dist_m", "req_speed_mps", "turn_cost",
        "a_last", "b_first", "panel", "note",
    ]
    with (args.out_dir / "candidates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (args.out_dir / "review_prompts.jsonl").open("w") as f:
        for row in prompt_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.players_csv, load_track_points(args.tracks_csv))
    units = load_units(rows, args.merge_map)
    candidates = generate_candidates(units, args)
    write_outputs(candidates, units, args)
    print(f"units: {len(units)}")
    print(f"candidates: {len(candidates)}")
    print(f"saved: {args.out_dir / 'candidates.csv'}")
    print(f"prompts: {args.out_dir / 'review_prompts.jsonl'}")
    if candidates:
        first_panel = args.out_dir / candidates[0].panel if candidates[0].panel else args.out_dir / "panels"
        print(f"panels: {args.out_dir / 'panels'}")


if __name__ == "__main__":
    main()
