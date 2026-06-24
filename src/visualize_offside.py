"""
オフサイド可視化動画を作る（点滅低減・中割り対応版）。

photo_annotator.py で仕上げた box_corrections.csv（緑/黒チーム）を読み、
  - 上段: カメラ映像にボックスを重ねる。オフサイド位置の選手は赤で強調
  - 下段: 頂上(真上)からのピッチ図。選手をドット表示し、
          両チームの最終守備ライン（=オフサイドライン, 2人目の守備者）を縦線で描く。
          オフサイドの選手は赤リングで強調
を1フレームに合成し、mp4 として書き出す。

★点滅対策:
  - 選手を時系列で追跡(最近傍)してIDを付与 → 短い検出欠落を補間 → 位置をEMA平滑化
  - 実フレーム間に中割り(補間)を K 枚挿入して動きを滑らかに
  - オフサイド判定をトラックごとに時間方向で多数決平滑化(ヒステリシス)＋マージン
  - 最終ラインの位置も時間方向で移動平均

攻撃方向はセグメント（スキップ区間で分割）ごとに自動判定する。
ボール座標が無いため「ボールより前」条件は省略し、
「相手の2人目の守備者より前 かつ 相手陣内」をオフサイド条件とする。

使い方:
  .venv/bin/python visualize_offside.py
  .venv/bin/python visualize_offside.py --start 212 --end 1311 --skip 955-1010 \
      --inbetween 4 --fps 8
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import photo_annotator as pa

PITCH_L = 105.0
PITCH_W = 68.0

# 出力レイアウト
CANVAS_W = 1280
CAM_H = 720
PPM = 11
PITCH_PX_W = int(PITCH_L * PPM)
PITCH_PX_H = int(PITCH_W * PPM)
PAD_X = (CANVAS_W - PITCH_PX_W) // 2
PAD_Y = 46
PANEL_H = PITCH_PX_H + PAD_Y + 40

# BGR
COL_GREEN = (60, 200, 60)
COL_BLACK = (30, 30, 30)
COL_OFFSIDE = (40, 40, 230)
COL_LINE_GREEN = (90, 230, 90)
COL_LINE_BLACK = (200, 200, 200)
COL_PITCH = (60, 120, 50)
COL_WHITE = (235, 235, 235)

TEAM_BGR = {1: COL_GREEN, 2: COL_BLACK}
TEAM_LINE = {1: COL_LINE_GREEN, 2: COL_LINE_BLACK}
TEAM_NAME = {1: "緑(A)", 2: "黒(B)"}


# ---- 最終ボックスの復元（photo_annotator と同じ手順） ----
def reconstruct_boxes() -> dict[int, list[dict]]:
    boxes: dict[int, list[dict]] = defaultdict(list)
    with pa.DET_CSV.open() as f:
        for r in csv.DictReader(f):
            try:
                frame = int(r["frame"]); key = int(r["detection_id"])
                x1, y1, x2, y2 = (float(r["x1"]), float(r["y1"]),
                                  float(r["x2"]), float(r["y2"]))
            except (ValueError, KeyError):
                continue
            g = pa.group_from_team_hint(r.get("team_hint", r.get("team", "")))
            boxes[frame].append({"key": key, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "g": g, "added": False, "deleted": False})
    index = {(f, b["key"]): b for f in boxes for b in boxes[f]}
    if pa.CORRECTIONS_CSV.exists():
        with pa.CORRECTIONS_CSV.open() as f:
            for r in csv.DictReader(f):
                fr = int(r["frame"]); key = int(r["box_key"]); act = r["action"]
                if act == "add":
                    b = {"key": key, "x1": float(r["x1"]), "y1": float(r["y1"]),
                         "x2": float(r["x2"]), "y2": float(r["y2"]),
                         "g": int(r["group"]), "added": True, "deleted": False}
                    boxes[fr].append(b); index[(fr, key)] = b
                else:
                    b = index.get((fr, key))
                    if b is None:
                        continue
                    if act == "delete":
                        b["deleted"] = True
                    elif act == "reassign":
                        b["g"] = int(r["group"])
                    elif act == "move":
                        b["x1"], b["y1"] = float(r["x1"]), float(r["y1"])
                        b["x2"], b["y2"] = float(r["x2"]), float(r["y2"])
    return boxes


def load_homography():
    data = json.loads(pa.POINTS_PATH.read_text())
    img = np.array(data["image"], dtype=np.float32)
    pitch = np.array(data["pitch"], dtype=np.float32)
    return cv2.getPerspectiveTransform(img, pitch)


def raw_players(boxes_for_frame, H):
    """[{X,Y,team,x1,y1,x2,y2}, ...]  team in (1,2) のみ。"""
    out = []
    pts = []
    metas = []
    for b in boxes_for_frame:
        if b["deleted"] or b["g"] not in (1, 2):
            continue
        fx = (b["x1"] + b["x2"]) / 2.0
        fy = b["y2"]
        pts.append([fx, fy])
        metas.append((b["g"], b["x1"], b["y1"], b["x2"], b["y2"]))
    if not pts:
        return out
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(arr, H).reshape(-1, 2)
    for (X, Y), (g, x1, y1, x2, y2) in zip(proj, metas):
        out.append({"X": float(X), "Y": float(Y), "team": g,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


# ---- セグメント分割 ----
def split_segments(frame_idxs: list[int]) -> list[list[int]]:
    segs = []
    cur = [frame_idxs[0]]
    for f in frame_idxs[1:]:
        if f == cur[-1] + 1:
            cur.append(f)
        else:
            segs.append(cur); cur = [f]
    segs.append(cur)
    return segs


# ---- トラッキング（チームごと最近傍・貪欲）----
def build_tracks(seg_frames, raw_by_frame, thresh=7.0, maxgap=5):
    """セグメント内で選手を追跡。track[tid] = {team, pos:{fr:(X,Y,box)}}。
    速度予測（直近2点）で次フレーム位置を見込んでマッチング → 高速移動でもIDが切れにくい。"""
    tracks: dict[int, dict] = {}
    active: dict[int, int] = {}   # tid -> last seen frame
    next_id = 0
    for fr in seg_frames:
        dets = raw_by_frame.get(fr, [])
        live = [tid for tid, last in active.items() if fr - last <= maxgap]
        pairs = []
        for tid in live:
            t = tracks[tid]
            fs = sorted(t["pos"])
            lf = fs[-1]
            px, py, _ = t["pos"][lf]
            # 速度予測: 直近2点から外挿して、消失中フレームぶん進めた位置を見込む
            if len(fs) >= 2:
                pf = fs[-2]
                qx, qy, _ = t["pos"][pf]
                span = max(1, lf - pf)
                vx, vy = (px - qx) / span, (py - qy) / span
                step = fr - lf
                predx, predy = px + vx * step, py + vy * step
            else:
                predx, predy = px, py
            for di, d in enumerate(dets):
                if d["team"] != t["team"]:
                    continue
                dist = ((predx - d["X"]) ** 2 + (predy - d["Y"]) ** 2) ** 0.5
                if dist <= thresh:
                    pairs.append((dist, tid, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid); used_d.add(di)
            d = dets[di]
            tracks[tid]["pos"][fr] = (d["X"], d["Y"],
                                      (d["x1"], d["y1"], d["x2"], d["y2"]))
            active[tid] = fr
        for di, d in enumerate(dets):
            if di in used_d:
                continue
            tid = next_id; next_id += 1
            tracks[tid] = {"team": d["team"],
                           "pos": {fr: (d["X"], d["Y"],
                                        (d["x1"], d["y1"], d["x2"], d["y2"]))}}
            active[tid] = fr
    return tracks


def fill_and_smooth(tracks, maxgap=5, alpha=0.5):
    """短い欠落を線形補間で埋め、位置をEMA平滑化する。"""
    for t in tracks.values():
        frames = sorted(t["pos"])
        # ギャップ補間
        filled = dict(t["pos"])
        for a, b in zip(frames, frames[1:]):
            gap = b - a
            if 1 < gap <= maxgap:
                Xa, Ya, ba = t["pos"][a]
                Xb, Yb, bb = t["pos"][b]
                for f in range(a + 1, b):
                    s = (f - a) / gap
                    box = tuple(ba[i] + (bb[i] - ba[i]) * s for i in range(4))
                    filled[f] = (Xa + (Xb - Xa) * s, Ya + (Yb - Ya) * s, box)
        # EMA平滑化
        order = sorted(filled)
        sm = {}
        prevX = prevY = None
        prevbox = None
        for f in order:
            X, Y, box = filled[f]
            if prevX is None:
                sX, sY, sbox = X, Y, box
            else:
                sX = alpha * X + (1 - alpha) * prevX
                sY = alpha * Y + (1 - alpha) * prevY
                sbox = tuple(alpha * box[i] + (1 - alpha) * prevbox[i] for i in range(4))
            sm[f] = (sX, sY, sbox)
            prevX, prevY, prevbox = sX, sY, sbox
        t["pos"] = sm


def players_at(tracks, fr):
    out = []
    for tid, t in tracks.items():
        if fr in t["pos"]:
            X, Y, box = t["pos"][fr]
            out.append({"tid": tid, "team": t["team"], "X": X, "Y": Y, "box": box})
    return out


def load_players_hi(path="outputs/players_hi.csv"):
    """安定player_id付きデータを {pid: {team, pos:{fr:(X,Y,box)}}} で返す。無ければNone。"""
    p = Path(path)
    if not p.exists():
        return None
    from collections import Counter
    raw = defaultdict(lambda: {"pos": {}, "team": Counter()})
    with p.open() as f:
        for r in csv.DictReader(f):
            pid = int(r["player_id"]); fr = int(r["frame"])
            box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            raw[pid]["pos"][fr] = (float(r["x_pitch"]), float(r["y_pitch"]), box)
            raw[pid]["team"][int(r["team"])] += 1
    tracks = {}
    for pid, d in raw.items():
        tracks[pid] = {"team": d["team"].most_common(1)[0][0], "pos": d["pos"]}
    return tracks


def tracks_for_segment(all_tracks, seg_frames, min_len=3):
    """全体トラックから、セグメント内のフレームだけ抜き出す。短すぎる断片は捨てる（チラつき低減）。"""
    segset = set(seg_frames)
    out = {}
    for pid, t in all_tracks.items():
        pos = {fr: v for fr, v in t["pos"].items() if fr in segset}
        if len(pos) >= min_len:
            out[pid] = {"team": t["team"], "pos": pos}
    return out


def detect_attack_dirs(seg_frames, tracks) -> dict[int, int]:
    vote0 = {1: 0, 2: 0}; vote105 = {1: 0, 2: 0}
    by_frame = defaultdict(lambda: defaultdict(list))
    for tid, t in tracks.items():
        for fr, (X, Y, _) in t["pos"].items():
            by_frame[fr][t["team"]].append(X)
    for fr in seg_frames:
        for tm, xs in by_frame.get(fr, {}).items():
            if len(xs) < 3:
                continue
            if min(xs) < PITCH_L - max(xs):
                vote0[tm] += 1
            else:
                vote105[tm] += 1
    dirs = {t: (1 if vote0[t] >= vote105[t] else -1) for t in (1, 2)}
    if dirs[1] == dirs[2]:
        m1 = abs(vote0[1] - vote105[1]); m2 = abs(vote0[2] - vote105[2])
        dirs[2 if m2 <= m1 else 1] *= -1
    return dirs


def compute_lines_offside(players, attack_dirs, margin):
    """lines={dfn:X}, offside_tids=set"""
    lines = {}
    offside = set()
    by_team = defaultdict(list)
    for p in players:
        by_team[p["team"]].append(p)
    for atk in (1, 2):
        dfn = 2 if atk == 1 else 1
        dt = attack_dirs[atk]
        defenders = by_team[dfn]
        if len(defenders) >= 2:
            ds = sorted(defenders, key=lambda q: dt * q["X"], reverse=True)
            line_x = ds[1]["X"]
            lines[dfn] = line_x
            for a in by_team[atk]:
                if dt * a["X"] > dt * (PITCH_L / 2) and dt * a["X"] > dt * line_x + margin:
                    offside.add(a["tid"])
    return lines, offside


def smooth_series(seq_by_frame, frames, win=1):
    """frames(連番)に沿って移動平均。seq_by_frame: fr->value(or None)。"""
    out = {}
    for i, fr in enumerate(frames):
        vals = []
        for j in range(max(0, i - win), min(len(frames), i + win + 1)):
            v = seq_by_frame.get(frames[j])
            if v is not None:
                vals.append(v)
        out[fr] = sum(vals) / len(vals) if vals else None
    return out


# ---- 描画 ----
def pitch_xy(X, Y):
    return PAD_X + int(X * PPM), PAD_Y + int(Y * PPM)


_FONT_CACHE = {}
def _get_font(size):
    from PIL import ImageFont
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in ("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                 "/System/Library/Fonts/Hiragino Sans GB.ttc",
                 "/Library/Fonts/Arial Unicode.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
        if Path(path).exists():
            f = ImageFont.truetype(path, size); _FONT_CACHE[size] = f; return f
    f = ImageFont.load_default(); _FONT_CACHE[size] = f; return f


def put_jp(img_bgr, text, org, color_bgr, size):
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((org[0], org[1] - size), text, font=_get_font(size),
           fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
    img_bgr[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_pitch_panel(players, lines, attack_dirs, offside, seg_label):
    panel = np.full((PANEL_H, CANVAS_W, 3), COL_PITCH, dtype=np.uint8)
    x0, y0 = pitch_xy(0, 0); x1, y1 = pitch_xy(PITCH_L, PITCH_W)
    cv2.rectangle(panel, (x0, y0), (x1, y1), COL_WHITE, 2)
    midx, _ = pitch_xy(PITCH_L / 2, 0)
    cv2.line(panel, (midx, y0), (midx, y1), COL_WHITE, 1)
    cx, cy = pitch_xy(PITCH_L / 2, PITCH_W / 2)
    cv2.circle(panel, (cx, cy), int(9.15 * PPM), COL_WHITE, 1)
    for gx in (0, PITCH_L):
        sign = 1 if gx == 0 else -1
        ax1, ay1 = pitch_xy(gx, (PITCH_W - 40.3) / 2)
        ax2, ay2 = pitch_xy(gx + sign * 16.5, (PITCH_W + 40.3) / 2)
        cv2.rectangle(panel, (ax1, ay1), (ax2, ay2), COL_WHITE, 1)

    for dfn_team, line_x in lines.items():
        lx, _ = pitch_xy(line_x, 0)
        col = TEAM_LINE[dfn_team]
        cv2.line(panel, (lx, y0), (lx, y1), col, 2, cv2.LINE_AA)
        put_jp(panel, f"{TEAM_NAME[dfn_team]}最終ライン", (lx + 4, y0 + 16), col, 14)

    for t in (1, 2):
        dt = attack_dirs[t]
        ay = y0 - 30 if t == 1 else y0 - 14
        ex = midx + (40 * dt)
        cv2.arrowedLine(panel, (midx, ay), (ex, ay), TEAM_LINE[t], 2, tipLength=0.4)
        put_jp(panel, f"{TEAM_NAME[t]}→", (midx - 150 if dt < 0 else midx + 70, ay + 6),
               TEAM_LINE[t], 13)

    for p in players:
        px, py = pitch_xy(p["X"], p["Y"])
        if not (x0 - 30 <= px <= x1 + 30 and y0 - 30 <= py <= y1 + 30):
            continue
        col = TEAM_BGR[p["team"]]
        cv2.circle(panel, (px, py), 7, col, -1)
        cv2.circle(panel, (px, py), 7, COL_WHITE, 1)
        if p["tid"] in offside:
            cv2.circle(panel, (px, py), 12, COL_OFFSIDE, 3)

    put_jp(panel, seg_label, (PAD_X, PANEL_H - 14), COL_WHITE, 16)
    return panel


def draw_camera(img, players, offside, dscale=1.0):
    # box は1920基準。カメラ画像が2560等なら dscale=frame_w/1920 で拡大して描く。
    for p in players:
        x1, y1, x2, y2 = (int(p["box"][0] * dscale), int(p["box"][1] * dscale),
                          int(p["box"][2] * dscale), int(p["box"][3] * dscale))
        if p["tid"] in offside:
            col = COL_OFFSIDE; w = 4
        else:
            col = TEAM_BGR[p["team"]]; w = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), col, w)
        fx, fy = (x1 + x2) // 2, y2
        cv2.circle(img, (fx, fy), 3, col, -1)
        if p["tid"] in offside:
            put_jp(img, "OFF", (x1, y1 - 2), COL_OFFSIDE, 22)
    return img


def parse_skips(skip_args):
    skips = []
    for s in skip_args or []:
        a, _, b = s.partition("-")
        skips.append((int(a), int(b or a)))
    return skips


def in_skip(img_no, skips):
    return any(a <= img_no <= b for a, b in skips)


def lerp(a, b, t):
    return a + (b - a) * t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=212)
    ap.add_argument("--end", type=int, default=1311)
    ap.add_argument("--skip", action="append", default=None)
    ap.add_argument("--inbetween", type=int, default=4, help="実フレーム間に挿す中割り枚数")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA係数(小さいほど滑らか)")
    ap.add_argument("--offside-margin", type=float, default=0.6, help="オフサイド判定の余裕(m)")
    ap.add_argument("--offside-win", type=int, default=2, help="オフサイド時間平滑の半幅(実フレーム)")
    ap.add_argument("--no-ids", action="store_true", help="players_hi.csvの安定IDを使わず毎回トラッキング")
    ap.add_argument("--min-track", type=int, default=4, help="この長さ未満の断片トラックは捨てる")
    ap.add_argument("--out", default="outputs/offside_vis.mp4")
    args = ap.parse_args()

    skips = parse_skips(args.skip if args.skip is not None else ["955-1010"])
    print("スキップ区間:", skips, flush=True)
    K = max(1, args.inbetween)

    # フレーム間隔fps（時刻表示用）と 1920基準→フレームpx 倍率
    eff_fps = 2.0
    try:
        md = json.loads((pa.FRAMES_DIR / "frames_metadata.json").read_text())
        if len(md) >= 2:
            dt = md[1]["timestamp"] - md[0]["timestamp"]
            if dt > 0:
                eff_fps = 1.0 / dt
    except Exception:
        pass
    sample = sorted(pa.FRAMES_DIR.glob("frame_*.jpg"))
    dscale = (cv2.imread(str(sample[0])).shape[1] / 1920.0) if sample else 1.0
    print(f"フレーム間隔fps={eff_fps:.1f}  1920→px倍率={dscale:.3f}", flush=True)

    boxes = reconstruct_boxes()
    H = load_homography()

    target_imgs = [n for n in range(args.start, args.end + 1)
                   if not in_skip(n, skips)
                   and (pa.FRAMES_DIR / f"frame_{n:06d}.jpg").exists()]
    target_frames = [n - 1 for n in target_imgs]
    if not target_frames:
        raise SystemExit("対象フレームがありません")

    raw_by_frame = {fr: raw_players(boxes.get(fr, []), H) for fr in target_frames}

    segs = split_segments(target_frames)

    # 安定ID（players_hi.csv）があればそれを使う＝再トラッキングしない → ブレ激減
    stable = None if args.no_ids else load_players_hi()
    print("ID元:", "players_hi.csv（安定ID）" if stable else "毎回トラッキング", flush=True)

    # セグメントごとに: トラッキング → 補間/平滑化 → 攻撃方向 → ライン/オフサイド(時間平滑)
    frame_data = {}   # fr -> {players, lines, dirs, offside, seg_idx}
    for si, seg in enumerate(segs):
        if stable is not None:
            tracks = tracks_for_segment(stable, seg, min_len=args.min_track)
        else:
            tracks = build_tracks(seg, raw_by_frame)
        fill_and_smooth(tracks, alpha=args.smooth)
        dirs = detect_attack_dirs(seg, tracks)
        print(f"セグメント{si+1}: 画像{seg[0]+1}〜{seg[-1]+1}  "
              f"攻撃方向 緑={'→105' if dirs[1]>0 else '→0'} "
              f"黒={'→105' if dirs[2]>0 else '→0'}  トラック{len(tracks)}", flush=True)

        # 実フレームのライン/オフサイドを計算
        all_frames = sorted({f for t in tracks.values() for f in t["pos"]})
        raw_lines = {1: {}, 2: {}}
        raw_offside_by_frame = {}
        for fr in all_frames:
            ps = players_at(tracks, fr)
            lines, off = compute_lines_offside(ps, dirs, args.offside_margin)
            for dfn in (1, 2):
                raw_lines[dfn][fr] = lines.get(dfn)
            raw_offside_by_frame[fr] = off
        # ライン時間平滑（移動平均）
        sm_lines = {dfn: smooth_series(raw_lines[dfn], all_frames, win=1) for dfn in (1, 2)}
        # オフサイド時間平滑（トラックごと多数決）
        off_frames_of = defaultdict(list)
        for fr in all_frames:
            for tid in raw_offside_by_frame[fr]:
                off_frames_of[tid].append(fr)
        sm_offside_by_frame = {fr: set() for fr in all_frames}
        present_of = defaultdict(list)
        for tid, t in tracks.items():
            present_of[tid] = sorted(t["pos"])
        for tid, present in present_of.items():
            offset = set(off_frames_of.get(tid, []))
            if not offset:
                continue
            for i, fr in enumerate(present):
                lo = max(0, i - args.offside_win)
                hi = min(len(present), i + args.offside_win + 1)
                window = present[lo:hi]
                cnt = sum(1 for f in window if f in offset)
                if cnt * 2 > len(window):   # 多数決
                    sm_offside_by_frame[fr].add(tid)

        for fr in seg:
            ps = players_at(tracks, fr)
            lines = {dfn: sm_lines[dfn].get(fr) for dfn in (1, 2)
                     if sm_lines[dfn].get(fr) is not None}
            frame_data[fr] = {"players": ps, "lines": lines, "dirs": dirs,
                              "offside": sm_offside_by_frame.get(fr, set()),
                              "seg_idx": si}

    # ---- レンダリング（中割り挿入）----
    total_h = CAM_H + PANEL_H
    total_h += total_h % 2
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (CANVAS_W, total_h))

    n_off_frames = 0
    written = 0
    for idx, fr in enumerate(target_frames):
        a = frame_data[fr]
        nb = target_frames[idx + 1] if idx + 1 < len(target_frames) else None
        interp_ok = nb is not None and nb == fr + 1  # 同一セグメント連続のみ中割り
        b = frame_data.get(nb) if interp_ok else None
        reps = K if interp_ok else 1

        img_no = fr + 1
        cam_base = cv2.imread(str(pa.FRAMES_DIR / f"frame_{img_no:06d}.jpg"))
        if cam_base is None:
            continue
        # トラックを tid で対応付け（中割り用）
        a_by = {p["tid"]: p for p in a["players"]}
        b_by = {p["tid"]: p for p in b["players"]} if b else {}

        if a["offside"]:
            n_off_frames += 1

        for k in range(reps):
            tt = k / K if interp_ok else 0.0
            # 選手位置を中割り
            players = []
            tids = set(a_by) | (set(b_by) if interp_ok else set())
            for tid in tids:
                pa_ = a_by.get(tid); pb_ = b_by.get(tid)
                if pa_ and pb_:
                    X = lerp(pa_["X"], pb_["X"], tt); Y = lerp(pa_["Y"], pb_["Y"], tt)
                    box = tuple(lerp(pa_["box"][i], pb_["box"][i], tt) for i in range(4))
                    team = pa_["team"]
                elif pa_:
                    X, Y, box, team = pa_["X"], pa_["Y"], pa_["box"], pa_["team"]
                else:
                    continue  # b のみの新規はこの区間では出さない（次区間で登場）
                players.append({"tid": tid, "team": team, "X": X, "Y": Y, "box": box})
            # ライン中割り
            lines = {}
            for dfn in (1, 2):
                la = a["lines"].get(dfn)
                lb = b["lines"].get(dfn) if b else None
                if la is not None and lb is not None and interp_ok:
                    lines[dfn] = lerp(la, lb, tt)
                elif la is not None:
                    lines[dfn] = la
            offside = a["offside"]  # 区間内は固定（境界で切替）

            cam = cam_base.copy()
            draw_camera(cam, players, offside, dscale)
            cam = cv2.resize(cam, (CANVAS_W, CAM_H))
            t = (fr + (tt if interp_ok else 0)) / eff_fps
            n_off = sum(1 for p in players if p["tid"] in offside)
            seg_label = (f"セグメント{a['seg_idx']+1}  画像{img_no}  t={t:.1f}s  "
                         f"選手{len(players)}  オフサイド{n_off}人")
            panel = draw_pitch_panel(players, lines, a["dirs"], offside, seg_label)
            canvas = np.zeros((total_h, CANVAS_W, 3), dtype=np.uint8)
            canvas[:CAM_H] = cam
            canvas[CAM_H:CAM_H + PANEL_H] = panel
            writer.write(canvas)
            written += 1

        if (idx + 1) % 100 == 0 or idx + 1 == len(target_frames):
            print(f"  {idx+1}/{len(target_frames)} 実フレーム（出力{written}枚）", flush=True)

    writer.release()
    print(f"\n完了: {args.out}", flush=True)
    print(f"  実{len(target_frames)}フレーム → 中割りx{K} = {written}枚 / {args.fps}fps "
          f"（{written/args.fps:.0f}秒, オフサイド検出 {n_off_frames}実フレーム）", flush=True)


if __name__ == "__main__":
    main()
