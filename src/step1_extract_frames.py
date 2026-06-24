"""
Step 1: 動画からフレームを切り出す

OpenCVで全フレームをPython側に読み込むと時間がかかるため、実際の
切り出しはffmpegに任せる。デフォルトは2FPSで、frames/にJPGを保存する。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_VIDEO = "緑黒1試合目.MOV"
DEFAULT_OUTPUT_DIR = "frames"
DEFAULT_TARGET_FPS = 2.0
DEFAULT_OUTPUT_WIDTH = 1920


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=True)


def probe_video(video_path: Path) -> dict[str, float | int | str]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])

    fps = parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    total_frames = stream.get("nb_frames")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "total_frames": int(total_frames) if total_frames else int(duration * fps),
        "duration": duration,
    }


def parse_rate(rate: str) -> float:
    numerator, denominator = rate.split("/")
    denominator_value = float(denominator)
    if denominator_value == 0:
        return 0.0
    return float(numerator) / denominator_value


def build_metadata(
    output_dir: Path, target_fps: float, start: float
) -> list[dict[str, float | int | str]]:
    metadata = []
    for index, path in enumerate(sorted(output_dir.glob("frame_*.jpg"))):
        timestamp = start + index / target_fps
        metadata.append(
            {
                "frame_idx": index,
                "timestamp": round(timestamp, 3),
                "filename": path.name,
            }
        )
    return metadata


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float,
    quality: int,
    start: float,
    duration: float | None,
    width: int | None,
) -> None:
    pattern = output_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
    ]
    if start > 0:
        command.extend(["-ss", str(start)])
    command.extend(
        [
            "-i",
            str(video_path),
        ]
    )
    if duration is not None:
        command.extend(["-t", str(duration)])
    filters = [f"fps={target_fps}"]
    if width is not None:
        filters.append(f"scale={width}:-2")
    command.extend(
        [
            "-vf",
            ",".join(filters),
            "-q:v",
            str(quality),
            str(pattern),
        ]
    )
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="動画から一定FPSでフレームを切り出す")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="入力動画パス")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="フレーム出力先")
    parser.add_argument("--fps", type=float, default=DEFAULT_TARGET_FPS, help="切り出しFPS")
    parser.add_argument("--start", type=float, default=0.0, help="開始秒")
    parser.add_argument("--duration", type=float, default=None, help="切り出す秒数。未指定なら最後まで")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_OUTPUT_WIDTH,
        help="出力画像の横幅。0なら元解像度のまま。デフォルトは1080p相当の1920px",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="ffmpegのJPEG品質。小さいほど高品質。2-5程度がおすすめ",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="出力先に既存JPGがある場合に削除して作り直す",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"動画が見つかりません: {video_path}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpegとffprobeが必要です")

    existing_frames = list(output_dir.glob("frame_*.jpg")) if output_dir.exists() else []
    if existing_frames and not args.overwrite:
        raise RuntimeError(
            f"{output_dir} に既存フレームが {len(existing_frames)} 枚あります。"
            "作り直すなら --overwrite、別保存なら --output-dir を指定してください。"
        )

    if existing_frames:
        for path in existing_frames:
            path.unlink()
    output_dir.mkdir(exist_ok=True)

    info = probe_video(video_path)
    effective_duration = args.duration if args.duration is not None else info["duration"] - args.start
    estimated_count = int(max(0, effective_duration) * args.fps) + 1

    print("動画情報:")
    print(f"  ファイル: {video_path}")
    print(f"  解像度: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  総フレーム数: {info['total_frames']}")
    print(f"  長さ: {info['duration']:.1f}秒 ({info['duration'] / 60:.1f}分)")
    print(f"  切り出しFPS: {args.fps}")
    if args.width == 0:
        args.width = None

    if args.width is not None:
        print(f"  出力横幅: {args.width}px")
    print(f"  開始秒: {args.start}")
    if args.duration is not None:
        print(f"  切り出し秒数: {args.duration}")
    print(f"  推定保存枚数: {estimated_count}")
    print(flush=True)

    extract_frames(
        video_path,
        output_dir,
        args.fps,
        args.quality,
        args.start,
        args.duration,
        args.width,
    )

    metadata = build_metadata(output_dir, args.fps, args.start)
    metadata_path = output_dir / "frames_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(metadata)}枚のフレームを {output_dir}/ に保存しました")
    print(f"メタデータ: {metadata_path}")


if __name__ == "__main__":
    main()
