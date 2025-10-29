#!/usr/bin/env python3
"""Trim the first N seconds (default 15) from a video, starting at 0.

Usage:
  python3 trim_first_15s.py [<input_video>] [-o OUTPUT] [-d DURATION]

If <input_video> is omitted, the script will try to auto-detect a single .mp4
in the `output/` folder relative to the current working directory.

Prefers ffmpeg for fast, lossless trimming. Falls back to moviepy if ffmpeg is
not available (re-encodes, slower).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a new video consisting of the first N seconds (default 15).",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to the input video. If omitted, auto-detect a single .mp4 in ./output/",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output video path. Default: <input_stem>_first15s<suffix> in the same directory.",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=15.0,
        help="Duration in seconds to keep from the start (default: 15)",
    )
    return parser.parse_args()


def find_single_mp4_in_output() -> Path | None:
    output_dir = Path.cwd() / "output"
    if not output_dir.exists():
        return None
    candidates = sorted(p for p in output_dir.glob("*.mp4") if p.is_file())
    if len(candidates) == 1:
        return candidates[0]
    return None


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def run_ffmpeg_copy(input_path: Path, output_path: Path, duration: float) -> None:
    # Copy streams without re-encoding; fastest when starting at t=0
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def run_ffmpeg_reencode(input_path: Path, output_path: Path, duration: float) -> None:
    # Re-encode for maximum compatibility if stream copy fails
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y",
        "-i",
        str(input_path),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def run_moviepy(input_path: Path, output_path: Path, duration: float) -> None:
    from moviepy.editor import VideoFileClip

    with VideoFileClip(str(input_path)) as clip:
        end_time = min(duration, float(clip.duration or duration))
        sub = clip.subclip(0, end_time)
        sub.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            preset="fast",
            threads=max(1, shutil.cpu_count() or 1),
            verbose=False,
            logger=None,
        )


def main() -> int:
    args = parse_args()

    input_path: Path
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
    else:
        detected = find_single_mp4_in_output()
        if detected is None:
            print(
                "Error: No input provided and could not uniquely detect a single .mp4 in ./output/.\n"
                "Provide the input path explicitly.",
                file=sys.stderr,
            )
            return 2
        input_path = detected.resolve()

    if not input_path.exists():
        print(f"Error: Input not found: {input_path}", file=sys.stderr)
        return 2

    duration = float(max(0.0, args.duration))
    if duration == 0.0:
        print("Error: Duration must be greater than 0.", file=sys.stderr)
        return 2

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        suffix = input_path.suffix or ".mp4"
        # Use integer seconds if it's an integer value, else keep as float
        dur_part = str(int(duration)) if duration.is_integer() else ("%g" % duration)
        output_basename = f"{input_path.stem}_first{dur_part}s{suffix}"
        output_path = (input_path.parent / output_basename).resolve()

    ensure_parent_dir(output_path)

    try:
        if has_ffmpeg():
            try:
                run_ffmpeg_copy(input_path, output_path, duration)
            except subprocess.CalledProcessError:
                # Fallback to re-encode if stream copy fails
                run_ffmpeg_reencode(input_path, output_path, duration)
        else:
            run_moviepy(input_path, output_path, duration)
    except FileNotFoundError as e:
        print(
            "Error: A required executable or module was not found.\n"
            "- If using ffmpeg: ensure it is installed and on your PATH.\n"
            "- If falling back to moviepy: ensure moviepy is installed.",
            file=sys.stderr,
        )
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"Failed to create trimmed video: {e}", file=sys.stderr)
        return 1

    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


