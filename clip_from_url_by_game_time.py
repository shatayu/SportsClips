#!/usr/bin/env python3
import argparse
import os
import sys
import time
from typing import Optional


# Reuse existing helpers and the GPT-vision binary search
from nfl_highlight_extractor import (
    ensure_dependencies,
    download_youtube_video,
    parse_user_game_time,
    ffmpeg_exists,
    trim_video_ffmpeg,
    trim_video_moviepy,
)

import binary_search_game_time as bsg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a video from a URL, locate a target NFL game time (e.g., 'Q1 14:12')\n"
            "using GPT vision binary search, and output a 5-second clip starting there."
        )
    )
    parser.add_argument("--video-url", required=True, help="HTTP/YouTube URL of the video")
    parser.add_argument(
        "--game-time",
        required=True,
        help="Target like 'Q1 14:12' or '2nd 14:12'",
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory to place the downloaded video and the output clip",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=5.0,
        help="Duration of output clip in seconds (default: 5)",
    )
    parser.add_argument(
        "--scan-step",
        type=float,
        default=5.0,
        help="Scan step in seconds when probing around a timestamp (default: 0.5)",
    )
    parser.add_argument(
        "--scan-window",
        type=float,
        default=30.0,
        help="Maximum +/- window in seconds to search around a probe (default: 3.0)",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=12,
        help="Maximum binary search iterations (default: 12)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        choices=["chrome", "firefox", "safari", "edge"],
        default=None,
        help="Pass cookies from your browser for age/region-gated videos",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to a Netscape cookie file (for sites like YouTube)",
    )

    args = parser.parse_args()

    t0 = time.time()

    # Ensure libraries are available (yt-dlp, cv2, openai, etc.)
    ensure_dependencies()

    os.makedirs(args.output_dir, exist_ok=True)

    # Download the video via yt-dlp (supports many sites, not just YouTube)
    print("[step] Downloading video...", flush=True)
    video_path = download_youtube_video(
        args.video_url,
        args.output_dir,
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies,
    )
    if not video_path or not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
        raise RuntimeError("Downloaded file is missing or empty")
    print(f"[done] Downloaded: {video_path}")

    # Parse target game time
    tq, tsec = parse_user_game_time(args.game_time)
    target_quarter = tq
    target_clock_str = f"{tsec // 60:02d}:{tsec % 60:02d}"

    print(
        f"Searching for {target_quarter} {target_clock_str} within video: {video_path}",
        flush=True,
    )

    # Use the existing GPT-vision binary search to locate the timestamp
    start_ts = bsg.binary_search_game_time(
        video_path=video_path,
        target_quarter=target_quarter,
        target_clock_str=target_clock_str,
        scan_step=float(args.scan_step),
        max_scan_window=float(args.scan_window),
        max_iterations=int(args.max_iters),
    )

    # Build output path for the 5s clip (or user-specified duration)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_clip = os.path.join(
        args.output_dir,
        f"{base_name}_{target_quarter}_{target_clock_str.replace(':', '-')}_5s.mp4",
    )

    print(f"Trimming {args.clip_duration:.1f}s starting at {start_ts:.2f}s → {out_clip}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if ffmpeg_exists():
        trim_video_ffmpeg(
            input_path=video_path,
            start_sec=start_ts,
            output_path=out_clip,
            duration=args.clip_duration,
            reencode=False,
        )
    else:
        out_clip = trim_video_moviepy(
            input_path=video_path,
            start_sec=start_ts,
            output_path=out_clip,
            duration=args.clip_duration,
        )

    t1 = time.time()
    total_s = t1 - t0

    # Report end-to-end runtime and GPT call count used by the binary search
    try:
        gpt_calls: Optional[int] = getattr(bsg, "gpt_call_count", None)
    except Exception:
        gpt_calls = None

    print(f"Done. Output: {out_clip}")
    if gpt_calls is not None:
        print(f"ChatGPT calls: {gpt_calls}")
    print(f"End-to-end time: {total_s:.2f} seconds")


if __name__ == "__main__":
    main()


