#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import llm_client


# Global counter for ChatGPT API calls (counts each request attempt)
gpt_call_count = 0


def _is_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def ensure_local_video(
    video_input: str,
    download_dir: str,
    *,
    cookies_from_browser: Optional[Union[str, Tuple[str, Optional[str], Optional[str], Optional[str]]]] = None,
    cookies_file: Optional[str] = None,
) -> str:
    """Return a local video path, downloading remote URLs when needed."""
    if _is_url(video_input):
        print(f"[download] Detected URL input → downloading via yt-dlp", flush=True)
        os.makedirs(download_dir, exist_ok=True)
        browser_spec = cookies_from_browser
        if isinstance(browser_spec, str):
            browser_spec = (browser_spec, None, None, None)
        local_path = download_youtube_video(
            video_input,
            download_dir,
            cookies_from_browser=browser_spec,
            cookies_file=cookies_file,
        )
        if not local_path or not os.path.exists(local_path):
            raise RuntimeError("Download reported success but file is missing")
        print(f"[download] Saved to {local_path}", flush=True)
        return local_path
    if not os.path.exists(video_input):
        raise FileNotFoundError(f"Video not found: {video_input}")
    return video_input


def ensure_package(pkg: str, import_name: Optional[str] = None) -> None:
    """Ensure a Python package is installed."""
    name = import_name or pkg
    try:
        __import__(name)
        return
    except Exception:
        pass
    print(f"[setup] Installing {pkg} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def ensure_dependencies() -> None:
    ensure_package("yt-dlp", "yt_dlp")
    ensure_package("opencv-python", "cv2")
    ensure_package("numpy")
    ensure_package("openai")
    ensure_package("python-dotenv", "dotenv")
    ensure_package("moviepy")


def download_youtube_video(
    url: str,
    out_dir: str,
    cookies_from_browser: Optional[Union[str, Tuple[str, Optional[str], Optional[str], Optional[str]]]] = None,
    cookies_file: Optional[str] = None,
) -> str:
    from yt_dlp import YoutubeDL

    os.makedirs(out_dir, exist_ok=True)
    preferred_format = (
        "bestvideo[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]"
        "+bestaudio[ext=m4a][protocol!=m3u8][protocol!=m3u8_native]"
        "/best[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]"
        "/bestvideo+bestaudio/best"
    )
    ydl_opts = {
        "format": preferred_format,
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noprogress": False,
        "quiet": False,
        "nocheckcertificate": True,
        "retries": 10,
        "fragment_retries": 5,
        "skip_unavailable_fragments": True,
        "concurrent_fragment_downloads": 1,
        "forceipv4": True,
        "geo_bypass": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    if cookies_from_browser:
        if isinstance(cookies_from_browser, (list, tuple)):
            spec: List[Optional[str]] = list(cookies_from_browser)[:4]
            while len(spec) < 4:
                spec.append(None)
            ydl_opts["cookiesfrombrowser"] = tuple(spec)
        else:
            ydl_opts["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    attempt_specs = [
        {"format": preferred_format, "extractor_args": {"youtube": {"player_client": ["web"]}}},
        {"format": preferred_format, "extractor_args": {"youtube": {"player_client": ["android"]}}},
        {"format": preferred_format, "extractor_args": {"youtube": {"player_client": ["ios"]}}},
        {"format": "22/18/best[ext=mp4]/best", "extractor_args": {"youtube": {"player_client": ["web"]}}},
        {"format": "bestvideo+bestaudio/best", "extractor_args": {"youtube": {"player_client": ["web"]}}},
    ]

    last_err: Optional[Exception] = None
    for spec in attempt_specs:
        opts = dict(ydl_opts)
        opts.update(spec)
        with YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                if "requested_downloads" in info and info["requested_downloads"]:
                    path = info["requested_downloads"][0].get("_filename")
                else:
                    title = info.get("title", "video")
                    ext = info.get("ext", "mp4")
                    path = os.path.join(out_dir, f"{title}.{ext}")
                if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                    raise RuntimeError("Downloaded file missing or empty after attempt")
                if not path.lower().endswith(".mp4"):
                    base, _ = os.path.splitext(path)
                    mp4_path = base + ".mp4"
                    if os.path.exists(mp4_path):
                        path = mp4_path
                return path
            except Exception as err:
                last_err = err
                continue

    raise RuntimeError(
        "Failed to download video due to HTTP 403/NSIG or format access. "
        "Try: --cookies-from-browser safari (or chrome), or --cookies <file>, or manually download and pass --video-path."
    ) from last_err


def sample_frame_from_video(video_path: str, ts_sec: float) -> Optional[object]:
    try:
        import cv2
        import numpy as np  # noqa: F401
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try_times = [max(0.0, ts_sec), max(0.0, ts_sec - 0.5), max(0.0, ts_sec + 0.5), 0.0]
    frame = None
    for t in try_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
            break
    cap.release()
    return frame


def parse_quarter(text: str) -> Optional[str]:
    t = text.lower()
    patterns = [
        r"\bq\s*([1-4])\b",
        r"\b([1-4])\s*quarter\b",
        r"\b(1st|2nd|3rd|4th)\b",
        r"\bqtr\s*([1-4])\b",
        r"\bot\b",
    ]
    for pat in patterns:
        m = re.search(pat, t)
        if m:
            if m.group(0) == "ot" or (m.groups() and m.groups()[0] and m.groups()[0].lower() == "ot"):
                return "OT"
            g = m.groups()[0] if m.groups() else m.group(0)
            g = str(g).lower()
            mapping = {"1st": "Q1", "2nd": "Q2", "3rd": "Q3", "4th": "Q4"}
            if g in mapping:
                return mapping[g]
            if g in {"1", "2", "3", "4"}:
                return f"Q{g}"
            if g == "ot":
                return "OT"
            if g.startswith("q") and len(g) == 2 and g[1] in "1234":
                return g.upper()
    return None


def parse_game_clock(text: str) -> Optional[Tuple[str, int]]:
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    if ss >= 60 or mm > 59:
        return None
    return (f"{mm:02d}:{ss:02d}", mm * 60 + ss)


def quarter_order(quarter: str) -> int:
    if quarter == "OT":
        return 5
    if quarter and quarter.startswith("Q") and len(quarter) == 2 and quarter[1] in "1234":
        return int(quarter[1])
    return 0


def quarter_duration_seconds(quarter: str) -> int:
    if quarter == "OT":
        return 10 * 60
    return 15 * 60


def compute_game_elapsed(quarter: Optional[str], game_clock_sec: Optional[int]) -> Optional[int]:
    if not quarter or game_clock_sec is None:
        return None
    q_idx = quarter_order(quarter)
    if q_idx == 0:
        return None
    elapsed_prior = 0
    for i in range(1, q_idx):
        elapsed_prior += quarter_duration_seconds(f"Q{i}")
    elapsed_this = quarter_duration_seconds(quarter) - game_clock_sec
    return elapsed_prior + max(0, elapsed_this)


def parse_user_game_time(s: str) -> Tuple[str, int]:
    s_norm = s.strip().replace("\u2013", "-")
    qm = re.search(r"\b(Q[1-4]|[1234](?:st|nd|rd|th)|OT)\b", s_norm, flags=re.IGNORECASE)
    tm = re.search(r"\b(\d{1,2}):(\d{2})\b", s_norm)
    if not qm or not tm:
        raise ValueError("Invalid game time format. Use e.g. 'Q2 05:23' or '2nd 5:23'.")
    q_raw = qm.group(1)
    q = parse_quarter(q_raw) or q_raw.upper()
    mm, ss = int(tm.group(1)), int(tm.group(2))
    if ss >= 60 or mm > 59:
        raise ValueError("Invalid clock time; expected MM:SS with SS<60 and MM<=59")
    return q, mm * 60 + ss


def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def trim_video_ffmpeg(
    input_path: str,
    start_sec: float,
    output_path: str,
    duration: Optional[float] = None,
    reencode: bool = False,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", input_path]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    if reencode:
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
        ]
    else:
        cmd += ["-c", "copy"]
    cmd += [output_path]
    subprocess.check_call(cmd)
    return output_path


def trim_video_moviepy(
    input_path: str,
    start_sec: float,
    output_path: str,
    duration: Optional[float] = None,
) -> str:
    from moviepy.editor import VideoFileClip

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with VideoFileClip(input_path) as clip:
        end = clip.duration if duration is None else min(clip.duration, start_sec + duration)
        sub = clip.subclip(start_sec, end)
        sub.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path


def get_video_duration_seconds(video_path: str) -> float:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("OpenCV not available; cannot read video duration") from e

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    if frame_count and fps:
        duration = float(frame_count) / float(fps)
    else:
        # Fallback: seek to end is not reliable; iterate
        duration = 0.0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            duration += 1.0 / (fps or 30.0)
    cap.release()
    duration = max(0.0, duration)
    print(f"[video] Duration detected: {duration:.2f}s", flush=True)
    return duration


def gpt_read_quarter_clock(frame_bgr) -> Tuple[Optional[str], Optional[str]]:
    """Return (quarter, clock_str) via a single GPT vision call.
    quarter in {Q1,Q2,Q3,Q4,OT} or None; clock_str as MM:SS or None.
    """
    # Crop to bottom 20% of the frame before sending to the model
    def _crop_bottom_20_percent(src_bgr):
        height, width = src_bgr.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Cannot crop an empty frame")
        top_row_index = int(0.8 * height)
        # Ensure at least 1 row is included
        top_row_index = min(max(0, top_row_index), height - 1)
        return src_bgr[top_row_index:height, 0:width]

    cropped_bgr = _crop_bottom_20_percent(frame_bgr)
    user_prompt = (
        "You are reading the broadcast scoreboard from an NFL TV frame. "
        "If visible and readable, return ONLY this JSON with no extra text: "
        '{"quarter":"Q1|Q2|Q3|Q4|OT or null","clock":"MM:SS or null"}. '
        "If the scoreboard/clock is missing, occluded, or unreadable, return: "
        '{"quarter":null,"clock":null}. '
        "Do not add commentary."
    )
    # Call out to the LLM client
    def _inc_attempt():
        global gpt_call_count
        gpt_call_count += 1

    text_response = llm_client.vision_ask(
        cropped_bgr,
        user_prompt,
        on_attempt=_inc_attempt,
        model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
    )

    # Parse strict JSON if present; else try regex fallback
    quarter: Optional[str] = None
    clock: Optional[str] = None
    if text_response:
        try:
            # Remove any surrounding backticks or code fences
            tr = text_response.strip()
            if tr.startswith("```"):
                tr = tr.strip("`\n ")
                # Might still include a language tag
                tr = re.sub(r"^[a-zA-Z]+\n", "", tr, count=1)
            data = json.loads(tr)
            q_raw = data.get("quarter")
            c_raw = data.get("clock")
            if isinstance(q_raw, str):
                quarter = q_raw.strip().upper()
                # Normalize forms like 1ST, 2ND etc.
                m = re.match(r"^(Q[1-4]|OT)$", quarter)
                if not m:
                    mapping = {"1ST": "Q1", "2ND": "Q2", "3RD": "Q3", "4TH": "Q4"}
                    quarter = mapping.get(quarter, None)
            if isinstance(c_raw, str):
                m = re.match(r"^(\d{1,2}):(\d{2})$", c_raw.strip())
                if m and int(m.group(2)) < 60:
                    mm = int(m.group(1))
                    ss = int(m.group(2))
                    clock = f"{mm:02d}:{ss:02d}"
        except Exception:
            # Regex fallback if model didn't return clean JSON
            qm = re.search(r"\b(Q[1-4]|OT)\b", text_response.upper())
            if qm:
                quarter = qm.group(1)
            tm = re.search(r"\b(\d{1,2}):(\d{2})\b", text_response)
            if tm and int(tm.group(2)) < 60:
                mm = int(tm.group(1))
                ss = int(tm.group(2))
                clock = f"{mm:02d}:{ss:02d}"

    print(f"[vision] Read → quarter={quarter}, clock={clock}", flush=True)

    return quarter, clock


class FrameReadingCache:
    def __init__(self) -> None:
        self._cache: Dict[int, Tuple[Optional[str], Optional[str]]] = {}

    @staticmethod
    def _key(ts: float) -> int:
        # Quantize timestamp to 40ms buckets to dedupe near-identical reads
        return int(round(ts * 25.0))

    def get(self, ts: float) -> Optional[Tuple[Optional[str], Optional[str]]]:
        return self._cache.get(self._key(ts))

    def put(self, ts: float, value: Tuple[Optional[str], Optional[str]]) -> None:
        self._cache[self._key(ts)] = value


def find_nearest_reading(
    video_path: str,
    ts_sec: float,
    scan_step: float,
    max_window: float,
    cache: FrameReadingCache,
) -> Optional[Tuple[float, Optional[str], Optional[str]]]:
    """Search left/right from ts_sec to find a frame where quarter/clock are readable.
    Returns (found_ts, quarter, clock) or None if none within window.
    """
    # Alternate right, left, right, left ... starting at ts
    directions = [0.0]
    k = 1
    while k * scan_step <= max_window + 1e-6 and k < 1000:
        directions.append(+k * scan_step)
        directions.append(-k * scan_step)
        k += 1

    for dt in directions:
        t = max(0.0, ts_sec + dt)
        cached = cache.get(t)
        print(
            f"[probe] t={t:.2f}s (cached={'yes' if cached is not None else 'no'})",
            flush=True,
        )
        if cached is None:
            frame = sample_frame_from_video(video_path, t)
            if frame is None:
                print("[probe]   no frame (decoder miss)", flush=True)
                continue
            qc = gpt_read_quarter_clock(frame)
            cache.put(t, qc)
        else:
            qc = cached

        q, c = qc
        print(f"[probe]   result: quarter={q}, clock={c}", flush=True)
        if q and c:
            return (t, q, c)
    return None


def binary_search_game_time(
    video_path: str,
    target_quarter: str,
    target_clock_str: str,
    *,
    scan_step: float = 0.5,
    max_scan_window: float = 3.0,
    max_iterations: int = 12,
) -> float:
    """Return approximate video timestamp (seconds) corresponding to target game time.
    Uses GPT vision to read scorebug and binary search over the video timeline.
    """
    duration = get_video_duration_seconds(video_path)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Could not determine video duration or duration is zero.")

    target_clock_mmss = target_clock_str
    # Normalize target clock
    m = re.match(r"^(\d{1,2}):(\d{2})$", target_clock_mmss.strip())
    if not m or int(m.group(2)) >= 60:
        raise ValueError("Target clock must be MM:SS with SS<60")
    target_clock_mmss = f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"

    # Convert target to absolute game elapsed seconds
    _, target_clock_sec = (target_clock_mmss, int(m.group(1)) * 60 + int(m.group(2)))
    target_elapsed = compute_game_elapsed(target_quarter, target_clock_sec)
    if target_elapsed is None:
        raise ValueError("Invalid target quarter/clock combination")

    low = 0.0
    high = duration
    cache = FrameReadingCache()

    # Bootstrap: find readings near the ends to seed direction
    print("[bootstrap] Probing near start/end...", flush=True)
    left_read = find_nearest_reading(
        video_path, low, scan_step=scan_step, max_window=max_scan_window, cache=cache
    )
    right_read = find_nearest_reading(
        video_path, high, scan_step=scan_step, max_window=max_scan_window, cache=cache
    )

    def elapsed_from(q: Optional[str], mmss: Optional[str]) -> Optional[int]:
        if not q or not mmss:
            return None
        tm = re.match(r"^(\d{2}):(\d{2})$", mmss)
        if not tm:
            return None
        sec = int(tm.group(1)) * 60 + int(tm.group(2))
        return compute_game_elapsed(q, sec)

    # Binary search loop
    for i in range(max_iterations):
        mid = (low + high) / 2.0
        print(
            f"[bisect] iter={i} low={low:.2f}s high={high:.2f}s mid={mid:.2f}s",
            flush=True,
        )
        mid_read = find_nearest_reading(
            video_path, mid, scan_step=scan_step, max_window=max_scan_window, cache=cache
        )

        if mid_read is None:
            print("[bisect]   mid unreadable; adjusting bounds", flush=True)
            # No readable frame around mid; shrink towards the side where we do have a reading
            if left_read and not right_read:
                high = mid
                continue
            if right_read and not left_read:
                low = mid
                continue
            # If neither side has readings yet, expand scan window slightly and retry once
            max_scan_window = min(8.0, max_scan_window + scan_step)
            print(f"[bisect]   expanding scan window to ±{max_scan_window:.2f}s", flush=True)
            continue

        mid_ts, mid_q, mid_clock = mid_read
        mid_elapsed = elapsed_from(mid_q, mid_clock)
        if mid_elapsed is None:
            # Treat as unreadable and continue
            # Also slightly increase search window to escape sparse regions
            max_scan_window = min(8.0, max_scan_window + scan_step)
            print("[bisect]   mid had no elapsed; expanding window", flush=True)
            continue

        if mid_elapsed == target_elapsed:
            print(
                f"[bisect]   exact match at {mid_ts:.2f}s → {mid_q} {mid_clock}",
                flush=True,
            )
            return max(0.0, mid_ts)
        if mid_elapsed < target_elapsed:
            print(
                f"[bisect]   mid {mid_q} {mid_clock} < target; move low → {mid_ts:.2f}s",
                flush=True,
            )
            low = mid_ts
            left_read = mid_read
        else:
            print(
                f"[bisect]   mid {mid_q} {mid_clock} > target; move high → {mid_ts:.2f}s",
                flush=True,
            )
            high = mid_ts
            right_read = mid_read

        if (high - low) <= max(scan_step, 0.35):
            print(
                f"[bisect]   interval narrowed to {(high - low):.2f}s → stopping",
                flush=True,
            )
            break

    # Final local scan around the narrowed region to find exact MM:SS match if possible
    center = (low + high) / 2.0
    print(f"[final] Local scan around {center:.2f}s", flush=True)
    final = find_nearest_reading(
        video_path, center, scan_step=0.25, max_window=2.0, cache=cache
    )
    if final:
        ts, fq, fclock = final
        if fq == target_quarter and fclock == target_clock_mmss:
            print("[final] Exact quarter/clock match in local scan", flush=True)
            return max(0.0, ts)

    # Otherwise, choose whichever side is closer in elapsed time
    candidates = [x for x in [left_read, right_read, final] if x is not None]
    best_ts = center
    best_delta = float("inf")
    for ts, fq, fclock in candidates:
        e = elapsed_from(fq, fclock)
        if e is None:
            continue
        d = abs(e - target_elapsed)
        if d < best_delta:
            best_delta = d
            best_ts = ts
    print(f"[final] Chose nearest by elapsed at {best_ts:.2f}s", flush=True)
    return max(0.0, best_ts)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find the video timestamp that matches a target NFL game time (quarter/MM:SS) "
            "via binary search using ChatGPT vision, then output a 5s clip."
        )
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Local video file path or HTTP/YouTube URL",
    )
    parser.add_argument(
        "--game-time",
        required=True,
        help='Target like "Q1 14:12" or "2nd 14:12"',
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Directory to place the output clip"
    )
    parser.add_argument(
        "--clip-duration", type=float, default=5.0, help="Duration of output clip in seconds"
    )
    parser.add_argument(
        "--scan-step",
        type=float,
        default=10.0,
        help="Linear scan step (seconds) when a frame lacks readable timestamp",
    )
    parser.add_argument(
        "--scan-window",
        type=float,
        default=30.0,
        help="Maximum +/- window (seconds) to search around a probe",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=12,
        help="Maximum binary search iterations",
    )
    parser.add_argument(
        "--cookies-from-browser",
        type=str,
        choices=["chrome", "firefox", "safari", "edge"],
        default=None,
        help="Pass cookies from your browser for age/region gated videos",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=None,
        help="Path to a Netscape cookie file for download authentication",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="Directory to place downloaded videos (default: output dir)",
    )

    args = parser.parse_args()

    ensure_dependencies()

    os.makedirs(args.output_dir, exist_ok=True)

    download_dir = args.download_dir or args.output_dir
    cookies_browser = args.cookies_from_browser.strip() if args.cookies_from_browser else None
    if cookies_browser:
        browser_spec = (cookies_browser, None, None, None)
    else:
        browser_spec = None

    video_path = ensure_local_video(
        args.video_path,
        download_dir,
        cookies_from_browser=browser_spec,
        cookies_file=args.cookies,
    )

    # Parse target game time
    tq, tsec = parse_user_game_time(args.game_time)
    target_quarter = tq
    target_clock_str = f"{tsec // 60:02d}:{tsec % 60:02d}"

    print(
        f"Searching for {target_quarter} {target_clock_str} within video: {video_path}",
        flush=True,
    )

    start_ts = binary_search_game_time(
        video_path=video_path,
        target_quarter=target_quarter,
        target_clock_str=target_clock_str,
        scan_step=float(args.scan_step),
        max_scan_window=float(args.scan_window),
        max_iterations=int(args.max_iters),
    )

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_clip = os.path.join(
        args.output_dir,
        f"{base_name}_{target_quarter}_{target_clock_str.replace(':', '-')}_5s.mp4",
    )

    print(f"Trimming 5s starting at {start_ts:.2f}s → {out_clip}", flush=True)
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

    print(f"Done. Output: {out_clip}")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    # Final execution statistics
    print(f"ChatGPT calls: {gpt_call_count}")
    print(f"Time taken: {end - start:.2f} seconds")


