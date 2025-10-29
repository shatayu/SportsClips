#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import sys
import subprocess
import io
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import llm_client

# ------------------------------
# Utilities: dynamic dependency install
# ------------------------------


def ensure_package(pkg: str, import_name: Optional[str] = None) -> None:
    """Ensure a Python package is available; install via pip if missing."""
    name = import_name or pkg
    try:
        __import__(name)
        return
    except Exception:
        pass
    print(f"[setup] Installing {pkg} ...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


def ensure_dependencies() -> None:
    # yt-dlp for downloading YouTube videos
    ensure_package("yt-dlp", "yt_dlp")
    # OpenCV for frame extraction
    ensure_package("opencv-python", "cv2")
    # EasyOCR for OCR
    ensure_package("easyocr")
    # numpy is a dependency for image processing
    ensure_package("numpy")
    # OpenAI for ROI inference
    ensure_package("openai")
    # dotenv to load API key from .env
    ensure_package("python-dotenv", "dotenv")


# ------------------------------
# Core data structures
# ------------------------------


@dataclass
class FrameIndexEntry:
    frame_index: int
    video_time_sec: float
    quarter: Optional[str]
    game_clock_str: Optional[str]
    game_clock_sec: Optional[int]
    game_elapsed_sec: Optional[int]
    ocr_text: str
    frame_path: Optional[str]


# ------------------------------
# YouTube download
# ------------------------------


def download_youtube_video(
    url: str,
    out_dir: str,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
) -> str:
    from yt_dlp import YoutubeDL

    os.makedirs(out_dir, exist_ok=True)
    # Prefer non-HLS MP4 to avoid m3u8 fragment issues; fall back later if needed
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
        # Accept common names: chrome, firefox, safari, edge
        ydl_opts["cookiesfrombrowser"] = cookies_from_browser
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
    # Attempt sequence: try multiple player clients, then progressive MP4, then best
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
                # Determine final file path
                if "requested_downloads" in info and info["requested_downloads"]:
                    path = info["requested_downloads"][0].get("_filename")
                else:
                    title = info.get("title", "video")
                    ext = info.get("ext", "mp4")
                    path = os.path.join(out_dir, f"{title}.{ext}")
                # Verify file exists and non-empty
                if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                    raise RuntimeError("Downloaded file missing or empty after attempt")
                # Normalize extension to .mp4 if merger happened
                if not path.lower().endswith(".mp4"):
                    base, _ = os.path.splitext(path)
                    mp4_path = base + ".mp4"
                    if os.path.exists(mp4_path):
                        path = mp4_path
                return path
            except Exception as e:
                last_err = e
                continue

    # If we got here, all attempts failed
    raise RuntimeError(
        "Failed to download video due to HTTP 403/NSIG or format access. "
        "Try: --cookies-from-browser safari (or chrome), or --cookies <file>, or manually download and pass --video-path."
    )

    # Normalize extension to .mp4 if merger happened
    if not path.lower().endswith(".mp4"):
        base, _ = os.path.splitext(path)
        mp4_path = base + ".mp4"
        if os.path.exists(mp4_path):
            path = mp4_path
    return path


# ------------------------------
# Frame extraction + OCR
# ------------------------------


def build_reader():
    import easyocr

    try:
        import torch  # noqa: F401
        gpu = False
        # Attempt GPU detection but default to CPU for portability
    except Exception:
        gpu = False
    print("[ocr] Initializing EasyOCR reader (first run may download models)...", flush=True)
    return easyocr.Reader(["en"], gpu=gpu)


def load_openai_api_key_from_env() -> Optional[str]:
    """Load OPENAI_API_KEY from a .env file (if present) or environment."""
    try:
        from dotenv import load_dotenv

        # Load .env from current working directory if available
        load_dotenv()
    except Exception:
        # If dotenv isn't available or .env not present, environment may still have the key
        pass
    return os.getenv("OPENAI_API_KEY")


def sample_frame_from_video(video_path: str, ts_sec: float) -> Optional[object]:
    """Grab a single frame at the requested timestamp (ms-based seek)."""
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


def _parse_bbox_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse (a, b, c, d) from model text; return floats if 4 numbers found."""
    if not text:
        return None
    m = re.search(r"\(([^)]*)\)", text)
    raw = m.group(1) if m else text
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", raw)
    if len(nums) < 4:
        return None
    a, b, c, d = [float(x) for x in nums[:4]]
    return (a, b, c, d)


def _normalize_bbox(a: float, b: float, c: float, d: float, w: int, h: int) -> Tuple[float, float, float, float]:
    """
    Convert possibly pixel-based (a,b,c,d) into fractional x,y,w,h in [0,1].
    Heuristic: if any value > 1.0, assume pixel units.
    """
    is_pixels = any(v > 1.0 for v in (a, b, c, d))
    if is_pixels:
        fx = max(0.0, min(1.0, a / max(1, w)))
        fy = max(0.0, min(1.0, b / max(1, h)))
        fw = max(0.0, min(1.0, c / max(1, w)))
        fh = max(0.0, min(1.0, d / max(1, h)))
        return (fx, fy, fw, fh)
    # already fractional
    fx = max(0.0, min(1.0, a))
    fy = max(0.0, min(1.0, b))
    fw = max(0.0, min(1.0, c))
    fh = max(0.0, min(1.0, d))
    return (fx, fy, fw, fh)


def infer_roi_via_chatgpt(frame_bgr: object) -> Optional[Tuple[float, float, float, float]]:
    """
    Make one call to ChatGPT to infer bounding box for score+quarter.
    Returns ROI as fractional (x, y, w, h) in [0,1] or None on failure.
    """
    prompt = (
        "given this frame, return the bounding box around the score and quarter only. "
        "just return the format (a, b, c, d) and nothing else"
    )
    try:
        text_response = llm_client.vision_ask(
            frame_bgr,
            prompt,
            model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
        )
    except Exception as e2:
        print(f"[ocr] ROI inference via ChatGPT failed: {e2}")
        return None

    bbox = _parse_bbox_from_text(text_response or "")
    if not bbox:
        print("[ocr] ROI inference: could not parse bounding box from response.")
        return None

    h, w = frame_bgr.shape[:2]
    fx, fy, fw, fh = _normalize_bbox(bbox[0], bbox[1], bbox[2], bbox[3], w, h)
    print(f"[ocr] ROI inferred via ChatGPT: ({fx:.3f}, {fy:.3f}, {fw:.3f}, {fh:.3f})")
    return (fx, fy, fw, fh)


def parse_quarter(text: str) -> Optional[str]:
    t = text.lower()
    # Common representations: Q1, Q2, Q3, Q4, 1st, 2nd, 3rd, 4th, OT
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
            # fallback
            if g.startswith("q") and len(g) == 2 and g[1] in "1234":
                return g.upper()
    return None


def parse_game_clock(text: str) -> Optional[Tuple[str, int]]:
    # Find MM:SS or M:SS
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    if ss >= 60 or mm > 59:
        return None
    total_seconds = mm * 60 + ss
    return (f"{mm:02d}:{ss:02d}", total_seconds)


def quarter_order(quarter: str) -> int:
    if quarter == "OT":
        return 5
    if quarter and quarter.startswith("Q") and len(quarter) == 2 and quarter[1] in "1234":
        return int(quarter[1])
    return 0


def quarter_duration_seconds(quarter: str) -> int:
    if quarter == "OT":
        # NFL OT period commonly 10 minutes in regular season
        return 10 * 60
    return 15 * 60


def compute_game_elapsed(quarter: Optional[str], game_clock_sec: Optional[int]) -> Optional[int]:
    if not quarter or game_clock_sec is None:
        return None
    q_idx = quarter_order(quarter)
    if q_idx == 0:
        return None
    # Elapsed up to start of this quarter
    elapsed_prior = 0
    for i in range(1, q_idx):
        elapsed_prior += quarter_duration_seconds(f"Q{i}")
    elapsed_this = quarter_duration_seconds(quarter) - game_clock_sec
    return elapsed_prior + max(0, elapsed_this)


def extract_frames_and_ocr(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    roi: Optional[Tuple[float, float, float, float]] = None,
    keep_frames: bool = True,
) -> List[FrameIndexEntry]:
    import cv2
    import numpy as np

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / real_fps if total_frames > 0 else (
        cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    )

    if duration_sec is None or not math.isfinite(duration_sec) or duration_sec <= 0:
        # Fallback: sample until no more frames
        duration_sec = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            duration_sec += 1.0 / real_fps
        cap.release()
        cap = cv2.VideoCapture(video_path)

    reader = build_reader()
    entries: List[FrameIndexEntry] = []

    step = max(1.0, float(1.0 / fps) if fps > 0 else 1.0)  # seconds per sample; default 1.0
    sample_times = [t for t in frange(0.0, duration_sec, 1.0)]
    total_samples = len(sample_times)
    if total_samples == 0:
        return []

    frame_counter = 0
    for idx, t in enumerate(sample_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if not ret:
            continue

        h, w = frame.shape[:2]
        if roi is None:
            # Default: bottom 22% height, centered width
            x, y, rw, rh = 0.0, 0.78, 1.0, 0.22
        else:
            x, y, rw, rh = roi
        x0 = max(0, int(x * w))
        y0 = max(0, int(y * h))
        x1 = min(w, int((x + rw) * w))
        y1 = min(h, int((y + rh) * h))
        crop = frame[y0:y1, x0:x1]

        # Basic pre-processing to improve OCR
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # EasyOCR expects RGB
        rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
        ocr_texts = reader.readtext(rgb, detail=0)
        joined = " ".join(ocr_texts)

        q = parse_quarter(joined) or parse_quarter(joined.replace("|", "1"))
        clock_parsed = parse_game_clock(joined)
        if clock_parsed is None:
            # Sometimes colon picked up as other char; normalize common OCR errors
            normalized = re.sub(r"[;lI]", ":", joined)
            clock_parsed = parse_game_clock(normalized)

        clock_str = None
        clock_sec = None
        elapsed = None
        if clock_parsed is not None:
            clock_str, clock_sec = clock_parsed
            elapsed = compute_game_elapsed(q, clock_sec)

        frame_path = os.path.join(frames_dir, f"frame_{int(t):06d}.jpg")
        cv2.imwrite(frame_path, frame)

        entry = FrameIndexEntry(
            frame_index=frame_counter,
            video_time_sec=float(t),
            quarter=q,
            game_clock_str=clock_str,
            game_clock_sec=clock_sec,
            game_elapsed_sec=elapsed,
            ocr_text=joined,
            frame_path=frame_path if keep_frames else None,
        )
        entries.append(entry)
        frame_counter += 1

        # Progress logging every ~5% or every 50 frames, whichever is smaller
        if total_samples > 0:
            step_mod = max(1, min(50, total_samples // 20))
            if (idx + 1) % step_mod == 0 or (idx + 1) == total_samples:
                pct = ((idx + 1) / total_samples) * 100.0
                print(f"[ocr] Processed {idx + 1}/{total_samples} frames ({pct:.1f}%)", flush=True)

    cap.release()

    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return entries


def frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    t = start
    while t <= stop:
        vals.append(round(t, 3))
        t += step
    return vals


# ------------------------------
# Indexing
# ------------------------------


def save_index(entries: List[FrameIndexEntry], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "frame_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)
    return index_path


# ------------------------------
# Search for frame by video or game time
# ------------------------------


def parse_user_game_time(s: str) -> Tuple[str, int]:
    # Accept formats: "Q2 05:23", "2nd 5:23", "Q3 5:23", "OT 08:15"
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


def find_latest_frame_by_video_time(entries: List[FrameIndexEntry], ts_sec: float) -> Optional[FrameIndexEntry]:
    eligible = [e for e in entries if e.video_time_sec <= ts_sec]
    if not eligible:
        return None
    return max(eligible, key=lambda e: e.video_time_sec)


def find_latest_frame_by_game_time(entries: List[FrameIndexEntry], quarter: str, clock_sec: int) -> Optional[FrameIndexEntry]:
    target_elapsed = compute_game_elapsed(quarter, clock_sec)
    if target_elapsed is None:
        return None
    eligible = [e for e in entries if e.game_elapsed_sec is not None and e.game_elapsed_sec <= target_elapsed]
    if not eligible:
        return None
    return max(eligible, key=lambda e: e.game_elapsed_sec)  # latest before or equal


# ------------------------------
# Trim video
# ------------------------------


def ffmpeg_exists() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def trim_video_ffmpeg(input_path: str, start_sec: float, output_path: str, duration: Optional[float] = None, reencode: bool = False) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-i", input_path]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy"]
    cmd += [output_path]
    subprocess.check_call(cmd)
    return output_path


def trim_video_moviepy(input_path: str, start_sec: float, output_path: str, duration: Optional[float] = None) -> str:
    from moviepy.editor import VideoFileClip

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with VideoFileClip(input_path) as clip:
        end = clip.duration if duration is None else min(clip.duration, start_sec + duration)
        sub = clip.subclip(start_sec, end)
        sub.write_videofile(output_path, codec="libx264", audio_codec="aac")
    return output_path


# ------------------------------
# Main CLI
# ------------------------------


def main():
    parser = argparse.ArgumentParser(description="Download NFL video, OCR frames for quarter/clock, index, and trim from the latest frame before a given timestamp +10s.")
    parser.add_argument("youtube_url", help="YouTube URL of the NFL highlight video")
    parser.add_argument("timestamp", help="Video timestamp to anchor from (seconds or MM:SS)")
    parser.add_argument("--output-dir", default="./output", help="Output directory for downloads and artifacts")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample for OCR (default 1 FPS)")
    parser.add_argument("--roi", type=str, default=None, help="ROI as x,y,w,h in 0..1 fractions (default bottom strip)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames on disk")
    parser.add_argument("--clip-duration", type=float, default=5.0, help="Clip duration in seconds (default: 5 seconds)")
    parser.add_argument("--reencode", action="store_true", help="Force re-encode when trimming (more accurate, slower)")
    parser.add_argument("--video-path", type=str, default=None, help="If provided, skip download and use this file path")
    parser.add_argument("--cookies-from-browser", type=str, choices=["chrome", "firefox", "safari", "edge"], default=None, help="Pass cookies from your browser for age/region-gated videos")
    parser.add_argument("--cookies", type=str, default=None, help="Path to a Netscape cookie file for YouTube")

    args = parser.parse_args()

    # Require a timestamp (positional). Accept seconds or MM:SS
    def parse_video_timestamp(ts: str) -> float:
        m = re.match(r"^(\d{1,4}):(\d{2})$", ts.strip())
        if m:
            mm = int(m.group(1))
            ss = int(m.group(2))
            if ss >= 60:
                raise ValueError("Invalid timestamp MM:SS; SS must be < 60")
            return float(mm * 60 + ss)
        try:
            return float(ts)
        except Exception as e:
            raise ValueError("Timestamp must be seconds or MM:SS") from e

    target_ts_sec = parse_video_timestamp(args.timestamp)

    ensure_dependencies()

    roi_tuple: Optional[Tuple[float, float, float, float]] = None
    if args.roi:
        try:
            parts = [float(x.strip()) for x in args.roi.split(",")]
            if len(parts) != 4:
                raise ValueError
            roi_tuple = tuple(parts)  # type: ignore
        except Exception:
            parser.error("--roi must be four comma-separated floats: x,y,w,h in 0..1")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Download
    if args.video_path:
        video_path = args.video_path
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Provided --video-path not found: {video_path}")
    else:
        print("[1/5] Downloading video...", flush=True)
        video_path = download_youtube_video(
            args.youtube_url,
            args.output_dir,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
        )
        print(f"Downloaded to: {video_path}")
        # Validate file isn't empty
        try:
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                raise RuntimeError("Downloaded file is empty. Try providing --cookies-from-browser safari (or chrome) or --cookies <file>.")
        except OSError:
            raise RuntimeError("Downloaded file missing or unreadable. Try passing browser cookies or use --video-path.")

    # If ROI not provided, infer once via ChatGPT using a sampled frame at the user timestamp
    if roi_tuple is None:
        try:
            import numpy as np  # noqa: F401
            frame = sample_frame_from_video(video_path, target_ts_sec)
            if frame is not None:
                inferred = infer_roi_via_chatgpt(frame)
                if inferred is not None:
                    roi_tuple = inferred
        except Exception as e:
            print(f"[ocr] Skipping ROI inference due to error: {e}")

    # Step 2: Extract and index
    print("[2/4] Extracting frames at 1 FPS and running OCR...", flush=True)
    entries = extract_frames_and_ocr(
        video_path=video_path,
        output_dir=args.output_dir,
        fps=args.fps,
        roi=roi_tuple,
        keep_frames=args.keep_frames,
    )
    index_path = save_index(entries, args.output_dir)
    print(f"Index saved: {index_path} (total {len(entries)} frames)")

    # Step 3: Determine trim start (exactly at provided timestamp)
    print("[3/4] Preparing trim...", flush=True)
    start_sec = max(0.0, float(target_ts_sec))
    print(f"Trimming 5 seconds starting at {start_sec:.2f}s", flush=True)

    # Determine output path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_clip = os.path.join(args.output_dir, f"{base_name}_from_{int(start_sec)}s.mp4")

    print("[4/4] Trimming video...", flush=True)
    if ffmpeg_exists():
        trim_video_ffmpeg(
            input_path=video_path,
            start_sec=start_sec,
            output_path=out_clip,
            duration=(args.clip_duration if args.clip_duration and args.clip_duration > 0 else 5.0),
            reencode=bool(args.reencode),
        )
    else:
        ensure_package("moviepy")
        out_clip = trim_video_moviepy(
            input_path=video_path,
            start_sec=start_sec,
            output_path=out_clip,
            duration=(args.clip_duration if args.clip_duration and args.clip_duration > 0 else 5.0),
        )

    print(f"Done. Output clip: {out_clip}")


if __name__ == "__main__":
    main()


