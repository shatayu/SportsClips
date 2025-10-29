#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import time
from typing import Dict, Optional, Tuple


# Reuse helpers from the existing script for video I/O and trimming
from nfl_highlight_extractor import (
    ensure_dependencies,
    load_openai_api_key_from_env,
    sample_frame_from_video,
    compute_game_elapsed,
    parse_user_game_time,
    ffmpeg_exists,
    trim_video_ffmpeg,
    trim_video_moviepy,
)


# Global counter for ChatGPT API calls (counts each request attempt)
gpt_call_count = 0


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


def encode_frame_to_data_url(frame_bgr) -> str:
    import base64
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def gpt_read_quarter_clock(frame_bgr) -> Tuple[Optional[str], Optional[str]]:
    """Return (quarter, clock_str) via a single GPT vision call.
    quarter in {Q1,Q2,Q3,Q4,OT} or None; clock_str as MM:SS or None.
    """
    api_key = load_openai_api_key_from_env()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment or .env; cannot use ChatGPT vision."
        )

    # Crop to bottom 20% of the frame before encoding/sending to the model
    def _crop_bottom_20_percent(src_bgr):
        height, width = src_bgr.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Cannot crop an empty frame")
        top_row_index = int(0.8 * height)
        # Ensure at least 1 row is included
        top_row_index = min(max(0, top_row_index), height - 1)
        return src_bgr[top_row_index:height, 0:width]

    cropped_bgr = _crop_bottom_20_percent(frame_bgr)
    data_url = encode_frame_to_data_url(cropped_bgr)
    user_prompt = (
        "You are reading the broadcast scoreboard from an NFL TV frame. "
        "If visible and readable, return ONLY this JSON with no extra text: "
        '{"quarter":"Q1|Q2|Q3|Q4|OT or null","clock":"MM:SS or null"}. '
        "If the scoreboard/clock is missing, occluded, or unreadable, return: "
        '{"quarter":null,"clock":null}. '
        "Do not add commentary."
    )

    content = [
        {"type": "text", "text": user_prompt},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    # Modern SDK with simple retry/backoff on rate limits (no legacy fallback)
    text_response: Optional[str] = None
    try:
        try:
            from openai import OpenAI, RateLimitError  # type: ignore
        except Exception:
            from openai import OpenAI  # type: ignore

            class RateLimitError(Exception):
                pass

        client = OpenAI(api_key=api_key, timeout=20.0)
        max_attempts = 6
        base_sleep = 5.0
        for attempt in range(1, max_attempts + 1):
            try:
                t0 = time.time()
                # Count each actual API request attempt
                global gpt_call_count
                gpt_call_count += 1
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": content}],
                    temperature=0,
                    max_tokens=40,
                )
                dt = (time.time() - t0) * 1000.0
                text_response = (resp.choices[0].message.content or "").strip()
                print(f"[vision] GPT call ok in {dt:.0f}ms", flush=True)
                break
            except RateLimitError as e:  # type: ignore
                sleep_s = min(8.0, base_sleep * (2 ** (attempt - 1)))
                print(
                    f"[vision] rate limited (attempt {attempt}/{max_attempts}); backing off {sleep_s:.2f}s",
                    flush=True,
                )
                time.sleep(sleep_s)
            except Exception as e:
                raise RuntimeError(f"ChatGPT vision call failed: {e}")
        if text_response is None:
            raise RuntimeError("ChatGPT vision call failed after retries")
    except Exception as e:
        raise

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
    parser.add_argument("--video-path", required=True, help="Local video file path")
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

    args = parser.parse_args()

    ensure_dependencies()

    video_path = args.video_path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

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


