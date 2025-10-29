#!/usr/bin/env python3
"""Download YouTube highlight videos, map each NFL game to its broadcast network,
and extract representative frames grouped by season-specific network folders."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def ensure_package(package: str, import_name: Optional[str] = None) -> None:
    """Install *package* with pip if *import_name* cannot be imported."""

    name = import_name or package
    try:
        __import__(name)
        return
    except Exception:
        pass

    print(f"[setup] Installing {package} …", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def ensure_dependencies() -> None:
    ensure_package("yt-dlp", "yt_dlp")
    ensure_package("opencv-python", "cv2")


@dataclass
class GameInfo:
    season: Optional[int]
    week_label: Optional[str]
    week_number: Optional[int]
    matchup_key: str
    matchup_key_reverse: str
    matchup_display: str
    network: str


@dataclass
class VideoSpec:
    header: str
    url: str


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def sanitize_component(value: str) -> str:
    value = normalize_text(value)
    value = value.replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    return value or "item"


def extract_network(detail: str) -> Optional[str]:
    text = detail.strip()
    # Remove leading time block (e.g., "8:20p", "9:30a")
    text = re.sub(r"^\s*\d{1,2}(?::\d{2})?[ap]\s*", "", text, flags=re.IGNORECASE)
    text = text.lstrip(".,;: ")
    if not text:
        return None
    primary = text.split(",")[0].strip()
    primary = primary.rstrip(".)")
    return primary or None


def split_matchup(raw: str) -> Optional[Tuple[str, str]]:
    text = raw.strip()
    if not text:
        return None
    fixed = text
    if " vs" not in fixed.lower() and " at" not in fixed.lower():
        # Handle stray periods between team names (typo in source file)
        fixed_period = re.sub(r"(?<=\w)\.\s+(?=[A-Z])", " vs ", fixed)
        if " vs" in fixed_period.lower() or " at" in fixed_period.lower():
            fixed = fixed_period
    parts = re.split(r"\b(?:vs\.?|at)\b", fixed, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    team1 = parts[0]
    team2 = parts[1]

    def _clean_team(name: str) -> str:
        cleaned = name.strip()
        cleaned = re.sub(r"\b\d{1,2}(?::\d{2})?\s*[ap]\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b\d{1,2}(?::\d{2})?\b", "", cleaned)
        cleaned = cleaned.strip(" .")
        return cleaned

    team1 = _clean_team(team1)
    team2 = _clean_team(team2)
    if not team1 or not team2:
        return None
    return team1, team2


def _parse_schedule_csv(schedule_path: str) -> List[GameInfo]:
    games: List[GameInfo] = []

    with open(schedule_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return games

        header_map = {((name or "").strip().lower()): name for name in reader.fieldnames}

        def pick(row: Dict[str, Optional[str]], *keys: str) -> str:
            for key in keys:
                header = header_map.get(key)
                if not header:
                    continue
                value = row.get(header)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
            return ""

        for row in reader:
            game = pick(row, "game", "matchup", "teams")
            if not game:
                continue

            network = pick(row, "network", "channel", "broadcaster")
            if not network:
                continue

            matchup = split_matchup(game)
            if not matchup:
                continue
            team1, team2 = matchup

            season_raw = pick(row, "season", "year")
            season: Optional[int] = None
            if season_raw:
                try:
                    season = int(season_raw)
                except ValueError:
                    match = re.search(r"20\d{2}", season_raw)
                    if match:
                        try:
                            season = int(match.group(0))
                        except ValueError:
                            season = None

            week_label = pick(row, "week", "week_label", "weekname")
            week_number_raw = pick(row, "week_number", "weeknum", "weekindex")
            week_number: Optional[int] = None
            for candidate in (week_label, week_number_raw):
                if not candidate:
                    continue
                match = re.search(r"\d+", candidate)
                if match:
                    try:
                        week_number = int(match.group(0))
                        break
                    except ValueError:
                        continue
            if not week_label and week_number is not None:
                week_label = f"Week {week_number}"

            display = f"{team1} vs {team2}"
            matchup_key = normalize_text(display)
            reverse_key = normalize_text(f"{team2} vs {team1}")

            games.append(
                GameInfo(
                    season=season,
                    week_label=week_label or None,
                    week_number=week_number,
                    matchup_key=matchup_key,
                    matchup_key_reverse=reverse_key,
                    matchup_display=display,
                    network=network,
                )
            )

    return games


def _parse_schedule_markdown(schedule_path: str) -> List[GameInfo]:
    games: List[GameInfo] = []
    current_season: Optional[int] = None
    current_week: Optional[str] = None
    current_week_number: Optional[int] = None

    with open(schedule_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            season_match = re.match(r"^nfl schedule\s+(\d{4})", line, flags=re.IGNORECASE)
            if season_match:
                try:
                    current_season = int(season_match.group(1))
                except ValueError:
                    current_season = None
                current_week = None
                current_week_number = None
                continue
            if line.lower().startswith("buy now"):
                continue
            if line.lower().startswith("week "):
                current_week = line
                m_week = re.search(r"week\s+(\d+)", line, flags=re.IGNORECASE)
                current_week_number = int(m_week.group(1)) if m_week else None
                continue
            if "(" not in line or ")" not in line:
                continue

            before_paren, inside_paren = line.split("(", 1)
            inside_paren = inside_paren.rsplit(")", 1)[0]

            matchup = split_matchup(before_paren)
            if not matchup:
                continue
            team1, team2 = matchup
            network = extract_network(inside_paren)
            if not network:
                continue

            display = f"{team1} vs {team2}"
            matchup_key = normalize_text(display)
            reverse_key = normalize_text(f"{team2} vs {team1}")
            games.append(
                GameInfo(
                    season=current_season,
                    week_label=current_week,
                    week_number=current_week_number,
                    matchup_key=matchup_key,
                    matchup_key_reverse=reverse_key,
                    matchup_display=display,
                    network=network,
                )
            )

    return games


def parse_schedule(schedule_path: str) -> List[GameInfo]:
    games = _parse_schedule_csv(schedule_path)
    source = "csv"
    if not games:
        games = _parse_schedule_markdown(schedule_path)
        source = "markdown"
    print(f"[schedule] Loaded {len(games)} games from {schedule_path} ({source})")
    return games


def load_video_sources(videos_path: str) -> List[VideoSpec]:
    videos: List[VideoSpec] = []
    current_header: Optional[str] = None

    with open(videos_path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                current_header = stripped.lstrip("# ") or "playlist"
                continue
            if stripped.startswith("http"):
                header = current_header or "playlist"
                videos.append(VideoSpec(header=header, url=stripped))

    print(f"[videos] Found {len(videos)} video entries in {videos_path}")
    return videos


def _apply_cookie_options(
    opts: Dict[str, object],
    cookies_from_browser: Optional[str],
    cookies_file: Optional[str],
) -> None:
    if cookies_from_browser:
        opts["cookiesfrombrowser"] = (cookies_from_browser, None, None, None)
    if cookies_file:
        opts["cookiefile"] = cookies_file


def download_video(
    url: str,
    out_dir: str,
    *,
    cookies_from_browser: Optional[str],
    cookies_file: Optional[str],
) -> Tuple[str, Dict[str, Any]]:
    from yt_dlp import YoutubeDL

    os.makedirs(out_dir, exist_ok=True)
    template = os.path.join(out_dir, "%(id)s_%(title)s.%(ext)s")
    preferred_format = (
        "bestvideo[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]"
        "+bestaudio[ext=m4a][protocol!=m3u8][protocol!=m3u8_native]"
        "/best[ext=mp4][protocol!=m3u8][protocol!=m3u8_native]"
        "/bestvideo+bestaudio/best"
    )

    base_opts: Dict[str, Any] = {
        "format": preferred_format,
        "outtmpl": template,
        "merge_output_format": "mp4",
        "quiet": False,
        "noprogress": False,
        "restrictfilenames": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "retries": 10,
        "fragment_retries": 5,
        "skip_unavailable_fragments": True,
        "concurrent_fragment_downloads": 1,
        "forceipv4": True,
        "geo_bypass": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    }
    _apply_cookie_options(base_opts, cookies_from_browser, cookies_file)

    attempt_specs = [
        {"format": preferred_format, "extractor_args": {"youtube": {"player_client": ["web"]}}},
        {"format": preferred_format, "extractor_args": {"youtube": {"player_client": ["android"]}}},
        {"format": preferred_format, "extractor_args": {"youtube": {"player_client": ["ios"]}}},
        {"format": "22/18/best[ext=mp4]/best", "extractor_args": {"youtube": {"player_client": ["web"]}}},
        {"format": "bestvideo+bestaudio/best", "extractor_args": {"youtube": {"player_client": ["web"]}}},
    ]

    last_err: Optional[Exception] = None
    for spec in attempt_specs:
        opts = dict(base_opts)
        opts.update(spec)
        with YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
                path: Optional[str]
                if "requested_downloads" in info and info["requested_downloads"]:
                    path = info["requested_downloads"][0].get("_filename")
                else:
                    path = ydl.prepare_filename(info)

                if not path:
                    raise RuntimeError("Download reported success but path was empty")
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    raise RuntimeError("Downloaded file missing or empty after attempt")
                if not path.lower().endswith(".mp4"):
                    base, _ = os.path.splitext(path)
                    mp4_path = base + ".mp4"
                    if os.path.exists(mp4_path):
                        path = mp4_path
                return path, info
            except Exception as err:
                last_err = err
                continue

    raise RuntimeError(
        "Failed to download video due to HTTP 403/NSIG or format access. "
        "Try providing --cookies-from-browser edge (or chrome/firefox/safari) "
        "or --cookies-file with a Netscape export."
    ) from last_err


def compute_week_from_title(title: str) -> Optional[int]:
    m = re.search(r"week\s*(\d{1,2})", title, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def compute_season_from_text(text: str) -> Optional[int]:
    years = re.findall(r"20\d{2}", text)
    for year_str in years:
        try:
            year = int(year_str)
        except ValueError:
            continue
        if 2000 <= year <= 2100:
            return year
    return None


def match_game(
    title: str,
    games: Sequence[GameInfo],
    *,
    header_hint: Optional[str] = None,
) -> Optional[GameInfo]:
    normalized_title = normalize_text(title)
    week_hint = compute_week_from_title(title)
    season_hint = compute_season_from_text(title)
    if season_hint is None and header_hint:
        season_hint = compute_season_from_text(header_hint)

    title_variants = {normalized_title}
    title_variants.add(normalized_title.replace(" at ", " vs "))
    title_variants.add(normalized_title.replace(" vs ", " at "))
    if header_hint:
        norm_header = normalize_text(header_hint)
        title_variants.add(norm_header)
        title_variants.add(norm_header.replace(" at ", " vs "))

    def iter_candidates(pool: Iterable[GameInfo]) -> List[GameInfo]:
        matches: List[GameInfo] = []
        for g in pool:
            keys = [g.matchup_key]
            if g.matchup_key_reverse and g.matchup_key_reverse != g.matchup_key:
                keys.append(g.matchup_key_reverse)
            if any(key and key in variant for key in keys for variant in title_variants):
                matches.append(g)
        return matches

    candidates = iter_candidates(games)
    if not candidates:
        return None

    if season_hint is not None:
        season_filtered = [g for g in candidates if g.season == season_hint]
        if len(season_filtered) == 1:
            return season_filtered[0]
        if season_filtered:
            candidates = season_filtered

    if week_hint is not None:
        week_filtered = [g for g in candidates if g.week_number == week_hint]
        if len(week_filtered) == 1:
            return week_filtered[0]
        if week_filtered:
            candidates = week_filtered

    if len(candidates) == 1:
        return candidates[0]

    # Fallback: prefer games with the longest matchup key match (more specific)
    def score(game: GameInfo) -> Tuple[int, int, int]:
        key_len = len(game.matchup_key)
        reverse_len = len(game.matchup_key_reverse)
        season_score = 1 if season_hint is not None and game.season == season_hint else 0
        return (season_score, max(key_len, reverse_len), key_len)

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def extract_frames(
    video_path: str,
    count: int,
) -> List[Tuple[int, Any]]:
    import cv2  # type: ignore

    frames: List[Tuple[int, Any]] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    duration = total_frames / fps if total_frames and fps else None

    if duration and duration > 0:
        timestamps = [duration * (i + 1) / (count + 1) for i in range(count)]
    elif total_frames > 0:
        frame_indices = [max(0, (total_frames * (i + 1)) // (count + 1)) for i in range(count)]
        timestamps = [idx / fps for idx in frame_indices]
    else:
        timestamps = [float(i) for i in range(count)]

    for idx, ts in enumerate(timestamps, start=1):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        success, frame = cap.read()
        if not success or frame is None:
            print(f"[frame] Unable to read frame at {ts:.2f}s for {video_path}")
            continue
        frames.append((idx, frame))

    cap.release()
    return frames


def save_frames(
    frames: List[Tuple[int, Any]],
    dest_dir: str,
    base_name: str,
    image_format: str,
) -> List[str]:
    import cv2  # type: ignore

    os.makedirs(dest_dir, exist_ok=True)
    saved_paths: List[str] = []
    ext = image_format.lower().lstrip(".")

    for index, frame in frames:
        filename = f"{base_name}_{index}.{ext}"
        path = os.path.join(dest_dir, filename)
        success = cv2.imwrite(path, frame)
        if not success:
            print(f"[frame] Failed to write {path}")
            continue
        saved_paths.append(path)

    return saved_paths


def process_video_entry(
    video: VideoSpec,
    games: Sequence[GameInfo],
    download_dir: str,
    output_dir: str,
    frames_per_video: int,
    image_format: str,
    keep_video: bool,
    cookies_from_browser: Optional[str],
    cookies_file: Optional[str],
) -> None:
    header_component = sanitize_component(video.header)

    print(f"[video] Downloading {video.url}")
    video_path, info = download_video(
        video.url,
        download_dir,
        cookies_from_browser=cookies_from_browser,
        cookies_file=cookies_file,
    )
    print(f"[video] Saved to {video_path}")

    title = info.get("title") if isinstance(info, dict) else None
    title = title or os.path.splitext(os.path.basename(video_path))[0]

    matched_game = match_game(title, games, header_hint=video.header)
    if not matched_game:
        print(f"[match] No schedule match for '{title}' — skipping frame extraction")
        if not keep_video and os.path.exists(video_path):
            os.remove(video_path)
        return

    video_component = sanitize_component(title)
    network_component = sanitize_component(matched_game.network)
    game_component = sanitize_component(matched_game.matchup_display)
    components = [video_component, game_component]
    base_name = "_".join([c for c in components if c]) or "frame"

    try:
        frames = extract_frames(video_path, frames_per_video)
    except Exception as exc:
        print(f"[frame] Failed to extract frames for {video_path}: {exc}")
        frames = []

    if not frames:
        print(f"[frame] No frames extracted for {title}")
    else:
        dest = os.path.join(output_dir, header_component, network_component)
        saved = save_frames(frames, dest, base_name, image_format)
        print(f"[frame] Saved {len(saved)} frames to {dest}")

    if not keep_video and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames from YouTube playlists and group them by NFL broadcast network.",
    )
    parser.add_argument(
        "--videos-file",
        default="videos.md",
        help="Markdown file listing playlist headers and URLs (default: videos.md)",
    )
    parser.add_argument(
        "--schedule-file",
        default="nfl_schedule.md",
        help="NFL schedule CSV/Markdown file used to map games to networks",
    )
    parser.add_argument(
        "--output-dir",
        default="images",
        help="Destination directory for extracted frames (default: images)",
    )
    parser.add_argument(
        "--download-dir",
        default="downloads",
        help="Directory where YouTube videos are downloaded temporarily",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=5,
        help="Number of frames to extract from each video (default: 5)",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        help="Image file format/extension for extracted frames (default: png)",
    )
    parser.add_argument(
        "--keep-video",
        action="store_true",
        help="Keep downloaded video files (default: delete after frames are saved)",
    )
    parser.add_argument(
        "--cookies-from-browser",
        choices=["chrome", "firefox", "safari", "edge"],
        default=None,
        help=(
            "Pull cookies directly from the specified browser. Useful for age/region restricted videos."
        ),
    )
    parser.add_argument(
        "--cookies-file",
        default=None,
        help="Path to a Netscape cookies.txt file to pass to yt-dlp",
    )

    args = parser.parse_args()

    ensure_dependencies()

    schedule_games = parse_schedule(args.schedule_file)
    if not schedule_games:
        raise SystemExit("No games parsed from schedule; cannot continue")

    video_entries = load_video_sources(args.videos_file)
    if not video_entries:
        raise SystemExit("No video URLs found in videos file")

    cookies_from_browser = args.cookies_from_browser
    cookies_file = args.cookies_file

    for video in video_entries:
        process_video_entry(
            video=video,
            games=schedule_games,
            download_dir=args.download_dir,
            output_dir=args.output_dir,
            frames_per_video=args.frames_per_video,
            image_format=args.image_format,
            keep_video=args.keep_video,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )

    print("[done] Frame extraction complete")


if __name__ == "__main__":
    main()

