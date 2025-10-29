## NFL highlight extractor (YouTube → OCR → indexed → clipped)

This CLI tool:
- Downloads a YouTube NFL highlight video
- Samples 1 frame per second and runs OCR on the broadcast overlay
- Extracts game quarter and game clock (MM:SS) when visible
- Indexes the frames with video time and game time
- Cuts a new video starting from the latest frame before a target timestamp, plus 10 seconds

### Install

Option A: Use the script's auto-installer (no manual step). It will `pip install` missing packages when you run it.

Option B: Install dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

For faster and more accurate trimming, install `ffmpeg` (recommended). On macOS with Homebrew:

```bash
brew install ffmpeg
```

### Usage

```bash
python3 nfl_highlight_extractor.py "<YOUTUBE_URL>" 600
# or with MM:SS timestamp
python3 nfl_highlight_extractor.py "<YOUTUBE_URL>" 10:00
```

Key options:
- Required positional: `<timestamp>` in seconds or `MM:SS`
- `--clip-duration <seconds>`: optionally limit output duration (default: until end)
- `--roi x,y,w,h`: crop region for OCR in normalized fractions (default is a bottom strip like `0,0.78,1,0.22`)
- `--keep-frames`: keep the sampled frames on disk (default: removed)
- `--video-path <file>`: use an existing local video instead of downloading
- `--reencode`: force re-encode during trimming for frame-accurate cuts (slower)
- `--cookies-from-browser {safari,chrome,firefox,edge}`: use browser cookies for age/region-gated videos
- `--cookies <file>`: path to a Netscape cookie file

Outputs:
- Downloaded video in `output/`
- Frame index JSON at `output/frame_index.json`
- Trimmed video: `output/<video>_from_<start>s.mp4`

### How the timestamp anchor works

The tool finds the latest indexed frame at or before the provided video timestamp, then starts the output clip at that frame’s video time +10 seconds.

### Troubleshooting YouTube downloads

- If you see many "fragment not found" lines or "The downloaded file is empty":
  - Pass cookies from your browser (often required for some YouTube videos):
    ```bash
    python3 nfl_highlight_extractor.py "<YOUTUBE_URL>" 300 --cookies-from-browser safari
    # or
    python3 nfl_highlight_extractor.py "<YOUTUBE_URL>" 300 --cookies-from-browser chrome
    ```
  - Or export a Netscape cookie file and pass `--cookies <file>`.
  - As a workaround, download the video yourself and pass `--video-path <file>`.
  - Make sure `yt-dlp` is up to date:
    ```bash
    python3 -m pip install -U yt-dlp
    ```
  - The tool now automatically tries multiple YouTube player clients (web/android/ios) and falls back to progressive MP4 formats (itag 22/18) to reduce 403/nsig errors.

### Notes and limitations

- OCR quality depends on the broadcast overlay and video quality. If results are noisy, try adjusting `--roi` to tightly bound the scorebug/clock area.
- If `ffmpeg` is not found, the script will fall back to `moviepy` for trimming (slower and re-encodes).
- EasyOCR downloads its model on first run; allow some time for that step.


