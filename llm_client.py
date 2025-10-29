#!/usr/bin/env python3
import os
import time
import base64
from typing import Callable, Optional


def _encode_frame_to_data_url(frame_bgr: object) -> str:
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("OpenCV is required to encode frames") from e

    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def vision_ask(
    frame_bgr: object,
    prompt_text: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    timeout: float = 20.0,
    max_attempts: int = 6,
    base_sleep: float = 5.0,
    on_attempt: Optional[Callable[[], None]] = None,
) -> str:
    """
    Send an image frame to a vision-capable chat model and return the text response.

    - frame_bgr: numpy ndarray in BGR color (as returned by OpenCV)
    - prompt_text: user prompt text to accompany the image
    - model: overrides default model; else reads from env VISION_MODEL or defaults to 'gpt-4o-mini'
    - provider: overrides default provider; else reads from env LLM_PROVIDER or defaults to 'openai'
    - on_attempt: optional callback invoked before each API request attempt (useful for counters)
    """
    chosen_model = model or os.getenv("VISION_MODEL", "gpt-4o-mini")
    chosen_provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()
    # Load .env if available so OPENAI_API_KEY is picked up when running locally
    if api_key is None:
        try:
            from dotenv import load_dotenv  # type: ignore

            load_dotenv()
        except Exception:
            pass
    key = api_key or os.getenv("OPENAI_API_KEY")

    if chosen_provider != "openai":
        raise NotImplementedError(f"Provider '{chosen_provider}' not implemented yet")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call vision model")

    data_url = _encode_frame_to_data_url(frame_bgr)
    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    # Prefer new SDK; define RateLimitError shim if needed
    try:
        from openai import OpenAI, RateLimitError  # type: ignore
    except Exception:
        from openai import OpenAI  # type: ignore

        class RateLimitError(Exception):
            pass

    client = OpenAI(api_key=key, timeout=timeout)
    last_err: Optional[Exception] = None

    for attempt in range(1, int(max_attempts) + 1):
        if on_attempt is not None:
            try:
                on_attempt()
            except Exception:
                pass
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=chosen_model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=64,
            )
            dt_ms = (time.time() - t0) * 1000.0
            text = (resp.choices[0].message.content or "").strip()
            print(f"[vision] {chosen_provider}:{chosen_model} call ok in {dt_ms:.0f}ms", flush=True)
            return text
        except RateLimitError as e:  # type: ignore
            last_err = e
            sleep_s = min(8.0, base_sleep * (2 ** (attempt - 1)))
            print(
                f"[vision] rate limited (attempt {attempt}/{max_attempts}); backing off {sleep_s:.2f}s",
                flush=True,
            )
            time.sleep(sleep_s)
        except Exception as e:
            last_err = e
            break

    raise RuntimeError(f"Vision model call failed: {last_err}")


