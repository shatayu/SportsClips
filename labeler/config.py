"""Application-level configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppConfig:
    """Typed configuration for the labeler application."""

    image_root: Path
    annotation_path: Path


def build_config(base_dir: Path | None = None) -> AppConfig:
    """Construct a configuration using environment overrides when available."""

    base_dir = base_dir or Path.cwd()

    image_root = Path(
        os.getenv("LABELER_IMAGE_ROOT", base_dir / "images")
    ).expanduser().resolve()

    annotation_path = Path(
        os.getenv("LABELER_ANNOTATION_FILE", base_dir / "labeler" / "data" / "annotations.json")
    ).expanduser().resolve()

    return AppConfig(image_root=image_root, annotation_path=annotation_path)

