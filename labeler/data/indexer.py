"""Utilities for grouping dataset images by broadcast folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(slots=True)
class ImageGroup:
    """Represents a collection of sibling images that share a broadcast folder."""

    folder_key: str
    folder_path: Path
    image_paths: List[Path]

    def __post_init__(self) -> None:
        # Ensure deterministic ordering for navigation.
        self.image_paths.sort()


class DatasetIndex:
    """Scans the image directory and groups files by their broadcast folder."""

    def __init__(self, image_root: Path, *, extensions: Iterable[str] | None = None) -> None:
        self.image_root = image_root
        self.extensions = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
        self._groups: Dict[str, ImageGroup] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Scan the filesystem and populate group metadata."""

        if not self.image_root.exists():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")

        grouped: Dict[str, List[Path]] = {}

        for file_path in self.image_root.rglob("*"):
            if not file_path.is_file():
                continue

            if file_path.suffix.lower() not in self.extensions:
                continue

            relative_folder = file_path.parent.relative_to(self.image_root)
            folder_key = relative_folder.as_posix()

            if folder_key not in grouped:
                grouped[folder_key] = []

            grouped[folder_key].append(file_path)

        self._groups = {
            key: ImageGroup(folder_key=key, folder_path=self.image_root / key, image_paths=paths)
            for key, paths in grouped.items()
        }

    def refresh(self) -> None:
        """Re-run the grouping scan."""

        self.build()

    def groups(self) -> Iterator[ImageGroup]:
        """Iterate through discovered image groups in sorted order."""

        for key in sorted(self._groups):
            yield self._groups[key]

    def get(self, folder_key: str) -> ImageGroup:
        """Return a specific group by key."""

        try:
            return self._groups[folder_key]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Unknown folder key: {folder_key}") from exc

    def __contains__(self, folder_key: str) -> bool:
        return folder_key in self._groups

    def __len__(self) -> int:
        return len(self._groups)


