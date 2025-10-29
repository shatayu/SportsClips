"""Annotation storage for folder-level and per-image bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, MutableMapping, Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def datetime_to_str(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime(ISO_FORMAT)


def datetime_from_str(raw: str) -> datetime:
    return datetime.strptime(raw, ISO_FORMAT).replace(tzinfo=timezone.utc)


@dataclass(slots=True)
class BoundingBox:
    """Normalized bounding box anchored at the top-left corner."""

    x: float
    y: float
    width: float
    height: float

    def clamp(self) -> "BoundingBox":
        """Return a clamped copy whose coordinates stay within [0, 1]."""

        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        width = min(max(self.width, 0.0), 1.0 - x)
        height = min(max(self.height, 0.0), 1.0 - y)
        return BoundingBox(x=x, y=y, width=width, height=height)

    def to_absolute(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert the normalized coordinates to absolute pixel positions."""

        left = int(round(self.x * width))
        top = int(round(self.y * height))
        box_width = int(round(self.width * width))
        box_height = int(round(self.height * height))
        return left, top, left + box_width, top + box_height

    def as_dict(self) -> Dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }

    @classmethod
    def from_dict(cls, payload: MutableMapping[str, float]) -> "BoundingBox":
        return cls(
            x=float(payload["x"]),
            y=float(payload["y"]),
            width=float(payload["width"]),
            height=float(payload["height"]),
        )


@dataclass(slots=True)
class BoxEntry:
    """Full metadata for a bounding box entry."""

    box: Optional[BoundingBox]
    updated_at: datetime
    source: str
    manual: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "box": self.box.as_dict() if self.box else None,
            "updated_at": datetime_to_str(self.updated_at),
            "source": self.source,
            "manual": self.manual,
        }

    @classmethod
    def from_dict(cls, payload: MutableMapping[str, object]) -> "BoxEntry":
        box_payload = payload.get("box")
        box = BoundingBox.from_dict(box_payload) if box_payload else None
        return cls(
            box=box,
            updated_at=datetime_from_str(payload["updated_at"]),
            source=str(payload.get("source", "unknown")),
            manual=bool(payload.get("manual", False)),
        )


@dataclass
class FolderAnnotation:
    """Annotation bundle for a single broadcast folder."""

    folder_key: str
    default_entry: Optional[BoxEntry] = None
    overrides: Dict[str, BoxEntry] = field(default_factory=dict)
    known_images: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def active_entry_for(self, image_name: str) -> Optional[BoxEntry]:
        if image_name in self.overrides:
            return self.overrides[image_name]
        return self.default_entry

    def get_box(self, image_name: str) -> Optional[BoundingBox]:
        entry = self.active_entry_for(image_name)
        return entry.box if entry else None

    def has_override(self, image_name: str) -> bool:
        return image_name in self.overrides

    def default_box(self) -> Optional[BoundingBox]:
        return self.default_entry.box if self.default_entry else None

    def override_entry(self, image_name: str) -> Optional[BoxEntry]:
        return self.overrides.get(image_name)

    def register_image(self, image_name: str) -> None:
        self.known_images.add(image_name)

    def sync_images(self, image_names: Iterable[str]) -> None:
        self.known_images.update(image_names)

    def image_boxes(self) -> Dict[str, Optional[BoundingBox]]:
        boxes: Dict[str, Optional[BoundingBox]] = {}
        for name in sorted(self.known_images):
            boxes[name] = self.get_box(name)
        return boxes

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def set_default(self, box: BoundingBox, *, source: str, manual: bool = False) -> None:
        self.default_entry = BoxEntry(box=box.clamp(), updated_at=utc_now(), source=source, manual=manual)

    def set_override(
        self,
        image_name: str,
        box: Optional[BoundingBox],
        *,
        source: str,
        manual: bool = True,
    ) -> None:
        self.register_image(image_name)
        self.overrides[image_name] = BoxEntry(
            box=box.clamp() if box else None,
            updated_at=utc_now(),
            source=source,
            manual=manual,
        )

    def clear_override(self, image_name: str) -> None:
        self.overrides.pop(image_name, None)

    def clear_default(self) -> None:
        self.default_entry = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {}
        if self.default_entry:
            data["default"] = self.default_entry.to_dict()
        if self.overrides:
            data["overrides"] = {name: entry.to_dict() for name, entry in self.overrides.items()}
        if self.known_images:
            data["images"] = {
                name: (box.as_dict() if box else None)
                for name, box in self.image_boxes().items()
            }
        return data

    @classmethod
    def from_dict(cls, folder_key: str, payload: MutableMapping[str, object]) -> "FolderAnnotation":
        default_payload = payload.get("default")
        overrides_payload = payload.get("overrides") or {}
        default_entry = BoxEntry.from_dict(default_payload) if default_payload else None
        overrides = {name: BoxEntry.from_dict(data) for name, data in overrides_payload.items()}
        images_payload = payload.get("images") or {}
        known_images = set(images_payload.keys()) or set(overrides.keys())
        return cls(
            folder_key=folder_key,
            default_entry=default_entry,
            overrides=overrides,
            known_images=known_images,
        )


class AnnotationStore:
    """Handles reading and writing annotation files with optimistic locking."""

    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._etag: Optional[int] = None
        self._folders: Dict[str, FolderAnnotation] = {}

    # ------------------------------------------------------------------
    # Loading / saving
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path) -> "AnnotationStore":
        store = cls(path)
        store.refresh()
        return store

    def refresh(self) -> None:
        if not self.path.exists():
            self._folders = {}
            self._etag = None
            return

        raw_text = self.path.read_text(encoding="utf-8")
        payload = json.loads(raw_text) if raw_text.strip() else {}

        if payload and payload.get("version", self.VERSION) != self.VERSION:
            raise RuntimeError(
                f"Unsupported annotation version: {payload.get('version')}, expected {self.VERSION}"
            )

        folder_payloads = payload.get("folders", {})
        self._folders = {
            folder_key: FolderAnnotation.from_dict(folder_key, folder_data)
            for folder_key, folder_data in folder_payloads.items()
        }

        stat = self.path.stat()
        self._etag = stat.st_mtime_ns

    def save(self, *, force: bool = False) -> None:
        if not force and self.path.exists():
            current_mtime = self.path.stat().st_mtime_ns
            if self._etag is not None and current_mtime != self._etag:
                raise RuntimeError(
                    "Annotation file was modified externally; reload before saving again."
                )

        payload = {
            "version": self.VERSION,
            "saved_at": datetime_to_str(utc_now()),
            "folders": {key: folder.to_dict() for key, folder in sorted(self._folders.items())},
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temp_path.replace(self.path)

        self._etag = self.path.stat().st_mtime_ns

    # ------------------------------------------------------------------
    # Folder management
    # ------------------------------------------------------------------
    def folder(self, folder_key: str) -> FolderAnnotation:
        if folder_key not in self._folders:
            self._folders[folder_key] = FolderAnnotation(folder_key=folder_key)
        return self._folders[folder_key]

    def folders(self) -> Iterator[FolderAnnotation]:
        for key in sorted(self._folders):
            yield self._folders[key]

    def update_default(self, folder_key: str, box: BoundingBox, *, source: str) -> None:
        folder = self.folder(folder_key)
        folder.set_default(box, source=source, manual=False)

    def update_override(
        self,
        folder_key: str,
        image_name: str,
        box: Optional[BoundingBox],
        *,
        source: str,
    ) -> None:
        folder = self.folder(folder_key)
        folder.set_override(image_name, box, source=source, manual=True)

    def clear_override(self, folder_key: str, image_name: str) -> None:
        folder = self.folder(folder_key)
        folder.clear_override(image_name)

    def clear_default(self, folder_key: str) -> None:
        folder = self.folder(folder_key)
        folder.clear_default()

    def active_box(self, folder_key: str, image_name: str) -> Optional[BoundingBox]:
        folder = self.folder(folder_key)
        return folder.get_box(image_name)


