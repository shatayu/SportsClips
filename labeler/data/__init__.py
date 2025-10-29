"""Data access layer for annotations and image indexing."""

from __future__ import annotations

from .annotations import AnnotationStore, BoundingBox, FolderAnnotation
from .indexer import DatasetIndex, ImageGroup

__all__ = [
    "AnnotationStore",
    "BoundingBox",
    "FolderAnnotation",
    "DatasetIndex",
    "ImageGroup",
]

