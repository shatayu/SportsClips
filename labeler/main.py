"""Application entry point for the bounding box labeler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6 import QtWidgets

from .config import AppConfig, build_config
from .data import AnnotationStore, DatasetIndex
from .ui.main_window import MainWindow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SportsClips bounding box labeler")
    parser.add_argument("--image-root", type=Path, help="Root directory containing season/network folders", default=None)
    parser.add_argument("--annotation-file", type=Path, help="Path to annotation JSON file", default=None)
    return parser.parse_args(argv)


def create_config(args: argparse.Namespace) -> AppConfig:
    base_config = build_config(Path.cwd())

    image_root = args.image_root or base_config.image_root
    annotation_path = args.annotation_file or base_config.annotation_path

    return AppConfig(image_root=image_root, annotation_path=annotation_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = create_config(args)

    dataset_index = DatasetIndex(config.image_root)
    dataset_index.build()

    annotation_store = AnnotationStore.load(config.annotation_path)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config, dataset_index, annotation_store)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

