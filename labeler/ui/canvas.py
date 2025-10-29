"""Canvas widget responsible for presenting and editing bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from ..data import BoundingBox


@dataclass(slots=True)
class CanvasState:
    image_path: Optional[Path] = None
    pixmap: Optional[QtGui.QPixmap] = None
    bounding_box: Optional[BoundingBox] = None


class BoundingBoxCanvas(QtWidgets.QWidget):
    """Lightweight image canvas with placeholder support for annotations."""

    box_finalized = QtCore.pyqtSignal(object)  # Emits BoundingBox | None

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)

        self._state = CanvasState()
        self._scaled_pixmap: Optional[QtGui.QPixmap] = None
        self._draw_rect = QtCore.QRectF()
        self._mode: Optional[str] = None
        self._drag_start_norm: Optional[QtCore.QPointF] = None
        self._drag_anchor_norm: Optional[QtCore.QPointF] = None
        self._initial_box: Optional[BoundingBox] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_image(self, image_path: Path) -> None:
        pixmap = QtGui.QPixmap(str(image_path))
        if pixmap.isNull():
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        self._state.image_path = image_path
        self._state.pixmap = pixmap
        self._scaled_pixmap = None
        self._draw_rect = QtCore.QRectF()
        self.update()

    def set_bounding_box(self, box: Optional[BoundingBox]) -> None:
        self._state.bounding_box = box
        self.update()

    def clear(self) -> None:
        self._state = CanvasState()
        self._scaled_pixmap = None
        self._draw_rect = QtCore.QRectF()
        self._mode = None
        self._drag_start_norm = None
        self._drag_anchor_norm = None
        self._initial_box = None
        self.update()

    def current_box(self) -> Optional[BoundingBox]:
        return self._state.bounding_box

    # ------------------------------------------------------------------
    # Qt Events
    # ------------------------------------------------------------------
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt naming
        super().paintEvent(event)

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#121212"))

        if not self._state.pixmap:
            return

        pixmap = self._scaled()
        top_left = self._pixmap_origin(pixmap)
        painter.drawPixmap(top_left, pixmap)

        if self._state.bounding_box:
            rect = self._to_device_rect(self._state.bounding_box, pixmap)
            pen = QtGui.QPen(QtGui.QColor(0, 180, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(rect)

            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 180, 255, 80)))
            painter.drawRect(rect)

            handle_pen = QtGui.QPen(QtGui.QColor("#ffffff"))
            handle_pen.setWidth(1)
            painter.setPen(handle_pen)
            painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
            for handle_rect in self._handle_rects(rect).values():
                painter.drawEllipse(handle_rect)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._scaled_pixmap = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt naming
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self._state.pixmap:
            return

        pos = event.position()
        if not self._draw_rect.contains(pos):
            return

        box = self._state.bounding_box
        if box:
            rect = self._to_device_rect(box, self._scaled())
            handles = self._handle_rects(rect)
            for name, handle_rect in handles.items():
                if handle_rect.contains(pos):
                    self._mode = "resizing"
                    self._drag_start_norm = self._to_normalized(pos)
                    self._drag_anchor_norm = self._opposite_corner_normalized(name, rect)
                    self._initial_box = box
                    event.accept()
                    return

            if rect.contains(pos):
                self._mode = "moving"
                self._drag_start_norm = self._to_normalized(pos)
                self._initial_box = box
                event.accept()
                return

        self._mode = "drawing"
        self._drag_start_norm = self._to_normalized(pos)
        self._initial_box = self._state.bounding_box
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt naming
        if not self._mode or not self._state.pixmap:
            return

        current_norm = self._to_normalized(event.position())

        if self._mode == "drawing" and self._drag_start_norm:
            self._update_box_drawing(current_norm)
        elif self._mode == "moving" and self._drag_start_norm and self._initial_box:
            self._update_box_moving(current_norm)
        elif self._mode == "resizing" and self._drag_start_norm and self._drag_anchor_norm:
            self._update_box_resizing(current_norm)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt naming
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self._mode:
            return

        previous = self._initial_box if self._initial_box else None
        self._mode = None
        self._drag_start_norm = None
        self._drag_anchor_norm = None
        self._initial_box = None

        current = self._state.bounding_box
        if current and current.width < 0.001 and current.height < 0.001:
            self._state.bounding_box = None
            current = None

        if previous != current:
            self.box_finalized.emit(current)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 - Qt naming
        if event.key() in {QtCore.Qt.Key.Key_Delete, QtCore.Qt.Key.Key_Backspace}:
            if self._state.bounding_box is not None:
                self._state.bounding_box = None
                self.update()
                self.box_finalized.emit(None)
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scaled(self) -> QtGui.QPixmap:
        if self._scaled_pixmap is None and self._state.pixmap:
            self._scaled_pixmap = self._state.pixmap.scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            origin = self._pixmap_origin(self._scaled_pixmap)
            self._draw_rect = QtCore.QRectF(
                float(origin.x()),
                float(origin.y()),
                float(self._scaled_pixmap.width()),
                float(self._scaled_pixmap.height()),
            )
        return self._scaled_pixmap or QtGui.QPixmap()

    def _pixmap_origin(self, pixmap: QtGui.QPixmap) -> QtCore.QPoint:
        x = (self.width() - pixmap.width()) / 2
        y = (self.height() - pixmap.height()) / 2
        return QtCore.QPoint(int(x), int(y))

    def _to_device_rect(self, box: BoundingBox, pixmap: QtGui.QPixmap) -> QtCore.QRectF:
        draw_w = pixmap.width()
        draw_h = pixmap.height()
        top_left = self._pixmap_origin(pixmap)

        rect = QtCore.QRectF(
            top_left.x() + box.x * draw_w,
            top_left.y() + box.y * draw_h,
            box.width * draw_w,
            box.height * draw_h,
        )
        return rect

    def _handle_rects(self, rect: QtCore.QRectF) -> Dict[str, QtCore.QRectF]:
        radius = 6
        size = QtCore.QSizeF(radius * 2, radius * 2)

        return {
            "top_left": QtCore.QRectF(rect.topLeft() - QtCore.QPointF(radius, radius), size),
            "top_right": QtCore.QRectF(rect.topRight() - QtCore.QPointF(radius, radius), size),
            "bottom_left": QtCore.QRectF(rect.bottomLeft() - QtCore.QPointF(radius, radius), size),
            "bottom_right": QtCore.QRectF(rect.bottomRight() - QtCore.QPointF(radius, radius), size),
        }

    def _opposite_corner_normalized(self, handle_name: str, rect: QtCore.QRectF) -> QtCore.QPointF:
        mapping = {
            "top_left": rect.bottomRight(),
            "top_right": rect.bottomLeft(),
            "bottom_left": rect.topRight(),
            "bottom_right": rect.topLeft(),
        }
        device_point = mapping.get(handle_name, rect.center())
        return self._to_normalized(device_point)

    def _to_normalized(self, pos: QtCore.QPointF) -> QtCore.QPointF:
        if self._draw_rect.isNull():
            return QtCore.QPointF(0.0, 0.0)

        x = (pos.x() - self._draw_rect.left()) / self._draw_rect.width()
        y = (pos.y() - self._draw_rect.top()) / self._draw_rect.height()

        return QtCore.QPointF(
            max(0.0, min(1.0, x)),
            max(0.0, min(1.0, y)),
        )

    def _update_box_drawing(self, current_norm: QtCore.QPointF) -> None:
        if not self._drag_start_norm:
            return

        x0 = min(self._drag_start_norm.x(), current_norm.x())
        y0 = min(self._drag_start_norm.y(), current_norm.y())
        x1 = max(self._drag_start_norm.x(), current_norm.x())
        y1 = max(self._drag_start_norm.y(), current_norm.y())

        self._state.bounding_box = BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0).clamp()
        self.update()

    def _update_box_moving(self, current_norm: QtCore.QPointF) -> None:
        if not self._drag_start_norm or not self._initial_box:
            return

        delta_x = current_norm.x() - self._drag_start_norm.x()
        delta_y = current_norm.y() - self._drag_start_norm.y()

        new_box = BoundingBox(
            x=self._initial_box.x + delta_x,
            y=self._initial_box.y + delta_y,
            width=self._initial_box.width,
            height=self._initial_box.height,
        ).clamp()

        self._state.bounding_box = new_box
        self.update()

    def _update_box_resizing(self, current_norm: QtCore.QPointF) -> None:
        if not self._drag_anchor_norm:
            return

        x0 = min(self._drag_anchor_norm.x(), current_norm.x())
        y0 = min(self._drag_anchor_norm.y(), current_norm.y())
        x1 = max(self._drag_anchor_norm.x(), current_norm.x())
        y1 = max(self._drag_anchor_norm.y(), current_norm.y())

        self._state.bounding_box = BoundingBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0).clamp()
        self.update()


