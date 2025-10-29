"""PyQt window hosting the bounding box labeling workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from ..config import AppConfig
from ..data import AnnotationStore, BoundingBox, DatasetIndex, FolderAnnotation, ImageGroup
from .canvas import BoundingBoxCanvas


@dataclass(slots=True)
class SelectionState:
    folder_key: Optional[str] = None
    image_path: Optional[Path] = None


@dataclass(slots=True)
class HistoryEntry:
    folder_key: str
    image_name: str
    previous_box: Optional[BoundingBox]
    previous_mode: str
    new_box: Optional[BoundingBox]
    new_mode: str


class MainWindow(QtWidgets.QMainWindow):
    """Primary UI surface for the labeler application."""

    def __init__(
        self,
        config: AppConfig,
        dataset_index: DatasetIndex,
        annotation_store: AnnotationStore,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.config = config
        self.dataset_index = dataset_index
        self.annotation_store = annotation_store
        self.state = SelectionState()

        self._history_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []
        self._is_replaying_history = False
        self._dirty = False
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setInterval(1500)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave_timeout)

        self.setWindowTitle("SportsClips Bounding Box Labeler")
        self.resize(1440, 900)

        self._build_ui()
        self._connect_shortcuts()
        self.canvas.box_finalized.connect(self._handle_box_finalized)
        self.propagate_action.toggled.connect(self._handle_propagation_toggle)
        self._populate_groups()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.toolbar = QtWidgets.QToolBar("Tools", self)
        self.toolbar.setMovable(False)
        self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        self.propagate_action = QtGui.QAction("Propagate edits", self)
        self.propagate_action.setCheckable(True)
        self.propagate_action.setChecked(True)
        self.propagate_action.setShortcut(QtGui.QKeySequence("Ctrl+P"))
        self.propagate_action.setStatusTip(
            "When enabled, new boxes replace the folder default. Disable to record per-image overrides instead."
        )
        self.toolbar.addAction(self.propagate_action)

        self.save_action = QtGui.QAction("Save", self)
        self.save_action.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        self.save_action.triggered.connect(self._manual_save)
        self.toolbar.addAction(self.save_action)

        self.undo_action = QtGui.QAction("Undo", self)
        self.undo_action.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        self.undo_action.setEnabled(False)
        self.undo_action.triggered.connect(self._undo)
        self.toolbar.addAction(self.undo_action)

        self.redo_action = QtGui.QAction("Redo", self)
        self.redo_action.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        self.redo_action.setEnabled(False)
        self.redo_action.triggered.connect(self._redo)
        self.toolbar.addAction(self.redo_action)

        self.clear_folder_action = QtGui.QAction("Clear folder boxes", self)
        self.clear_folder_action.setToolTip("Remove the shared box and all overrides for the current folder")
        self.clear_folder_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+X"))
        self.clear_folder_action.triggered.connect(self._clear_current_folder_boxes)
        self.clear_folder_action.setEnabled(False)
        self.toolbar.addAction(self.clear_folder_action)

        self.clear_image_action = QtGui.QAction("Clear image box", self)
        self.clear_image_action.setToolTip("Remove the bounding box for the current image")
        self.clear_image_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+D"))
        self.clear_image_action.triggered.connect(self._clear_current_image_box)
        self.clear_image_action.setEnabled(False)
        self.toolbar.addAction(self.clear_image_action)

        central = QtWidgets.QWidget(self)
        central_layout = QtWidgets.QHBoxLayout(central)
        central_layout.setContentsMargins(8, 8, 8, 8)
        central_layout.setSpacing(12)

        self.group_list = QtWidgets.QListWidget(central)
        self.group_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.group_list.setMinimumWidth(240)
        self.group_list.itemSelectionChanged.connect(self._handle_group_change)

        self.image_list = QtWidgets.QListWidget(central)
        self.image_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.image_list.setMinimumWidth(240)
        self.image_list.itemSelectionChanged.connect(self._handle_image_change)

        self.canvas = BoundingBoxCanvas(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.group_list)
        splitter.addWidget(self.image_list)
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)

        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Ready")

    def _connect_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Right), self, activated=self.next_image)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Left), self, activated=self.previous_image)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Down), self, activated=self.next_group)
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key.Key_Up), self, activated=self.previous_group)

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------
    def _populate_groups(self) -> None:
        self.group_list.blockSignals(True)
        self.group_list.clear()

        for group in self.dataset_index.groups():
            item = QtWidgets.QListWidgetItem(group.folder_key)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, group.folder_key)
            item.setToolTip(str(group.folder_path))
            self.group_list.addItem(item)

        self.group_list.blockSignals(False)

        if self.group_list.count() > 0:
            self.group_list.setCurrentRow(0)

    def _populate_images(self, group: ImageGroup) -> None:
        self.image_list.blockSignals(True)
        self.image_list.clear()

        folder = self.annotation_store.folder(group.folder_key)
        folder.sync_images(image_path.name for image_path in group.image_paths)

        for image_path in group.image_paths:
            item = QtWidgets.QListWidgetItem(self._format_image_label(folder, image_path))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, image_path)
            self.image_list.addItem(item)

        self.image_list.blockSignals(False)

        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def current_group(self) -> Optional[ImageGroup]:
        folder_key = self.state.folder_key
        if not folder_key:
            return None
        try:
            return self.dataset_index.get(folder_key)
        except KeyError:
            return None

    def current_image(self) -> Optional[Path]:
        return self.state.image_path

    def next_group(self) -> None:
        row = self.group_list.currentRow()
        if row < self.group_list.count() - 1:
            self.group_list.setCurrentRow(row + 1)

    def previous_group(self) -> None:
        row = self.group_list.currentRow()
        if row > 0:
            self.group_list.setCurrentRow(row - 1)

    def next_image(self) -> None:
        row = self.image_list.currentRow()
        if row < self.image_list.count() - 1:
            self.image_list.setCurrentRow(row + 1)

    def previous_image(self) -> None:
        row = self.image_list.currentRow()
        if row > 0:
            self.image_list.setCurrentRow(row - 1)

    # ------------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------------
    def _handle_group_change(self) -> None:
        selected_items = self.group_list.selectedItems()
        if not selected_items:
            self.state.folder_key = None
            self.state.image_path = None
            self.canvas.clear()
            self._update_action_states()
            return

        item = selected_items[0]
        folder_key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.state.folder_key = folder_key
        self.statusBar().showMessage(f"Folder: {folder_key}")

        try:
            group = self.dataset_index.get(folder_key)
        except KeyError:
            self.image_list.clear()
            self.canvas.clear()
            self.state.image_path = None
            self._update_action_states()
            return

        self._populate_images(group)

        # Update canvas default box for the first image
        current_item = self.image_list.currentItem()
        if current_item:
            self._handle_image_change()
        else:
            self.state.image_path = None
            self._update_action_states()

    def _handle_image_change(self) -> None:
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            self.state.image_path = None
            self.canvas.set_bounding_box(None)
            self._update_action_states()
            return

        image_path: Path = selected_items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        self.state.image_path = image_path

        try:
            self.canvas.set_image(image_path)
        except FileNotFoundError as exc:
            QtWidgets.QMessageBox.warning(self, "Image missing", str(exc))
            return

        folder_key = self.state.folder_key
        image_name = image_path.name

        bounding_box: Optional[BoundingBox] = None
        entry_desc = "missing"
        if folder_key:
            folder = self.annotation_store.folder(folder_key)
            folder.register_image(image_name)
            bounding_box = folder.get_box(image_name)
            if bounding_box:
                entry_desc = "override" if folder.has_override(image_name) else "default"
            elif folder.has_override(image_name):
                entry_desc = "override-blank"

        self.canvas.set_bounding_box(bounding_box)
        self.statusBar().showMessage(f"Viewing {image_name} [{entry_desc}]")
        self._refresh_image_list_labels()
        self._update_action_states()

    # ------------------------------------------------------------------
    # Annotation updates
    # ------------------------------------------------------------------
    def _handle_box_finalized(self, payload: object) -> None:
        folder_key = self.state.folder_key
        image_path = self.state.image_path

        if not folder_key or not image_path:
            return

        folder = self.annotation_store.folder(folder_key)
        image_name = image_path.name

        prev_box = self._clone_box(folder.get_box(image_name))
        prev_mode = self._entry_mode(folder, image_name)

        if payload is None:
            self.annotation_store.update_override(folder_key, image_name, None, source="user-override-none")
            self.statusBar().showMessage(f"Removed bounding box for {image_name}")
            self.canvas.set_bounding_box(None)
        else:
            assert isinstance(payload, BoundingBox)
            if self._should_propagate(folder, image_name):
                self.annotation_store.update_default(folder_key, payload, source="user-default")
                # Ensure current image is aligned with default by removing stale override.
                self.annotation_store.clear_override(folder_key, image_name)
                self.statusBar().showMessage(f"Updated folder default from {image_name}")
            else:
                self.annotation_store.update_override(folder_key, image_name, payload, source="user-override")
                self.statusBar().showMessage(f"Saved override for {image_name}")
            self.canvas.set_bounding_box(payload)

        self._refresh_image_list_labels()

        folder_after = self.annotation_store.folder(folder_key)
        new_box = self._clone_box(folder_after.get_box(image_name))
        new_mode = self._entry_mode(folder_after, image_name)
        self._record_history(folder_key, image_name, prev_box, prev_mode, new_box, new_mode)
        self._mark_dirty()
        self._update_action_states()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _refresh_image_list_labels(self) -> None:
        folder_key = self.state.folder_key
        if not folder_key:
            return

        folder = self.annotation_store.folder(folder_key)
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            image_path = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not isinstance(image_path, Path):
                continue
            item.setText(self._format_image_label(folder, image_path))

    def _format_image_label(self, folder: FolderAnnotation, image_path: Path) -> str:
        label = image_path.name
        override_entry = folder.override_entry(image_path.name)
        if override_entry:
            if override_entry.box is None:
                label += " [override blank]"
            else:
                label += " [override]"
        elif folder.default_box() is None:
            label += " [missing]"
        return label

    def _should_propagate(self, folder: FolderAnnotation, image_name: str) -> bool:
        if not self.propagate_action.isChecked():
            return False
        if folder.has_override(image_name):
            return False
        return True

    def _handle_propagation_toggle(self, checked: bool) -> None:
        mode = "folder default" if checked else "per-image overrides"
        self.statusBar().showMessage(f"Edits will update {mode}.")

    def _clear_current_image_box(self) -> None:
        if not self.state.folder_key or not self.state.image_path:
            self.statusBar().showMessage("No image selected", 2000)
            return
        self._handle_box_finalized(None)
        self._update_action_states()

    def _clear_current_folder_boxes(self) -> None:
        folder_key = self.state.folder_key
        if not folder_key:
            return

        try:
            group = self.dataset_index.get(folder_key)
        except KeyError:
            return

        folder = self.annotation_store.folder(folder_key)
        folder.sync_images(image_path.name for image_path in group.image_paths)

        default_before = self._clone_box(folder.default_box())
        overrides_before = {name: self._clone_box(entry.box) for name, entry in folder.overrides.items()}

        if not default_before and not overrides_before:
            self.statusBar().showMessage("No boxes to clear for this folder", 2000)
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear folder boxes",
            "This will remove the shared box and all per-image overrides for the current folder. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # Apply changes
        self.annotation_store.clear_default(folder_key)
        for image_name in list(folder.overrides.keys()):
            self.annotation_store.clear_override(folder_key, image_name)

        # Update history entries for each affected image
        for image_path in group.image_paths:
            image_name = image_path.name

            prev_box: Optional[BoundingBox]
            prev_mode: str

            if image_name in overrides_before:
                prev_box = overrides_before[image_name]
                prev_mode = "override"
            elif default_before is not None:
                prev_box = default_before
                prev_mode = "default"
            else:
                prev_box = None
                prev_mode = "none"

            new_box = None
            new_mode = "none"

            self._record_history(folder_key, image_name, prev_box, prev_mode, new_box, new_mode)

        self._refresh_image_list_labels()
        if self.state.image_path and self.state.image_path.parent == group.folder_path:
            self.canvas.set_bounding_box(None)

        self.statusBar().showMessage(f"Cleared boxes for {folder_key}")
        self._mark_dirty()
        self._update_action_states()

    def _entry_mode(self, folder: FolderAnnotation, image_name: str) -> str:
        if folder.has_override(image_name):
            return "override"
        if folder.get_box(image_name):
            return "default"
        return "none"

    def _clone_box(self, box: Optional[BoundingBox]) -> Optional[BoundingBox]:
        if box is None:
            return None
        return BoundingBox(**box.as_dict())

    def _record_history(
        self,
        folder_key: str,
        image_name: str,
        previous_box: Optional[BoundingBox],
        previous_mode: str,
        new_box: Optional[BoundingBox],
        new_mode: str,
    ) -> None:
        if self._is_replaying_history:
            return
        if previous_mode == new_mode and previous_box == new_box:
            return

        entry = HistoryEntry(
            folder_key=folder_key,
            image_name=image_name,
            previous_box=previous_box,
            previous_mode=previous_mode,
            new_box=new_box,
            new_mode=new_mode,
        )
        self._history_stack.append(entry)
        if len(self._history_stack) > 200:
            self._history_stack.pop(0)
        self._redo_stack.clear()
        self._update_history_actions()

    def _undo(self) -> None:
        if not self._history_stack:
            return

        entry = self._history_stack.pop()
        self._redo_stack.append(entry)
        self._apply_history_state(entry.folder_key, entry.image_name, entry.previous_box, entry.previous_mode)
        self._update_history_actions()

    def _redo(self) -> None:
        if not self._redo_stack:
            return

        entry = self._redo_stack.pop()
        self._history_stack.append(entry)
        self._apply_history_state(entry.folder_key, entry.image_name, entry.new_box, entry.new_mode)
        self._update_history_actions()

    def _apply_history_state(
        self,
        folder_key: str,
        image_name: str,
        target_box: Optional[BoundingBox],
        mode: str,
    ) -> None:
        self._is_replaying_history = True
        try:
            if mode == "override":
                if target_box is not None:
                    self.annotation_store.update_override(folder_key, image_name, target_box, source="history")
                else:
                    self.annotation_store.clear_override(folder_key, image_name)
            elif mode == "default":
                if target_box is not None:
                    self.annotation_store.update_default(folder_key, target_box, source="history")
                    self.annotation_store.clear_override(folder_key, image_name)
                else:
                    self.annotation_store.clear_default(folder_key)
            elif mode == "none":
                self.annotation_store.clear_override(folder_key, image_name)
                self.annotation_store.clear_default(folder_key)
            else:  # pragma: no cover - defensive programming
                raise ValueError(f"Unknown history mode: {mode}")
        finally:
            self._is_replaying_history = False

        self._after_history_apply(folder_key, image_name)

    def _after_history_apply(self, folder_key: str, image_name: str) -> None:
        if self.state.folder_key == folder_key:
            folder = self.annotation_store.folder(folder_key)
            if self.state.image_path and self.state.image_path.name == image_name:
                self.canvas.set_bounding_box(folder.get_box(image_name))
            self._refresh_image_list_labels()
        self._mark_dirty()

    def _update_history_actions(self) -> None:
        self.undo_action.setEnabled(bool(self._history_stack))
        self.redo_action.setEnabled(bool(self._redo_stack))

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._autosave_timer.start()

    def _autosave_timeout(self) -> None:
        self._perform_autosave(force=False)

    def _manual_save(self) -> None:
        self._perform_autosave(force=True)

    def _perform_autosave(self, *, force: bool) -> bool:
        if not force and not self._dirty:
            return False
        try:
            self.annotation_store.save(force=force)
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return False
        else:
            self._dirty = False
            self.statusBar().showMessage("Annotations saved", 1500)
            return True

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt naming
        if self._dirty:
            if not self._perform_autosave(force=True):
                event.ignore()
                return
        super().closeEvent(event)

    def _update_action_states(self) -> None:
        has_folder = self.state.folder_key is not None
        has_image = self.state.image_path is not None
        self.clear_folder_action.setEnabled(has_folder)
        self.clear_image_action.setEnabled(has_image)


