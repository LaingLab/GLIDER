"""The project's one screen: every image in a grid, with stats, filters and bulk actions.

Clicking a tile opens it in the editor (the window's other page); Cmd/Ctrl- or
Shift-click selects, and a selection gets a bar to move tiles to another split
or skip them.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.animal_annotator.gallery import _SPLIT_COLOUR, _STATUS_COLOUR, GalleryView
from glider.gui.animal_annotator.style import muted, secondary
from glider.gui.styles import colors
from glider.vision.training.project import StudioProject
from glider.vision.training.splits import SPLITS, leak_notice, split_counts

__all__ = ["GridPage", "short_path"]

#: Status chips: gallery status, label.
_STATUSES = (("labelled", "Annotated"), ("todo", "Unannotated"), ("skipped", "Skipped"))


def short_path(path: Path, limit: int = 70) -> str:
    text = str(path)
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


class _SplitBar(QWidget):
    """A thin bar in three colours, one segment per split, sized by its frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.counts = dict.fromkeys(SPLITS, 0)
        self.setFixedSize(240, 6)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(colors.SURFACE_2))
        painter.drawRoundedRect(QRectF(self.rect()), 3, 3)
        total = sum(self.counts.values())
        x = 0.0
        for split in SPLITS if total else ():
            w = self.width() * self.counts[split] / total
            painter.setBrush(QColor(_SPLIT_COLOUR[split]))
            painter.drawRect(QRectF(x, 0, w, self.height()))
            x += w
        painter.end()


def _chip(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setCheckable(True)
    button.setProperty("chip", True)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    return button


def _stat(caption: str, colour: str | None = None) -> tuple[QWidget, QLabel]:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    value = QLabel("0")
    value.setProperty("stat", True)
    if colour:
        value.setStyleSheet(f"color: {colour};")
    layout.addWidget(value)
    layout.addWidget(muted(QLabel(caption), small=True))
    return box, value


class GridPage(QWidget):
    """Header, stats, filter chips, the gallery and a selection bar."""

    frame_activated = pyqtSignal(int)
    add_images_requested = pyqtSignal()
    export_requested = pyqtSignal()
    train_requested = pyqtSignal()
    skeleton_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    #: rows, split
    move_requested = pyqtSignal(list, str)
    #: rows, skipped
    skip_requested = pyqtSignal(list, bool)
    #: video key
    remove_video_requested = pyqtSignal(str)
    #: rows to delete, images and labels
    delete_requested = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: StudioProject | None = None
        self._busy = False
        self._status: str | None = None
        self._split: str | None = None
        self._video: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # header
        header = QHBoxLayout()
        header.setContentsMargins(16, 12, 16, 12)
        header.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.title = QLabel("Animal Annotator")
        self.title.setProperty("studioTitle", True)
        titles.addWidget(self.title)
        self.summary = muted(QLabel(), small=True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        titles.addWidget(self.summary)
        header.addLayout(titles, 1)
        self.status_text = muted(QLabel(), small=True)
        header.addWidget(self.status_text)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        header.addWidget(self.progress)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setFlat(True)
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel)
        header.addWidget(self.cancel_btn)
        self.skeleton_btn = secondary(QPushButton("Skeleton…"))
        self.skeleton_btn.setToolTip("Rename, recolour and rearrange this project's keypoints")
        self.skeleton_btn.clicked.connect(self.skeleton_requested)
        header.addWidget(self.skeleton_btn)
        self.add_btn = secondary(QPushButton("Add Images / Videos…"))
        self.add_btn.setToolTip(
            "Add image files as they are, or pull frames worth labelling out of videos"
        )
        self.add_btn.clicked.connect(self.add_images_requested)
        header.addWidget(self.add_btn)
        self.export_btn = secondary(QPushButton("Export Dataset"))
        self.export_btn.setToolTip(
            "Write data.yaml and the train/val/test lists, to train with Ultralytics yourself"
        )
        self.export_btn.clicked.connect(self.export_requested)
        header.addWidget(self.export_btn)
        self.train_btn = QPushButton("Train Model…")
        self.train_btn.setToolTip(
            "Export the annotated images and train a YOLO model on them, here in GLIDER"
        )
        self.train_btn.clicked.connect(self.train_requested)
        header.addWidget(self.train_btn)
        layout.addLayout(header)

        # stats strip
        strip = QFrame()
        strip.setProperty("studioStrip", True)
        stats = QHBoxLayout(strip)
        stats.setContentsMargins(16, 10, 16, 10)
        stats.setSpacing(22)
        self._stats: dict[str, QLabel] = {}
        for key, caption, colour in (
            ("total", "images", None),
            ("labelled", "annotated", _STATUS_COLOUR["labelled"]),
            ("todo", "unannotated", None),
            ("skipped", "skipped", _STATUS_COLOUR["skipped"]),
        ):
            box, self._stats[key] = _stat(caption, colour)
            stats.addWidget(box)
        stats.addStretch(1)
        split_box = QVBoxLayout()
        split_box.setSpacing(4)
        split_box.addWidget(muted(QLabel("split"), small=True))
        self._split_bar = _SplitBar()
        split_box.addWidget(self._split_bar)
        self._split_text = muted(QLabel(), small=True)
        self._split_text.setTextFormat(Qt.TextFormat.RichText)
        split_box.addWidget(self._split_text)
        stats.addLayout(split_box)
        layout.addWidget(strip)

        # filter chips
        chips = QHBoxLayout()
        chips.setContentsMargins(16, 10, 16, 6)
        chips.setSpacing(6)
        self.all_chip = _chip("All")
        self.all_chip.clicked.connect(self._clear_filters)
        chips.addWidget(self.all_chip)
        self.status_chips: dict[str, QPushButton] = {}
        for status, label in _STATUSES:
            chip = self.status_chips[status] = _chip(label)
            chip.setProperty("label", label)
            chip.clicked.connect(lambda _c=False, s=status: self._toggle_status(s))
            chips.addWidget(chip)
        chips.addWidget(self._separator())
        self.split_chips: dict[str, QPushButton] = {}
        for split in SPLITS:
            chip = self.split_chips[split] = _chip(split.capitalize())
            chip.setProperty("label", split.capitalize())
            chip.clicked.connect(lambda _c=False, s=split: self._toggle_split(s))
            chips.addWidget(chip)
        chips.addWidget(self._separator())
        self.video_btn = QToolButton()
        self.video_btn.setProperty("chip", True)
        self.video_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.video_menu = QMenu(self.video_btn)
        self.video_btn.setMenu(self.video_menu)
        chips.addWidget(self.video_btn)
        chips.addStretch(1)
        self.search = QLineEdit()
        self.search.setProperty("studioSearch", True)
        self.search.setPlaceholderText("Search images…")
        self.search.setToolTip("Match a file name, video name or frame number (e.g. f120)")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(220)
        self.search.textChanged.connect(self._on_search)
        chips.addWidget(self.search)
        layout.addLayout(chips)

        self.notice = QLabel()
        self.notice.setProperty("notice", True)
        self.notice.setWordWrap(True)
        self.notice.setContentsMargins(16, 2, 16, 4)
        self.notice.setVisible(False)
        layout.addWidget(self.notice)

        # the gallery, or one call to action when there is nothing to show
        self._body = QStackedWidget()
        self._body.setContentsMargins(16, 2, 16, 8)
        self.gallery = GalleryView()
        self.gallery.setToolTip("Click an image to label it; Cmd/Ctrl- or Shift-click to select")
        self.gallery.frame_activated.connect(self.frame_activated)
        self.gallery.selection_changed.connect(self._update_selection_bar)
        self._body.addWidget(self.gallery)
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.addStretch(1)
        empty_layout.addWidget(
            muted(QLabel("No images yet. Add image files, or pull frames from videos, to start.")),
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        self.empty_add_btn = QPushButton("Add Images / Videos…")
        self.empty_add_btn.clicked.connect(self.add_images_requested)
        empty_layout.addWidget(self.empty_add_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch(1)
        self._body.addWidget(empty)
        layout.addWidget(self._body, 1)

        # selection bar
        self.selection_bar = QFrame()
        self.selection_bar.setProperty("studioSelection", True)
        bar = QHBoxLayout(self.selection_bar)
        bar.setContentsMargins(16, 8, 16, 8)
        bar.setSpacing(8)
        self._selection_label = QLabel()
        bar.addWidget(self._selection_label)
        bar.addStretch(1)
        self._bulk_buttons: list[QPushButton] = []
        for split in SPLITS:
            button = secondary(QPushButton(f"Move to {split.capitalize()}"))
            button.clicked.connect(lambda _c=False, s=split: self._move(s))
            self._bulk_buttons.append(button)
            bar.addWidget(button)
        for label, skipped in (("Skip", True), ("Unskip", False)):
            button = secondary(QPushButton(label))
            button.clicked.connect(lambda _c=False, k=skipped: self._skip(k))
            self._bulk_buttons.append(button)
            bar.addWidget(button)
        self.delete_btn = secondary(QPushButton("Delete…"))
        self.delete_btn.setProperty("danger", True)
        self.delete_btn.setToolTip("Delete the selected images and their labels (Delete key)")
        self.delete_btn.clicked.connect(self._delete)
        self._bulk_buttons.append(self.delete_btn)
        bar.addWidget(self.delete_btn)
        self.selection_bar.setVisible(False)
        layout.addWidget(self.selection_bar)

        self.gallery.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.gallery.customContextMenuRequested.connect(self._context_menu)
        self.gallery.delete_pressed.connect(self._delete)

        self.set_project(None)

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFixedHeight(18)
        line.setStyleSheet(f"color: {colors.BORDER};")
        return line

    # -- project ---------------------------------------------------------

    def set_project(self, project: StudioProject | None) -> None:
        self._project = project
        self._status = self._split = self._video = None
        self.search.blockSignals(True)
        self.search.clear()
        self.search.blockSignals(False)
        self.gallery.set_search("")
        self.gallery.set_project(project)
        self.gallery.empty_text = (
            "No images yet.\n\nAdd videos and extract frames to start labelling."
            if project is not None
            else "No project open.\n\nStart a new project or open an existing one (File menu)."
        )
        if project is None:
            self.title.setText("Animal Annotator")
            self.summary.setText("No project open")
            self.summary.setToolTip("")
        else:
            if project.is_segment:
                kind = f"Segmentation · {len(project.class_names)} classes"
            else:
                kind = f"Pose · {len(project.keypoint_names)} keypoints"
            self.title.setText(project.name)
            self.summary.setText(f"{kind} · {short_path(project.folder)}")
            self.summary.setToolTip(str(project.folder))
        self.skeleton_btn.setVisible(project is not None and not project.is_segment)
        self.status_text.setText("")
        self.set_busy(False)
        self.refresh()

    def refresh(self) -> None:
        """Stats, chip counts and notice; re-apply the filter, so edits show."""
        project = self._project
        if project is not None and self._video not in (None, *project.videos):
            self._video = None  # that video was removed
        self.update_stats()
        self._rebuild_video_menu()
        self._apply_filter()
        has_frames = project is not None and bool(project.frames)
        self._body.setCurrentIndex(0 if project is None or has_frames else 1)

    def update_stats(self) -> None:
        """The numbers only; the filter is left alone (a label edit is in progress)."""
        project = self._project
        counts = project.counts() if project else {"labelled": 0, "todo": 0, "skipped": 0}
        splits = split_counts(project) if project else dict.fromkeys(SPLITS, 0)
        total = len(project.frames) if project else 0
        self._stats["total"].setText(str(total))
        for key in ("labelled", "todo", "skipped"):
            self._stats[key].setText(str(counts[key]))
            chip = self.status_chips[key]
            chip.setText(f"{chip.property('label')}  {counts[key]}")
        for split, chip in self.split_chips.items():
            chip.setText(f"{chip.property('label')}  {splits[split]}")
        self.all_chip.setText(f"All  {total}")
        self._split_bar.counts = splits
        self._split_bar.update()
        self._split_text.setText(
            " &nbsp;".join(
                f"<span style='color:{_SPLIT_COLOUR[s]}'>●</span> {s.capitalize()} {splits[s]}"
                for s in SPLITS
            )
        )
        notice = leak_notice(project) if project else ""
        self.notice.setText(notice)
        self.notice.setVisible(bool(notice))

    # -- busy --------------------------------------------------------------

    def set_busy(self, busy: bool, text: str = "") -> None:
        """Extraction running: progress and Cancel in the header, everything else off."""
        self._busy = busy
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
        self.cancel_btn.setVisible(busy)
        self.cancel_btn.setEnabled(busy)
        self.status_text.setText(text)
        ready = not busy and self._project is not None
        for widget in (
            self.add_btn,
            self.empty_add_btn,
            self.export_btn,
            self.train_btn,
            self.skeleton_btn,
            self.video_btn,
            *self._bulk_buttons,
        ):
            widget.setEnabled(ready)

    def set_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def _cancel(self) -> None:
        self.cancel_btn.setEnabled(False)
        self.status_text.setText("Stopping…")
        self.cancel_requested.emit()

    # -- filters -----------------------------------------------------------

    def _toggle_status(self, status: str) -> None:
        self._status = None if self._status == status else status
        self._apply_filter()

    def _toggle_split(self, split: str) -> None:
        self._split = None if self._split == split else split
        self._apply_filter()

    def _clear_filters(self) -> None:
        self._status = self._split = self._video = None
        self._rebuild_video_menu()
        self.search.clear()  # re-applies the filter through _on_search
        self._apply_filter()

    def _on_search(self, text: str) -> None:
        self.gallery.set_search(text)
        self._apply_filter()

    def _set_video(self, key: str | None) -> None:
        self._video = key
        self._rebuild_video_menu()
        self._apply_filter()

    def _apply_filter(self) -> None:
        for status, chip in self.status_chips.items():
            chip.setChecked(status == self._status)
        for split, chip in self.split_chips.items():
            chip.setChecked(split == self._split)
        self.all_chip.setChecked(
            self._status is None
            and self._split is None
            and not self._video
            and not self.gallery.search_text
        )
        self.gallery.set_filter(self._status, self._split, self._video)
        self._update_selection_bar()

    def _rebuild_video_menu(self) -> None:
        menu = self.video_menu
        menu.clear()
        videos = self._project.videos if self._project else {}
        group = QActionGroup(menu)

        def label_for(key: str, path: str) -> str:
            folder = self._project is not None and self._project.is_image_folder(key)
            return f"{Path(path).name} (images)" if folder else Path(path).name

        for key, label in (
            (None, "All videos"),
            *((k, label_for(k, p)) for k, p in videos.items()),
        ):
            action = QAction(label, menu)
            action.setCheckable(True)
            action.setChecked(key == self._video)
            action.triggered.connect(lambda _c=False, k=key: self._set_video(k))
            group.addAction(action)
            menu.addAction(action)
        menu.addSeparator()
        remove = menu.addAction("Remove this video…")
        remove.setEnabled(self._video is not None)
        remove.triggered.connect(lambda: self.remove_video_requested.emit(self._video))
        name = Path(videos[self._video]).name if self._video else "all"
        self.video_btn.setText(f"Video: {name} ▾")

    # -- selection ---------------------------------------------------------

    def _update_selection_bar(self) -> None:
        rows = self.gallery.selected_rows()
        self._selection_label.setText(f"<b>{len(rows)} selected</b>")
        self.selection_bar.setVisible(bool(rows))

    def _move(self, split: str) -> None:
        rows = self.gallery.selected_rows()
        if rows:
            self.move_requested.emit(rows, split)

    def _skip(self, skipped: bool) -> None:
        rows = self.gallery.selected_rows()
        if rows:
            self.skip_requested.emit(rows, skipped)

    def _delete(self) -> None:
        rows = self.gallery.selected_rows()
        if rows and not self._busy and self._project is not None:
            self.delete_requested.emit(rows)

    def _context_menu(self, pos) -> None:
        """Right-click: act on the selection, or on the tile under the cursor if it isn't in it."""
        if self._busy or self._project is None:
            return
        index = self.gallery.indexAt(pos)
        if not index.isValid():
            return
        if not self.gallery.selectionModel().isSelected(index):
            self.gallery.setCurrentIndex(index)  # selects only that tile
        rows = self.gallery.selected_rows()
        if not rows:
            return
        noun = f"{len(rows)} images" if len(rows) > 1 else "image"
        menu = QMenu(self.gallery)
        if len(rows) == 1:
            menu.addAction("Open", lambda: self.frame_activated.emit(rows[0]))
            menu.addSeparator()
        for split in SPLITS:
            menu.addAction(f"Move to {split.capitalize()}", lambda s=split: self._move(s))
        menu.addSeparator()
        menu.addAction("Skip", lambda: self._skip(True))
        menu.addAction("Unskip", lambda: self._skip(False))
        menu.addSeparator()
        menu.addAction(f"Delete {noun}…", self._delete)
        menu.exec(self.gallery.viewport().mapToGlobal(pos))
