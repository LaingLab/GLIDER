"""Animal Annotator: label frames from your videos and export a YOLO pose dataset.

Phase 1 of #224. The window has two steps, one tab each:

1. **Frames** -- add videos and pull frames worth labelling out of them.
2. **Label** -- place keypoints on each frame, for every animal in it.

**Export Dataset** writes ``data.yaml`` with the train/val lists, ready for
Ultralytics. Training inside GLIDER comes next, on this same project folder.

Labels are saved as you go, on every edit; there is no Save button to forget.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.core.presets import slugify
from glider.gui.animal_annotator.canvas import SEQUENTIAL, SKELETON, LabelCanvas, read_image
from glider.gui.animal_annotator.gallery import GalleryView
from glider.gui.animal_annotator.history import LABELS, SKIP, Edit, History
from glider.gui.animal_annotator.skeleton_editor import (
    SkeletonCanvas,
    SkeletonEditorDialog,
    TemplatePicker,
)
from glider.gui.animal_annotator.style import muted, secondary, section, studio_stylesheet
from glider.gui.qthreads import retire_thread
from glider.gui.styles import colors
from glider.gui.widgets.pastel_glyphs import lucide_icon
from glider.vision.training.dataset import DatasetError, export_dataset
from glider.vision.training.frames import ExtractionCancelledError, extract_frames
from glider.vision.training.project import PROJECT_FILE, StudioProject, StudioProjectError
from glider.vision.training.templates import SkeletonTemplate, TemplateKeypoint, palette_colour

logger = logging.getLogger(__name__)

__all__ = ["AnimalAnnotatorWindow", "NewProjectDialog"]


def _all_templates() -> list[SkeletonTemplate]:
    from glider.vision.training.templates import builtin_templates, list_custom_templates

    return builtin_templates() + list_custom_templates()


_SETTINGS_LAST = "animal_annotator/last_project"
_SETTINGS_MODE = "animal_annotator/label_mode"
_VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.m4v);;All files (*)"

#: Frame-list filters: label, and which statuses each shows.
_FILTERS = (
    ("All frames", {"todo", "labelled", "skipped"}),
    ("To do", {"todo"}),
    ("Labelled", {"labelled"}),
    ("Skipped", {"skipped"}),
)
_STATUS_COLOUR = {"labelled": "#4ade80", "todo": "#64748b", "skipped": "#f59e0b"}


def _status_icon(status: str) -> QIcon:
    """A small coloured dot: green labelled, grey to do, amber skipped."""
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(_STATUS_COLOUR[status]))
    painter.drawEllipse(2, 2, 8, 8)
    painter.end()
    return QIcon(pixmap)


def _swatch(colour: QColor) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(colour)
    painter.drawEllipse(1, 1, 10, 10)
    painter.end()
    return QIcon(pixmap)


def _short_path(path: Path, limit: int = 70) -> str:
    text = str(path)
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


def _panel() -> tuple[QFrame, QVBoxLayout]:
    """A bordered side panel with comfortable padding."""
    frame = QFrame()
    frame.setProperty("studioPanel", True)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(6)
    return frame, layout


# ----------------------------------------------------------------------
# new project
# ----------------------------------------------------------------------


class NewProjectDialog(QDialog):
    """Name, location and skeleton template for a new project."""

    def __init__(self, default_location: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Animal Annotator Project")
        self.setStyleSheet(studio_stylesheet())
        self._template: SkeletonTemplate | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 18, 20, 16)
        outer.setSpacing(12)
        title = QLabel("New project")
        title.setProperty("studioTitle", True)
        outer.addWidget(title)
        outer.addWidget(
            muted(
                QLabel("A project holds the frames you label and the skeleton you label them with.")
            )
        )

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. OFT mice, top view")
        self._name.textChanged.connect(self._update_folder)
        form.addRow("Name", self._name)

        location = QHBoxLayout()
        self._location = QLineEdit(str(default_location))
        location.addWidget(self._location, 1)
        browse = secondary(QPushButton("Browse…"))
        browse.clicked.connect(self._browse)
        location.addWidget(browse)
        form.addRow("Location", location)

        self._folder_label = muted(QLabel(), small=True)
        self._folder_label.setWordWrap(True)
        form.addRow("", self._folder_label)
        self._location.textChanged.connect(self._update_folder)

        skeleton = QHBoxLayout()
        skeleton.setSpacing(12)
        preview_frame, preview_layout = _panel()
        preview_layout.setContentsMargins(4, 4, 4, 4)
        self._preview = SkeletonCanvas(SkeletonTemplate(""), read_only=True)
        self._preview.setFixedSize(132, 132)
        preview_layout.addWidget(self._preview)
        skeleton.addWidget(preview_frame, 0, Qt.AlignmentFlag.AlignTop)
        right = QVBoxLayout()
        right.setSpacing(6)
        self._picker = TemplatePicker()
        self._picker.setToolTip("Choose a skeleton template, or make your own with New Template…")
        self._picker.template_chosen.connect(self._on_template)
        right.addWidget(self._picker, 0, Qt.AlignmentFlag.AlignLeft)
        self._keypoints_label = muted(QLabel())
        self._keypoints_label.setWordWrap(True)
        right.addWidget(self._keypoints_label)
        right.addWidget(
            muted(
                QLabel(
                    "Keypoint order is fixed once you start labelling: the model "
                    "learns keypoints by position."
                ),
                small=True,
            )
        )
        right.itemAt(right.count() - 1).widget().setWordWrap(True)
        right.addStretch()
        skeleton.addLayout(right, 1)
        form.addRow("Skeleton", skeleton)
        outer.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setFlat(True)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self._ok = QPushButton("Create Project")
        self._ok.setDefault(True)
        self._ok.clicked.connect(self._accept)
        buttons.addWidget(self._ok)
        outer.addLayout(buttons)

        self._on_template(self._picker.current.copy(self._picker.current.name))
        self._update_folder()
        self.resize(600, self.sizeHint().height())

    @property
    def name(self) -> str:
        return self._name.text().strip()

    @property
    def folder(self) -> Path:
        return Path(self._location.text().strip()) / slugify(self.name or "project")

    @property
    def template(self) -> SkeletonTemplate:
        return self._template

    @property
    def keypoint_names(self) -> list[str]:
        return self._template.names

    def _update_folder(self) -> None:
        self._folder_label.setText(f"Saved in {_short_path(self.folder, 80)}")
        self._folder_label.setToolTip(str(self.folder))
        self._ok.setEnabled(bool(self.name))

    def _on_template(self, template: SkeletonTemplate) -> None:
        self._template = template
        self._preview.set_template(template)
        names = template.names
        self._keypoints_label.setText(", ".join(names))

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Project location", self._location.text())
        if path:
            self._location.setText(path)

    def _accept(self) -> None:
        problem = self._template.problem()
        if problem:
            QMessageBox.warning(self, "Keypoints", f"These keypoints can't be used: {problem}.")
            return
        if (self.folder / PROJECT_FILE).exists():
            QMessageBox.warning(
                self,
                "Project exists",
                f"{self.folder} already holds a project. Open it instead, or pick another name.",
            )
            return
        self.accept()


# ----------------------------------------------------------------------
# extraction worker
# ----------------------------------------------------------------------


class ExtractWorker(QObject):
    """Runs :func:`extract_frames` over several videos off the UI thread."""

    progress = pyqtSignal(int, int)  # steps done, total, within the current video
    video_started = pyqtSignal(int, int, str)  # index, count, file name
    finished = pyqtSignal(int, list)  # frames added, error messages
    cancelled = pyqtSignal(int)  # frames added before the cancel

    def __init__(self, project: StudioProject, videos: list[Path], per_video: int, method: str):
        super().__init__()
        self._project = project
        self._videos = videos
        self._per_video = per_video
        self._method = method
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        added, errors = 0, []
        for i, video in enumerate(self._videos):
            self.video_started.emit(i, len(self._videos), video.name)
            try:
                added += len(
                    extract_frames(
                        self._project,
                        video,
                        self._per_video,
                        self._method,
                        progress_cb=self.progress.emit,
                        cancel_cb=self._cancel.is_set,
                    )
                )
            except ExtractionCancelledError:
                self.cancelled.emit(added)
                return
            except Exception as e:  # report and carry on with the next video
                logger.exception("frame extraction failed for %s", video)
                errors.append(f"{video.name}: {e}")
        self.finished.emit(added, errors)


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------


class AnimalAnnotatorWindow(QMainWindow):
    """Label frames and build a pose dataset."""

    def __init__(self, parent=None, settings: QSettings | None = None):
        super().__init__(parent)
        self.setWindowTitle("Animal Annotator")
        self.resize(1280, 820)
        self._settings = settings if settings is not None else QSettings()
        self._project: StudioProject | None = None
        self._thread: QThread | None = None
        self._worker: ExtractWorker | None = None
        self._frame_index = -1  # into self._project.frames
        self._loading_frame = False
        self._history = History()
        #: The shown frame's labels as last saved: what an edit is undone to.
        self._baseline: list = []

        self.setStyleSheet(studio_stylesheet())
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(10)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self._project_title = QLabel("Animal Annotator")
        self._project_title.setProperty("studioTitle", True)
        titles.addWidget(self._project_title)
        self._project_label = muted(QLabel(), small=True)
        self._project_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        titles.addWidget(self._project_label)
        bar.addLayout(titles, 1)
        new_btn = secondary(QPushButton("New Project…"))
        new_btn.clicked.connect(self._new_project)
        bar.addWidget(new_btn)
        open_btn = secondary(QPushButton("Open…"))
        open_btn.setToolTip("Open an existing Animal Annotator project folder")
        open_btn.clicked.connect(self._open_project_dialog)
        bar.addWidget(open_btn)
        self._skeleton_btn = secondary(QPushButton("Edit Skeleton…"))
        self._skeleton_btn.setToolTip("Rename, recolour and rearrange this project's keypoints")
        self._skeleton_btn.clicked.connect(self._edit_skeleton)
        bar.addWidget(self._skeleton_btn)
        bar.addSpacing(6)
        self._export_btn = QPushButton("Export Dataset")
        self._export_btn.setToolTip("Write data.yaml and the train/val lists for Ultralytics")
        self._export_btn.clicked.connect(self._export)
        bar.addWidget(self._export_btn)
        layout.addLayout(bar)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_frames_page(), "1. Images")
        self._tabs.addTab(self._build_label_page(), "2. Label")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs, 1)
        self.setCentralWidget(central)

        self._install_undo_shortcuts()
        self._set_project(None)
        last = self._settings.value(_SETTINGS_LAST, "", type=str)
        if last and (Path(last) / PROJECT_FILE).exists():
            self.open_project(Path(last), quiet=True)

    # ------------------------------------------------------------------
    # frames page
    # ------------------------------------------------------------------

    def _build_frames_page(self) -> QWidget:
        """The Images step: every frame in a grid, and the videos they come from."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self._add_videos_btn = secondary(QPushButton("Add Videos…"))
        self._add_videos_btn.setToolTip(
            "Several sessions, lighting conditions and animals make a model that "
            "works on the next recording, not just these"
        )
        self._add_videos_btn.clicked.connect(self._add_videos)
        controls.addWidget(self._add_videos_btn)
        controls.addStretch(1)
        controls.addWidget(muted(QLabel("Frames per video")))
        self._per_video = QSpinBox()
        self._per_video.setRange(1, 2000)
        self._per_video.setValue(20)
        self._per_video.setFixedWidth(84)
        self._per_video.setToolTip("Frames added from each selected video (or every video)")
        controls.addWidget(self._per_video)
        controls.addSpacing(6)
        self._method = QComboBox()
        self._method.addItem("Most varied", "diverse")
        self._method.addItem("Evenly spaced", "uniform")
        self._method.setToolTip(
            "Most varied skips near-identical frames, such as a mouse sitting still"
        )
        controls.addWidget(self._method)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(180)
        self._progress.setTextVisible(False)
        self._progress.setVisible(False)
        controls.addWidget(self._progress)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFlat(True)
        self._cancel_btn.clicked.connect(self._cancel_extract)
        self._cancel_btn.setVisible(False)
        controls.addWidget(self._cancel_btn)
        self._extract_btn = QPushButton("Extract Frames")
        self._extract_btn.clicked.connect(self._extract)
        controls.addWidget(self._extract_btn)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        videos, videos_layout = _panel()
        videos_layout.addWidget(section("Videos"))
        self._video_list = QListWidget()
        self._video_list.setProperty("compact", True)
        self._video_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._video_list.setWordWrap(True)
        self._video_list.setToolTip("Select videos to extract from only those; none selected: all")
        videos_layout.addWidget(self._video_list, 1)
        self._remove_video_btn = secondary(QPushButton("Remove"))
        self._remove_video_btn.setToolTip("Remove the selected videos and their frames")
        self._remove_video_btn.clicked.connect(self._remove_videos)
        videos_layout.addWidget(self._remove_video_btn)
        splitter.addWidget(videos)

        self._gallery = GalleryView()
        self._gallery.setToolTip("Double-click an image to label it")
        self._gallery.frame_activated.connect(self._label_frame)
        splitter.addWidget(self._gallery)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 1000])
        layout.addWidget(splitter, 1)

        self._frames_status = muted(QLabel(""), small=True)
        self._frames_status.setWordWrap(True)
        layout.addWidget(self._frames_status)
        # Videos added but not yet extracted from: not in the project until
        # they have frames, so a mistaken add leaves nothing behind.
        self._pending_videos: list[Path] = []
        return page

    def _label_frame(self, index: int) -> None:
        """Open frame ``index`` on the Label step."""
        if self._project is None or not self._tabs.isTabEnabled(1):
            return
        self._frame_index = index
        self._tabs.setCurrentIndex(1)
        self._refresh_frame_list()
        self._show_frame(index)

    def _refresh_videos(self) -> None:
        self._video_list.clear()
        if self._project is None:
            return
        counts: dict[str, list[int]] = {}
        for frame in self._project.frames:
            entry = counts.setdefault(frame.video, [0, 0])
            entry[0] += 1
            entry[1] += self._project.is_labelled(frame)
        for key, path in self._project.videos.items():
            total, labelled = counts.get(key, [0, 0])
            item = QListWidgetItem(f"{Path(path).name}\n{total} frames · {labelled} labelled")
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, ("project", key))
            self._video_list.addItem(item)
        for path in self._pending_videos:
            item = QListWidgetItem(f"{path.name}\nnot extracted yet")
            item.setToolTip(str(path))
            item.setData(Qt.ItemDataRole.UserRole, ("pending", str(path)))
            self._video_list.addItem(item)

    def _add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", _VIDEO_FILTER)
        known = {Path(p) for p in self._project.videos.values()} | set(self._pending_videos)
        for p in paths:
            path = Path(p).resolve()
            if path not in known:
                self._pending_videos.append(path)
        self._refresh_videos()

    def _remove_videos(self) -> None:
        items = self._video_list.selectedItems()
        if not items or self._project is None:
            return
        keys = [
            k for kind, k in (i.data(Qt.ItemDataRole.UserRole) for i in items) if kind == "project"
        ]
        labelled = sum(
            1 for f in self._project.frames if f.video in keys and self._project.is_labelled(f)
        )
        if keys:
            text = f"Remove {len(keys)} video(s) and their extracted frames from the project?"
            if labelled:
                text += f"\n\n{labelled} labelled frame(s) will be deleted with them."
            answer = QMessageBox.question(self, "Remove videos", text)
            if answer != QMessageBox.StandardButton.Yes:
                return
        for item in items:
            kind, value = item.data(Qt.ItemDataRole.UserRole)
            if kind == "project":
                self._project.remove_video(value)
            else:
                self._pending_videos = [p for p in self._pending_videos if str(p) != value]
        self._project.save()
        self._frame_index = -1
        self._refresh_all()

    def _selected_or_all_videos(self) -> list[Path]:
        items = self._video_list.selectedItems() or [
            self._video_list.item(i) for i in range(self._video_list.count())
        ]
        videos = []
        for item in items:
            kind, value = item.data(Qt.ItemDataRole.UserRole)
            videos.append(Path(self._project.videos[value]) if kind == "project" else Path(value))
        return videos

    def _extract(self) -> None:
        if self._project is None or self._thread is not None:
            return
        videos = self._selected_or_all_videos()
        if not videos:
            QMessageBox.information(self, "Extract frames", "Add a video first.")
            return
        missing = [v for v in videos if not v.exists()]
        if missing:
            QMessageBox.warning(
                self,
                "Video not found",
                "These videos can't be found (moved, or on a drive that isn't connected):\n\n"
                + "\n".join(str(v) for v in missing),
            )
            return
        self._save_current_frame()
        self._worker = ExtractWorker(
            self._project, videos, self._per_video.value(), self._method.currentData()
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_extract_progress)
        self._worker.video_started.connect(self._on_video_started)
        self._worker.finished.connect(self._on_extract_finished)
        self._worker.cancelled.connect(self._on_extract_cancelled)
        self._set_busy(True)
        self._thread.start()

    def _on_video_started(self, index: int, count: int, name: str) -> None:
        self._frames_status.setText(f"Video {index + 1} of {count}: {name}")
        self._progress.setRange(0, 0)

    def _on_extract_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(done)

    def _on_extract_finished(self, added: int, errors: list) -> None:
        self._end_extract()
        text = f"Added {added} frame(s)."
        if errors:
            text += " Some videos failed:\n" + "\n".join(errors)
        self._frames_status.setText(text)
        if errors:
            QMessageBox.warning(self, "Extract frames", text)

    def _on_extract_cancelled(self, added: int) -> None:
        self._end_extract()
        self._frames_status.setText(f"Cancelled. {added} frame(s) were added before stopping.")

    def _cancel_extract(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._frames_status.setText("Stopping…")

    def _end_extract(self) -> None:
        thread, worker = self._thread, self._worker
        self._thread = self._worker = None
        if thread is not None:
            retire_thread(thread, worker)
        # The worker added frames for the pending videos: they are in the project now.
        self._pending_videos = [
            p for p in self._pending_videos if str(p) not in self._project.videos.values()
        ]
        self._set_busy(False)
        self._refresh_all()

    def _set_busy(self, busy: bool) -> None:
        self._progress.setVisible(busy)
        self._cancel_btn.setVisible(busy)
        self._cancel_btn.setEnabled(busy)
        for widget in (
            self._extract_btn,
            self._add_videos_btn,
            self._remove_video_btn,
            self._export_btn,
            self._skeleton_btn,
        ):
            widget.setEnabled(not busy and self._project is not None)
        # The worker is appending to the project; labelling meanwhile would race it.
        self._tabs.setTabEnabled(1, not busy and self._project is not None)

    # ------------------------------------------------------------------
    # label page
    # ------------------------------------------------------------------

    def _build_label_page(self) -> QWidget:
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 10, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(6)
        outer.addWidget(splitter)

        left, left_layout = _panel()
        left_layout.addWidget(section("Frames"))
        self._filter = QComboBox()
        for label, _statuses in _FILTERS:
            self._filter.addItem(label)
        self._filter.currentIndexChanged.connect(self._refresh_frame_list)
        left_layout.addWidget(self._filter)
        self._frame_list = QListWidget()
        self._frame_list.setProperty("compact", True)
        self._frame_list.currentItemChanged.connect(self._on_frame_chosen)
        left_layout.addWidget(self._frame_list, 1)
        self._counts_label = muted(QLabel(), small=True)
        self._counts_label.setWordWrap(True)
        left_layout.addWidget(self._counts_label)
        splitter.addWidget(left)

        middle = QWidget()
        middle_layout = QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(8)
        top_row = QHBoxLayout()
        top_row.setSpacing(0)
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QPushButton] = {}
        for mode, label, tip, side in (
            (SEQUENTIAL, "Sequential", "Click each keypoint in turn", "left"),
            (
                SKELETON,
                "Skeleton",
                "Drop the whole skeleton, then move, tilt and scale it",
                "right",
            ),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("segment", side)
            button.setToolTip(f"{tip} (M switches)")
            button.clicked.connect(lambda _c=False, m=mode: self._set_mode(m))
            self._mode_group.addButton(button)
            self._mode_buttons[mode] = button
            top_row.addWidget(button)
        top_row.addSpacing(10)
        undo_key = QKeySequence(QKeySequence.StandardKey.Undo).toString(
            QKeySequence.SequenceFormat.NativeText
        )
        redo_key = QKeySequence(QKeySequence.StandardKey.Redo).toString(
            QKeySequence.SequenceFormat.NativeText
        )
        self._undo_btn = QToolButton()
        self._undo_btn.setProperty("stepper", True)
        self._undo_btn.setIcon(lucide_icon("undo-2", colors.TEXT_SECONDARY, 16))
        self._undo_btn.setToolTip(f"Undo ({undo_key})")
        self._undo_btn.clicked.connect(self._undo)
        top_row.addWidget(self._undo_btn)
        top_row.addSpacing(4)
        self._redo_btn = QToolButton()
        self._redo_btn.setProperty("stepper", True)
        self._redo_btn.setIcon(lucide_icon("redo-2", colors.TEXT_SECONDARY, 16))
        self._redo_btn.setToolTip(f"Redo ({redo_key})")
        self._redo_btn.clicked.connect(self._redo)
        top_row.addWidget(self._redo_btn)
        top_row.addSpacing(14)
        self._hint = QLabel()
        self._hint.setTextFormat(Qt.TextFormat.RichText)
        top_row.addWidget(self._hint, 1)
        middle_layout.addLayout(top_row)
        self._canvas_holder = QVBoxLayout()
        middle_layout.addLayout(self._canvas_holder, 1)
        self._canvas: LabelCanvas | None = None
        self._help_text = muted(QLabel(), small=True)
        self._help_text.setWordWrap(True)
        middle_layout.addWidget(self._help_text)
        self._mode = self._settings.value(_SETTINGS_MODE, SEQUENTIAL, type=str)
        if self._mode not in (SEQUENTIAL, SKELETON):
            self._mode = SEQUENTIAL
        self._mode_buttons[self._mode].setChecked(True)
        splitter.addWidget(middle)

        right, right_layout = _panel()
        right_layout.addWidget(section("Keypoints"))
        self._keypoint_list = QListWidget()
        self._keypoint_list.setProperty("compact", True)
        self._keypoint_list.setToolTip("Pick a keypoint to place or move it with the next click")
        self._keypoint_list.itemClicked.connect(self._on_keypoint_clicked)
        right_layout.addWidget(self._keypoint_list, 1)

        right_layout.addSpacing(4)
        right_layout.addWidget(section("Animal"))
        animal_row = QHBoxLayout()
        self._prev_animal_btn = QToolButton()
        self._prev_animal_btn.setProperty("stepper", True)
        self._prev_animal_btn.setIcon(lucide_icon("chevron-left", colors.TEXT_SECONDARY, 16))
        self._prev_animal_btn.setToolTip("Previous animal on this frame")
        self._prev_animal_btn.clicked.connect(lambda: self._step_animal(-1))
        animal_row.addWidget(self._prev_animal_btn)
        self._animal_label = QLabel()
        self._animal_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        animal_row.addWidget(self._animal_label, 1)
        self._next_animal_btn = QToolButton()
        self._next_animal_btn.setProperty("stepper", True)
        self._next_animal_btn.setIcon(lucide_icon("chevron-right", colors.TEXT_SECONDARY, 16))
        self._next_animal_btn.setToolTip("Next animal on this frame")
        self._next_animal_btn.clicked.connect(lambda: self._step_animal(1))
        animal_row.addWidget(self._next_animal_btn)
        right_layout.addLayout(animal_row)
        animal_actions = QHBoxLayout()
        self._new_animal_btn = secondary(QPushButton("Add"))
        self._new_animal_btn.setToolTip("Start another animal on this frame (N)")
        self._new_animal_btn.clicked.connect(self._new_animal)
        animal_actions.addWidget(self._new_animal_btn)
        self._delete_animal_btn = secondary(QPushButton("Delete"))
        self._delete_animal_btn.setToolTip("Remove the current animal and its keypoints")
        self._delete_animal_btn.clicked.connect(self._delete_animal)
        animal_actions.addWidget(self._delete_animal_btn)
        right_layout.addLayout(animal_actions)

        right_layout.addSpacing(4)
        right_layout.addWidget(section("Frame"))
        nav = QHBoxLayout()
        prev_btn = secondary(QPushButton("← Previous"))
        prev_btn.clicked.connect(lambda: self._step_frame(-1))
        nav.addWidget(prev_btn)
        next_btn = secondary(QPushButton("Next →"))
        next_btn.clicked.connect(lambda: self._step_frame(1))
        nav.addWidget(next_btn)
        right_layout.addLayout(nav)
        self._skip_btn = secondary(QPushButton("Skip Frame"))
        self._skip_btn.setCheckable(True)
        self._skip_btn.setToolTip(
            "Leave this frame out of the dataset: blurred, no animal visible… (S)"
        )
        self._skip_btn.toggled.connect(self._on_skip_toggled)
        right_layout.addWidget(self._skip_btn)
        todo_btn = QPushButton("Next To Do")
        todo_btn.setToolTip("Jump to the next frame that isn't labelled or skipped")
        todo_btn.clicked.connect(self._next_todo)
        right_layout.addWidget(todo_btn)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([220, 820, 230])

        for keys, slot in (
            (("Right", "D"), lambda: self._step_frame(1)),
            (("Left", "A"), lambda: self._step_frame(-1)),
            (("N",), self._new_animal),
            (("S",), lambda: self._skip_btn.toggle()),
            (("Delete", "Backspace"), self._remove_point),
            (("M",), self._toggle_mode),
            (("[",), lambda: self._canvas and self._canvas.rotate_current(-5)),
            (("]",), lambda: self._canvas and self._canvas.rotate_current(5)),
            (("-",), lambda: self._canvas and self._canvas.scale_current(1 / 1.05)),
            (("=", "+"), lambda: self._canvas and self._canvas.scale_current(1.05)),
        ):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), page)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(slot)
        return page

    def _make_canvas(self) -> None:
        """Build a canvas for the open project's keypoints, replacing any old one."""
        if self._canvas is not None:
            self._canvas_holder.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        self._keypoint_list.clear()
        if self._project is None:
            return
        self._canvas = LabelCanvas(
            self._project.keypoint_names,
            self._project.keypoint_colours or None,
        )
        self._canvas.set_layout([(k.x, k.y) for k in self._project_template().keypoints])
        self._canvas.set_mode(self._mode)
        self._canvas.changed.connect(self._on_labels_changed)
        self._canvas.state_changed.connect(self._refresh_label_state)
        self._canvas_holder.addWidget(self._canvas)
        for i, name in enumerate(self._project.keypoint_names):
            self._keypoint_list.addItem(QListWidgetItem(_swatch(self._canvas.colour(i)), name))

    def _status(self, frame) -> str:
        if frame.skipped:
            return "skipped"
        return "labelled" if self._project.is_labelled(frame) else "todo"

    def _frame_text(self, frame) -> str:
        """``Frame 120``, with the video's name when the project has several."""
        if len(self._project.videos) > 1:
            return f"Frame {frame.frame}  ·  {frame.video}"
        return f"Frame {frame.frame}"

    def _refresh_frame_list(self) -> None:
        self._frame_list.blockSignals(True)
        self._frame_list.clear()
        if self._project is not None:
            shown = _FILTERS[self._filter.currentIndex()][1]
            for i, frame in enumerate(self._project.frames):
                status = self._status(frame)
                if status not in shown and i != self._frame_index:
                    continue
                item = QListWidgetItem(_status_icon(status), self._frame_text(frame))
                item.setToolTip(frame.name)
                item.setData(Qt.ItemDataRole.UserRole, i)
                self._frame_list.addItem(item)
                if i == self._frame_index:
                    self._frame_list.setCurrentItem(item)
        self._frame_list.blockSignals(False)
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        if self._project is None:
            self._counts_label.setText("")
            return
        c = self._project.counts()
        self._counts_label.setText(
            f"{c['labelled']} labelled · {c['skipped']} skipped · {c['todo']} to do"
        )

    def _update_frame_item(self, index: int) -> None:
        self._gallery.gallery_model.invalidate(index)
        for row in range(self._frame_list.count()):
            item = self._frame_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == index:
                frame = self._project.frames[index]
                item.setIcon(_status_icon(self._status(frame)))
        self._refresh_counts()

    def _on_frame_chosen(self, item: QListWidgetItem | None, _previous=None) -> None:
        if item is None:
            return
        self._show_frame(item.data(Qt.ItemDataRole.UserRole))

    def _show_frame(self, index: int) -> None:
        if self._project is None or self._canvas is None:
            return
        if not 0 <= index < len(self._project.frames):
            self._frame_index = -1
            self._canvas.set_frame(None, [])
            return
        self._frame_index = index
        frame = self._project.frames[index]
        image = read_image(self._project.image_path(frame))
        try:
            instances = self._project.read_instances(frame)
        except ValueError as e:
            QMessageBox.warning(
                self,
                "Label file",
                f"The labels for {frame.name} can't be read ({e}); "
                "editing this frame will replace them.",
            )
            instances = []
        self._loading_frame = True
        self._skip_btn.setChecked(frame.skipped)
        self._loading_frame = False
        self._baseline = instances
        self._canvas.set_frame(image, instances)
        self._canvas.setFocus()
        if image is None:
            self._hint.setText(f"<b>{frame.name}.jpg</b> is missing from the project folder.")
        for row in range(self._frame_list.count()):
            if self._frame_list.item(row).data(Qt.ItemDataRole.UserRole) == index:
                self._frame_list.blockSignals(True)
                self._frame_list.setCurrentRow(row)
                self._frame_list.blockSignals(False)

    def _save_current_frame(self) -> None:
        """Write the canvas's labels for the frame on screen."""
        if self._project is None or self._canvas is None or self._frame_index < 0:
            return
        if not self._canvas.has_image:
            return
        frame = self._project.frames[self._frame_index]
        after = self._canvas.instances()
        if after == self._baseline:
            return  # nothing changed: nothing to save or undo
        try:
            self._project.write_instances(frame, after)
        except OSError as e:
            QMessageBox.warning(self, "Save labels", f"Could not save labels for {frame.name}: {e}")
            return
        self._history.push(Edit(self._frame_index, LABELS, self._baseline, after))
        self._baseline = after
        self._update_frame_item(self._frame_index)
        self._refresh_undo_buttons()

    def _on_labels_changed(self) -> None:
        self._save_current_frame()

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._mode_buttons[mode].setChecked(True)
        self._settings.setValue(_SETTINGS_MODE, mode)
        if self._canvas is not None:
            self._canvas.set_mode(mode)
            self._canvas.setFocus()
        self._refresh_label_state()

    def _toggle_mode(self) -> None:
        self._set_mode(SKELETON if self._mode == SEQUENTIAL else SEQUENTIAL)

    def _refresh_label_state(self) -> None:
        common = (
            "Drag a point: move · Right-click: hidden/visible · Delete: remove point · "
            "Wheel: zoom · Middle-drag: pan · ←/→: frame · S: skip · M: switch mode · "
            + QKeySequence(QKeySequence.StandardKey.Undo).toString(
                QKeySequence.SequenceFormat.NativeText
            )
            + ": undo"
        )
        if self._mode == SKELETON:
            self._help_text.setText(
                "Click: drop skeleton · Drag a box: drop it sized · Inside the box: move · "
                "Corners: scale · Top handle: tilt · [ ]: tilt 5° · − =: scale · " + common
            )
        else:
            self._help_text.setText("Click: place next keypoint · N: next animal · " + common)
        canvas = self._canvas
        if canvas is None or not canvas.has_image:
            self._hint.setText("Pick a frame on the left." if self._project else "")
            self._animal_label.setText("")
            return
        names = self._project.keypoint_names
        placed = canvas.placed()
        for i in range(self._keypoint_list.count()):
            item = self._keypoint_list.item(i)
            font = item.font()
            font.setBold(i == canvas.current_keypoint)
            item.setFont(font)
            item.setText(f"{names[i]}  ✓" if placed[i] else names[i])
            item.setForeground(QColor(colors.TEXT_PRIMARY if placed[i] else colors.TEXT_SECONDARY))
        if self._mode == SKELETON:
            self._keypoint_list.clearSelection()
            if not any(placed):
                self._hint.setText(
                    "Click the animal to drop the <b>whole skeleton</b>, or drag a box round it."
                )
            else:
                self._hint.setText(
                    "Move, tilt or scale with the box, then drag keypoints into place. "
                    "Click elsewhere for another animal."
                )
        elif canvas.current_keypoint >= 0:
            self._keypoint_list.setCurrentRow(canvas.current_keypoint)
            self._hint.setText(f"Click to place <b>{names[canvas.current_keypoint]}</b>")
        else:
            self._keypoint_list.clearSelection()
            self._hint.setText(
                "All points placed. <b>N</b> for another animal, <b>→</b> for the next frame."
            )
        self._animal_label.setText(f"Animal {canvas.current_animal + 1} of {canvas.animal_count}")
        self._prev_animal_btn.setEnabled(canvas.current_animal > 0)
        self._next_animal_btn.setEnabled(canvas.current_animal < canvas.animal_count - 1)

    def _on_keypoint_clicked(self, item: QListWidgetItem) -> None:
        if self._canvas is not None:
            self._canvas.set_current_keypoint(self._keypoint_list.row(item))
            self._canvas.setFocus()

    def _new_animal(self) -> None:
        if self._canvas is not None:
            self._canvas.new_animal()

    def _delete_animal(self) -> None:
        if self._canvas is not None:
            self._canvas.delete_animal()

    def _step_animal(self, delta: int) -> None:
        if self._canvas is not None:
            self._canvas.select_animal(self._canvas.current_animal + delta)

    def _remove_point(self) -> None:
        if self._canvas is not None:
            self._canvas.remove_selected_point()

    def _on_skip_toggled(self, checked: bool) -> None:
        if self._loading_frame or self._project is None or self._frame_index < 0:
            return
        self._project.frames[self._frame_index].skipped = checked
        self._project.save()
        self._history.push(Edit(self._frame_index, SKIP, not checked, checked))
        self._update_frame_item(self._frame_index)
        self._refresh_undo_buttons()

    def _visible_indices(self) -> list[int]:
        return [
            self._frame_list.item(r).data(Qt.ItemDataRole.UserRole)
            for r in range(self._frame_list.count())
        ]

    def _step_frame(self, delta: int) -> None:
        indices = self._visible_indices()
        if not indices:
            return
        if self._frame_index in indices:
            pos = indices.index(self._frame_index) + delta
        else:
            pos = 0 if delta > 0 else len(indices) - 1
        if 0 <= pos < len(indices):
            self._show_frame(indices[pos])

    def _next_todo(self) -> None:
        if self._project is None:
            return
        frames = self._project.frames
        n = len(frames)
        for step in range(1, n + 1):
            i = (self._frame_index + step) % n
            if self._status(frames[i]) == "todo":
                if self._filter.currentIndex() not in (0, 1):
                    self._filter.setCurrentIndex(0)
                self._show_frame(i)
                return
        QMessageBox.information(self, "Labelling", "Every frame is labelled or skipped.")

    # ------------------------------------------------------------------
    # projects
    # ------------------------------------------------------------------

    def _default_location(self) -> Path:
        from glider.first_run import default_data_dir

        return default_data_dir() / "animal_annotator"

    def _new_project(self) -> None:
        dialog = NewProjectDialog(self._default_location(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            template = dialog.template
            project = StudioProject.create(
                dialog.folder,
                dialog.name,
                template.names,
                colours=template.colours,
                template_name=template.name,
                layout=[(k.x, k.y) for k in template.keypoints],
            )
        except (StudioProjectError, OSError) as e:
            QMessageBox.warning(self, "New project", str(e))
            return
        self._set_project(project)
        self._tabs.setCurrentIndex(0)

    def _open_project_dialog(self) -> None:
        start = str(self._project.folder.parent) if self._project else str(self._default_location())
        path = QFileDialog.getExistingDirectory(self, "Open Animal Annotator project", start)
        if path:
            self.open_project(Path(path))

    def open_project(self, folder: Path, quiet: bool = False) -> bool:
        """Open the project in ``folder``; says why if it can't, unless ``quiet``."""
        try:
            project = StudioProject.load(folder)
        except StudioProjectError as e:
            if not quiet:
                QMessageBox.warning(self, "Open project", str(e))
            return False
        self._set_project(project)
        return True

    # ------------------------------------------------------------------
    # undo and redo
    # ------------------------------------------------------------------

    def _install_undo_shortcuts(self) -> None:
        """Ctrl+Z / Cmd+Z undo; Ctrl+Y, Shift+Ctrl+Z / Shift+Cmd+Z redo.

        One shortcut per platform binding: Qt treats two shortcuts on the same
        key as ambiguous and fires neither, so Shift+Ctrl+Z is only added
        where the platform's Redo does not already include it.
        """
        bindings = {
            self._undo: QKeySequence.keyBindings(QKeySequence.StandardKey.Undo),
            self._redo: QKeySequence.keyBindings(QKeySequence.StandardKey.Redo),
        }
        shift_z = QKeySequence("Ctrl+Shift+Z")
        if shift_z not in bindings[self._redo]:
            bindings[self._redo].append(shift_z)
        for slot, keys in bindings.items():
            for key in keys:
                shortcut = QShortcut(key, self)
                shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
                shortcut.activated.connect(slot)

    def _refresh_undo_buttons(self) -> None:
        self._undo_btn.setEnabled(self._history.can_undo)
        self._redo_btn.setEnabled(self._history.can_redo)

    def _undo(self) -> None:
        self._save_current_frame()  # an edit still in flight is undone first
        edit = self._history.undo()
        if edit is not None:
            self._restore(edit, edit.before)

    def _redo(self) -> None:
        edit = self._history.redo()
        if edit is not None:
            self._restore(edit, edit.after)

    def _restore(self, edit: Edit, value) -> None:
        """Put ``edit``'s frame back to ``value``, going to that frame if needed."""
        project = self._project
        if project is None or not 0 <= edit.frame < len(project.frames):
            self._refresh_undo_buttons()
            return
        frame = project.frames[edit.frame]
        on_screen = edit.frame == self._frame_index and self._canvas is not None
        if edit.kind == SKIP:
            frame.skipped = value
            project.save()
            if on_screen:
                self._loading_frame = True
                self._skip_btn.setChecked(value)
                self._loading_frame = False
        else:
            try:
                project.write_instances(frame, value)
            except OSError as e:
                QMessageBox.warning(self, "Undo", f"Could not restore {frame.name}: {e}")
                return
            if on_screen:
                self._baseline = value
                self._canvas.set_instances(value)
        if not on_screen and self._tabs.currentIndex() == 1:
            self._show_frame(edit.frame)
        self._update_frame_item(edit.frame)
        self._refresh_undo_buttons()

    def _set_project(self, project: StudioProject | None) -> None:
        self._save_current_frame()
        self._history.clear()
        self._baseline = []
        self._project = project
        self._pending_videos = []
        self._frame_index = -1
        if project is not None:
            self._settings.setValue(_SETTINGS_LAST, str(project.folder))
            self._project_title.setText(project.name)
            self._project_label.setText(_short_path(project.folder))
            self._project_label.setToolTip(str(project.folder))
            self.setWindowTitle(f"Animal Annotator — {project.name}")
        else:
            self._project_title.setText("Animal Annotator")
            self._project_label.setText("No project open")
            self._project_label.setToolTip("")
            self.setWindowTitle("Animal Annotator")
        self._make_canvas()
        self._gallery.set_project(project)
        self._gallery.empty_text = (
            "No images yet.\n\nAdd videos and extract frames to start labelling."
            if project is not None
            else "No project open.\n\nStart a new project or open an existing one."
        )
        self._set_busy(False)
        self._refresh_all()
        self._refresh_undo_buttons()

    def _refresh_all(self) -> None:
        self._gallery.gallery_model.reload()
        self._refresh_videos()
        self._refresh_frame_list()
        if self._frame_index >= 0:
            self._show_frame(self._frame_index)
        self._refresh_label_state()

    def _on_tab_changed(self, index: int) -> None:
        if index == 1 and self._frame_index < 0 and self._project and self._project.frames:
            self._next_todo_or_first()
        elif index == 0:
            self._save_current_frame()
            self._refresh_videos()

    def _next_todo_or_first(self) -> None:
        frames = self._project.frames
        first = next((i for i, f in enumerate(frames) if self._status(f) == "todo"), 0)
        self._show_frame(first)

    def _project_template(self) -> SkeletonTemplate:
        """The open project's keypoints as a template, for the editor.

        Positions are the project's stored layout. Projects made before it was
        stored fall back to their template when the names still match, else a grid.
        """
        project = self._project
        names = project.keypoint_names
        source = next(
            (t for t in _all_templates() if t.name == project.template_name and t.names == names),
            None,
        )
        cols = max(1, math.ceil(math.sqrt(len(names))))
        keypoints = []
        for i, name in enumerate(names):
            if len(project.keypoint_layout) == len(names):
                x, y = project.keypoint_layout[i]
            elif source is not None:
                x, y = source.keypoints[i].x, source.keypoints[i].y
            else:
                x, y = (i % cols + 0.5) / cols, (i // cols + 0.5) / cols
            colour = (
                project.keypoint_colours[i]
                if i < len(project.keypoint_colours)
                else palette_colour(i)
            )
            keypoints.append(TemplateKeypoint(name, x, y, colour))
        return SkeletonTemplate(
            project.template_name or project.name,
            keypoints,
            description=source.description if source else "",
        )

    def _edit_skeleton(self) -> None:
        if self._project is None:
            return
        self._save_current_frame()
        project = self._project
        locked = any(project.is_labelled(f) for f in project.frames)

        def apply(template: SkeletonTemplate) -> str | None:
            try:
                project.apply_template(
                    template.names,
                    template.colours,
                    template.name,
                    layout=[(k.x, k.y) for k in template.keypoints],
                )
            except (StudioProjectError, OSError) as e:
                return str(e)
            return None

        dialog = SkeletonEditorDialog(
            self._project_template(),
            on_save=apply,
            locked=locked,
            title="Edit Project Skeleton",
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            index = self._frame_index
            self._make_canvas()
            self._frame_index = index
            self._refresh_all()

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------

    def _export(self) -> None:
        if self._project is None:
            return
        self._save_current_frame()
        try:
            result = export_dataset(self._project)
        except (DatasetError, OSError) as e:
            QMessageBox.warning(self, "Export dataset", str(e))
            return
        text = (
            f"Wrote {result.data_yaml}\n\n"
            f"{len(result.train)} training frame(s), {len(result.val)} validation frame(s)."
        )
        if not result.split_by_video:
            text += (
                "\n\nAll frames come from one video, so validation frames are from the "
                "same session as training ones and the scores will look better than "
                "they are. Add frames from a second video for an honest check."
            )
        QMessageBox.information(self, "Export dataset", text)

    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._save_current_frame()
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            retire_thread(self._thread, self._worker)
            self._thread = self._worker = None
        super().closeEvent(event)
