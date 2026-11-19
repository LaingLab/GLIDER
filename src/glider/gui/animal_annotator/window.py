"""Animal Annotator: label frames from your videos and export a YOLO dataset.

Phase 1 of #224. A project labels either keypoints (pose) or outlines
(instance segmentation), chosen when it is made. The window has two steps,
one tab each:

1. **Frames** -- add videos and pull frames worth labelling out of them.
2. **Label** -- place keypoints, or draw outlines, on every animal in each
   frame. Outlines can be drawn by hand or proposed by Segment Anything
   from a click or a box (the Auto tool).

**Export Dataset** writes ``data.yaml`` with the train/val lists, ready for
Ultralytics. Training inside GLIDER comes next, on this same project folder.

Labels are saved as you go, on every edit; there is no Save button to forget.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
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
from glider.gui.animal_annotator.seg_canvas import AUTO, POLYGON, SegmentCanvas, class_colour
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
from glider.vision.training.project import (
    PROJECT_FILE,
    TASK_POSE,
    TASK_SEGMENT,
    StudioProject,
    StudioProjectError,
)
from glider.vision.training.sam import DEFAULT_SAM_MODEL, SAM_MODELS
from glider.vision.training.templates import SkeletonTemplate, TemplateKeypoint, palette_colour

logger = logging.getLogger(__name__)

__all__ = ["AnimalAnnotatorWindow", "NewProjectDialog"]


def _all_templates() -> list[SkeletonTemplate]:
    from glider.vision.training.templates import builtin_templates, list_custom_templates

    return builtin_templates() + list_custom_templates()


_SETTINGS_LAST = "animal_annotator/last_project"
_SETTINGS_MODE = "animal_annotator/label_mode"
_SETTINGS_SEG_TOOL = "animal_annotator/seg_tool"
_SETTINGS_SAM_MODEL = "animal_annotator/sam_model"
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

        self._task = QComboBox()
        self._task.addItem("Pose — keypoints", TASK_POSE)
        self._task.addItem("Instance segmentation — outlines", TASK_SEGMENT)
        self._task.setToolTip(
            "Pose trains a model that finds body parts; segmentation trains one that "
            "outlines each animal. A project does one or the other."
        )
        self._task.currentIndexChanged.connect(self._on_task_changed)
        form.addRow("Labels", self._task)

        self._classes = QLineEdit("animal")
        self._classes.setToolTip(
            "What you will outline, separated by commas, e.g. 'mouse, pup'. "
            "More can be added later."
        )
        form.addRow("Classes", self._classes)
        self._form = form

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
        self._skeleton_row = form.rowCount() - 1
        self._classes_row = form.rowCount() - 2
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
        self._on_task_changed()
        self.resize(600, self.sizeHint().height())

    @property
    def name(self) -> str:
        return self._name.text().strip()

    @property
    def task(self) -> str:
        return self._task.currentData()

    def set_task(self, task: str) -> None:
        self._task.setCurrentIndex(max(0, self._task.findData(task)))

    @property
    def class_names(self) -> list[str]:
        """The classes typed, for a segmentation project; ``["animal"]`` for pose."""
        if self.task != TASK_SEGMENT:
            return ["animal"]
        return [n.strip() for n in self._classes.text().split(",") if n.strip()]

    def _on_task_changed(self, _index: int = 0) -> None:
        segment = self.task == TASK_SEGMENT
        self._form.setRowVisible(self._skeleton_row, not segment)
        self._form.setRowVisible(self._classes_row, segment)
        self.adjustSize()

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
        if self.task == TASK_SEGMENT:
            names = self.class_names
            if not names or len(set(names)) != len(names):
                QMessageBox.warning(
                    self, "Classes", "Give at least one class, each name only once."
                )
                return
            problem = None
        else:
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


class SamWorker(QObject):
    """Runs Segment Anything off the UI thread for the Auto tool.

    Requests arrive through a queued signal, one per change of clicks. Only
    the newest matters, so any request older than :attr:`latest` (set from the
    UI thread) is dropped unrun: a burst of clicks costs one SAM call, not one
    per click.
    """

    #: request id, polygon in image pixels (list) or None, error text ("" if none)
    result = pyqtSignal(int, object, str)
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.latest = 0
        self._segmenter = None

    @pyqtSlot(int, str, str, object, object, object, object)
    def run(self, request: int, model: str, key: str, image, points, labels, box) -> None:
        if request < self.latest:
            return
        from glider.vision.training.sam import SAM_MODELS, SamSegmenter, mask_to_polygon

        try:
            if self._segmenter is None or self._segmenter.model != model:
                info = next((m for m in SAM_MODELS if m.file == model), None)
                self.status.emit(
                    f"Loading {info.label if info else model}… "
                    f"(first use downloads about {info.size_mb if info else '?'} MB)"
                )
                self._segmenter = SamSegmenter(model)
                self._segmenter.load()
            if self._segmenter.image_key != key:
                self.status.emit("Reading the frame…")
                self._segmenter.set_image(key, image)
            if request < self.latest:
                return
            self.status.emit("Segmenting…")
            mask = self._segmenter.segment(points, labels, box)
            polygon = mask_to_polygon(mask) if mask is not None else None
            self.result.emit(request, polygon, "")
        except ImportError as e:
            logger.exception("SAM unavailable")
            self.result.emit(request, None, f"Segment Anything needs Ultralytics and PyTorch: {e}")
        except Exception as e:  # report it in the window and keep the tool usable
            logger.exception("SAM failed")
            self._segmenter = None
            self.result.emit(request, None, str(e) or type(e).__name__)


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------


class AnimalAnnotatorWindow(QMainWindow):
    """Label frames and build a pose or instance-segmentation dataset."""

    #: request id, model file, frame key, BGR image, points, labels, box
    _sam_request = pyqtSignal(int, str, str, object, object, object, object)

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
        # Auto (SAM) tool: started on first use, kept for the window's life.
        self._sam_thread: QThread | None = None
        self._sam_worker: SamWorker | None = None
        self._sam_request_id = 0

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
        # Segmentation projects: the same two-part switch, with their tools.
        self._tool_group = QButtonGroup(self)
        self._tool_buttons: dict[str, QPushButton] = {}
        for tool, label, tip, side in (
            (POLYGON, "Polygon", "Click round the animal to outline it", "left"),
            (
                AUTO,
                "Auto",
                "Segment Anything: click the animal, or drag a box round it",
                "right",
            ),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("segment", side)
            button.setToolTip(f"{tip} (M switches)")
            button.clicked.connect(lambda _c=False, t=tool: self._set_tool(t))
            self._tool_group.addButton(button)
            self._tool_buttons[tool] = button
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

        # The Auto tool's bar: which SAM model, accept, and what SAM is doing.
        self._auto_bar = QWidget()
        auto_row = QHBoxLayout(self._auto_bar)
        auto_row.setContentsMargins(0, 0, 0, 0)
        auto_row.setSpacing(8)
        auto_row.addWidget(muted(QLabel("Model")))
        self._sam_model = QComboBox()
        for model in SAM_MODELS:
            self._sam_model.addItem(model.label, model.file)
        saved = self._settings.value(_SETTINGS_SAM_MODEL, DEFAULT_SAM_MODEL, type=str)
        self._sam_model.setCurrentIndex(max(0, self._sam_model.findData(saved)))
        self._sam_model.setToolTip(
            "The Segment Anything model that proposes outlines. Each downloads once, "
            "on first use. Bigger is more accurate and slower on a CPU."
        )
        self._sam_model.currentIndexChanged.connect(self._on_sam_model_changed)
        auto_row.addWidget(self._sam_model)
        self._accept_btn = QPushButton("Accept")
        self._accept_btn.setToolTip("Keep the proposed outline (Enter)")
        self._accept_btn.clicked.connect(self._commit)
        auto_row.addWidget(self._accept_btn)
        self._clear_prompts_btn = secondary(QPushButton("Clear"))
        self._clear_prompts_btn.setToolTip("Drop the clicks and the proposal (Esc)")
        self._clear_prompts_btn.clicked.connect(self._cancel)
        auto_row.addWidget(self._clear_prompts_btn)
        self._sam_status = muted(QLabel(), small=True)
        auto_row.addWidget(self._sam_status, 1)
        middle_layout.addWidget(self._auto_bar)

        self._canvas_holder = QVBoxLayout()
        middle_layout.addLayout(self._canvas_holder, 1)
        self._canvas: LabelCanvas | SegmentCanvas | None = None
        self._help_text = muted(QLabel(), small=True)
        self._help_text.setWordWrap(True)
        middle_layout.addWidget(self._help_text)
        self._mode = self._settings.value(_SETTINGS_MODE, SEQUENTIAL, type=str)
        if self._mode not in (SEQUENTIAL, SKELETON):
            self._mode = SEQUENTIAL
        self._mode_buttons[self._mode].setChecked(True)
        self._tool = self._settings.value(_SETTINGS_SEG_TOOL, AUTO, type=str)
        if self._tool not in (POLYGON, AUTO):
            self._tool = AUTO
        self._tool_buttons[self._tool].setChecked(True)
        splitter.addWidget(middle)

        right, right_layout = _panel()
        # Pose projects: keypoints and animals.
        self._pose_panel = QWidget()
        pose_layout = QVBoxLayout(self._pose_panel)
        pose_layout.setContentsMargins(0, 0, 0, 0)
        pose_layout.setSpacing(6)
        right_layout.addWidget(self._pose_panel, 1)
        self._seg_panel = self._build_seg_panel()
        right_layout.addWidget(self._seg_panel, 1)
        right_layout = pose_layout  # the pose controls below go in their own panel
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

        right_layout = right.layout()  # back to the shared panel
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
            (("[",), lambda: self._pose_canvas() and self._pose_canvas().rotate_current(-5)),
            (("]",), lambda: self._pose_canvas() and self._pose_canvas().rotate_current(5)),
            (("-",), lambda: self._pose_canvas() and self._pose_canvas().scale_current(1 / 1.05)),
            (("=", "+"), lambda: self._pose_canvas() and self._pose_canvas().scale_current(1.05)),
            (("Return", "Enter"), self._commit),
            (("Escape",), self._cancel),
        ):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), page)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(slot)
        return page

    def _build_seg_panel(self) -> QWidget:
        """Segmentation projects' side panel: classes and the outlines on this frame."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(section("Classes"))
        self._class_list = QListWidget()
        self._class_list.setProperty("compact", True)
        self._class_list.setToolTip(
            "The class new outlines get; picking one also changes the selected outline's "
            "class. Double-click to rename."
        )
        self._class_list.itemClicked.connect(self._on_class_clicked)
        self._class_list.itemDoubleClicked.connect(self._rename_class)
        layout.addWidget(self._class_list, 1)
        class_actions = QHBoxLayout()
        add_class = secondary(QPushButton("Add Class…"))
        add_class.clicked.connect(self._add_class)
        class_actions.addWidget(add_class)
        layout.addLayout(class_actions)

        layout.addSpacing(4)
        layout.addWidget(section("Outlines"))
        self._shape_list = QListWidget()
        self._shape_list.setProperty("compact", True)
        self._shape_list.setToolTip("The animals outlined on this frame; pick one to edit it")
        self._shape_list.currentRowChanged.connect(self._on_shape_chosen)
        # Clicking the row already current still counts as picking that outline.
        self._shape_list.itemClicked.connect(
            lambda item: self._on_shape_chosen(self._shape_list.row(item))
        )
        layout.addWidget(self._shape_list, 1)
        self._delete_shape_btn = secondary(QPushButton("Delete Outline"))
        self._delete_shape_btn.setToolTip("Remove the selected outline")
        self._delete_shape_btn.clicked.connect(self._delete_shape)
        layout.addWidget(self._delete_shape_btn)
        return panel

    def _is_segment(self) -> bool:
        return self._project is not None and self._project.is_segment

    def _pose_canvas(self) -> LabelCanvas | None:
        return self._canvas if isinstance(self._canvas, LabelCanvas) else None

    def _seg_canvas(self) -> SegmentCanvas | None:
        return self._canvas if isinstance(self._canvas, SegmentCanvas) else None

    def _show_task_controls(self) -> None:
        """Show the pose or the segmentation controls, whichever the project uses."""
        segment = self._is_segment()
        for button in self._mode_buttons.values():
            button.setVisible(not segment)
        for button in self._tool_buttons.values():
            button.setVisible(segment)
        self._pose_panel.setVisible(not segment)
        self._seg_panel.setVisible(segment)
        self._auto_bar.setVisible(segment and self._tool == AUTO)
        self._skeleton_btn.setVisible(not segment)

    def _make_canvas(self) -> None:
        """Build a canvas for the open project, replacing any old one."""
        if self._canvas is not None:
            # Hidden now, not just when deleteLater gets round to it, so a
            # stale canvas never draws over the toolbar in the meantime.
            self._canvas.hide()
            self._canvas_holder.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        self._keypoint_list.clear()
        self._class_list.clear()
        self._shape_list.clear()
        self._show_task_controls()
        if self._project is None:
            return
        if self._project.is_segment:
            canvas = SegmentCanvas(self._project.class_names)
            canvas.set_tool(self._tool)
            canvas.changed.connect(self._on_labels_changed)
            canvas.state_changed.connect(self._refresh_label_state)
            canvas.prompt_changed.connect(self._request_sam)
            self._canvas = canvas
            self._canvas_holder.addWidget(canvas)
            self._refresh_class_list()
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
        # Any SAM answer still on its way is for the previous frame's clicks.
        self._sam_request_id += 1
        self._sam_status.setText("")
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
        canvas = self._pose_canvas()
        if canvas is not None:
            canvas.set_mode(mode)
            canvas.setFocus()
        self._refresh_label_state()

    def _toggle_mode(self) -> None:
        if self._is_segment():
            self._set_tool(AUTO if self._tool == POLYGON else POLYGON)
            return
        self._set_mode(SKELETON if self._mode == SEQUENTIAL else SEQUENTIAL)

    def _set_tool(self, tool: str) -> None:
        """Segmentation: switch between the Polygon and Auto tools."""
        self._tool = tool
        self._tool_buttons[tool].setChecked(True)
        self._settings.setValue(_SETTINGS_SEG_TOOL, tool)
        self._auto_bar.setVisible(self._is_segment() and tool == AUTO)
        self._sam_status.setText("")
        canvas = self._seg_canvas()
        if canvas is not None:
            canvas.set_tool(tool)
            canvas.setFocus()
        self._refresh_label_state()

    def _commit(self) -> None:
        """Enter: close the polygon being drawn, or accept SAM's proposal."""
        canvas = self._seg_canvas()
        if canvas is not None:
            canvas.commit()
            canvas.setFocus()

    def _cancel(self) -> None:
        """Esc: drop the polygon being drawn, or the Auto tool's clicks."""
        canvas = self._seg_canvas()
        if canvas is not None:
            canvas.cancel()
            self._sam_status.setText("")
            canvas.setFocus()

    # ------------------------------------------------------------------
    # segmentation: classes, outlines and SAM
    # ------------------------------------------------------------------

    def _refresh_class_list(self) -> None:
        if self._project is None or not self._project.is_segment:
            return
        canvas = self._seg_canvas()
        current = canvas.current_class if canvas is not None else 0
        self._class_list.blockSignals(True)
        self._class_list.clear()
        for i, name in enumerate(self._project.class_names):
            self._class_list.addItem(QListWidgetItem(_swatch(class_colour(i)), name))
        self._class_list.setCurrentRow(current)
        self._class_list.blockSignals(False)

    def _on_class_clicked(self, item: QListWidgetItem) -> None:
        canvas = self._seg_canvas()
        if canvas is not None:
            canvas.set_class(self._class_list.row(item))
            canvas.setFocus()

    def _add_class(self) -> None:
        if self._project is None:
            return
        name, ok = QInputDialog.getText(self, "Add class", "Class name:")
        if not ok or not name.strip():
            return
        self._set_class_names([*self._project.class_names, name.strip()])

    def _rename_class(self, item: QListWidgetItem) -> None:
        if self._project is None:
            return
        row = self._class_list.row(item)
        name, ok = QInputDialog.getText(
            self, "Rename class", "Class name:", text=self._project.class_names[row]
        )
        if not ok or not name.strip():
            return
        names = list(self._project.class_names)
        names[row] = name.strip()
        self._set_class_names(names)

    def _set_class_names(self, names: list[str]) -> None:
        try:
            self._project.set_class_names(names)
        except (StudioProjectError, OSError) as e:
            QMessageBox.warning(self, "Classes", str(e))
            return
        canvas = self._seg_canvas()
        if canvas is not None:
            canvas.set_class_names(self._project.class_names)
        self._refresh_class_list()
        self._refresh_label_state()

    def _on_shape_chosen(self, row: int) -> None:
        canvas = self._seg_canvas()
        if canvas is not None and row >= 0:
            canvas.select_shape(row)

    def _delete_shape(self) -> None:
        canvas = self._seg_canvas()
        if canvas is not None:
            canvas.delete_shape()
            canvas.setFocus()

    def _on_sam_model_changed(self, _index: int) -> None:
        self._settings.setValue(_SETTINGS_SAM_MODEL, self._sam_model.currentData())
        self._request_sam()

    def _ensure_sam_worker(self) -> SamWorker:
        if self._sam_worker is None:
            self._sam_worker = SamWorker()
            self._sam_thread = QThread(self)
            self._sam_worker.moveToThread(self._sam_thread)
            self._sam_request.connect(self._sam_worker.run, Qt.ConnectionType.QueuedConnection)
            self._sam_worker.result.connect(self._on_sam_result)
            self._sam_worker.status.connect(self._sam_status.setText)
            self._sam_thread.start()
        return self._sam_worker

    def _request_sam(self) -> None:
        """The Auto tool's clicks changed: ask SAM for an outline (or clear it)."""
        canvas = self._seg_canvas()
        if canvas is None or canvas.tool != AUTO or self._frame_index < 0:
            return
        self._sam_request_id += 1
        points, labels, box = canvas.prompts()
        if not points and box is None:
            if self._sam_worker is not None:
                self._sam_worker.latest = self._sam_request_id
            canvas.busy = False
            canvas.set_proposal(None)
            return
        image = canvas.image
        if image is None:
            return
        worker = self._ensure_sam_worker()
        worker.latest = self._sam_request_id
        canvas.busy = True
        key = self._project.frames[self._frame_index].name
        self._sam_request.emit(
            self._sam_request_id,
            self._sam_model.currentData(),
            key,
            image,
            points,
            labels,
            box,
        )

    def _on_sam_result(self, request: int, polygon, error: str) -> None:
        if request != self._sam_request_id:
            return  # the clicks changed again since; a newer answer is coming
        canvas = self._seg_canvas()
        if canvas is None:
            return
        canvas.busy = False
        if error:
            self._sam_status.setText(f"Auto failed: {error}")
            return
        if not polygon:
            self._sam_status.setText("No outline found here; try another click or a box.")
            canvas.set_proposal(None)
            return
        self._sam_status.setText(
            "Enter to accept · click to add · right-click to take away · Esc to clear"
        )
        canvas.set_proposal(polygon)

    def _refresh_seg_state(self) -> None:
        canvas = self._seg_canvas()
        common = (
            "Drag a vertex: move · Shift-click an edge: add vertex · Delete: remove · "
            "Wheel: zoom · Middle-drag: pan · ←/→: frame · S: skip · M: switch tool · "
            + QKeySequence(QKeySequence.StandardKey.Undo).toString(
                QKeySequence.SequenceFormat.NativeText
            )
            + ": undo"
        )
        if self._tool == AUTO:
            self._help_text.setText(
                "Click: on the animal · Right-click: not the animal · Drag: box round it · "
                "Enter: accept · Esc: clear · " + common
            )
        else:
            self._help_text.setText(
                "Click: add vertex · Click the first vertex or Enter: close · Esc: abandon · "
                "Right-click a vertex: remove it · " + common
            )
        if canvas is None or not canvas.has_image:
            self._hint.setText("Pick a frame on the left." if self._project else "")
            self._shape_list.clear()
            return
        names = self._project.class_names
        cls_name = names[canvas.current_class] if canvas.current_class < len(names) else "?"
        if canvas.drafting:
            self._hint.setText(
                f"Outlining a <b>{cls_name}</b>: {canvas.draft_length} vertices. "
                "Click the first vertex or press <b>Enter</b> to close it."
            )
        elif self._tool == AUTO:
            if canvas.has_proposal:
                self._hint.setText(
                    f"<b>Enter</b> to keep this <b>{cls_name}</b>, or click to refine it."
                )
            elif canvas.has_prompts:
                self._hint.setText("Finding the outline…")
            else:
                self._hint.setText(
                    f"Click a <b>{cls_name}</b>, or drag a box round it, to outline it."
                )
        else:
            self._hint.setText(f"Click round a <b>{cls_name}</b> to outline it.")
        self._accept_btn.setEnabled(canvas.has_proposal)
        self._clear_prompts_btn.setEnabled(canvas.has_prompts or canvas.has_proposal)
        self._shape_list.blockSignals(True)
        self._shape_list.clear()
        for i in range(canvas.shape_count):
            c = canvas.shape_class(i)
            name = names[c] if c < len(names) else f"class {c}"
            self._shape_list.addItem(
                QListWidgetItem(
                    _swatch(class_colour(c)),
                    f"{i + 1}. {name}  ·  {canvas.shape_vertex_count(i)} pts",
                )
            )
        self._shape_list.setCurrentRow(canvas.current_shape)
        self._shape_list.blockSignals(False)
        self._delete_shape_btn.setEnabled(canvas.current_shape >= 0)
        self._class_list.blockSignals(True)
        self._class_list.setCurrentRow(canvas.current_class)
        self._class_list.blockSignals(False)

    def _refresh_label_state(self) -> None:
        if self._is_segment():
            self._refresh_seg_state()
            return
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
        canvas = self._pose_canvas()
        if canvas is not None:
            canvas.set_current_keypoint(self._keypoint_list.row(item))
            canvas.setFocus()

    def _new_animal(self) -> None:
        seg = self._seg_canvas()
        if seg is not None:
            # A fresh start: nothing selected, nothing half-drawn.
            seg.cancel()
            seg.select_shape(-1)
            return
        canvas = self._pose_canvas()
        if canvas is not None:
            canvas.new_animal()

    def _delete_animal(self) -> None:
        canvas = self._pose_canvas()
        if canvas is not None:
            canvas.delete_animal()

    def _step_animal(self, delta: int) -> None:
        canvas = self._pose_canvas()
        if canvas is not None:
            canvas.select_animal(canvas.current_animal + delta)

    def _remove_point(self) -> None:
        seg = self._seg_canvas()
        if seg is not None:
            seg.remove()
            return
        canvas = self._pose_canvas()
        if canvas is not None:
            canvas.remove_selected_point()

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
            if dialog.task == TASK_SEGMENT:
                project = StudioProject.create(
                    dialog.folder,
                    dialog.name,
                    [],
                    task=TASK_SEGMENT,
                    class_names=dialog.class_names,
                )
            else:
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
        if self._sam_thread is not None:
            self._sam_worker.latest = self._sam_request_id + 1  # drop anything queued
            retire_thread(self._sam_thread, self._sam_worker)
            self._sam_thread = self._sam_worker = None
        super().closeEvent(event)
