"""Animal Annotator: label frames from your videos and export a YOLO dataset.

Phase 1 of #224. A project labels either keypoints (pose) or outlines
(instance segmentation), chosen when it is made. The window has two pages:

1. **The grid** -- every image in the project, filterable by status, split and
   video. Add Images pulls frames worth labelling out of videos; a selection
   can be moved to another split or skipped.
2. **The editor** -- opens on a tile click: place keypoints, or draw outlines,
   on every animal in the frame. Outlines can be drawn by hand or proposed by
   Segment Anything from a click or a box (the Auto tool). Esc goes back.

**Export Dataset** writes ``data.yaml`` with the stored splits, ready for
Ultralytics. **Train Model** exports and then trains a YOLO model on it in a
child process (see :mod:`.train_dialog`); runs land in ``<project>/runs/``.

Labels are saved as you go, on every edit; there is no Save button to forget.
"""

from __future__ import annotations

import html
import logging
import math
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, QRect, QSettings, QSize, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QActionGroup,
    QColor,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.core.presets import slugify
from glider.gui.animal_annotator.add_images import AddImagesDialog
from glider.gui.animal_annotator.canvas import SEQUENTIAL, SKELETON, LabelCanvas, read_image
from glider.gui.animal_annotator.gallery import _SPLIT_COLOUR, GalleryView
from glider.gui.animal_annotator.grid_page import GridPage, short_path
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
from glider.vision.training.frames import (
    ExtractionCancelledError,
    extract_frames,
    import_images,
)
from glider.vision.training.project import (
    PROJECT_FILE,
    TASK_POSE,
    TASK_SEGMENT,
    StudioProject,
    StudioProjectError,
)
from glider.vision.training.sam import DEFAULT_SAM_MODEL, SAM_MODELS
from glider.vision.training.splits import SPLITS, ensure_splits, leak_notice
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


class _Pill(QLabel):
    """The canvas hint: a pill over the top of the image, hidden when it has nothing to say."""

    def __init__(self):
        super().__init__()
        self.setProperty("studioHint", True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setVisible(False)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's name
        super().setText(text)
        self.setVisible(bool(text))


def _glyph_icon(glyph: str, size: int = 22) -> QIcon:
    """``glyph`` drawn as an icon: grey normally, accent when the tool is on."""
    icon = QIcon()
    for colour, state in (
        (colors.TEXT_SECONDARY, QIcon.State.Off),
        (colors.ACCENT, QIcon.State.On),
    ):
        for scale in (1, 2):
            pixmap = QPixmap(size * scale, size * scale)
            pixmap.setDevicePixelRatio(scale)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            font = painter.font()
            font.setPixelSize(size - 2)
            painter.setFont(font)
            painter.setPen(QColor(colour))
            painter.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, glyph)
            painter.end()
            icon.addPixmap(pixmap, QIcon.Mode.Normal, state)
    return icon


def _rail_button(glyph: str, name: str, tip: str) -> QToolButton:
    """A tool on the editor's rail: its glyph with its name underneath."""
    button = QToolButton()
    button.setIcon(_glyph_icon(glyph))
    button.setIconSize(QSize(22, 22))
    button.setText(name)
    button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
    button.setToolTip(tip)
    button.setCheckable(True)
    button.setProperty("railTool", True)
    return button


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
        self._folder_label.setText(f"Saved in {short_path(self.folder, 80)}")
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

    def __init__(
        self,
        project: StudioProject,
        videos: list[Path],
        per_video: int,
        method: str,
        images: list[Path] | None = None,
    ):
        super().__init__()
        self._project = project
        self._videos = videos
        self._images = list(images or [])
        self._per_video = per_video
        self._method = method
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        added, errors = 0, []
        known = set(self._project.videos)
        # Register every video (and image folder) first, so splits are assigned
        # knowing the real count (a lone registered video gets per-frame splits).
        for video in self._videos:
            self._project.video_key(video)
        for image in self._images:
            self._project.video_key(Path(image).parent)
        steps = len(self._videos) + (1 if self._images else 0)
        try:
            if self._images:
                self.video_started.emit(0, steps, f"{len(self._images)} image(s)")
                try:
                    added += len(
                        import_images(
                            self._project,
                            self._images,
                            progress_cb=self.progress.emit,
                            cancel_cb=self._cancel.is_set,
                        )
                    )
                except ExtractionCancelledError:
                    self._drop_empty(known)
                    self.cancelled.emit(added)
                    return
                except Exception as e:  # report and carry on with the videos
                    logger.exception("image import failed")
                    errors.append(f"images: {e}")
            first = steps - len(self._videos)
            for i, video in enumerate(self._videos):
                self.video_started.emit(first + i, steps, video.name)
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
                    self._drop_empty(known)
                    self.cancelled.emit(added)
                    return
                except Exception as e:  # report and carry on with the next video
                    logger.exception("frame extraction failed for %s", video)
                    errors.append(f"{video.name}: {e}")
        except BaseException:
            self._drop_empty(known)
            raise
        self._drop_empty(known)
        self.finished.emit(added, errors)

    def _drop_empty(self, known: set[str]) -> None:
        """Forget videos registered for this run that ended with no frames; save."""
        used = {f.video for f in self._project.frames}
        for key in [k for k in self._project.videos if k not in known and k not in used]:
            del self._project.videos[key]
        try:
            self._project.save()
        except OSError:
            logger.exception("could not save the project after extraction")


class _NoWheelComboBox(QComboBox):
    """A combo that only changes on a deliberate pick, never on a stray wheel turn."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class SamWorker(QObject):
    """Runs Segment Anything off the UI thread for the Auto tool.

    Requests arrive through a queued signal, one per change of clicks. Only
    the newest matters, so any request older than :attr:`latest` (set from the
    UI thread) is dropped unrun: a burst of clicks costs one SAM call, not one
    per click.
    """

    #: request id, polygon in image pixels (list) or None, error text ("" if none)
    result = pyqtSignal(int, object, str)
    #: request id, status text
    status = pyqtSignal(int, str)

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
                    request,
                    f"Loading {info.label if info else model}… "
                    f"(first use downloads about {info.size_mb if info else '?'} MB)",
                )
                self._segmenter = SamSegmenter(model)
                self._segmenter.load()
            if self._segmenter.image_key != key:
                self.status.emit(request, "Reading the frame…")
                self._segmenter.set_image(key, image)
            if request < self.latest:
                return
            self.status.emit(request, "Segmenting…")
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
        file_menu = self.menuBar().addMenu("&File")
        self._file_actions = [
            file_menu.addAction("New Project…", self._new_project),
            file_menu.addAction("Open Project…", self._open_project_dialog),
        ]

        self._grid = GridPage()
        self._gallery = self._grid.gallery
        self._grid.frame_activated.connect(self.show_editor)
        self._grid.add_images_requested.connect(self._add_images)
        self._grid.export_requested.connect(self._export)
        self._grid.train_requested.connect(self._train)
        self._grid.skeleton_requested.connect(self._edit_skeleton)
        self._grid.cancel_requested.connect(self._cancel_extract)
        self._grid.move_requested.connect(self._move_frames)
        self._grid.skip_requested.connect(self._skip_frames)
        self._grid.remove_video_requested.connect(self._remove_video)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._grid)
        label_page = self._build_label_page()
        label_page.setContentsMargins(14, 0, 14, 10)
        self._stack.addWidget(label_page)
        self.setCentralWidget(self._stack)

        self._install_undo_shortcuts()
        self._set_project(None)
        last = self._settings.value(_SETTINGS_LAST, "", type=str)
        if last and (Path(last) / PROJECT_FILE).exists():
            self.open_project(Path(last), quiet=True)

    # ------------------------------------------------------------------
    # grid and editor pages
    # ------------------------------------------------------------------

    def show_editor(self, index: int) -> None:
        """Open frame ``index`` in the editor; not while frames are being added."""
        if self._project is None or self._thread is not None:
            return
        if not 0 <= index < len(self._project.frames):
            return
        self._frame_index = index
        # The filmstrip shows what the grid shows; stepping walks the grid's rows.
        self._filmstrip.set_filter(*self._gallery.current_filter)
        self._stack.setCurrentIndex(1)
        self._show_frame(index)

    def show_grid(self) -> None:
        """Back to the grid where it was scrolled, with the filter re-applied."""
        self._save_current_frame()
        bar = self._gallery.verticalScrollBar()
        scroll = bar.value()
        self._stack.setCurrentIndex(0)
        self._grid.refresh()
        self._gallery.doItemsLayout()
        bar.setValue(scroll)

    def _move_frames(self, rows: list, split: str) -> None:
        """Bulk move to a split: saved at once, not undoable."""
        if self._project is None or self._thread is not None:
            return
        for row in rows:
            self._project.frames[row].split = split
        self._save_bulk(rows)

    def _skip_frames(self, rows: list, skipped: bool) -> None:
        """Bulk skip or unskip: saved at once, not undoable."""
        if self._project is None or self._thread is not None:
            return
        for row in rows:
            self._project.frames[row].skipped = skipped
        self._save_bulk(rows)

    def _save_bulk(self, rows: list) -> None:
        try:
            self._project.save()
        except OSError as e:
            QMessageBox.warning(self, "Save project", f"Could not save the project: {e}")
        for row in rows:
            self._gallery.gallery_model.invalidate(row)
        self._grid.refresh()

    def _remove_video(self, key: str) -> None:
        if self._project is None or self._thread is not None or key not in self._project.videos:
            return
        labelled = sum(
            1 for f in self._project.frames if f.video == key and self._project.is_labelled(f)
        )
        text = f"Remove {Path(self._project.videos[key]).name} and its extracted frames?"
        if labelled:
            text += f"\n\n{labelled} labelled frame(s) will be deleted with it."
        if QMessageBox.question(self, "Remove video", text) != QMessageBox.StandardButton.Yes:
            return
        self._save_current_frame()
        self._project.remove_video(key)
        self._project.save()
        self._frame_index = -1
        self._history.clear()  # edits index into project.frames, which just shifted
        self._refresh_undo_buttons()
        self._refresh_all()

    def _add_images(self) -> None:
        if self._project is None or self._thread is not None:
            return
        dialog = AddImagesDialog(self._project, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted and (dialog.videos or dialog.images):
            self.extract(dialog.videos, dialog.per_video, dialog.method, images=dialog.images)

    def extract(
        self,
        videos: list[Path],
        per_video: int,
        method: str,
        images: list[Path] | None = None,
    ) -> None:
        """Copy in ``images`` and pull ``per_video`` frames from each of ``videos``,
        in the background."""
        images = list(images or [])
        if self._project is None or self._thread is not None or not (videos or images):
            return
        missing = [v for v in [*videos, *images] if not Path(v).exists()]
        if missing:
            QMessageBox.warning(
                self,
                "File not found",
                "These files can't be found (moved, or on a drive that isn't connected):\n\n"
                + "\n".join(str(v) for v in missing),
            )
            return
        self._save_current_frame()
        self._worker = ExtractWorker(
            self._project,
            [Path(v) for v in videos],
            per_video,
            method,
            images=[Path(i) for i in images],
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._grid.set_progress)
        self._worker.video_started.connect(self._on_video_started)
        self._worker.finished.connect(self._on_extract_finished)
        self._worker.cancelled.connect(self._on_extract_cancelled)
        self._grid.set_busy(True, "Starting…")
        self._set_file_actions(False)
        self._stack.setCurrentIndex(0)
        self._thread.start()

    def _on_video_started(self, index: int, count: int, name: str) -> None:
        self._grid.set_busy(True, f"{index + 1} of {count}: {name}")

    def _on_extract_finished(self, added: int, errors: list) -> None:
        text = f"Added {added} frame(s)."
        if errors:
            text += " Some videos failed:\n" + "\n".join(errors)
        self._end_extract(text.splitlines()[0])
        if errors:
            QMessageBox.warning(self, "Extract frames", text)

    def _on_extract_cancelled(self, added: int) -> None:
        self._end_extract(f"Cancelled. {added} frame(s) were added before stopping.")

    def _cancel_extract(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _end_extract(self, text: str) -> None:
        thread, worker = self._thread, self._worker
        self._thread = self._worker = None
        if thread is not None:
            retire_thread(thread, worker)
        self._set_file_actions(True)
        self._grid.set_busy(False, text)
        self._refresh_all()

    def _set_file_actions(self, enabled: bool) -> None:
        for action in self._file_actions:
            action.setEnabled(enabled)

    # ------------------------------------------------------------------
    # label page
    # ------------------------------------------------------------------

    def _build_label_page(self) -> QWidget:
        """The editor: tool rail, top bar, canvas with its hint, side panel, filmstrip."""
        page = QWidget()
        outer = QHBoxLayout(page)
        outer.setContentsMargins(0, 10, 0, 0)
        outer.setSpacing(10)

        # -- tool rail: the project's two tools, undo and redo at the bottom
        rail = QFrame()
        rail.setProperty("studioPanel", True)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(6, 8, 6, 8)
        rail_layout.setSpacing(4)
        rail_title = QLabel("TOOL")
        rail_title.setProperty("textRole", "section")
        rail_title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addWidget(rail_title)
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QToolButton] = {}
        for mode, glyph, name, tip in (
            (SEQUENTIAL, "•", "Points", "Point by point: click each keypoint in turn"),
            (
                SKELETON,
                "⌖",
                "Skeleton",
                "Drop skeleton: drop the whole skeleton, then move, tilt and scale it",
            ),
        ):
            button = _rail_button(glyph, name, f"{tip} (M switches)")
            button.clicked.connect(lambda _c=False, m=mode: self._set_mode(m))
            self._mode_group.addButton(button)
            self._mode_buttons[mode] = button
            rail_layout.addWidget(button)
        self._tool_group = QButtonGroup(self)
        self._tool_buttons: dict[str, QToolButton] = {}
        for tool, glyph, name, tip in (
            (POLYGON, "⬠", "Polygon", "Polygon: click round the animal to outline it by hand"),
            (
                AUTO,
                "✦",
                "Auto",
                "Auto: Segment Anything outlines what you click or box. "
                "The arrow picks the model; each downloads once, on first use.",
            ),
        ):
            button = _rail_button(glyph, name, f"{tip} (M switches)")
            button.clicked.connect(lambda _c=False, t=tool: self._set_tool(t))
            self._tool_group.addButton(button)
            self._tool_buttons[tool] = button
            rail_layout.addWidget(button)
        saved = self._settings.value(_SETTINGS_SAM_MODEL, DEFAULT_SAM_MODEL, type=str)
        files = [m.file for m in SAM_MODELS]
        self._sam_model_file = saved if saved in files else DEFAULT_SAM_MODEL
        sam_menu = QMenu(self._tool_buttons[AUTO])
        sam_group = QActionGroup(sam_menu)
        for model in SAM_MODELS:
            action = sam_menu.addAction(f"{model.label} · {model.size_mb} MB")
            action.setCheckable(True)
            action.setChecked(model.file == self._sam_model_file)
            action.triggered.connect(lambda _c=False, f=model.file: self._set_sam_model(f))
            sam_group.addAction(action)
        self._tool_buttons[AUTO].setMenu(sam_menu)
        self._tool_buttons[AUTO].setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._tool_buttons[AUTO].setProperty("withMenu", True)
        rail_layout.addStretch(1)
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
        rail_layout.addWidget(self._undo_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        self._redo_btn = QToolButton()
        self._redo_btn.setProperty("stepper", True)
        self._redo_btn.setIcon(lucide_icon("redo-2", colors.TEXT_SECONDARY, 16))
        self._redo_btn.setToolTip(f"Redo ({redo_key})")
        self._redo_btn.clicked.connect(self._redo)
        rail_layout.addWidget(self._redo_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(rail)

        # -- centre: top bar, canvas with the hint over it, filmstrip
        middle = QVBoxLayout()
        middle.setSpacing(8)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        back = secondary(QPushButton("← Grid"))
        back.setToolTip("Back to every image (Esc)")
        back.clicked.connect(self.show_grid)
        top_row.addWidget(back)
        self._frame_title = QLabel()
        self._frame_title.setTextFormat(Qt.TextFormat.RichText)
        top_row.addWidget(self._frame_title)
        self._split_badge = QLabel()
        self._split_badge.setProperty("splitBadge", True)
        top_row.addWidget(self._split_badge)
        top_row.addStretch(1)
        self._skip_btn = secondary(QPushButton("Skip"))
        self._skip_btn.setCheckable(True)
        self._skip_btn.setToolTip(
            "Leave this image out of the dataset: blurred, no animal visible… (S)"
        )
        self._skip_btn.toggled.connect(self._on_skip_toggled)
        top_row.addWidget(self._skip_btn)
        for icon, tip, delta in (
            ("chevron-left", "Previous image (← or A)", -1),
            ("chevron-right", "Next image (→ or D)", 1),
        ):
            button = QToolButton()
            button.setProperty("stepper", True)
            button.setIcon(lucide_icon(icon, colors.TEXT_SECONDARY, 16))
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, d=delta: self._step_frame(d))
            top_row.addWidget(button)
        middle.addLayout(top_row)

        stage = QWidget()
        self._canvas_holder = QGridLayout(stage)
        self._canvas_holder.setContentsMargins(0, 0, 0, 0)
        self._hint = _Pill()
        self._canvas_holder.addWidget(
            self._hint, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        middle.addWidget(stage, 1)
        self._canvas: LabelCanvas | SegmentCanvas | None = None

        self._filmstrip = GalleryView(strip=True)
        self._filmstrip.share_model(self._gallery)
        self._filmstrip.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self._filmstrip.setToolTip("The images the grid's filter shows; click one to open it")
        self._filmstrip.frame_activated.connect(self._show_frame)
        middle.addWidget(self._filmstrip)
        outer.addLayout(middle, 1)

        self._mode = self._settings.value(_SETTINGS_MODE, SEQUENTIAL, type=str)
        if self._mode not in (SEQUENTIAL, SKELETON):
            self._mode = SEQUENTIAL
        self._mode_buttons[self._mode].setChecked(True)
        self._tool = self._settings.value(_SETTINGS_SEG_TOOL, AUTO, type=str)
        if self._tool not in (POLYGON, AUTO):
            self._tool = AUTO
        self._tool_buttons[self._tool].setChecked(True)

        # -- side panel: keypoints and animals, or classes and outlines; the split
        right, right_layout = _panel()
        right.setFixedWidth(250)
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
        right_layout.addWidget(section("Split"))
        self._split_combo = _NoWheelComboBox()
        self._split_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        for split in SPLITS:
            self._split_combo.addItem(split.capitalize(), split)
        self._split_combo.setToolTip("Which part of the dataset this image goes in")
        self._split_combo.activated.connect(lambda _i: self._canvas.setFocus())
        self._split_combo.currentIndexChanged.connect(self._on_split_changed)
        right_layout.addWidget(self._split_combo)
        key_line = muted(QLabel("←/→ image · S skip · N animal · ? all shortcuts"), small=True)
        key_line.setWordWrap(True)
        right_layout.addWidget(key_line)
        outer.addWidget(right)

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
            (("?",), lambda: self._shortcuts_dialog().open()),
        ):
            for key in keys:
                shortcut = QShortcut(QKeySequence(key), page)
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                shortcut.activated.connect(slot)
        return page

    def _shortcuts_dialog(self) -> QDialog:
        """What ``?`` opens: every key and click the editor understands."""
        undo = QKeySequence(QKeySequence.StandardKey.Undo).toString(
            QKeySequence.SequenceFormat.NativeText
        )
        redo = QKeySequence(QKeySequence.StandardKey.Redo).toString(
            QKeySequence.SequenceFormat.NativeText
        )
        groups = [
            (
                "Images",
                [
                    ("← → or A D", "Previous / next image in the grid's filter"),
                    ("S", "Toggle skip: leave the image out of the dataset"),
                    ("Esc", "Cancel what you are drawing, else back to the grid"),
                    (undo, "Undo"),
                    (redo, "Redo"),
                    ("M", "Switch tool"),
                    ("Wheel · Middle-drag", "Zoom · pan"),
                    ("?", "This list"),
                ],
            )
        ]
        if self._is_segment():
            groups += [
                (
                    "Polygon",
                    [
                        ("Click", "Add a vertex"),
                        ("Click the first vertex or Enter", "Close the outline"),
                        ("Esc", "Abandon the outline"),
                        ("Right-click a vertex", "Remove it"),
                    ],
                ),
                (
                    "Auto",
                    [
                        ("Click · Right-click", "On the animal · not the animal"),
                        ("Drag", "A box round it"),
                        ("Enter · Esc", "Accept · clear"),
                    ],
                ),
                (
                    "Outlines",
                    [
                        ("Drag a vertex", "Move it"),
                        ("Shift-click an edge", "Add a vertex"),
                        ("Delete", "Remove"),
                        ("N", "Start a fresh outline"),
                    ],
                ),
            ]
        else:
            groups += [
                (
                    "Point by point",
                    [("Click", "Place the next keypoint"), ("N", "Next animal")],
                ),
                (
                    "Drop skeleton",
                    [
                        ("Click · Drag a box", "Drop the skeleton · drop it sized"),
                        ("Inside the box · Corners · Top handle", "Move · scale · tilt"),
                        ("[ ]", "Tilt 5°"),
                        ("− =", "Scale"),
                    ],
                ),
                (
                    "Keypoints",
                    [
                        ("Drag a point", "Move it"),
                        ("Right-click", "Hidden / visible"),
                        ("Delete", "Remove the point"),
                    ],
                ),
            ]
        rows = "".join(
            f"<tr><td colspan=2 style='padding-top:8px'><b>{title}</b></td></tr>"
            + "".join(
                f"<tr><td style='padding-right:16px; color:{colors.ACCENT}'>{key}</td>"
                f"<td>{action}</td></tr>"
                for key, action in items
            )
            for title, items in groups
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Shortcuts")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 16, 20, 16)
        label = QLabel(f"<table>{rows}</table>")
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)
        close = QPushButton("Close")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        return dialog

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
            self._canvas_holder.addWidget(canvas, 0, 0)
            self._hint.raise_()
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
        self._canvas_holder.addWidget(self._canvas, 0, 0)
        self._hint.raise_()
        for i, name in enumerate(self._project.keypoint_names):
            self._keypoint_list.addItem(QListWidgetItem(_swatch(self._canvas.colour(i)), name))

    def _update_frame_item(self, index: int) -> None:
        self._gallery.gallery_model.invalidate(index)
        self._grid.update_stats()

    def _refresh_frame_header(self) -> None:
        """``Frame 120 · video.mp4 · 4 of 30`` (place in the grid's filter) and the split badge."""
        if self._project is None or not 0 <= self._frame_index < len(self._project.frames):
            self._frame_title.setText("")
            self._split_badge.setText("")
            return
        frame = self._project.frames[self._frame_index]
        where = Path(self._project.videos.get(frame.video, frame.video)).name
        if frame.source:  # an imported image: its own file name says more than a number
            parts = [f"<b>{html.escape(frame.source)}</b>", html.escape(where)]
        else:
            parts = [f"<b>Frame {frame.frame}</b>", html.escape(where)]
        rows = self._gallery.visible_rows()
        if self._frame_index in rows:
            parts.append(f"{rows.index(self._frame_index) + 1} of {len(rows)}")
        self._frame_title.setText(" · ".join(parts))
        split = frame.split or ""
        self._split_badge.setText(split.upper())
        self._split_badge.setStyleSheet(
            f"color: {_SPLIT_COLOUR.get(split, colors.TEXT_SECONDARY)};"
        )

    def _on_split_changed(self, _index: int) -> None:
        """The side panel's Split: saved at once and not undoable, like a bulk move."""
        if self._project is None or not 0 <= self._frame_index < len(self._project.frames):
            return
        frame = self._project.frames[self._frame_index]
        split = self._split_combo.currentData()
        if frame.split == split:
            return
        frame.split = split
        try:
            self._project.save()
        except OSError as e:
            QMessageBox.warning(self, "Save project", f"Could not save the project: {e}")
        # Not grid.refresh(): that re-filters, and the open image must stay in the steps.
        self._update_frame_item(self._frame_index)
        self._refresh_frame_header()

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
        self._split_combo.blockSignals(True)
        self._split_combo.setCurrentIndex(max(0, self._split_combo.findData(frame.split)))
        self._split_combo.blockSignals(False)
        self._refresh_frame_header()
        self._filmstrip.select_row(index)
        self._baseline = instances
        # Any SAM answer still on its way is for the previous frame's clicks.
        self._sam_request_id += 1
        self._canvas.set_frame(image, instances)
        self._canvas.setFocus()
        if image is None:
            self._hint.setText(f"<b>{frame.name}.jpg</b> is missing from the project folder.")

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
        """Esc: drop the polygon being drawn or the Auto tool's clicks; else back to the grid."""
        canvas = self._seg_canvas()
        if canvas is not None and (canvas.drafting or canvas.has_prompts or canvas.has_proposal):
            canvas.cancel()
            canvas.setFocus()
            return
        self.show_grid()

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

    def _set_sam_model(self, file: str) -> None:
        """The Auto tool's menu: which SAM model proposes outlines, remembered."""
        self._sam_model_file = file
        self._settings.setValue(_SETTINGS_SAM_MODEL, file)
        self._request_sam()

    def _ensure_sam_worker(self) -> SamWorker:
        if self._sam_worker is None:
            self._sam_worker = SamWorker()
            self._sam_thread = QThread(self)
            self._sam_worker.moveToThread(self._sam_thread)
            self._sam_request.connect(self._sam_worker.run, Qt.ConnectionType.QueuedConnection)
            self._sam_worker.result.connect(self._on_sam_result)
            self._sam_worker.status.connect(self._on_sam_status)
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
            self._sam_model_file,
            key,
            image,
            points,
            labels,
            box,
        )

    def _on_sam_status(self, request: int, text: str) -> None:
        if request == self._sam_request_id:  # a stale request's progress is not news
            self._hint.setText(text)

    def _on_sam_result(self, request: int, polygon, error: str) -> None:
        if request != self._sam_request_id:
            return  # the clicks changed again since; a newer answer is coming
        canvas = self._seg_canvas()
        if canvas is None:
            return
        canvas.busy = False
        if error:
            self._hint.setText(f"Auto failed: {error}")
            return
        if not polygon:
            canvas.set_proposal(None)  # refreshes the hint, so the news goes after
            self._hint.setText("No outline found here; try another click or a box.")
            return
        canvas.set_proposal(polygon)  # the hint now says Enter keeps it

    def _refresh_seg_state(self) -> None:
        canvas = self._seg_canvas()
        if canvas is None or not canvas.has_image:
            self._hint.setText("")
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
        canvas = self._canvas
        if canvas is None or not canvas.has_image:
            self._hint.setText("")
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

    def _step_frame(self, delta: int) -> None:
        """Previous / next image among the grid's rows, as filtered when the editor opened.

        If the open image is no longer among them, step from where it would sit.
        """
        rows = self._gallery.visible_rows()
        here = self._frame_index
        if here in rows:
            pos = rows.index(here) + delta
            target = rows[pos] if 0 <= pos < len(rows) else None
        elif delta > 0:
            target = next((r for r in rows if r > here), None)
        else:
            target = next((r for r in reversed(rows) if r < here), None)
        if target is not None:
            self._show_frame(target)

    # ------------------------------------------------------------------
    # projects
    # ------------------------------------------------------------------

    def _default_location(self) -> Path:
        from glider.first_run import default_data_dir

        return default_data_dir() / "animal_annotator"

    def _new_project(self) -> None:
        if self._thread is not None:
            return
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

    def _open_project_dialog(self) -> None:
        if self._thread is not None:
            return
        start = str(self._project.folder.parent) if self._project else str(self._default_location())
        path = QFileDialog.getExistingDirectory(self, "Open Animal Annotator project", start)
        if path:
            self.open_project(Path(path))

    def open_project(self, folder: Path, quiet: bool = False) -> bool:
        """Open the project in ``folder``; says why if it can't, unless ``quiet``."""
        if self._thread is not None:
            return False
        try:
            project = StudioProject.load(folder)
        except StudioProjectError as e:
            if not quiet:
                QMessageBox.warning(self, "Open project", str(e))
            return False
        if ensure_splits(project):  # a project from before splits were stored
            try:
                project.save()
            except OSError as e:
                logger.warning("could not save the split migration for %s: %s", folder, e)
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
        if not on_screen and self._stack.currentIndex() == 1:
            self._show_frame(edit.frame)
        self._update_frame_item(edit.frame)
        self._refresh_undo_buttons()

    def _set_project(self, project: StudioProject | None) -> None:
        self._save_current_frame()
        self._history.clear()
        self._baseline = []
        self._project = project
        self._frame_index = -1
        if project is not None:
            self._settings.setValue(_SETTINGS_LAST, str(project.folder))
            self.setWindowTitle(f"Animal Annotator — {project.name}")
        else:
            self.setWindowTitle("Animal Annotator")
        self._make_canvas()
        self._grid.set_project(project)
        self._stack.setCurrentIndex(0)
        self._refresh_all()
        self._refresh_undo_buttons()

    def _refresh_all(self) -> None:
        self._gallery.gallery_model.reload()
        self._grid.refresh()
        if self._frame_index >= 0:
            self._show_frame(self._frame_index)
        self._refresh_label_state()

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
            f"{len(result.train)} training frame(s), {len(result.val)} validation frame(s)"
            f"{f', {len(result.test)} test frame(s)' if result.test else ''}."
        )
        if notice := leak_notice(self._project):
            text += "\n\n" + notice
        text += (
            "\n\nTo train on it here, use Train Model. Elsewhere: "
            f'yolo train data="{result.data_yaml}" model=yolo11n-'
            f"{'seg' if self._project.is_segment else 'pose'}.pt"
        )
        QMessageBox.information(self, "Export dataset", text)

    def _train(self) -> None:
        if self._project is None or self._thread is not None:
            return
        from glider.gui.animal_annotator.train_dialog import TrainDialog

        self._save_current_frame()
        TrainDialog(self._project, parent=self).exec()

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
