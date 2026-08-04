"""Batch Pose Tracking window.

Point it at directories of videos, pick a trained YOLO-pose ``.pt``, and it
writes a DeepLabCut CSV beside each video by driving
:func:`glider.vision.pose.batch.run_batch` on a QThread.

Import cost: this module pulls in the pose stack, so ``MainWindow`` imports it
lazily inside its menu handler rather than at startup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PyQt6.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from glider.gui.pose_batch.calibration_table import CalibrationTable
from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import (
    GUTTER,
    Card,
    RunRail,
    ToolHeader,
    apply_tool_theme,
    attach_empty_state,
    hint,
    labelled_row,
    scroll_column,
    set_button_role,
    set_text_role,
)
from glider.vision.calibration import CameraCalibration
from glider.vision.calibration_set import CalibrationSet, CalibrationSetError
from glider.vision.pose import batch as batch_core

logger = logging.getLogger(__name__)

# Reading a model's own keypoint names landed on the pose-kpt-names branch.
# TEMPORARY: delete this seam (and the None checks below) once that merges to
# main — the tool degrades to manual entry without it, it does not break.
try:
    from glider.vision.pose.model_meta import read_pose_model_meta
except ImportError:  # pragma: no cover - depends on branch state
    read_pose_model_meta = None

_SETTINGS_PREFIX = "pose_batch/keypoints"
_INVALID_STYLE = f"border: 1px solid {colors.ERROR};"


class _DropList(QListWidget):
    """A list of video directories/files that accepts drag-and-drop."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setToolTip("Drag video folders or files here")

    # Qt camelCase overrides (N802 is suppressed project-wide).
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return
        dropped = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        self.add_paths(dropped)
        event.acceptProposedAction()

    def add_paths(self, paths) -> None:
        """Add paths, ignoring ones already listed."""
        existing = {self.item(i).text() for i in range(self.count())}
        added = False
        for path in paths:
            text = str(Path(path))
            if text not in existing:
                self.addItem(text)
                existing.add(text)
                added = True
        if added:
            self.changed.emit()

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self.changed.emit()

    def clear_all(self) -> None:
        self.clear()
        self.changed.emit()

    def paths(self) -> list[Path]:
        return [Path(self.item(i).text()) for i in range(self.count())]


class _MetaWorker(QObject):
    """Reads a model's keypoint metadata off the GUI thread.

    Reading a ``.pt`` loads the checkpoint through ultralytics, which imports
    torch and takes seconds — never do it on the UI thread.
    """

    done = pyqtSignal(object)  # PoseModelMeta | None

    def __init__(self, model_path: Path):
        super().__init__()
        self._model_path = model_path

    def run(self) -> None:
        if read_pose_model_meta is None:
            self.done.emit(None)
            return
        try:
            self.done.emit(read_pose_model_meta(self._model_path))
        except Exception:  # never block the user on a metadata read
            logger.warning("could not read pose metadata", exc_info=True)
            self.done.emit(None)


class PoseBatchWindow(QMainWindow):
    """Batch pose inference over directories of videos."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Pose Tracking")
        # Wide enough for both columns. The old 820 was sized for a single
        # stacked column; against the config/run split it left the config side
        # ~370px and clipped the weights field, the confidence spin box and
        # half the calibration table. The minimum stops a drag doing the same.
        self.resize(1180, 900)
        self.setMinimumWidth(940)

        self._model_path: Path | None = None
        self._meta = None
        self._videos: list[Path] = []
        self._calibrations = CalibrationSet()
        self._loaded_master: Path | None = None
        # The last value we defaulted into the master field. While the field
        # still holds it, the path is ours to keep in step with the videos;
        # once it differs, the operator owns it and we never touch it again.
        self._auto_master_text: str | None = None
        self._thread: QThread | None = None
        self._worker = None
        self._meta_thread: QThread | None = None
        self._meta_worker: _MetaWorker | None = None

        self._build_ui()
        self._refresh_videos()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("ToolPage")
        page = QVBoxLayout(central)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)

        header = ToolHeader(
            "Batch Pose Tracking",
            "Run a YOLO-pose model over folders of video and write a DLC CSV beside each one",
        )
        page.addWidget(header)

        body = QHBoxLayout()
        body.setContentsMargins(GUTTER, GUTTER, GUTTER, GUTTER)
        body.setSpacing(GUTTER)
        page.addLayout(body, 1)

        # --- left: configuration ----------------------------------------
        area, column = scroll_column()
        column.addWidget(self._build_model_group())
        # Videos and calibration still share a splitter -- the calibration
        # table is the one thing here whose useful height genuinely depends on
        # the cohort size, and trading rows against it is worth a handle. The
        # log moved to the run rail, where output belongs.
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(GUTTER)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_sources_group())
        splitter.addWidget(self._build_calibration_group())
        splitter.setSizes([220, 340])
        column.addWidget(splitter, 1)
        column.addWidget(self._build_filter_group())
        column.addWidget(self._build_options_group())
        column.addStretch(0)
        body.addWidget(area, 1)

        # --- right: run rail --------------------------------------------
        rail = RunRail("Track poses")
        self._rail = rail
        self._run_button = rail.button
        self._run_button.clicked.connect(self._start)
        self._cancel_button = rail.secondary
        self._cancel_button.setVisible(True)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel)

        # Hidden until a batch starts: two empty bars sitting at zero read as
        # a run that has stalled rather than one that has not begun.
        self._overall_bar = QProgressBar()
        self._overall_bar.setFormat("%v / %m videos")
        self._overall_bar.setVisible(False)
        rail.card.add(self._overall_bar)

        self._video_bar = QProgressBar()
        self._video_bar.setFormat("%v / %m frames")
        self._video_bar.setVisible(False)
        rail.card.add(self._video_bar)

        log_card = Card("Progress log")
        self._log = QPlainTextEdit()
        self._log.setObjectName("LogPane")
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Progress will appear here.")
        log_card.add(self._log, 1)
        rail.add(log_card, 1)

        rail.setFixedWidth(400)
        body.addWidget(rail)

        self.setCentralWidget(central)
        # Opened with parent=None, so nothing hands it the app theme.
        apply_tool_theme(self)

    def _build_model_group(self) -> Card:
        card = Card("Model")

        self._model_field = QLineEdit()
        self._model_field.setReadOnly(True)
        self._model_field.setPlaceholderText("Select a trained YOLO-pose .pt")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_model)
        card.add_row("Weights", self._model_field, browse)

        self._names_field = QLineEdit()
        self._names_field.setPlaceholderText("nose, l_ear, r_ear, …")
        self._names_field.textChanged.connect(self._validate)
        edit_schema = QPushButton("Edit…")
        edit_schema.setToolTip(
            "Arrange the bodyparts on a figure instead of typing the order, and "
            "save the layout for reuse. These names are baked into every CSV "
            "this batch writes, so a wrong order propagates downstream."
        )
        edit_schema.clicked.connect(self._edit_keypoint_schema)
        card.add_row("Bodyparts", self._names_field, edit_schema)

        self._names_status = hint("Select a model to load its keypoints.")
        card.add(self._names_status)
        card.add_separator()

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self._device_combo.setMaximumWidth(150)

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)  # matches infer_video's default

        # Two short numeric settings on one line: each is narrow, and stacking
        # them cost a full row apiece for a value four characters wide.
        inference = QHBoxLayout()
        inference.setSpacing(8)
        inference.addWidget(self._device_combo)
        inference.addSpacing(8)
        inference.addWidget(set_text_role(QLabel("Confidence"), "caption"))
        inference.addWidget(self._conf_spin)
        inference.addStretch(1)
        card.add(labelled_row("Device", inference))

        self._require_gpu = QCheckBox("Fail if no GPU is available")
        self._require_gpu.setToolTip(
            "Prevents a silent CPU fallback turning an overnight batch into a multi-day one."
        )
        card.add(self._require_gpu)
        return card

    def _build_sources_group(self) -> Card:
        card = Card("Videos", "drop folders or files here")
        self._sources_card = card

        self._sources = _DropList()
        self._sources.changed.connect(self._refresh_videos)
        attach_empty_state(
            self._sources, "Drop video folders or files here,\nor use the buttons below."
        )
        card.add(self._sources, 1)

        add_dir = QPushButton("Add Directory…")
        add_dir.clicked.connect(self._choose_directory)
        add_files = QPushButton("Add Files…")
        add_files.clicked.connect(self._choose_files)
        remove = QPushButton("Remove")
        set_button_role(remove, "ghost")
        remove.clicked.connect(self._sources.remove_selected)
        clear = QPushButton("Clear")
        set_button_role(clear, "ghost")
        clear.clicked.connect(self._sources.clear_all)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        for button in (add_dir, add_files, remove, clear):
            buttons.addWidget(button)
        buttons.addStretch(1)
        card.add(buttons)

        self._recursive = QCheckBox("Include subdirectories")
        self._recursive.setChecked(True)
        self._recursive.toggled.connect(self._refresh_videos)
        card.add(self._recursive)

        # Kept as the count of record; the card badge is where it is read now.
        self._count_label = QLabel("No videos found")
        self._count_label.setVisible(False)
        card.add(self._count_label)
        return card

    def _build_calibration_group(self) -> Card:
        card = Card("Pixel-to-distance calibration", "required for every video")
        self._calibration_card = card
        card.setToolTip(
            "Every video needs a scale before the batch can run. The DLC CSVs "
            "stay in pixels; the scale is written to the master calibration file."
        )

        self._cal_table = CalibrationTable()
        self._cal_table.set_calibration_set(self._calibrations)
        self._cal_table.calibrate_requested.connect(self._open_calibration)
        # ~4-5 rows: enough to see calibration status at a glance even if the
        # splitter above gets dragged down to its floor.
        self._cal_table.setMinimumHeight(140)
        card.add(self._cal_table, 1)

        calibrate = QPushButton("Calibrate…")
        calibrate.setToolTip("Calibrate the selected video (or double-click its row)")
        calibrate.clicked.connect(self._calibrate_selected)
        copy_btn = QPushButton("Copy to Selected")
        copy_btn.setToolTip(
            "Stamp one calibration onto the other selected videos — for videos "
            "shot on the same rig at the same camera height."
        )
        copy_btn.clicked.connect(self._copy_calibration_to_selected)
        clear_btn = QPushButton("Clear")
        set_button_role(clear_btn, "ghost")
        clear_btn.clicked.connect(self._clear_selected_calibrations)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        for button in (calibrate, copy_btn, clear_btn):
            buttons.addWidget(button)
        buttons.addStretch(1)
        card.add(buttons)
        card.add_separator()

        self._master_field = QLineEdit()
        self._master_field.setPlaceholderText("Master calibration file")
        # A typed or pasted path must get the same load-if-exists treatment as
        # Browse and Load, or Run silently overwrites a master it never read.
        # editingFinished (not textChanged) so it fires once the path is whole.
        self._master_field.editingFinished.connect(self._master_path_edited)
        master_browse = QPushButton("Browse…")
        master_browse.clicked.connect(self._choose_master_path)
        load_master = QPushButton("Load")
        set_button_role(load_master, "ghost")
        load_master.clicked.connect(self._load_master_clicked)
        save_master = QPushButton("Save")
        set_button_role(save_master, "ghost")
        save_master.clicked.connect(self._save_master_clicked)
        card.add_row("Master file", self._master_field, master_browse, load_master, save_master)
        return card

    def _build_filter_group(self) -> Card:
        card = Card("Post-process filtering", "optional")
        self._filter_card = card

        # Still a checkable QGroupBox: it is the enable switch the rest of the
        # section hangs off, and _filter_settings reads isChecked().
        group = QGroupBox("Smooth and gap-fill the tracked keypoints")
        group.setCheckable(True)
        group.setChecked(False)
        group.setToolTip(
            "Writes the smoothed result as the main CSV and keeps the raw "
            "inference alongside it as *_raw.csv."
        )
        group.toggled.connect(lambda on: card.set_badge("on" if on else "off"))
        box = QVBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)

        self._filter_conf = QDoubleSpinBox()
        self._filter_conf.setRange(0.0, 1.0)
        self._filter_conf.setSingleStep(0.05)
        self._filter_conf.setValue(0.5)
        box.addLayout(labelled_row("Min confidence", self._filter_conf))

        self._filter_gap = QSpinBox()
        self._filter_gap.setRange(0, 500)
        self._filter_gap.setValue(5)
        self._filter_gap.setSuffix(" frames")
        box.addLayout(labelled_row("Max gap", self._filter_gap))

        self._filter_window = QSpinBox()
        self._filter_window.setRange(3, 99)
        self._filter_window.setSingleStep(2)
        self._filter_window.setValue(5)
        # median_filter rejects even windows, so keep the widget odd-only.
        self._filter_window.valueChanged.connect(self._force_odd_window)
        box.addLayout(labelled_row("Median window", self._filter_window))

        card.add(group)
        card.set_badge("off")
        self._filter_group = group
        return card

    def _build_options_group(self) -> Card:
        card = Card("Options")
        self._overwrite = QCheckBox("Overwrite existing outputs")
        self._overwrite.setToolTip(
            "Off by default, so an interrupted batch resumes without redoing finished videos."
        )
        self._overwrite.toggled.connect(self._validate)
        card.add(self._overwrite)
        card.add(hint("Leave off to resume an interrupted batch without redoing finished videos."))
        return card

    # ------------------------------------------------------------------
    # inputs
    # ------------------------------------------------------------------

    def _force_odd_window(self, value: int) -> None:
        if value % 2 == 0:
            self._filter_window.setValue(value + 1)

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO-pose weights", "", "PyTorch weights (*.pt)"
        )
        if not path:
            return
        self._model_path = Path(path)
        self._model_field.setText(str(self._model_path))
        self._names_status.setText("Reading keypoints from the model…")
        self._start_meta_read()
        self._validate()

    def _choose_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add video directory")
        if path:
            self._sources.add_paths([Path(path)])

    def _choose_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(batch_core.VIDEO_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", f"Videos ({exts})")
        if paths:
            self._sources.add_paths([Path(p) for p in paths])

    def _refresh_videos(self) -> None:
        # Same call Run uses, so the count can never disagree with the work
        # actually done. On a very large tree this touches disk synchronously.
        try:
            self._videos = batch_core.discover_videos(
                self._sources.paths(), recursive=self._recursive.isChecked()
            )
        except OSError:
            logger.warning("could not scan for videos", exc_info=True)
            self._videos = []
        count = len(self._videos)
        self._count_label.setText(
            "No videos found" if count == 0 else f"{count} video{'s' if count != 1 else ''} found"
        )
        self._sources_card.set_badge(
            "none yet" if count == 0 else f"{count} video{'s' if count != 1 else ''}"
        )
        self._cal_table.set_videos(self._videos)
        self._sync_master_field()
        self._validate()

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def _open_calibration(self, video: Path) -> None:
        """Draw measurement lines on a frame scrubbed out of *video*."""
        from glider.gui.dialogs.calibration_dialog import CalibrationDialog
        from glider.vision.frame_provider import VideoFrameProvider

        provider = VideoFrameProvider(video)
        if not provider.is_connected:
            provider.release()
            QMessageBox.warning(
                self,
                "Cannot Open Video",
                f"{video.name} could not be opened for calibration.",
            )
            return

        # Edit a copy: Cancel must leave the stored calibration untouched.
        existing = self._calibrations.get(video)
        working = (
            CameraCalibration.from_dict(existing.to_dict())
            if existing is not None
            else CameraCalibration()
        )
        dialog = None
        try:
            dialog = CalibrationDialog(
                frame_provider=provider,
                calibration=working,
                parent=self,
                show_file_buttons=False,
            )
            dialog.setWindowTitle(f"Calibrate — {video.name}")
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._calibrations.set(video, dialog.get_calibration())
                ppm = self._calibrations.px_per_mm(video)
                self._log.appendPlainText(
                    f"{video.name}: {ppm:.3f} px/mm"
                    if ppm
                    else f"{video.name}: no usable scale drawn"
                )
        finally:
            if dialog is not None:
                # parent=self hands ownership to the window, so dropping the
                # Python reference leaves the C++ dialog — and the full-res
                # frame and pixmap it holds — alive for the session.
                dialog.deleteLater()
            provider.release()

        self._cal_table.refresh()
        self._validate()

    def _calibrate_selected(self) -> None:
        selected = self._cal_table.selected_videos()
        if not selected:
            QMessageBox.information(
                self, "Calibrate", "Select a video in the calibration table first."
            )
            return
        self._open_calibration(selected[0])

    def _retarget_calibration(
        self, template: CameraCalibration, video: Path
    ) -> CameraCalibration | None:
        """A copy of *template* reconstructed at *video*'s own resolution.

        Lines are stored normalized, so on a same-rig video the ruler spans the
        same fraction of the frame whatever the recording resolution — but
        ``pixels_per_mm`` reconstructs pixels at ``calibration_width``. Carrying
        the source's resolution over would report the source's scale for a video
        that does not have it. Returns None when the video will not open, since
        guessing its resolution is exactly the error being fixed.
        """
        from glider.vision.video_source import VideoFileSource

        reader = VideoFileSource()
        try:
            if not reader.load(video):
                return None
            width, height = reader.resolution
        finally:
            reader.release()
        if width <= 0 or height <= 0:
            return None

        # from_dict(to_dict()) is a deep copy: the videos must not share
        # a CameraCalibration, or editing one silently edits the others.
        copy = CameraCalibration.from_dict(template.to_dict())
        copy.calibration_width = width
        copy.calibration_height = height
        return copy

    def _copy_calibration_to_selected(self) -> None:
        """Stamp the one calibrated selected video onto the rest of the selection."""
        selected = self._cal_table.selected_videos()
        sources = [v for v in selected if self._calibrations.px_per_mm(v) is not None]
        if not sources:
            QMessageBox.information(
                self,
                "Copy Calibration",
                "Select one calibrated video plus the videos to copy it to.",
            )
            return

        source = sources[0]
        template = self._calibrations.get(source)
        targets = [v for v in selected if v != source]
        already = {v for v in targets if self._calibrations.px_per_mm(v) is not None}

        if already and not self._confirm_overwrite(source, len(already)):
            # Declined: filling the blanks is what "copy to selected" is for,
            # and it is the half of the job that cannot destroy anything.
            targets = [v for v in targets if v not in already]

        filled = 0
        overwritten = 0
        skipped: list[Path] = []
        for video in targets:
            retargeted = self._retarget_calibration(template, video)
            if retargeted is None:
                skipped.append(video)
                continue
            self._calibrations.set(video, retargeted)
            if video in already:
                overwritten += 1
            else:
                filled += 1

        self._log.appendPlainText(
            f"Copied {source.name}'s calibration: {filled} uncalibrated video(s) filled, "
            f"{overwritten} existing calibration(s) overwritten."
        )
        if skipped:
            names = ", ".join(v.name for v in skipped)
            self._log.appendPlainText(
                f"Not copied to {len(skipped)} video(s) that could not be opened to "
                f"read their resolution: {names}"
            )
        self._cal_table.refresh()
        self._validate()

    def _confirm_overwrite(self, source: Path, count: int) -> bool:
        """Ask before replacing calibrations the operator drew themselves."""
        answer = QMessageBox.question(
            self,
            "Overwrite Calibrations?",
            f"{count} of the selected video(s) already have their own calibration.\n\n"
            f"Overwrite them with {source.name}'s?\n"
            "Choose No to fill only the uncalibrated videos.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _clear_selected_calibrations(self) -> None:
        for video in self._cal_table.selected_videos():
            self._calibrations.discard(video)
        self._cal_table.refresh()
        self._validate()

    # ------------------------------------------------------------------
    # master calibration file
    # ------------------------------------------------------------------

    def _default_master_path(self) -> Path | None:
        """``pose_calibration.json`` in the videos' common parent."""
        if not self._videos:
            return None
        try:
            parent = Path(os.path.commonpath([str(v.parent) for v in self._videos]))
        except ValueError:
            # Videos span drive roots on Windows; commonpath refuses.
            parent = self._videos[0].parent
        return parent / "pose_calibration.json"

    def _master_path(self) -> Path | None:
        text = self._master_field.text().strip()
        return Path(text) if text else self._default_master_path()

    @staticmethod
    def _exists(path: Path) -> bool:
        """Path.exists() that also survives a path the OS rejects outright.

        The field is free text, so an embedded null byte reaches here as a
        ValueError rather than a plain "no".
        """
        try:
            return path.exists()
        except (OSError, ValueError):
            return False

    def _master_is_auto(self) -> bool:
        """True while the field still holds a value we defaulted, not one typed."""
        return self._master_field.text().strip() == (self._auto_master_text or "")

    def _sync_master_field(self) -> None:
        """Keep an auto-defaulted path following the videos, then load if it exists."""
        default = self._default_master_path()
        # Only ever re-point a path we chose: an operator's own path must not
        # move under them, but a stale default belonging to a folder that is no
        # longer listed would write the master where nobody will look for it.
        if default is not None and self._master_is_auto():
            text = str(default)
            if text != self._master_field.text():
                self._auto_master_text = text
                self._master_field.setText(text)

        path = self._master_path()
        # Load an existing master once per path, so a re-run costs no re-drawing.
        if path is not None and path != self._loaded_master and self._exists(path):
            self._loaded_master = path
            self._load_master(path)

    def _master_path_edited(self) -> None:
        """Route a hand-typed path through the same load-if-exists as Browse."""
        if not self._master_field.text().strip():
            # Cleared: hand the field back to the auto-default machinery.
            self._auto_master_text = None
            self._sync_master_field()
            return
        path = self._master_path()
        # _loaded_master is the guard against re-loading, so a focus-out that
        # changed nothing costs nothing and no path can reach Run unread.
        if path is None or path == self._loaded_master or not self._exists(path):
            return
        self._loaded_master = path
        self._load_master(path)

    def _load_master(self, path: Path) -> None:
        try:
            loaded = CalibrationSet.load(path, known_videos=self._videos)
        except CalibrationSetError as e:
            # Never half-apply: leave the current state alone and say why.
            self._log.appendPlainText(f"Could not read {path.name}: {e}")
            QMessageBox.warning(self, "Calibration File", str(e))
            return

        self._calibrations.entries.update(loaded.entries)
        self._log.appendPlainText(
            f"Loaded calibration for {len(loaded.entries)} video(s) from {path.name}."
        )
        self._cal_table.refresh()
        self._validate()

    def _choose_master_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Master calibration file",
            self._master_field.text(),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self._master_field.setText(path)
            self._auto_master_text = None  # deliberately chosen: stop re-defaulting it
            self._loaded_master = None  # a new path may want loading
            self._sync_master_field()

    def _load_master_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load master calibration",
            self._master_field.text(),
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self._master_field.setText(path)
            self._auto_master_text = None  # deliberately chosen: stop re-defaulting it
            self._loaded_master = Path(path)
            self._load_master(Path(path))

    def _save_master_clicked(self) -> None:
        path = self._master_path()
        if path is None or not self._videos:
            QMessageBox.information(
                self, "Master File", "Add videos first so the file has somewhere to go."
            )
            return
        if not self._write_master(path):
            return
        self._log.appendPlainText(f"Wrote {path}")

    def _write_master(self, path: Path) -> bool:
        """Write the master file. Reports and returns False on failure."""
        try:
            # Only the listed batch: the set can also hold videos from folders
            # visited earlier this session, which this file does not describe.
            self._calibrations.subset(self._videos).save(path, model=self._model_path)
        except (OSError, ValueError) as e:
            self._log.appendPlainText(f"Could not write {path}: {e}")
            QMessageBox.critical(self, "Master File", f"Could not write the calibration file:\n{e}")
            return False
        self._loaded_master = path
        return True

    # ------------------------------------------------------------------
    # keypoint names
    # ------------------------------------------------------------------

    def _start_meta_read(self) -> None:
        self._stop_meta_thread()
        if self._model_path is None:
            return
        self._meta_thread = QThread(self)
        self._meta_worker = _MetaWorker(self._model_path)
        self._meta_worker.moveToThread(self._meta_thread)
        self._meta_thread.started.connect(self._meta_worker.run)
        self._meta_worker.done.connect(self._apply_meta)
        self._meta_thread.start()

    def _stop_meta_thread(self) -> None:
        if self._meta_thread is not None:
            self._meta_thread.quit()
            self._meta_thread.wait(5000)
            self._meta_thread.deleteLater()
            self._meta_thread = None
            self._meta_worker = None

    def _apply_meta(self, meta) -> None:
        """Resolve bodypart names: remembered -> model's own -> generated."""
        self._meta = meta
        self._stop_meta_thread()

        remembered = self._remembered_names()
        if remembered:
            names, source = remembered, "remembered from your last run"
        elif meta is not None and meta.keypoint_names:
            names, source = list(meta.keypoint_names), f"read from the model ({meta.source})"
        elif meta is not None and meta.n_keypoints:
            names = [f"kp{i}" for i in range(meta.n_keypoints)]
            source = (
                f"the model reports {meta.n_keypoints} keypoints but no names — "
                "rename these to match your training data.yaml"
            )
        else:
            names, source = [], "could not read the model's keypoints — enter them manually"

        self._names_field.setText(", ".join(names))
        self._names_status.setText(source)
        self._validate()

    def _edit_keypoint_schema(self) -> None:
        """Arrange the bodyparts on a figure, then write their order back.

        This is where the names are first chosen and baked into every CSV the
        batch writes, so getting the order right here stops a wrong schema
        propagating into every downstream artifact.
        """
        from glider.analysis.behavior.keypoint_schema import Keypoint, KeypointSchema
        from glider.gui.behavior.keypoint_editor import KeypointEditorDialog

        current = self._current_names()
        if current:
            step = 1.0 / (len(current) + 1)
            schema = KeypointSchema(
                [Keypoint(n, 0.5, step * (i + 1)) for i, n in enumerate(current)]
            )
        else:
            schema = KeypointSchema.default_mouse()

        dialog = KeypointEditorDialog(schema, parent=self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._names_field.setText(", ".join(dialog.names()))
        finally:
            dialog.deleteLater()

    def _settings_key(self) -> str | None:
        if self._model_path is None:
            return None
        try:
            resolved = self._model_path.resolve()
        except OSError:
            resolved = self._model_path
        return f"{_SETTINGS_PREFIX}/{resolved}"

    def _remembered_names(self) -> list[str]:
        key = self._settings_key()
        if key is None:
            return []
        stored = QSettings().value(key, "")
        return self._parse_names(stored) if isinstance(stored, str) else []

    def _remember_names(self, names) -> None:
        key = self._settings_key()
        if key is not None:
            QSettings().setValue(key, ", ".join(names))

    @staticmethod
    def _parse_names(text: str) -> list[str]:
        return [part.strip() for part in text.split(",") if part.strip()]

    def _current_names(self) -> list[str]:
        return self._parse_names(self._names_field.text())

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Disable Run — and say why — rather than failing an overnight batch."""
        if self._thread is not None:
            return  # a run is in flight; Run stays disabled

        names = self._current_names()
        expected = getattr(self._meta, "n_keypoints", None)

        problem = None
        names_bad = False
        if self._model_path is None:
            problem = "Select a model."
        elif not names:
            problem = "Enter the bodypart names, in the model's keypoint order."
            names_bad = True
        elif len(set(names)) != len(names):
            problem = "Bodypart names must be unique."
            names_bad = True
        elif expected and len(names) != expected:
            problem = f"The model has {expected} keypoints but {len(names)} names were given."
            names_bad = True
        elif not self._videos:
            problem = "Add at least one video or directory."
        else:
            uncalibrated = self._calibrations.missing(self._videos)
            if uncalibrated:
                problem = f"{len(uncalibrated)} video(s) still need calibration."

        self._names_field.setStyleSheet(_INVALID_STYLE if names_bad else "")
        self._run_button.setEnabled(problem is None)
        self._run_button.setToolTip(problem or "")
        # Shown next to the button as well as on hover: a disabled Run with the
        # reason only in a tooltip is a dead end you have to go hunting in.
        self._rail.set_blocker(problem or "")

        if self._videos:
            missing = len(self._calibrations.missing(self._videos))
            done = len(self._videos) - missing
            self._calibration_card.set_badge(f"{done} / {len(self._videos)} calibrated")
        else:
            self._calibration_card.set_badge("")

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------

    def _filter_settings(self):
        if not self._filter_group.isChecked():
            return None
        window = self._filter_window.value()
        return batch_core.FilterSettings(
            confidence_threshold=self._filter_conf.value(),
            max_gap=self._filter_gap.value(),
            median_window=window if window % 2 else window + 1,
        )

    def _start(self) -> None:
        if self._thread is not None:
            return

        # Written before inference begins — the calibration is complete and
        # known now, so it survives a cancelled or failed batch. A path we
        # cannot write is a reason not to start at all, rather than to discover
        # it after an hour of GPU time.
        master = self._master_path()
        if master is not None and not self._write_master(master):
            return

        self._start_worker()

    def _start_worker(self) -> None:
        names = self._current_names()
        device = self._device_combo.currentText()

        from glider.gui.pose_batch.worker import PoseBatchWorker

        self._log.clear()
        self._log.appendPlainText(f"Starting {len(self._videos)} video(s)…")
        self._overall_bar.setRange(0, len(self._videos))
        self._overall_bar.setValue(0)
        self._video_bar.setRange(0, 0)

        self._worker = PoseBatchWorker(
            self._videos,
            self._model_path,
            names,
            conf=self._conf_spin.value(),
            device=None if device == "auto" else device,
            require_gpu=self._require_gpu.isChecked(),
            overwrite=self._overwrite.isChecked(),
            filtering=self._filter_settings(),
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.video_progress.connect(self._on_video_progress)
        self._worker.log.connect(self._log.appendPlainText)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._overall_bar.setVisible(True)
        self._video_bar.setVisible(True)
        self._rail.status.set_state("running", "Tracking")
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_button.setEnabled(False)
            self._rail.status.set_state("warn", "Cancelling")
            self._log.appendPlainText("Cancelling after the current frame…")

    def _on_progress(self, index: int, total: int) -> None:
        self._overall_bar.setRange(0, total)
        self._overall_bar.setValue(index)

    def _on_video_progress(self, done: int, total: int) -> None:
        # total == 0 means OpenCV couldn't report a frame count: show a busy bar.
        self._video_bar.setRange(0, total)
        self._video_bar.setValue(done)

    def _on_finished(self, result) -> None:
        self._overall_bar.setValue(self._overall_bar.maximum())
        self._video_bar.setRange(0, 1)
        self._video_bar.setValue(1)
        if result.completed:
            self._remember_names(self._current_names())
        if result.failed:
            self._rail.status.set_state("warn", f"{len(result.failed)} failed")
        elif result.completed:
            self._rail.status.set_state("ok", "Done")
        else:
            self._rail.status.set_state("idle", "Cancelled")
        for video, message in result.failed:
            self._log.appendPlainText(f"FAILED {video.name}: {message}")
        self._teardown_thread()

    def _on_failed(self, message: str) -> None:
        self._rail.status.set_state("error", "Failed")
        self._log.appendPlainText(f"Batch could not start: {message}")
        QMessageBox.critical(self, "Batch Pose Tracking", message)
        self._teardown_thread()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
            self._thread = None
        self._worker = None
        self._video_bar.setRange(0, 1)
        self._cancel_button.setEnabled(False)
        self._validate()

    def closeEvent(self, event):
        """Never let a running batch outlive its window."""
        if self._thread is not None and self._worker is not None:
            self._worker.cancel()
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
            self._worker = None
        self._stop_meta_thread()
        super().closeEvent(event)
