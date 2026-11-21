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
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glider.core.experiment_types import ANALYSIS_PUP_RETRIEVAL, has_analysis
from glider.gui.pose_batch.calibration_table import CalibrationTable
from glider.gui.qthreads import retire_thread
from glider.gui.styles import colors
from glider.gui.widgets.tool_ui import (
    GUTTER,
    Card,
    StatusPill,
    ToolHeader,
    apply_tool_theme,
    attach_empty_state,
    hint,
    labelled_row,
    scroll_column,
    set_button_role,
    set_text_role,
)
from glider.vision.arena import ArenaCalibration, DegenerateArenaError
from glider.vision.calibration_set import CalibrationSet, CalibrationSetError
from glider.vision.pose import batch as batch_core
from glider.vision.pose.spec import read_pose_model_meta

if TYPE_CHECKING:  # pragma: no cover - typing only
    from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)

_SETTINGS_PREFIX = "pose_batch/keypoints"
_INVALID_STYLE = f"border: 1px solid {colors.ERROR};"


class _Sources:
    """The video folders/files the user added, in order, without duplicates.

    Plain data: the table shows the videos they expand to, so nothing here is a
    widget. ``on_change`` fires whenever the set of sources changes.
    """

    def __init__(self, on_change) -> None:
        self._paths: list[Path] = []
        self._on_change = on_change

    def add_paths(self, paths) -> None:
        """Add paths, ignoring ones already listed."""
        added = False
        for path in map(Path, paths):
            if path not in self._paths:
                self._paths.append(path)
                added = True
        if added:
            self._on_change()

    def clear_all(self) -> None:
        self._paths.clear()
        self._on_change()

    def paths(self) -> list[Path]:
        return list(self._paths)


class _StepStrip(QFrame):
    """Model -> Videos -> Arenas -> Track, with the first unfinished step lit.

    Drawn like Behavior Analysis's step tabs -- a numbered badge (a tick once
    done) beside each title, the current step underlined -- so every
    step-by-step tool reads the same. Here it is one page, so the steps are
    progress, not tabs to click.

    Display only: the window's validation decides what is done and hands the
    counts in, so this can never disagree with what the Track button allows.
    """

    _NAMES = ("Model", "Videos", "Arenas", "Track")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StepStrip")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._items: list[QFrame] = []
        self._badges: list[QLabel] = []
        self._chips: list[QLabel] = []
        for i, name in enumerate(self._NAMES):
            item = QFrame()
            item.setObjectName("StepItem")
            inner = QHBoxLayout(item)
            inner.setContentsMargins(10, 9, 18, 9)
            inner.setSpacing(8)
            badge = QLabel(str(i + 1))
            badge.setObjectName("StepNumber")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedSize(18, 18)
            title = QLabel(name)
            title.setObjectName("StepTitle")
            inner.addWidget(badge)
            inner.addWidget(title)
            self._items.append(item)
            self._badges.append(badge)
            self._chips.append(title)
            row.addWidget(item)
        row.addStretch(1)
        self._current = 0
        self._tracking: tuple[int, int] | None = None
        self.set_progress(model_ok=False, videos=0, arenas=0)

    def set_tracking(self, tracking: tuple[int, int] | None) -> None:
        """Show "Tracking k of N" on the last step while a batch runs; None to clear."""
        self._tracking = tracking
        self.set_progress(**self._last)

    def set_progress(self, *, model_ok: bool, videos: int, arenas: int) -> None:
        self._last = {"model_ok": model_ok, "videos": videos, "arenas": arenas}
        arenas_ok = videos > 0 and arenas == videos
        done = [model_ok, videos > 0, arenas_ok, False]
        # The badge carries the tick, so the titles only add counts.
        texts = [
            "Model",
            f"Videos · {videos}" if videos else "Videos",
            f"Arenas · {arenas} of {videos}" if videos else "Arenas",
            f"Tracking {self._tracking[0]} of {self._tracking[1]}" if self._tracking else "Track",
        ]
        self._current = next((i for i, d in enumerate(done) if not d), len(done) - 1)
        if self._tracking:
            self._current = len(done) - 1
        for i, text in enumerate(texts):
            self._chips[i].setText(text)
            self._badges[i].setText("✓" if done[i] else str(i + 1))
            badge_state = "done" if done[i] else ("current" if i == self._current else "")
            for widget, state in (
                (self._badges[i], badge_state),
                (self._chips[i], "current" if i == self._current else ""),
                (self._items[i], "current" if i == self._current else ""),
            ):
                widget.setProperty("state", state)
                widget.style().unpolish(widget)
                widget.style().polish(widget)

    def done(self) -> list[bool]:
        """Which steps show a tick."""
        return [badge.text() == "✓" for badge in self._badges]

    def steps_text(self) -> list[str]:
        return [chip.text() for chip in self._chips]

    def current(self) -> int:
        """Index of the first step still to do (Track once the rest are done)."""
        return self._current


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
        self._run_videos: list[Path] = []
        self._calibrations = CalibrationSet()
        # Side of the centre zone written for any video with a drawn arena.
        # Follows whatever the operator last chose in the arena dialog.
        self._zone_cm = 10.0
        self._loaded_master: Path | None = None
        # The last value we defaulted into the master field. While the field
        # still holds it, the path is ours to keep in step with the videos;
        # once it differs, the operator owns it and we never touch it again.
        self._auto_master_text: str | None = None
        self._thread: QThread | None = None
        self._worker = None
        self._meta_thread: QThread | None = None
        self._meta_worker: _MetaWorker | None = None
        # The post-hoc re-gate runs on its own thread rather than borrowing the
        # batch one: it is I/O over a handful of CSVs with no cancel, and the
        # two would otherwise share a teardown that means different things.
        self._regate_thread: QThread | None = None
        self._regate_worker = None
        # The export is derived and read-only over the source CSVs, so it
        # shares no half-written-primary hazard with Run or the re-gate --
        # but it still gets its own thread rather than blocking the GUI, on
        # the same reasoning as the re-gate: I/O over a handful of CSVs.
        self._export_thread: QThread | None = None
        self._export_worker = None
        # The open experiment's type, pushed in by whoever owns the experiment
        # (MainWindow). "" when the window runs standalone.
        self._experiment_type = ""

        self._build_ui()
        self._refresh_videos()

    def set_experiment_type(self, experiment_type: str) -> None:
        """Follow the open experiment's type (what it unlocks is up to the preset)."""
        self._experiment_type = experiment_type or ""
        self._pup_card.setVisible(has_analysis(self._experiment_type, ANALYSIS_PUP_RETRIEVAL))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("ToolPage")
        page = QVBoxLayout(central)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(0)

        # Re-gate and Export live in the header's More menu. Neither is part of a
        # batch: both read CSVs that are already on disk, and the cohort the
        # re-gate exists for is already tracked, so it must stay reachable
        # without a model or a run.
        self._regate_action = QAction("Re-gate tracked CSVs…", self)
        self._regate_action.setEnabled(False)
        self._regate_action.triggered.connect(lambda: self._start_regate())

        # Opt-in: the four-row file is derived fresh from the per-animal CSVs
        # every time this runs, so it can never drift from them the way a
        # parallel primary could.
        self._export_action = QAction("Export multi-animal DLC CSV…", self)
        self._export_action.setEnabled(False)
        self._export_action.triggered.connect(lambda: self._start_export())

        self._remove_all_action = QAction("Remove all videos", self)
        self._remove_all_action.setEnabled(False)
        self._remove_all_action.triggered.connect(lambda: self._remove_all())

        self._header = ToolHeader(
            "Batch Pose Tracking",
            "Run a YOLO-pose model over folders of video and write a DLC CSV beside each one",
        )
        page.addWidget(self._header)

        self._tour = None
        self._tour_btn = QPushButton("Tutorial")
        self._tour_btn.setToolTip("Walk through setting up and running a batch.")
        set_button_role(self._tour_btn, "ghost")
        self._tour_btn.clicked.connect(self.start_tour)
        self._header.add_action(self._tour_btn)

        # Non-default settings, shown outside the popup so they are not forgotten.
        self._settings_summary = QLabel("")
        set_text_role(self._settings_summary, "muted")
        self._header.add_action(self._settings_summary)

        self._settings_button = QPushButton("⚙ Settings")
        self._settings_button.setToolTip(
            "Device, confidence, smoothing, subfolders, overwrite and the arenas file."
        )
        set_button_role(self._settings_button, "ghost")
        self._settings_button.clicked.connect(self._show_settings)
        self._header.add_action(self._settings_button)

        self._more_button = QPushButton("More ▾")
        # The literal arrow sits beside the label; hide Qt's detached indicator.
        self._more_button.setStyleSheet("QPushButton::menu-indicator { image: none; width: 0; }")
        set_button_role(self._more_button, "ghost")
        more_menu = QMenu(self._more_button)
        more_menu.addAction(self._regate_action)
        more_menu.addAction(self._export_action)
        more_menu.addAction(self._remove_all_action)
        self._more_button.setMenu(more_menu)
        self._header.add_action(self._more_button)

        # Full width under the header, like Behavior Analysis's step tabs.
        self._steps = _StepStrip()
        page.addWidget(self._steps)

        # --- configuration, full width ------------------------------------
        area, column = scroll_column()
        column.addWidget(self._build_model_group())
        column.addWidget(self._build_pup_group())
        column.addWidget(self._build_videos_group(), 1)
        column.addStretch(0)
        body = QHBoxLayout()
        body.setContentsMargins(GUTTER, GUTTER, GUTTER, GUTTER)
        body.addWidget(area, 1)
        page.addLayout(body, 1)

        page.addWidget(self._build_footer())

        self._build_settings_popup()
        self.setCentralWidget(central)
        # Opened with parent=None, so nothing hands it the app theme.
        apply_tool_theme(self)
        apply_tool_theme(self._settings_popup)
        for signal in (
            self._filter_group.toggled,
            self._require_gpu.toggled,
            self._device_combo.currentTextChanged,
            self._conf_spin.valueChanged,
        ):
            signal.connect(self._update_settings_summary)

    def _build_footer(self) -> QFrame:
        """Next-step message + Track; while a run is in flight, progress + Cancel.

        The progress panel stays up after a run ends so the outcome (Done, n
        failed) is still readable; only the Track row swaps out while busy.
        """
        footer = QFrame()
        footer.setObjectName("RunFooter")
        footer.setStyleSheet(f"QFrame#RunFooter {{ border-top: 1px solid {colors.BORDER}; }}")
        column = QVBoxLayout(footer)
        column.setContentsMargins(GUTTER, 10, GUTTER, 10)
        column.setSpacing(8)

        # Hidden until a batch starts: two empty bars sitting at zero read as
        # a run that has stalled rather than one that has not begun.
        self._progress_panel = QWidget()
        panel = QVBoxLayout(self._progress_panel)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(6)
        top = QHBoxLayout()
        self._status = StatusPill("Idle")
        top.addWidget(self._status)
        self._video_label = QLabel("")
        set_text_role(self._video_label, "muted")
        top.addWidget(self._video_label, 1)
        self._cancel_button = QPushButton("Cancel")
        set_button_role(self._cancel_button, "danger")
        self._cancel_button.setEnabled(False)
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self._cancel)
        top.addWidget(self._cancel_button)
        panel.addLayout(top)
        self._video_bar = QProgressBar()
        self._video_bar.setFormat("%v / %m frames")
        panel.addWidget(self._video_bar)
        self._overall_bar = QProgressBar()
        self._overall_bar.setFormat("%v / %m videos")
        panel.addWidget(self._overall_bar)
        self._progress_panel.setVisible(False)
        column.addWidget(self._progress_panel)

        row = QHBoxLayout()
        self._blocker_label = hint("")
        self._blocker_label.setVisible(False)
        row.addWidget(self._blocker_label, 1)
        row.addStretch(0)
        self._run_button = QPushButton("Track 0 Videos")
        set_button_role(self._run_button, "primary")
        self._run_button.setMinimumHeight(38)
        self._run_button.clicked.connect(self._start)
        row.addWidget(self._run_button)
        column.addLayout(row)

        self._log = QPlainTextEdit()
        self._log.setObjectName("LogPane")
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Progress will appear here.")
        self._log.setMaximumHeight(110)
        self._log.setVisible(False)
        self._log_toggle = QPushButton()
        set_button_role(self._log_toggle, "ghost")
        self._log_toggle.setCheckable(True)
        self._log_toggle.toggled.connect(self._log.setVisible)
        self._log_toggle.toggled.connect(lambda _: self._sync_log_toggle())
        self._log.textChanged.connect(self._sync_log_toggle)
        self._sync_log_toggle()
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self._log_toggle)
        toggle_row.addStretch(1)
        column.addLayout(toggle_row)
        column.addWidget(self._log)
        return footer

    def _update_settings_summary(self, *_args) -> None:
        active = [
            text
            for on, text in (
                (self._filter_group.isChecked(), "Smoothing on"),
                (self._overwrite.isChecked(), "Overwrite on"),
                (self._require_gpu.isChecked(), "GPU required"),
                (not self._recursive.isChecked(), "Subfolders off"),
                (self._device_combo.currentText() != "auto", self._device_combo.currentText()),
                (abs(self._conf_spin.value() - 0.25) > 1e-9, f"Conf {self._conf_spin.value():.2f}"),
            )
            if on
        ]
        self._settings_summary.setText(" · ".join(active))

    def _sync_log_toggle(self) -> None:
        lines = self._log.document().blockCount() if self._log.toPlainText() else 0
        arrow = "▾" if self._log_toggle.isChecked() else "▸"
        self._log_toggle.setText(f"{arrow} Log ({lines} line{'s' if lines != 1 else ''})")

    def _add_log(self, text: str) -> None:
        """Append a line; an error opens the folded log so it is not missed."""
        self._log.appendPlainText(text)
        lowered = text.lower()
        if "failed" in lowered or "could not" in lowered or "error" in lowered:
            self._log_toggle.setChecked(True)

    def _sync_footer(self) -> None:
        """Swap Track for progress while a pass owns the CSVs; gate removals."""
        busy = self._busy()
        self._run_button.setVisible(not busy)
        self._blocker_label.setVisible(not busy and bool(self._blocker_label.text()))
        if busy:
            self._progress_panel.setVisible(True)
        running = self._thread is not None
        self._cancel_button.setVisible(running)
        self._video_bar.setVisible(running)
        self._video_label.setVisible(running)
        if not running:
            self._steps.set_tracking(None)
        # Removing videos mid-run would pull the rug from the worker's list.
        self._remove_button.setEnabled(not busy)
        self._remove_all_action.setEnabled(bool(self._videos) and not busy)

    def _build_model_group(self) -> Card:
        """The model as two short rows: the weights (and Animals), the bodyparts."""
        card = Card("Model")
        self._model_card = card

        # Shows the file name only; the full path is in the tooltip.
        self._model_field = QLineEdit()
        self._model_field.setReadOnly(True)
        self._model_field.setPlaceholderText("Select a trained YOLO-pose .pt")
        self._model_button = QPushButton("Choose model…")
        self._model_button.clicked.connect(self._choose_model)

        self._animals_spin = QSpinBox()
        self._animals_spin.setRange(1, 16)
        self._animals_spin.setValue(1)
        self._animals_spin.setToolTip(
            "How many animals are in each video. This is not detected -- it is "
            "the constraint that lets tracking stitch a fragmented track back "
            "into one animal. Too high and one animal is split into two that "
            "each vanish for half the video; too low and a real animal is "
            "dropped entirely."
        )
        card.add_row(
            "Weights",
            self._model_field,
            self._model_button,
            set_text_role(QLabel("Animals"), "caption"),
            self._animals_spin,
        )

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
        return card

    def _build_settings_popup(self) -> None:
        """The rarely-touched settings, in a window of their own.

        Built once and kept, so every widget in it stays a member of this
        window: values set while it is closed (the master file defaulting beside
        the videos) still land, and Run reads them the same way it always did.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Pose Settings")
        dialog.setModal(False)
        dialog.setMinimumWidth(520)
        self._settings_popup = dialog
        column = QVBoxLayout(dialog)
        column.setContentsMargins(GUTTER, GUTTER, GUTTER, GUTTER)
        column.setSpacing(10)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        self._device_combo.setMaximumWidth(150)
        column.addLayout(labelled_row("Device", self._device_combo))

        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)  # matches infer_video's default
        column.addLayout(labelled_row("Confidence", self._conf_spin))

        self._require_gpu = QCheckBox("Fail if no GPU is available")
        self._require_gpu.setToolTip(
            "Prevents a silent CPU fallback turning an overnight batch into a multi-day one."
        )
        column.addWidget(self._require_gpu)

        column.addWidget(self._build_filter_group())

        self._recursive = QCheckBox("Include subfolders")
        self._recursive.setChecked(True)
        self._recursive.toggled.connect(self._refresh_videos)
        column.addWidget(self._recursive)

        self._overwrite = QCheckBox("Overwrite existing outputs")
        self._overwrite.setToolTip(
            "Off by default, so an interrupted batch resumes without redoing finished videos."
        )
        self._overwrite.toggled.connect(self._validate)
        column.addWidget(self._overwrite)

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
        column.addLayout(
            labelled_row("Arenas file", self._master_field, master_browse, load_master, save_master)
        )

    def _show_settings(self) -> None:
        popup = self._settings_popup
        popup.show()
        popup.raise_()
        popup.activateWindow()

    def _build_pup_group(self) -> Card:
        card = Card("Pup segmentation")
        self._pup_card = card
        # Shown only when the experiment type unlocks pup retrieval
        # (set_experiment_type); every other experiment never sees it.
        card.setVisible(False)

        self._pup_model_path: Path | None = None
        self._pup_model_field = QLineEdit()
        self._pup_model_field.setReadOnly(True)
        self._pup_model_field.setPlaceholderText("Optional: a semantic pup model (.pt)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_pup_model)
        self._pup_clear = QPushButton("Clear")
        set_button_role(self._pup_clear, "ghost")
        self._pup_clear.clicked.connect(lambda: self._set_pup_model(None))
        card.add_row("Weights", self._pup_model_field, browse, self._pup_clear)

        self._litter_spin = QSpinBox()
        self._litter_spin.setRange(1, 20)
        self._litter_spin.setValue(5)
        card.add_row("Litter size", self._litter_spin)
        card.add(
            hint(
                "Writes <video>_pups.npz beside each video, for scoring pup retrieval in Session Review."
            )
        )
        self._set_pup_model(None)
        return card

    def _build_videos_group(self) -> Card:
        """One table for the batch: a row per video, with its arena and scale."""
        card = Card("Videos", "an arena is required for every video")
        # The walkthrough's "videos" and "calibration" steps are the same card now.
        self._sources_card = self._calibration_card = card
        self._sources = _Sources(self._refresh_videos)
        self._excluded: set[Path] = set()

        self._cal_table = CalibrationTable()
        self._cal_table.set_calibration_set(self._calibrations)
        self._cal_table.arena_requested.connect(self._open_arena)
        self._cal_table.paths_dropped.connect(self._add_paths)
        self._cal_table.itemSelectionChanged.connect(self._update_selection_label)
        # ~4-5 rows: enough to see arena status at a glance.
        self._cal_table.setMinimumHeight(140)
        attach_empty_state(self._cal_table, "Drop video folders here, or Add Folder…")

        add_dir = QPushButton("Add Folder…")
        add_dir.clicked.connect(self._choose_directory)
        add_files = QPushButton("Add Files…")
        add_files.clicked.connect(self._choose_files)

        # The arena is the calibration: it is what Run requires, and it supplies
        # the scale as a by-product (#212 retired the separate line tool here).
        arena = QPushButton("Draw Arena…")
        set_button_role(arena, "primary")
        arena.setToolTip(
            "Required. Outline the arena floor of the selected video, square or "
            "round (or double-click its row).\nGives a centre zone that means the "
            "same patch of floor in every video, and a scale that accounts for "
            "perspective."
        )
        arena.clicked.connect(self._arena_selected)
        copy_arena_btn = QPushButton("Copy Arena to Selected")
        copy_arena_btn.setToolTip(
            "Stamp one video's floor perimeter onto the other selected videos.\n"
            "The copies land unconfirmed: an arena that does not fit gives no "
            "warning of its own, so open each one and check the outline before "
            "the batch will run."
        )
        copy_arena_btn.clicked.connect(self._copy_arena_to_selected)
        remove = self._remove_button = QPushButton("Remove")
        set_button_role(remove, "ghost")
        remove.clicked.connect(self._remove_selected)

        self._selection_label = QLabel("")
        set_text_role(self._selection_label, "hint")

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(add_dir)
        toolbar.addWidget(add_files)
        toolbar.addStretch(1)
        toolbar.addWidget(self._selection_label)
        for button in (arena, copy_arena_btn, remove):
            toolbar.addWidget(button)
        card.add(toolbar)
        card.add(self._cal_table, 1)

        # Clear Arena is not worth a button of its own: it is the exception.
        self._cal_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._cal_table.customContextMenuRequested.connect(self._table_menu)

        # Kept as the count of record; the card badge is where it is read now.
        self._count_label = QLabel("No videos found")
        self._count_label.setVisible(False)
        card.add(self._count_label)
        return card

    def _build_filter_group(self) -> QGroupBox:
        # Still a checkable QGroupBox: it is the enable switch the rest of the
        # section hangs off, and _filter_settings reads isChecked().
        group = QGroupBox("Smooth and gap-fill the tracked keypoints")
        group.setCheckable(True)
        group.setChecked(False)
        group.setToolTip(
            "Writes the smoothed result as the main CSV and keeps the raw "
            "inference alongside it as *_raw.csv."
        )
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

        self._filter_group = group
        return group

    # ------------------------------------------------------------------
    # walkthrough
    # ------------------------------------------------------------------

    def tour_targets(self) -> dict[str, QWidget | None]:
        """Widgets the Batch Pose walkthrough spotlights, by step key.

        Mostly whole cards rather than single fields: the lesson is which
        section of the form to fill in next, and outlining one line edit inside
        a card says less than outlining the card.
        """
        return {
            "model": self._model_card,
            "bodyparts": self._names_field,
            "videos": self._sources_card,
            "calibration": self._calibration_card,
            "filter": self._settings_button,
            "run": self._run_button,
        }

    def start_tour(self) -> None:
        """Run the Batch Pose walkthrough (also the Tutorial button)."""
        from glider.gui.onboarding.tour import (
            POSE_BATCH_TOUR_COMPLETE_KEY,
            Tour,
            pose_batch_steps,
        )

        # Held on the instance: the overlay is parented to this window, and a
        # Tour that goes out of scope takes its overlay with it mid-step.
        self._tour = Tour(self, steps=pose_batch_steps(), complete_key=POSE_BATCH_TOUR_COMPLETE_KEY)
        self._tour.start()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Offer the walkthrough the first time this window is opened."""
        super().showEvent(event)
        from glider.gui.onboarding.tour import (
            POSE_BATCH_TOUR_COMPLETE_KEY,
            offer_tour_once,
            pose_batch_steps,
        )

        offer_tour_once(self, pose_batch_steps(), POSE_BATCH_TOUR_COMPLETE_KEY)

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
        self._model_field.setText(self._model_path.name)
        self._model_field.setToolTip(str(self._model_path))
        self._model_button.setText("Change model…")
        self._names_status.setText("Reading keypoints from the model…")
        self._start_meta_read()
        self._validate()

    def _choose_pup_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a pup segmentation model", "", "PyTorch weights (*.pt)"
        )
        if path:
            self._set_pup_model(Path(path))

    def _set_pup_model(self, path: Path | None) -> None:
        self._pup_model_path = path
        self._pup_model_field.setText("" if path is None else path.name)
        # The litter size only means something to a pass that will run.
        self._litter_spin.setEnabled(path is not None)

    def _choose_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Add video directory")
        if path:
            self._add_paths([Path(path)])

    def _choose_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(batch_core.VIDEO_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", "", f"Videos ({exts})")
        if paths:
            self._add_paths([Path(p) for p in paths])

    def _discover(self, sources) -> list[Path]:
        # Same call Run uses, so the count can never disagree with the work
        # actually done. On a very large tree this touches disk synchronously.
        try:
            return batch_core.discover_videos(sources, recursive=self._recursive.isChecked())
        except OSError:
            logger.warning("could not scan for videos", exc_info=True)
            return []

    def _add_paths(self, paths) -> None:
        """Add folders/files; anything in them the user removed earlier comes back."""
        paths = list(paths)
        self._excluded -= set(self._discover(paths))
        before = self._sources.paths()
        self._sources.add_paths(paths)  # refreshes itself when something is new
        if self._sources.paths() == before:
            self._refresh_videos()  # only exclusions changed

    def _refresh_videos(self) -> None:
        self._videos = [v for v in self._discover(self._sources.paths()) if v not in self._excluded]
        count = len(self._videos)
        self._count_label.setText(
            "No videos found" if count == 0 else f"{count} video{'s' if count != 1 else ''} found"
        )
        self._cal_table.set_videos(self._videos)
        self._update_selection_label()
        self._sync_master_field()
        self._validate()

    def _update_selection_label(self) -> None:
        n = len(self._cal_table.selected_videos())
        self._selection_label.setText(f"{n} selected" if n else "")

    def _remove_selected(self) -> None:
        """Drop the selected videos, and their arenas with them."""
        if self._busy():
            return
        selected = self._cal_table.selected_videos()
        for video in selected:
            self._excluded.add(video)
            self._calibrations.discard(video)
            self._calibrations.discard_arena(video)
        if selected:
            self._refresh_videos()

    def _remove_all(self) -> None:
        """Empty the table, and drop every arena and scale along with it."""
        if self._busy():
            return
        if any(self._calibrations.get_arena(v) is not None for v in self._videos):
            answer = QMessageBox.question(
                self,
                "Remove all videos",
                "Remove every video and the arenas drawn for them? This cannot be undone.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        for video in {*self._videos, *self._calibrations.videos()}:
            self._calibrations.discard(video)
            self._calibrations.discard_arena(video)
        self._excluded.clear()
        self._sources.clear_all()

    def _table_menu(self, pos) -> None:
        menu = QMenu(self._cal_table)
        menu.addAction("Draw Arena…", self._arena_selected)
        menu.addAction("Clear Arena", self._clear_selected_calibrations).setEnabled(
            not self._busy()
        )
        menu.exec(self._cal_table.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def _open_arena(self, video: Path) -> None:
        """Click the four floor corners of *video*'s arena on a scrubbed frame."""
        from glider.gui.dialogs.arena_dialog import ArenaDialog
        from glider.vision.video_source import VideoFileSource

        reader = VideoFileSource()
        frame = None
        try:
            if reader.load(video):
                # A quarter in: rigs are often still being adjusted at the
                # start, and a hand in shot hides the corner to be clicked.
                frame = reader.read_frame(int(reader.frame_count * 0.25))
        finally:
            reader.release()

        if frame is None:
            QMessageBox.warning(
                self,
                "Cannot Open Video",
                f"{video.name} could not be opened to draw its arena.",
            )
            return

        dialog = None
        try:
            dialog = ArenaDialog(frame, title=video.name, parent=self)
            existing = self._calibrations.get_arena(video)
            if existing is not None:
                dialog.set_arena(existing)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                arena = dialog.calibration()
                if arena is not None:
                    # Explicit, because this is the only thing that confirms a
                    # copied arena: an operator has now seen the overlay sit on
                    # this video's own floor, which residuals() cannot tell them.
                    self._calibrations.set_arena(video, arena, confirmed=True)
                    self._zone_cm = dialog.zone_size_cm()
                    self._add_log(
                        f"{video.name}: arena drawn, "
                        f"{arena.px_per_cm_centre / 10:.3f} px/mm at the floor centre"
                    )
        finally:
            if dialog is not None:
                # parent=self hands ownership to the window, so dropping the
                # Python reference leaves the C++ dialog — and the full-res
                # frame it holds — alive for the session.
                dialog.deleteLater()

        self._cal_table.refresh()
        self._validate()

    def _arena_selected(self) -> None:
        selected = self._cal_table.selected_videos()
        if not selected:
            QMessageBox.information(self, "Arena", "Select a video in the calibration table first.")
            return
        self._open_arena(selected[0])

    def _arena_map(self) -> dict[Path, ArenaCalibration]:
        """Arenas for the batch, keyed the way run_batch resolves videos.

        Only the listed videos: the set can also hold arenas from folders
        visited earlier this session, and handing those over would key the gate
        off files this run knows nothing about. Run is gated on every listed
        video having a confirmed arena, so in practice this covers all of them.
        """
        arenas: dict[Path, ArenaCalibration] = {}
        for video in self._videos:
            arena = self._calibrations.get_arena(video)
            if arena is not None:
                arenas[Path(video).resolve()] = arena
        return arenas

    def _zone_configs(self) -> dict[Path, ZoneConfiguration]:
        """Centre zones for the batch, keyed the way run_batch resolves videos.

        Only videos with a drawn arena appear; the rest simply get no zone
        CSVs. A perimeter that does not yet describe a quadrilateral is skipped
        with a note rather than failing the run — the pose inference is the
        expensive part and must not be held hostage to a half-drawn zone.
        """
        from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

        configs: dict[Path, ZoneConfiguration] = {}
        for video in self._videos:
            arena = self._calibrations.get_arena(video)
            if arena is None:
                continue
            try:
                vertices = arena.centre_zone_vertices(self._zone_cm)
            except (DegenerateArenaError, ValueError) as e:
                self._add_log(f"{video.name}: no zone written ({e})")
                continue
            config = ZoneConfiguration()
            config.config_width, config.config_height = arena.frame_size
            config.add_zone(
                Zone(
                    id=str(uuid.uuid4()),
                    name=f"Centre {self._zone_cm:g}cm",
                    shape=ZoneShape.POLYGON,
                    vertices=vertices,
                    color=(0, 165, 255),
                )
            )
            configs[Path(video).resolve()] = config
        return configs

    def _copy_arena_to_selected(self) -> None:
        """Stamp the first selected video's arena onto the rest of the selection.

        The copies land unconfirmed, so Run stays blocked until each has been
        opened and checked — see :func:`arena_actions.copy_arena_to` for why.

        Asks before replacing an arena that is already there. Selecting a whole cohort
        and clicking Copy is one action with no undo, and what it would
        otherwise destroy is exactly the per-video human judgement this window
        now makes mandatory.
        """
        from glider.gui.pose_batch import arena_actions

        selected = self._cal_table.selected_videos()
        sources = [v for v in selected if self._calibrations.get_arena(v) is not None]
        if not sources:
            QMessageBox.information(
                self,
                "Copy Arena",
                "Select one video with a drawn arena plus the videos to copy it to.",
            )
            return

        source = sources[0]
        targets = [v for v in selected if v != source]
        already = {v for v in targets if self._calibrations.get_arena(v) is not None}

        if already and not self._confirm_overwrite(source, len(already), noun="arena"):
            # Declined: filling the blanks is what "copy to selected" is for,
            # and it is the half of the job that cannot destroy anything.
            targets = [v for v in targets if v not in already]

        skipped = arena_actions.copy_arena_to(self._calibrations, source, targets)

        copied = [v for v in targets if v not in skipped]
        overwritten = sum(1 for v in copied if v in already)
        self._add_log(
            f"Copied {source.name}'s arena: {len(copied) - overwritten} video(s) with no "
            f"arena filled, {overwritten} existing arena(s) overwritten. Each copy is "
            "marked unconfirmed: open it and check the outline sits on that video's floor."
        )
        if skipped:
            names = ", ".join(v.name for v in skipped)
            self._add_log(
                f"Not copied to {len(skipped)} video(s) that could not be opened to "
                f"read their resolution: {names}"
            )
        self._cal_table.refresh()
        self._validate()

    def _confirm_overwrite(self, source: Path, count: int, *, noun: str = "calibration") -> bool:
        """Ask before replacing work the operator drew themselves.

        Shared by the line and arena copies rather than duplicated: the two
        already drifted apart once, and the half without the prompt is the one
        that destroys data.
        """
        answer = QMessageBox.question(
            self,
            f"Overwrite {noun.capitalize()}s?",
            f"{count} of the selected video(s) already have their own {noun}.\n\n"
            f"Overwrite them with {source.name}'s?\n"
            f"Choose No to fill only the videos with no {noun}.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _clear_selected_calibrations(self) -> None:
        for video in self._cal_table.selected_videos():
            self._calibrations.discard(video)
            self._calibrations.discard_arena(video)
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
            self._add_log(f"Could not read {path.name}: {e}")
            QMessageBox.warning(self, "Calibration File", str(e))
            return

        # Not a hand-rolled two-map update: the confirmed state is a third map
        # that has to travel with the arenas, or a copied arena comes back
        # confirmed after one save/reload cycle.
        self._calibrations.merge(loaded)
        self._add_log(
            f"Loaded calibration for {len(loaded.entries)} video(s) "
            f"and {len(loaded.arenas)} arena(s) from {path.name}."
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
        self._add_log(f"Wrote {path}")

    def _write_master(self, path: Path) -> bool:
        """Write the master file. Reports and returns False on failure."""
        try:
            # Only the listed batch: the set can also hold videos from folders
            # visited earlier this session, which this file does not describe.
            self._calibrations.subset(self._videos).save(path, model=self._model_path)
        except (OSError, ValueError) as e:
            self._add_log(f"Could not write {path}: {e}")
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
            # A model load cannot be interrupted: do not block the GUI on it,
            # and do not let a superseded model's names land later.
            try:
                self._meta_worker.done.disconnect(self._apply_meta)
            except TypeError:
                pass  # already disconnected
            retire_thread(self._meta_thread, self._meta_worker, timeout_ms=0)
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

    def _busy(self) -> bool:
        """True while either pass owns the pose CSVs.

        Both write the same primary files: ``run_batch`` writes the primary,
        and the re-gate renames it to ``_ungated`` and rewrites it in place.
        Overlapping them can capture a half-written primary as the permanent
        "pristine original", and the re-gate has no cancel, so the window is
        tens of seconds wide on a real cohort. Asking one question keeps the
        Run gate and the button-disabling in ``_start_regate`` from drifting
        apart the way they already did once.
        """
        return (
            self._thread is not None
            or self._regate_thread is not None
            or self._export_thread is not None
        )

    def _validate(self) -> None:
        """Disable Run — and say why — rather than failing an overnight batch."""
        self._sync_footer()
        self._update_settings_summary()
        if self._busy():
            return  # a run or a re-gate is in flight; Run stays disabled

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
            # An arena, not just a line: only the perimeter can place a zone or
            # gate a detection, and on this cohort the line ran systematically
            # 2.34% high. A usable arena carries the scale by construction, so
            # this replaces the old line check rather than adding to it.
            no_arena = self._calibrations.missing_arenas(self._videos)
            if no_arena:
                problem = f"{len(no_arena)} video(s) still need an arena drawn."

        self._names_field.setStyleSheet(_INVALID_STYLE if names_bad else "")
        self._steps.set_progress(
            model_ok=self._model_path is not None and bool(names) and not names_bad,
            videos=len(self._videos),
            arenas=len(self._videos) - len(self._calibrations.missing_arenas(self._videos)),
        )
        self._run_button.setEnabled(problem is None)
        self._run_button.setToolTip(problem or "")
        # Shown next to the button as well as on hover: a disabled Run with the
        # reason only in a tooltip is a dead end you have to go hunting in.
        self._blocker_label.setText(problem or "")
        self._blocker_label.setVisible(problem is not None)
        n = len(self._videos)
        self._run_button.setText(f"Track {n} Video{'s' if n != 1 else ''}")

        if self._videos:
            missing = len(self._calibrations.missing_arenas(self._videos))
            done = len(self._videos) - missing
            n = len(self._videos)
            self._sources_card.set_badge(
                f"{n} video{'s' if n != 1 else ''} · {done} / {n} arenas drawn"
            )
        else:
            self._sources_card.set_badge("none yet")

        self._validate_regate()

    def _validate_regate(self) -> None:
        """Re-gate needs arenas and existing tracks — not a model.

        Deliberately not gated on ``problem``: this pass touches CSVs that are
        already on disk, so demanding the weights that produced them would make
        the button unavailable on exactly the archived cohort it is for.
        """
        from glider.gui.pose_batch.arena_actions import regatable

        ready = (
            not self._busy()
            and bool(self._videos)
            and not self._calibrations.missing_arenas(self._videos)
            and bool(regatable(self._videos))
        )
        self._regate_action.setEnabled(ready)
        self._regate_action.setToolTip(
            "Apply the drawn arenas to the pose CSVs already beside these "
            "videos. Originals are kept as _ungated.csv."
            if ready
            else "Needs a confirmed arena on every video and a pose CSV to re-gate."
        )
        self._validate_export()

    def _validate_export(self) -> None:
        """Export needs multi-animal sessions -- not a model, not an arena.

        Deliberately not gated on ``problem`` for the same reason as the
        re-gate: this reads per-animal CSVs already on disk, so demanding the
        weights that produced them would make the button unavailable on
        exactly the archived cohort it is for.
        """
        from glider.gui.pose_batch.export_actions import exportable

        ready = not self._busy() and bool(exportable(self._videos))
        self._export_action.setEnabled(ready)
        self._export_action.setToolTip(
            "Rebuild the four-row DLC CSV that SimBA, DeepLabCut and "
            "Keypoint-MoSeq read as a multi-animal session, from the "
            "per-animal CSVs already beside these videos."
            if ready
            else "Needs at least one video with more than one animal tracked."
        )

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
        # Not just a run already in flight: a re-gate is rewriting the very
        # CSVs this would write. Checked before the master file is written, so
        # a click that cannot start also cannot touch the disk.
        if self._busy():
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
        from glider.vision.arena_gate import ArenaGateSettings

        self._run_videos = list(self._videos)  # the worker's copy; _videos stays live
        self._log.clear()
        self._log_toggle.setChecked(False)
        self._add_log(f"Starting {len(self._videos)} video(s)…")
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
            zones=self._zone_configs(),
            arenas=self._arena_map(),
            n_animals=self._animals_spin.value(),
            # Defaults, and no UI to tune them: gating is not optional here.
            # Zones below are scored from the gated pose — centre time computed
            # from bench-floor detections is meaningless — and the arena also
            # re-ranks multi-detection frames during inference, which is the
            # one part no post-hoc pass can recover.
            gate=ArenaGateSettings(),
            # A model picked under pup retrieval stays in the hidden card when the
            # type changes; it must not run a pass the user can no longer see.
            pup_model=(
                self._pup_model_path
                if has_analysis(self._experiment_type, ANALYSIS_PUP_RETRIEVAL)
                else None
            ),
            litter_size=self._litter_spin.value(),
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.video_progress.connect(self._on_video_progress)
        self._worker.log.connect(self._add_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self._run_button.setEnabled(False)
        self._regate_action.setEnabled(False)  # _validate returns early mid-run
        self._export_action.setEnabled(False)
        self._cancel_button.setEnabled(True)
        self._status.set_state("running", "Tracking")
        self._sync_footer()
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._cancel_button.setEnabled(False)
            self._status.set_state("warn", "Cancelling")
            self._add_log("Cancelling after the current frame…")

    def _on_progress(self, index: int, total: int) -> None:
        if self._thread is not None:  # the batch, not a re-gate or export
            self._steps.set_tracking((index + 1, total))
            if 0 <= index < len(self._run_videos):
                self._video_label.setText(self._run_videos[index].name)
        self._overall_bar.setRange(0, total)
        # The batch counts the video in flight ("2 of 5", as the step strip does).
        self._overall_bar.setValue(index + 1 if self._thread is not None else index)

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
            self._status.set_state("warn", f"{len(result.failed)} failed")
        elif result.completed:
            self._status.set_state("ok", "Done")
        else:
            self._status.set_state("idle", "Cancelled")
        for video, message in result.failed:
            self._add_log(f"FAILED {video.name}: {message}")
        self._teardown_thread()

    def _on_failed(self, message: str) -> None:
        self._status.set_state("error", "Failed")
        self._add_log(f"Batch could not start: {message}")
        QMessageBox.critical(self, "Batch Pose Tracking", message)
        self._teardown_thread()

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            retire_thread(self._thread, self._worker)
            self._thread = None
        self._worker = None
        self._video_bar.setRange(0, 1)
        self._cancel_button.setEnabled(False)
        self._validate()

    # ------------------------------------------------------------------
    # post-hoc re-gating
    # ------------------------------------------------------------------

    def _start_regate(self) -> None:
        from glider.gui.pose_batch.arena_actions import regatable
        from glider.gui.pose_batch.regate_worker import RegateWorker

        if self._busy():
            return
        videos = regatable(self._videos)
        if not videos:
            return

        # Confirmed because it rewrites files the operator did not just make.
        # Naming what survives is the point of the prompt: without the second
        # sentence the honest answer to "can I undo this?" is not visible
        # anywhere in the UI.
        answer = QMessageBox.question(
            self,
            "Re-gate tracked CSVs",
            f"Rewrite {len(videos)} pose CSV(s) in place?\n\n"
            f"Originals are kept as _ungated.csv.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._add_log(f"Re-gating {len(videos)} tracked CSV(s)…")
        self._overall_bar.setRange(0, len(videos))
        self._overall_bar.setValue(0)

        self._regate_worker = RegateWorker(videos, self._calibrations)
        self._regate_thread = QThread(self)
        self._regate_worker.moveToThread(self._regate_thread)
        self._regate_thread.started.connect(self._regate_worker.run)
        self._regate_worker.progress.connect(self._on_progress)
        self._regate_worker.log.connect(self._add_log)
        self._regate_worker.finished.connect(self._on_regate_finished)
        self._regate_worker.failed.connect(self._on_regate_failed)

        self._run_button.setEnabled(False)
        self._regate_action.setEnabled(False)
        self._export_action.setEnabled(False)
        self._status.set_state("running", "Re-gating")
        self._sync_footer()
        self._regate_thread.start()

    def _on_regate_finished(self, gated: int, skipped: int) -> None:
        self._overall_bar.setValue(self._overall_bar.maximum())
        self._add_log(f"Re-gated {gated}, skipped {skipped}.")
        self._status.set_state("warn" if skipped else "ok", "Done")
        self._teardown_regate()

    def _on_regate_failed(self, message: str) -> None:
        self._status.set_state("error", "Failed")
        self._add_log(f"Re-gate could not run: {message}")
        QMessageBox.critical(self, "Re-gate tracked CSVs", message)
        self._teardown_regate()

    def _teardown_regate(self) -> None:
        if self._regate_thread is not None:
            retire_thread(self._regate_thread, self._regate_worker)
            self._regate_thread = None
        self._regate_worker = None
        self._validate()

    # ------------------------------------------------------------------
    # multi-animal export
    # ------------------------------------------------------------------

    def _start_export(self) -> None:
        from glider.gui.pose_batch.export_actions import exportable
        from glider.gui.pose_batch.export_worker import ExportWorker

        if self._busy():
            return
        videos = exportable(self._videos)
        if not videos:
            return

        # No confirmation prompt: unlike the re-gate, this never touches the
        # per-animal CSVs it reads from -- it only (re)writes the derived
        # four-row file, which is exactly what running it again is for.
        self._add_log(f"Exporting {len(videos)} multi-animal session(s)…")
        self._overall_bar.setRange(0, len(videos))
        self._overall_bar.setValue(0)

        self._export_worker = ExportWorker(videos)
        self._export_thread = QThread(self)
        self._export_worker.moveToThread(self._export_thread)
        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.progress.connect(self._on_progress)
        self._export_worker.log.connect(self._add_log)
        self._export_worker.finished.connect(self._on_export_finished)
        self._export_worker.failed.connect(self._on_export_failed)

        self._run_button.setEnabled(False)
        self._regate_action.setEnabled(False)
        self._export_action.setEnabled(False)
        self._status.set_state("running", "Exporting")
        self._sync_footer()
        self._export_thread.start()

    def _on_export_finished(self, exported: int, skipped: int) -> None:
        self._overall_bar.setValue(self._overall_bar.maximum())
        self._add_log(f"Exported {exported}, skipped {skipped}.")
        self._status.set_state("warn" if skipped else "ok", "Done")
        self._teardown_export()

    def _on_export_failed(self, message: str) -> None:
        self._status.set_state("error", "Failed")
        self._add_log(f"Export could not run: {message}")
        QMessageBox.critical(self, "Export multi-animal DLC CSV", message)
        self._teardown_export()

    def _teardown_export(self) -> None:
        if self._export_thread is not None:
            retire_thread(self._export_thread, self._export_worker)
            self._export_thread = None
        self._export_worker = None
        self._validate()

    def closeEvent(self, event):
        """Never let a running batch outlive its window."""
        # Cancel lands only between frames, and the re-gate/export have no
        # cancel at all, so a thread still running is parked until it ends
        # (retire_thread) rather than joined or destroyed.
        if self._thread is not None and self._worker is not None:
            self._worker.cancel()
            retire_thread(self._thread, self._worker)
            self._thread = None
            self._worker = None
        if self._regate_thread is not None:
            retire_thread(self._regate_thread, self._regate_worker, timeout_ms=0)
            self._regate_thread = None
            self._regate_worker = None
        if self._export_thread is not None:
            retire_thread(self._export_thread, self._export_worker, timeout_ms=0)
            self._export_thread = None
            self._export_worker = None
        self._stop_meta_thread()
        super().closeEvent(event)
