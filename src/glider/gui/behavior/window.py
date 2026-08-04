"""Behavior Analysis window: Annotate / Train / Apply tabs.

Drives the Qt-free behavior cores (:mod:`glider.analysis.behavior`) through
the :class:`~PyQt6.QtCore.QThread` workers in :mod:`glider.gui.behavior.workers`,
following the worker-on-a-thread shape used throughout GLIDER (see
``glider.gui.panels.camera_panel``'s tracking-run thread).

This window is only ever opened when the optional ``[behavior]`` extra is
installed — the menu item that launches it checks
:func:`glider.gui.behavior.availability.behavior_available` first — but the
module itself stays import-light: everything that pulls in sklearn / umap /
hdbscan / cv2 (``train_model``, ``classify``, ``propose_clips_multi``,
``AnnotatorWindow``, ...) is imported lazily inside button handlers, not at
module scope, so ``import glider.gui.behavior.window`` succeeds under a bare
``[pc]`` install.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from glider.gui.widgets.tool_ui import (
    GUTTER,
    Card,
    RunRail,
    ToolHeader,
    apply_tool_theme,
    attach_empty_state,
    hint,
    labelled_row,
    path_label,
    scroll_column,
    set_button_role,
    set_path_text,
    set_text_role,
)
from glider.vision.pose.batch import VIDEO_EXTS, VIDEO_FILTER, find_pose_csv

logger = logging.getLogger(__name__)

# Video extensions and the matching dialog filter come from the pose batch
# module rather than being restated here, so this tool and Batch Pose Tracking
# can never disagree about what counts as a video. That module keeps its heavy
# imports (ultralytics, torch, pandas) inside run_batch, so importing it at
# menu-build time stays cheap.
_VIDEO_EXTS = VIDEO_EXTS
_VIDEO_FILTER = VIDEO_FILTER

# Classifier cadence shown in the Apply tab. Mirrors
# glider.analysis.behavior.classify.pipeline.LiveInferenceConfig.predict_every,
# which this module can't import at module scope (it pulls in cv2/sklearn).
# test_apply_tab_cadence_default_matches_the_pipeline pins the two together.
_DEFAULT_PREDICT_EVERY = 3

# Majority-vote window over the classifier's predictions. The pipeline's own
# default is 1 (off) for backwards compatibility; the Apply tab defaults to 5
# because a per-frame classifier flickers about ten times faster than a mouse
# changes behavior, and at 5 the time budget shifts by <0.5 percentage points
# while the switch rate roughly halves.
_DEFAULT_SMOOTH_WINDOW = 5

# Session lists hold one row per animal, so a cohort is dozens of rows and
# both adding and removing have to work on a selection rather than a row.
_MULTI_SELECT = QAbstractItemView.SelectionMode.ExtendedSelection


class _Knob(NamedTuple):
    """One row of the advanced LightGBM dialog.

    ``decimals == 0`` picks a :class:`QSpinBox` (integer knob), anything
    higher a :class:`QDoubleSpinBox`. ``special`` is the label shown when an
    integer knob sits at its minimum, for the knobs where the minimum is a
    sentinel rather than a quantity (``max_depth = -1`` means "no limit").
    """

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    decimals: int
    step: float
    tooltip: str
    special: str = ""


# The tunable LightGBM surface, in the order it reads best: capacity knobs
# first, then the regularizers. Defaults mirror
# glider.analysis.behavior.pipeline.LgbmReg (plus train_model's n_estimators),
# duplicated because this module stays import-light -- importing pipeline at
# module scope would pull sklearn/pandas into a bare [pc] install.
# test_lgbm_knobs_cover_the_dataclass pins the names and defaults together.
_LGBM_KNOBS: tuple[_Knob, ...] = (
    _Knob(
        "n_estimators",
        "Boosting rounds:",
        200,
        10,
        5000,
        0,
        10,
        "How many trees are boosted in sequence. More rounds fit the training "
        "data harder; pair a higher count with a lower learning rate.",
    ),
    _Knob(
        "learning_rate",
        "Learning rate:",
        0.1,
        0.001,
        1.0,
        3,
        0.01,
        "How much each tree contributes. Lower generalizes better but needs "
        "more boosting rounds to reach the same fit.",
    ),
    _Knob(
        "num_leaves",
        "Leaves per tree:",
        31,
        2,
        1024,
        0,
        1,
        "Maximum leaves in one tree -- LightGBM's main capacity dial. Raising "
        "it lets the model carve finer distinctions and overfit sooner.",
    ),
    _Knob(
        "max_depth",
        "Max tree depth:",
        -1,
        -1,
        64,
        0,
        1,
        "Hard cap on tree depth. -1 leaves depth unlimited (leaves-per-tree is "
        "then the only limit); cap it to blunt deep, session-specific splits.",
        special="No limit",
    ),
    _Knob(
        "min_child_samples",
        "Min samples per leaf:",
        50,
        1,
        1000,
        0,
        5,
        "Fewest training frames a leaf may cover. Raising it stops the model "
        "memorizing rare one-off poses. Raise it if you have few labeled bouts.",
    ),
    _Knob(
        "min_split_gain",
        "Min split gain:",
        0.0,
        0.0,
        10.0,
        3,
        0.01,
        "Minimum improvement a split must buy to be kept. Above 0 this prunes "
        "splits that only fit noise.",
    ),
    _Knob(
        "feature_fraction",
        "Feature fraction:",
        0.8,
        0.1,
        1.0,
        2,
        0.05,
        "Fraction of features each tree may sample. Below 1.0 it decorrelates "
        "the trees, which matters because the windowed features overlap heavily.",
    ),
    _Knob(
        "bagging_fraction",
        "Row fraction:",
        0.8,
        0.1,
        1.0,
        2,
        0.05,
        "Fraction of training rows each tree samples. Below 1.0 adds variance "
        "between trees and reduces overfitting.",
    ),
    _Knob(
        "reg_lambda",
        "L2 regularization:",
        1.0,
        0.0,
        100.0,
        2,
        0.5,
        "L2 penalty on leaf weights. Higher shrinks confident leaves toward "
        "the mean, trading training accuracy for cross-session stability.",
    ),
)


class LgbmAdvancedDialog(QDialog):
    """Per-knob LightGBM hyperparameters, opened from the Train tab.

    Deliberately a modal dialog rather than an inline group: these are
    rarely-touched knobs where a wrong value quietly produces a worse model,
    so they get their own surface with a tooltip per knob and a one-click
    path back to the defaults.
    """

    def __init__(self, values: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced LightGBM settings")
        self._spins: dict[str, QSpinBox | QDoubleSpinBox] = {}
        current = dict(values or {})

        layout = QVBoxLayout(self)
        blurb = QLabel(
            "GLIDER's defaults are mildly regularized relative to stock LightGBM, "
            "which trades training accuracy for generalization to sessions the "
            "model has not seen. Hover a field to see what it trades off."
        )
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        form = QFormLayout()
        for knob in _LGBM_KNOBS:
            spin = self._build_spin(knob)
            spin.setValue(current.get(knob.name, knob.default))
            self._spins[knob.name] = spin
            form.addRow(knob.label, spin)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.RestoreDefaults
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self.restore_defaults
        )
        layout.addWidget(buttons)

    @staticmethod
    def _build_spin(knob: _Knob) -> QSpinBox | QDoubleSpinBox:
        if knob.decimals == 0:
            spin: QSpinBox | QDoubleSpinBox = QSpinBox()
            spin.setRange(int(knob.minimum), int(knob.maximum))
            if knob.special:
                spin.setSpecialValueText(knob.special)
        else:
            spin = QDoubleSpinBox()
            spin.setDecimals(knob.decimals)
            spin.setRange(float(knob.minimum), float(knob.maximum))
        spin.setSingleStep(knob.step)
        spin.setToolTip(knob.tooltip)
        return spin

    def restore_defaults(self) -> None:
        for knob in _LGBM_KNOBS:
            self._spins[knob.name].setValue(knob.default)

    def values(self) -> dict:
        """Every knob's current value, keyed by its ``train_model`` kwarg name."""
        return {knob.name: self._spins[knob.name].value() for knob in _LGBM_KNOBS}


class BehaviorAnalysisWindow(QMainWindow):
    """Top-level window for the Behavior Analysis tool."""

    # What each tab is for, shown in the header so the three stages read as one
    # pipeline rather than three unrelated screens.
    _TAB_BLURBS = (
        "Label behavior on sampled clips to build a training set",
        "Fit a classifier from labeled sessions",
        "Score recorded video and write the ethogram",
    )

    def __init__(self, project_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Analysis")
        self.resize(1240, 840)

        if project_dir is None:
            from glider.core.config import get_config

            project_dir = get_config().paths.behavior_projects_dir
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(AnnotateTab(self.project_dir), "Annotate")
        self.tabs.addTab(TrainTab(), "Train")
        self.tabs.addTab(ApplyTab(), "Apply")

        self._header = ToolHeader("Behavior Analysis", self._TAB_BLURBS[0])
        self.tabs.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        central.setObjectName("ToolPage")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._header)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # These windows open with parent=None, so nothing hands them the app
        # theme -- see tool_ui.apply_tool_theme.
        apply_tool_theme(self)

    def _on_tab_changed(self, index: int) -> None:
        if 0 <= index < len(self._TAB_BLURBS):
            self._header.set_subtitle(self._TAB_BLURBS[index])


def _workspace(parent: QWidget, rail: RunRail, *, rail_width: int = 380) -> QVBoxLayout:
    """Lay a tab out as ``[ scrolling config column | pinned run rail ]``.

    Returns the layout to add cards to. The rail keeps the primary action and
    its output on screen no matter how far the configuration scrolls, which is
    the whole point of the two-column shape.
    """
    outer = QHBoxLayout(parent)
    outer.setContentsMargins(GUTTER, GUTTER, GUTTER, GUTTER)
    outer.setSpacing(GUTTER)

    area, column = scroll_column()
    outer.addWidget(area, 1)

    rail.setFixedWidth(rail_width)
    outer.addWidget(rail)
    return column


class AnnotateTab(QWidget):
    """Pick a videos folder + pose-CSV folder and launch the clip annotator."""

    # The clip queue the tab asked for before it was configurable. Kept as the
    # default so an existing habit produces an unchanged session.
    DEFAULT_CLIP_COUNT = 50

    def __init__(self, project_dir: Path, parent=None):
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self._videos_dir: Path | None = None
        self._poses_dir: Path | None = None
        # Keep a reference so the launched AnnotatorWindow survives GC.
        self._annotator_window = None

        rail = RunRail("Launch annotator")
        self._launch_btn = rail.button
        self._launch_btn.clicked.connect(self._on_launch)
        rail.status.setVisible(False)  # nothing runs here; the annotator opens
        rail.card.add(
            hint(
                "Opens the clip annotator in its own window. Labels are saved "
                "beside each pose CSV as <name>_annotations.csv — the same "
                "files the Train tab reads."
            )
        )
        column = _workspace(self, rail, rail_width=340)

        # --- sources ----------------------------------------------------
        sources = Card("Sources", "where the clips come from")
        self._videos_label = path_label("(none)")
        videos_btn = QPushButton("Choose…")
        videos_btn.clicked.connect(self._on_choose_videos)
        sources.add_row("Videos folder", self._videos_label, videos_btn)

        self._poses_label = path_label("(defaults to videos folder)")
        poses_btn = QPushButton("Choose…")
        poses_btn.clicked.connect(self._on_choose_poses)
        sources.add_row("Pose CSV folder", self._poses_label, poses_btn)
        sources.add(
            hint(
                "Pose data comes from Tools ▸ Batch Pose Tracking. Both "
                "<name>.csv and <name>DLC_<model>.csv naming are accepted."
            )
        )
        column.addWidget(sources)

        # --- clips ------------------------------------------------------
        clips = Card("Clips", "what the annotator will show you")
        # Reviewing existing work and sampling new material are the two
        # reasons to open the annotator, and the tab could only do the second.
        self._review_check = QCheckBox("Review annotations already saved for these videos")
        self._review_check.setToolTip(
            "Replay every saved behavior zone instead of sampling new clips.\n"
            "Zones are read from <pose CSV folder>/<name>_annotations.csv — the\n"
            "same files training reads."
        )
        self._review_check.toggled.connect(self._on_review_toggled)
        clips.add(self._review_check)
        clips.add_separator()

        self._clip_count = QSpinBox()
        # A 30-video cohort wants four figures of clips, so the ceiling has to
        # be well clear of anything a labeller would actually ask for.
        self._clip_count.setRange(1, 100_000)
        self._clip_count.setValue(self.DEFAULT_CLIP_COUNT)
        self._clip_count.setToolTip(
            "Total clips sampled across ALL videos, not per video.\n"
            "Raised to the number of videos if you ask for fewer."
        )
        self._clip_count_label = QLabel("Clips to sample (across all videos):")
        self._clip_count_label.setVisible(False)  # the row caption says it now
        clips.add_row("Clips to sample", self._clip_count)
        clips.add(hint("Across all videos, not per video."))

        self._skip_labelled_check = QCheckBox("Skip regions already labelled")
        self._skip_labelled_check.setToolTip(
            "Keep the sampler off frames you have already annotated, so a\n"
            "second pass over a cohort proposes new material."
        )
        clips.add(self._skip_labelled_check)
        column.addWidget(clips)

        # --- speed trace ------------------------------------------------
        speed = Card("Speed trace", "optional reference while labelling")
        # On by default: the labeller is judging speed either way, and doing it
        # against the numbers the scoring run uses is the whole point. Opting
        # out skips reading pose data altogether.
        self._speed_check = QCheckBox("Show speed trace under the clip")
        self._speed_check.setChecked(True)
        self._speed_check.setToolTip(
            "Draw each clip's speed beside the video. Reads the pose CSVs,\n"
            "which is why it can be turned off."
        )
        self._speed_check.toggled.connect(self._on_speed_toggled)
        speed.add(self._speed_check)

        self._cohort_path: Path | None = None
        self._cohort_label = path_label("(none — trace has no reference lines)")
        cohort_btn = QPushButton("Choose…")
        cohort_btn.clicked.connect(self._on_choose_cohort)
        self._cohort_row = speed.add_row("Speed thresholds", self._cohort_label, cohort_btn)

        self._calibration_master: Path | None = None
        self._calibration_label = path_label("(none — trace in px/frame)")
        calib_btn = QPushButton("Choose…")
        calib_btn.clicked.connect(self._on_choose_calibration)
        self._calibration_row = speed.add_row("Calibration", self._calibration_label, calib_btn)
        self._speed_children = (cohort_btn, calib_btn)
        column.addWidget(speed)

        column.addStretch(1)
        self._on_speed_toggled(self._speed_check.isChecked())

    def _on_speed_toggled(self, enabled: bool) -> None:
        """The trace's inputs mean nothing when the trace itself is off."""
        for widget in (
            self._cohort_label,
            self._calibration_label,
            *self._speed_children,
        ):
            widget.setEnabled(enabled)

    def _on_review_toggled(self, reviewing: bool) -> None:
        """Review mode replays saved zones, so the sampling controls don't apply."""
        for widget in (self._clip_count, self._clip_count_label, self._skip_labelled_check):
            widget.setEnabled(not reviewing)

    def _on_choose_cohort(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose cohort threshold file", "", "Cohort thresholds (*.json)"
        )
        if not path:
            return
        self._cohort_path = Path(path)
        set_path_text(self._cohort_label, Path(path).name, filled=True)
        self._cohort_label.setToolTip(path)

    def _on_choose_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose calibration file", "", "Calibration (*.json)"
        )
        if not path:
            return
        self._calibration_master = Path(path)
        set_path_text(self._calibration_label, Path(path).name, filled=True)
        self._calibration_label.setToolTip(path)

    def _on_choose_videos(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose videos folder")
        if not path:
            return
        self._videos_dir = Path(path)
        set_path_text(self._videos_label, _short_path(self._videos_dir), filled=True)
        self._videos_label.setToolTip(path)

    def _on_choose_poses(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose pose CSV folder")
        if not path:
            return
        self._poses_dir = Path(path)
        set_path_text(self._poses_label, _short_path(self._poses_dir), filled=True)
        self._poses_label.setToolTip(path)

    def _on_launch(self) -> None:
        if self._videos_dir is None:
            QMessageBox.warning(self, "Annotate", "Choose a videos folder first.")
            return
        poses_dir = self._poses_dir or self._videos_dir

        videos = sorted(p for p in self._videos_dir.iterdir() if p.suffix.lower() in _VIDEO_EXTS)
        if not videos:
            QMessageBox.warning(self, "Annotate", f"No videos found in {self._videos_dir}")
            return
        # find_pose_csv accepts both namings, so a folder Batch Pose Tracking
        # just filled works here without the operator renaming anything.
        located = [(v, find_pose_csv(v, poses_dir)) for v in videos]
        missing = [v.name for v, csv in located if csv is None]
        if missing:
            QMessageBox.warning(
                self,
                "Annotate",
                "No pose CSV found for:\n"
                + "\n".join(missing)
                + f"\n\nLooked in {poses_dir} for <name>.csv and "
                "<name>DLC_<model>.csv.\n\nPose data comes from "
                "Tools ▸ Batch Pose Tracking — run that over these videos "
                "first, then come back here.",
            )
            return
        sessions = [(v, csv) for v, csv in located if csv is not None]

        # Deferred: propose_clips_multi pulls in sklearn; AnnotatorWindow
        # pulls in cv2 via the clip player.
        from glider.analysis.behavior.annotations import AnnotationStore
        from glider.analysis.behavior.cohort_speed import (
            CohortSpeedError,
            CohortSpeedThresholds,
        )
        from glider.analysis.behavior.units import load_px_per_mm
        from glider.analysis.behavior.vocabulary import Vocabulary
        from glider.gui.behavior.annotator.app import (
            annotation_path_for,
            build_review_clips,
            make_more_sampler,
        )
        from glider.gui.behavior.annotator.capture_cache import VideoCaptureCache
        from glider.gui.behavior.annotator.main_window import AnnotatorWindow
        from glider.gui.behavior.annotator.resume_cache import ResumeCache
        from glider.gui.behavior.annotator.sampler import ProposedClip, propose_clips_multi
        from glider.vision.pose.dlc import DEFAULT_FPS, fps_for_csv

        # Clip lengths and the trim window are specified in seconds, so the
        # annotator needs the rate the video was actually recorded at. Take it
        # from what pose inference measured; fall back only for CSVs written
        # before that was recorded.
        rates = {fps_for_csv(csv) for _v, csv in sessions}
        rates.discard(None)
        fps = float(next(iter(rates))) if len(rates) == 1 else DEFAULT_FPS

        # Annotations live next to the POSE CSV — same place training reads
        # them from (mirrors annotator/app.py's run()).
        videos_meta = {v: annotation_path_for(p) for v, p in sessions}
        pairs = [(p, v) for v, p in sessions]  # (pose_csv, video) for the sampler
        try:
            if self._review_check.isChecked():
                # Every saved zone becomes a replayable clip. Nothing is
                # sampled: the point is to see the work already on disk.
                clips = build_review_clips(videos_meta, fps)
                if not clips:
                    QMessageBox.warning(
                        self,
                        "Annotate",
                        "Review mode found no annotations for these videos.\n\n"
                        f"Looked for <name>_annotations.csv in {poses_dir}.\n\n"
                        "Uncheck 'Review annotations already saved' to sample "
                        "fresh clips and start labelling.",
                    )
                    return
            else:
                # propose_clips_multi refuses a total below the video count,
                # and a labeller asking for "10 clips" across 30 videos means
                # "a few", not "crash".
                n_total = max(int(self._clip_count.value()), len(pairs))
                exclude_labeled = self._skip_labelled_check.isChecked()
                exclude_zones = None
                if exclude_labeled:
                    exclude_zones = [
                        [
                            (z.start_frame, z.end_frame)
                            for z in AnnotationStore.load_csv(videos_meta[video])
                        ]
                        for _pose_csv, video in pairs
                    ]

                # Record the sampled queue, exactly as the CLI path does. The
                # annotations CSV only ever holds the clips that got LABELLED,
                # so without this the other two hundred are gone the moment the
                # window closes and the next launch starts somewhere else.
                cache_inputs = {
                    "videos": sorted(str(v) for _p, v in pairs),
                    "n_clips": int(n_total),
                    "window": 30,
                    "fps": float(fps),
                    "random_state": 42,
                    "spatial_weight": 1.0,
                    "min_frame_gap": None,
                    "exclude_labeled": bool(exclude_labeled),
                }
                resume = ResumeCache(self._videos_dir)
                cached = resume.load(inputs=cache_inputs)
                if cached is not None:
                    clips = [ProposedClip(**c) for c in cached["clips"]]
                else:
                    clips = propose_clips_multi(
                        sessions=pairs,
                        n_clips_total=n_total,
                        fps=fps,
                        exclude_zones_by_session=exclude_zones,
                    )
                    try:
                        resume.save(inputs=cache_inputs, clip_payload=[c.__dict__ for c in clips])
                    except OSError as e:
                        # A read-only or full drive costs the resume, not the
                        # session — the clips are already in hand.
                        logger.warning("could not write the clip queue: %s", e)
        except Exception as e:  # noqa: BLE001 - surfaced to the user, not fatal
            QMessageBox.critical(self, "Annotate", f"Could not build the clip list:\n{e}")
            return

        # The window's "render more" button only exists when it is given a
        # sampler; without one, a review session is a dead end and a sampled
        # session can never be extended.
        clip_sampler = make_more_sampler(sessions, fps=fps)

        # Speed trace inputs. All three are optional and independent: no pose
        # CSVs means no trace, no cohort file means no reference lines, no
        # calibration means the trace is in px/frame.
        pose_csvs: dict[Path, Path] = {}
        scales: dict[Path, float] = {}
        cohort = None
        if self._speed_check.isChecked():
            pose_csvs = dict(sessions)
            if self._calibration_master is not None:
                for video, _pose in sessions:
                    scale = load_px_per_mm(self._calibration_master, video)
                    if scale and scale > 0:
                        scales[video] = float(scale)
            if self._cohort_path is not None:
                try:
                    cohort = CohortSpeedThresholds.load(self._cohort_path)
                except CohortSpeedError as e:
                    # Costs the reference lines, not the labelling session.
                    QMessageBox.warning(
                        self,
                        "Annotate",
                        f"Could not read the cohort threshold file:\n{e}\n\n"
                        "Opening without speed reference lines.",
                    )

        # Vocabulary fallback: a sibling of the first video, same rule as
        # annotator/app.py's run().
        vocab = Vocabulary()
        vocab_path: Path | None = videos[0].parent / f"{videos[0].stem}_behaviors.yaml"
        if vocab_path.exists():
            try:
                vocab = Vocabulary.load(vocab_path)
            except Exception:  # noqa: BLE001
                vocab_path = None
        else:
            vocab_path = None

        capture_cache = VideoCaptureCache(max_open=3)
        self._annotator_window = AnnotatorWindow(
            clips=clips,
            videos_meta=videos_meta,
            fps=fps,
            vocab=vocab,
            vocab_path=vocab_path,
            capture_cache=capture_cache,
            clip_sampler=clip_sampler,
            pose_csvs=pose_csvs,
            cohort=cohort,
            px_per_mm=scales,
        )
        self._annotator_window.show()
        self._annotator_window.warn_about_load_errors()


class TrainTab(QWidget):
    """Fit a behavior classifier from labeled sessions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sessions: list[tuple[Path, Path]] = []
        self._holdout: list[tuple[Path, Path]] = []
        self._output_path: Path | None = None
        self._train_thread: QThread | None = None
        self._train_worker = None
        # None until the user accepts the Advanced dialog at least once.
        # Staying None keeps the library's own defaults the single source of
        # truth instead of freezing today's values into every trained model.
        self._lgbm_advanced: dict | None = None

        # --- run rail ---------------------------------------------------
        rail = RunRail("Fit model")
        self._fit_btn = rail.button
        self._fit_btn.clicked.connect(self._on_fit)
        self._rail = rail

        # Cross-validation is its own action, not a mode of Fit: it
        # deliberately produces NO model, only a measurement, and folding it
        # into the same button would invite saving something that does not
        # exist. It is the only way to get a number with a spread on it — a
        # single cross-session holdout on this kind of data varies by ~0.09
        # macro F1 depending purely on which animals land in it.
        self._folds_spin = QSpinBox()
        self._folds_spin.setRange(2, 100)
        self._folds_spin.setValue(5)
        self._folds_spin.setToolTip(
            "Whole sessions are split across folds, never frames within a\n"
            "session, so every fold is scored on animals the model never saw.\n"
            "At or above the number of sessions this is leave-one-out."
        )
        self._cv_btn = QPushButton("Cross-validate")
        set_button_role(self._cv_btn, "ghost")
        self._cv_btn.setToolTip(
            "Measure generalization across all training sessions.\n"
            "Produces no model — use Fit for that."
        )
        self._cv_btn.clicked.connect(self._on_cross_validate)
        rail.card.add(_button_row(QLabel("Folds"), self._folds_spin, self._cv_btn))

        # Refitting on everything after measuring is the standard way to turn a
        # CV estimate into a deployable model. Done as two separate actions it
        # assembles the features twice — and with motion features that means
        # decoding every source video twice — so it is offered here instead.
        self._cv_fit_check = QCheckBox("…and fit a model on all sessions")
        self._cv_fit_check.setToolTip(
            "Refit on every session after measuring, and save the bundle.\n"
            "One feature pass for both, and the saved model carries the\n"
            "cross-validated score instead of a meaningless 1.000 train\n"
            "accuracy.\n\n"
            "The CV number understates this model slightly: it saw every\n"
            "session, while each fold model saw only a fraction."
        )
        rail.card.add(self._cv_fit_check)
        rail.card.add(
            hint(
                "Cross-validation splits the TRAINING sessions itself and "
                "ignores the holdout list. It reports a mean and a spread "
                "instead of one split's number, and saves nothing."
            )
        )

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        rail.card.add(self._progress)

        results_card = Card("Results")
        self._results = QTextEdit()
        self._results.setObjectName("LogPane")
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Training results will appear here.")
        results_card.add(self._results, 1)
        rail.add(results_card, 1)

        column = _workspace(self, rail)

        # --- training sessions ------------------------------------------
        sessions_group = Card("Training sessions", "pose CSV + annotations CSV")
        self._sessions_card = sessions_group
        self._sessions_list = QListWidget()
        self._sessions_list.setMinimumHeight(96)
        self._sessions_list.setMaximumHeight(150)
        self._sessions_list.setSelectionMode(_MULTI_SELECT)
        attach_empty_state(
            self._sessions_list,
            "No sessions yet.\nAdd pose CSVs; annotations are found beside them.",
        )
        sessions_group.add(self._sessions_list)
        add_btn = QPushButton("Add sessions…")
        add_btn.setToolTip(
            "Pick any number of pose CSVs. Each session's annotations are\n"
            "taken from <name>_annotations.csv beside it — the same file\n"
            "training reads."
        )
        add_btn.clicked.connect(self._on_add_session)
        remove_btn = QPushButton("Remove selected")
        set_button_role(remove_btn, "ghost")
        remove_btn.clicked.connect(self._on_remove_session)
        sessions_group.add(_button_row(add_btn, remove_btn))
        column.addWidget(sessions_group)

        # --- holdout ----------------------------------------------------
        holdout_group = Card("Holdout sessions", "optional cross-session test set")
        self._holdout_card = holdout_group
        self._holdout_list = QListWidget()
        self._holdout_list.setMinimumHeight(80)
        self._holdout_list.setMaximumHeight(130)
        self._holdout_list.setSelectionMode(_MULTI_SELECT)
        attach_empty_state(
            self._holdout_list,
            "No holdout sessions.\nAccuracy will be reported on training data only.",
        )
        holdout_group.add(self._holdout_list)
        add_holdout_btn = QPushButton("Add holdout sessions…")
        add_holdout_btn.clicked.connect(self._on_add_holdout)
        remove_holdout_btn = QPushButton("Remove selected")
        set_button_role(remove_holdout_btn, "ghost")
        remove_holdout_btn.clicked.connect(self._on_remove_holdout)
        holdout_group.add(_button_row(add_holdout_btn, remove_holdout_btn))
        holdout_group.add(
            hint(
                "Sessions held back from fitting, so the reported accuracy is "
                "measured on animals the model has never seen."
            )
        )
        column.addWidget(holdout_group)

        # --- classifier -------------------------------------------------
        options = Card("Classifier")
        self._classifier_combo = QComboBox()
        # train_model(classifier_type=...) accepts exactly "rf"
        # (RandomForestClassifier) or "lightgbm" (LGBMClassifier, and the
        # library-side default) — see
        # glider.analysis.behavior.pipeline.train_model docstring.
        self._classifier_combo.addItems(["rf", "lightgbm"])
        self._classifier_combo.currentTextChanged.connect(self._on_classifier_changed)
        self._advanced_btn = QPushButton("Advanced…")
        self._advanced_btn.clicked.connect(self._on_advanced)
        options.add_row("Backend", self._classifier_combo, self._advanced_btn)

        self._background_check = QCheckBox("Include background class")
        self._background_check.setToolTip(
            "Score unlabelled frames as an explicit 'background' behavior "
            "rather than leaving them out of the fit."
        )
        self._mirror_check = QCheckBox("Mirror augment")
        self._mirror_check.setToolTip(
            "Double the training set by left-right mirroring every pose, so a "
            "behavior is learned independently of which way the animal faced."
        )
        options.add(self._background_check)
        options.add(self._mirror_check)

        # Pipeline knobs, deliberately here rather than in the Advanced
        # dialog: that dialog is LightGBM-only and is disabled for the
        # RandomForest backend, which would leave these unreachable. They also
        # move the score far more than the LightGBM knobs do — measured over
        # five cross-session folds on a real cohort, window + class weight +
        # mirroring together were worth +0.063 macro F1, winning 5 folds of 5.
        self._window_spin = QSpinBox()
        self._window_spin.setRange(1, 600)
        self._window_spin.setValue(30)
        self._window_spin.setToolTip(
            "How many frames of context each prediction sees.\n"
            "Shorter reacts faster and captures brief bouts; longer is\n"
            "steadier but smears the boundaries between behaviors."
        )
        self._window_spin.valueChanged.connect(self._refresh_window_hint)
        options.add_row("Window (frames)", self._window_spin)
        self._window_hint = hint("")
        options.add(self._window_hint)
        self._refresh_window_hint()

        self._class_weight_combo = QComboBox()
        self._class_weight_combo.addItem("none", None)
        self._class_weight_combo.addItem("balanced", "balanced")
        self._class_weight_combo.setToolTip(
            "'balanced' weights each class by the inverse of how often it\n"
            "occurs, so a rare behavior is not simply ignored in favour of\n"
            "the common ones. Worth trying whenever one class is scarce."
        )
        options.add_row("Class weight", self._class_weight_combo)

        self._test_split_spin = QDoubleSpinBox()
        self._test_split_spin.setRange(0.0, 0.5)
        self._test_split_spin.setSingleStep(0.05)
        self._test_split_spin.setDecimals(2)
        self._test_split_spin.setValue(0.0)
        self._test_split_spin.setToolTip(
            "Fraction of each session held back for testing when no holdout\n"
            "SESSIONS are set. Ignored when they are — a cross-session\n"
            "holdout is the stronger test. At 0 with no holdout sessions the\n"
            "run reports training accuracy only, which is not a\n"
            "generalization estimate."
        )
        options.add_row("Within-session test split", self._test_split_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999_999)
        self._seed_spin.setValue(42)
        self._seed_spin.setToolTip("Change to check how much of a score is fit noise.")
        options.add_row("Random seed", self._seed_spin)

        # Feature families the pipeline can compute but never did: all three
        # default to False in train_model and had no way to be switched on.
        self._traj_check = QCheckBox("Trajectory shape features")
        self._traj_check.setToolTip(
            "Straightness, path length, net displacement, radius of gyration\n"
            "and total turning over each window — the shape of the path\n"
            "rather than the pose. Aimed at telling travelling apart from\n"
            "milling about in one spot."
        )
        self._motion_check = QCheckBox("Motion features")
        self._motion_check.setToolTip(
            "Frame-differencing features read from the source video.\n\n"
            "Cannot be combined with mirror augmentation: the pipeline\n"
            "mirrors the pose but not the video, so the two would disagree.\n"
            "train_model refuses the combination outright."
        )
        self._freq_check = QCheckBox("Frequency features")
        self._freq_check.setToolTip(
            "Dominant frequency and spectral flatness per kinematic column —\n"
            "for rhythmic behaviors such as grooming."
        )
        options.add(self._traj_check)
        options.add(self._motion_check)
        options.add(self._freq_check)
        # train_model raises on motion + mirror together, and the raise lands
        # minutes into a fit with the model lost. Enforce it here, where the
        # click happens, rather than letting the run start and fail.
        self._mirror_check.toggled.connect(self._on_motion_mirror_toggled)
        self._motion_check.toggled.connect(self._on_motion_mirror_toggled)
        self._motion_mirror_hint = hint("")
        options.add(self._motion_mirror_hint)
        self._on_motion_mirror_toggled()
        options.add(
            hint(
                "These three are computed but off by default, so no existing "
                "model used them. Turning one on changes the feature set, and "
                "the resulting model is not comparable to one trained without it."
            )
        )

        column.addWidget(options)
        self._on_classifier_changed()

        # --- output -----------------------------------------------------
        output = Card("Output")
        self._output_label = path_label("(none)")
        output_btn = QPushButton("Choose…")
        output_btn.clicked.connect(self._on_choose_output)
        output.add_row("Model bundle", self._output_label, output_btn)
        output.add(hint("A .pkl the Apply tab loads to score new video."))
        column.addWidget(output)

        column.addStretch(1)
        self._refresh_counts()

    def _refresh_counts(self) -> None:
        """Keep the card badges showing how much material is loaded."""
        n = len(self._sessions)
        self._sessions_card.set_badge(f"{n} session{'' if n == 1 else 's'}" if n else "none yet")
        n = len(self._holdout)
        self._holdout_card.set_badge(f"{n} session{'' if n == 1 else 's'}" if n else "none yet")
        self._rail.set_blocker("" if self._sessions else "Add at least one training session.")

    def _on_remove_session(self) -> None:
        _remove_selected(self._sessions_list, self._sessions)
        self._refresh_counts()

    def _on_remove_holdout(self) -> None:
        _remove_selected(self._holdout_list, self._holdout)
        self._refresh_counts()

    def _on_motion_mirror_toggled(self, *_args) -> None:
        """Keep motion features and mirror augmentation mutually exclusive.

        The pipeline mirrors the pose but not the source video, so motion
        features read from the video would disagree with the mirrored pose;
        ``train_model`` refuses the pair. Whichever the operator picked most
        recently wins, and the other is cleared and greyed out with the reason
        on screen — a disabled box with no explanation reads as a bug.
        """
        mirror, motion = self._mirror_check, self._motion_check
        # Signals are blocked while clearing so the two handlers do not
        # bounce the state between them.
        if mirror.isChecked() and motion.isChecked():
            loser = motion if self.sender() is mirror else mirror
            loser.blockSignals(True)
            loser.setChecked(False)
            loser.blockSignals(False)

        motion.setEnabled(not mirror.isChecked())
        mirror.setEnabled(not motion.isChecked())
        if mirror.isChecked():
            note = "Motion features are unavailable while mirror augment is on."
        elif motion.isChecked():
            note = "Mirror augment is unavailable while motion features are on."
        else:
            note = "Motion features and mirror augment cannot be used together."
        self._motion_mirror_hint.setText(note)

    def _refresh_window_hint(self) -> None:
        """Say what the window is in seconds; frames alone mean nothing."""
        frames = self._window_spin.value()
        self._window_hint.setText(f"{frames / 30.0:g} s at 30 fps")

    def _on_classifier_changed(self, *_args) -> None:
        """The advanced knobs are LightGBM-only; RandomForest silently ignores them."""
        lgbm = self._classifier_combo.currentText() == "lightgbm"
        self._advanced_btn.setEnabled(lgbm)
        self._advanced_btn.setToolTip(
            "Tune LightGBM's capacity and regularization per knob."
            if lgbm
            else "LightGBM only — the RandomForest backend ignores these knobs."
        )

    def _on_advanced(self) -> None:
        dialog = LgbmAdvancedDialog(self._lgbm_advanced, self)
        if dialog.exec():
            self._lgbm_advanced = dialog.values()

    def _lgbm_options(self) -> dict:
        """``train_model`` kwargs for the advanced knobs, or ``{}`` when untouched.

        Returns nothing for the RandomForest backend even when the dialog has
        been filled in: ``_make_classifier`` ignores ``lgbm_reg`` there, so
        passing it would imply an effect the run won't have.
        """
        if not self._lgbm_advanced or self._classifier_combo.currentText() != "lightgbm":
            return {}
        # Imported here, not at module scope: pipeline pulls in sklearn/pandas
        # and this window must import cleanly under a bare [pc] install.
        from glider.analysis.behavior.pipeline import LgbmReg

        values = dict(self._lgbm_advanced)
        # n_estimators is a train_model kwarg, the rest are LgbmReg fields.
        n_estimators = int(values.pop("n_estimators"))
        return {"n_estimators": n_estimators, "lgbm_reg": LgbmReg(**values)}

    def _on_add_session(self) -> None:
        self._add_sessions("training", self._sessions, self._sessions_list)

    def _on_add_holdout(self) -> None:
        self._add_sessions("holdout", self._holdout, self._holdout_list)

    def _add_sessions(
        self, kind: str, backing: list[tuple[Path, Path]], list_widget: QListWidget
    ) -> None:
        """Add every selected session, reporting whatever could not be added.

        Silence would be the wrong default here: selecting thirty files and
        getting twenty-eight rows is invisible unless the two missing ones are
        named.
        """
        pairs, skipped = _pick_session_pairs(self, kind)
        if not pairs and not skipped:
            return  # cancelled

        already = {pose for pose, _ann in backing}
        added = 0
        for pose, ann_path in pairs:
            if pose in already:
                skipped.append(f"{pose.name} (already added)")
                continue
            already.add(pose)
            backing.append((pose, ann_path))
            self._add_session_item(list_widget, (pose, ann_path))
            added += 1
        self._refresh_counts()

        if skipped:
            QMessageBox.warning(
                self,
                "Train",
                f"Added {added} {kind} session(s).\n\n"
                f"Skipped {len(skipped)}:\n" + "\n".join(f"  • {s}" for s in skipped),
            )

    @staticmethod
    def _add_session_item(list_widget: QListWidget, pair: tuple[Path, Path]) -> None:
        """One row per session, named rather than spelled out in full.

        Two absolute paths joined by a pipe overflowed the row every time, so
        the list showed a scrollbar and no filenames. The names identify the
        session; the tooltip still carries both paths in full.
        """
        list_widget.addItem(f"{pair[0].name}  ·  {pair[1].name}")
        item = list_widget.item(list_widget.count() - 1)
        item.setToolTip(f"{pair[0]}\n{pair[1]}")

    def _on_choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model bundle", "", "Model files (*.pkl);;All files (*)"
        )
        if not path:
            return
        self._output_path = Path(path)
        set_path_text(self._output_label, _short_path(self._output_path), filled=True)
        self._output_label.setToolTip(path)

    def _shared_options(self) -> dict:
        """Settings both Fit and Cross-validate accept.

        Shared deliberately: a cross-validation run configured differently
        from the fit it is meant to estimate is not an estimate of anything.
        ``test_split`` and ``holdout_sessions`` are NOT here — cross-validation
        takes neither, and it makes its own splits.
        """
        options: dict = {
            "classifier_type": self._classifier_combo.currentText(),
            "include_background": self._background_check.isChecked(),
            "mirror_augment": self._mirror_check.isChecked(),
            "window": int(self._window_spin.value()),
            # currentData, not currentText: the pipeline wants None, and the
            # string "none" would be passed through to sklearn as a weighting
            # scheme it does not recognise.
            "class_weight": self._class_weight_combo.currentData(),
            "random_state": int(self._seed_spin.value()),
            "traj_features": self._traj_check.isChecked(),
            "motion_features": self._motion_check.isChecked(),
            "freq_features": self._freq_check.isChecked(),
        }
        options.update(self._lgbm_options())
        return options

    def _on_cross_validate(self) -> None:
        """Measure generalization over the training sessions. Saves nothing."""
        if not self._sessions:
            QMessageBox.warning(self, "Train", "Add at least one training session first.")
            return
        also_fit = self._cv_fit_check.isChecked()
        if also_fit and self._output_path is None:
            QMessageBox.warning(
                self,
                "Train",
                "Choose a model output file first, or untick "
                "'…and fit a model on all sessions' to measure only.",
            )
            return

        from glider.gui.behavior import workers as workers_mod

        options = self._shared_options()
        options["n_folds"] = int(self._folds_spin.value())

        self._train_thread = QThread()
        self._train_worker = workers_mod.CrossValidateWorker(
            list(self._sessions), options, self._output_path if also_fit else None
        )
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.finished.connect(self._on_cv_finished)
        self._train_worker.failed.connect(self._on_train_failed)

        self._results.clear()
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._fit_btn.setEnabled(False)
        self._cv_btn.setEnabled(False)
        self._rail.status.set_state("running", f"Cross-validating ({options['n_folds']} folds)")
        self._train_thread.start()

    def _on_cv_finished(self, result: object) -> None:
        from glider.analysis.behavior.summary_text import format_cv_summary

        self._teardown_train_thread()
        self._progress.setVisible(False)
        self._fit_btn.setEnabled(True)
        self._cv_btn.setEnabled(True)
        self._rail.status.set_state("ok", "Done")
        self._results.setPlainText(format_cv_summary(result))

    def _on_fit(self) -> None:
        if not self._sessions:
            QMessageBox.warning(self, "Train", "Add at least one training session first.")
            return
        if self._output_path is None:
            QMessageBox.warning(self, "Train", "Choose a model output file first.")
            return

        from glider.gui.behavior.workers import TrainWorker

        options = self._shared_options()
        options["test_split"] = float(self._test_split_spin.value())
        if self._holdout:
            options["holdout_sessions"] = list(self._holdout)

        self._train_thread = QThread()
        self._train_worker = TrainWorker(list(self._sessions), self._output_path, options)
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._on_train_finished)
        self._train_worker.failed.connect(self._on_train_failed)

        self._results.clear()
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate until fit finishes
        self._fit_btn.setEnabled(False)
        self._rail.status.set_state("running", "Fitting")
        self._train_thread.start()

    def _on_train_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)

    def _on_train_finished(self, summary: object) -> None:
        self._teardown_train_thread()
        self._progress.setVisible(False)
        self._fit_btn.setEnabled(True)
        self._rail.status.set_state("ok", "Done")
        # Deferred like the rest of the analysis imports, though this one is
        # cheap by design — it exists so the results pane never needs pandas.
        from glider.analysis.behavior.summary_text import format_training_summary

        self._results.setPlainText(format_training_summary(summary))

    def _on_train_failed(self, message: str) -> None:
        # Shared by Fit and Cross-validate, so both buttons come back —
        # otherwise a failed cross-validation leaves its own button dead.
        self._cv_btn.setEnabled(True)
        self._teardown_train_thread()
        self._progress.setVisible(False)
        self._fit_btn.setEnabled(True)
        self._rail.status.set_state("error", "Failed")
        QMessageBox.critical(self, "Training failed", message)

    def _teardown_train_thread(self) -> None:
        if self._train_thread is not None:
            self._train_thread.quit()
            self._train_thread.wait(5000)
            self._train_thread.deleteLater()
            self._train_thread = None
        if self._train_worker is not None:
            self._train_worker.deleteLater()
            self._train_worker = None


class ApplyTab(QWidget):
    """Classify recorded video(s) with a trained model and write the ethogram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model_path: Path | None = None
        self._yolo_path: Path | None = None
        self._videos: list[Path] = []
        self._output_dir: Path | None = None
        self._calibration_master: Path | None = None
        self._cohort_path: Path | None = None
        self._cohort_thread: QThread | None = None
        self._cohort_worker = None
        self._apply_thread: QThread | None = None
        self._apply_worker = None
        # Videos process one at a time (ApplyWorker takes a single video);
        # this is the remaining queue for the current Run, each writing to
        # its own <output_dir>/<video stem>/ subfolder.
        self._queue: list[Path] = []

        # --- run rail ---------------------------------------------------
        rail = RunRail("Score videos")
        self._run_btn = rail.button
        self._run_btn.clicked.connect(self._on_run)
        self._rail = rail

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        rail.card.add(self._progress)

        results_card = Card("Output")
        self._results = QTextEdit()
        self._results.setObjectName("LogPane")
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Output paths will appear here.")
        results_card.add(self._results, 1)
        rail.add(results_card, 1)

        column = _workspace(self, rail)

        # --- models -----------------------------------------------------
        # Both models are optional, and the labels say so where the operator
        # is deciding: no bundle scores freezing/darting alone, and no weights
        # is fine as long as the poses already exist. Clearing is offered
        # because choosing one is otherwise irreversible, and switching to a
        # speed-only run after picking a bundle is a normal thing to want.
        # No "both optional" subtitle: each field's own placeholder already says
        # what happens without it, and more precisely than the card could.
        models = Card("Models")
        self._model_label = QLabel("Model bundle: (none — scores freezing/darting only)")
        self._model_label.setVisible(False)  # kept as the text of record
        self._model_path_label = path_label("(none — scores freezing/darting only)")
        model_btn = QPushButton("Choose…")
        model_btn.clicked.connect(self._on_choose_model)
        model_clear = QPushButton("Clear")
        set_button_role(model_clear, "ghost")
        model_clear.setToolTip(
            "Run without a behaviour model: freezing and darting are read from "
            "the speed trace, which needs no classifier."
        )
        model_clear.clicked.connect(self._on_clear_model)
        models.add_row("Model bundle", self._model_path_label, model_btn, model_clear)

        self._yolo_label = QLabel("YOLO weights: (none — needed only to track poses)")
        self._yolo_label.setVisible(False)
        self._yolo_path_label = path_label("(none — needed only to track poses)")
        yolo_btn = QPushButton("Choose…")
        yolo_btn.clicked.connect(self._on_choose_yolo)
        yolo_clear = QPushButton("Clear")
        set_button_role(yolo_clear, "ghost")
        yolo_clear.setToolTip(
            "Only tracking uses the pose weights. A run whose videos all have "
            "a pose CSV never needs them."
        )
        yolo_clear.clicked.connect(self._on_clear_yolo)
        models.add_row("YOLO weights", self._yolo_path_label, yolo_btn, yolo_clear)
        column.addWidget(models)

        # --- videos -----------------------------------------------------
        videos_group = Card("Videos to classify")
        self._videos_card = videos_group
        self._videos_list = QListWidget()
        self._videos_list.setMinimumHeight(96)
        self._videos_list.setMaximumHeight(160)
        attach_empty_state(self._videos_list, "No videos yet.\nAdd the recordings you want scored.")
        videos_group.add(self._videos_list)
        add_videos_btn = QPushButton("Add video(s)…")
        add_videos_btn.clicked.connect(self._on_add_videos)
        remove_video_btn = QPushButton("Remove selected")
        set_button_role(remove_video_btn, "ghost")
        remove_video_btn.clicked.connect(self._on_remove_video)
        videos_group.add(_button_row(add_videos_btn, remove_video_btn))
        column.addWidget(videos_group)

        # --- keypoints --------------------------------------------------
        keypoints = Card("Keypoints", "must match the model's own order")
        self._keypoints_edit = QLineEdit()
        self._keypoints_edit.setPlaceholderText("nose, left_ear, right_ear, ... (comma-separated)")
        edit_kp = QPushButton("Edit…")
        edit_kp.setToolTip(
            "Arrange the keypoints on a figure instead of typing the order, "
            "and save the layout for reuse."
        )
        edit_kp.clicked.connect(self._edit_keypoint_schema)
        keypoints.add_row("Names", self._keypoints_edit, edit_kp)
        keypoints.add(hint("Filled in automatically when a model bundle is chosen."))
        column.addWidget(keypoints)

        # --- pose reuse -------------------------------------------------
        # Checked by default: Batch Pose Tracking has usually already produced
        # these, and re-deriving them is the biggest avoidable cost in a run.
        poses = Card("Pose data")
        self._reuse_poses = QCheckBox("Reuse already-tracked pose CSVs (skips tracking)")
        self._reuse_poses.setChecked(True)
        self._reuse_poses.setToolTip(
            "Reads the poses Batch Pose Tracking already wrote instead of "
            "running the pose model again — by far the biggest cost in a run. "
            "Falls back to tracking for any video without one."
        )
        self._reuse_poses.toggled.connect(self._on_reuse_toggled)
        poses.add(self._reuse_poses)

        # Batch Pose Tracking writes its CSVs wherever it was pointed, which
        # is routinely a poses folder rather than beside the videos. Matching
        # by name from one folder beats copying a CSV per video by hand.
        self._pose_dir: Path | None = None
        self._pose_dir_label = QLabel("Pose CSV folder: beside each video")
        self._pose_dir_label.setVisible(False)  # kept as the text of record
        self._pose_dir_label.setWordWrap(False)
        self._pose_dir_value = path_label("beside each video")
        self._pose_dir_btn = QPushButton("Choose…")
        self._pose_dir_btn.clicked.connect(self._on_choose_pose_dir)
        self._pose_dir_clear = QPushButton("Use video folder")
        set_button_role(self._pose_dir_clear, "ghost")
        self._pose_dir_clear.clicked.connect(self._on_clear_pose_dir)
        poses.add_row("CSV folder", self._pose_dir_value, self._pose_dir_btn, self._pose_dir_clear)

        self._pose_match_label = hint("")
        poses.add(self._pose_match_label)
        column.addWidget(poses)

        # --- scoring ----------------------------------------------------
        scoring = Card("Scoring")
        scoring.add(self._build_cadence_row())
        scoring.add(self._build_time_range_row())
        scoring.add_separator()
        scoring.add(self._build_stability_group())
        column.addWidget(scoring)

        # --- speed axis -------------------------------------------------
        speed_card = Card("Freeze / dart speed axis", "optional")
        self._speed_card = speed_card
        speed_card.add(self._build_speed_group())
        column.addWidget(speed_card)

        # --- output -----------------------------------------------------
        output = Card("Output")
        self._output_label = QLabel("Output folder: (none)")
        self._output_label.setVisible(False)  # kept as the text of record
        self._output_value = path_label("(none)")
        output_btn = QPushButton("Choose…")
        output_btn.clicked.connect(self._on_choose_output_dir)
        output.add_row("Folder", self._output_value, output_btn)
        output.add(hint("Each video gets its own subfolder here."))

        # Encoding an annotated MP4 costs more wall-clock than the inference
        # itself on a long recording, and it is a spot-checking aid rather than
        # an analysis artifact -- so it is off unless asked for.
        self._render_video = QCheckBox("Also render an annotated video (slow)")
        self._render_video.setChecked(False)
        self._render_video.setToolTip(
            "Writes annotated.mp4 beside the ethogram. Useful for checking a "
            "single video; a large cost per video across a batch."
        )
        output.add(self._render_video)
        column.addWidget(output)

        column.addStretch(1)
        self._refresh_blocker()

    def _refresh_blocker(self) -> None:
        """Say why Run is unavailable, in the rail, before it is ever clicked.

        ``_run_blocker`` already knew all of this; it was only ever consulted
        after the click, so the operator configured the whole screen and then
        found out. Shown continuously it becomes a checklist instead.
        """
        n = len(self._videos)
        self._videos_card.set_badge(f"{n} video{'' if n == 1 else 's'}" if n else "none yet")
        self._speed_card.set_badge("on" if self._speed_group.isChecked() else "off")
        if not self._videos:
            message = "Add at least one video."
        elif self._output_dir is None:
            message = "Choose an output folder."
        else:
            message = self._run_blocker() or ""
        self._rail.set_blocker(message)

    def _on_choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose model bundle", "", "Model files (*.pkl);;All files (*)"
        )
        if not path:
            return
        self._model_path = Path(path)
        self._model_label.setText(f"Model bundle: {path}")
        set_path_text(self._model_path_label, _short_path(self._model_path), filled=True)
        self._model_path_label.setToolTip(path)
        self._autofill_keypoints()
        self._refresh_blocker()

    def _on_clear_model(self) -> None:
        self._model_path = None
        self._model_label.setText("Model bundle: (none — scores freezing/darting only)")
        set_path_text(self._model_path_label, "(none — scores freezing/darting only)", filled=False)
        self._model_path_label.setToolTip("")
        self._refresh_blocker()

    def _on_clear_yolo(self) -> None:
        self._yolo_path = None
        self._yolo_label.setText("YOLO weights: (none — needed only to track poses)")
        set_path_text(self._yolo_path_label, "(none — needed only to track poses)", filled=False)
        self._yolo_path_label.setToolTip("")
        self._refresh_blocker()

    def _autofill_keypoints(self) -> None:
        """Write the bundle's own keypoint order into the field.

        The bundle records the order it was trained with, so there is nothing
        for the operator to remember or retype. Not making them guess is a
        stronger safeguard than warning about a wrong guess.
        """
        from glider.analysis.behavior.classify.features_stream import expected_keypoint_order
        from glider.analysis.behavior.model import BehaviorModel

        try:
            expected = expected_keypoint_order(BehaviorModel.load(self._model_path))
        except Exception:
            logger.debug("could not read keypoint order from the bundle", exc_info=True)
            return
        if not expected:
            return
        current = [n.strip() for n in self._keypoints_edit.text().split(",") if n.strip()]
        if current == expected:
            return
        self._keypoints_edit.setText(",".join(expected))
        if current:
            # Say so rather than silently rewriting something they typed.
            self._results.append(f"Keypoint names set from the model bundle: {','.join(expected)}")

    def _on_choose_yolo(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose YOLO weights", "", "Weights (*.pt);;All files (*)"
        )
        if not path:
            return
        self._yolo_path = Path(path)
        self._yolo_label.setText(f"YOLO weights: {path}")
        set_path_text(self._yolo_path_label, _short_path(self._yolo_path), filled=True)
        self._yolo_path_label.setToolTip(path)
        self._refresh_blocker()

    def _on_add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose video(s)", "", _VIDEO_FILTER)
        for path in paths:
            self._videos.append(Path(path))
            self._videos_list.addItem(Path(path).name)
            self._videos_list.item(self._videos_list.count() - 1).setToolTip(path)
        self._refresh_pose_match()

    def _confirm_unmatched_poses(self) -> bool:
        """Ask before tracking videos whose poses could not be found.

        Falling back to tracking is the documented behaviour, but it is also
        the difference between seconds and tens of minutes per video, and a
        misspelt folder looks exactly like a slow run. Default is No.
        """
        if not self._reuse_poses.isChecked():
            return True
        _matched, unmatched = self._match_poses()
        if not unmatched:
            return True
        where = str(self._pose_dir) if self._pose_dir else "each video's own folder"
        names = "\n".join(f"  {v.name}" for v in unmatched[:12])
        more = f"\n  ...and {len(unmatched) - 12} more" if len(unmatched) > 12 else ""
        answer = QMessageBox.question(
            self,
            "Pose CSVs not found",
            f"No pose CSV was found for {len(unmatched)} of {len(self._videos)} "
            f"video(s) in {where}:\n\n{names}{more}\n\n"
            "Those will be tracked from scratch, which takes far longer than "
            "reusing existing poses. Looked for <name>.csv and "
            "<name>DLC_<model>.csv.\n\nRun anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _scale_advisories(self) -> list[str]:
        """Quiet mis-scalings worth knowing about before a cohort is scored.

        Both failures produce a plausible ethogram rather than an error, so
        nothing downstream will ever flag them. Never raises: a diagnostic
        that breaks a run is worse than one that stays silent.
        """
        from glider.analysis.behavior.scale_guard import (
            calibration_spread_warning,
            scale_warning,
        )

        messages: list[str] = []
        try:
            if self._videos and self._model_path is not None:
                pose_csv = find_pose_csv(self._videos[0], self._pose_dir)
                if pose_csv is not None:
                    from glider.analysis.behavior.classify.pipeline import _load_behavior_model
                    from glider.vision.pose.dlc import from_dlc_csv

                    message = scale_warning(
                        _load_behavior_model(self._model_path), from_dlc_csv(pose_csv)
                    )
                    if message:
                        messages.append(message)
        except Exception:  # noqa: BLE001 - advisory only
            logger.debug("scale check skipped", exc_info=True)
        try:
            if self._calibration_master is not None:
                message = calibration_spread_warning(self._calibration_master)
                if message:
                    messages.append(message)
        except Exception:  # noqa: BLE001 - advisory only
            logger.debug("calibration spread check skipped", exc_info=True)
        message = self._cohort_window_mismatch()
        if message:
            messages.append(message)
        return messages

    def _cohort_window_mismatch(self) -> str | None:
        """Whether the cohort thresholds cover a different stretch than this run.

        Freezing is a percentile of the pooled speed, so it moves with the
        stretch it was pooled over — by a third on a real cohort between
        whole recordings and minutes two to seven. Scoring one window against
        another's cut-offs is not an error and produces no symptom, which is
        exactly why it is worth saying out loud.
        """
        if self._cohort_path is None:
            return None
        try:
            from glider.analysis.behavior.cohort_speed import CohortSpeedThresholds

            thresholds = CohortSpeedThresholds.load(self._cohort_path)
        except Exception:  # noqa: BLE001 - advisory only
            logger.debug("could not read the cohort thresholds", exc_info=True)
            return None

        start_s, end_s = self._time_range()
        run_window = None if start_s is None and end_s is None else (start_s, end_s)
        if run_window == thresholds.window:
            return None
        run_described = "the whole recording"
        if run_window is not None:
            tail = f"{end_s / 60:g} min" if end_s is not None else "the end"
            run_described = f"{(start_s or 0.0) / 60:g}–{tail}"
        return (
            f"The cohort thresholds were pooled over {thresholds.describe_window()}, "
            f"but this run scores {run_described}.\n\n"
            "Freezing is a percentile of the pooled speed, so it moves with the "
            "stretch it came from — on a 30-animal cohort the freezing cut-off "
            "differed by a third between the whole recording and minutes 2–7. "
            "Nothing will look wrong in the output.\n\n"
            "Re-pool the thresholds over the same window, or clear the window."
        )

    def _confirm_scale_advisories(self) -> bool:
        messages = self._scale_advisories()
        if not messages:
            return True
        answer = QMessageBox.question(
            self,
            "Before running",
            "\n\n———\n\n".join(messages) + "\n\nRun anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _on_remove_video(self) -> None:
        _remove_selected(self._videos_list, self._videos)
        self._refresh_pose_match()
        self._refresh_blocker()

    # ------------------------------------------------------------------
    # Pose CSV folder
    # ------------------------------------------------------------------

    def _on_choose_pose_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder of pose CSVs")
        if not path:
            return
        self._pose_dir = Path(path)
        self._pose_dir_label.setText(f"Pose CSV folder: {_short_path(self._pose_dir)}")
        self._pose_dir_label.setToolTip(path)
        set_path_text(self._pose_dir_value, _short_path(self._pose_dir), filled=True)
        self._pose_dir_value.setToolTip(path)
        self._refresh_pose_match()

    def _on_clear_pose_dir(self) -> None:
        self._pose_dir = None
        self._pose_dir_label.setText("Pose CSV folder: beside each video")
        self._pose_dir_label.setToolTip("")
        set_path_text(self._pose_dir_value, "beside each video", filled=False)
        self._pose_dir_value.setToolTip("")
        self._refresh_pose_match()

    def _on_reuse_toggled(self, checked: bool) -> None:
        for w in (
            self._pose_dir_label,
            self._pose_dir_value,
            self._pose_dir_btn,
            self._pose_dir_clear,
        ):
            w.setEnabled(checked)
        self._refresh_pose_match()

    def _match_poses(self) -> tuple[list[Path], list[Path]]:
        """``(matched, unmatched)`` videos for the current folder setting."""
        matched, unmatched = [], []
        for video in self._videos:
            try:
                found = find_pose_csv(video, self._pose_dir)
            except OSError:  # an unreachable share must not break the summary
                found = None
            (matched if found is not None else unmatched).append(video)
        return matched, unmatched

    def _refresh_pose_match(self) -> None:
        """Say how many videos this folder covers, before anything is run.

        Worth showing continuously rather than at Run: an unmatched video
        silently costs a full tracking pass, which is the single most
        expensive thing an apply run can do.
        """
        if not self._reuse_poses.isChecked():
            set_text_role(self._pose_match_label, "warning")
            self._pose_match_label.setText("Tracking will run for every video.")
            self._refresh_blocker()
            return
        if not self._videos:
            set_text_role(self._pose_match_label, "hint")
            self._pose_match_label.setText("")
            self._refresh_blocker()
            return
        matched, unmatched = self._match_poses()
        where = _short_path(self._pose_dir) if self._pose_dir else "each video's own folder"
        self._pose_match_label.setToolTip(str(self._pose_dir) if self._pose_dir else "")
        if not unmatched:
            set_text_role(self._pose_match_label, "success")
            self._pose_match_label.setText(
                f"✓ Matched a pose CSV for all {len(matched)} video(s) in {where}."
            )
            self._refresh_blocker()
            return
        names = ", ".join(v.name for v in unmatched[:4])
        more = f" (+{len(unmatched) - 4} more)" if len(unmatched) > 4 else ""
        set_text_role(self._pose_match_label, "warning")
        self._pose_match_label.setText(
            f"⚠ Matched {len(matched)} of {len(self._videos)} in {where}. "
            f"No pose CSV for: {names}{more} — those would be tracked from scratch."
        )
        self._refresh_blocker()

    def _on_choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not path:
            return
        self._output_dir = Path(path)
        self._output_label.setText(f"Output folder: {path}")
        set_path_text(self._output_value, _short_path(self._output_dir), filled=True)
        self._output_value.setToolTip(path)
        self._refresh_blocker()

    def _needs_tracking(self) -> list[Path]:
        """Videos this run would have to track, because no pose CSV covers them."""
        if not self._reuse_poses.isChecked():
            return list(self._videos)
        return self._match_poses()[1]

    def _run_blocker(self) -> str | None:
        """Why this run cannot start yet, or None.

        Neither model is unconditionally required. Without a behaviour bundle
        the run scores freezing and darting alone, which is the whole analysis
        for plenty of work and needs no classifier. Without pose weights it
        scores keypoints that are already on disk, and tracking is the only
        thing the weights were for. Each is required exactly when something
        the run must do actually depends on it.
        """
        if self._model_path is None:
            if not self._speed_group.isChecked():
                return (
                    "Choose a model bundle — or, to score freezing and darting "
                    "with no model at all, turn on the freeze/dart speed axis "
                    "and set its thresholds."
                )
            if not (self._score_freezing.isChecked() or self._score_darting.isChecked()):
                return (
                    "A run with no model bundle scores the speed axis, so tick "
                    "freezing, darting, or both."
                )
            if self._render_video.isChecked():
                return (
                    "An annotated video draws the model's labels on each frame, "
                    "so it needs a model bundle. Untick it, or choose one."
                )
        if (
            self._speed_group.isChecked()
            and self._speed_mode_value() == "cohort"
            and self._cohort_path is None
        ):
            # Otherwise the axis is simply absent from every ethogram, which
            # looks like a run that scored nothing rather than one that was
            # never told where its cut-offs live.
            return (
                "Cohort mode takes its cut-offs from a file — choose or compute "
                "one, or switch the threshold mode."
            )
        untracked = self._needs_tracking()
        if self._yolo_path is None and untracked:
            names = ", ".join(v.name for v in untracked[:4])
            more = f" (+{len(untracked) - 4} more)" if len(untracked) > 4 else ""
            return (
                f"Choose YOLO weights: {len(untracked)} video(s) have no pose CSV "
                f"to score and would have to be tracked — {names}{more}."
            )
        return None

    def _on_run(self) -> None:
        blocker = self._run_blocker()
        if blocker:
            QMessageBox.warning(self, "Apply", blocker)
            return
        if not self._videos:
            QMessageBox.warning(self, "Apply", "Add at least one video first.")
            return
        if self._output_dir is None:
            QMessageBox.warning(self, "Apply", "Choose an output folder first.")
            return
        keypoint_names = [
            name.strip() for name in self._keypoints_edit.text().split(",") if name.strip()
        ]
        if not keypoint_names:
            QMessageBox.warning(self, "Apply", "Enter at least one keypoint name.")
            return
        if not self._confirm_unmatched_poses():
            return
        if not self._confirm_scale_advisories():
            return
        blocker = self._keypoint_blocker(keypoint_names)
        if blocker:
            # Refused rather than warned: a wrong order produces an empty
            # ethogram with no error, after a full inference pass per video.
            QMessageBox.critical(self, "Keypoint names", blocker)
            return
        if not self._confirm_keypoints(keypoint_names):
            return

        self._results.clear()
        self._queue = list(self._videos)
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._rail.status.set_state("running", "Scoring")
        self._keypoint_names = keypoint_names
        self._run_next()

    def _edit_keypoint_schema(self) -> None:
        """Arrange the schema on a figure, then write its order into the field."""
        from glider.analysis.behavior.keypoint_schema import Keypoint, KeypointSchema
        from glider.gui.behavior.keypoint_editor import KeypointEditorDialog

        typed = [n.strip() for n in self._keypoints_edit.text().split(",") if n.strip()]
        if typed:
            # Seed from what is already in the field so the editor opens on the
            # user's actual schema, spread down the figure to be draggable.
            step = 1.0 / (len(typed) + 1)
            schema = KeypointSchema([Keypoint(n, 0.5, step * (i + 1)) for i, n in enumerate(typed)])
        else:
            schema = KeypointSchema.default_mouse()

        dialog = KeypointEditorDialog(schema, parent=self)
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self._keypoints_edit.setText(",".join(dialog.names()))
        finally:
            dialog.deleteLater()

    def _keypoint_blocker(self, keypoint_names) -> str | None:
        """Why this run must not start, or None.

        Only the bundle can veto: it defines which feature columns must exist,
        so names it cannot use guarantee a blank ethogram. A pose CSV that
        disagrees is reported by :meth:`_keypoint_warning` instead, because a
        CSV written by our own Batch Pose Tracking inherits whatever names were
        typed there and so is not independent evidence.
        """
        from glider.analysis.behavior.classify.features_stream import (
            expected_keypoint_order,
            keypoint_order_problem,
        )
        from glider.analysis.behavior.model import BehaviorModel

        if self._model_path is None:
            return None  # a speed-only run has no bundle to disagree with
        try:
            model = BehaviorModel.load(self._model_path)
        except Exception:
            # A bundle we cannot read must not block work; the labelled-frame
            # check still runs.
            logger.debug("could not read the bundle to check keypoints", exc_info=True)
            return None
        problem = keypoint_order_problem(model, keypoint_names)
        if problem is None:
            return None
        expected = expected_keypoint_order(model)
        return (
            f"These keypoint names cannot produce the features this model needs: "
            f"{problem}.\n\n"
            f"The model expects, in this order:\n{','.join(expected)}\n\n"
            "Running anyway would spend a full pass per video and write an "
            "ethogram with every label blank."
        )

    def _pose_csv_disagreement(self, keypoint_names) -> str | None:
        """Whether the first video's pose CSV names its parts differently.

        Looks in the chosen pose folder, not just beside the video: pointing
        the run at a folder elsewhere would otherwise silently retire this
        cross-check exactly when it is most useful, since a foreign folder is
        the likeliest source of foreign keypoint names.
        """
        from glider.analysis.behavior.classify.features_stream import pose_csv_bodyparts
        from glider.vision.pose.batch import find_pose_csv

        if not self._videos:
            return None
        csv_path = find_pose_csv(self._videos[0], self._pose_dir)
        if csv_path is None:
            return None
        bodyparts = pose_csv_bodyparts(csv_path)
        if not bodyparts or bodyparts == list(keypoint_names):
            return None
        return (
            f"{csv_path.name} names its bodyparts {','.join(bodyparts)}, which "
            "differs from the names being used. If that CSV came from a "
            "different pose project, check the labels below sit on the right "
            "body parts."
        )

    def _keypoint_warning(self, keypoint_names) -> str | None:
        """What the bundle says is wrong with these names, if anything.

        Read from the behavior model itself, which records the order it was
        trained with. Never blocks the run on its own — the labelled frame is
        the real check, and a model we cannot introspect must not stop work.
        """
        try:
            return self._pose_csv_disagreement(keypoint_names)
        except Exception:
            logger.debug("could not cross-check the pose CSV", exc_info=True)
            return None

    def _confirm_keypoints(self, keypoint_names) -> bool:
        """Show a labelled frame and require the operator to confirm it.

        Labelled from the pose CSV this video will actually be scored from,
        when there is one: those are the coordinates in question, and reading
        them costs nothing next to loading torch to re-derive them.
        """
        from glider.gui.behavior.keypoint_confirm import KeypointConfirmDialog

        pose_csv = None
        if self._reuse_poses.isChecked():
            try:
                pose_csv = find_pose_csv(self._videos[0], self._pose_dir)
            except OSError:  # an unreachable share falls back to the pose model
                pose_csv = None

        dialog = KeypointConfirmDialog(
            self._videos[0],
            self._yolo_path,
            keypoint_names,
            parent=self,
            warning=self._keypoint_warning(keypoint_names),
            pose_csv=pose_csv,
        )
        try:
            return dialog.exec() == QDialog.DialogCode.Accepted
        finally:
            dialog.deleteLater()

    def _build_cadence_row(self) -> QHBoxLayout:
        """Classifier cadence: how many tracked frames per prediction.

        The pipeline samples the model every 3 frames by default (~10 Hz on
        30 fps video), which is plenty for a live overlay but coarser than
        some analyses want. Exposed here so a per-frame ethogram is a spin
        box rather than a code change. Bout durations are corrected for the
        cadence downstream either way, so this trades runtime for temporal
        resolution and nothing else.
        """
        self._predict_every = QSpinBox()
        self._predict_every.setRange(1, 30)
        self._predict_every.setValue(_DEFAULT_PREDICT_EVERY)
        self._predict_every.setSuffix(" frame(s)")
        self._predict_every.setToolTip(
            "How often the behavior model is asked for a prediction.\n"
            f"{_DEFAULT_PREDICT_EVERY} (default) is ample for scoring bouts and keeps "
            "inference fast.\n"
            "1 classifies every frame: one ethogram row per video frame, at "
            "proportionally more classifier calls.\n"
            "Pose tracking and feature extraction run on every frame regardless."
        )
        self._cadence_hint = hint()
        self._predict_every.setFixedWidth(120)
        self._predict_every.valueChanged.connect(self._on_cadence_changed)

        inner = QHBoxLayout()
        inner.setSpacing(8)
        inner.addWidget(self._predict_every)
        inner.addWidget(self._cadence_hint, 1)
        row = labelled_row("Classify every", inner)
        self._on_cadence_changed()
        return row

    def _build_time_range_row(self) -> QHBoxLayout:
        """Score only part of each recording — a drug window, a post-stimulus
        period — instead of all of it.

        In minutes rather than frames, and resolved per video against that
        video's own rate, so one setting means the same clock time across a
        cohort filmed at different frame rates. Everything upstream of the
        prediction still runs over the whole recording, so the window's first
        frames are scored exactly as a whole-session run would have scored
        them rather than from a cold start.
        """
        self._range_on = QCheckBox("Limit to")
        self._range_on.setToolTip(
            "Score a stretch of each video instead of all of it.\n"
            "Applies to every selected video, measured from its own start."
        )
        self._range_on.toggled.connect(self._on_range_toggled)

        self._range_start = QDoubleSpinBox()
        self._range_start.setRange(0.0, 600.0)
        self._range_start.setDecimals(2)
        self._range_start.setSingleStep(0.5)
        self._range_start.setSuffix(" min")
        self._range_end = QDoubleSpinBox()
        self._range_end.setRange(0.0, 600.0)
        self._range_end.setDecimals(2)
        self._range_end.setSingleStep(0.5)
        self._range_end.setValue(5.0)
        self._range_end.setSuffix(" min")
        self._range_end.setSpecialValueText("to the end")
        for spin in (self._range_start, self._range_end):
            spin.setEnabled(False)
            spin.valueChanged.connect(self._on_range_changed)

        self._range_hint = hint("")
        for spin in (self._range_start, self._range_end):
            spin.setFixedWidth(110)

        inner = QHBoxLayout()
        inner.setSpacing(8)
        inner.addWidget(self._range_on)
        inner.addWidget(self._range_start)
        inner.addWidget(set_text_role(QLabel("to"), "muted"))
        inner.addWidget(self._range_end)
        inner.addWidget(self._range_hint, 1)
        row = labelled_row("Time window", inner)
        return row

    def _on_range_toggled(self, checked: bool) -> None:
        self._range_start.setEnabled(checked)
        self._range_end.setEnabled(checked)
        self._on_range_changed()

    def _on_range_changed(self, *_args) -> None:
        if not self._range_on.isChecked():
            set_text_role(self._range_hint, "hint")
            self._range_hint.setText("whole recording")
            return
        start, end = self._range_start.value(), self._range_end.value()
        if end and end <= start:
            set_text_role(self._range_hint, "error")
            self._range_hint.setText("⚠ the window ends before it starts")
            return
        set_text_role(self._range_hint, "hint")
        span = f"{(end - start) * 60:.0f} s of each video" if end else "to the end of each video"
        self._range_hint.setText(span)

    def _time_range(self) -> tuple[float | None, float | None]:
        """``(start_s, end_s)`` for classify(), or ``(None, None)``."""
        if not self._range_on.isChecked():
            return None, None
        start = self._range_start.value() * 60.0
        end = self._range_end.value() * 60.0
        # The end spin's minimum doubles as "to the end", so 0 means open.
        return (start or None) if start else 0.0, (end or None)

    def _build_stability_group(self) -> QWidget:
        """How much frame-to-frame flicker to absorb before reporting bouts.

        A per-frame classifier switches label far more often than an animal
        switches behavior — measured at ~100 switches/minute on a real
        cohort, with a median bout of 0.17 s. The time budget (fraction of
        session per behavior) is barely affected by that, but every
        bout-level number — counts, mean and median duration, the transition
        matrix — is dominated by it. Both knobs exist in the pipeline; until
        now neither was reachable from here.
        """
        group = QWidget()
        group.setObjectName("CardSection")
        box = QVBoxLayout(group)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        box.addWidget(set_text_role(QLabel("Label stability"), "caption"))

        self._smooth_window = QSpinBox()
        self._smooth_window.setRange(1, 61)
        self._smooth_window.setValue(_DEFAULT_SMOOTH_WINDOW)
        self._smooth_window.setSingleStep(2)
        self._smooth_window.setSuffix(" prediction(s)")
        self._smooth_window.setToolTip(
            "Majority vote over the last N predictions.\n"
            "1 = off (raw per-prediction labels).\n"
            f"{_DEFAULT_SMOOTH_WINDOW} (default) roughly halves the switch rate while moving "
            "each behavior's share of the session by well under a percentage point.\n"
            "Applies to the ethogram, bouts, stats and the annotated video alike."
        )
        box.addLayout(labelled_row("Majority vote", self._smooth_window))

        self._min_bout_s = QDoubleSpinBox()
        self._min_bout_s.setRange(0.0, 10.0)
        self._min_bout_s.setValue(0.0)
        self._min_bout_s.setSingleStep(0.1)
        self._min_bout_s.setDecimals(2)
        self._min_bout_s.setSuffix(" s")
        self._min_bout_s.setSpecialValueText("off")
        self._min_bout_s.setToolTip(
            "Discard bouts shorter than this from bouts.csv and stats.csv.\n"
            "Off by default: it changes what counts as a bout, which is a "
            "scoring decision, not a display one.\n"
            "The per-frame ethogram_raw.csv is never altered — this only "
            "filters the summaries, so the raw record stays auditable."
        )
        box.addLayout(labelled_row("Minimum bout", self._min_bout_s))
        return group

    def _on_cadence_changed(self, *_args) -> None:
        """Restate the cadence as a rate, which is what people actually reason in."""
        n = self._predict_every.value()
        hz = 30.0 / n
        every_frame = " — every frame" if n == 1 else ""
        self._cadence_hint.setText(f"≈ {hz:.1f} predictions/s on 30 fps video{every_frame}")

    def _build_speed_group(self) -> QGroupBox:
        """Optional freeze/dart axis, set in real units.

        The live detector works in pixels per frame, which means nothing
        physical and changes with camera height. Entering mm/s and converting
        through the batch calibration keeps the numbers comparable across rigs
        and sessions. Off by default — the pre-existing behaviour.
        """
        # Stays a checkable QGroupBox: its per-row show/hide (_on_speed_mode_
        # changed) is driven through QFormLayout.setRowVisible with the row
        # indices captured below, and the tests address those rows by index.
        group = QGroupBox("Score freezing and darting from speed")
        group.setCheckable(True)
        group.setChecked(False)
        group.setToolTip(
            "Adds a speed column to the ethogram and shows freezing/darting "
            "over the postural label. Needs a pixel-to-distance calibration."
        )
        group.toggled.connect(self._on_speed_group_toggled)
        form = QFormLayout(group)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._speed_form = form

        self._speed_mode = QComboBox()
        self._speed_mode.addItem("Absolute (cm/s)", "absolute")
        self._speed_mode.addItem("Percentile of this video", "percentile")
        self._speed_mode.addItem("Cohort thresholds (file)", "cohort")
        self._speed_mode.setToolTip(
            "Absolute is comparable across sessions but needs a calibration. "
            "Percentile self-adjusts to each video and needs none, but the "
            "thresholds then mean something different per recording -- which "
            "is circular in a treatment study. Cohort derives one set of "
            "cut-offs from every session at once and applies them unchanged."
        )
        self._speed_mode.currentIndexChanged.connect(self._on_speed_mode_changed)
        form.addRow("Threshold mode:", self._speed_mode)

        # Which of the two this run is about. Both by default, but a fear-
        # conditioning session scores freezing and has no darting cut-off
        # anyone could defend — and scoring one it cannot justify is worse
        # than scoring neither.
        self._score_freezing = QCheckBox("Freezing")
        self._score_freezing.setChecked(True)
        self._score_darting = QCheckBox("Darting")
        self._score_darting.setChecked(True)
        for box in (self._score_freezing, self._score_darting):
            box.setToolTip(
                "Untick one to leave it out of the ethogram entirely. Its "
                "threshold is then neither needed nor applied."
            )
            box.toggled.connect(self._on_speed_mode_changed)
        score_row = QHBoxLayout()
        score_row.addWidget(self._score_freezing)
        score_row.addWidget(self._score_darting)
        score_row.addStretch(1)
        form.addRow("Score:", score_row)

        self._freeze_cm_s = QDoubleSpinBox()
        self._freeze_cm_s.setRange(0.0, 10000.0)
        self._freeze_cm_s.setDecimals(2)
        self._freeze_cm_s.setSuffix(" cm/s")
        self._freeze_cm_s.setValue(1.0)
        form.addRow("Freezing below:", self._freeze_cm_s)
        self._freeze_abs_row = form.rowCount() - 1

        self._dart_cm_s = QDoubleSpinBox()
        self._dart_cm_s.setRange(0.0, 10000.0)
        self._dart_cm_s.setDecimals(2)
        self._dart_cm_s.setSuffix(" cm/s")
        self._dart_cm_s.setValue(15.0)
        form.addRow("Darting above:", self._dart_cm_s)
        self._dart_abs_row = form.rowCount() - 1

        row = QHBoxLayout()
        row.setSpacing(8)
        self._calibration_label = path_label("(none)")
        cal_btn = QPushButton("Choose…")
        cal_btn.clicked.connect(self._on_choose_calibration)
        row.addWidget(self._calibration_label, 1)
        row.addWidget(cal_btn)
        self._calibration_row = row
        form.addRow("Calibration file:", row)
        self._calibration_hint = hint(
            "Used for cm/s thresholds, and for the ethogram's speed_cm_s column in either mode."
        )
        # A wrapping QLabel in a QFormLayout field column reports a narrow
        # sizeHint and gets taken at its word, so this wrapped to three short
        # lines with the rest of the row empty beside it.
        self._calibration_hint.setMinimumWidth(380)
        form.addRow("", self._calibration_hint)

        # Percentiles of the video's own causal-speed distribution. Defaults
        # match the offline labeller so live and offline agree by default.
        self._freeze_pct = QDoubleSpinBox()
        self._freeze_pct.setRange(0.0, 100.0)
        self._freeze_pct.setDecimals(1)
        self._freeze_pct.setSuffix(" %")
        self._freeze_pct.setValue(10.0)
        form.addRow("Freezing percentile:", self._freeze_pct)
        self._freeze_pct_row = form.rowCount() - 1

        self._dart_pct = QDoubleSpinBox()
        self._dart_pct.setRange(0.0, 100.0)
        self._dart_pct.setDecimals(1)
        self._dart_pct.setSuffix(" %")
        self._dart_pct.setValue(99.5)
        form.addRow("Darting percentile:", self._dart_pct)
        self._dart_pct_row = form.rowCount() - 1

        cohort_row = QHBoxLayout()
        cohort_row.setSpacing(8)
        self._cohort_label = path_label("(none)")
        pick = QPushButton("Choose…")
        pick.clicked.connect(self._on_choose_cohort)
        build = QPushButton("Compute…")
        build.setToolTip(
            "Pool the speed of every pose CSV in a folder and take the cohort "
            "percentiles. Existing CSVs are used as-is, so tracking is not re-run."
        )
        build.clicked.connect(self._on_build_cohort)
        cohort_row.addWidget(self._cohort_label, 1)
        cohort_row.addWidget(pick)
        cohort_row.addWidget(build)
        form.addRow("Cohort file:", cohort_row)
        self._cohort_rows = [form.rowCount() - 1]

        # Seconds, not frames: a bout minimum is an ethological duration, and a
        # frame count means something different at 30 vs 60 fps.
        self._freeze_min_s = QDoubleSpinBox()
        self._freeze_min_s.setRange(0.0, 600.0)
        self._freeze_min_s.setDecimals(2)
        self._freeze_min_s.setSuffix(" s")
        self._freeze_min_s.setValue(1.0)
        form.addRow("Freezing lasts at least:", self._freeze_min_s)
        self._freeze_min_row = form.rowCount() - 1

        self._dart_min_s = QDoubleSpinBox()
        self._dart_min_s.setRange(0.0, 600.0)
        self._dart_min_s.setDecimals(2)
        self._dart_min_s.setSuffix(" s")
        self._dart_min_s.setValue(0.1)
        form.addRow("Darting lasts at least:", self._dart_min_s)
        self._dart_min_row = form.rowCount() - 1

        self._speed_group = group
        self._on_speed_mode_changed()
        return group

    def _on_speed_group_toggled(self, _checked: bool) -> None:
        """Keep the card badge and the Run blocker in step with the axis."""
        self._refresh_blocker()

    def _on_choose_cohort(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose cohort thresholds", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._cohort_path = Path(path)
            self._show_cohort_file()

    def _show_cohort_file(self) -> None:
        """Put what the chosen file actually says on the label.

        A path is not an answer to "which cut-offs is this run using". The
        numbers exist only inside the JSON, so choosing a file used to mean
        opening it in a text editor to find out — and the unit matters most of
        all, because a cohort pooled without a calibration is in px/frame and
        looks nothing like the cm/s the operator typed elsewhere.
        """
        from glider.analysis.behavior.cohort_speed import CohortSpeedError, CohortSpeedThresholds

        self._cohort_label.setToolTip(str(self._cohort_path))
        try:
            thresholds = CohortSpeedThresholds.load(self._cohort_path)
        except CohortSpeedError as e:
            set_path_text(
                self._cohort_label, f"{self._cohort_path.name} — unreadable: {e}", filled=True
            )
            return
        note = ""
        if not thresholds.is_calibrated:
            missing = thresholds.n_uncalibrated
            note = "  ⚠ pooled in pixels" + (
                f" — {missing} session(s) had no pixel scale" if missing else ""
            )
        set_path_text(
            self._cohort_label,
            f"{self._cohort_path.name} — {thresholds.describe()}{note}",
            filled=True,
        )
        self._refresh_blocker()

    def _on_build_cohort(self) -> None:
        """Pool every pose CSV in a folder into one set of cut-offs.

        Reads existing CSVs rather than re-tracking: a cohort that has already
        been through Batch Pose Tracking should not pay for inference twice.
        The pooling runs on a worker thread — it is minutes of work on a real
        cohort, and a UI frozen that long is indistinguishable from a crash.
        """
        folder = QFileDialog.getExistingDirectory(self, "Folder of pose CSVs")
        if not folder:
            return
        csvs = _unique_pose_csvs(Path(folder))
        if not csvs:
            QMessageBox.warning(
                self,
                "Cohort thresholds",
                f"No pose CSVs found under {folder}.\n\n"
                "Run Batch Pose Tracking first, or pick the folder its CSVs were "
                "written to.",
            )
            return

        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save cohort thresholds",
            str(Path(folder) / "cohort_speed.json"),
            "JSON Files (*.json)",
        )
        if not out:
            return

        from glider.gui.behavior.workers import CohortSpeedWorker

        window = "the whole recording"
        if self._range_on.isChecked():
            end = self._range_end.value()
            window = (
                f"minutes {self._range_start.value():g}–{end:g}"
                if end
                else f"minute {self._range_start.value():g} to the end"
            )
        self._results.append(
            f"Pooling speed from {len(csvs)} session(s) over {window}… this takes a minute or two."
        )
        self._progress.setVisible(True)
        self._progress.setRange(0, len(csvs))
        self._progress.setValue(0)
        self._run_btn.setEnabled(False)

        # Pool the same stretch the run will score. Thresholds describe the
        # behaviour they are applied to, and the two windows disagreeing is
        # not a small effect: on a real cohort, pooling whole recordings and
        # scoring minutes two to seven moved the freezing cut-off by a third,
        # because the settling-in period the ethogram never covers is where
        # the stillest frames are.
        start_s, end_s = self._time_range()

        self._cohort_thread = QThread()
        self._cohort_worker = CohortSpeedWorker(
            csvs,
            out,
            freeze_pct=self._freeze_pct.value(),
            dart_pct=self._dart_pct.value(),
            calibration_master=self._calibration_master,
            start_s=start_s,
            end_s=end_s,
        )
        self._cohort_worker.moveToThread(self._cohort_thread)
        self._cohort_thread.started.connect(self._cohort_worker.run)
        self._cohort_worker.progress.connect(lambda done, _total: self._progress.setValue(done))
        self._cohort_worker.finished.connect(
            lambda thresholds, path=out: self._on_cohort_done(thresholds, path)
        )
        self._cohort_worker.failed.connect(self._on_cohort_failed)
        # The thread retires itself rather than being parented to this widget:
        # closing the tab mid-pool would otherwise destroy a running QThread.
        self._cohort_worker.finished.connect(self._cohort_thread.quit)
        self._cohort_worker.failed.connect(self._cohort_thread.quit)
        self._cohort_thread.finished.connect(self._cohort_worker.deleteLater)
        self._cohort_thread.finished.connect(self._cohort_thread.deleteLater)
        self._cohort_thread.start()

    def _end_cohort_run(self) -> None:
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)

    def _on_cohort_done(self, thresholds, path) -> None:
        self._end_cohort_run()
        self._cohort_path = Path(path)
        self._show_cohort_file()
        missing = getattr(thresholds, "n_uncalibrated", 0)
        note = (
            ""
            if thresholds.is_calibrated
            else f"\n\n{missing or 'Some'} of the sessions had no pixel scale, so the "
            "WHOLE pool fell back to px/frame — mixing units would be meaningless. "
            "That is only valid if every video shares one rig geometry. To get "
            "cm/s, choose a calibration file that covers every session."
        )
        summary = (
            f"Pooled {thresholds.n_samples:,} samples from "
            f"{thresholds.n_sessions} session(s).\n\n"
            f"freezing below {thresholds.freeze:.3f} {thresholds.unit}\n"
            f"darting above {thresholds.dart:.3f} {thresholds.unit}{note}"
        )
        self._results.append(summary.replace("\n", " "))
        QMessageBox.information(self, "Cohort thresholds", summary)

    def _on_cohort_failed(self, message: str) -> None:
        self._end_cohort_run()
        self._results.append(f"Cohort thresholds failed: {message}")
        QMessageBox.critical(self, "Cohort thresholds", message)

    def _speed_mode_value(self) -> str:
        return self._speed_mode.currentData() or "absolute"

    def _on_speed_mode_changed(self, *_args) -> None:
        """Show only the fields the chosen mode and the ticked halves use.

        A threshold for a behaviour this run is not scoring is not a setting
        with a bad value — it is a setting with no meaning, and leaving it on
        screen invites someone to tune it and wonder why nothing changed.
        """
        mode = self._speed_mode_value()
        freezing = self._score_freezing.isChecked()
        darting = self._score_darting.isChecked()
        for row, visible in (
            (self._freeze_abs_row, mode == "absolute" and freezing),
            (self._dart_abs_row, mode == "absolute" and darting),
            (self._freeze_pct_row, mode == "percentile" and freezing),
            (self._dart_pct_row, mode == "percentile" and darting),
            (self._freeze_min_row, freezing),
            (self._dart_min_row, darting),
        ):
            self._speed_form.setRowVisible(row, visible)
        for row in self._cohort_rows:
            self._speed_form.setRowVisible(row, mode == "cohort")
        # The calibration rows stay visible in every mode: percentile
        # thresholds do not need a scale, but speed_cm_s does.

    def _on_choose_calibration(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose master calibration file",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if path:
            self._calibration_master = Path(path)
            set_path_text(
                self._calibration_label, _short_path(self._calibration_master), filled=True
            )
            self._calibration_label.setToolTip(path)

    def _speed_opts(self) -> dict:
        """Speed-axis kwargs for ApplyWorker, or {} when the axis is off."""
        if not self._speed_group.isChecked():
            return {}
        freezing = self._score_freezing.isChecked()
        darting = self._score_darting.isChecked()
        if not (freezing or darting):
            return {}
        mode = self._speed_mode_value()
        if mode == "cohort":
            opts = {"cohort_thresholds": self._cohort_path}
        elif mode == "percentile":
            # A side that is not being scored sends no threshold at all, so
            # nothing downstream has a stale number it could apply by mistake.
            opts = {
                "freeze_pct": self._freeze_pct.value() if freezing else None,
                "dart_pct": self._dart_pct.value() if darting else None,
            }
        else:
            opts = {
                "freeze_cm_s": self._freeze_cm_s.value() if freezing else None,
                "dart_cm_s": self._dart_cm_s.value() if darting else None,
            }
        opts["score_freezing"] = freezing
        opts["score_darting"] = darting
        # Sent in every mode. Percentile thresholds need no scale, but the
        # ethogram's speed_cm_s column does, and wanting real units in the
        # output is independent of how the cut-offs were chosen.
        opts["calibration_master"] = self._calibration_master
        if freezing:
            opts["freeze_min_s"] = self._freeze_min_s.value()
        if darting:
            opts["dart_min_s"] = self._dart_min_s.value()
        return opts

    def _run_next(self) -> None:
        if not self._queue:
            self._progress.setVisible(False)
            self._run_btn.setEnabled(True)
            if self._rail.status.property("state") == "running":
                self._rail.status.set_state("ok", "Done")
            return

        from glider.gui.behavior.workers import ApplyWorker

        video = self._queue.pop(0)
        remaining = len(self._videos) - len(self._queue)
        self._rail.status.set_state("running", f"{remaining} / {len(self._videos)}")
        video_output_dir = self._output_dir / video.stem
        start_s, end_s = self._time_range()

        self._apply_thread = QThread()
        self._apply_worker = ApplyWorker(
            video=video,
            model_path=self._model_path,
            yolo_path=self._yolo_path,
            keypoint_names=self._keypoint_names,
            output_dir=video_output_dir,
            speed_opts=self._speed_opts(),
            predict_every=self._predict_every.value(),
            reuse_existing_poses=self._reuse_poses.isChecked(),
            pose_dir=self._pose_dir if self._reuse_poses.isChecked() else None,
            write_annotated=self._render_video.isChecked(),
            smooth_window=self._smooth_window.value(),
            min_bout_s=self._min_bout_s.value() or None,
            start_s=start_s,
            end_s=end_s,
        )
        self._apply_worker.moveToThread(self._apply_thread)
        self._apply_thread.started.connect(self._apply_worker.run)
        self._apply_worker.progress.connect(self._on_apply_progress)
        self._apply_worker.finished.connect(
            lambda result, v=video, out=video_output_dir: self._on_apply_finished(result, v, out)
        )
        self._apply_worker.failed.connect(lambda msg, v=video: self._on_apply_failed(msg, v))

        self._progress.setRange(0, 0)  # indeterminate until this video finishes
        self._apply_thread.start()

    def _on_apply_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)

    def _on_apply_finished(self, result: object, video: Path, output_dir: Path) -> None:
        self._teardown_apply_thread()
        lines = [f"{video.name}:"]
        n_frames = getattr(getattr(result, "ethogram", None), "__len__", lambda: None)()
        if n_frames is not None:
            lines.append(f"  frames classified: {n_frames}")
        for label, filename in (
            ("annotated video", "annotated.mp4"),
            ("ethogram (raw)", "ethogram_raw.csv"),
            ("bouts", "bouts.csv"),
            ("stats", "stats.csv"),
            ("transitions", "transitions.csv"),
        ):
            out_path = output_dir / filename
            if out_path.exists():
                lines.append(f"  {label}: {out_path}")
        self._results.append("\n".join(lines))
        self._run_next()

    def _on_apply_failed(self, message: str, video: Path) -> None:
        self._teardown_apply_thread()
        self._rail.status.set_state("error", "Failed")
        self._results.append(f"{video.name}: FAILED - {message}")
        QMessageBox.critical(self, "Apply failed", f"{video.name}:\n{message}")
        self._run_next()

    def _teardown_apply_thread(self) -> None:
        if self._apply_thread is not None:
            self._apply_thread.quit()
            self._apply_thread.wait(5000)
            self._apply_thread.deleteLater()
            self._apply_thread = None
        if self._apply_worker is not None:
            self._apply_worker.deleteLater()
            self._apply_worker = None


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _short_path(path: Path, keep: int = 3) -> str:
    """The tail of a path, enough to recognise it without filling a row.

    Lab paths are UNC shares nested several folders deep; the last few
    components are what tells one cohort from another.
    """
    parts = Path(path).parts
    return str(path) if len(parts) <= keep else "…" + "\\".join(parts[-keep:])


def _unique_pose_csvs(folder: Path) -> list[Path]:
    """Pose CSVs under *folder*, one per session, shallowest first.

    A recursive scan of a working folder finds the same session more than
    once: an apply run copies or writes poses into its own output subfolder,
    so ``videos/t1_d2DLC_exp-7.csv`` and ``videos/output/t1_d2/...`` are the
    same animal. Pooling both weights that session twice in the cohort
    percentiles, and the copies usually sit where their video cannot be found,
    which costs the whole pool its pixel scale — cm/s cut-offs silently become
    px/frame ones because of a duplicate.

    Keyed on the file name, and the shallowest copy wins, because that is the
    one that lives beside its video.
    """
    seen: dict[str, Path] = {}
    for path in sorted(folder.rglob("*.csv"), key=lambda p: (len(p.relative_to(folder).parts), p)):
        if "DLC_" in path.stem and path.name not in seen:
            seen[path.name] = path
    return sorted(seen.values())


def _row(*widgets: QWidget) -> QHBoxLayout:
    """A QHBoxLayout with the first widget stretched and the rest packed tight."""
    row = QHBoxLayout()
    for i, w in enumerate(widgets):
        row.addWidget(w, 1 if i == 0 else 0)
    return row


def _button_row(*buttons: QWidget) -> QHBoxLayout:
    """Buttons packed left, so a two-button pair does not stretch to the width."""
    row = QHBoxLayout()
    row.setSpacing(8)
    for button in buttons:
        row.addWidget(button)
    row.addStretch(1)
    return row


def _remove_selected(list_widget: QListWidget, backing: list) -> None:
    """Remove every selected row from a QListWidget and its backing list in step.

    Rows are taken bottom-up: deleting from the front would shift the indices
    of everything still to be removed, so a top-down pass silently deletes the
    wrong entries as soon as more than one row is selected.
    """
    rows = sorted((index.row() for index in list_widget.selectedIndexes()), reverse=True)
    for row in rows:
        if 0 <= row < len(backing):
            list_widget.takeItem(row)
            del backing[row]


def _annotations_beside(pose_csv: Path) -> Path:
    """Where training expects this pose CSV's annotations to live.

    Same rule as the annotator's ``annotation_path_for`` and the training
    pipeline, so a session added here is one training can actually read.
    """
    return pose_csv.parent / f"{pose_csv.stem}_annotations.csv"


def _pick_session_pairs(parent: QWidget, kind: str) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Prompt for pose CSVs and pair each with its annotations.

    Returns ``(pairs, skipped)``. One multi-select dialog: pairing a whole
    cohort by hand cost two dialogs per session, which is sixty for thirty
    animals.

    Selecting exactly one pose CSV with no annotations beside it falls back to
    asking for the annotations explicitly, so a layout that keeps them
    somewhere unrelated is still reachable. That fallback is deliberately not
    offered for a multi-selection: answering it once would say nothing about
    the other files, and asking N times is the tax this replaces.
    """
    chosen, _ = QFileDialog.getOpenFileNames(
        parent, f"Choose {kind} pose CSVs", "", "CSV files (*.csv);;All files (*)"
    )
    if not chosen:
        return [], []

    poses = [Path(p) for p in chosen]
    pairs: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    for pose in poses:
        # These folders hold pose and annotation CSVs side by side, and the
        # dialog cannot tell them apart. Taking one as a pose CSV would look
        # for "<stem>_annotations_annotations.csv" and report it missing,
        # which is a confusing way to say "you picked the wrong file".
        if pose.stem.endswith("_annotations"):
            skipped.append(f"{pose.name} (that is an annotations file, not pose data)")
            continue
        ann_path = _annotations_beside(pose)
        if ann_path.exists():
            pairs.append((pose, ann_path))
            continue
        if len(poses) == 1:
            picked, _ = QFileDialog.getOpenFileName(
                parent,
                f"Choose {kind} annotations CSV for {pose.name}",
                "",
                "CSV files (*.csv);;All files (*)",
            )
            if picked:
                pairs.append((pose, Path(picked)))
                continue
            return [], []
        skipped.append(f"{pose.name} (no {ann_path.name} beside it)")
    return pairs, skipped
