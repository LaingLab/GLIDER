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
import pprint
from pathlib import Path
from typing import NamedTuple

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
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

    def __init__(self, project_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Analysis")
        self.resize(1000, 700)

        if project_dir is None:
            from glider.core.config import get_config

            project_dir = get_config().paths.behavior_projects_dir
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)

        self.tabs = QTabWidget()
        self.tabs.addTab(AnnotateTab(self.project_dir), "Annotate")
        self.tabs.addTab(TrainTab(), "Train")
        self.tabs.addTab(ApplyTab(), "Apply")
        self.setCentralWidget(self.tabs)


class AnnotateTab(QWidget):
    """Pick a videos folder + pose-CSV folder and launch the clip annotator."""

    def __init__(self, project_dir: Path, parent=None):
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self._videos_dir: Path | None = None
        self._poses_dir: Path | None = None
        # Keep a reference so the launched AnnotatorWindow survives GC.
        self._annotator_window = None

        layout = QVBoxLayout(self)

        self._videos_label = QLabel("Videos folder: (none)")
        videos_btn = QPushButton("Choose videos folder...")
        videos_btn.clicked.connect(self._on_choose_videos)
        layout.addLayout(_row(self._videos_label, videos_btn))

        self._poses_label = QLabel("Pose CSV folder: (defaults to videos folder)")
        poses_btn = QPushButton("Choose pose CSV folder...")
        poses_btn.clicked.connect(self._on_choose_poses)
        layout.addLayout(_row(self._poses_label, poses_btn))

        self._launch_btn = QPushButton("Launch annotator")
        self._launch_btn.clicked.connect(self._on_launch)
        layout.addWidget(self._launch_btn)
        layout.addStretch(1)

    def _on_choose_videos(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose videos folder")
        if not path:
            return
        self._videos_dir = Path(path)
        self._videos_label.setText(f"Videos folder: {path}")

    def _on_choose_poses(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose pose CSV folder")
        if not path:
            return
        self._poses_dir = Path(path)
        self._poses_label.setText(f"Pose CSV folder: {path}")

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
        from glider.analysis.behavior.vocabulary import Vocabulary
        from glider.gui.behavior.annotator.app import annotation_path_for
        from glider.gui.behavior.annotator.capture_cache import VideoCaptureCache
        from glider.gui.behavior.annotator.main_window import AnnotatorWindow
        from glider.gui.behavior.annotator.sampler import propose_clips_multi
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
            clips = propose_clips_multi(sessions=pairs, n_clips_total=max(50, len(pairs)), fps=fps)
        except Exception as e:  # noqa: BLE001 - surfaced to the user, not fatal
            QMessageBox.critical(self, "Annotate", f"Could not sample clips:\n{e}")
            return

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

        layout = QVBoxLayout(self)

        sessions_group = QGroupBox("Training sessions (pose CSV + annotations CSV)")
        sessions_layout = QVBoxLayout(sessions_group)
        self._sessions_list = QListWidget()
        sessions_layout.addWidget(self._sessions_list)
        add_btn = QPushButton("Add session...")
        add_btn.clicked.connect(self._on_add_session)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(lambda: _remove_selected(self._sessions_list, self._sessions))
        sessions_layout.addLayout(_row(add_btn, remove_btn))
        layout.addWidget(sessions_group)

        holdout_group = QGroupBox("Holdout sessions (optional cross-session test set)")
        holdout_layout = QVBoxLayout(holdout_group)
        self._holdout_list = QListWidget()
        holdout_layout.addWidget(self._holdout_list)
        add_holdout_btn = QPushButton("Add holdout session...")
        add_holdout_btn.clicked.connect(self._on_add_holdout)
        remove_holdout_btn = QPushButton("Remove selected")
        remove_holdout_btn.clicked.connect(
            lambda: _remove_selected(self._holdout_list, self._holdout)
        )
        holdout_layout.addLayout(_row(add_holdout_btn, remove_holdout_btn))
        layout.addWidget(holdout_group)

        options_row = QHBoxLayout()
        options_row.addWidget(QLabel("Classifier:"))
        self._classifier_combo = QComboBox()
        # train_model(classifier_type=...) accepts exactly "rf"
        # (RandomForestClassifier) or "lightgbm" (LGBMClassifier, and the
        # library-side default) — see
        # glider.analysis.behavior.pipeline.train_model docstring.
        self._classifier_combo.addItems(["rf", "lightgbm"])
        self._classifier_combo.currentTextChanged.connect(self._on_classifier_changed)
        options_row.addWidget(self._classifier_combo)
        self._advanced_btn = QPushButton("Advanced...")
        self._advanced_btn.clicked.connect(self._on_advanced)
        options_row.addWidget(self._advanced_btn)
        self._background_check = QCheckBox("Include background class")
        self._mirror_check = QCheckBox("Mirror augment")
        options_row.addWidget(self._background_check)
        options_row.addWidget(self._mirror_check)
        options_row.addStretch(1)
        layout.addLayout(options_row)
        self._on_classifier_changed()

        self._output_label = QLabel("Model output: (none)")
        output_btn = QPushButton("Choose output file...")
        output_btn.clicked.connect(self._on_choose_output)
        layout.addLayout(_row(self._output_label, output_btn))

        self._fit_btn = QPushButton("Fit")
        self._fit_btn.clicked.connect(self._on_fit)
        layout.addWidget(self._fit_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Training results will appear here.")
        layout.addWidget(self._results, 1)

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
        pair = _pick_session_pair(self, "training")
        if pair is not None:
            self._sessions.append(pair)
            self._sessions_list.addItem(f"{pair[0]}  |  {pair[1]}")

    def _on_add_holdout(self) -> None:
        pair = _pick_session_pair(self, "holdout")
        if pair is not None:
            self._holdout.append(pair)
            self._holdout_list.addItem(f"{pair[0]}  |  {pair[1]}")

    def _on_choose_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model bundle", "", "Model files (*.pkl);;All files (*)"
        )
        if not path:
            return
        self._output_path = Path(path)
        self._output_label.setText(f"Model output: {path}")

    def _on_fit(self) -> None:
        if not self._sessions:
            QMessageBox.warning(self, "Train", "Add at least one training session first.")
            return
        if self._output_path is None:
            QMessageBox.warning(self, "Train", "Choose a model output file first.")
            return

        from glider.gui.behavior.workers import TrainWorker

        options: dict = {
            "classifier_type": self._classifier_combo.currentText(),
            "include_background": self._background_check.isChecked(),
            "mirror_augment": self._mirror_check.isChecked(),
        }
        options.update(self._lgbm_options())
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
        self._train_thread.start()

    def _on_train_progress(self, done: int, total: int) -> None:
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(done)

    def _on_train_finished(self, summary: object) -> None:
        self._teardown_train_thread()
        self._progress.setVisible(False)
        self._fit_btn.setEnabled(True)
        self._results.setPlainText(pprint.pformat(summary))

    def _on_train_failed(self, message: str) -> None:
        self._teardown_train_thread()
        self._progress.setVisible(False)
        self._fit_btn.setEnabled(True)
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

        layout = QVBoxLayout(self)

        self._model_label = QLabel("Model bundle: (none)")
        model_btn = QPushButton("Choose model bundle...")
        model_btn.clicked.connect(self._on_choose_model)
        layout.addLayout(_row(self._model_label, model_btn))

        self._yolo_label = QLabel("YOLO weights: (none)")
        yolo_btn = QPushButton("Choose YOLO weights...")
        yolo_btn.clicked.connect(self._on_choose_yolo)
        layout.addLayout(_row(self._yolo_label, yolo_btn))

        videos_group = QGroupBox("Video(s) to classify")
        videos_layout = QVBoxLayout(videos_group)
        self._videos_list = QListWidget()
        videos_layout.addWidget(self._videos_list)
        add_videos_btn = QPushButton("Add video(s)...")
        add_videos_btn.clicked.connect(self._on_add_videos)
        remove_video_btn = QPushButton("Remove selected")
        remove_video_btn.clicked.connect(self._on_remove_video)
        videos_layout.addLayout(_row(add_videos_btn, remove_video_btn))
        layout.addWidget(videos_group)

        self._keypoints_edit = QLineEdit()
        self._keypoints_edit.setPlaceholderText("nose, left_ear, right_ear, ... (comma-separated)")
        keypoints_row = QHBoxLayout()
        keypoints_row.addWidget(QLabel("Keypoint names:"))
        keypoints_row.addWidget(self._keypoints_edit, 1)
        edit_kp = QPushButton("Edit…")
        edit_kp.setToolTip(
            "Arrange the keypoints on a figure instead of typing the order, "
            "and save the layout for reuse."
        )
        edit_kp.clicked.connect(self._edit_keypoint_schema)
        keypoints_row.addWidget(edit_kp)
        layout.addLayout(keypoints_row)

        layout.addLayout(self._build_cadence_row())
        layout.addLayout(self._build_time_range_row())
        layout.addWidget(self._build_stability_group())

        layout.addWidget(self._build_speed_group())

        # Encoding an annotated MP4 costs more wall-clock than the inference
        # itself on a long recording, and it is a spot-checking aid rather than
        # an analysis artifact -- so it is off unless asked for.
        # Checked by default: Batch Pose Tracking has usually already produced
        # these, and re-deriving them is the biggest avoidable cost in a run.
        self._reuse_poses = QCheckBox("Reuse already-tracked pose CSVs (skips tracking)")
        self._reuse_poses.setChecked(True)
        self._reuse_poses.setToolTip(
            "Reads the poses Batch Pose Tracking already wrote instead of "
            "running the pose model again — by far the biggest cost in a run. "
            "Falls back to tracking for any video without one."
        )
        self._reuse_poses.toggled.connect(self._on_reuse_toggled)
        layout.addWidget(self._reuse_poses)

        # Batch Pose Tracking writes its CSVs wherever it was pointed, which
        # is routinely a poses folder rather than beside the videos. Matching
        # by name from one folder beats copying a CSV per video by hand.
        self._pose_dir: Path | None = None
        self._pose_dir_label = QLabel("Pose CSV folder: beside each video")
        # Not wrapped: a UNC share path is long enough to wrap to four lines
        # and shove the buttons around. The tail identifies the folder; the
        # tooltip carries the whole thing.
        self._pose_dir_label.setWordWrap(False)
        self._pose_dir_btn = QPushButton("Choose pose CSV folder…")
        self._pose_dir_btn.clicked.connect(self._on_choose_pose_dir)
        self._pose_dir_clear = QPushButton("Use video folder")
        self._pose_dir_clear.clicked.connect(self._on_clear_pose_dir)
        pose_row = _row(self._pose_dir_label, self._pose_dir_btn)
        pose_row.addWidget(self._pose_dir_clear)
        layout.addLayout(pose_row)
        self._pose_match_label = QLabel("")
        self._pose_match_label.setWordWrap(True)
        layout.addWidget(self._pose_match_label)

        self._render_video = QCheckBox("Also render an annotated video (slow)")
        self._render_video.setChecked(False)
        self._render_video.setToolTip(
            "Writes annotated.mp4 beside the ethogram. Useful for checking a "
            "single video; a large cost per video across a batch."
        )
        layout.addWidget(self._render_video)

        self._output_label = QLabel("Output folder: (none)")
        output_btn = QPushButton("Choose output folder...")
        output_btn.clicked.connect(self._on_choose_output_dir)
        layout.addLayout(_row(self._output_label, output_btn))

        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._results = QTextEdit()
        self._results.setReadOnly(True)
        self._results.setPlaceholderText("Output paths will appear here.")
        layout.addWidget(self._results, 1)

    def _on_choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose model bundle", "", "Model files (*.pkl);;All files (*)"
        )
        if not path:
            return
        self._model_path = Path(path)
        self._model_label.setText(f"Model bundle: {path}")
        self._autofill_keypoints()

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

    def _on_add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose video(s)", "", _VIDEO_FILTER)
        for path in paths:
            self._videos.append(Path(path))
            self._videos_list.addItem(path)
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
        self._refresh_pose_match()

    def _on_clear_pose_dir(self) -> None:
        self._pose_dir = None
        self._pose_dir_label.setText("Pose CSV folder: beside each video")
        self._pose_dir_label.setToolTip("")
        self._refresh_pose_match()

    def _on_reuse_toggled(self, checked: bool) -> None:
        for w in (self._pose_dir_label, self._pose_dir_btn, self._pose_dir_clear):
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
            self._pose_match_label.setText("Tracking will run for every video.")
            return
        if not self._videos:
            self._pose_match_label.setText("")
            return
        matched, unmatched = self._match_poses()
        where = _short_path(self._pose_dir) if self._pose_dir else "each video's own folder"
        self._pose_match_label.setToolTip(str(self._pose_dir) if self._pose_dir else "")
        if not unmatched:
            self._pose_match_label.setText(
                f"Matched a pose CSV for all {len(matched)} video(s) in {where}."
            )
            return
        names = ", ".join(v.name for v in unmatched[:4])
        more = f" (+{len(unmatched) - 4} more)" if len(unmatched) > 4 else ""
        self._pose_match_label.setText(
            f"Matched {len(matched)} of {len(self._videos)} in {where}. "
            f"No pose CSV for: {names}{more} — those would be tracked from scratch."
        )

    def _on_choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not path:
            return
        self._output_dir = Path(path)
        self._output_label.setText(f"Output folder: {path}")

    def _on_run(self) -> None:
        if self._model_path is None:
            QMessageBox.warning(self, "Apply", "Choose a model bundle first.")
            return
        if self._yolo_path is None:
            QMessageBox.warning(self, "Apply", "Choose YOLO weights first.")
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
        """Show a labelled frame and require the operator to confirm it."""
        from glider.gui.behavior.keypoint_confirm import KeypointConfirmDialog

        dialog = KeypointConfirmDialog(
            self._videos[0],
            self._yolo_path,
            keypoint_names,
            parent=self,
            warning=self._keypoint_warning(keypoint_names),
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
        self._cadence_hint = QLabel()
        self._predict_every.valueChanged.connect(self._on_cadence_changed)

        row = QHBoxLayout()
        row.addWidget(QLabel("Classify every:"))
        row.addWidget(self._predict_every)
        row.addWidget(self._cadence_hint, 1)
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
        self._range_on = QCheckBox("Analyse only")
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

        self._range_hint = QLabel("")
        self._range_hint.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(self._range_on)
        row.addWidget(self._range_start)
        row.addWidget(QLabel("to"))
        row.addWidget(self._range_end)
        row.addWidget(self._range_hint, 1)
        return row

    def _on_range_toggled(self, checked: bool) -> None:
        self._range_start.setEnabled(checked)
        self._range_end.setEnabled(checked)
        self._on_range_changed()

    def _on_range_changed(self, *_args) -> None:
        if not self._range_on.isChecked():
            self._range_hint.setText("whole recording")
            return
        start, end = self._range_start.value(), self._range_end.value()
        if end and end <= start:
            self._range_hint.setText("⚠ the window ends before it starts")
            return
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

    def _build_stability_group(self) -> QGroupBox:
        """How much frame-to-frame flicker to absorb before reporting bouts.

        A per-frame classifier switches label far more often than an animal
        switches behavior — measured at ~100 switches/minute on a real
        cohort, with a median bout of 0.17 s. The time budget (fraction of
        session per behavior) is barely affected by that, but every
        bout-level number — counts, mean and median duration, the transition
        matrix — is dominated by it. Both knobs exist in the pipeline; until
        now neither was reachable from here.
        """
        group = QGroupBox("Label stability")
        form = QFormLayout(group)

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
        form.addRow("Majority-vote smoothing:", self._smooth_window)

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
        form.addRow("Minimum bout duration:", self._min_bout_s)
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
        group = QGroupBox("Freeze / dart speed axis (optional)")
        group.setCheckable(True)
        group.setChecked(False)
        group.setToolTip(
            "Adds a speed column to the ethogram and shows freezing/darting "
            "over the postural label. Needs a pixel-to-distance calibration."
        )
        form = QFormLayout(group)
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

        self._freeze_cm_s = QDoubleSpinBox()
        self._freeze_cm_s.setRange(0.0, 10000.0)
        self._freeze_cm_s.setDecimals(2)
        self._freeze_cm_s.setSuffix(" cm/s")
        self._freeze_cm_s.setValue(1.0)
        form.addRow("Freezing below:", self._freeze_cm_s)
        self._absolute_rows = [form.rowCount() - 1]

        self._dart_cm_s = QDoubleSpinBox()
        self._dart_cm_s.setRange(0.0, 10000.0)
        self._dart_cm_s.setDecimals(2)
        self._dart_cm_s.setSuffix(" cm/s")
        self._dart_cm_s.setValue(15.0)
        form.addRow("Darting above:", self._dart_cm_s)
        self._absolute_rows.append(form.rowCount() - 1)

        row = QHBoxLayout()
        self._calibration_label = QLabel("(none)")
        self._calibration_label.setWordWrap(True)
        cal_btn = QPushButton("Choose...")
        cal_btn.clicked.connect(self._on_choose_calibration)
        row.addWidget(self._calibration_label, 1)
        row.addWidget(cal_btn)
        self._calibration_row = row
        form.addRow("Calibration file:", row)
        self._calibration_hint = QLabel(
            "Used for cm/s thresholds, and for the ethogram's speed_cm_s " "column in either mode."
        )
        self._calibration_hint.setWordWrap(True)
        form.addRow("", self._calibration_hint)

        # Percentiles of the video's own causal-speed distribution. Defaults
        # match the offline labeller so live and offline agree by default.
        self._freeze_pct = QDoubleSpinBox()
        self._freeze_pct.setRange(0.0, 100.0)
        self._freeze_pct.setDecimals(1)
        self._freeze_pct.setSuffix(" %")
        self._freeze_pct.setValue(10.0)
        form.addRow("Freezing percentile:", self._freeze_pct)
        self._percentile_rows = [form.rowCount() - 1]

        self._dart_pct = QDoubleSpinBox()
        self._dart_pct.setRange(0.0, 100.0)
        self._dart_pct.setDecimals(1)
        self._dart_pct.setSuffix(" %")
        self._dart_pct.setValue(99.5)
        form.addRow("Darting percentile:", self._dart_pct)
        self._percentile_rows.append(form.rowCount() - 1)

        cohort_row = QHBoxLayout()
        self._cohort_label = QLabel("(none)")
        self._cohort_label.setWordWrap(True)
        pick = QPushButton("Choose...")
        pick.clicked.connect(self._on_choose_cohort)
        build = QPushButton("Compute...")
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

        self._dart_min_s = QDoubleSpinBox()
        self._dart_min_s.setRange(0.0, 600.0)
        self._dart_min_s.setDecimals(2)
        self._dart_min_s.setSuffix(" s")
        self._dart_min_s.setValue(0.1)
        form.addRow("Darting lasts at least:", self._dart_min_s)

        self._speed_group = group
        self._on_speed_mode_changed()
        return group

    def _on_choose_cohort(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose cohort thresholds", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._cohort_path = Path(path)
            self._cohort_label.setText(str(self._cohort_path))

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
        csvs = sorted(p for p in Path(folder).rglob("*.csv") if "DLC_" in p.stem)
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
            f"Pooling speed from {len(csvs)} session(s) over {window}… "
            "this takes a minute or two."
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
        self._cohort_label.setText(str(self._cohort_path))
        note = (
            ""
            if thresholds.is_calibrated
            else "\n\nNo calibration covered these sessions, so the thresholds are "
            "in px/frame. That is only valid if every video shares one rig geometry."
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
        """Show only the fields the chosen mode actually uses."""
        absolute = self._speed_mode_value() == "absolute"
        for row in self._absolute_rows:
            self._speed_form.setRowVisible(row, absolute)
        mode = self._speed_mode_value()
        for row in self._percentile_rows:
            self._speed_form.setRowVisible(row, mode == "percentile")
        for row in self._cohort_rows:
            self._speed_form.setRowVisible(row, mode == "cohort")
        # The calibration rows stay visible in both modes: percentile
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
            self._calibration_label.setText(str(self._calibration_master))

    def _speed_opts(self) -> dict:
        """Speed-axis kwargs for ApplyWorker, or {} when the axis is off."""
        if not self._speed_group.isChecked():
            return {}
        mode = self._speed_mode_value()
        if mode == "cohort":
            opts = {"cohort_thresholds": self._cohort_path}
        elif mode == "percentile":
            opts = {
                "freeze_pct": self._freeze_pct.value(),
                "dart_pct": self._dart_pct.value(),
            }
        else:
            opts = {
                "freeze_cm_s": self._freeze_cm_s.value(),
                "dart_cm_s": self._dart_cm_s.value(),
            }
        # Sent in BOTH modes. Percentile thresholds need no scale, but the
        # ethogram's speed_cm_s column does, and wanting real units in the
        # output is independent of how the cut-offs were chosen.
        opts["calibration_master"] = self._calibration_master
        opts["freeze_min_s"] = self._freeze_min_s.value()
        opts["dart_min_s"] = self._dart_min_s.value()
        return opts

    def _run_next(self) -> None:
        if not self._queue:
            self._progress.setVisible(False)
            self._run_btn.setEnabled(True)
            return

        from glider.gui.behavior.workers import ApplyWorker

        video = self._queue.pop(0)
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


def _row(*widgets: QWidget) -> QHBoxLayout:
    """A QHBoxLayout with the first widget stretched and the rest packed tight."""
    row = QHBoxLayout()
    for i, w in enumerate(widgets):
        row.addWidget(w, 1 if i == 0 else 0)
    return row


def _remove_selected(list_widget: QListWidget, backing: list) -> None:
    """Remove the current row from a QListWidget and its backing list in step."""
    row = list_widget.currentRow()
    if row < 0:
        return
    list_widget.takeItem(row)
    del backing[row]


def _pick_session_pair(parent: QWidget, kind: str) -> tuple[Path, Path] | None:
    """Prompt for a (pose CSV, annotations CSV) pair, or None if cancelled."""
    pose_path, _ = QFileDialog.getOpenFileName(
        parent, f"Choose {kind} pose CSV", "", "CSV files (*.csv);;All files (*)"
    )
    if not pose_path:
        return None
    ann_path, _ = QFileDialog.getOpenFileName(
        parent, f"Choose {kind} annotations CSV", "", "CSV files (*.csv);;All files (*)"
    )
    if not ann_path:
        return None
    return (Path(pose_path), Path(ann_path))
