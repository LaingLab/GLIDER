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
        remove_video_btn.clicked.connect(lambda: _remove_selected(self._videos_list, self._videos))
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

        layout.addWidget(self._build_speed_group())

        # Encoding an annotated MP4 costs more wall-clock than the inference
        # itself on a long recording, and it is a spot-checking aid rather than
        # an analysis artifact -- so it is off unless asked for.
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

    def _keypoint_warning(self, keypoint_names) -> str | None:
        """What the bundle says is wrong with these names, if anything.

        Read from the behavior model itself, which records the order it was
        trained with. Never blocks the run on its own — the labelled frame is
        the real check, and a model we cannot introspect must not stop work.
        """
        try:
            from glider.analysis.behavior.classify.features_stream import (
                expected_keypoint_order,
                keypoint_order_problem,
            )
            from glider.analysis.behavior.model import BehaviorModel

            model = BehaviorModel.load(self._model_path)
            problem = keypoint_order_problem(model, keypoint_names)
            if problem is None:
                return None
            expected = expected_keypoint_order(model)
            return f"{problem}. This model expects: {','.join(expected)}"
        except Exception:
            logger.debug("could not check keypoint order against the bundle", exc_info=True)
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
        self._speed_mode.setToolTip(
            "Absolute is comparable across sessions but needs a calibration. "
            "Percentile self-adjusts to each video and needs none, but the "
            "thresholds then mean something different per recording."
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

    def _speed_mode_value(self) -> str:
        return self._speed_mode.currentData() or "absolute"

    def _on_speed_mode_changed(self, *_args) -> None:
        """Show only the fields the chosen mode actually uses."""
        absolute = self._speed_mode_value() == "absolute"
        for row in self._absolute_rows:
            self._speed_form.setRowVisible(row, absolute)
        for row in self._percentile_rows:
            self._speed_form.setRowVisible(row, not absolute)
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
        if self._speed_mode_value() == "percentile":
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

        self._apply_thread = QThread()
        self._apply_worker = ApplyWorker(
            video=video,
            model_path=self._model_path,
            yolo_path=self._yolo_path,
            keypoint_names=self._keypoint_names,
            output_dir=video_output_dir,
            speed_opts=self._speed_opts(),
            predict_every=self._predict_every.value(),
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
