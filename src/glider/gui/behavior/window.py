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

import pprint
from pathlib import Path

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Video extensions the annotator / apply pickers accept, kept in sync with
# glider.analysis.behavior.project.VIDEO_EXTS (module-level, no heavy deps).
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
_VIDEO_FILTER = "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)"


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
        sessions = [(v, poses_dir / f"{v.stem}.csv") for v in videos]
        missing = [str(p) for _v, p in sessions if not p.exists()]
        if missing:
            QMessageBox.warning(
                self,
                "Annotate",
                "Missing pose CSV(s) for:\n" + "\n".join(missing),
            )
            return

        # Deferred: propose_clips_multi pulls in sklearn; AnnotatorWindow
        # pulls in cv2 via the clip player.
        from glider.analysis.behavior.vocabulary import Vocabulary
        from glider.gui.behavior.annotator.app import annotation_path_for
        from glider.gui.behavior.annotator.capture_cache import VideoCaptureCache
        from glider.gui.behavior.annotator.main_window import AnnotatorWindow
        from glider.gui.behavior.annotator.sampler import propose_clips_multi

        # Annotations live next to the POSE CSV — same place training reads
        # them from (mirrors annotator/app.py's run()).
        videos_meta = {v: annotation_path_for(p) for v, p in sessions}
        pairs = [(p, v) for v, p in sessions]  # (pose_csv, video) for the sampler
        try:
            clips = propose_clips_multi(sessions=pairs, n_clips_total=max(50, len(pairs)))
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
            vocab=vocab,
            vocab_path=vocab_path,
            capture_cache=capture_cache,
        )
        self._annotator_window.show()


class TrainTab(QWidget):
    """Fit a behavior classifier from labeled sessions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sessions: list[tuple[Path, Path]] = []
        self._holdout: list[tuple[Path, Path]] = []
        self._output_path: Path | None = None
        self._train_thread: QThread | None = None
        self._train_worker = None

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
        # train_model(classifier_type=...) accepts exactly "rf" (default,
        # RandomForestClassifier) or "lightgbm" (LGBMClassifier) — see
        # glider.analysis.behavior.pipeline.train_model docstring.
        self._classifier_combo.addItems(["rf", "lightgbm"])
        options_row.addWidget(self._classifier_combo)
        self._background_check = QCheckBox("Include background class")
        self._mirror_check = QCheckBox("Mirror augment")
        options_row.addWidget(self._background_check)
        options_row.addWidget(self._mirror_check)
        options_row.addStretch(1)
        layout.addLayout(options_row)

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
        layout.addLayout(keypoints_row)

        layout.addWidget(self._build_speed_group())

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

        self._results.clear()
        self._queue = list(self._videos)
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._keypoint_names = keypoint_names
        self._run_next()

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

        self._freeze_mm_s = QDoubleSpinBox()
        self._freeze_mm_s.setRange(0.0, 100000.0)
        self._freeze_mm_s.setDecimals(1)
        self._freeze_mm_s.setSuffix(" mm/s")
        self._freeze_mm_s.setValue(10.0)
        form.addRow("Freezing below:", self._freeze_mm_s)

        self._dart_mm_s = QDoubleSpinBox()
        self._dart_mm_s.setRange(0.0, 100000.0)
        self._dart_mm_s.setDecimals(1)
        self._dart_mm_s.setSuffix(" mm/s")
        self._dart_mm_s.setValue(150.0)
        form.addRow("Darting above:", self._dart_mm_s)

        row = QHBoxLayout()
        self._calibration_label = QLabel("(none)")
        self._calibration_label.setWordWrap(True)
        cal_btn = QPushButton("Choose...")
        cal_btn.clicked.connect(self._on_choose_calibration)
        row.addWidget(self._calibration_label, 1)
        row.addWidget(cal_btn)
        form.addRow("Calibration file:", row)

        self._speed_group = group
        return group

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
        return {
            "freeze_mm_s": self._freeze_mm_s.value(),
            "dart_mm_s": self._dart_mm_s.value(),
            "calibration_master": self._calibration_master,
        }

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
