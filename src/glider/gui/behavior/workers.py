"""QObject workers that run the Qt-free behavior cores off the UI thread.

Move an instance onto a :class:`~PyQt6.QtCore.QThread` and call :meth:`run`
via the thread's ``started`` signal, following the shape of
:mod:`glider.gui.panels.video_tracking_worker`. Each worker reports success
through ``finished`` and any error through ``failed`` — a worker must never
crash its thread, so ``run()`` catches broadly and forwards the message.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from glider.analysis.behavior import train_model
from glider.analysis.behavior.classify import classify
from glider.vision.pose import infer_video, smooth

logger = logging.getLogger(__name__)


class _BaseWorker(QObject):
    progress = pyqtSignal(int, int)  # done, total
    finished = pyqtSignal(object)  # result payload
    failed = pyqtSignal(str)  # error message


class ConvertWorker(_BaseWorker):
    """Run YOLO-pose inference over a video and write a DLC pose CSV."""

    def __init__(self, video, model, keypoint_names, output, *, device=None):
        super().__init__()
        self._args = (video, model, keypoint_names, output, device)

    def run(self) -> None:
        try:
            video, model, names, output, device = self._args
            pose = infer_video(
                model_path=str(model),
                video_path=str(video),
                keypoint_names=names,
                device=device,
            )
            pose = smooth(pose)
            pose_csv = Path(output)
            from glider.vision.pose import dlc

            dlc.to_dlc_csv(pose, pose_csv)
            self.finished.emit(str(pose_csv))
        except Exception as e:  # surface as a UI message, never crash the thread
            self.failed.emit(str(e))


def report_dir_for(model_path) -> Path:
    """Where a run's report folder goes for a given model bundle.

    A sibling of the bundle named after it, so the numbers a model was
    accepted on stay next to the model itself. Two bundles in one folder keep
    separate reports, and copying the pair keeps them together.
    """
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_report")


def _write_report(result, model_path) -> Path | None:
    """Write the run's report folder, or return None if it could not be written.

    Never raises. The model is already on disk by the time this runs, and a
    report is a convenience: losing a ten-minute fit because a chart could not
    be rendered, or because the output folder is a read-only share, would be a
    far worse outcome than having no report.
    """
    from glider.analysis.behavior import write_training_report

    try:
        return write_training_report(result, report_dir_for(model_path))
    except Exception:  # noqa: BLE001 - the model is saved; the report is a bonus
        logger.warning("could not write the training report", exc_info=True)
        return None


class TrainWorker(_BaseWorker):
    """Fit a behavior classifier from labeled sessions and save the bundle."""

    #: Emitted with the report folder once it is on disk, before ``finished``.
    #: A separate signal rather than a richer ``finished`` payload: that one is
    #: the summary dict and several callers already read it positionally.
    report_ready = pyqtSignal(object)  # Path | None

    # sessions: list[(pose_csv, annotations_csv)] pairs (train_model's SessionPair).
    def __init__(self, sessions, output, options):
        super().__init__()
        self._sessions, self._output, self._options = sessions, output, options

    def run(self) -> None:
        try:
            result = train_model(self._sessions, **self._options)  # -> TrainResult
            result.model.save(self._output)  # train_model does NOT write
            # On this thread deliberately: the report renders charts through
            # matplotlib, and doing that on the GUI thread would freeze the
            # window at the moment the run looks finished.
            self.report_ready.emit(_write_report(result, self._output))
            self.finished.emit(result.summary)
        except Exception as e:
            self.failed.emit(str(e))


#: Options that describe the model a run produces, not the measurement.
#: cross_validate_sessions fits no final model and does not accept them.
_MODEL_ONLY_OPTIONS = frozenset({"embedding"})


class CrossValidateWorker(_BaseWorker):
    """Session-grouped K-fold cross-validation over the training sessions.

    Measures only — :func:`cross_validate_sessions` deliberately returns no
    model. Kept separate from :class:`TrainWorker` for that reason: a caller
    cannot accidentally treat the result as something to save.
    """

    #: See :attr:`TrainWorker.report_ready`. Emitted with None on the
    #: measure-only path, which produces no model and so has nowhere to put a
    #: report — the Review tab needs to hear that as much as it needs a path.
    report_ready = pyqtSignal(object)  # Path | None

    def __init__(self, sessions, options, output=None):
        super().__init__()
        self._sessions, self._options, self._output = sessions, options, output

    def run(self) -> None:
        try:
            if self._output is None:
                from glider.analysis.behavior import cross_validate_sessions

                # Measuring produces no model, so options that only describe
                # one are not just unused here — cross_validate_sessions does
                # not accept them, and passing one through is a TypeError
                # minutes into a run.
                measure_only = {
                    k: v for k, v in self._options.items() if k not in _MODEL_ONLY_OPTIONS
                }
                self.report_ready.emit(None)
                self.finished.emit(cross_validate_sessions(self._sessions, **measure_only))
                return

            # Measure and produce in one pass. Two separate calls would
            # assemble the feature matrix twice, which with motion features
            # means decoding every source video twice.
            from glider.analysis.behavior import cross_validate_and_train

            cv_result, trained = cross_validate_and_train(self._sessions, **self._options)
            trained.model.save(self._output)
            # The bundle's own summary carries the cross-validated score, so
            # the report describes the model that was actually saved.
            self.report_ready.emit(_write_report(trained, self._output))
            self.finished.emit(cv_result)
        except Exception as e:  # surface as a UI message, never crash the thread
            self.failed.emit(str(e))


class ApplyWorker(_BaseWorker):
    """Classify a recorded video with a trained model and write the ethogram.

    ``speed_opts`` carries the optional freeze/dart axis — ``freeze_mm_s`` /
    ``dart_mm_s`` plus a ``calibration_master`` or ``px_per_mm`` to convert
    them. Empty leaves the axis off, which is the pre-existing behaviour.

    ``predict_every`` is the classifier cadence in tracked frames: 3 (the
    pipeline default) samples the model at ~10 Hz on 30 fps video, 1 gives a
    prediction on every frame. Kept separate from ``speed_opts`` because it
    reaches ``LiveInferenceConfig`` through ``classify``'s ``**opts``, not
    through the speed-threshold resolver.
    """

    def __init__(
        self,
        video,
        model_path,
        yolo_path,
        keypoint_names,
        output_dir,
        *,
        device=None,
        speed_opts=None,
        predict_every=None,
        write_annotated=False,
        reuse_existing_poses=False,
        pose_dir=None,
        smooth_window=None,
        offline_smooth_window=None,
        min_bout_s=None,
        start_s=None,
        end_s=None,
    ):
        super().__init__()
        self._args = (video, model_path, yolo_path, keypoint_names, output_dir, device)
        self._speed_opts = dict(speed_opts or {})
        # Off by default: encoding the annotated MP4 costs more than the
        # inference on a long recording, and it is a spot-check aid.
        self._write_annotated = bool(write_annotated)
        self._reuse_existing_poses = bool(reuse_existing_poses)
        # Where to look for those poses. None = beside each video.
        self._pose_dir = pose_dir
        # None for both = leave the pipeline defaults as the single source of
        # truth, exactly as predict_every does below.
        self._smooth_window = smooth_window
        # Centred vote over the finished predictions. Offline-only, and worth
        # 0.780 -> 0.823 macro F1 on held-out data, so the Apply tab turns it
        # on by default -- scoring a recording is precisely the case where
        # reading the frames after each one is free and correct.
        self._offline_smooth_window = offline_smooth_window
        self._min_bout_s = min_bout_s
        # Inclusive analysis window in seconds; None either side = open.
        self._start_s = start_s
        self._end_s = end_s
        # None = don't pass it at all, so the pipeline default stays the single
        # source of truth for the cadence.
        self._predict_every = predict_every

    def run(self) -> None:
        try:
            video, model_path, yolo_path, names, output_dir, device = self._args
            opts = dict(self._speed_opts)
            if self._predict_every is not None:
                opts["predict_every"] = int(self._predict_every)
            if self._smooth_window is not None:
                opts["smooth_window"] = int(self._smooth_window)
            if self._offline_smooth_window is not None:
                opts["offline_smooth_window"] = int(self._offline_smooth_window)
            result = classify(
                video,
                model_path=model_path,
                yolo_path=yolo_path,
                keypoint_names=names,
                output_dir=output_dir,
                device=device,
                write_annotated=self._write_annotated,
                reuse_existing_poses=self._reuse_existing_poses,
                pose_dir=self._pose_dir,
                min_bout_s=self._min_bout_s,
                start_s=self._start_s,
                end_s=self._end_s,
                **opts,
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class CohortSpeedWorker(_BaseWorker):
    """Pool a cohort's pose CSVs into one set of freeze/dart thresholds.

    On a worker thread because it is not fast: a session is ~5 s of per-frame
    Python, so a 22-video cohort is minutes. Run on the UI thread it stops Qt
    pumping events, Windows paints the window "Not Responding", and the
    operator reasonably concludes it crashed and kills it before it saves.
    """

    def __init__(
        self,
        pose_csvs,
        output,
        *,
        freeze_pct,
        dart_pct,
        calibration_master=None,
        start_s=None,
        end_s=None,
    ):
        super().__init__()
        self._pose_csvs = list(pose_csvs)
        self._output = output
        self._freeze_pct = float(freeze_pct)
        self._dart_pct = float(dart_pct)
        self._calibration_master = calibration_master
        # Pool the same stretch the run will score; see _on_compute_cohort.
        self._start_s = start_s
        self._end_s = end_s

    def run(self) -> None:
        try:
            from glider.analysis.behavior.cohort_speed import compute_cohort_thresholds

            thresholds = compute_cohort_thresholds(
                self._pose_csvs,
                freeze_pct=self._freeze_pct,
                dart_pct=self._dart_pct,
                calibration_master=self._calibration_master,
                start_s=self._start_s,
                end_s=self._end_s,
                progress=lambda done, total, _name: self.progress.emit(done, total),
            )
            thresholds.save(self._output)
            self.finished.emit(thresholds)
        except Exception as e:
            self.failed.emit(str(e))
