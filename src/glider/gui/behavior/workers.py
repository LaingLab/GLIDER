"""QObject workers that run the Qt-free behavior cores off the UI thread.

Move an instance onto a :class:`~PyQt6.QtCore.QThread` and call :meth:`run`
via the thread's ``started`` signal, following the shape of
:mod:`glider.gui.panels.video_tracking_worker`. Each worker reports success
through ``finished`` and any error through ``failed`` — a worker must never
crash its thread, so ``run()`` catches broadly and forwards the message.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from glider.analysis.behavior import train_model
from glider.analysis.behavior.classify import classify
from glider.vision.pose import infer_video, smooth


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


class TrainWorker(_BaseWorker):
    """Fit a behavior classifier from labeled sessions and save the bundle."""

    # sessions: list[(pose_csv, annotations_csv)] pairs (train_model's SessionPair).
    def __init__(self, sessions, output, options):
        super().__init__()
        self._sessions, self._output, self._options = sessions, output, options

    def run(self) -> None:
        try:
            result = train_model(self._sessions, **self._options)  # -> TrainResult
            result.model.save(self._output)  # train_model does NOT write
            self.finished.emit(result.summary)
        except Exception as e:
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
    ):
        super().__init__()
        self._args = (video, model_path, yolo_path, keypoint_names, output_dir, device)
        self._speed_opts = dict(speed_opts or {})
        # Off by default: encoding the annotated MP4 costs more than the
        # inference on a long recording, and it is a spot-check aid.
        self._write_annotated = bool(write_annotated)
        self._reuse_existing_poses = bool(reuse_existing_poses)
        # None = don't pass it at all, so the pipeline default stays the single
        # source of truth for the cadence.
        self._predict_every = predict_every

    def run(self) -> None:
        try:
            video, model_path, yolo_path, names, output_dir, device = self._args
            opts = dict(self._speed_opts)
            if self._predict_every is not None:
                opts["predict_every"] = int(self._predict_every)
            result = classify(
                video,
                model_path=model_path,
                yolo_path=yolo_path,
                keypoint_names=names,
                output_dir=output_dir,
                device=device,
                write_annotated=self._write_annotated,
                reuse_existing_poses=self._reuse_existing_poses,
                **opts,
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
