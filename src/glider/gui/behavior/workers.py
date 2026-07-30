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
    ):
        super().__init__()
        self._args = (video, model_path, yolo_path, keypoint_names, output_dir, device)
        self._speed_opts = dict(speed_opts or {})

    def run(self) -> None:
        try:
            video, model_path, yolo_path, names, output_dir, device = self._args
            result = classify(
                video,
                model_path=model_path,
                yolo_path=yolo_path,
                keypoint_names=names,
                output_dir=output_dir,
                device=device,
                **self._speed_opts,
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))
