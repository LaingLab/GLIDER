"""
VideoTrackingWorker — Qt adapter around VideoTrackingRunner.

Owns no business logic; it forwards the runner's progress/finished/failed as
Qt signals and exposes cancel(). Move it to a QThread (or call run() in a
worker thread) so the batch pass never blocks the UI.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from glider.vision.cv_processor import CVProcessor
from glider.vision.video_tracking_runner import VideoTrackingConfig, VideoTrackingRunner

logger = logging.getLogger(__name__)


class VideoTrackingWorker(QObject):
    progress = pyqtSignal(int, int)  # done, total
    finished = pyqtSignal(str)  # output_dir
    failed = pyqtSignal(str)  # message

    def __init__(
        self, config: VideoTrackingConfig, cv_processor: CVProcessor | None = None
    ) -> None:
        super().__init__()
        self._runner = VideoTrackingRunner(config, cv_processor=cv_processor)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            out = self._runner.run(
                progress_cb=lambda d, t: self.progress.emit(d, t),
                cancel_cb=lambda: self._cancelled,
            )
            self.finished.emit(str(out))
        except Exception as exc:  # surfaced to the panel, never crashes the UI
            logger.exception("VideoTrackingWorker failed")
            self.failed.emit(str(exc))
