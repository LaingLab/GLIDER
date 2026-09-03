"""QObject worker running the Qt-free batch core off the UI thread.

Follows :mod:`glider.gui.behavior.workers`: move an instance onto a QThread,
call :meth:`PoseBatchWorker.run` from the thread's ``started`` signal, and never
let an exception escape into the thread — ``run()`` catches broadly and reports
through ``failed``.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from glider.vision.pose import batch as batch_core

# A 30 fps hour-long video would emit ~108,000 cross-thread signals; throttle
# per-frame updates so the Qt event queue isn't flooded by the progress bar.
_PROGRESS_INTERVAL_S = 0.1


class PoseBatchWorker(QObject):
    """Drives :func:`glider.vision.pose.batch.run_batch` and reports progress."""

    progress = pyqtSignal(int, int)  # video index, total videos
    video_progress = pyqtSignal(int, int)  # frames done, total frames (0 = unknown)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)  # BatchResult
    failed = pyqtSignal(str)  # preflight failure — no video ran

    def __init__(
        self,
        videos,
        model_path,
        keypoint_names,
        *,
        conf=0.25,
        device=None,
        require_gpu=False,
        overwrite=False,
        filtering=None,
        zones=None,
    ):
        super().__init__()
        self._videos = list(videos)
        self._model_path = Path(model_path)
        self._names = list(keypoint_names)
        self._conf = conf
        self._device = device
        self._require_gpu = require_gpu
        self._overwrite = overwrite
        self._filtering = filtering
        # Centre zones per video, when arenas have been drawn. Scored from the
        # track as each video finishes, so no second pass over the video.
        self._zones = zones or {}
        self._cancel = threading.Event()
        self._last_emit = 0.0

    def cancel(self) -> None:
        """Ask the batch to stop. Thread-safe: called from the GUI thread."""
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def _on_event(self, event: batch_core.BatchEvent) -> None:
        kind = batch_core.EventKind
        name = event.video.name
        position = f"[{event.index + 1}/{event.total}]"

        if event.kind is kind.STARTED:
            self.progress.emit(event.index, event.total)
            self._last_emit = 0.0
            self.video_progress.emit(0, 0)
            self.log.emit(f"{position} {name}")
        elif event.kind is kind.WROTE:
            self.log.emit(f"    wrote {event.output.name}")
        elif event.kind is kind.SKIPPED:
            self.log.emit(f"{position} {name} — skipped (output exists)")
        elif event.kind is kind.FAILED:
            self.log.emit(f"    FAILED: {event.message}")
        elif event.kind is kind.CANCELLED:
            self.log.emit("Cancelled.")

    def _on_frame(self, done: int, total: int) -> None:
        now = time.monotonic()
        # Always emit the final frame so the bar lands on 100%.
        if now - self._last_emit >= _PROGRESS_INTERVAL_S or (total and done >= total):
            self._last_emit = now
            self.video_progress.emit(done, total)

    def run(self) -> None:
        try:
            result = batch_core.run_batch(
                self._videos,
                self._model_path,
                self._names,
                conf=self._conf,
                device=self._device,
                require_gpu=self._require_gpu,
                overwrite=self._overwrite,
                filtering=self._filtering,
                on_event=self._on_event,
                cancel_cb=self._cancel.is_set,
                progress_cb=self._on_frame,
                zones=self._zones,
            )
        except Exception as e:  # surface as a UI message, never crash the thread
            self.failed.emit(str(e))
            return
        self.log.emit(result.summary)
        self.finished.emit(result)
