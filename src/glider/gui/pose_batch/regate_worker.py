"""QObject worker for the post-hoc arena re-gate.

The walk itself lives in :mod:`glider.gui.pose_batch.arena_actions`, which is
Qt-free and therefore testable without building a thread; this file is only the
wrapper that carries its callbacks across to the GUI as signals.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from glider.gui.pose_batch.arena_actions import regate_videos


class RegateWorker(QObject):
    """Re-gate tracked CSVs off the GUI thread.

    A sibling to :class:`~glider.gui.pose_batch.worker.PoseBatchWorker` rather
    than a mode of it: that one owns a GPU-bound inference run with per-frame
    progress and a cancel that has to land between frames, this one is I/O over
    a handful of CSVs. Folding them together would put two cancellation stories
    in one object.
    """

    progress = pyqtSignal(int, int)  # videos done, total videos
    log = pyqtSignal(str)
    finished = pyqtSignal(int, int)  # gated, skipped
    failed = pyqtSignal(str)  # nothing ran — a bug, not one awkward session

    def __init__(self, videos, calibrations, *, settings=None):
        super().__init__()
        self._videos = list(videos)
        self._calibrations = calibrations
        self._settings = settings

    def run(self) -> None:
        try:
            gated, skipped = regate_videos(
                self._videos,
                self._calibrations,
                settings=self._settings,
                on_log=self.log.emit,
                on_progress=self.progress.emit,
            )
        except Exception as e:  # surface as a UI message, never crash the thread
            self.failed.emit(str(e))
            return
        self.finished.emit(gated, skipped)
