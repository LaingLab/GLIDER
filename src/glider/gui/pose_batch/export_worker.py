"""QObject worker for the multi-animal DLC CSV export.

The walk itself lives in :mod:`glider.gui.pose_batch.export_actions`, which is
Qt-free and therefore testable without building a thread; this file is only
the wrapper that carries its callbacks across to the GUI as signals. A sibling
to :class:`~glider.gui.pose_batch.regate_worker.RegateWorker`, for the same
reason that one is not folded into :class:`~glider.gui.pose_batch.worker.PoseBatchWorker`:
this is I/O over a handful of already-written CSVs, with no cancel and no
GPU-bound per-frame progress.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from glider.gui.pose_batch.export_actions import export_sessions


class ExportWorker(QObject):
    """Export every multi-animal session in a video list, off the GUI thread."""

    progress = pyqtSignal(int, int)  # videos done, total videos
    log = pyqtSignal(str)
    finished = pyqtSignal(int, int)  # exported, skipped
    failed = pyqtSignal(str)  # nothing ran — a bug, not one awkward session

    def __init__(self, videos):
        super().__init__()
        self._videos = list(videos)

    def run(self) -> None:
        try:
            exported, skipped = export_sessions(
                self._videos,
                on_log=self.log.emit,
                on_progress=self.progress.emit,
            )
        except Exception as e:  # surface as a UI message, never crash the thread
            self.failed.emit(str(e))
            return
        self.finished.emit(exported, skipped)
