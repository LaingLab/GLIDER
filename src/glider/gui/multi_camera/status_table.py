"""Per-camera status for the Multi-Camera Recording window.

Its own module so neither this nor ``window.py`` grows unwieldy, mirroring
:mod:`glider.gui.pose_batch.calibration_table`: this owns how per-camera state
is *displayed*, the window owns what to do about it.

The column that earns this table is Dropped. :class:`MultiVideoRecorder` counts
dropped frames per camera and logs a total when recording stops, which is the
wrong time to learn about it - on a sixteen-camera rig one starved writer
silently shortens one animal's recording while everything on screen looks
healthy. Zero FPS during a run is the same failure a step earlier: the capture
thread has stopped and the file simply stops growing.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from glider.gui.styles import colors

_COLUMNS = ("Camera", "FPS", "Queue", "Dropped", "State")


class CameraStatusTable(QTableWidget):
    """One row per camera: is it keeping up, and has it lost anything."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[str, int] = {}
        self._flagged: set[str] = set()

        self.setColumnCount(len(_COLUMNS))
        self.setHorizontalHeaderLabels(_COLUMNS)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(_COLUMNS)):
            self.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def set_cameras(self, camera_ids) -> None:
        """Replace the rows, in the order given."""
        ids = list(camera_ids)
        self._rows = {cam: i for i, cam in enumerate(ids)}
        self._flagged.clear()
        self.setRowCount(len(ids))
        for row, cam in enumerate(ids):
            self._set(row, 0, cam)
            for col in range(1, len(_COLUMNS)):
                self._set(row, col, "—")

    def update_status(
        self,
        camera_id: str,
        *,
        fps: float,
        queue_depth: int,
        dropped: int,
        recording: bool,
    ) -> None:
        """Refresh one camera's row. Unknown cameras are ignored.

        A camera can disappear between a status poll being scheduled and it
        arriving - a USB device can be unplugged mid-session - and that should
        not raise into the timer that drives this.
        """
        row = self._rows.get(camera_id)
        if row is None:
            return

        stalled = recording and fps <= 0.0
        flagged = dropped > 0 or stalled
        if flagged:
            self._flagged.add(camera_id)
        else:
            self._flagged.discard(camera_id)

        if not recording:
            state = "idle"
        elif stalled:
            state = "STALLED"
        elif dropped:
            state = "dropping"
        else:
            state = "recording"

        self._set(row, 1, f"{fps:.1f}")
        self._set(row, 2, str(queue_depth))
        self._set(row, 3, str(dropped), warn=dropped > 0)
        self._set(row, 4, state, warn=flagged)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def is_flagged(self, camera_id: str) -> bool:
        """Whether this camera has dropped frames or stalled."""
        return camera_id in self._flagged

    def any_flagged(self) -> bool:
        """Whether any camera needs attention — the one-glance answer."""
        return bool(self._flagged)

    def flagged_cameras(self) -> list[str]:
        return sorted(self._flagged)

    # ------------------------------------------------------------------

    def _set(self, row: int, col: int, text: str, *, warn: bool = False) -> None:
        item = QTableWidgetItem(text)
        if col:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if warn:
            item.setForeground(Qt.GlobalColor.red)
            item.setToolTip("Frames are being lost — stop and check before relying on this file")
        self.setItem(row, col, item)


__all__ = ["CameraStatusTable"]


# Kept for callers that want the same palette as the rest of the tool windows.
WARN_COLOUR = colors.WARNING
