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

It is also where each camera is switched on or off and recorded on its own:
the checkbox beside a camera's name, and the button at the end of its row.
"""

from __future__ import annotations

from contextlib import contextmanager

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QPushButton,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
)

from glider.gui.styles import colors

_COLUMNS = ("Camera", "FPS", "Queue", "Dropped", "Time", "State", "")
_FPS, _QUEUE, _DROPPED, _TIME, _STATE, _RECORD = range(1, len(_COLUMNS))


def format_elapsed(seconds: float) -> str:
    """mm:ss, or h:mm:ss once a take runs past an hour."""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class _WholeCellDelegate(QStyledItemDelegate):
    """Lay a cell's widget over the whole cell.

    The default delegate fits it inside the stylesheet's item padding, which
    under the tool theme left the Record buttons shorter than their own text.
    """

    def updateEditorGeometry(self, editor, option, index):  # noqa: N802 - Qt override
        editor.setGeometry(option.rect.adjusted(2, 2, -2, -2))


class CameraStatusTable(QTableWidget):
    """One row per camera: is it on, is it keeping up, and has it lost anything."""

    #: The operator ticked or unticked a camera: ``(camera_id, on)``.
    enabled_toggled = pyqtSignal(str, bool)
    #: The operator pressed a camera's Record / Stop button.
    record_toggled = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: dict[str, int] = {}
        self._flagged: set[str] = set()
        self._enabled: dict[str, bool] = {}
        self._recording: set[str] = set()
        self._saving: set[str] = set()
        self._buttons: dict[str, QPushButton] = {}
        self._updating = False

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
        self.setItemDelegateForColumn(_RECORD, _WholeCellDelegate(self))
        self.itemChanged.connect(self._on_item_changed)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def set_cameras(self, camera_ids, names: dict[str, str] | None = None) -> None:
        """Replace the rows, in the order given.

        Cameras keep whether they were switched on across a rebuild; a camera
        not seen before starts on.
        """
        ids = list(camera_ids)
        names = names or {}
        self._rows = {cam: i for i, cam in enumerate(ids)}
        self._flagged.clear()
        self._enabled = {cam: self._enabled.get(cam, True) for cam in ids}
        self._recording &= set(ids)
        self._saving &= set(ids)
        self._buttons = {}
        self._fit_record_buttons()

        with self._quietly():
            self.setRowCount(len(ids))
            for row, cam in enumerate(ids):
                name_item = QTableWidgetItem(names.get(cam) or cam)
                name_item.setData(Qt.ItemDataRole.UserRole, cam)
                self.setItem(row, 0, name_item)
                for col in range(1, len(_COLUMNS)):
                    self._set(row, col, "" if col == _RECORD else "—")

                button = QPushButton("Record")
                button.clicked.connect(lambda _=False, c=cam: self.record_toggled.emit(c))
                self.setCellWidget(row, _RECORD, button)
                self._buttons[cam] = button

        for cam in ids:
            self._refresh_controls(cam)
            if not self._enabled[cam]:
                self._show_off(cam)

    def set_display_name(self, camera_id: str, name: str) -> None:
        """Show the operator's name for this camera instead of its id."""
        row = self._rows.get(camera_id)
        if row is None:
            return
        item = self.item(row, 0)
        if item is not None:
            with self._quietly():
                item.setText(name or camera_id)

    def set_enabled(self, camera_id: str, enabled: bool) -> None:
        """Tick or untick a camera, without announcing it as the operator's doing."""
        if camera_id not in self._rows:
            return
        self._enabled[camera_id] = enabled
        self._refresh_controls(camera_id)
        if not enabled:
            self._recording.discard(camera_id)
            self._saving.discard(camera_id)
            self._show_off(camera_id)

    def update_status(
        self,
        camera_id: str,
        *,
        fps: float,
        queue_depth: int,
        dropped: int,
        recording: bool,
        elapsed: float | None = None,
        saving: bool = False,
    ) -> None:
        """Refresh one camera's row. Unknown and switched-off cameras are ignored.

        A camera can disappear between a status poll being scheduled and it
        arriving - a USB device can be unplugged mid-session - and that should
        not raise into the timer that drives this.

        ``elapsed`` is how long this camera's take has run; None leaves the
        Time column as it is.
        """
        row = self._rows.get(camera_id)
        if row is None or not self._enabled.get(camera_id, True):
            return

        stalled = recording and fps <= 0.0
        flagged = dropped > 0 or stalled
        if flagged:
            self._flagged.add(camera_id)
        else:
            self._flagged.discard(camera_id)

        if saving and not recording:
            state = "saving…"
        elif not recording:
            state = "idle"
        elif stalled:
            state = "STALLED"
        elif dropped:
            state = "dropping"
        else:
            state = "recording"

        self._set(row, _FPS, f"{fps:.1f}")
        self._set(row, _QUEUE, str(queue_depth))
        self._set(row, _DROPPED, str(dropped), warn=dropped > 0)
        if elapsed is not None:
            self._set(row, _TIME, format_elapsed(elapsed) if elapsed > 0 else "—")
        self._set(row, _STATE, state, warn=flagged)

        changed = (camera_id in self._recording) != recording or (
            camera_id in self._saving
        ) != saving
        if recording:
            self._recording.add(camera_id)
        else:
            self._recording.discard(camera_id)
        if saving:
            self._saving.add(camera_id)
        else:
            self._saving.discard(camera_id)
        if changed:
            self._refresh_controls(camera_id)

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def is_enabled(self, camera_id: str) -> bool:
        """Whether this camera is ticked."""
        return self._enabled.get(camera_id, False)

    def record_button(self, camera_id: str) -> QPushButton | None:
        """The Record / Stop button on this camera's row."""
        return self._buttons.get(camera_id)

    def is_flagged(self, camera_id: str) -> bool:
        """Whether this camera has dropped frames or stalled."""
        return camera_id in self._flagged

    def any_flagged(self) -> bool:
        """Whether any camera needs attention — the one-glance answer."""
        return bool(self._flagged)

    def flagged_cameras(self) -> list[str]:
        return sorted(self._flagged)

    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """The operator ticked or unticked a camera."""
        if self._updating or item.column() != 0:
            return
        camera_id = item.data(Qt.ItemDataRole.UserRole)
        if camera_id not in self._rows:
            return
        on = item.checkState() == Qt.CheckState.Checked
        if on == self._enabled.get(camera_id, True):
            return
        self.set_enabled(camera_id, on)
        self.enabled_toggled.emit(camera_id, on)

    def _refresh_controls(self, camera_id: str) -> None:
        """Bring a row's checkbox and button into line with its state.

        A camera cannot be switched off while it has a file open: switching
        off disconnects it, which would end that file without finishing it.
        """
        row = self._rows.get(camera_id)
        if row is None:
            return
        enabled = self._enabled.get(camera_id, True)
        recording = camera_id in self._recording
        saving = camera_id in self._saving

        item = self.item(row, 0)
        if item is not None:
            with self._quietly():
                flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                if not (recording or saving):
                    flags |= Qt.ItemFlag.ItemIsUserCheckable
                item.setFlags(flags)
                item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
                if recording or saving:
                    item.setToolTip("Stop recording this camera before switching it off")
                else:
                    item.setToolTip(
                        "Untick to switch this camera off: it is disconnected, "
                        "so it is neither previewed nor recorded"
                    )
                # None hands the colour back to the stylesheet.
                item.setData(
                    Qt.ItemDataRole.ForegroundRole,
                    None if enabled else QColor(colors.TEXT_DISABLED),
                )

        button = self._buttons.get(camera_id)
        if button is not None:
            if saving and not recording:
                button.setText("Saving…")
                button.setToolTip("Finishing this camera's file")
            elif recording:
                button.setText("Stop")
                button.setToolTip("Stop recording this camera; the others carry on")
            else:
                button.setText("Record")
                button.setToolTip(
                    "Record this camera on its own"
                    if enabled
                    else "Tick this camera to switch it on first"
                )
            button.setEnabled(enabled and not (saving and not recording))

    def _fit_record_buttons(self) -> None:
        """Make the rows and the button column big enough for the buttons.

        A cell widget does not size its row or column. Under the tool theme's
        button padding, rows sized for text cut every Record button down to
        a sliver of its label.

        The allowance is :class:`_WholeCellDelegate`'s 2 px each side, plus the
        grid line that takes a pixel of every cell.
        """
        probe = QPushButton("Saving…", self)
        probe.ensurePolished()
        hint = probe.sizeHint()
        probe.deleteLater()
        header = self.verticalHeader()
        header.setDefaultSectionSize(max(header.defaultSectionSize(), hint.height() + 6))
        self.horizontalHeader().setSectionResizeMode(_RECORD, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(_RECORD, hint.width() + 8)

    def _show_off(self, camera_id: str) -> None:
        """Blank a switched-off camera's numbers: it has none."""
        row = self._rows.get(camera_id)
        if row is None:
            return
        self._flagged.discard(camera_id)
        for col in (_FPS, _QUEUE, _DROPPED, _TIME):
            self._set(row, col, "—", muted=True)
        self._set(row, _STATE, "off", muted=True)

    def _set(
        self, row: int, col: int, text: str, *, warn: bool = False, muted: bool = False
    ) -> None:
        item = QTableWidgetItem(text)
        if col:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if warn:
            item.setForeground(Qt.GlobalColor.red)
            item.setToolTip("Frames are being lost — stop and check before relying on this file")
        elif muted:
            item.setForeground(QColor(colors.TEXT_DISABLED))
        with self._quietly():
            self.setItem(row, col, item)

    @contextmanager
    def _quietly(self):
        """Change items without it reading as the operator ticking a box."""
        previous, self._updating = self._updating, True
        try:
            yield
        finally:
            self._updating = previous


__all__ = ["CameraStatusTable", "format_elapsed"]


# Kept for callers that want the same palette as the rest of the tool windows.
WARN_COLOUR = colors.WARNING
