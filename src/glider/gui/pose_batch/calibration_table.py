"""Per-video calibration status for the Batch Pose Tracking window.

Its own module so neither this nor ``window.py`` grows unwieldy: this owns how
calibration state is *displayed and selected*, the window owns what to do
about it.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from glider.vision.calibration_set import CalibrationSet

_COLUMNS = ("Video", "Resolution", "px/mm", "Status")


class CalibrationTable(QTableWidget):
    """One row per video: resolution, scale, and whether it still needs work."""

    calibrate_requested = pyqtSignal(object)  # Path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._videos: list[Path] = []
        self._calibrations = CalibrationSet()

        self.setColumnCount(len(_COLUMNS))
        self.setHorizontalHeaderLabels(_COLUMNS)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(_COLUMNS)):
            self.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.setToolTip("Double-click a video to calibrate it")

        self.itemDoubleClicked.connect(self._on_double_click)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def set_calibration_set(self, calibrations: CalibrationSet) -> None:
        """Share the window's set by reference, so refresh() sees edits."""
        self._calibrations = calibrations
        self.refresh()

    def set_videos(self, videos) -> None:
        self._videos = [Path(v) for v in videos]
        self.refresh()

    def videos(self) -> list[Path]:
        return list(self._videos)

    def selected_videos(self) -> list[Path]:
        rows = sorted({index.row() for index in self.selectedIndexes()})
        return [self._videos[r] for r in rows if 0 <= r < len(self._videos)]

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        self.setRowCount(len(self._videos))
        for row, video in enumerate(self._videos):
            calibration = self._calibrations.get(video)
            ppm = self._calibrations.px_per_mm(video)

            if calibration is not None and calibration.calibration_width:
                resolution = f"{calibration.calibration_width}x{calibration.calibration_height}"
            else:
                resolution = "—"

            self._set_cell(row, 0, video.name, tooltip=str(video))
            self._set_cell(row, 1, resolution)
            self._set_cell(row, 2, f"{ppm:.3f}" if ppm else "—")
            self._set_cell(
                row,
                3,
                "✓ Calibrated" if ppm else "⚠ Needs calibration",
            )

    def _set_cell(self, row: int, col: int, text: str, *, tooltip: str = "") -> None:
        item = QTableWidgetItem(text)
        if tooltip:
            item.setToolTip(tooltip)
        if col:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, col, item)

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if 0 <= row < len(self._videos):
            self.calibrate_requested.emit(self._videos[row])
