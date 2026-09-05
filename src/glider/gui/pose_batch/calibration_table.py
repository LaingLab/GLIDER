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

from glider.vision.arena import DegenerateArenaError
from glider.vision.calibration_set import CalibrationSet

_COLUMNS = ("Video", "Resolution", "px/mm", "Arena", "Status")

#: Index of the arena column. Double-clicking it asks for the perimeter rather
#: than the line, so both dialogs are reachable without a modifier key.
ARENA_COLUMN = 3
_STATUS_COLUMN = 4


class CalibrationTable(QTableWidget):
    """One row per video: resolution, scale, and whether it still needs work."""

    calibrate_requested = pyqtSignal(object)  # Path
    arena_requested = pyqtSignal(object)  # Path

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
        self.setToolTip(
            "Double-click a video to draw its measurement line, "
            "or its Arena cell to draw the floor perimeter"
        )

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

            arena_text, arena_tip = self._arena_cell(video)

            self._set_cell(row, 0, video.name, tooltip=str(video))
            self._set_cell(row, 1, resolution)
            self._set_cell(row, 2, f"{ppm:.3f}" if ppm else "—", tooltip=self._scale_tip(video))
            self._set_cell(row, ARENA_COLUMN, arena_text, tooltip=arena_tip)
            self._set_cell(row, _STATUS_COLUMN, self._status_text(video))

    def _status_text(self, video: Path) -> str:
        """Run-readiness, not merely whether a scale exists.

        The Run gate now asks for a confirmed arena, so a row that shows only
        "Calibrated" while Run stays disabled sends the operator hunting.
        """
        if self._calibrations.get_arena(video) is None:
            return "⚠ Needs arena"
        if not self._calibrations.is_arena_confirmed(video):
            return "⚠ Copied — confirm it"
        return "✓ Calibrated" if self._calibrations.px_per_mm(video) else "⚠ Needs calibration"

    def _arena_cell(self, video: Path) -> tuple[str, str]:
        """Text and tooltip for the arena column.

        A perimeter that does not yet describe a quadrilateral is reported, not
        raised: the operator is mid-draw, and the table must keep painting.
        """
        arena = self._calibrations.get_arena(video)
        if arena is None:
            return "—", "No floor perimeter drawn"
        try:
            residuals = arena.residuals()
        except DegenerateArenaError as exc:
            return "⚠ unusable", str(exc)

        size = f"{arena.width_cm:g}x{arena.height_cm:g} cm"
        lines = [
            f"{size} floor",
            f"opposite edges {residuals['edge_ratio']:.2f}x",
            f"scale across floor {residuals['scale_ratio']:.2f}x",
        ]
        if residuals["clipped"]:
            lines.append("extends beyond the frame")
        if residuals["suspect"]:
            lines.append("This quad looks unlikely — check the corners.")
            return "⚠ check", "\n".join(lines)
        return f"✓ {size}", "\n".join(lines)

    def _scale_tip(self, video: Path) -> str:
        """Say which drawing the scale came from, since the arena outranks the line."""
        if self._calibrations.get_arena(video) is not None:
            return "Scale from the floor perimeter"
        if self._calibrations.get(video) is not None:
            return "Scale from the measurement line"
        return ""

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
            signal = (
                self.arena_requested if item.column() == ARENA_COLUMN else self.calibrate_requested
            )
            signal.emit(self._videos[row])
