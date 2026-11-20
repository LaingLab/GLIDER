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

_COLUMNS = ("Video", "Folder", "Arena", "Scale")

#: Index of the arena column.
ARENA_COLUMN = 2
_SCALE_COLUMN = 3

#: The three arena states Run cares about, one phrase each.
NEEDS_ARENA = "Needs arena — double-click"
UNCONFIRMED_ARENA = "Copied — open to confirm"


class CalibrationTable(QTableWidget):
    """The batch's one video list: a row per video with its arena and scale.

    It is also the drop target for adding videos: dropped files and folders are
    reported through ``paths_dropped`` and the window decides what they expand to.
    """

    arena_requested = pyqtSignal(object)  # Path
    paths_dropped = pyqtSignal(list)  # list[Path]

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
        self.setAcceptDrops(True)

        self.itemDoubleClicked.connect(self._on_double_click)

    # Qt camelCase overrides (N802 is suppressed project-wide).
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            super().dropEvent(event)
            return
        dropped = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        self.paths_dropped.emit(dropped)
        event.acceptProposedAction()

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
            ppm = self._calibrations.px_per_mm(video)
            arena_text, arena_tip = self._arena_cell(video)

            self._set_cell(row, 0, video.name, tooltip=str(video))
            self._set_cell(row, 1, video.parent.name, tooltip=str(video.parent))
            self._set_cell(row, ARENA_COLUMN, arena_text, tooltip=arena_tip)
            self._set_cell(
                row,
                _SCALE_COLUMN,
                f"{ppm:.2f} px/mm" if ppm else "—",
                tooltip=self._scale_tip(video),
            )

    def _arena_cell(self, video: Path) -> tuple[str, str]:
        """Text and tooltip for the arena column: one phrase per Run-relevant state.

        A perimeter that does not yet describe a quadrilateral is reported, not
        raised: the operator is mid-draw, and the table must keep painting.
        """
        arena = self._calibrations.get_arena(video)
        if arena is None:
            return NEEDS_ARENA, "No floor perimeter drawn"
        try:
            residuals = arena.residuals()
        except DegenerateArenaError as exc:
            return "⚠ unusable", str(exc)

        if arena.shape == "ellipse":
            size = f"⌀{arena.width_cm:g} cm floor"
        elif arena.shape == "plus":
            size = (
                f"{arena.width_cm:g}x{arena.height_cm:g} cm plus maze, "
                f"{arena.arm_width_cm:g} cm arms"
            )
        else:
            size = f"{arena.width_cm:g}x{arena.height_cm:g} cm floor"
        lines = [
            size,
            f"opposite edges {residuals['edge_ratio']:.2f}x",
            f"scale across floor {residuals['scale_ratio']:.2f}x",
        ]
        if residuals["clipped"]:
            lines.append("extends beyond the frame")
        if not self._calibrations.is_arena_confirmed(video):
            lines.append("Copied from another video; open it and check the outline.")
            return UNCONFIRMED_ARENA, "\n".join(lines)
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
        if col == _SCALE_COLUMN:
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(row, col, item)

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if 0 <= row < len(self._videos):
            self.arena_requested.emit(self._videos[row])
