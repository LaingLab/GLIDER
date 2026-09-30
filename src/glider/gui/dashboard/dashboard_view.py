"""The desktop live-run dashboard: a fixed, equal 2x2 grid.

    +----------------------+----------------------------+
    | Camera Feed          | Device States & Timer      |
    +----------------------+----------------------------+
    | Start, Stop,         | Manual Control             |
    | Metadata             |                            |
    +----------------------+----------------------------+

Panels are constructed by the caller (they need ``core``/main-window slots);
DashboardView only frames and places them, and owns the run header (elapsed
timer, state pill, REC) at the top of the Device States quadrant. The header
shows whatever text ``set_time`` is handed — RunControlPanel's
``elapsed_updated`` is the one source of that text.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from glider.gui.styles import colors

QUADRANT_TITLES = {
    "top_left": "Camera Feed",
    "top_right": "Device States & Timer",
    "bottom_left": "Start, Stop, Metadata",
    "bottom_right": "Manual Control",
}


class DashboardView(QWidget):
    """Fixed 2x2 grid: camera, device states + timer, run control, manual control."""

    def __init__(
        self,
        camera: QWidget,
        device_states: QWidget,
        run_control: QWidget,
        manual_controls: QWidget,
        parent=None,
    ):
        super().__init__(parent)
        self._panels = (camera, device_states, run_control, manual_controls)
        self._quadrants: dict[str, QFrame] = {}

        grid = QGridLayout(self)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)
        for row in (0, 1):
            grid.setRowStretch(row, 1)
        for col in (0, 1):
            grid.setColumnStretch(col, 1)

        top_right = QWidget()
        tr_layout = QVBoxLayout(top_right)
        tr_layout.setContentsMargins(0, 0, 0, 0)
        tr_layout.setSpacing(0)
        tr_layout.addWidget(self._build_header())
        tr_layout.addWidget(device_states, 1)

        for key, widget, row, col in (
            ("top_left", camera, 0, 0),
            ("top_right", top_right, 0, 1),
            ("bottom_left", run_control, 1, 0),
            ("bottom_right", manual_controls, 1, 1),
        ):
            frame = self._frame(QUADRANT_TITLES[key], widget)
            self._quadrants[key] = frame
            grid.addWidget(frame, row, col)

    def _frame(self, title: str, widget: QWidget) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        label = QLabel(title)
        label.setProperty("textRole", "section")
        label.setContentsMargins(10, 6, 10, 2)
        layout.addWidget(label)
        # Panels can be taller than a quadrant; let the grid, not them, decide.
        widget.setMinimumHeight(0)
        layout.addWidget(widget, 1)
        return frame

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setProperty("runnerHeader", True)
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        row = QHBoxLayout(header)
        row.setContentsMargins(12, 4, 12, 4)
        row.setSpacing(10)

        self._timer = QLabel("00:00.00")
        self._timer.setProperty("timer", True)
        self._timer.setStyleSheet(
            f"color: {colors.SUCCESS}; font-size: 36px; font-weight: bold; font-family: monospace;"
        )
        row.addWidget(self._timer)
        row.addStretch(1)

        self._rec = QLabel("● REC")
        self._rec.setProperty("recording", True)
        self._rec.hide()
        row.addWidget(self._rec)

        self._state = QLabel("IDLE")
        self._state.setProperty("runnerStatus", True)
        self._state.setProperty("statusState", "IDLE")
        row.addWidget(self._state)
        return header

    # --- public API ---

    def quadrant(self, key: str) -> QFrame:
        """The framed quadrant (``top_left`` ... ``bottom_right``)."""
        return self._quadrants[key]

    def set_time(self, text: str) -> None:
        self._timer.setText(text)

    def update_state(self, state_name: str, recording: bool = False) -> None:
        self._state.setText(state_name)
        self._state.setProperty("statusState", state_name)
        self._state.style().unpolish(self._state)
        self._state.style().polish(self._state)
        self._rec.setVisible(state_name == "RUNNING" and recording)
        for widget in self._panels:
            if hasattr(widget, "update_state"):
                widget.update_state(state_name)
