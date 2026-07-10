"""Run banner - persistent slim strip shown during a live recorded run.

Displayed across both Runner pages (Dashboard + Manual) so the operator never
loses the timer/state/STOP when switching pages. This is a pure-UI widget: it
receives pre-formatted strings via setters and exposes a ``stop_requested``
signal. It owns no timer and drives nothing - callers push time/state updates
in. There is deliberately no emergency-stop button here; emergency stop stays
on the Dashboard.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class RunBanner(QWidget):
    """Slim persistent banner showing timer/state/REC and a STOP button."""

    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("runnerRunBanner")
        self.setProperty("runnerRunBanner", True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(48)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 12, 4)
        layout.setSpacing(10)

        self._time = QLabel("00:00.00")
        self._time.setProperty("timer", True)
        self._time.setStyleSheet("font-family: monospace; font-weight: bold; font-size: 18px;")
        layout.addWidget(self._time)

        self._state = QLabel("IDLE")
        self._state.setProperty("runnerStatus", True)
        self._state.setProperty("statusState", "IDLE")
        layout.addWidget(self._state)

        self._rec = QLabel("● REC")
        self._rec.setProperty("recording", True)
        self._rec.hide()
        layout.addWidget(self._rec)

        layout.addStretch()

        self._stop = QPushButton("■ STOP")
        self._stop.setProperty("runnerAction", "stop")
        self._stop.clicked.connect(self.stop_requested.emit)
        layout.addWidget(self._stop)

    def set_time(self, text: str) -> None:
        """Set the pre-formatted elapsed-time text (e.g. "01:23.45")."""
        self._time.setText(text)

    def set_state(self, name: str, recording: bool) -> None:
        """Update the state pill text and REC indicator visibility."""
        self._state.setText(name)
        self._state.setProperty("statusState", name)
        self._state.style().unpolish(self._state)
        self._state.style().polish(self._state)
        self._rec.setVisible(recording)
