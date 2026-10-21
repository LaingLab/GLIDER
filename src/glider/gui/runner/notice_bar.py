"""A one-line-or-two notice bar for the Pi's header.

Shows a short title and message at a level (``info`` | ``warning`` |
``critical``), styled by the ``level`` property. A tap dismisses it. A lower
level never replaces a visible ``critical`` notice: an "UNSAFE STATE" warning
has to stay up until somebody has seen it, not be buried by a "Saved".
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel

from glider.gui.styles import restyle

_RANK = {"info": 0, "warning": 1, "critical": 2}


class NoticeBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("piNoticeBar")
        self.setProperty("level", "info")
        self._label = QLabel()
        self._label.setWordWrap(True)
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        row = QHBoxLayout(self)
        row.addWidget(self._label)
        self.hide()

    def show_notice(self, title: str, message: str, level: str) -> None:
        if self.property("level") == "critical" and not self.isHidden() and _RANK[level] < 2:
            return
        self._label.setText(f"{title}: {message}")
        self.setProperty("level", level)
        # The label too: touch.qss colours it by the bar's level, and Qt
        # re-resolves a descendant selector only when the descendant is polished.
        restyle(self)
        restyle(self._label)
        self.show()

    def text(self) -> str:
        return self._label.text()

    def dismiss(self) -> None:
        self.hide()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.dismiss()
        super().mouseReleaseEvent(event)
