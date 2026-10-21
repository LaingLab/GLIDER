"""A one-line-or-two notice bar for the Pi's header.

Shows a level glyph, a short title and message at a level (``info`` |
``warning`` | ``critical``), styled by the ``level`` property, and a ✕. A tap
anywhere on it dismisses it. Info and warning take one line, elided, with the
whole text in the tooltip; a critical notice wraps to two lines, so more of an
error is readable before anything is cut. Either way the bar never grows past
two lines or widens the window. A lower level never replaces a visible
``critical`` notice: an "UNSAFE STATE" warning has to stay up until somebody has
seen it, not be buried by a "Saved".
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMouseEvent, QTextLayout, QTextOption
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton

from glider.gui.styles import restyle

_RANK = {"info": 0, "warning": 1, "critical": 2}
_GLYPH = {"info": "ⓘ", "warning": "⚠", "critical": "⚠"}


class NoticeBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("piNoticeBar")
        self.setProperty("level", "info")
        self._full = ""
        self._label = QLabel()
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        # A long unbreakable token must clip or elide, not widen the window.
        self._label.setMinimumWidth(1)
        self._close = QToolButton()
        self._close.setObjectName("piNoticeClose")
        self._close.setText("✕")
        self._close.setAccessibleName("Dismiss notice")
        self._close.setFixedSize(44, 44)
        self._close.clicked.connect(self.dismiss)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._label, 1)
        row.addWidget(self._close)
        self.hide()

    def show_notice(self, title: str, message: str, level: str) -> None:
        if self.property("level") == "critical" and not self.isHidden() and _RANK[level] < 2:
            return
        self._full = f"{title}: {message}"
        self._label.setToolTip(self._full)
        self.setProperty("level", level)
        # The label and ✕ too: touch.qss colours them by the bar's level, and Qt
        # re-resolves a descendant selector only when the descendant is polished.
        for widget in (self, self._label, self._close):
            restyle(widget)
        self._fit()
        self.show()

    def text(self) -> str:
        return self._full

    def dismiss(self) -> None:
        self.hide()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        """Lay the text out in one line, or two for critical, cut to the width."""
        level = self.property("level")
        label = self._label
        width = max(1, label.width())
        text = f"{_GLYPH[level]}  {' '.join(self._full.split())}"
        if level == "critical":
            layout = QTextLayout(text, label.font())
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            layout.setTextOption(option)
            layout.beginLayout()
            first = layout.createLine()
            first.setLineWidth(width)
            layout.endLayout()
            cut = first.textLength()
            if cut < len(text):
                tail = label.fontMetrics().elidedText(
                    text[cut:], Qt.TextElideMode.ElideRight, width
                )
                text = f"{text[:cut].rstrip()}\n{tail}"
        else:
            text = label.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, width)
        label.setText(text)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.MouseButton.LeftButton:
            self.dismiss()
        super().mouseReleaseEvent(event)
