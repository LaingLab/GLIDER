"""
Touch-friendly number pad dialog.

A large-button numeric entry for the Runner touchscreen. Returns an int via the
``get_int`` classmethod::

    n = NumberPadDialog.get_int("Revolutions", value=1, minimum=1, maximum=1000)
    if n is not None:
        ...
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class NumberPadDialog(QDialog):
    """Big-button integer entry for touchscreens."""

    def __init__(self, title, value=0, minimum=0, maximum=1_000_000, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(280, 380)
        self._min = int(minimum)
        self._max = int(maximum)
        self._text = str(self._clamp(int(value)))

        layout = QVBoxLayout(self)

        prompt = QLabel(title)
        prompt.setProperty("textRole", "muted")
        layout.addWidget(prompt)

        self._display = QLabel(self._text)
        self._display.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._display.setMinimumHeight(56)
        self._display.setStyleSheet("font-size: 28px; padding: 4px 12px;")
        layout.addWidget(self._display)

        grid = QGridLayout()
        grid.setSpacing(6)
        buttons = [
            ("7", 0, 0),
            ("8", 0, 1),
            ("9", 0, 2),
            ("4", 1, 0),
            ("5", 1, 1),
            ("6", 1, 2),
            ("1", 2, 0),
            ("2", 2, 1),
            ("3", 2, 2),
            ("C", 3, 0),
            ("0", 3, 1),
            ("⌫", 3, 2),
        ]
        for label, r, c in buttons:
            btn = QPushButton(label)
            btn.setMinimumSize(72, 64)
            btn.setStyleSheet("font-size: 22px;")
            btn.clicked.connect(lambda _checked=False, key=label: self._on_key(key))
            grid.addWidget(btn, r, c)
        layout.addLayout(grid)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _clamp(self, n: int) -> int:
        return max(self._min, min(self._max, n))

    def _on_key(self, key: str) -> None:
        if key == "C":
            self._text = "0"
        elif key == "⌫":
            self._text = self._text[:-1] or "0"
        else:  # a digit
            self._text = (self._text + key).lstrip("0") or "0"
        self._display.setText(self._text)

    def value(self) -> int:
        return self._clamp(int(self._text or 0))

    @classmethod
    def get_int(cls, title, value=0, minimum=0, maximum=1_000_000, parent=None):
        """Show the pad; return the entered int, or None if cancelled."""
        dialog = cls(title, value=value, minimum=minimum, maximum=maximum, parent=parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.value()
        return None
