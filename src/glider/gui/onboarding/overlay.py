"""Spotlight overlay for the interactive walkthrough.

A single translucent child widget laid over the whole main window. It paints a
dark scrim with a rounded "hole" cut out around the step's target widget (the
spotlight), draws a highlight border around the hole, and shows a small callout
card with the step text + Back / Skip / Next. Passive by design: the scrim
swallows clicks to the app behind it, so the user advances with the card's
buttons (or Esc to skip).

The overlay owns no step logic — :class:`~glider.gui.onboarding.tour.Tour`
drives it via :meth:`show_step` and the ``next_requested`` / ``back_requested``
/ ``skip_requested`` signals.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors

# Padding around the target rect for the cutout, and the corner radius.
_CUTOUT_PAD = 6
_CUTOUT_RADIUS = 8
# Gap between the spotlight and the callout card.
_CALLOUT_GAP = 14
_CALLOUT_WIDTH = 340


class TourOverlay(QWidget):
    """Full-window scrim + spotlight cutout + step callout."""

    next_requested = pyqtSignal()
    back_requested = pyqtSignal()
    skip_requested = pyqtSignal()

    def __init__(self, host: QWidget):
        super().__init__(host)
        self._host = host
        self._target_rect = QRect()  # empty => no cutout (centered step)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._build_callout()
        self._sync_geometry()
        host.installEventFilter(self)

    # --- Public API (driven by Tour) ---

    def show_step(
        self,
        target: QWidget | None,
        title: str,
        body: str,
        index: int,
        total: int,
    ) -> None:
        """Render one step: spotlight ``target`` and fill the callout."""
        self._target_rect = self._rect_for(target)
        self._title.setText(title)
        self._body.setText(body)
        self._counter.setText(f"Step {index + 1} of {total}")
        self._back_btn.setEnabled(index > 0)
        is_last = index == total - 1
        self._next_btn.setText("Finish" if is_last else "Next")
        self._skip_btn.setVisible(not is_last)

        self._sync_geometry()
        self._position_callout()
        self.raise_()
        self.show()
        self.setFocus()
        self.update()

    # --- Geometry ---

    def _rect_for(self, target: QWidget | None) -> QRect:
        """Target's rect mapped into overlay (== host) coordinates."""
        if target is None or not target.isVisible():
            return QRect()
        top_left = target.mapTo(self._host, QPoint(0, 0))
        return QRect(top_left, target.size())

    def _sync_geometry(self) -> None:
        self.setGeometry(self._host.rect())

    def _position_callout(self) -> None:
        card = self._callout
        card.adjustSize()
        w, h = card.width(), card.height()
        host = self.rect()

        if self._target_rect.isEmpty():
            # Centered step (welcome / finish).
            x = host.center().x() - w // 2
            y = host.center().y() - h // 2
        else:
            t = self._target_rect
            # Prefer below the spotlight; flip above if it would clip.
            if t.bottom() + _CALLOUT_GAP + h <= host.height():
                y = t.bottom() + _CALLOUT_GAP
            else:
                y = t.top() - _CALLOUT_GAP - h
            x = t.left()
        # Clamp inside the window with a small margin.
        x = max(12, min(x, host.width() - w - 12))
        y = max(12, min(y, host.height() - h - 12))
        card.move(x, y)

    def eventFilter(self, obj, event):  # noqa: N802 (Qt signature)
        from PyQt6.QtCore import QEvent

        if obj is self._host and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
        ):
            # Host resized/moved: the target moved with it, so re-sync.
            self._sync_geometry()
            self._position_callout()
            self.update()
        return super().eventFilter(obj, event)

    # --- Painting ---

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        scrim = QColor(0, 0, 0, 165)

        if self._target_rect.isEmpty():
            p.fillRect(self.rect(), scrim)
            return

        hole = QRectF(
            self._target_rect.adjusted(-_CUTOUT_PAD, -_CUTOUT_PAD, _CUTOUT_PAD, _CUTOUT_PAD)
        )
        path = QPainterPath()
        path.addRect(QRectF(self.rect()))
        cut = QPainterPath()
        cut.addRoundedRect(hole, _CUTOUT_RADIUS, _CUTOUT_RADIUS)
        p.fillPath(path.subtracted(cut), scrim)

        # Accent ring around the spotlight.
        p.setPen(QPen(colors.Q_ACCENT, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(hole, _CUTOUT_RADIUS, _CUTOUT_RADIUS)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.skip_requested.emit()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.next_requested.emit()
        else:
            super().keyPressEvent(event)

    # --- Callout card ---

    def _build_callout(self) -> None:
        card = QFrame(self)
        card.setObjectName("tourCallout")
        card.setFixedWidth(_CALLOUT_WIDTH)
        card.setStyleSheet(f"""
            QFrame#tourCallout {{
                background-color: {colors.SURFACE_2};
                border: 1px solid {colors.ACCENT};
                border-radius: 10px;
            }}
            """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self._counter = QLabel()
        self._counter.setStyleSheet(f"color: {colors.ACCENT}; font-size: 10px; font-weight: bold;")

        self._title = QLabel()
        self._title.setWordWrap(True)
        self._title.setStyleSheet(
            f"color: {colors.TEXT_PRIMARY}; font-size: 15px; font-weight: bold;"
        )

        self._body = QLabel()
        self._body.setWordWrap(True)
        self._body.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-size: 12px;")

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.setProperty("buttonRole", "secondary")
        self._skip_btn.clicked.connect(self.skip_requested)
        self._back_btn = QPushButton("Back")
        self._back_btn.setProperty("buttonRole", "secondary")
        self._back_btn.clicked.connect(self.back_requested)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self.next_requested)

        buttons.addWidget(self._skip_btn)
        buttons.addStretch()
        buttons.addWidget(self._back_btn)
        buttons.addWidget(self._next_btn)

        layout.addWidget(self._counter)
        layout.addWidget(self._title)
        layout.addWidget(self._body)
        layout.addLayout(buttons)

        self._callout = card
