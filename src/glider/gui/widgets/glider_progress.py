"""A progress bar with a running mouse: in a wheel until progress is known, then along a track.

A ``QProgressBar`` subclass, so callers keep ``setRange(0, 0)`` for "busy",
``setRange``/``setValue``/``setFormat`` for progress, and nothing else changes.
Only the painting is different. The mouse is drawn from shapes (the spike in
``tools/splash-animation`` was the mockup), so it stays crisp at any size and
can follow real progress, which a pre-rendered loop could not.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QProgressBar, QSizePolicy

from glider.gui.styles import colors

__all__ = ["GliderProgress"]

FRAME_MS = 33
STRIDE_FRAMES = 8  # ticks per full stride
EASE = 0.12  # fraction of the remaining distance covered per tick
WHEEL_FADE_TICKS = 9  # ~300 ms
HEIGHT = 64

_FG = QColor(colors.TEXT_PRIMARY)
_FAR = QColor(colors.TEXT_MUTED)
_DIM = QColor(colors.TEXT_DISABLED)
_EYE = QColor(colors.BASE)
_NOSE = QColor("#c98a8a")
_TRACK = QColor(colors.BORDER)
_FILL = QColor(colors.ACCENT)

# Mouse model units: nose at +x, feet at about +42, ears at about -40.
_FEET = 42.0
_TAIL = 140.0
_NOSE_X = 112.0


class GliderProgress(QProgressBar):
    """Drop-in ``QProgressBar`` that paints a running mouse."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._phase = 0  # stride tick count
        self._shown = 0.0  # eased track position, 0..1
        self._wheel = 1.0 if self._busy() else 0.0  # wheel opacity, 0..1
        self._timer = QTimer(self)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(self._tick)

    # -- state --------------------------------------------------------------

    def _busy(self) -> bool:
        return self.minimum() == 0 and self.maximum() == 0

    def fraction(self) -> float:
        """Where the mouse is heading, 0..1. Busy (0, 0) is 0."""
        lo, hi = self.minimum(), self.maximum()
        if hi <= lo:
            return 0.0 if self._busy() else 1.0
        return min(1.0, max(0.0, (self.value() - lo) / (hi - lo)))

    def _arrived(self) -> bool:
        return not self._busy() and self.fraction() >= 1.0 and self._shown > 0.995

    def _tick(self) -> None:
        step = 1.0 / WHEEL_FADE_TICKS
        if self._busy():
            self._wheel = min(1.0, self._wheel + step)
            self._shown = 0.0  # a new busy spell starts the next run from the line
        else:
            self._wheel = max(0.0, self._wheel - step)
            self._shown += (self.fraction() - self._shown) * EASE
        if not self._arrived():
            self._phase += 1
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().hideEvent(event)
        self._timer.stop()

    # -- painting -----------------------------------------------------------

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        pad = 8.0
        track_s = 0.36  # mouse scale on the track
        ground = h - 10.0

        label = self.text() if self.isTextVisible() and not self._busy() else ""
        font = QFont(self.font())
        font.setPointSizeF(max(font.pointSizeF(), 10.0))
        p.setFont(font)
        label_w = p.fontMetrics().horizontalAdvance(label) + 12 if label else 0

        left = pad + _TAIL * track_s * 0.6
        right = max(left + 1, w - pad - label_w - _NOSE_X * track_s)
        x_track = left + (right - left) * self._shown

        # track and the distance flown
        p.setPen(QPen(_TRACK, 3, cap=Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(left, ground), QPointF(right, ground))
        if self._shown > 0.002:
            p.setPen(QPen(_FILL, 3, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(left, ground), QPointF(x_track, ground))
        if label:
            p.setPen(_FG)
            p.drawText(
                QRectF(w - pad - label_w, 0, label_w, ground + 6),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                label,
            )

        # the wheel, fading out once progress arrives
        r = (h - 10) / 2.0
        wc = QPointF(left, h / 2.0)
        wheel_s = 0.2
        if self._wheel > 0:
            p.save()
            p.setOpacity(self._wheel)
            p.setPen(QPen(_DIM, 2))
            spin = -self._phase * 9.0
            for i in range(8):
                a = math.radians(spin + i * 45)
                p.drawLine(wc, wc + QPointF(math.cos(a) * r, math.sin(a) * r))
            p.setPen(QPen(_FAR, 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(wc, r, r)
            p.restore()

        # the mouse: wheel position blends into track position as the wheel fades
        t = self._wheel
        s = wheel_s * t + track_s * (1 - t)
        mx = wc.x() * t + x_track * (1 - t)
        my = (wc.y() + r - _FEET * wheel_s - 1) * t + (ground - _FEET * track_s - 1) * (1 - t)
        stride = 0 if self._arrived() else self._phase
        _draw_mouse(p, mx, my, s, stride)
        p.end()


def _leg(p: QPainter, x: float, y: float, phase: float, color: QColor) -> None:
    # Thigh swings; shin folds back on the return so the foot lifts.
    a = math.sin(phase) * 0.7
    k = a - 0.5 - max(0.0, math.cos(phase)) * 0.9
    knee = QPointF(x + math.sin(a) * 16, y + math.cos(a) * 16)
    foot = knee + QPointF(math.sin(k) * 15, math.cos(k) * 15)
    pen = QPen(color, 7, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline([QPointF(x, y), knee, foot])


def _draw_mouse(p: QPainter, x: float, y: float, scale: float, tick: int) -> None:
    """Side-view mouse facing right, body centre at (x, y), in model units * scale."""
    ph = tick / STRIDE_FRAMES * 2 * math.pi
    bounce = -abs(math.sin(ph)) * 5
    sway = math.sin(ph * 0.5) * 6
    p.save()
    p.translate(x, y + bounce * scale)
    p.scale(scale, scale)

    _leg(p, -34, 10, ph + 0.5, _FAR)
    _leg(p, 30, 12, ph + math.pi + 0.5, _FAR)

    tail = QPainterPath(QPointF(-52, 0))
    tail.cubicTo(-85, -12 + sway, -105, 18 - sway, -140, 4 + sway)
    p.setPen(QPen(_FG, 4, cap=Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(tail)

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_FG)
    p.drawEllipse(QPointF(-4, -2), 54, 28)
    head = QPainterPath(QPointF(28, -24))
    head.quadTo(70, -24, 92, 2)
    head.quadTo(70, 14, 30, 18)
    head.closeSubpath()
    p.drawPath(head)
    p.drawEllipse(QPointF(44, -26), 14, 14)
    p.setBrush(_DIM)
    p.drawEllipse(QPointF(44, -26), 7, 7)
    p.setBrush(_EYE)
    p.drawEllipse(QPointF(66, -6), 4, 4)
    p.setBrush(_NOSE)
    p.drawEllipse(QPointF(92, 2), 4, 4)
    p.setPen(QPen(_FAR, 1.5, cap=Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(86, 4), QPointF(112, -2))
    p.drawLine(QPointF(86, 6), QPointF(112, 8))

    _leg(p, -34, 10, ph, _FG)
    _leg(p, 30, 12, ph + math.pi, _FG)
    p.restore()
