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
_EAR = QColor("#cbd5e1")
_TRACK = QColor(colors.BORDER)
_FILL = QColor(colors.ACCENT)

# Mouse model units: nose at +x, feet at about +42, ears at about -40.
_FEET = 38.0
_TAIL = 126.0
_NOSE_X = 108.0


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


def _limb(
    p: QPainter,
    hip: QPointF,
    phase: float,
    upper: float,
    lower: float,
    width: float,
    color: QColor,
) -> None:
    """A two-joint leg with a little foot; the knee folds as the leg comes forward."""
    swing = math.sin(phase) * 0.75  # radians off vertical, + is forward
    fold = 0.3 + max(0.0, math.cos(phase)) * 1.1  # lifts the foot on the recovery
    knee = hip + QPointF(math.sin(swing) * upper, math.cos(swing) * upper)
    shin = swing - fold
    ankle = knee + QPointF(math.sin(shin) * lower, math.cos(shin) * lower)
    toe = ankle + QPointF(width * 1.3, 0)
    p.setPen(QPen(color, width, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPolyline([hip, knee, ankle, toe])


def _body() -> QPainterPath:
    """One silhouette, snout to rump, so the head and body don't read as parts."""
    path = QPainterPath(QPointF(96, 4))
    path.cubicTo(82, -10, 64, -22, 40, -26)  # snout up to the forehead
    path.cubicTo(10, -34, -30, -36, -52, -18)  # back arch
    path.cubicTo(-66, -6, -60, 16, -40, 22)  # rump
    path.cubicTo(-10, 28, 30, 24, 58, 16)  # belly
    path.cubicTo(74, 12, 88, 10, 96, 4)  # chin
    path.closeSubpath()
    return path


_BODY = _body()


def _draw_mouse(p: QPainter, x: float, y: float, scale: float, tick: int) -> None:
    """Side-view mouse facing right, body centre at (x, y), in model units * scale."""
    ph = tick / STRIDE_FRAMES * 2 * math.pi
    bounce = -abs(math.sin(ph)) * 4
    pitch = math.sin(ph) * 3  # nose dips as the hind legs push
    sway = math.sin(ph * 0.5) * 5
    p.save()
    p.translate(x, y + bounce * scale)
    p.scale(scale, scale)
    p.rotate(pitch)

    # far side first, dimmer: a gallop, hind pair together, fronts opposite
    _limb(p, QPointF(-34, 12), ph + 0.4, 14, 13, 7, _FAR)
    _limb(p, QPointF(42, 14), ph + math.pi + 0.4, 11, 10, 5, _FAR)

    tail = QPainterPath(QPointF(-56, 8))
    tail.cubicTo(-80, 16 + sway, -100, -2 - sway, -126, 6 + sway)
    p.setPen(QPen(_FG, 3, cap=Qt.PenCapStyle.RoundCap))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawPath(tail)

    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(_FG)
    p.drawPath(_BODY)

    # ear: a shade off the body, tilted back, its base sunk into the head
    p.save()
    p.translate(28, -26)
    p.rotate(-18)
    p.setBrush(_EAR)
    p.drawEllipse(QPointF(0, -4), 12, 15)
    p.setBrush(_NOSE)
    p.drawEllipse(QPointF(0, -5), 6, 9)
    p.restore()

    p.setBrush(_EYE)
    p.drawEllipse(QPointF(62, -8), 3.5, 3.5)
    p.setBrush(_NOSE)
    p.drawEllipse(QPointF(96, 3), 3.5, 3.5)
    p.setPen(QPen(_FAR, 1.2, cap=Qt.PenCapStyle.RoundCap))
    p.drawLine(QPointF(84, 2), QPointF(108, -4))
    p.drawLine(QPointF(84, 5), QPointF(108, 7))

    _limb(p, QPointF(-30, 14), ph, 14, 13, 7, _FG)
    _limb(p, QPointF(46, 14), ph + math.pi, 11, 10, 5, _FG)
    p.restore()
