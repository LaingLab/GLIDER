"""Speed trace drawn under the trim bar, with the cohort's cut-offs on it.

A labeller deciding "is this a dart?" is judging speed from a looping video
with nothing to compare against, while the scoring run applies a numeric
cut-off the labeller cannot see. This puts the two side by side: the speed
over the same padded window the trim bar spans, the freeze/dart lines the
cohort file carries, and a playhead following the loop.

The shape is the point. A single number over a looping clip is jumpy and says
nothing about whether the animal accelerated into the window or was already
moving when it opened.

Frame-to-pixel mapping is imported from :mod:`trim_bar` rather than
reimplemented, so the two stacked timelines cannot drift apart.

**Units are refused rather than guessed.** Threshold lines are drawn only when
the cut-offs and the trace are in the same unit. A cm/s cohort threshold over
an uncalibrated px/frame trace would put the lines at arbitrary heights while
looking entirely convincing, so in that case the lines are dropped and the
status line says why.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from glider.gui.behavior.annotator.speed_source import SessionSpeed
from glider.gui.behavior.annotator.trim_bar import TRACK_MARGIN, frame_to_x, track_width

# Vertical padding inside the widget, leaving room for the axis labels.
_PAD_TOP = 14
_PAD_BOTTOM = 6

_COLOR_TRACE = QColor("#2563eb")
_COLOR_DART = QColor("#dc2626")
_COLOR_FREEZE = QColor("#0891b2")
_COLOR_PLAYHEAD = QColor("#111827")
_COLOR_TEXT = QColor("#6b7280")
_COLOR_BG = QColor("#f8fafc")


class SpeedTrace(QWidget):
    """Per-frame speed over the trim window, with optional threshold lines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._session: SessionSpeed | None = None
        self._win_start = 0
        self._win_end = 1
        self._playhead: int | None = None
        self._state = "absent"  # absent | loading | ready | failed
        self._error = ""
        # Thresholds as supplied, plus the unit they are expressed in. They
        # are only *applied* when that unit matches the trace's own.
        self._freeze: float | None = None
        self._dart: float | None = None
        self._threshold_unit: str = ""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def set_session(self, session: SessionSpeed | None) -> None:
        self._session = session
        self._state = "ready" if session is not None else "absent"
        self._error = ""
        self.update()

    def set_loading(self) -> None:
        self._state = "loading"
        self.update()

    def set_failed(self, reason: str) -> None:
        self._state = "failed"
        self._error = str(reason)
        self._session = None
        self.update()

    def set_window(self, win_start: int, win_end: int) -> None:
        self._win_start = int(win_start)
        self._win_end = max(int(win_end), self._win_start + 1)
        self.update()

    def set_playhead(self, frame: int | None) -> None:
        self._playhead = None if frame is None else int(frame)
        self.update()

    def set_thresholds(self, freeze: float | None, dart: float | None, unit: str = "") -> None:
        self._freeze = None if freeze is None else float(freeze)
        self._dart = None if dart is None else float(dart)
        self._threshold_unit = str(unit or "")
        self.update()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def has_data(self) -> bool:
        return self._state == "ready" and self._session is not None

    def unit(self) -> str:
        return self._session.unit if self._session is not None else ""

    def thresholds(self) -> tuple[float | None, float | None]:
        """The cut-offs actually drawable, i.e. in the trace's own unit."""
        if self._session is None or not self._threshold_unit:
            return (None, None)
        if self._threshold_unit != self._session.unit:
            return (None, None)
        return (self._freeze, self._dart)

    def value_at_playhead(self) -> float:
        if self._session is None or self._playhead is None:
            return float("nan")
        return self._session.at(self._playhead)

    def status_text(self) -> str:
        """One line explaining whatever the widget is currently not showing."""
        if self._state == "loading":
            return "loading pose data…"
        if self._state == "failed":
            return f"no speed trace: {self._error}"
        if self._state == "absent" or self._session is None:
            return "no pose data for this video"
        if self._threshold_unit and self._threshold_unit != self._session.unit:
            return (
                f"thresholds are in {self._threshold_unit} but this trace is in "
                f"{self._session.unit} — choose a calibration to place them"
            )
        return ""

    def readout_text(self) -> str:
        """The live number, and what it implies where that is knowable."""
        if not self.has_data():
            return ""
        value = self.value_at_playhead()
        if np.isnan(value):
            return f"—  {self.unit()}"
        text = f"{value:.3g} {self.unit()}"
        freeze, dart = self.thresholds()
        if dart is not None and value > dart:
            return f"{text}  (darting)"
        if freeze is not None and value < freeze:
            return f"{text}  (freezing)"
        return text

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _frame_to_x(self, frame: int) -> int:
        return frame_to_x(frame, self._win_start, self._win_end, self.width())

    def _plot_top(self) -> int:
        return _PAD_TOP

    def _plot_bottom(self) -> int:
        return max(self._plot_top() + 1, self.height() - _PAD_BOTTOM)

    def _y_range(self, samples: np.ndarray) -> tuple[float, float]:
        """``(lo, hi)`` for the vertical axis, always a non-zero span.

        The axis includes any drawable threshold, so a clip that never
        approaches the dart line still shows where that line is rather than
        rescaling it off the top.
        """
        finite = samples[np.isfinite(samples)] if samples.size else samples
        values = [0.0]
        if finite.size:
            values.extend([float(finite.min()), float(finite.max())])
        for t in self.thresholds():
            if t is not None:
                values.append(float(t))
        lo, hi = min(values), max(values)
        if hi <= lo:
            hi = lo + 1.0  # a flat trace still needs a span to divide by
        return lo, hi * 1.05

    def _value_to_y(self, value: float, lo: float, hi: float) -> int:
        top, bottom = self._plot_top(), self._plot_bottom()
        frac = (value - lo) / (hi - lo)
        return int(bottom - frac * (bottom - top))

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), _COLOR_BG)

        font = QFont(self.font())
        font.setPointSize(8)
        p.setFont(font)

        if not self.has_data():
            p.setPen(_COLOR_TEXT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.status_text() or "no data")
            p.end()
            return

        samples = self._session.window(self._win_start, self._win_end)
        lo, hi = self._y_range(samples)

        self._paint_thresholds(p, lo, hi)
        self._paint_trace(p, samples, lo, hi)
        self._paint_playhead(p)

        p.setPen(_COLOR_TEXT)
        p.drawText(TRACK_MARGIN, 10, f"speed ({self.unit()})")
        readout = self.readout_text()
        if readout:
            p.drawText(
                self.rect().adjusted(0, 0, -TRACK_MARGIN, 0),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                readout,
            )
        note = self.status_text()
        if note:
            p.drawText(
                self.rect().adjusted(TRACK_MARGIN, 0, -TRACK_MARGIN, -_PAD_BOTTOM),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                note,
            )
        p.end()

    def _paint_thresholds(self, p: QPainter, lo: float, hi: float) -> None:
        freeze, dart = self.thresholds()
        x0 = TRACK_MARGIN
        x1 = TRACK_MARGIN + track_width(self.width())
        for value, color, name in ((dart, _COLOR_DART, "dart"), (freeze, _COLOR_FREEZE, "freeze")):
            if value is None:
                continue
            y = self._value_to_y(value, lo, hi)
            pen = QPen(color)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(x0, y, x1, y)
            p.setPen(color)
            p.drawText(x0 + 2, max(8, y - 2), name)

    def _paint_trace(self, p: QPainter, samples: np.ndarray, lo: float, hi: float) -> None:
        pen = QPen(_COLOR_TRACE)
        pen.setWidth(2)
        p.setPen(pen)
        # Dropouts break the line rather than being interpolated across: a
        # straight segment over missing frames is an invented measurement.
        prev: tuple[int, int] | None = None
        for i, value in enumerate(samples):
            if not np.isfinite(value):
                prev = None
                continue
            point = (self._frame_to_x(self._win_start + i), self._value_to_y(value, lo, hi))
            if prev is not None:
                p.drawLine(prev[0], prev[1], point[0], point[1])
            prev = point

    def _paint_playhead(self, p: QPainter) -> None:
        if self._playhead is None:
            return
        if not (self._win_start <= self._playhead <= self._win_end):
            return
        x = self._frame_to_x(self._playhead)
        p.setPen(QPen(_COLOR_PLAYHEAD, 1))
        p.drawLine(x, self._plot_top(), x, self._plot_bottom())
