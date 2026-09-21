"""Visible-window arithmetic for the review timeline. Qt-free.

The timeline's axis is whatever it draws on -- flow-relative milliseconds when
the session has a frame map, frames otherwise -- so nothing here knows a unit.
Callers pass ``min_span`` in the same unit (30 frames' worth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Viewport", "format_seconds", "format_timecode", "snap", "tick_spacing"]


@dataclass
class Viewport:
    """Which stretch of the session axis is on screen."""

    lo: float
    hi: float
    min_span: float
    start: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            self.lo, self.hi = self.hi, self.lo
        self.min_span = max(0.0, min(float(self.min_span), self.hi - self.lo))
        if self.end <= self.start:
            self.fit()

    @property
    def span(self) -> float:
        return self.end - self.start

    def fit(self) -> None:
        self.start, self.end = float(self.lo), float(self.hi)

    def _place(self, start: float, span: float) -> None:
        span = min(max(span, self.min_span), self.hi - self.lo)
        start = max(self.lo, min(start, self.hi - span))
        self.start, self.end = start, start + span

    def show(self, start: float, end: float) -> None:
        """Show ``[start, end]``, clamped to the session and the minimum span."""
        start, end = sorted((float(start), float(end)))
        span = min(max(end - start, self.min_span), self.hi - self.lo)
        self._place((start + end) / 2 - span / 2, span)

    def zoom(self, factor: float, about: float) -> None:
        """Scale the span by ``factor`` (< 1 zooms in), keeping ``about`` fixed on screen."""
        about = min(max(float(about), self.start), self.end)
        fraction = 0.0 if self.span <= 0 else (about - self.start) / self.span
        span = min(max(self.span * factor, self.min_span), self.hi - self.lo)
        self._place(about - fraction * span, span)

    def pan(self, delta: float) -> None:
        self._place(self.start + float(delta), self.span)

    def follow(self, t: float) -> None:
        """Page so ``t`` is on screen, making it the left edge -- Resolve's page-follow."""
        if not self.start <= t <= self.end:
            self.pan(float(t) - self.start)

    def x_of(self, t: float, width: float) -> float:
        return 0.0 if self.span <= 0 else (float(t) - self.start) / self.span * width

    def t_at(self, x: float, width: float) -> float:
        return self.start if width <= 0 else self.start + float(x) / width * self.span


def snap(t: float, candidates, tolerance: float) -> float:
    """The candidate nearest ``t`` within ``tolerance``, else ``t``.

    ``candidates`` must be sorted ascending.
    """
    values = np.asarray(candidates, dtype=float)
    if values.size == 0:
        return float(t)
    i = int(np.searchsorted(values, t))
    near = [values[j] for j in (i - 1, i) if 0 <= j < values.size]
    best = min(near, key=lambda v: abs(v - t))
    return float(best) if abs(best - t) <= tolerance else float(t)


def format_timecode(seconds: float, fps: float) -> str:
    """``HH:MM:SS:FF`` -- FF is the frame within the second -- with a sign before 0."""
    sign = "-" if seconds < 0 else ""
    s = abs(float(seconds))
    whole = int(math.floor(s + 1e-9))
    frames = 0
    if fps > 0:
        frames = min(
            int(math.floor((s - whole) * fps + 1e-6)),
            max(0, math.ceil(fps) - 1),
        )
    return f"{sign}{whole // 3600:02d}:{whole // 60 % 60:02d}:{whole % 60:02d}:{frames:02d}"


def format_seconds(seconds: float) -> str:
    """``m:ss.ss`` for durations and read-outs."""
    sign = "-" if seconds < 0 else ""
    minutes, centis = divmod(round(abs(float(seconds)) * 100), 6000)
    return f"{sign}{minutes}:{centis / 100:05.2f}"


def tick_spacing(span_s: float) -> tuple[int, int]:
    """(major, minor) ruler tick spacing in seconds for a visible span."""
    if span_s <= 90:
        return 10, 1
    if span_s <= 600:
        return 30, 5
    if span_s <= 3600:
        return 60, 10
    return 600, 60
