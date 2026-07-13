"""FpsMeter — a rolling frames-per-second estimate from cumulative counts.

Given a monotonically increasing frame count sampled at arbitrary times (e.g.
a batch tracker's progress callback), report the processing rate between the
last two samples. Kept Qt-free and pure so it is trivially unit-testable.
"""

from __future__ import annotations


class FpsMeter:
    """Estimate FPS from successive (cumulative_count, timestamp) samples."""

    def __init__(self) -> None:
        self._last_count = 0
        self._last_t: float | None = None

    def reset(self, now: float) -> None:
        """Start a fresh measurement window anchored at ``now`` (count 0)."""
        self._last_count = 0
        self._last_t = now

    def update(self, count: int, now: float) -> float | None:
        """Record a sample; return FPS since the previous sample, or None.

        Returns ``None`` on the first sample of an un-reset meter (baseline
        only) and when the time delta is non-positive.
        """
        if self._last_t is None:
            self._last_count = count
            self._last_t = now
            return None
        dt = now - self._last_t
        if dt <= 0:
            return None
        fps = (count - self._last_count) / dt
        self._last_count = count
        self._last_t = now
        return fps
