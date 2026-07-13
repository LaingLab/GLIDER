"""Tests for FpsMeter — rolling frames-per-second from cumulative counts."""

from glider.gui.panels.fps_meter import FpsMeter


def test_first_update_has_no_rate():
    """The first sample only establishes a baseline; no rate yet."""
    m = FpsMeter()
    assert m.update(count=0, now=100.0) is None


def test_rate_from_two_samples():
    """FPS is (Δframes / Δseconds) between consecutive samples."""
    m = FpsMeter()
    m.update(count=0, now=100.0)
    assert m.update(count=15, now=101.0) == 15.0  # 15 frames in 1.0s
    assert m.update(count=20, now=101.5) == 10.0  # 5 frames in 0.5s


def test_non_positive_dt_returns_none():
    """A zero/negative time delta yields no rate (avoids div-by-zero)."""
    m = FpsMeter()
    m.update(count=0, now=100.0)
    assert m.update(count=5, now=100.0) is None


def test_reset_reestablishes_baseline():
    """After reset, the next update is a baseline again."""
    m = FpsMeter()
    m.update(count=10, now=100.0)
    m.update(count=20, now=101.0)
    m.reset(now=200.0)
    assert m.update(count=100, now=201.0) == 100.0  # counts measured from reset baseline
