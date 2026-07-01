"""Tests for the per-clip trim editor (timeline) widget.

The pure geometry/clamping helpers are tested without a Qt event loop;
the widget-level behavior uses the shared QApplication pattern.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Pure helpers — no Qt required
# ---------------------------------------------------------------------------


def test_compute_window_pads_both_sides():
    from glider.gui.behavior.annotator.trim_bar import compute_window

    assert compute_window(480, 500, pad=30) == (450, 530)


def test_compute_window_clamps_low_to_zero():
    from glider.gui.behavior.annotator.trim_bar import compute_window

    assert compute_window(10, 30, pad=30) == (0, 60)


def test_compute_window_clamps_high_to_n_frames_when_known():
    from glider.gui.behavior.annotator.trim_bar import compute_window

    assert compute_window(480, 500, pad=30, n_frames=510) == (450, 510)


def test_compute_window_unknown_n_frames_leaves_high_unclamped():
    from glider.gui.behavior.annotator.trim_bar import compute_window

    assert compute_window(480, 500, pad=30, n_frames=None) == (450, 530)


def test_clamp_trim_bounds_within_window_unchanged():
    from glider.gui.behavior.annotator.trim_bar import clamp_trim_bounds

    assert clamp_trim_bounds(472, 508, 450, 530) == (472, 508)


def test_clamp_trim_bounds_clips_to_window_edges():
    from glider.gui.behavior.annotator.trim_bar import clamp_trim_bounds

    assert clamp_trim_bounds(440, 460, 450, 530) == (450, 460)
    assert clamp_trim_bounds(500, 560, 450, 530) == (500, 530)


def test_clamp_trim_bounds_enforces_in_before_out():
    from glider.gui.behavior.annotator.trim_bar import clamp_trim_bounds

    in_f, out_f = clamp_trim_bounds(508, 472, 450, 530)
    assert in_f < out_f
    assert out_f - in_f >= 1


# ---------------------------------------------------------------------------
# TrimBar widget — needs a QApplication
# ---------------------------------------------------------------------------


_APP = None  # module-global so the QApplication wrapper isn't GC'd mid-test


def _app():
    global _APP
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _make_bar():
    _app()
    from glider.gui.behavior.annotator.trim_bar import TrimBar

    bar = TrimBar()
    bar.set_window(450, 530)
    bar.set_bounds(472, 508)
    return bar


def test_trimbar_reports_bounds():
    bar = _make_bar()
    assert bar.bounds() == (472, 508)


def test_trimbar_set_bounds_clamps_to_window():
    bar = _make_bar()
    bar.set_bounds(400, 999)  # both outside the [450, 530] window
    assert bar.bounds() == (450, 530)


def test_trimbar_nudge_handles():
    bar = _make_bar()
    bar.nudge_in(+3)
    bar.nudge_out(-4)
    assert bar.bounds() == (475, 504)


def test_trimbar_nudge_in_cannot_cross_out():
    bar = _make_bar()
    bar.set_bounds(500, 502)
    bar.nudge_in(+50)  # would shoot past out
    in_f, out_f = bar.bounds()
    assert in_f < out_f


def test_trimbar_emits_bounds_changed():
    bar = _make_bar()
    seen = []
    bar.bounds_changed.connect(lambda i, o: seen.append((i, o)))
    bar.nudge_out(-2)
    assert seen and seen[-1] == bar.bounds()
