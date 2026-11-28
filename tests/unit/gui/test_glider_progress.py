"""The running-mouse progress bar: it must stay a working QProgressBar."""

from __future__ import annotations

import pytest

from glider.gui.widgets.glider_progress import WHEEL_FADE_TICKS, GliderProgress


@pytest.fixture
def bar(qtbot):
    widget = GliderProgress()
    widget.resize(500, 64)
    qtbot.addWidget(widget)
    return widget


def _run(bar, ticks=120):
    for _ in range(ticks):
        bar._tick()


@pytest.mark.parametrize("busy", [True, False])
def test_paints_in_both_modes(bar, busy):
    bar.setRange(0, 0) if busy else (bar.setRange(0, 10), bar.setValue(4))
    _run(bar, 5)
    assert not bar.grab().isNull()


def test_the_mouse_eases_to_the_reported_progress(bar):
    bar.setRange(0, 10)
    bar.setValue(4)
    bar._tick()
    assert 0 < bar._shown < 0.4  # no jump
    _run(bar)
    assert bar._shown == pytest.approx(0.4, abs=0.01)


def test_wheel_until_progress_then_track_and_back(bar):
    bar.setRange(0, 0)
    _run(bar, WHEEL_FADE_TICKS)
    assert bar._wheel == 1.0
    bar.setRange(0, 5)
    bar.setValue(5)
    _run(bar)
    assert bar._wheel == 0.0
    assert bar._shown == pytest.approx(1.0, abs=0.01)
    bar.setRange(0, 0)  # a new run
    bar._tick()
    assert bar._shown == 0.0


@pytest.mark.parametrize(
    ("lo", "hi", "value", "expected"),
    [(0, 0, 0, 0.0), (0, 1, 1, 1.0), (5, 5, 5, 1.0), (0, 4, -1, 0.0)],
)
def test_fraction_edges(bar, lo, hi, value, expected):
    bar.setRange(lo, hi)
    bar.setValue(value)
    assert bar.fraction() == expected


def test_reset_reads_as_the_start(bar):
    bar.setRange(0, 4)
    bar.setValue(2)
    bar.reset()
    assert bar.fraction() == 0.0


def test_the_callers_format_is_kept(bar):
    bar.setRange(0, 5)
    bar.setValue(2)
    bar.setFormat("%v / %m videos")
    assert bar.text() == "2 / 5 videos"


def test_animates_only_while_visible(bar):
    assert not bar._timer.isActive()
    bar.show()
    assert bar._timer.isActive()
    bar.hide()
    assert not bar._timer.isActive()
