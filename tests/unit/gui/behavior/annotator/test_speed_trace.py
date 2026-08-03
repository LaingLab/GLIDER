"""The speed trace widget drawn under the trim bar."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from glider.gui.behavior.annotator.speed_source import SessionSpeed  # noqa: E402
from glider.gui.behavior.annotator.speed_trace import SpeedTrace  # noqa: E402
from glider.gui.behavior.annotator.trim_bar import TrimBar, frame_to_x  # noqa: E402


def _session(values, fps=30.0, px_per_mm=None):
    return SessionSpeed(px_per_frame=np.asarray(values, dtype=float), fps=fps, px_per_mm=px_per_mm)


# ---------------------------------------------------------------------------
# Alignment with the trim bar
# ---------------------------------------------------------------------------


def test_trim_bar_and_trace_map_frames_to_the_same_pixel(qtbot):
    """Two stacked timelines that disagree by a few pixels are worse than one."""
    bar = TrimBar()
    qtbot.addWidget(bar)
    bar.resize(400, 44)
    bar.set_window(100, 200)

    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.resize(400, 70)
    trace.set_window(100, 200)

    for frame in (100, 137, 199, 200):
        assert bar._frame_to_x(frame) == trace._frame_to_x(frame)


def test_frame_to_x_helper_matches_the_widget(qtbot):
    bar = TrimBar()
    qtbot.addWidget(bar)
    bar.resize(321, 44)
    bar.set_window(10, 60)
    assert bar._frame_to_x(35) == frame_to_x(35, 10, 60, 321)


# ---------------------------------------------------------------------------
# What it shows
# ---------------------------------------------------------------------------


def test_starts_with_no_data_and_says_so(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    assert trace.has_data() is False
    assert "no pose" in trace.status_text().lower()


def test_reports_loading_while_the_csv_is_being_parsed(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_loading()
    assert trace.has_data() is False
    assert "loading" in trace.status_text().lower()


def test_shows_the_failure_reason_rather_than_an_empty_box(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_failed("could not parse pose.csv")
    assert "could not parse" in trace.status_text()


def test_holds_the_session_it_is_given(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 1.0, 2.0, 3.0]))
    trace.set_window(0, 4)
    assert trace.has_data() is True


def test_reads_the_speed_at_the_playhead(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 5.0, 9.0]))
    trace.set_window(0, 3)
    trace.set_playhead(1)
    assert trace.value_at_playhead() == pytest.approx(5.0)


def test_playhead_value_is_nan_off_the_end_of_the_recording(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 5.0]))
    trace.set_window(0, 10)
    trace.set_playhead(7)
    assert np.isnan(trace.value_at_playhead())


def test_readout_reports_the_unit(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 1.0], fps=30.0, px_per_mm=2.0))
    trace.set_window(0, 2)
    trace.set_playhead(1)
    assert "cm/s" in trace.readout_text()


def test_readout_reports_pixels_when_uncalibrated(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 4.0]))
    trace.set_window(0, 2)
    trace.set_playhead(1)
    assert "px/frame" in trace.readout_text()


# ---------------------------------------------------------------------------
# Threshold lines
# ---------------------------------------------------------------------------


def test_thresholds_are_absent_until_given(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    assert trace.thresholds() == (None, None)


def test_thresholds_apply_when_units_agree(qtbot):
    """A cm/s cohort file against a calibrated cm/s trace: lines are drawable."""
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 1.0], fps=30.0, px_per_mm=2.0))
    trace.set_thresholds(freeze=0.5, dart=27.7, unit="cm/s")
    assert trace.thresholds() == (0.5, 27.7)


def test_thresholds_are_dropped_when_the_trace_is_in_other_units(qtbot):
    """cm/s cut-offs over a px/frame trace would be lines in the wrong place."""
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 1.0]))  # uncalibrated -> px/frame
    trace.set_thresholds(freeze=0.5, dart=27.7, unit="cm/s")
    assert trace.thresholds() == (None, None)
    assert "calibration" in trace.status_text().lower()


def test_readout_names_the_behavior_the_speed_implies(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 40.0], fps=30.0, px_per_mm=2.0))
    trace.set_thresholds(freeze=0.5, dart=10.0, unit="cm/s")
    trace.set_window(0, 2)
    trace.set_playhead(1)
    assert "darting" in trace.readout_text()


def test_readout_says_freezing_below_the_freeze_line(qtbot):
    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.set_session(_session([0.0, 0.0], fps=30.0, px_per_mm=2.0))
    trace.set_thresholds(freeze=0.5, dart=10.0, unit="cm/s")
    trace.set_window(0, 2)
    trace.set_playhead(1)
    assert "freezing" in trace.readout_text()


# ---------------------------------------------------------------------------
# Painting must not raise -- offscreen render of each state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prepare",
    [
        lambda t: None,
        lambda t: t.set_loading(),
        lambda t: t.set_failed("boom"),
        lambda t: (t.set_session(_session([0.0, 1.0, 8.0, 2.0])), t.set_window(0, 4)),
        lambda t: (
            t.set_session(_session([0.0, np.nan, 8.0, 2.0])),
            t.set_window(0, 4),
            t.set_playhead(2),
        ),
        # A window entirely past the recording: every sample is NaN.
        lambda t: (t.set_session(_session([1.0, 2.0])), t.set_window(500, 540)),
        # A flat trace -- zero range, which is where a naive y-scale divides by nothing.
        lambda t: (t.set_session(_session([3.0, 3.0, 3.0])), t.set_window(0, 3)),
    ],
)
def test_paints_without_raising(qtbot, prepare):
    from PyQt6.QtGui import QPixmap

    trace = SpeedTrace()
    qtbot.addWidget(trace)
    trace.resize(400, 70)
    prepare(trace)
    trace.render(QPixmap(trace.size()))
