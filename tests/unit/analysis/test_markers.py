"""Review markers: the time rule, the files, and the stores. Qt-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from glider.analysis import Session
from glider.analysis.markers import frame_at, frames_in, seconds_at, t0_of, uses_ms
from glider.analysis.timeline import build_timeline

from .conftest import RecordingSpec, write_synthetic_recording

FPS = 30.0

# ---------------------------------------------------------------------------
# the time rule


def test_without_a_frame_map_seconds_are_frames_over_fps():
    assert seconds_at(None, FPS, 90) == pytest.approx(3.0)
    assert frame_at(None, FPS, 3.0) == 90
    assert frame_at(None, FPS, 3.01) == 91  # the first frame at or after
    assert t0_of(None) == "video_start"
    assert not uses_ms(None)


def test_a_missing_fps_reads_as_thirty():
    assert seconds_at(None, 0.0, 30) == pytest.approx(1.0)


def test_flow_start_is_zero_on_a_recording_with_one(synthetic_recording: Path):
    timeline = build_timeline(Session.load(synthetic_recording))
    assert uses_ms(timeline)
    assert t0_of(timeline) == "flow_start"
    # 30 pre-flow frames, numbered from 1: frame 31 is the first in flow.
    assert seconds_at(timeline, FPS, 31) == pytest.approx(0.0, abs=0.002)
    assert seconds_at(timeline, FPS, 1) == pytest.approx(-1.0, abs=0.002)


def test_a_recording_without_a_flow_start_is_video_relative(tmp_path: Path):
    directory = write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_events=False))
    timeline = build_timeline(Session.load(directory))
    assert t0_of(timeline) == "video_start"
    assert seconds_at(timeline, FPS, 1) == pytest.approx(0.0, abs=0.002)


@pytest.mark.parametrize("flow", [True, False])
def test_seconds_and_frames_round_trip(tmp_path: Path, flow: bool):
    directory = write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_events=flow))
    timeline = build_timeline(Session.load(directory))
    for frame in range(1, 151, 7):
        assert frame_at(timeline, FPS, seconds_at(timeline, FPS, frame)) == frame


def test_round_trip_without_a_frame_map():
    for frame in range(0, 400, 13):
        assert frame_at(None, FPS, seconds_at(None, FPS, frame)) == frame


def test_the_frame_after_the_last_still_has_a_time(synthetic_recording: Path):
    """A range that runs to the session's end ends one frame past its last."""
    timeline = build_timeline(Session.load(synthetic_recording))
    last = int(timeline.frame_map.frames[-1])
    step = seconds_at(timeline, FPS, last + 1) - seconds_at(timeline, FPS, last)
    assert step == pytest.approx(1 / FPS, abs=1e-3)
    assert frame_at(timeline, FPS, seconds_at(timeline, FPS, last + 1)) == last + 1


def test_frames_in_is_the_inclusive_frames_of_a_half_open_span():
    assert frames_in(None, FPS, 1.0, 2.0, (0, 299)) == (30, 59)


def test_frames_in_is_clipped_to_the_session():
    assert frames_in(None, FPS, 9.0, 12.0, (0, 299)) == (270, 299)


def test_frames_in_is_none_outside_the_session():
    assert frames_in(None, FPS, 20.0, 30.0, (0, 299)) is None


def test_a_selection_survives_seconds_and_back(synthetic_recording: Path):
    timeline = build_timeline(Session.load(synthetic_recording))
    start, end = 40, 99
    span = seconds_at(timeline, FPS, start), seconds_at(timeline, FPS, end + 1)
    assert frames_in(timeline, FPS, *span, (1, 150)) == (start, end)
