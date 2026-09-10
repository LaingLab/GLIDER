"""Tests for the Qt-free timeline model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glider.analysis import Session
from glider.analysis.timeline import build_frame_map, build_timeline, hardware_lanes

from .conftest import RecordingSpec, write_synthetic_recording


def test_frame_map_comes_from_tracking(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    fm = build_frame_map(s)
    assert fm is not None
    assert fm.source == "tracking"


def test_frame_map_round_trips_a_frame(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    fm = build_frame_map(s)
    # Any frame the map covers must survive frame -> ms -> frame.
    frame = int(fm.frames[len(fm.frames) // 2])
    assert fm.frame_at(fm.ms_of(frame)) == frame


def test_frame_map_applies_the_flow_offset(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    plain = build_frame_map(s)
    shifted = build_frame_map(s, flow_offset_ms=1000.0)
    frame = int(plain.frames[0])
    assert shifted.ms_of(frame) == pytest.approx(plain.ms_of(frame) - 1000.0)


def test_frame_map_survives_a_dropped_frame(tmp_path: Path):
    """A gap in the frame column must not shift later frames.

    This is the whole reason the map is built from the tracking CSV
    rather than from a nominal frame rate: dropping frame 40 means every
    later frame's true time is unchanged, but frame_index / fps would
    place them all one frame early.
    """
    directory = write_synthetic_recording(tmp_path / "rec", RecordingSpec())
    tracking = next(p for p in directory.glob("*_tracking.csv"))
    lines = tracking.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [ln for ln in lines if not ln.startswith("40,")]
    assert len(kept) < len(lines), "fixture changed: no frame 40 row to drop"
    tracking.write_text("".join(kept), encoding="utf-8")

    s = Session.load(directory)
    fm = build_frame_map(s)
    assert 40 not in set(fm.frames.astype(int))
    # The fixture writes 1-based frame numbers against 0-based elapsed time
    # (conftest.py: `frame = i + 1`, `elapsed_ms = i / fps * 1000`), so frame
    # 41 sits at 40/30 s. Do not "correct" this to 41/30. A nominal-fps
    # implementation would place frame 41 at 1300.0 ms (41/30), which is
    # 33 ms away — outside the 5 ms tolerance — so this assertion catches
    # the regression it names.
    assert fm.ms_of(41) == pytest.approx(40 / 30.0 * 1000.0, abs=5.0)


def test_frame_map_is_none_without_tracking(tmp_path: Path):
    directory = write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_tracking=False))
    s = Session.load(directory)
    assert build_frame_map(s) is None


def _rec(tmp_path: Path, events, name="rec") -> Path:
    return write_synthetic_recording(tmp_path / name, RecordingSpec(extra_events=tuple(events)))


def test_hardware_lane_per_device(tmp_path: Path):
    directory = _rec(
        tmp_path,
        [
            (500.0, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "1"),
            (600.0, "output_write", "board0", "pump", "Pump", "6", "DIGITAL", "1"),
        ],
    )
    lanes = hardware_lanes(Session.load(directory))
    assert {lane.key for lane in lanes} == {"led1", "pump"}


def test_value_holds_until_the_next_event(tmp_path: Path):
    """A digital pin written high at 500ms and low at 1500ms is on for 1s."""
    directory = _rec(
        tmp_path,
        [
            (500.0, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "1"),
            (1500.0, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "0"),
        ],
    )
    s = Session.load(directory)
    # Times are flow-relative, and the fixture has a pre-flow period, so the
    # offset has to come from the flow marker — the same value build_timeline
    # will compute once and hand to both builders.
    marker = s.events[(s.events["source"] == "flow_marker") & (s.events["value"] == "start")]
    flow_offset_ms = float(marker["elapsed_ms"].iloc[0])
    lane = hardware_lanes(s, flow_offset_ms=flow_offset_ms)[0]
    on = [seg for seg in lane.segments if seg.value > 0]
    assert len(on) == 1
    assert on[0].start_ms == pytest.approx(500.0, abs=2.0)
    assert on[0].end_ms == pytest.approx(1500.0, abs=2.0)
    assert on[0].level == pytest.approx(1.0)


def test_pwm_normalises_against_full_scale(tmp_path: Path):
    """128 on a PWM pin is half height, not full — per-lane max would
    draw a device that never exceeded 128 as though it were saturated."""
    directory = _rec(
        tmp_path,
        [
            (500.0, "output_write", "board0", "fan", "Fan", "9", "PWM", "128"),
            (1500.0, "output_write", "board0", "fan", "Fan", "9", "PWM", "0"),
        ],
    )
    lane = hardware_lanes(Session.load(directory))[0]
    assert lane.segments[0].level == pytest.approx(128.0 / 255.0)


def test_analog_beyond_assumed_range_falls_back_to_observed(tmp_path: Path):
    """A 12-bit board reads to 4095. Clipping at the 10-bit 1023 would
    draw the whole session as one saturated row."""
    directory = _rec(
        tmp_path,
        [
            (500.0, "input_change", "board0", "ldr", "Photoresistor", "0", "ANALOG", "4095"),
            (1500.0, "input_change", "board0", "ldr", "Photoresistor", "0", "ANALOG", "2048"),
        ],
    )
    lane = hardware_lanes(Session.load(directory))[0]
    assert lane.segments[0].level == pytest.approx(1.0)
    assert lane.segments[1].level == pytest.approx(2048.0 / 4095.0)


def test_non_numeric_value_becomes_a_marker(tmp_path: Path):
    directory = _rec(
        tmp_path,
        [
            (500.0, "input_change", "board0", "reader", "RFID", "2", "", "tag-A7"),
        ],
    )
    lane = hardware_lanes(Session.load(directory))[0]
    assert lane.segments == []
    assert [m.label for m in lane.markers] == ["tag-A7"]


def test_event_without_a_frame_still_makes_a_segment(tmp_path: Path):
    """Device-init writes land before the first camera frame, so their
    `frame` cell is empty. They are drawn from their timestamp."""
    directory = _rec(
        tmp_path,
        [(500.0, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "1")],
    )
    s = Session.load(directory)
    s.events.loc[s.events["source"] == "output_write", "frame"] = np.nan
    assert hardware_lanes(s)[0].segments


def test_flow_markers_are_not_a_lane(tmp_path: Path):
    """Flow boundaries are drawn as rules across every lane, not as a lane."""
    directory = _rec(
        tmp_path,
        [(500.0, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "1")],
    )
    lanes = hardware_lanes(Session.load(directory))
    assert {lane.key for lane in lanes} == {"led1"}
    labels = {m.label for lane in lanes for m in lane.markers}
    assert not labels & {"start", "end"}


def test_no_events_gives_no_lanes(tmp_path: Path):
    directory = write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_events=False))
    assert hardware_lanes(Session.load(directory)) == []


def test_device_id_falls_back_to_board_and_pin(tmp_path: Path):
    directory = _rec(
        tmp_path,
        [(500.0, "output_write", "board0", "", "", "7", "DIGITAL", "1")],
    )
    assert hardware_lanes(Session.load(directory))[0].key == "board0:pin7"


def test_segment_never_ends_before_it_starts(tmp_path: Path):
    """A session that ends before its last event must not make a negative span.

    The camera can stop before the flow tears down, so `end_ms` legitimately
    precedes a late event. A renderer computing `end - start` would draw an
    inverted rect."""
    directory = _rec(
        tmp_path,
        [(500.0, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "1")],
    )
    lane = hardware_lanes(Session.load(directory), end_ms=0.0)[0]
    assert all(seg.end_ms >= seg.start_ms for seg in lane.segments)


def test_empty_value_is_neither_level_nor_marker(tmp_path: Path):
    """The event logger writes "" when a device reports no value."""
    directory = _rec(
        tmp_path,
        [(500.0, "input_change", "board0", "probe", "Probe", "3", "", "")],
    )
    lanes = hardware_lanes(Session.load(directory))
    lane = next(ln for ln in lanes if ln.key == "probe")
    assert lane.segments == []
    assert lane.markers == []


def test_flow_start_is_time_zero(synthetic_recording: Path):
    t = build_timeline(Session.load(synthetic_recording))
    assert t.flow_start_ms == pytest.approx(0.0, abs=1.0)
    assert t.flow_end_ms is not None and t.flow_end_ms > 0


def test_pre_flow_events_keep_negative_times(tmp_path: Path):
    """Device-init writes happen before flow start and must stay visible.

    The default spec has 1s of pre-flow, so an event at flow_ms=-500
    cannot be expressed through extra_events; shift the whole axis instead
    by asserting the timeline starts before zero because the tracking CSV
    itself begins in the pre-flow period.
    """
    t = build_timeline(Session.load(write_synthetic_recording(tmp_path / "rec", RecordingSpec())))
    assert t.start_ms < 0.0


def test_behavior_lane_from_tracking(synthetic_recording: Path):
    t = build_timeline(Session.load(synthetic_recording))
    sources = [lane.source for lane in t.behavior]
    assert "tracking" in sources


def test_timeline_without_a_session_is_empty_but_valid(tmp_path: Path):
    t = build_timeline(None)
    assert t.lanes == []
    assert t.behavior == []
    assert t.frame_map is None
    assert t.start_ms == 0.0 and t.end_ms == 0.0


def test_no_flow_marker_leaves_boundaries_none(tmp_path: Path):
    directory = write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_events=False))
    t = build_timeline(Session.load(directory))
    assert t.flow_start_ms is None
    assert t.flow_end_ms is None


def test_multi_object_tracking_gets_one_lane_per_object(tmp_path: Path):
    """Two tracked mice must not collapse to one object's behaviour.

    A shared frame column across objects makes a naive
    drop_duplicates(subset="frame") keep only the first object's state,
    silently discarding the second mouse's labels entirely.
    """
    directory = write_synthetic_recording(tmp_path / "rec", RecordingSpec(n_objects=2))
    t = build_timeline(Session.load(directory))
    tracking_lanes = {
        lane.source: lane for lane in t.behavior if lane.source.startswith("tracking")
    }
    assert set(tracking_lanes) == {"tracking[0]", "tracking[1]"}
    obj0, obj1 = tracking_lanes["tracking[0]"], tracking_lanes["tracking[1]"]
    assert len(obj0.labels) == len(obj1.labels) > 0
    # Object 1's labels are the fixture's "<state>_obj1" — not object 0's
    # labels repeated under a different source string.
    assert obj0.labels != obj1.labels
    assert all(label.endswith("_obj1") for label in obj1.labels)


def test_single_object_tracking_keeps_bare_source(synthetic_recording: Path):
    """Task 5 depends on the single-object source staying the bare string
    "tracking", not "tracking[0]"."""
    t = build_timeline(Session.load(synthetic_recording))
    tracking_lanes = [lane for lane in t.behavior if lane.source == "tracking"]
    assert len(tracking_lanes) == 1
    assert not any(lane.source.startswith("tracking[") for lane in t.behavior)
