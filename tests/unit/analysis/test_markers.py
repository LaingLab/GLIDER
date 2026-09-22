"""Review markers: the time rule, the files, and the stores. Qt-free."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from glider.analysis import Session
from glider.analysis.markers import (
    COHORT_FILE,
    SESSION_FILE,
    SWATCHES,
    Marker,
    MarkerFileError,
    MarkerStore,
    frame_at,
    frames_in,
    load_markers,
    save_markers,
    seconds_at,
    stack_rows,
    t0_of,
    uses_ms,
)
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


# ---------------------------------------------------------------------------
# the model and its files


def test_a_marker_survives_the_round_trip(tmp_path: Path):
    path = tmp_path / SESSION_FILE
    markers = [
        Marker("point", 12.5, name="door stuck", color="red", note="lever jammed"),
        Marker("range", 60.0, 260.0, name="Stim", color="cyan", scope="cohort"),
    ]
    save_markers(path, markers, t0="flow_start")
    loaded, t0 = load_markers(path)
    assert loaded == markers
    assert t0 == "flow_start"


def test_a_point_has_no_end_and_no_duration():
    point = Marker("point", 3.0)
    assert point.end_s is None and not point.is_range and point.duration_s == 0.0
    assert Marker("range", 1.0, 4.5).duration_s == pytest.approx(3.5)


def test_every_marker_gets_its_own_id():
    assert Marker("point", 1.0).id != Marker("point", 1.0).id


def test_a_missing_file_is_no_markers(tmp_path: Path):
    assert load_markers(tmp_path / SESSION_FILE) == ([], None)


def test_a_cohort_file_has_no_t0(tmp_path: Path):
    path = tmp_path / COHORT_FILE
    save_markers(path, [Marker("range", 0.0, 60.0, scope="cohort")])
    assert "t0" not in json.loads(path.read_text())


def test_writing_leaves_no_temp_file_behind(tmp_path: Path):
    save_markers(tmp_path / SESSION_FILE, [Marker("point", 1.0)], t0="video_start")
    assert [p.name for p in tmp_path.iterdir()] == [SESSION_FILE]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_saving_over_a_group_readable_file_keeps_its_mode(tmp_path: Path):
    path = tmp_path / SESSION_FILE
    save_markers(path, [Marker("point", 1.0)], t0="video_start")
    path.chmod(0o664)
    save_markers(path, [Marker("point", 2.0)], t0="video_start")
    assert (path.stat().st_mode & 0o777) == 0o664


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes only")
def test_saving_a_fresh_file_is_group_readable_not_owner_only(tmp_path: Path):
    path = tmp_path / SESSION_FILE
    save_markers(path, [Marker("point", 1.0)], t0="video_start")
    umask = os.umask(0)
    os.umask(umask)
    assert (path.stat().st_mode & 0o777) == (0o666 & ~umask)


def test_a_failed_write_keeps_the_old_file(tmp_path: Path, monkeypatch):
    path = tmp_path / SESSION_FILE
    save_markers(path, [Marker("point", 1.0)], t0="video_start")
    before = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("glider.analysis.markers.json.dump", boom)
    with pytest.raises(OSError):
        save_markers(path, [Marker("point", 2.0)], t0="video_start")
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [SESSION_FILE]


@pytest.mark.parametrize(
    "text",
    [
        "{not json",
        '{"version": 1}',
        '{"version": 1, "markers": [{"kind": "range", "start_s": 5.0}]}',
        '{"version": 1, "markers": [{"kind": "blob", "start_s": 5.0}]}',
        '{"markers": []}',
    ],
)
def test_an_unreadable_file_raises(tmp_path: Path, text: str):
    path = tmp_path / SESSION_FILE
    path.write_text(text)
    with pytest.raises(MarkerFileError):
        load_markers(path)


def test_a_newer_file_is_refused(tmp_path: Path):
    path = tmp_path / SESSION_FILE
    path.write_text('{"version": 2, "markers": []}')
    with pytest.raises(MarkerFileError, match="newer"):
        load_markers(path)


def test_an_unknown_colour_loads_as_the_default(tmp_path: Path):
    path = tmp_path / SESSION_FILE
    path.write_text(
        '{"version": 1, "markers": [{"kind": "point", "start_s": 1.0, "color": "teal"}]}'
    )
    (marker,), _ = load_markers(path)
    assert marker.color == SWATCHES[0]


# ---------------------------------------------------------------------------
# the store


def test_a_store_over_an_unreadable_file_is_read_only_and_never_writes(tmp_path: Path):
    path = tmp_path / SESSION_FILE
    path.write_text("{not json")
    store = MarkerStore(path, t0="video_start")
    assert store.error and not store.writable and store.markers == []
    store.put(Marker("point", 1.0))
    with pytest.raises(MarkerFileError):
        store.save()
    assert path.read_text() == "{not json"


def test_a_store_puts_replaces_and_removes(tmp_path: Path):
    store = MarkerStore(tmp_path / SESSION_FILE, t0="video_start")
    marker = Marker("point", 1.0, name="a")
    store.put(marker)
    store.put(Marker("point", 2.0, name="b", id=marker.id))
    assert [m.name for m in store.markers] == ["b"]
    assert store.find(marker.id).start_s == 2.0
    store.remove(marker.id)
    assert store.markers == [] and store.find(marker.id) is None


def test_a_store_saves_and_reloads(tmp_path: Path):
    store = MarkerStore(tmp_path / SESSION_FILE, t0="flow_start")
    store.put(Marker("range", 1.0, 2.0, name="Stim"))
    store.save()
    again = MarkerStore(tmp_path / SESSION_FILE, t0="flow_start")
    assert [m.name for m in again.markers] == ["Stim"] and not again.t0_changed


def test_a_changed_zero_is_flagged_but_still_loads(tmp_path: Path):
    save_markers(tmp_path / SESSION_FILE, [Marker("point", 1.0)], t0="video_start")
    store = MarkerStore(tmp_path / SESSION_FILE, t0="flow_start")
    assert store.t0_changed and len(store.markers) == 1 and store.writable


def test_a_cohort_store_never_flags_a_zero(tmp_path: Path):
    save_markers(tmp_path / COHORT_FILE, [Marker("range", 0.0, 1.0, scope="cohort")])
    assert not MarkerStore(tmp_path / COHORT_FILE).t0_changed


# ---------------------------------------------------------------------------
# stacking range markers into sub-rows


def test_overlapping_ranges_stack_into_two_rows():
    # (0,10) row 0; (1,2) overlaps it: row 1; (5,15) overlaps row 0: row 1;
    # (12,20) starts after row 0 ended: row 0.
    assert stack_rows([(0, 10), (5, 15), (12, 20), (1, 2)]) == [0, 1, 0, 1]


def test_a_range_overlapping_both_rows_goes_on_the_second():
    assert stack_rows([(0, 10), (1, 20), (5, 8)]) == [0, 1, 1]
