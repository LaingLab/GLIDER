"""Tests for the Qt-free timeline model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glider.analysis import Session
from glider.analysis.timeline import FrameMap, build_frame_map

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
    directory = write_synthetic_recording(
        tmp_path / "rec", RecordingSpec(write_tracking=False)
    )
    s = Session.load(directory)
    assert build_frame_map(s) is None
