"""Tests for compute_bouts + compute_state_transitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glider.analysis import Session, compute_bouts, compute_state_transitions

from .conftest import RecordingSpec, write_synthetic_recording


def test_bouts_returns_dict_when_state_unspecified(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    bouts = s.bouts()
    assert isinstance(bouts, dict)
    # Default schedule has resting + active.
    assert set(bouts.keys()) == {"resting", "active"}


def test_bouts_returns_series_when_state_specified(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    resting = s.bouts(state="resting")
    assert isinstance(resting, pd.Series)
    # Two resting bouts (start + tail).
    assert len(resting) == 2
    assert (resting > 0).all()


def test_bouts_empty_series_for_unknown_state(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    bouts = s.bouts(state="not_a_real_state")
    assert isinstance(bouts, pd.Series)
    assert len(bouts) == 0


def test_state_transitions_default_schedule(synthetic_recording: Path):
    """resting → active → resting yields 2 transitions."""
    s = Session.load(synthetic_recording)
    trans = s.state_transitions()

    assert len(trans) == 2
    pairs = set(zip(trans["from_state"], trans["to_state"], strict=True))
    assert pairs == {("resting", "active"), ("active", "resting")}


def test_state_transitions_single_state(tmp_path: Path):
    spec = RecordingSpec(state_schedule=((0.0, "resting"),))
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")
    trans = s.state_transitions()
    assert trans.empty


def test_state_transitions_aggregates_counts(tmp_path: Path):
    """resting → active → resting → active should count resting→active twice."""
    spec = RecordingSpec(
        state_schedule=(
            (0.0, "resting"),
            (1000.0, "active"),
            (2000.0, "resting"),
            (3000.0, "active"),
        ),
    )
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")
    trans = s.state_transitions()

    r_to_a = trans[(trans["from_state"] == "resting") & (trans["to_state"] == "active")]
    assert len(r_to_a) == 1
    assert r_to_a.iloc[0]["count"] == 2


def test_compute_bouts_direct_smoke():
    intervals = pd.DataFrame(
        {
            "object_id": [0, 0, 0],
            "state": ["a", "b", "a"],
            "start_ms": [0.0, 100.0, 200.0],
            "end_ms": [100.0, 200.0, 300.0],
            "duration_ms": [100.0, 100.0, 100.0],
        }
    )
    bouts = compute_bouts(intervals)
    assert set(bouts.keys()) == {"a", "b"}
    assert len(bouts["a"]) == 2
    assert len(bouts["b"]) == 1


def test_compute_state_transitions_direct_smoke():
    intervals = pd.DataFrame(
        {
            "object_id": [0, 0, 0],
            "state": ["a", "b", "a"],
            "start_ms": [0.0, 100.0, 200.0],
            "end_ms": [100.0, 200.0, 300.0],
            "duration_ms": [100.0, 100.0, 100.0],
        }
    )
    trans = compute_state_transitions(intervals)
    assert len(trans) == 2
