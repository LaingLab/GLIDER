"""Tests for kinematics: velocity, speed distribution, cumulative distance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from glider.analysis import (
    Session,
    compute_speed_distribution,
    compute_velocity_series,
    extract_cumulative_distance,
)

from .conftest import RecordingSpec, write_synthetic_recording


def test_velocity_series_has_flow_time_axis(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    vel = s.velocity(use_frame_rate=False)

    assert list(vel.columns) == ["flow_elapsed_ms", "velocity"]
    assert len(vel) == 120  # post-flow frames
    assert vel["flow_elapsed_ms"].iloc[0] == 0.0  # first post-flow frame at flow t=0


def test_velocity_scales_with_frame_rate(synthetic_recording: Path):
    """frame_rate=30 should multiply px/frame by 30 to give px/s."""
    s = Session.load(synthetic_recording)
    vel_pf = s.velocity(use_frame_rate=False)
    vel_ps = s.velocity(use_frame_rate=True)

    # Active state has velocity_px_frame=5; *30 fps = 150 px/s.
    nz_pf = vel_pf[vel_pf["velocity"] > 0]
    nz_ps = vel_ps[vel_ps["velocity"] > 0]
    # frame_rate from fixture is ~30.3 due to ms quantization; just check ratio.
    if not nz_pf.empty:
        ratio = nz_ps["velocity"].iloc[0] / nz_pf["velocity"].iloc[0]
        assert 29 < ratio < 32


def test_velocity_empty_for_no_tracking(tmp_path: Path):
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_tracking=False))
    s = Session.load(tmp_path / "rec")
    vel = s.velocity()
    assert vel.empty


def test_speed_distribution_returns_bins(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    dist = s.speed_distribution(bins=10, use_frame_rate=False)

    assert list(dist.columns) == ["bin_center", "count"]
    assert len(dist) == 10
    assert dist["count"].sum() == 120  # all post-flow frames binned


def test_speed_distribution_bins_zero_and_nonzero(synthetic_recording: Path):
    """Default fixture has 0-velocity (resting) and 5-velocity (active) bins."""
    s = Session.load(synthetic_recording)
    dist = s.speed_distribution(bins=20, use_frame_rate=False)
    # Lowest bin (0 px/frame) should have many entries — most frames are resting.
    # Some bin near 5 px/frame should be nonzero for the active stretch.
    assert (dist["count"] > 0).sum() >= 2


def test_cumulative_distance_returns_flow_time_indexed(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    dist = s.cumulative_distance()

    assert list(dist.columns) == ["flow_elapsed_ms", "cumulative_mm"]
    assert len(dist) == 120


def test_cumulative_distance_empty_when_no_tracking(tmp_path: Path):
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_tracking=False))
    s = Session.load(tmp_path / "rec")
    assert s.cumulative_distance().empty


def test_compute_velocity_series_direct():
    df = pd.DataFrame(
        {
            "object_id": [0, 0, 0],
            "flow_elapsed_ms": [0.0, 100.0, 200.0],
            "velocity_px_frame": [1.0, 2.0, 3.0],
        }
    )
    vel = compute_velocity_series(df, frame_rate=30.0)
    np.testing.assert_allclose(vel["velocity"], [30.0, 60.0, 90.0])


def test_compute_speed_distribution_handles_all_nan():
    """All-NaN velocities give an empty distribution, not a crash."""
    df = pd.DataFrame(
        {
            "object_id": [0, 0],
            "flow_elapsed_ms": [0.0, 100.0],
            "velocity_px_frame": [np.nan, np.nan],
        }
    )
    dist = compute_speed_distribution(df)
    assert dist.empty


def test_extract_cumulative_distance_direct():
    df = pd.DataFrame(
        {
            "object_id": [0, 0, 0],
            "flow_elapsed_ms": [0.0, 100.0, 200.0],
            "cumulative_mm": [0.0, 5.0, 12.0],
        }
    )
    out = extract_cumulative_distance(df)
    np.testing.assert_allclose(out["cumulative_mm"], [0.0, 5.0, 12.0])
