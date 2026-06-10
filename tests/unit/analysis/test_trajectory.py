"""Tests for trajectory + occupancy + zone analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from glider.analysis import (
    Session,
    compute_occupancy,
    compute_trajectory,
    compute_zone_dwell,
    compute_zone_transitions,
)

from .conftest import RecordingSpec, write_synthetic_recording


def test_trajectory_returns_position_columns(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    traj = s.trajectory()

    assert list(traj.columns) == ["frame", "flow_elapsed_ms", "center_x", "center_y"]
    # Default spec has 120 post-flow frames, no pre-flow.
    assert len(traj) == 120


def test_trajectory_excludes_pre_flow_by_default(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    traj = s.trajectory()
    assert traj["flow_elapsed_ms"].notna().all()


def test_trajectory_includes_pre_flow_when_requested(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    traj = compute_trajectory(s.tracking, include_pre_flow=True)
    # 30 pre-flow + 120 post-flow.
    assert len(traj) == 150


def test_trajectory_position_walks_across_frame(synthetic_recording: Path):
    """Default fixture interpolates position from (100,100) to (500,300)."""
    s = Session.load(synthetic_recording)
    traj = s.trajectory()
    # First post-flow position is at frame index 30 / 150 = 20% of the walk.
    # Last post-flow position is at frame index 149 / 150 — basically at end.
    assert traj["center_x"].iloc[0] < traj["center_x"].iloc[-1]
    assert traj["center_y"].iloc[0] < traj["center_y"].iloc[-1]
    assert traj["center_x"].iloc[-1] > 400  # near 500


def test_trajectory_empty_for_missing_object(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    traj = s.trajectory(object_id=999)
    assert traj.empty


def test_occupancy_returns_heatmap_and_edges(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    heatmap, x_edges, y_edges = s.occupancy(bins=20)

    assert heatmap.shape == (20, 20)
    assert len(x_edges) == 21
    assert len(y_edges) == 21
    assert heatmap.sum() == 120  # number of post-flow frames


def test_occupancy_with_frame_size_anchors_extent(synthetic_recording: Path):
    """Passing frame_size makes the histogram cover the full frame, not just
    the visited portion — important for cross-recording comparisons."""
    s = Session.load(synthetic_recording)
    _, x_edges, y_edges = s.occupancy(bins=10, frame_size=(640, 480))
    assert x_edges[0] == 0
    assert x_edges[-1] == 640
    assert y_edges[0] == 0
    assert y_edges[-1] == 480


def test_occupancy_empty_when_no_tracking(tmp_path: Path):
    write_synthetic_recording(tmp_path / "rec", RecordingSpec(write_tracking=False))
    s = Session.load(tmp_path / "rec")
    heatmap, _, _ = s.occupancy()
    assert heatmap.sum() == 0


def test_zone_dwell_single_zone_recording(synthetic_recording: Path):
    """Default fixture keeps the subject in zone1 the whole flow."""
    s = Session.load(synthetic_recording)
    dwell = s.zone_dwell()

    assert len(dwell) == 1
    assert dwell.iloc[0]["zone"] == "zone1"
    assert dwell.iloc[0]["n_entries"] == 1
    assert dwell.iloc[0]["total_ms"] > 3000  # ~4s of flow


def test_zone_dwell_multi_zone_via_schedule(tmp_path: Path):
    """Custom zone_schedule: zone1 0-2s, zone2 2-3s, zone1 3-4s."""
    spec = RecordingSpec(
        zone_schedule=(
            (0.0, "zone1"),
            (2000.0, "zone2"),
            (3000.0, "zone1"),
        ),
    )
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")
    dwell = s.zone_dwell()

    by_zone = dwell.set_index("zone")
    assert "zone1" in by_zone.index
    assert "zone2" in by_zone.index
    # zone1 has TWO entries (start, then re-entry after zone2)
    assert by_zone.loc["zone1", "n_entries"] == 2
    assert by_zone.loc["zone2", "n_entries"] == 1


def test_zone_transitions_pairwise(tmp_path: Path):
    """zone1 → zone2 → zone1 yields two transitions."""
    spec = RecordingSpec(
        zone_schedule=(
            (0.0, "zone1"),
            (2000.0, "zone2"),
            (3000.0, "zone1"),
        ),
    )
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")
    trans = s.zone_transitions()

    assert len(trans) == 2
    assert set(zip(trans["from_zone"], trans["to_zone"], strict=True)) == {
        ("zone1", "zone2"),
        ("zone2", "zone1"),
    }
    assert (trans["count"] == 1).all()


def test_zone_transitions_empty_for_single_zone(synthetic_recording: Path):
    s = Session.load(synthetic_recording)
    trans = s.zone_transitions()
    assert trans.empty


def test_zone_transitions_aggregates_repeats(tmp_path: Path):
    """Multiple zone1→zone2 visits should count as a single row with count=2."""
    spec = RecordingSpec(
        zone_schedule=(
            (0.0, "zone1"),
            (1000.0, "zone2"),
            (2000.0, "zone1"),
            (3000.0, "zone2"),
        ),
    )
    write_synthetic_recording(tmp_path / "rec", spec)
    s = Session.load(tmp_path / "rec")
    trans = s.zone_transitions()

    # zone1→zone2 happens twice
    z1_to_z2 = trans[(trans["from_zone"] == "zone1") & (trans["to_zone"] == "zone2")]
    assert len(z1_to_z2) == 1
    assert z1_to_z2.iloc[0]["count"] == 2


def test_compute_occupancy_direct_smoke():
    """Hand-crafted DataFrame: 4 points in a single bin should yield count=4."""
    df = pd.DataFrame(
        {
            "object_id": [0, 0, 0, 0],
            "flow_elapsed_ms": [0.0, 100.0, 200.0, 300.0],
            "center_x": [10.0, 11.0, 12.0, 13.0],
            "center_y": [20.0, 21.0, 22.0, 23.0],
        }
    )
    heatmap, _, _ = compute_occupancy(df, bins=1)
    assert heatmap[0, 0] == 4


def test_compute_zone_dwell_direct_smoke():
    df = pd.DataFrame(
        {
            "object_id": [0, 0, 0],
            "flow_elapsed_ms": [0.0, 100.0, 200.0],
            "zone_ids": ["zoneA", "zoneA", "zoneA"],
        }
    )
    dwell = compute_zone_dwell(df)
    assert len(dwell) == 1
    assert dwell.iloc[0]["zone"] == "zoneA"


def test_compute_zone_transitions_direct_smoke():
    df = pd.DataFrame(
        {
            "object_id": [0, 0, 0, 0],
            "flow_elapsed_ms": [0.0, 100.0, 200.0, 300.0],
            "zone_ids": ["a", "b", "a", "b"],
        }
    )
    trans = compute_zone_transitions(df)
    # a→b appears twice; b→a once.
    by_pair = trans.set_index(["from_zone", "to_zone"])
    assert by_pair.loc[("a", "b"), "count"] == 2
    assert by_pair.loc[("b", "a"), "count"] == 1
