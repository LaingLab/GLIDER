"""Identity flags. Consolidation actively invents links between fragments, so
it owes the analyst more honesty than the live tracker does, not less."""

import numpy as np

from glider.vision.pose.core import PoseData
from glider.vision.pose.identity import (
    identity_flags,
    identity_output_path,
    write_identity_csv,
)
from glider.vision.pose.tracks import PoseTracks

NAMES = ["snout", "tail"]


def track(positions):
    """positions: (F, 2) centroid per frame; NaN row means undetected."""
    positions = np.asarray(positions, dtype=float)
    xy = np.repeat(positions[:, None, :], len(NAMES), axis=1)
    conf = np.where(np.isfinite(xy[:, :, 0]), 1.0, 0.0)
    return PoseData(xy=xy, confidence=conf, keypoint_names=list(NAMES), fps=30.0)


def tracks_from(*position_arrays):
    return PoseTracks(tracks={i: track(p) for i, p in enumerate(position_arrays)}, fps=30.0)


def test_animals_well_apart_get_no_rows_at_all():
    # A column that always fires says nothing.
    t = tracks_from([[0, 0]] * 5, [[500, 500]] * 5)
    assert identity_flags(t, stitched={0: set(), 1: set()}, min_separation_px=60.0) == []


def test_close_fires_for_both_animals_on_the_frame_they_touch():
    t = tracks_from([[0, 0], [0, 0]], [[500, 500], [10, 0]])
    rows = identity_flags(t, stitched={0: set(), 1: set()}, min_separation_px=60.0)
    assert sorted(rows) == [(1, 0, "close"), (1, 1, "close")]


def test_stitched_marks_exactly_the_joined_frames():
    t = tracks_from([[0, 0]] * 4, [[900, 900]] * 4)
    rows = identity_flags(t, stitched={0: {2, 3}, 1: set()}, min_separation_px=60.0)
    assert sorted(rows) == [(2, 0, "stitched"), (3, 0, "stitched")]


def test_a_frame_with_no_detection_is_a_gap():
    t = tracks_from([[0, 0], [np.nan, np.nan]], [[900, 900]] * 2)
    rows = identity_flags(t, stitched={0: set(), 1: set()}, min_separation_px=60.0)
    assert (1, 0, "gap") in rows


def test_a_gap_is_never_also_close_since_there_is_no_position():
    t = tracks_from([[np.nan, np.nan]], [[0, 0]])
    rows = identity_flags(t, stitched={0: {0}, 1: set()}, min_separation_px=60.0)
    assert [r for r in rows if r[1] == 0] == [(0, 0, "gap")]


def test_several_causes_join_in_a_fixed_order():
    t = tracks_from([[0, 0]], [[10, 0]])
    rows = identity_flags(t, stitched={0: {0}, 1: set()}, min_separation_px=60.0)
    assert (0, 0, "close+stitched") in rows


def test_the_sidecar_sits_beside_the_pose_csv(tmp_path):
    assert identity_output_path(tmp_path / "vidDLC_m.csv").name == "vidDLC_m_identity.csv"


def test_write_identity_csv_round_trips(tmp_path):
    path = write_identity_csv(tmp_path / "x_identity.csv", [(3, 1, "close")])
    assert path.read_text().splitlines() == ["frame,individual,identity_flag", "3,animal1,close"]


def test_the_sidecar_is_not_mistaken_for_a_pose_csv():
    from glider.vision.pose.dlc import NOT_POSE_SUFFIXES

    assert "_identity" in NOT_POSE_SUFFIXES
