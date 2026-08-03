"""Zones, occupancy and dwell for sessions that came from a video.

The spatial suite has existed since the live rig but read only
``Session.tracking``, so a cohort scored from recorded video had no
time-in-zone, no entries and no heatmap at all. These tests pin the bridge —
and specifically that it re-expresses the data rather than re-implementing
the analysis, since two implementations of "time in zone" is how a cohort
ends up with two answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.session_view import SessionView
from glider.analysis.behavior.spatial import (
    SpatialError,
    occupancy_grid,
    position_track,
    tracking_frame,
    zone_occupancy,
)
from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

NAMES = ["nose", "l_ear", "r_ear", "tail_base"]


def _view(tmp_path, xs, ys, labels=None, resolution=(640, 480), fps=30.0):
    """A session whose animal walks a path we choose."""
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    folder = tmp_path / "v"
    folder.mkdir(parents=True, exist_ok=True)
    n = len(xs)
    pd.DataFrame({"frame": range(n), "behavior": labels or ["groom"] * n}).to_csv(
        folder / "ethogram_raw.csv", index=False
    )

    xy = np.zeros((n, len(NAMES), 2))
    xy[:, :, 0] = np.asarray(xs, dtype=float)[:, None]
    xy[:, :, 1] = np.asarray(ys, dtype=float)[:, None]
    to_dlc_csv(
        PoseData(
            xy=xy,
            confidence=np.ones((n, len(NAMES))),
            keypoint_names=NAMES,
            fps=fps,
            metadata={"resolution": resolution} if resolution else {},
        ),
        folder / "vDLC_exp-7.csv",
    )
    return SessionView.load(folder / "ethogram_raw.csv")


def _centre_zone():
    config = ZoneConfiguration()
    config.add_zone(
        Zone(
            id="centre",
            name="centre",
            shape=ZoneShape.RECTANGLE,
            vertices=[(0.25, 0.25), (0.75, 0.75)],
        )
    )
    return config


class TestTheTrackingShape:
    def test_it_produces_the_columns_the_spatial_suite_reads(self, tmp_path):
        view = _view(tmp_path, [100] * 10, [100] * 10)
        frame = tracking_frame(view)
        assert {
            "object_id",
            "frame",
            "flow_elapsed_ms",
            "center_x",
            "center_y",
            "zone_ids",
            "behavioral_state",
        } <= set(frame.columns)

    def test_positions_are_pixels_like_the_live_logger(self, tmp_path):
        view = _view(tmp_path, [320] * 5, [240] * 5)
        frame = tracking_frame(view)
        assert frame["center_x"].iloc[0] == pytest.approx(320.0)
        assert frame["center_y"].iloc[0] == pytest.approx(240.0)

    def test_times_come_from_the_frame_rate(self, tmp_path):
        view = _view(tmp_path, [1] * 4, [1] * 4, fps=25.0)
        frame = tracking_frame(view)
        assert frame["flow_elapsed_ms"].iloc[2] == pytest.approx(2 / 25.0 * 1000)

    def test_a_named_keypoint_can_stand_for_the_animal(self, tmp_path):
        """A nose crossing a zone is a different event from a body crossing."""
        view = _view(tmp_path, [10] * 4, [10] * 4)
        view.xy[:, 0, :] = 500.0  # move only the nose
        assert position_track(view, "nose")[0][0] == pytest.approx(500.0)
        assert position_track(view)[0][0] != pytest.approx(500.0)

    def test_an_unknown_keypoint_is_refused_by_name(self, tmp_path):
        view = _view(tmp_path, [1] * 3, [1] * 3)
        with pytest.raises(SpatialError, match="snout"):
            position_track(view, "snout")

    def test_a_session_without_poses_cannot_have_a_position(self, tmp_path):
        folder = tmp_path / "noposes"
        folder.mkdir()
        pd.DataFrame({"frame": range(5), "behavior": ["a"] * 5}).to_csv(
            folder / "ethogram_raw.csv", index=False
        )
        view = SessionView.load(folder / "ethogram_raw.csv")
        with pytest.raises(SpatialError, match="no poses"):
            tracking_frame(view)


class TestZoneMembership:
    def test_a_point_inside_the_zone_is_named(self, tmp_path):
        view = _view(tmp_path, [320] * 5, [240] * 5)  # dead centre of 640x480
        frame = tracking_frame(view, _centre_zone())
        assert set(frame["zone_ids"]) == {"centre"}

    def test_a_point_outside_every_zone_is_blank(self, tmp_path):
        view = _view(tmp_path, [10] * 5, [10] * 5)
        frame = tracking_frame(view, _centre_zone())
        assert set(frame["zone_ids"]) == {""}

    def test_without_a_resolution_membership_is_not_guessed(self, tmp_path):
        """Normalising needs the arena size; inventing one would misplace it."""
        view = _view(tmp_path, [320] * 5, [240] * 5, resolution=None)
        frame = tracking_frame(view, _centre_zone())
        assert set(frame["zone_ids"]) == {""}

    def test_a_dropped_frame_is_not_outside_every_zone(self, tmp_path):
        view = _view(tmp_path, [320] * 5, [240] * 5)
        view.xy[2] = np.nan
        frame = tracking_frame(view, _centre_zone())
        assert frame["zone_ids"].iloc[2] == ""
        assert frame["zone_ids"].iloc[1] == "centre"


class TestZoneOccupancy:
    def _in_and_out(self, tmp_path, n_in=60, n_out=40):
        xs = [320] * n_in + [10] * n_out
        return _view(tmp_path, xs, [240] * n_in + [10] * n_out)

    def test_time_in_zone_is_seconds(self, tmp_path):
        view = self._in_and_out(tmp_path)
        rows = zone_occupancy(view, _centre_zone()).set_index("zone")
        assert rows.loc["centre", "total_s"] == pytest.approx(60 / 30.0)

    def test_fractions_cover_the_window(self, tmp_path):
        view = self._in_and_out(tmp_path)
        rows = zone_occupancy(view, _centre_zone())
        assert rows["fraction"].sum() == pytest.approx(1.0)

    def test_entries_are_counted(self, tmp_path):
        # in, out, in again
        xs = [320] * 20 + [10] * 20 + [320] * 20
        ys = [240] * 20 + [10] * 20 + [240] * 20
        rows = zone_occupancy(_view(tmp_path, xs, ys), _centre_zone()).set_index("zone")
        assert int(rows.loc["centre", "n_entries"]) == 2

    def test_latency_is_measured_from_the_window_start(self, tmp_path):
        xs = [10] * 30 + [320] * 30
        ys = [10] * 30 + [240] * 30
        rows = zone_occupancy(_view(tmp_path, xs, ys), _centre_zone()).set_index("zone")
        assert rows.loc["centre", "latency_s"] == pytest.approx(1.0)  # 30 frames @ 30 fps

    def test_a_zone_never_entered_has_no_latency(self, tmp_path):
        """Never entered is not the same as entered at time zero."""
        view = _view(tmp_path, [10] * 30, [10] * 30)
        rows = zone_occupancy(view, _centre_zone()).set_index("zone")
        assert "centre" not in rows.index or np.isnan(rows.loc["centre", "latency_s"])

    def test_a_window_restricts_the_measurement(self, tmp_path):
        view = self._in_and_out(tmp_path)
        rows = zone_occupancy(view, _centre_zone(), start_frame=60, end_frame=99)
        assert set(rows["zone"]) == {"(outside)"}

    def test_the_window_is_measured_in_absolute_frames(self, tmp_path):
        view = self._in_and_out(tmp_path)
        rows = zone_occupancy(view, _centre_zone(), start_frame=0, end_frame=29).set_index("zone")
        assert rows.loc["centre", "total_s"] == pytest.approx(30 / 30.0)


class TestOccupancyGrid:
    def test_it_bins_every_frame(self, tmp_path):
        view = _view(tmp_path, [320] * 50, [240] * 50)
        grid, _x, _y = occupancy_grid(view, bins=10)
        assert grid.sum() == 50

    def test_it_is_anchored_to_the_arena_not_the_track(self, tmp_path):
        """Two animals' heatmaps must be drawn on the same arena to compare."""
        view = _view(tmp_path, [320] * 20, [240] * 20)
        _grid, x_edges, y_edges = occupancy_grid(view, bins=10)
        assert x_edges[0] == pytest.approx(0.0)
        assert x_edges[-1] == pytest.approx(640.0)
        assert y_edges[-1] == pytest.approx(480.0)

    def test_a_window_bins_only_that_window(self, tmp_path):
        view = _view(tmp_path, [320] * 50, [240] * 50)
        grid, _x, _y = occupancy_grid(view, bins=10, start_frame=0, end_frame=9)
        assert grid.sum() == 10
