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


class TestVectorisedMembershipMatchesTheScalarForm:
    """The fast path must agree with Zone.contains_point exactly.

    A cohort is thirty sessions of ~45,000 frames, so the scalar form is over
    a million Python-level calls per window change — but a faster answer that
    disagrees at a boundary is worse than a slow one.
    """

    def _agree(self, zone, n=2000, seed=0):
        from glider.analysis.behavior.spatial import _zone_mask

        rng = np.random.default_rng(seed)
        xs, ys = rng.random(n), rng.random(n)
        fast = _zone_mask(zone, xs, ys)
        slow = np.array([zone.contains_point(x, y) for x, y in zip(xs, ys, strict=True)])
        assert np.array_equal(fast, slow)

    def test_rectangles_agree(self):
        self._agree(
            Zone(id="r", name="r", shape=ZoneShape.RECTANGLE, vertices=[(0.2, 0.3), (0.8, 0.6)])
        )

    def test_rectangles_agree_with_reversed_corners(self):
        self._agree(
            Zone(id="r", name="r", shape=ZoneShape.RECTANGLE, vertices=[(0.8, 0.6), (0.2, 0.3)])
        )

    def test_circles_agree(self):
        self._agree(
            Zone(id="c", name="c", shape=ZoneShape.CIRCLE, vertices=[(0.5, 0.5), (0.5, 0.8)])
        )

    def test_polygons_agree(self):
        self._agree(
            Zone(
                id="p",
                name="p",
                shape=ZoneShape.POLYGON,
                vertices=[(0.2, 0.2), (0.8, 0.25), (0.7, 0.75), (0.25, 0.7)],
            )
        )

    def test_a_degenerate_zone_contains_nothing(self):
        from glider.analysis.behavior.spatial import _zone_mask

        zone = Zone(id="d", name="d", shape=ZoneShape.RECTANGLE, vertices=[])
        assert not _zone_mask(zone, np.array([0.5]), np.array([0.5])).any()


class TestTheEthogramNamesTheFrameNotTheRow:
    """Positions are looked up by frame number, never by row position.

    The pose track has one row per video frame; the ethogram has one row per
    *scored* frame, which is neither the same count nor the same origin. A run
    over minutes 2–7 starts at frame 3600, and a cadence of 3 emits every third
    frame. Zipping the two by position paired every label with the wrong place:
    a heatmap of minutes 2–7 was really minutes 0–5, and every zone number came
    out of it.
    """

    def _view(self, tmp_path, *, first, count, stride=1, n_pose=None, x_origin=0.0):
        """A session whose ethogram covers part of a longer pose track.

        ``x_origin`` shifts the track so the scored window lands inside the
        arena; without it a frame number doubles as an x far outside 640 and
        the occupancy histogram discards every point.
        """
        from glider.vision.pose.core import PoseData
        from glider.vision.pose.dlc import to_dlc_csv

        folder = tmp_path / "v"
        folder.mkdir(parents=True, exist_ok=True)
        frames = list(range(first, first + count * stride, stride))
        n_pose = n_pose if n_pose is not None else frames[-1] + 1
        pd.DataFrame({"frame": frames, "behavior": [f"b{f}" for f in frames]}).to_csv(
            folder / "ethogram_raw.csv", index=False
        )
        # x == the frame number, so a misread position is unmistakable.
        xy = np.zeros((n_pose, len(NAMES), 2))
        xy[:, :, 0] = (np.arange(n_pose, dtype=float) + x_origin)[:, None]
        xy[:, :, 1] = 100.0
        to_dlc_csv(
            PoseData(
                xy=xy,
                confidence=np.ones((n_pose, len(NAMES))),
                keypoint_names=NAMES,
                fps=30.0,
                metadata={"resolution": (640, 480)},
            ),
            folder / "vDLC_exp-7.csv",
        )
        return SessionView.load(folder / "ethogram_raw.csv")

    def test_a_windowed_run_reads_its_own_window(self, tmp_path):
        view = self._view(tmp_path, first=3600, count=100)
        frame = tracking_frame(view, None)
        assert frame["frame"].iloc[0] == 3600
        assert frame["center_x"].iloc[0] == pytest.approx(3600.0)
        assert frame["center_x"].iloc[-1] == pytest.approx(3699.0)

    def test_the_label_still_belongs_to_the_position(self, tmp_path):
        view = self._view(tmp_path, first=3600, count=10)
        frame = tracking_frame(view, None)
        assert frame["behavioral_state"].iloc[0] == "b3600"
        assert frame["behavioral_state"].iloc[-1] == "b3609"

    def test_a_sparse_cadence_lands_on_the_frames_it_scored(self, tmp_path):
        """predict_every=3 emits every third frame, not the first third."""
        view = self._view(tmp_path, first=0, count=50, stride=3)
        frame = tracking_frame(view, None)
        assert frame["center_x"].tolist() == pytest.approx([float(f) for f in range(0, 150, 3)])

    def test_a_frame_past_the_end_of_the_track_is_dropped(self, tmp_path):
        """Poses shorter than the ethogram must not wrap or raise."""
        view = self._view(tmp_path, first=0, count=100, n_pose=60)
        frame = tracking_frame(view, None)
        assert len(frame) == 60
        assert frame["frame"].iloc[-1] == 59

    def test_the_heatmap_bins_the_window_that_was_asked_for(self, tmp_path):
        # x runs 100..199 over the scored frames, so reading the track from
        # row zero instead puts every point at a negative x and the histogram
        # comes back empty.
        view = self._view(tmp_path, first=3600, count=100, x_origin=100.0 - 3600)
        grid, _x, _y = occupancy_grid(view, bins=8, start_frame=3600, end_frame=3649)
        assert grid.sum() == 50


def _demo_grid():
    """A deliberately non-square grid, so a transpose cannot hide."""
    grid = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (nx=3, ny=2)
    x_edges = np.array([0.0, 10.0, 20.0, 30.0])  # centres 5, 15, 25
    y_edges = np.array([0.0, 8.0, 16.0])  # centres 4, 12
    return grid, x_edges, y_edges


def test_export_csv_is_rows_y_columns_x(tmp_path):
    from glider.analysis.behavior.spatial import write_occupancy_export

    grid, xe, ye = _demo_grid()
    _png, csv_path = write_occupancy_export(grid, xe, ye, tmp_path / "h")

    table = pd.read_csv(csv_path, index_col=0)
    assert table.shape == (2, 3)  # ny rows, nx columns
    assert [float(c) for c in table.columns] == [5.0, 15.0, 25.0]
    assert list(table.index) == [4.0, 12.0]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            assert table.iat[j, i] == grid[i, j], f"cell (row {j}, col {i})"


def test_export_writes_counts_as_integers(tmp_path):
    from glider.analysis.behavior.spatial import write_occupancy_export

    grid, xe, ye = _demo_grid()
    _png, csv_path = write_occupancy_export(grid, xe, ye, tmp_path / "h")
    # The demo centres (x 5/15/25, y 4/12) are picked so none contains the
    # substring "1.0" -- otherwise this passes or fails for the wrong reason.
    # Actual output: ,5.0,15.0,25.0 / 4.0,1,3,5 / 12.0,2,4,6
    assert "1.0" not in csv_path.read_text()


def test_export_base_suffix_is_replaced_not_appended(tmp_path):
    """Typing heatmap.csv in the dialog must not yield heatmap.csv.csv."""
    from glider.analysis.behavior.spatial import write_occupancy_export

    grid, xe, ye = _demo_grid()
    _png, csv_path = write_occupancy_export(grid, xe, ye, tmp_path / "heatmap.csv")
    assert csv_path == tmp_path / "heatmap.csv"
    assert not (tmp_path / "heatmap.csv.csv").exists()
    assert (tmp_path / "heatmap.png").exists()


def test_export_writes_a_png_without_leaking_a_pyplot_figure(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from glider.analysis.behavior.spatial import write_occupancy_export

    for num in plt.get_fignums():
        plt.close(num)

    grid, xe, ye = _demo_grid()
    png_path, _csv = write_occupancy_export(grid, xe, ye, tmp_path / "h", title="demo")

    assert png_path == tmp_path / "h.png"
    assert png_path.stat().st_size > 0
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "grid,xe,ye",
    [
        (np.zeros((4, 4)), np.arange(5.0), np.arange(5.0)),  # nobody moved
        (np.zeros((60, 60)), np.array([]), np.array([])),  # what an empty track returns
        (np.full((4, 4), np.nan), np.arange(5.0), np.arange(5.0)),  # all-NaN
    ],
    ids=["all-zero", "no-edges", "all-nan"],
)
def test_export_refuses_a_degenerate_grid(tmp_path, grid, xe, ye):
    from glider.analysis.behavior.spatial import write_occupancy_export

    with pytest.raises(ValueError):
        write_occupancy_export(grid, xe, ye, tmp_path / "h")
    assert not list(tmp_path.iterdir())  # and leaves no half-written file


def test_export_without_matplotlib_still_writes_the_csv(tmp_path, monkeypatch):
    """The data half is the half worth keeping."""
    import sys

    from glider.analysis.behavior.spatial import write_occupancy_export

    for name in list(sys.modules):
        if name == "matplotlib" or name.startswith("matplotlib."):
            monkeypatch.setitem(sys.modules, name, None)

    grid, xe, ye = _demo_grid()
    png_path, csv_path = write_occupancy_export(grid, xe, ye, tmp_path / "h")

    assert png_path is None
    assert csv_path.exists()


def test_export_propagates_a_write_failure(tmp_path):
    from glider.analysis.behavior.spatial import write_occupancy_export

    grid, xe, ye = _demo_grid()
    with pytest.raises(OSError):
        write_occupancy_export(grid, xe, ye, tmp_path / "no-such-dir" / "h")
