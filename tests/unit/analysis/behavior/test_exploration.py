"""Grid exploration over the drawn arena: how many floor cells the animal entered."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.exploration import (
    arena_file_for,
    cell_polygon,
    explore,
    find_session_arena,
    floor_cells,
    grid_segments,
    read_arena,
)
from glider.analysis.behavior.session_view import SessionView
from glider.vision.arena import ArenaCalibration

NAMES = ["nose", "l_ear", "r_ear", "tail_base"]


def _view(tmp_path, xs, ys, resolution=(640, 480), fps=30.0):
    """A session whose animal walks a path we choose."""
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    folder = tmp_path / "v"
    folder.mkdir(parents=True, exist_ok=True)
    n = len(xs)
    pd.DataFrame({"frame": range(n), "behavior": ["groom"] * n}).to_csv(
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
            metadata={"resolution": resolution},
        ),
        folder / "vDLC_exp-7.csv",
    )
    return SessionView.load(folder / "ethogram_raw.csv")


def _full_frame(shape="rectangle"):
    """An arena filling the whole 640x480 frame, 40 x 30 cm."""
    return ArenaCalibration(
        corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
        width_cm=40.0,
        height_cm=30.0,
        frame_size=(640, 480),
        shape=shape,
    )


class TestGeometry:
    def test_a_rectangle_is_all_floor(self):
        assert floor_cells(_full_frame(), 3).all()

    def test_a_round_arena_drops_the_box_corners(self):
        floor = floor_cells(_full_frame("ellipse"), 4)
        assert not floor[0, 0] and not floor[3, 3]
        assert floor[1:3, 1:3].all()

    def test_grid_lines_split_the_floor(self):
        lines = grid_segments(_full_frame(), 3)
        assert len(lines) == 4  # two vertical, two horizontal
        xs = sorted(round(line[0][0], 6) for line in lines if line[0][0] == line[-1][0])
        assert xs == pytest.approx([1 / 3, 2 / 3])

    def test_no_lines_without_a_grid(self):
        assert grid_segments(_full_frame(), 1) == []

    def test_grid_follows_perspective(self):
        """A far wall drawn narrower gives far cells drawn smaller."""
        tilted = ArenaCalibration(
            corners=[(0.3, 0.1), (0.7, 0.1), (0.9, 0.9), (0.1, 0.9)],
            width_cm=30.0,
            height_cm=30.0,
        )
        top = cell_polygon(tilted, 3, 1, 0)
        bottom = cell_polygon(tilted, 3, 1, 2)
        assert abs(top[1][0] - top[0][0]) < abs(bottom[1][0] - bottom[0][0])


class TestExplore:
    def test_counts_the_cells_entered(self, tmp_path):
        # Along the top row of a 3x3 grid only: three of nine cells.
        xs = np.linspace(10, 630, 90)
        view = _view(tmp_path, xs, np.full(90, 40.0))
        result = explore(view, _full_frame(), 3)
        assert result.cells == 9
        assert result.cells_visited == 3
        assert result.fraction == pytest.approx(1 / 3)
        assert result.full_coverage_s is None
        assert result.visited[0].all() and not result.visited[1:].any()

    def test_time_in_cells_adds_up_to_the_window(self, tmp_path):
        view = _view(tmp_path, np.linspace(10, 630, 90), np.full(90, 40.0))
        result = explore(view, _full_frame(), 3)
        assert np.nansum(result.seconds) == pytest.approx(90 / 30.0)

    def test_full_coverage_time(self, tmp_path):
        # Visit the four cells of a 2x2 grid, 30 frames each, in turn.
        xs = np.repeat([100, 500, 500, 100], 30)
        ys = np.repeat([100, 100, 400, 400], 30)
        result = explore(_view(tmp_path, xs, ys), _full_frame(), 2)
        assert result.fraction == 1.0
        assert result.full_coverage_s == pytest.approx(90 / 30.0)

    def test_the_window_limits_it(self, tmp_path):
        xs = np.repeat([100, 500, 500, 100], 30)
        ys = np.repeat([100, 100, 400, 400], 30)
        result = explore(_view(tmp_path, xs, ys), _full_frame(), 2, start_frame=0, end_frame=59)
        assert result.cells_visited == 2

    def test_positions_off_the_floor_do_not_count(self, tmp_path):
        """An arena in the middle of the frame: the animal outside it explores nothing."""
        arena = ArenaCalibration(corners=[(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
        result = explore(_view(tmp_path, np.full(30, 20.0), np.full(30, 20.0)), arena, 3)
        assert result.cells_visited == 0

    def test_a_grid_needs_a_cell(self, tmp_path):
        with pytest.raises(ValueError):
            explore(_view(tmp_path, [1, 2], [1, 2]), _full_frame(), 0)


class TestFindingTheArena:
    def test_reads_a_bare_and_a_wrapped_arena(self, tmp_path):
        arena = _full_frame()
        bare, wrapped = tmp_path / "a.json", tmp_path / "b.json"
        bare.write_text(json.dumps(arena.to_dict()))
        wrapped.write_text(json.dumps({"arena": arena.to_dict(), "px_per_cm_centre": 16.0}))
        assert read_arena(bare) == arena
        assert read_arena(wrapped) == arena

    def test_an_unreadable_file_is_none(self, tmp_path):
        junk = tmp_path / "x.json"
        junk.write_text("not json")
        assert read_arena(junk) is None

    def test_a_saved_arena_beside_the_session_is_found(self, tmp_path):
        view = _view(tmp_path, [1, 2], [1, 2])
        path = arena_file_for(tmp_path / "v" / "ethogram_raw.csv")
        assert path == tmp_path / "v" / "v_arena.json"
        path.write_text(json.dumps(_full_frame().to_dict()))
        found = find_session_arena(tmp_path / "v" / "ethogram_raw.csv", view)
        assert found == (_full_frame(), path)

    def test_the_batch_calibration_is_found_by_video_stem(self, tmp_path):
        """The master file keys arenas by the video the batch saw, often elsewhere now."""
        from glider.vision.calibration import CameraCalibration
        from glider.vision.calibration_set import CalibrationSet

        view = _view(tmp_path, [1, 2], [1, 2])
        calibrations = CalibrationSet()
        elsewhere = tmp_path / "gone" / "v.mp4"
        calibrations.set(elsewhere, CameraCalibration())
        calibrations.set_arena(elsewhere, _full_frame())
        calibrations.save(tmp_path / "pose_calibration.json")
        found = find_session_arena(tmp_path / "v" / "ethogram_raw.csv", view)
        assert found is not None and found[0] == _full_frame()

    def test_none_when_nothing_was_drawn(self, tmp_path):
        view = _view(tmp_path, [1, 2], [1, 2])
        assert find_session_arena(tmp_path / "v" / "ethogram_raw.csv", view) is None
