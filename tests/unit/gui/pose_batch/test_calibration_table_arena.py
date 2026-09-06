"""CalibrationTable: the arena column and how it reaches the dialog."""

from __future__ import annotations

import pytest

from glider.gui.pose_batch.calibration_table import ARENA_COLUMN, CalibrationTable
from glider.vision.arena import ArenaCalibration
from glider.vision.calibration import CameraCalibration, LengthUnit
from glider.vision.calibration_set import CalibrationSet

TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]
# Opposite edges disagree badly enough to read as a dragged corner.
LOPSIDED = [(0.3, 0.1), (0.95, 0.1), (0.5, 0.9), (0.2, 0.9)]


@pytest.fixture
def table(qtbot):
    widget = CalibrationTable()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "t1_d2.mp4"
    path.touch()
    return path


def _line(px_per_mm: float = 6.4) -> CameraCalibration:
    cal = CameraCalibration()
    cal.add_line(
        start=(0, 240),
        end=(640, 240),
        length=640 / px_per_mm,
        unit=LengthUnit.MILLIMETERS,
        resolution=(640, 480),
    )
    return cal


def _with_arena(video, corners=TRAPEZOID, *, confirmed: bool = True) -> CalibrationSet:
    cal_set = CalibrationSet()
    cal_set.set_arena(
        video,
        ArenaCalibration(corners=corners, frame_size=(640, 480)),
        confirmed=confirmed,
    )
    return cal_set


class TestArenaColumn:
    def test_no_arena_shows_a_dash(self, table, video):
        table.set_videos([video])
        assert table.item(0, ARENA_COLUMN).text() == "—"

    def test_a_drawn_arena_is_marked(self, table, video):
        table.set_calibration_set(_with_arena(video))
        table.set_videos([video])
        assert "✓" in table.item(0, ARENA_COLUMN).text()

    def test_a_suspect_arena_is_flagged(self, table, video):
        table.set_calibration_set(_with_arena(video, LOPSIDED))
        table.set_videos([video])
        assert "⚠" in table.item(0, ARENA_COLUMN).text()

    def test_a_degenerate_arena_does_not_crash_the_table(self, table, video):
        # A half-drawn perimeter must not take the whole window down.
        table.set_calibration_set(_with_arena(video, [(0.5, 0.5)] * 4))
        table.set_videos([video])
        assert table.item(0, ARENA_COLUMN) is not None


class TestScaleSource:
    def test_an_arena_supplies_the_scale_column(self, table, video):
        cal_set = _with_arena(video)
        table.set_calibration_set(cal_set)
        table.set_videos([video])
        expected = cal_set.px_per_mm(video)
        assert f"{expected:.3f}" in table.item(0, 2).text()

    def test_a_video_with_only_an_arena_reads_as_calibrated(self, table, video):
        table.set_calibration_set(_with_arena(video))
        table.set_videos([video])
        assert "Calibrated" in table.item(0, 4).text()

    def test_the_arena_scale_wins_over_the_line(self, table, video):
        cal_set = _with_arena(video)
        cal_set.set(video, _line(6.4))
        table.set_calibration_set(cal_set)
        table.set_videos([video])
        assert "6.400" not in table.item(0, 2).text()


class TestStatusColumn:
    """The status column answers "can this run", not "is there a scale".

    A row reading "Calibrated" beside a disabled Run button sends the operator
    hunting for a blocker the table refused to name.
    """

    @pytest.mark.parametrize(
        "state, expected",
        [
            (None, "Needs arena"),
            ("unconfirmed", "confirm it"),
        ],
    )
    def test_status_column_reports_arena_state(self, table, video, state, expected):
        cal_set = _with_arena(video, confirmed=False) if state else CalibrationSet()
        table.set_calibration_set(cal_set)
        table.set_videos([video])
        assert expected in table.item(0, 4).text()

    def test_a_confirmed_arena_still_reads_as_calibrated(self, table, video):
        table.set_calibration_set(_with_arena(video))
        table.set_videos([video])
        assert "Calibrated" in table.item(0, 4).text()

    def test_a_drawn_line_alone_does_not_read_as_calibrated(self, table, video):
        """The line still supplies a scale, but Run will not accept it."""
        cal_set = CalibrationSet()
        cal_set.set(video, _line())
        table.set_calibration_set(cal_set)
        table.set_videos([video])
        assert "Needs arena" in table.item(0, 4).text()


class TestRequests:
    def test_double_click_on_the_arena_column_asks_for_the_arena(self, table, video, qtbot):
        table.set_videos([video])
        with qtbot.waitSignal(table.arena_requested, timeout=500) as blocker:
            table.itemDoubleClicked.emit(table.item(0, ARENA_COLUMN))
        assert blocker.args == [video]

    def test_double_click_elsewhere_still_asks_for_the_line(self, table, video, qtbot):
        table.set_videos([video])
        with qtbot.waitSignal(table.calibrate_requested, timeout=500) as blocker:
            table.itemDoubleClicked.emit(table.item(0, 0))
        assert blocker.args == [video]
