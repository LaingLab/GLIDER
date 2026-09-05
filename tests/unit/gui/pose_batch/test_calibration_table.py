"""CalibrationTable: per-video calibration status for the batch window."""

from __future__ import annotations

import pytest

from glider.gui.pose_batch.calibration_table import CalibrationTable
from glider.vision.arena import ArenaCalibration
from glider.vision.calibration import CameraCalibration, LengthUnit
from glider.vision.calibration_set import CalibrationSet

TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


def _arena() -> ArenaCalibration:
    return ArenaCalibration(corners=TRAPEZOID, frame_size=(640, 480))


def _cal(px_per_mm: float = 6.4) -> CameraCalibration:
    cal = CameraCalibration()
    cal.add_line(
        start=(0, 240),
        end=(640, 240),
        length=640 / px_per_mm,
        unit=LengthUnit.MILLIMETERS,
        resolution=(640, 480),
    )
    return cal


@pytest.fixture
def table(qtbot):
    widget = CalibrationTable()
    qtbot.addWidget(widget)
    return widget


def test_one_row_per_video(table, tmp_path):
    table.set_videos([tmp_path / "a.mp4", tmp_path / "b.mp4"])
    assert table.rowCount() == 2


def test_uncalibrated_videos_are_flagged(table, tmp_path):
    table.set_videos([tmp_path / "a.mp4"])
    assert "—" in table.item(0, 2).text()
    # The status column reports Run-readiness, and the first thing a bare video
    # needs is the perimeter — the scale follows from it.
    assert "Needs arena" in table.item(0, 4).text()


def test_a_drawn_line_still_shows_its_scale(table, tmp_path):
    """A line supplies px/mm; it just no longer satisfies Run on its own."""
    video = tmp_path / "a.mp4"
    cal_set = CalibrationSet()
    cal_set.set(video, _cal(6.4))
    table.set_calibration_set(cal_set)
    table.set_videos([video])
    assert "6.4" in table.item(0, 2).text()
    assert "640x480" in table.item(0, 1).text()
    assert "Needs arena" in table.item(0, 4).text()


def test_refresh_picks_up_a_new_calibration(table, tmp_path):
    video = tmp_path / "a.mp4"
    cal_set = CalibrationSet()
    table.set_calibration_set(cal_set)
    table.set_videos([video])
    assert "Needs arena" in table.item(0, 4).text()

    cal_set.set_arena(video, _arena())
    table.refresh()
    assert "Calibrated" in table.item(0, 4).text()


def test_selected_videos_reports_the_selection(table, tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    table.set_videos([a, b])
    table.selectRow(1)
    assert table.selected_videos() == [b]


def test_double_click_requests_calibration(table, qtbot, tmp_path):
    video = tmp_path / "a.mp4"
    table.set_videos([video])
    with qtbot.waitSignal(table.calibrate_requested, timeout=1000) as blocker:
        table.itemDoubleClicked.emit(table.item(0, 0))
    assert blocker.args[0] == video
