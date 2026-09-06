"""Per-camera status while recording.

At sixteen cameras the failure that matters is one starved writer: the run
looks fine, and one animal's file is quietly short. Drops are counted already;
this is what puts them where an operator can see them mid-session.
"""

from __future__ import annotations

import pytest

from glider.gui.multi_camera.status_table import CameraStatusTable


@pytest.fixture
def table(qtbot):
    widget = CameraStatusTable()
    qtbot.addWidget(widget)
    return widget


class TestRows:
    def test_one_row_per_camera(self, table):
        table.set_cameras([f"cam_{i}" for i in range(16)])
        assert table.rowCount() == 16

    def test_rows_follow_the_order_given(self, table):
        table.set_cameras(["cam_2", "cam_0", "cam_1"])
        assert [table.item(r, 0).text() for r in range(3)] == ["cam_2", "cam_0", "cam_1"]

    def test_setting_cameras_again_replaces_them(self, table):
        table.set_cameras(["cam_0", "cam_1"])
        table.set_cameras(["cam_5"])
        assert table.rowCount() == 1
        assert table.item(0, 0).text() == "cam_5"

    def test_no_cameras_is_not_an_error(self, table):
        table.set_cameras([])
        assert table.rowCount() == 0


class TestStatus:
    def test_fps_and_drops_are_shown(self, table):
        table.set_cameras(["cam_0"])
        table.update_status("cam_0", fps=29.9, queue_depth=3, dropped=0, recording=True)
        row = [table.item(0, c).text() for c in range(table.columnCount())]
        assert "29.9" in row[1]
        assert "3" in row[2]
        assert "0" in row[3]

    def test_a_camera_with_drops_is_flagged(self, table):
        # The whole point: this has to be visible without reading the log.
        table.set_cameras(["cam_0", "cam_1"])
        table.update_status("cam_0", fps=30.0, queue_depth=0, dropped=0, recording=True)
        table.update_status("cam_1", fps=30.0, queue_depth=0, dropped=12, recording=True)
        assert not table.is_flagged("cam_0")
        assert table.is_flagged("cam_1")

    def test_a_stalled_camera_is_flagged(self, table):
        # Zero fps while recording means the capture thread has died; the file
        # keeps existing but stops growing.
        table.set_cameras(["cam_0"])
        table.update_status("cam_0", fps=0.0, queue_depth=0, dropped=0, recording=True)
        assert table.is_flagged("cam_0")

    def test_zero_fps_before_recording_is_not_a_problem(self, table):
        table.set_cameras(["cam_0"])
        table.update_status("cam_0", fps=0.0, queue_depth=0, dropped=0, recording=False)
        assert not table.is_flagged("cam_0")

    def test_updating_an_unknown_camera_is_ignored(self, table):
        table.set_cameras(["cam_0"])
        table.update_status("cam_9", fps=30.0, queue_depth=0, dropped=0, recording=True)
        assert table.rowCount() == 1

    def test_any_flagged_summarises_the_whole_rig(self, table):
        table.set_cameras(["cam_0", "cam_1"])
        table.update_status("cam_0", fps=30.0, queue_depth=0, dropped=0, recording=True)
        assert not table.any_flagged()
        table.update_status("cam_1", fps=30.0, queue_depth=0, dropped=5, recording=True)
        assert table.any_flagged()
