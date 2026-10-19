"""Per-camera status while recording.

At sixteen cameras the failure that matters is one starved writer: the run
looks fine, and one animal's file is quietly short. Drops are counted already;
this is what puts them where an operator can see them mid-session.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt

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


def _col(table, name: str) -> int:
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    return headers.index(name)


class TestSwitchingCamerasOnAndOff:
    def test_every_camera_starts_ticked(self, table):
        table.set_cameras(["cam_0", "cam_1"])
        assert table.is_enabled("cam_0") and table.is_enabled("cam_1")
        assert table.item(1, 0).checkState() == Qt.CheckState.Checked

    def test_unticking_announces_it(self, table, qtbot):
        table.set_cameras(["cam_0", "cam_1"])
        with qtbot.waitSignal(table.enabled_toggled) as blocker:
            table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
        assert blocker.args == ["cam_1", False]
        assert not table.is_enabled("cam_1")

    def test_setting_it_from_code_does_not_read_as_the_operator(self, table):
        # Otherwise every refresh would look like someone clicking every box.
        table.set_cameras(["cam_0", "cam_1"])
        heard = []
        table.enabled_toggled.connect(lambda *a: heard.append(a))
        table.set_enabled("cam_1", False)
        table.update_status("cam_0", fps=30.0, queue_depth=0, dropped=0, recording=True)
        assert heard == []

    def test_a_switched_off_camera_reads_off(self, table):
        table.set_cameras(["cam_0"])
        table.set_enabled("cam_0", False)
        assert table.item(0, _col(table, "State")).text() == "off"
        assert table.item(0, 0).checkState() == Qt.CheckState.Unchecked

    def test_a_switched_off_camera_is_never_flagged(self, table):
        table.set_cameras(["cam_0"])
        table.set_enabled("cam_0", False)
        table.update_status("cam_0", fps=0.0, queue_depth=0, dropped=5, recording=True)
        assert not table.is_flagged("cam_0")
        assert table.item(0, _col(table, "State")).text() == "off"

    def test_it_stays_off_across_a_rebuild(self, table):
        table.set_cameras(["cam_0", "cam_1"])
        table.set_enabled("cam_1", False)
        table.set_cameras(["cam_0", "cam_1"])
        assert not table.is_enabled("cam_1")

    def test_a_recording_camera_cannot_be_unticked(self, table):
        # Unticking disconnects; that would cut its file off unfinished.
        table.set_cameras(["cam_0"])
        table.update_status("cam_0", fps=30.0, queue_depth=0, dropped=0, recording=True)
        assert not (table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable)

    def test_it_can_be_unticked_again_once_stopped(self, table):
        table.set_cameras(["cam_0"])
        table.update_status("cam_0", fps=30.0, queue_depth=0, dropped=0, recording=True)
        table.update_status("cam_0", fps=30.0, queue_depth=0, dropped=0, recording=False)
        assert table.item(0, 0).flags() & Qt.ItemFlag.ItemIsUserCheckable


class TestRecordButtons:
    def test_each_camera_has_one(self, table):
        table.set_cameras(["cam_0", "cam_1", "cam_2"])
        assert all(table.record_button(c) is not None for c in ("cam_0", "cam_1", "cam_2"))

    def test_it_names_its_camera(self, table, qtbot):
        table.set_cameras(["cam_0", "cam_1"])
        with qtbot.waitSignal(table.record_toggled) as blocker:
            table.record_button("cam_1").click()
        assert blocker.args == ["cam_1"]

    def test_it_follows_the_camera(self, table):
        table.set_cameras(["cam_0"])
        button = table.record_button("cam_0")
        assert button.text() == "Record" and button.isEnabled()

        table.update_status("cam_0", fps=30.0, queue_depth=0, dropped=0, recording=True)
        assert button.text() == "Stop" and button.isEnabled()

        table.update_status(
            "cam_0", fps=30.0, queue_depth=0, dropped=0, recording=False, saving=True
        )
        assert button.text() == "Saving…" and not button.isEnabled()
        assert table.item(0, _col(table, "State")).text() == "saving…"

    def test_a_switched_off_camera_cannot_be_recorded(self, table):
        table.set_cameras(["cam_0"])
        table.set_enabled("cam_0", False)
        assert not table.record_button("cam_0").isEnabled()


class TestTimeAndNames:
    def test_the_time_column_shows_how_long_the_take_has_run(self, table):
        table.set_cameras(["cam_0"])
        table.update_status(
            "cam_0", fps=30.0, queue_depth=0, dropped=0, recording=True, elapsed=75.0
        )
        assert table.item(0, _col(table, "Time")).text() == "01:15"

    def test_a_name_replaces_the_id(self, table):
        table.set_cameras(["cam_0", "cam_1"], names={"cam_1": "arena B"})
        assert table.item(1, 0).text() == "arena B"
        table.set_display_name("cam_0", "arena A")
        assert table.item(0, 0).text() == "arena A"


class TestButtonsFit:
    def test_a_padded_button_is_not_cut_off(self, qtbot):
        # Found in screenshots, twice: rows sized for text cut every Record
        # button down to "ecor", and then the tool theme's item padding still
        # shrank the button inside a row tall enough for it. The padding here
        # is exaggerated so the test does not depend on the installed fonts.
        from PyQt6.QtWidgets import QApplication, QMainWindow

        window = QMainWindow()
        qtbot.addWidget(window)
        window.setStyleSheet(
            "QPushButton { padding: 20px 40px; } QTableView::item { padding: 10px 16px; }"
        )
        table = CameraStatusTable()
        window.setCentralWidget(table)
        table.set_cameras(["cam_0", "cam_1"])
        window.resize(900, 400)
        window.show()
        qtbot.waitExposed(window)
        QApplication.processEvents()

        button = table.record_button("cam_0")
        assert button.height() >= button.sizeHint().height()
        assert button.width() >= button.sizeHint().width()
