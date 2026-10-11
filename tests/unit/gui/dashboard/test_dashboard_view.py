import pytest
from PyQt6.QtWidgets import QLabel

from glider.gui.dashboard.dashboard_view import DashboardView


class _StatePanel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.last_state = None

    def update_state(self, name):
        self.last_state = name


@pytest.fixture
def panels():
    return {k: _StatePanel(k) for k in ("camera", "header", "run_control", "device_controls")}


@pytest.fixture
def view(qtbot, panels):
    v = DashboardView(**panels)
    qtbot.addWidget(v)
    return v


def _within(widget, ancestor):
    while widget is not None:
        if widget is ancestor:
            return True
        widget = widget.parent()
    return False


def test_stage_holds_header_and_camera_rail_holds_controls(view, panels):
    assert _within(panels["header"], view.stage())
    assert _within(panels["camera"], view.stage())
    assert _within(panels["run_control"], view.rail())
    assert _within(panels["device_controls"], view.rail())


def test_camera_gets_most_of_the_width(qtbot, view, panels):
    view.resize(1400, 800)
    view.show()
    qtbot.waitExposed(view)
    assert panels["camera"].width() > 2 * view.rail().width()


def test_update_state_fans_out(view, panels):
    view.update_state("RUNNING")
    assert all(p.last_state == "RUNNING" for p in panels.values())


def test_health_overlay(view):
    view.set_health(30.0, 0, recording=True)
    assert view._fps.text() == "30.0 fps"
    assert view._dropped.text() == "0 dropped"
    assert view._dropped.property("warn") is False
    assert view._rec.isVisibleTo(view)
    view.set_health(None, 14, recording=False)
    assert view._fps.text() == "-- fps"
    assert view._dropped.property("warn") is True
    assert not view._rec.isVisibleTo(view)
