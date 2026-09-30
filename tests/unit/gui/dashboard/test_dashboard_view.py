import pytest
from PyQt6.QtWidgets import QLabel

from glider.gui.dashboard.dashboard_view import QUADRANT_TITLES, DashboardView


class _StatePanel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.last_state = None

    def update_state(self, name):
        self.last_state = name


@pytest.fixture
def panels():
    return {
        key: _StatePanel(key)
        for key in ("camera", "device_states", "run_control", "manual_controls")
    }


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


def test_panels_have_fixed_quadrants(view, panels):
    expected = {
        "top_left": "camera",
        "top_right": "device_states",
        "bottom_left": "run_control",
        "bottom_right": "manual_controls",
    }
    for quadrant, key in expected.items():
        assert _within(panels[key], view.quadrant(quadrant)), (quadrant, key)
    assert _within(view._timer, view.quadrant("top_right"))


def test_quadrants_are_equal(qtbot, view):
    view.resize(1000, 800)
    view.show()
    qtbot.waitExposed(view)
    sizes = {view.quadrant(q).size() for q in QUADRANT_TITLES}
    assert len(sizes) == 1


def test_timer_visible_in_every_state(view):
    for state in ("IDLE", "READY", "RUNNING", "PAUSED", "STOPPED"):
        view.update_state(state)
        assert view._timer.isVisibleTo(view), state
        assert view._state.text() == state


def test_set_time_updates_header(view):
    view.set_time("01:23.45")
    assert view._timer.text() == "01:23.45"


def test_rec_shows_only_while_recording_a_run(view):
    view.update_state("RUNNING", recording=True)
    assert view._rec.isVisibleTo(view)
    view.update_state("RUNNING", recording=False)
    assert not view._rec.isVisibleTo(view)
    view.update_state("STOPPED", recording=True)
    assert not view._rec.isVisibleTo(view)


def test_update_state_fans_out_to_panels(view, panels):
    view.update_state("RUNNING")
    for panel in panels.values():
        assert panel.last_state == "RUNNING"
