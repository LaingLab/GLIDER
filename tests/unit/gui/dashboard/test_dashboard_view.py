import pytest
from PyQt6.QtWidgets import QLabel

from glider.gui.dashboard.dashboard_view import DashboardView
from glider.gui.dashboard.layout import QUADRANTS, default_layout
from glider.gui.dashboard.layout_store import load_layout
from glider.gui.dashboard.panel_registry import PANEL_KEYS


class _StatePanel(QLabel):
    def __init__(self):
        super().__init__()
        self.last_state = None

    def update_state(self, name):
        self.last_state = name


def _panels():
    return {key: QLabel(key) for key in PANEL_KEYS}


@pytest.fixture
def view(qtbot, tmp_path):
    v = DashboardView(_panels(), save_path=tmp_path / "layout.json")
    qtbot.addWidget(v)
    return v


def test_four_quadrants_show_default_panels(view):
    shown = {q: view.host(q).current_panel_key for q in QUADRANTS}
    assert shown == default_layout().assignment


def test_benched_panel_is_alive_but_not_in_a_quadrant(view):
    benched = view.current_layout().benched_panel()
    assert benched == "manual_controls"
    assert benched not in {view.host(q).current_panel_key for q in QUADRANTS}
    assert view.panel(benched).parent() is not None


def test_pick_swaps_and_persists(view, tmp_path):
    view.host("top_left").trigger_pick("device_states")
    assert view.host("top_left").current_panel_key == "device_states"
    assert view.host("top_right").current_panel_key == "run_control"
    assert load_layout(tmp_path / "layout.json").assignment["top_left"] == "device_states"


def test_drag_swap_exchanges_quadrants(view):
    view.host("top_left").swap_requested.emit("top_left", "bottom_right")
    assert view.host("top_left").current_panel_key == "experiment_info"
    assert view.host("bottom_right").current_panel_key == "run_control"
    assert view.panel("experiment_info").parent() is not None
    assert view.panel("run_control").parent() is not None


def test_pick_benched_panel_moves_it_in(view):
    view.host("top_left").trigger_pick("manual_controls")
    assert view.host("top_left").current_panel_key == "manual_controls"
    assert view.current_layout().benched_panel() == "run_control"
    assert view.panel("run_control").parent() is not None


def test_update_state_fans_out_including_benched(qtbot):
    panels = {key: _StatePanel() for key in PANEL_KEYS}
    from glider.gui.dashboard.dashboard_view import DashboardView

    v = DashboardView(panels, save_path=None)
    qtbot.addWidget(v)
    v.update_state("RUNNING")
    for key in PANEL_KEYS:  # benched included
        assert panels[key].last_state == "RUNNING"


def test_set_banner_time_forwards_to_banner(qtbot):
    from PyQt6.QtWidgets import QLabel

    class _Banner(QLabel):
        def __init__(self):
            super().__init__()
            self.time_text = None

        def set_time(self, text):
            self.time_text = text

    banner = _Banner()
    v = DashboardView(_panels(), save_path=None, banner=banner)
    qtbot.addWidget(v)
    v.set_banner_time("01:23.45")
    assert banner.time_text == "01:23.45"


def test_restores_saved_layout_on_construction(qtbot, tmp_path):
    from glider.gui.dashboard.layout import default_layout as _dl
    from glider.gui.dashboard.layout_store import save_layout

    custom = _dl().with_assignment(
        {
            "top_left": "camera",
            "top_right": "run_control",
            "bottom_left": "device_states",
            "bottom_right": "manual_controls",
        }
    )
    save_layout(custom, tmp_path / "layout.json")
    v = DashboardView(_panels(), save_path=tmp_path / "layout.json")
    qtbot.addWidget(v)
    assert v.host("top_left").current_panel_key == "camera"
