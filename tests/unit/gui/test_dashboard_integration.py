import pytest

pytestmark = pytest.mark.usefixtures("qtbot")


def test_main_window_builds_dashboard_view(qtbot, main_window_factory):
    window = main_window_factory()
    from glider.gui.dashboard.dashboard_view import DashboardView

    assert isinstance(window._dashboard_view, DashboardView)
    assert window._stack.widget(1) is window._dashboard_view


class _FakeState:
    def __init__(self, name):
        self.name = name


def test_state_change_fans_out_to_dashboard(qtbot, main_window_factory):
    window = main_window_factory()
    window.show()
    window._on_core_state_change(_FakeState("RUNNING"))
    assert window._run_control_panel._state_name == "RUNNING"


def test_banner_visible_only_when_running_and_run_control_not_shown(qtbot, main_window_factory):
    window = main_window_factory()
    window.switch_to_runner()
    window.show()
    window._dashboard_view.host("top_left").trigger_pick("manual_controls")  # bench run_control
    window._on_core_state_change(_FakeState("RUNNING"))
    assert window._dashboard_view._banner.isVisibleTo(window._dashboard_view)


def test_banner_hidden_when_run_control_is_shown(qtbot, main_window_factory):
    window = main_window_factory()
    window.switch_to_runner()
    window.show()
    window._on_core_state_change(_FakeState("RUNNING"))  # default layout shows run_control top_left
    assert not window._dashboard_view._banner.isVisibleTo(window._dashboard_view)


def test_can_switch_builder_dashboard_bidirectionally(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)  # builds desktop docks at startup
    window.switch_to_runner()
    assert window._stack.currentIndex() == 1
    window.switch_to_builder()
    assert window._stack.currentIndex() == 0
    window.switch_to_runner()  # must still work the second time
    assert window._stack.currentIndex() == 1


def test_toggle_view_switches_both_ways(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    window.switch_to_builder()
    window._toggle_view()  # -> dashboard
    assert window._stack.currentIndex() == 1
    window._toggle_view()  # -> builder
    assert window._stack.currentIndex() == 0


def test_entering_dashboard_refreshes_hardware(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    calls = []
    window._dash_hardware_panel.refresh_tree = lambda: calls.append(1)
    window.switch_to_builder()
    calls.clear()
    window._toggle_view()  # interactive entry into dashboard
    assert window._stack.currentIndex() == 1
    assert len(calls) >= 1  # refresh fired on entry via the menu-toggle path


def _is_descendant(widget, ancestor):
    p = widget.parent()
    while p is not None:
        if p is ancestor:
            return True
        p = p.parent()
    return False


def test_single_camera_panel_reparented_between_views(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    cam = window._camera_panel
    assert cam is not None
    window.switch_to_runner()
    assert _is_descendant(cam, window._dashboard_view)
    window.switch_to_builder()
    assert _is_descendant(cam, window._camera_dock)
    assert window._camera_panel is cam  # still the same single instance


def test_switch_to_desktop_mode_keeps_camera_in_dock_on_repeat(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=False)  # start in runner (no docks yet)
    cam = window._camera_panel
    window._switch_to_desktop_mode()  # first: builds docks, camera -> dock
    assert _is_descendant(cam, window._camera_dock)
    window.switch_to_runner()  # camera -> dashboard slot
    assert _is_descendant(cam, window._dashboard_view)
    window._switch_to_desktop_mode()  # repeat: docks exist; camera must STILL move to dock
    assert _is_descendant(cam, window._camera_dock)
    assert window._camera_panel is cam  # still the single instance
