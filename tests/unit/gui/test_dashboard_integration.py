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
    window = main_window_factory(desktop_mode=True)  # builds the builder panels
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


def test_camera_host_does_not_linger_over_the_dashboard(qtbot, main_window_factory):
    """The Builder's Camera tab is empty while the dashboard has the panel.

    That used to need hiding, because the camera lived in a dock beside the
    stack. It is inside the Builder frame now, and the Builder frame is the
    stack page the dashboard replaced -- so there is nothing on screen to hide.
    """
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()
    slot = window._camera_tab_slot
    assert _is_descendant(window._camera_panel, slot)

    window.switch_to_runner()  # dashboard borrows the CameraPanel
    assert not slot.isVisible()  # the empty host went with the Builder page

    window.switch_to_builder()  # panel returns to the tab
    assert _is_descendant(window._camera_panel, slot)


def _builder_panels(window):
    return (
        window._node_library_panel,
        window._hardware_panel,
        window._device_control_panel,
        window._files_panel,
        window._properties_host,
    )


def test_builder_panels_do_not_linger_over_the_operator_view(qtbot, main_window_factory):
    """Issue #39, now guarded by structure rather than by a hide/restore helper."""
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()
    assert window._builder_view.isVisible()

    window.switch_to_runner()  # issue #39: builder panels must not linger
    assert not window._builder_view.isVisible()
    for panel in _builder_panels(window):
        assert not panel.isVisible(), type(panel).__name__

    window.switch_to_builder()  # the frame comes back with the builder view
    assert window._builder_view.isVisible()
    assert window._node_library_panel.isVisible()  # the left panel's first tab


def test_switch_to_runner_preserves_the_chosen_panel_tabs(qtbot, main_window_factory):
    """A round trip must not rearrange the Builder the user set up.

    The predecessor of this test guarded a Files *dock* the desktop kept hidden
    by default. Files is a tab now and costs nothing to leave in place, so the
    arrangement worth preserving is which tab each panel is showing.
    """
    window = main_window_factory(desktop_mode=True)
    window.show()
    window.switch_to_builder()
    window._builder_view.left.set_current("files")
    window._builder_view.right.set_current("camera")

    window.switch_to_runner()
    window.switch_to_builder()

    assert window._builder_view.left.current_key() == "files"
    assert window._builder_view.right.current_key() == "camera"


def test_switch_to_desktop_mode_shows_the_builder_frame(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=True)
    window.show()
    window._toggle_view()  # F11 into the dashboard
    assert not window._builder_view.isVisible()
    window._switch_to_desktop_mode()  # bypasses switch_to_builder
    assert window._builder_view.isVisible()
    assert window._node_library_panel.isVisible()
    assert window._stack.currentIndex() == 0
    assert window._view_manager.is_desktop_mode


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
    assert _is_descendant(cam, window._builder_view.right)
    assert window._camera_panel is cam  # still the same single instance


def test_switch_to_desktop_mode_keeps_camera_in_the_tab_on_repeat(qtbot, main_window_factory):
    window = main_window_factory(desktop_mode=False)  # start in runner (no panels yet)
    cam = window._camera_panel
    window._switch_to_desktop_mode()  # first: builds the panels, camera -> tab
    assert _is_descendant(cam, window._builder_view.right)
    window.switch_to_runner()  # camera -> operator view (RunnerShell Camera tab)
    assert _is_descendant(cam, window._runner_shell)
    window._switch_to_desktop_mode()  # repeat: panels exist; camera must STILL move back
    assert _is_descendant(cam, window._builder_view.right)
    assert window._camera_panel is cam  # still the single instance
