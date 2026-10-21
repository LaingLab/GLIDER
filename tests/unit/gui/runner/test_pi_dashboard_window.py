"""The Pi's Dashboard tab as MainWindow builds it, and Switch to Desktop.

In runner mode stack page 0 is the graph over panel tabs, holding the real
Builder widgets. Switching to the desktop layout re-hosts those same objects in
the desktop frame; nothing is built twice.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QPushButton

from glider.gui.main_window import PAGE_BUILDER
from glider.gui.runner.pi_dashboard import PiDashboard


def _is_descendant(child, ancestor):
    w = child
    while w is not None:
        if w is ancestor:
            return True
        w = w.parentWidget()
    return False


def _pi(qtbot, main_window_factory):
    w = main_window_factory(desktop_mode=False)
    w.resize(480, 800)
    w.show()
    # A runner window shows itself before its central widget exists, so Qt only
    # shows (and lays out) the header on the next pass of the event loop.
    qtbot.waitUntil(w._pi_header.isVisible)
    return w


def test_dashboard_tab_is_graph_over_panel_tabs(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w.switch_to_builder()
    page = w._stack.widget(PAGE_BUILDER)
    assert isinstance(page, PiDashboard)
    assert _is_descendant(w._graph_view, page)
    assert page.keys() == ["nodes", "properties", "hardware", "camera", "control"]
    assert _is_descendant(w._camera_panel, page)
    assert w._graph_view.dragMode().name == "ScrollHandDrag"
    # The desktop frame is off the stack, so it no longer sets the width floor.
    assert w.minimumSizeHint().width() <= 480


def test_panel_tabs_hold_the_real_widgets(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    tabs = w._pi_dashboard.tabs()
    assert tabs.widget(0) is w._node_library_panel
    assert tabs.widget(1) is w._properties_host
    assert tabs.widget(3) is w._camera_tab_slot
    assert tabs.widget(4) is w._runner_device_controls
    hardware = tabs.widget(2)
    assert _is_descendant(w._hardware_panel, hardware)
    (connect,) = [b for b in hardware.findChildren(QPushButton) if b.text() == "Connect boards…"]
    assert connect.minimumHeight() >= 44


def test_connect_boards_opens_the_board_dialog(qtbot, main_window_factory, monkeypatch):
    w = _pi(qtbot, main_window_factory)
    opened = []
    monkeypatch.setattr(w._hardware_panel, "show_board_settings_dialog", lambda: opened.append(1))
    hardware = w._pi_dashboard.tabs().widget(2)
    (connect,) = [b for b in hardware.findChildren(QPushButton) if b.text() == "Connect boards…"]
    connect.click()
    assert opened == [1]


def test_control_tab_refreshes_the_manual_controls_when_shown(
    qtbot, main_window_factory, monkeypatch
):
    w = _pi(qtbot, main_window_factory)
    calls = []
    monkeypatch.setattr(w._runner_device_controls, "refresh", lambda: calls.append(1))
    w._pi_dashboard.set_current("hardware")
    assert calls == []
    w._pi_dashboard.set_current("control")
    assert calls == [1]


def test_graph_starts_locked_and_the_lock_covers_properties_and_hardware(
    qtbot, main_window_factory
):
    w = _pi(qtbot, main_window_factory)
    assert w._graph_view.is_locked()
    assert not w._properties_host.isEnabled()
    assert w._hardware_panel._add_row.isHidden()
    w._graph_view.set_locked(False)
    assert w._properties_host.isEnabled()
    assert not w._hardware_panel._add_row.isHidden()
    w._graph_view.set_locked(True)
    assert not w._properties_host.isEnabled()
    assert w._hardware_panel._add_row.isHidden()


def test_switch_to_desktop_rehosts_without_duplicates(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    hw, cam, graph, lib = w._hardware_panel, w._camera_panel, w._graph_view, w._node_library_panel
    w._switch_to_desktop_mode()
    assert w._stack.widget(PAGE_BUILDER) is w._builder_view
    assert w._builder_view.centre is graph
    assert (w._hardware_panel, w._camera_panel, w._node_library_panel) == (hw, cam, lib)
    assert w._builder_view.left.widget_for("hardware") is hw
    assert cam.parent() is w._camera_tab_slot
    assert _is_descendant(cam, w._builder_view.right)
    assert w._tab_bar.buttons()["analyze"].isVisibleTo(w)
    assert not w._gear_button.isVisibleTo(w)
    assert graph.dragMode().name == "RubberBandDrag" and not graph.is_locked()
    # The Pi lowered the graph's floor for landscape; the desktop gets its own back.
    assert (graph.minimumWidth(), graph.minimumHeight()) == (400, 300)


def test_switch_to_desktop_restores_the_desktop_chrome(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._switch_to_desktop_mode()
    strip = w._status_strip
    assert strip.maximumHeight() == 40
    assert w._tab_bar.parentWidget() is strip.parentWidget()
    assert strip.parentWidget().objectName() == "shellChromeRow"
    assert not w._pi_header.isVisibleTo(w) and not w._notice_bar.isVisibleTo(w)
    # On the Dashboard the desktop shows its panel toggles again.
    assert strip.left_toggle().isVisibleTo(w)
    assert strip.palette_hint().isVisibleTo(w)


def test_switch_to_desktop_is_refused_mid_run(qtbot, main_window_factory, monkeypatch):
    w = _pi(qtbot, main_window_factory)
    monkeypatch.setattr(w, "_run_is_live", lambda: True)
    w._switch_to_desktop_mode()
    assert isinstance(w._stack.widget(PAGE_BUILDER), PiDashboard)
    assert w.is_runner_mode
    assert _is_descendant(w._graph_view, w._pi_dashboard)
    assert w._notice_bar.text().startswith("Experiment running")
