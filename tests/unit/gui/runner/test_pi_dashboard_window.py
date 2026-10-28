"""The Pi's Dashboard tab as MainWindow builds it, and Switch to Desktop.

In runner mode stack page 0 is the graph over panel tabs, holding the real
Builder widgets. Switching to the desktop layout re-hosts those same objects in
the desktop frame; nothing is built twice.
"""

from __future__ import annotations

import pytest
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


@pytest.mark.parametrize("live", [False, True])
def test_connect_boards_sets_the_ports_then_connects_unless_live(
    qtbot, main_window_factory, monkeypatch, live
):
    """Both doors, the gear item and the Hardware tab's button, connect.

    The board dialog only edits ports; START waits for a connected board.
    """
    w = _pi(qtbot, main_window_factory)
    calls = []
    monkeypatch.setattr(
        w._hardware_panel, "show_board_settings_dialog", lambda: calls.append("dialog")
    )
    monkeypatch.setattr(w, "_on_connect_hardware", lambda: calls.append("connect"))
    monkeypatch.setattr(w, "_run_is_live", lambda: live)
    gear = next(a for a in w._gear_button.menu().actions() if a.text() == "Connect boards…")
    hardware = w._pi_dashboard.tabs().widget(2)
    (button,) = [b for b in hardware.findChildren(QPushButton) if b.text() == "Connect boards…"]
    for tap in (gear.trigger, button.click):
        calls.clear()
        tap()
        assert calls == (["dialog"] if live else ["dialog", "connect"])


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
    assert w._tab_bar.buttons()["tools"].isVisibleTo(w)
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


def test_startup_with_a_file_draws_its_graph_on_the_pi(qtbot, tmp_path):
    """``glider --runner --file X`` showed an empty graph: only the desktop
    branch opened the file the core had already loaded."""
    import asyncio
    import shutil
    from pathlib import Path

    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from glider.core.glider_core import GliderCore
    from glider.gui.main_window import PAGE_OPERATOR, MainWindow
    from glider.gui.view_manager import ViewManager, ViewMode

    experiment = tmp_path / "box_pi_side.glider"
    shutil.copy(Path(__file__).parents[4] / "examples" / "box_pi_side.glider", experiment)
    core = GliderCore()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(core.initialize(load_plugins=False))
    try:
        loop.run_until_complete(core.load_experiment(experiment))  # what --file does
        vm = ViewManager(QApplication.instance())
        vm.mode = ViewMode.RUNNER
        settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
        window = MainWindow(core, view_manager=vm, settings=settings)
        qtbot.addWidget(window, before_close_func=lambda _w: core.session._mark_clean())

        expected = {n.id for n in core.session.flow.nodes}
        assert expected
        assert set(window._graph_view.nodes) == expected
        assert window._stack.currentIndex() == PAGE_OPERATOR  # still opens on Run
    finally:
        loop.run_until_complete(core.shutdown())
        loop.close()
