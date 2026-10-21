"""The Pi's Run tab as MainWindow builds it.

In runner mode stack page 1 is the Pi Run page, not the old four-tab shell. A
live run forces the graph lock, the header pill carries the run clock while you
are elsewhere, and the camera box only appears when a stream is running.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QStatusBar

from glider.gui.main_window import PAGE_OPERATOR
from glider.gui.runner.run_page import PiRunPage


def _pi(qtbot, main_window_factory):
    w = main_window_factory(desktop_mode=False)
    w.resize(480, 800)
    w.show()
    # The header lays out one event-loop pass after show.
    qtbot.waitUntil(w._pi_header.isVisible)
    return w


def _state(name):
    return type("S", (), {"name": name})()


def test_run_tab_is_the_pi_run_page_with_one_nav_bar(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    assert isinstance(w._stack.widget(PAGE_OPERATOR), PiRunPage)
    assert w._stack.currentIndex() == PAGE_OPERATOR  # the kiosk opens on Run
    assert not hasattr(w, "_runner_shell") or w._runner_shell is None


def test_run_page_is_the_runs_header_and_controls(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    page = w._pi_run_page
    assert w._run_header is page.header
    assert w._run_control_panel is page.control
    assert w._camera_slot is page.camera_slot


def test_a_run_forces_the_lock_and_frees_the_toggle_after(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    g = w._graph_view
    g.set_locked(False)
    w._on_core_state_change(_state("RUNNING"))
    assert g.is_locked() and not g._overlay.lock_btn.isEnabled()
    w._on_core_state_change(_state("READY"))
    assert g.is_locked() and g._overlay.lock_btn.isEnabled()


def test_run_state_reaches_the_page_and_the_manual_controls(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    seen = []
    w._runner_device_controls.update_state = seen.append
    w._on_core_state_change(_state("RUNNING"))
    assert seen == ["RUNNING"]
    assert w._pi_run_page.clock.property("live") is True


def test_rec_pill_shows_minutes_and_goes_to_run(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._on_core_state_change(_state("RUNNING"))
    w._pi_run_page.control.elapsed_updated.emit("12:41.36")
    assert w._status_strip.run_detail() == "12:41"
    w.switch_to_builder()
    w._status_strip.pill_clicked.emit()
    assert w._stack.currentIndex() == PAGE_OPERATOR


def test_pill_does_nothing_when_no_run_is_live(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._pi_run_page.control.elapsed_updated.emit("00:03.10")
    assert w._status_strip.run_detail() != "00:03"
    w.switch_to_builder()
    w._status_strip.pill_clicked.emit()
    assert w._stack.currentIndex() != PAGE_OPERATOR


def test_no_empty_camera_box_without_a_stream(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._pi_run_page.set_camera_visible(True)  # left up by a stream that has since stopped
    w._enter_dashboard()
    assert not w._pi_run_page.camera_slot.isVisibleTo(w)


def test_camera_box_shows_while_a_stream_runs(qtbot, main_window_factory, monkeypatch):
    w = _pi(qtbot, main_window_factory)
    monkeypatch.setattr(type(w._core.camera_manager), "is_streaming", property(lambda _: True))
    w._enter_dashboard()
    assert w._pi_run_page.camera_slot.isVisibleTo(w)
    w._pi_run_page.set_camera_visible(False)
    w._on_core_state_change(_state("RUNNING"))
    assert w._pi_run_page.camera_slot.isVisibleTo(w)


def test_camera_on_the_run_page_is_video_only(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    cam = w._camera_panel
    w.switch_to_builder()
    assert cam._refresh_btn.isVisibleTo(cam)  # Dashboard's Camera tab: full panel
    w._enter_dashboard()
    assert cam.parentWidget() is w._pi_run_page.camera_slot
    assert not cam._refresh_btn.isVisibleTo(cam)


def test_desktop_switch_unhooks_the_removed_dashboard(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._pi_dashboard.set_current("nodes")
    w._switch_to_desktop_mode()
    w._graph_view.node_selected.emit("n1")
    assert w._pi_dashboard.current() == "nodes"


def test_camera_status_does_not_grow_a_status_bar(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._camera_panel._show_status("Camera settings saved")
    assert w.findChild(QStatusBar) is None


def test_camera_status_still_reaches_the_desktop_status_bar(qtbot, main_window_factory):
    w = main_window_factory(desktop_mode=True)
    w.switch_to_builder()
    w._camera_panel._show_status("Camera settings saved")
    assert w.statusBar().currentMessage() == "Camera settings saved"
