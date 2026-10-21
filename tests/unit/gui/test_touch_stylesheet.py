"""What touch.qss has to get right on the Pi, measured with the sheet applied."""

from __future__ import annotations

from PyQt6.QtGui import QPalette

from glider.gui.styles import colors, load_stylesheet


def test_the_graph_lock_is_a_44px_target_and_shows_all_its_text(qtbot, main_window_factory):
    w = main_window_factory(desktop_mode=False)
    w.resize(480, 800)
    w.show()
    qtbot.waitUntil(w._pi_header.isVisible)
    w.switch_to_builder()
    lock = w._graph_view._overlay.lock_btn
    qtbot.waitUntil(lock.isVisible)
    # The sheet lands after the button was first sized; it must re-fit.
    assert lock.height() >= 44
    assert lock.width() >= lock.sizeHint().width()


def test_the_notice_text_takes_the_colour_of_a_new_level(qtbot):
    from glider.gui.runner.notice_bar import NoticeBar

    bar = NoticeBar()
    qtbot.addWidget(bar)
    bar.setStyleSheet(load_stylesheet("touch"))
    bar.show_notice("Saved", "box_pi_side.glider", "info")
    bar.show_notice("UNSAFE STATE", "Hardware may still be active", "critical")
    text = bar._label.palette().color(QPalette.ColorRole.WindowText)
    assert text.name() == colors.ERROR


def test_the_styled_strip_still_fits_five_boards_beside_the_gear(qtbot):
    from glider.gui.shell.status_strip import StatusStrip

    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.setStyleSheet(load_stylesheet("touch"))
    strip.set_touch_mode(True)
    strip.set_experiment("x" * 60, dirty=True)
    strip.set_run_state("recording", "12:41")
    strip.set_devices([(f"board_{i}", "error", "") for i in range(5)])
    strip.ensurePolished()
    for child in strip.findChildren(type(strip.pill())):
        child.ensurePolished()
    assert strip.minimumSizeHint().width() <= 480 - 44


def test_the_pi_run_box_wraps_its_board_line(qtbot, mock_core):
    from glider.gui.dashboard.panels.run_control_panel import RunControlPanel

    touch = RunControlPanel(mock_core, touch=True)
    desktop = RunControlPanel(mock_core)
    qtbot.addWidget(touch)
    qtbot.addWidget(desktop)
    assert touch._sub.wordWrap()
    assert not desktop._sub.wordWrap()
