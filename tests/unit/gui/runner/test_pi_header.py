"""The Pi's two-row header and notice bar, as MainWindow builds them in runner mode.

Row one is the status strip beside a gear menu, row two is the tabs, and the
notice bar sits under both. Until the operator view is rebuilt the window itself
is still as wide as the desktop shell's minimum, so these assert on the header's
own geometry rather than the window's width.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QStatusBar, QToolButton


def _pi(qtbot, main_window_factory):
    w = main_window_factory(desktop_mode=False)
    w.resize(480, 800)
    w.show()
    # A runner window shows itself before its central widget exists, so Qt only
    # shows (and lays out) the header on the next pass of the event loop.
    qtbot.waitUntil(w._pi_header.isVisible)
    return w


def test_header_is_two_rows_with_no_overlap(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    strip, bar = w._status_strip, w._tab_bar
    strip_bottom = strip.mapTo(w, QPoint(0, 0)).y() + strip.height()
    assert strip_bottom <= bar.mapTo(w, QPoint(0, 0)).y()
    assert not w._tab_bar.buttons()["analyze"].isVisibleTo(w)


def test_desktop_controls_stay_hidden_on_every_page(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    for go in (w.switch_to_builder, w._show_experiment_tab, w._enter_dashboard):
        go()
        assert not w._status_strip.left_toggle().isVisibleTo(w)
        assert not w._status_strip.palette_hint().isVisibleTo(w)


def test_gear_menu_items(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    texts = [a.text() for a in w._gear_button.menu().actions() if not a.isSeparator()]
    assert texts == [
        "New",
        "Open…",
        "Save",
        "Save As…",
        "Connect boards…",
        "Help",
        "Switch to Desktop",
        "Exit",
    ]


def test_gear_is_a_touch_sized_instant_popup(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    gear = w._gear_button
    assert gear.objectName() == "piGearButton"
    assert gear.width() >= 44 and gear.height() >= 44
    assert gear.popupMode() == QToolButton.ToolButtonPopupMode.InstantPopup
    assert w._pi_header.objectName() == "piHeader"


def test_gear_menu_has_separators_before_connect_and_help(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    actions = w._gear_button.menu().actions()
    for text in ("Connect boards…", "Help"):
        before = actions[[a.text() for a in actions].index(text) - 1]
        assert before.isSeparator()


def test_connect_boards_resolves_the_panel_when_clicked(qtbot, main_window_factory, monkeypatch):
    w = _pi(qtbot, main_window_factory)
    opened = []
    monkeypatch.setattr(w._hardware_panel, "show_board_settings_dialog", lambda: opened.append(1))
    connect = next(a for a in w._gear_button.menu().actions() if a.text() == "Connect boards…")
    connect.trigger()
    assert opened == [1]


def test_notice_bar_sits_directly_under_the_header(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    column = w.centralWidget().layout()
    assert column.indexOf(w._notice_bar) == column.indexOf(w._pi_header) + 1


def test_notices_use_the_bar_not_a_status_bar(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._notify_user("Start Error", "No board", level="warning")
    assert w.findChild(QStatusBar) is None
    assert w._notice_bar.text() == "Start Error: No board"


def test_long_name_does_not_widen_the_window(qtbot, main_window_factory):
    w = _pi(qtbot, main_window_factory)
    w._core.session.metadata.name = "x" * 60
    w._refresh_strip_experiment()
    assert w._pi_header.minimumSizeHint().width() <= 480
