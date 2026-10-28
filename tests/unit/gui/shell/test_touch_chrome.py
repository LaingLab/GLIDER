"""Touch mode for the status strip and the tab bar.

The Pi's two-row header reuses both widgets, so each grows a reversible
``set_touch_mode``. These tests pin what it changes on the Pi (480 px wide) and
that switching it back off restores the desktop look.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from glider.gui.shell.status_strip import STRIP_HEIGHT, StatusStrip
from glider.gui.shell.tab_bar import ShellTabBar

#: The header's first row, less the gear button beside the strip.
_STRIP_BUDGET = 480 - 44

_FIVE_BOARDS = [(f"board_{i}", "ok", "") for i in range(5)]


def _shown(qtbot) -> StatusStrip:
    """A strip that is on screen, so Qt re-lays it out when a flag flips.

    An unshown widget never gets the layout requests that carry a child's new
    size up to its parents, so a test that flips touch mode and then measures
    would read the size from before the flip.
    """
    strip = StatusStrip()
    qtbot.addWidget(strip)
    with qtbot.waitExposed(strip):
        strip.show()
    return strip


def test_touch_strip_hides_desktop_controls_and_shows_dirty(qtbot):
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.set_touch_mode(True)
    strip.set_experiment("box_pi_side", dirty=True)
    assert not strip.left_toggle().isVisibleTo(strip)
    assert not strip.right_toggle().isVisibleTo(strip)
    assert not strip.palette_hint().isVisibleTo(strip)
    assert strip.dirty_label().isVisibleTo(strip)


def test_touch_strip_fits_480_with_a_long_name_and_five_boards(qtbot):
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.set_touch_mode(True)
    strip.set_experiment("x" * 60)
    strip.set_devices(_FIVE_BOARDS)
    assert strip.minimumSizeHint().width() <= _STRIP_BUDGET


def test_touch_strip_fits_480_while_recording_and_edited(qtbot):
    """The worst case: dirty marker and a running pill beside five boards."""
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.set_touch_mode(True)
    strip.set_experiment("x" * 60, dirty=True)
    strip.set_run_state("recording", "12:41")
    strip.set_devices(_FIVE_BOARDS)
    assert strip.minimumSizeHint().width() <= _STRIP_BUDGET


def test_touch_mode_reaches_boards_that_were_already_listed(qtbot):
    strip = _shown(qtbot)
    strip.set_experiment("x" * 60, dirty=True)
    strip.set_run_state("recording", "12:41")
    strip.set_devices(_FIVE_BOARDS)
    strip.set_touch_mode(True)
    qtbot.wait(0)
    assert strip.minimumSizeHint().width() <= _STRIP_BUDGET


def test_touch_strip_keeps_full_names_for_anything_that_reads_them(qtbot):
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.set_touch_mode(True)
    strip.set_experiment("x" * 60)
    strip.set_devices([("a_rather_long_board_name", "ok", "")])
    assert strip.name_label().text() == "x" * 60
    assert strip.device_names() == ["a_rather_long_board_name"]
    assert "a_rather_long_board_name" in strip.device_chips()[0].toolTip()


def test_touch_strip_is_reversible(qtbot):
    strip = _shown(qtbot)
    strip.set_experiment("x" * 60, dirty=True)
    strip.set_devices(_FIVE_BOARDS)
    desktop_width = strip.minimumSizeHint().width()

    strip.set_touch_mode(True)
    qtbot.wait(0)
    assert strip.minimumSizeHint().width() < desktop_width
    strip.set_touch_mode(False)
    qtbot.wait(0)

    assert strip.left_toggle().isVisibleTo(strip)
    assert strip.right_toggle().isVisibleTo(strip)
    assert strip.palette_hint().isVisibleTo(strip)
    assert not strip.dirty_label().isVisibleTo(strip)  # desktop never shows it
    assert strip.minimumSizeHint().width() == desktop_width


def test_leaving_touch_mode_keeps_a_toggle_the_window_had_hidden(qtbot):
    """MainWindow hides the panel toggles off the Dashboard; the way out of
    touch mode must not show them there."""
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.left_toggle().setVisible(False)

    strip.set_touch_mode(True)
    strip.set_touch_mode(True)  # twice must not record "hidden" as the desktop state
    strip.set_touch_mode(False)

    assert not strip.left_toggle().isVisibleTo(strip)
    assert strip.right_toggle().isVisibleTo(strip)
    assert strip.palette_hint().isVisibleTo(strip)


def test_boards_built_in_touch_mode_lose_their_floor_on_the_way_back(qtbot):
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.set_touch_mode(True)
    strip.set_devices(_FIVE_BOARDS)
    names = strip.findChildren(QLabel, "statusStripDeviceName")
    assert len(names) == 5
    assert all(label.minimumWidth() > 0 for label in names)

    strip.set_touch_mode(False)

    assert all(label.minimumWidth() == 0 for label in names)
    assert strip.name_label().minimumWidth() == 0


def test_pill_click_emits(qtbot):
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.set_run_state("recording", "12:41")
    with qtbot.waitSignal(strip.pill_clicked):
        qtbot.mouseClick(strip.pill(), Qt.MouseButton.LeftButton)


def test_only_a_left_click_on_the_pill_emits(qtbot):
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.set_run_state("recording", "12:41")
    with qtbot.assertNotEmitted(strip.pill_clicked):
        qtbot.mouseClick(strip.pill(), Qt.MouseButton.RightButton)


def test_touch_tab_bar_drops_tools_and_fills(qtbot):
    bar = ShellTabBar()
    qtbot.addWidget(bar)
    bar.resize(480, 46)
    bar.set_touch_mode(True)
    bar.show()
    buttons = bar.buttons()
    assert not buttons["tools"].isVisibleTo(bar)
    widths = {buttons[k].width() for k in ("experiment", "dashboard", "run")}
    assert max(widths) - min(widths) <= 2
    assert sum(buttons[k].width() for k in ("experiment", "dashboard", "run")) >= 480 - 2 * 8 - 8
    bar.set_touch_mode(False)
    assert buttons["tools"].isVisibleTo(bar)


def test_tab_bar_touch_mode_is_reversible(qtbot):
    bar = ShellTabBar()
    qtbot.addWidget(bar)
    bar.resize(900, 46)
    bar.show()
    before = {k: (b.geometry(), b.sizePolicy()) for k, b in bar.buttons().items()}

    bar.set_touch_mode(True)
    bar.set_touch_mode(False)
    bar.layout().activate()

    assert {k: (b.geometry(), b.sizePolicy()) for k, b in bar.buttons().items()} == before


def test_touch_strip_is_a_tap_target_tall_and_desktop_stays_40(qtbot):
    strip = StatusStrip()
    qtbot.addWidget(strip)
    assert strip.height() == STRIP_HEIGHT == 40
    strip.set_touch_mode(True)
    assert strip.minimumHeight() == strip.maximumHeight() == 44
    strip.set_touch_mode(False)
    assert strip.minimumHeight() == strip.maximumHeight() == STRIP_HEIGHT
