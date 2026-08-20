"""What the composed Builder frame guarantees.

**Every test here runs against a shown shell.** Qt gives a widget its final
geometry on the *show* event, and this file is mostly about widths -- an
unshown splitter has no sizes to assert on. :func:`_shown` therefore shows and
waits for exposure, and every fixture holds the top-level widget for the whole
test: ``qtbot.addWidget`` keeps only a weak reference, and a dropped top level
takes its children with it ("wrapped C/C++ object has been deleted").

Two of these tests exist because the composition is where the pieces stop
behaving the way they do alone:

* A :class:`SidePanel` inside a :class:`QSplitter` cannot keep its own width.
  ``setFixedWidth`` wins on collapse, but on expand the splitter re-imposes
  whatever it last handed out, silently discarding the width the user dragged.
  :func:`test_a_dragged_width_survives_a_collapse_and_an_expand` asserts the
  pixels, not a flag.
* The strip's toggles reflect the panels; they do not own them. Without the
  echo back, the first collapse by any other route -- a rail click, a restored
  layout -- desynchronises the button from the panel it claims to describe.
"""

from __future__ import annotations

import logging

import pytest
from PyQt6 import sip
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QLabel, QWidget

from glider.gui.shell.app_shell import MIN_CENTRE_WIDTH, AppShell
from glider.gui.shell.side_panel import DEFAULT_WIDTH, MIN_EXPANDED_WIDTH, RAIL_WIDTH

#: (key, label, icon) for each side, matching what the Builder will carry.
LEFT_TABS = [("nodes", "Nodes", "N"), ("hardware", "Hardware", "H"), ("files", "Files", "F")]
RIGHT_TABS = [("properties", "Properties", "P"), ("camera", "Camera", "C")]

#: Wide enough that both panels can hold a dragged width and still leave the
#: centre a usable surface -- roughly a laptop screen.
_SHELL_WIDTH = 1200
_SHELL_HEIGHT = 700

#: The widest a laptop-class Builder may demand. A window that cannot be made
#: this narrow is unusable on the machines this app runs on.
_LAPTOP_WIDTH = 1024


def _shown(qtbot, *, width: int = _SHELL_WIDTH) -> AppShell:
    """A populated shell, sized, shown and exposed.

    The shell is the top-level widget, so the caller holding it is what keeps
    it (and every widget under test inside it) alive; see the module docstring.
    """
    centre = QWidget()
    centre.setObjectName("centre")
    shell = AppShell(centre)
    qtbot.addWidget(shell)
    for key, label, icon in LEFT_TABS:
        shell.left.add_tab(key, label, QLabel(label), icon)
    for key, label, icon in RIGHT_TABS:
        shell.right.add_tab(key, label, QLabel(label), icon)

    shell.resize(width, _SHELL_HEIGHT)
    with qtbot.waitExposed(shell):
        shell.show()
    qtbot.waitUntil(lambda: shell.splitter.width() > 0)
    return shell


def _drag(qtbot, shell: AppShell, *, left: int | None = None, right: int | None = None) -> None:
    """Move the splitter handles as a user dragging them would."""
    splitter = shell.splitter
    left_w = shell.left.width() if left is None else left
    right_w = shell.right.width() if right is None else right
    available = splitter.width() - splitter.handleWidth() * 2
    splitter.setSizes([left_w, available - left_w - right_w, right_w])
    qtbot.waitUntil(lambda: shell.left.width() == left_w and shell.right.width() == right_w)


def _settle(qtbot, shell: AppShell) -> None:
    """Force the layout pass a width assertion must survive.

    A panel resizing *itself* holds until something makes the splitter lay out
    again -- and then the splitter re-imposes its own sizes. Every width here
    is asserted after that has happened, because ``waitUntil`` alone would be
    satisfied by the transient value and pass on a shell that loses the width a
    moment later.
    """
    qtbot.wait(1)
    shell.resize(shell.width() + 1, shell.height())
    qtbot.wait(1)
    shell.resize(shell.width() - 1, shell.height())
    qtbot.wait(1)


@pytest.fixture
def shell(qtbot) -> AppShell:
    return _shown(qtbot)


@pytest.fixture
def settings(tmp_path) -> QSettings:
    """An ini under ``tmp_path``. No test may touch the real settings."""
    return QSettings(str(tmp_path / "shell.ini"), QSettings.Format.IniFormat)


# ------------------------------------------------------------------ composition


def test_the_shell_is_a_strip_above_a_panel_centre_panel_splitter(shell):
    assert shell.strip is not None
    assert shell.splitter.count() == 3
    assert shell.splitter.widget(0) is shell.left
    assert shell.splitter.widget(2) is shell.right
    # The centre is reachable but is not the shell's business beyond hosting.
    assert shell.centre.objectName() == "centre"
    assert shell.strip.y() < shell.splitter.y()


def test_the_shell_does_not_know_what_the_centre_is(qtbot):
    """Any widget will do -- the shell must not assume a node graph."""
    plain = QLabel("anything at all")
    shell = AppShell(plain)
    qtbot.addWidget(shell)
    with qtbot.waitExposed(shell):
        shell.show()

    assert shell.centre is plain


def test_set_centre_replaces_without_leaking_the_previous_widget(qtbot, shell):
    old = shell.centre
    new = QWidget()
    new.setObjectName("replacement")

    shell.set_centre(new)

    assert shell.centre is new
    assert shell.splitter.widget(1).findChild(QWidget, "replacement") is new
    qtbot.waitUntil(lambda: sip.isdeleted(old))


# ------------------------------------------------------------------ the widths


def test_a_dragged_width_survives_a_collapse_and_an_expand(qtbot, shell):
    """The width the user chose comes back, in pixels.

    A drag is the *easy* case -- it moves the splitter's own stored size, so
    the splitter happens to hand 320 px back by itself today. The hard case is
    :func:`test_a_remembered_width_the_splitter_never_saw_survives_an_expand`
    below. This one pins the guarantee a user would describe.
    """
    _drag(qtbot, shell, left=320)
    assert shell.left.width() == 320

    shell.left.set_expanded(False)
    _settle(qtbot, shell)
    assert shell.left.width() == RAIL_WIDTH

    shell.left.set_expanded(True)
    _settle(qtbot, shell)
    assert shell.left.width() == 320


def test_a_remembered_width_the_splitter_never_saw_survives_an_expand(qtbot, settings):
    """The way a panel inside a splitter really loses a width.

    A width that reached the panel by any route other than a drag -- here a
    restored layout, which is the route that matters after a restart -- is a
    width the splitter has never stored. Expanding makes the splitter lay out
    again, and it re-imposes what it does know: the panel snaps to its 160 px
    minimum and the user's 300 px is gone. Measured: 160, not 300.
    """
    settings.setValue("shell/left_width", 300)
    settings.setValue("shell/left_expanded", False)

    shell = _shown(qtbot)
    shell.restore_layout(settings)
    _settle(qtbot, shell)
    assert shell.left.width() == RAIL_WIDTH

    shell.left.set_expanded(True)
    _settle(qtbot, shell)

    assert shell.left.width() == 300


def test_both_sides_keep_their_own_dragged_widths(qtbot, shell):
    _drag(qtbot, shell, left=300, right=200)

    shell.left.set_expanded(False)
    shell.right.set_expanded(False)
    _settle(qtbot, shell)
    assert (shell.left.width(), shell.right.width()) == (RAIL_WIDTH, RAIL_WIDTH)

    shell.right.set_expanded(True)
    shell.left.set_expanded(True)
    _settle(qtbot, shell)

    assert (shell.left.width(), shell.right.width()) == (300, 200)


def test_collapsing_a_panel_gives_its_space_to_the_centre_at_once(qtbot, shell):
    """Not on the next window resize -- then, a collapse leaves a 226 px hole
    beside the rail until the user happens to resize something."""
    before = shell.centre.width()

    shell.left.set_expanded(False)
    qtbot.waitUntil(lambda: shell.left.width() == RAIL_WIDTH)

    assert shell.centre.width() > before


# ------------------------------------------------------------------ the toggles


def test_the_strips_toggle_collapses_the_panel_it_describes(qtbot, shell):
    shell.strip.left_toggle().click()
    qtbot.waitUntil(lambda: not shell.left.expanded)

    assert shell.strip.left_toggle().isChecked() is False
    assert shell.right.expanded is True
    assert shell.strip.right_toggle().isChecked() is True

    shell.strip.left_toggle().click()
    qtbot.waitUntil(lambda: shell.left.expanded)
    assert shell.strip.left_toggle().isChecked() is True


def test_a_rail_click_keeps_the_strips_toggle_in_step(qtbot, shell):
    """The echo the strip cannot supply for itself.

    A panel collapsed by any route other than the toggle -- a rail click here,
    a restored layout elsewhere -- must still move the button, or the button
    starts describing a panel state that is not the one on screen.
    """
    shell.left.set_expanded(False)
    qtbot.waitUntil(lambda: not shell.left.expanded)
    assert shell.strip.left_toggle().isChecked() is False

    shell.left.rail_buttons()[2].click()  # "Files"

    qtbot.waitUntil(lambda: shell.left.expanded)
    assert shell.strip.left_toggle().isChecked() is True
    assert shell.left.current_key() == "files"


def test_the_right_side_is_wired_the_same_way(qtbot, shell):
    shell.strip.right_toggle().click()
    qtbot.waitUntil(lambda: not shell.right.expanded)
    assert shell.strip.right_toggle().isChecked() is False

    shell.right.set_expanded(True)
    assert shell.strip.right_toggle().isChecked() is True


# ------------------------------------------------------------------ persistence


def test_the_layout_round_trips_through_settings(qtbot, shell, settings):
    _drag(qtbot, shell, left=300, right=210)
    shell.left.set_current("hardware")
    shell.right.set_current("camera")
    shell.right.set_expanded(False)

    shell.save_layout(settings)

    restored = _shown(qtbot)
    restored.restore_layout(settings)

    assert restored.left.expanded is True
    assert restored.right.expanded is False
    assert restored.left.current_key() == "hardware"
    assert restored.right.current_key() == "camera"
    _settle(qtbot, restored)
    assert restored.left.width() == 300

    restored.right.set_expanded(True)
    _settle(qtbot, restored)
    assert restored.right.width() == 210


def test_the_window_geometry_round_trips(qtbot, shell, settings):
    available = QGuiApplication.primaryScreen().availableGeometry()
    target = available.adjusted(20, 20, -20, -20)
    shell.setGeometry(target)
    qtbot.waitUntil(lambda: shell.geometry() == target)

    shell.save_layout(settings)

    restored = _shown(qtbot)
    restored.restore_layout(settings)

    qtbot.waitUntil(lambda: restored.geometry() == target)


def test_a_saved_tab_that_no_longer_exists_falls_back_to_the_first(qtbot, shell, settings):
    shell.left.set_current("files")
    shell.save_layout(settings)
    settings.setValue("shell/left_tab", "a-panel-that-was-removed")

    restored = _shown(qtbot)
    restored.restore_layout(settings)

    assert restored.left.current_key() == "nodes"


def test_geometry_entirely_off_screen_is_ignored_and_the_window_centred(qtbot, settings):
    """Real on a rig whose second monitor is not always plugged in."""
    settings.setValue("shell/geometry", "9000,9000,600,400")
    available = QGuiApplication.primaryScreen().availableGeometry()

    # Small enough to fit the screen as it is, so this test is about *position*
    # and the one below is about size.
    shell = _shown(qtbot, width=min(600, available.width()))
    size_before = shell.size()
    shell.restore_layout(settings)

    qtbot.waitUntil(lambda: available.intersects(shell.geometry()))
    assert shell.size() == size_before
    assert abs(shell.geometry().center().x() - available.center().x()) <= 2
    assert abs(shell.geometry().center().y() - available.center().y()) <= 2


# -- three ways a saved geometry can be well-formed and still unusable --------
#
# The type checks above are only half of it. A rectangle can parse, have area
# and touch a screen while still leaving the window unreachable, and the last of
# these is the *fallback path* -- the one that runs when everything else has
# already been rejected.


def test_an_absurd_saved_size_is_clamped_to_the_screen(qtbot, settings):
    """A rectangle with area is not the same as a rectangle with a usable size.

    ``0,0,99999999,99999999`` parses, has area and overlaps every screen, and
    gives a 16 777 215 px window with nothing on it reachable -- which
    ``save_layout`` then writes back on close, so it survives restarts.
    """
    settings.setValue("shell/geometry", "0,0,99999999,99999999")
    available = QGuiApplication.primaryScreen().availableGeometry()

    shell = _shown(qtbot)
    shell.restore_layout(settings)

    qtbot.waitUntil(lambda: available.contains(shell.geometry()))
    assert shell.width() <= available.width()
    assert shell.height() <= available.height()


def test_a_geometry_hanging_off_the_corner_is_not_on_a_screen(qtbot, settings):
    """One pixel of overlap is not "on a screen".

    Bottom-right corner of the screen, 1400x900: a single pixel intersects, so
    the overlap test passed and the window was restored with its title bar --
    and everything else -- past the edge.
    """
    available = QGuiApplication.primaryScreen().availableGeometry()
    settings.setValue("shell/geometry", f"{available.right()},{available.bottom()},1400,900")

    shell = _shown(qtbot, width=min(600, available.width()))
    shell.restore_layout(settings)

    qtbot.waitUntil(lambda: available.contains(shell.geometry()))


def test_a_geometry_saved_left_of_the_screen_comes_back_onto_it(qtbot, settings):
    """The reported symptom, from the one direction that hides most.

    A window at a negative x still overlaps its screen by plenty, so it passes
    :func:`_on_a_screen` and reaches the clamp rather than the rescue path --
    and what it costs is the *left* edge, where the Builder keeps its first
    panel tab and the start of every label under it. Off the right or the
    bottom the user at least still has the controls; off the left they have
    neither the tab nor the words.
    """
    available = QGuiApplication.primaryScreen().availableGeometry()
    width = max(available.width() // 2, MIN_CENTRE_WIDTH)
    settings.setValue(
        "shell/geometry",
        f"{available.x() - width // 2},{available.y() + 20},{width},{available.height() // 2}",
    )

    shell = _shown(qtbot, width=width)
    shell.restore_layout(settings)

    qtbot.waitUntil(lambda: available.contains(shell.geometry()))
    assert shell.geometry().x() >= available.x()


def test_centring_a_window_larger_than_the_screen_shrinks_it_to_fit(qtbot, settings):
    """The fallback has to land somewhere usable, or rejecting is pointless.

    1400x900 is the configured default size. Centred on a 1024x768 lab PC that
    puts the title bar at y = -66, so the one path that exists to rescue a bad
    geometry produced another one.
    """
    settings.setValue("shell/geometry", "not-a-rectangle")  # forces the fallback
    available = QGuiApplication.primaryScreen().availableGeometry()

    shell = _shown(qtbot, width=available.width() + 400)
    shell.restore_layout(settings)

    qtbot.waitUntil(lambda: available.contains(shell.geometry()))
    assert shell.geometry().x() >= available.x()
    assert shell.geometry().y() >= available.y()


def test_malformed_values_fall_back_to_defaults_and_log(qtbot, settings, caplog):
    for key, junk in (
        ("shell/left_expanded", "perhaps"),
        ("shell/right_expanded", "¯\\_(ツ)_/¯"),
        ("shell/left_width", "wide-ish"),
        ("shell/right_width", ""),
        ("shell/geometry", "not-a-rectangle"),
    ):
        settings.setValue(key, junk)

    shell = _shown(qtbot)
    with caplog.at_level(logging.WARNING, logger="glider.gui.shell.app_shell"):
        shell.restore_layout(settings)

    assert shell.left.expanded is True
    assert shell.right.expanded is True
    _settle(qtbot, shell)
    assert shell.left.width() == DEFAULT_WIDTH
    assert len(caplog.records) >= 5


def test_missing_keys_leave_the_defaults(qtbot, settings):
    shell = _shown(qtbot)

    shell.restore_layout(settings)  # nothing was ever saved

    assert shell.left.expanded is True
    assert shell.right.expanded is True
    assert shell.left.current_key() == "nodes"
    assert shell.right.current_key() == "properties"


def test_a_partial_layout_restores_what_it_has(qtbot, settings):
    settings.setValue("shell/left_expanded", False)

    shell = _shown(qtbot)
    shell.restore_layout(settings)

    assert shell.left.expanded is False
    assert shell.strip.left_toggle().isChecked() is False
    assert shell.right.expanded is True


def test_an_absurd_saved_width_is_clamped_rather_than_obeyed(qtbot, settings):
    settings.setValue("shell/left_width", 5)

    shell = _shown(qtbot)
    shell.restore_layout(settings)
    _settle(qtbot, shell)

    assert shell.left.width() >= MIN_EXPANDED_WIDTH


def test_restoring_an_empty_panel_does_not_raise(qtbot, settings):
    """A shell whose tabs failed to construct still has to start."""
    settings.setValue("shell/left_tab", "nodes")
    shell = AppShell(QWidget())
    qtbot.addWidget(shell)
    with qtbot.waitExposed(shell):
        shell.show()

    shell.restore_layout(settings)

    assert shell.left.current_key() == ""


# ------------------------------------------------------------------ the strip


def test_a_long_experiment_name_cannot_force_the_window_wider_than_a_laptop(qtbot, shell):
    """A QLabel reports its whole text as its minimum width, and that minimum
    propagates out through the strip to the window. Eliding was rejected --
    it would corrupt what ``name_label().text()`` reads back -- so the cap is
    structural and lives here, where the Builder's minimum width belongs.
    """
    name = "Chronic social defeat, cohort 3, habituation day 2, arena B, camera overhead"

    shell.strip.set_experiment(name, dirty=True)

    assert shell.strip.name_label().text() == name
    assert shell.minimumSizeHint().width() <= _LAPTOP_WIDTH
    assert shell.strip.minimumSizeHint().width() <= _LAPTOP_WIDTH


def test_the_run_pill_can_carry_an_elapsed_clock(shell):
    """The mockup's pill reads "Recording 04:12". Nothing here runs a timer --
    the strip shows what it is given."""
    shell.strip.set_run_state("recording", "04:12")

    assert shell.strip.pill().text() == "Recording 04:12"
    assert shell.strip.run_state() == "recording"
