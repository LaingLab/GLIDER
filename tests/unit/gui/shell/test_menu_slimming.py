"""What comes off the menu bar, and what has to stay on it.

The bar is **File, Edit, Experiment, View, Tools, Help**. Hardware and Run come
off, under a rule worth stating in full because the first version of this change
did not have one:

    a menu leaves the bar only when its actions already have another *visible*
    home -- a panel button or a toolbar button. The palette does not count.

Hardware and Run satisfy it: the Hardware panel carries Add Board and Add
Device, and the toolbar carries Connect, Start and Stop. Experiment and Tools
were taken off with them and did not, which is what this file's rule now exists
to prevent -- eight actions, including the plugin manager, whose only remaining
route was a command palette you have to know about before you can use it.

The original risk of taking *any* menu off is still here and still the sharpest
thing in this file: the palette used to source its corpus by walking the real
``QMenuBar``, so deleting a menu would have deleted its actions from the palette
*as well* -- the place they were supposed to have gone. The actions would still
exist, still work when triggered, and be reachable from nowhere. Nothing raises.
Nothing else in the suite notices.

So the load-bearing tests here are :func:`test_every_relocated_action_is_still_reachable`,
-- written to derive its list from the window rather than from names typed
into this file, so that it walks the window's own menu registry for whichever
categories are currently relocated and demands each action back out of the
palette's corpus *by identity*. An action added to the Hardware menu next year
is covered the day it is added.

Whether a menu should be off the bar at all is the other half, and it lives in
``test_menu_paths_are_followable.py``.

That derivation cannot see an action that was deleted outright, so it is paired
with :func:`test_the_menus_that_came_off_the_bar_kept_their_actions`, which
holds the sixteen names as they stood before this change. Together: the derived
test catches additions, the recorded floor catches disappearances.

One thing the plan did not anticipate. A ``QAction`` in a ``QMenu`` that is not
on the menu bar has no *visible* associated widget, and Qt will not dispatch its
shortcut -- so taking the Run menu off the bar would have quietly killed F5.
:func:`test_f5_still_starts_a_run_from_the_keyboard` presses the key.

Every window here is ``show()``n and held for the whole test body:
``qtbot.addWidget`` keeps only a weak reference, and a dropped top level takes
its children with it.
"""

from __future__ import annotations

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QPushButton

from glider.gui.main_window import MENU_BAR_TITLES, RELOCATED_MENU_TITLES

pytestmark = pytest.mark.usefixtures("qtbot")


#: The actions these four menus carried on the day they came off the bar.
#: Experiment and Tools have since gone back on it and are kept here anyway:
#: the point of the list is that *deleting* one of these actions cannot pass
#: quietly, and that is worth having whether or not the menu is on the bar.
#: A floor, not an inventory -- later additions are the derived test's job.
ACTIONS_THE_REMOVED_MENUS_CARRIED = {
    "Experiment": ["Experiment Settings...", "Add Subject...", "Lab Setup..."],
    "Hardware": [
        "Add Board...",
        "Add Device...",
        "New Custom Device Type...",
        "Connect All",
        "Disconnect All",
    ],
    "Run": ["Start", "Stop", "Emergency Stop"],
    "Tools": [
        "Behavior Analysis…",
        "Batch Pose Tracking…",
        "Session Review…",
        "GPU / Device Check…",
        "Plugins…",
    ],
}


# ---------------------------------------------------------------------- helpers


def _builder(qtbot, main_window_factory):
    """A shown, **activated** desktop MainWindow on the Builder page.

    The activation is load-bearing for every keystroke below, not defensive:
    both ``Ctrl+K`` and the relocated shortcuts are bound with ``WindowShortcut``
    context, so Qt matches them only while the window is the active one. Without
    it the key press is delivered, matches nothing, and the test reports an
    unwired shortcut that is merely unfocused. Same dance, and same reason, as
    ``_active_builder`` in ``test_command_palette.py``.
    """
    window = main_window_factory(desktop_mode=True)
    with qtbot.waitExposed(window):
        window.show()
    window.switch_to_builder()
    window.activateWindow()
    qtbot.waitActive(window)
    return window


def _plain(text: str) -> str:
    """An action or menu title as the user reads it, mnemonic stripped."""
    return text.replace("&&", "\x00").replace("&", "").replace("\x00", "&").strip()


def _bar_titles(window) -> list[str]:
    return [_plain(action.text()) for action in window.menuBar().actions()]


def _registry_titles(window) -> list[str]:
    return [_plain(menu.title()) for menu in window.menus()]


def _menu(window, title: str):
    for menu in window.menus():
        if _plain(menu.title()) == title:
            return menu
    return None


def _relocated_actions(window) -> list[QAction]:
    """Every action in a relocated menu, read off the window itself.

    Derived rather than listed: this is what makes the reachability test cover
    an action nobody has written yet.
    """
    found: list[QAction] = []
    for title in RELOCATED_MENU_TITLES:
        menu = _menu(window, title)
        assert menu is not None, f"{title} menu is gone entirely"
        found.extend(action for action in menu.actions() if not action.isSeparator())
    return found


def _open_palette(qtbot, window):
    qtbot.keyClick(window, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    palette = window.command_palette()
    assert palette is not None
    qtbot.waitUntil(palette.isVisible)
    return palette


# ------------------------------------------------------------------ the bar


def test_the_menu_bar_is_the_menus_with_something_only_they_can_show(qtbot, main_window_factory):
    """Experiment and Tools are on it because nothing else shows their actions."""
    window = _builder(qtbot, main_window_factory)
    assert _bar_titles(window) == ["File", "Edit", "Experiment", "View", "Tools", "Help"]
    assert tuple(_bar_titles(window)) == MENU_BAR_TITLES
    del window


def test_the_relocated_menus_are_not_on_the_bar(qtbot, main_window_factory):
    window = _builder(qtbot, main_window_factory)
    on_the_bar = set(_bar_titles(window))
    for title in RELOCATED_MENU_TITLES:
        assert title not in on_the_bar
    del window


def test_the_window_still_builds_all_eight_menus_in_menu_order(qtbot, main_window_factory):
    """The registry keeps the original order, so the unfiltered palette still
    reads File, Edit, Experiment, … -- and so the bar, which is a filter over
    this list, comes out in menu order rather than build-accident order."""
    window = _builder(qtbot, main_window_factory)
    assert _registry_titles(window) == [
        "File",
        "Edit",
        "Experiment",
        "View",
        "Hardware",
        "Run",
        "Tools",
        "Help",
    ]
    del window


# --------------------------------------------------------------- reachability


def test_every_relocated_action_is_still_reachable(qtbot, main_window_factory):
    """The one that matters. Derived from the window, asserted by identity.

    ``window.commands()`` is what the palette opens onto, so an action present
    here is an action a user can find and run. An action absent here exists and
    does nothing for anybody.
    """
    window = _builder(qtbot, main_window_factory)

    relocated = _relocated_actions(window)
    # Hardware's five and Run's three, as of the day Experiment and Tools went
    # back on the bar. A floor, so that a menu quietly emptying is not mistaken
    # for a menu whose actions are all reachable.
    assert len(relocated) >= 8, "the relocated menus came back empty"

    reachable = {id(command.action) for command in window.commands()}
    unreachable = [_plain(action.text()) for action in relocated if id(action) not in reachable]
    assert unreachable == []
    del window


def test_the_menus_that_came_off_the_bar_kept_their_actions(qtbot, main_window_factory):
    """The floor. The derived test above cannot see an action that was deleted."""
    window = _builder(qtbot, main_window_factory)

    for title, expected in ACTIONS_THE_REMOVED_MENUS_CARRIED.items():
        menu = _menu(window, title)
        assert menu is not None, f"{title} menu is gone entirely"
        texts = [_plain(action.text()) for action in menu.actions() if not action.isSeparator()]
        missing = [name for name in expected if name not in texts]
        assert missing == [], f"{title}: {missing}"
    del window


# ------------------------------------------------------------------- palette


def test_the_palette_lists_every_relocated_action(qtbot, main_window_factory):
    window = _builder(qtbot, main_window_factory)
    expected = [_plain(action.text()) for action in _relocated_actions(window)]

    palette = _open_palette(qtbot, window)

    shown = palette.shown_texts()
    assert [name for name in expected if name not in shown] == []
    del window


def test_a_relocated_row_still_names_the_menu_it_came_from(qtbot, main_window_factory):
    """After this change the category is the only thing left that says where a
    command used to live."""
    window = _builder(qtbot, main_window_factory)

    palette = _open_palette(qtbot, window)
    categories = {command.text: command.category for command in palette.commands()}

    assert categories["Add Subject..."] == "Experiment"
    assert categories["Connect All"] == "Hardware"
    assert categories["Emergency Stop"] == "Run"
    assert categories["Plugins…"] == "Tools"
    del window


def test_the_palette_reads_a_relocated_actions_enabled_state_live(qtbot, main_window_factory):
    """Sourcing from the real actions has to survive them leaving the bar: the
    palette re-reads on every open, so a command greys the moment the app says
    it cannot be run."""
    window = _builder(qtbot, main_window_factory)
    connect_all = next(
        action for action in _relocated_actions(window) if _plain(action.text()) == "Connect All"
    )

    palette = _open_palette(qtbot, window)
    before = next(c for c in palette.commands() if c.text == "Connect All")
    assert before.enabled is True
    palette.dismiss()

    connect_all.setEnabled(False)
    palette = _open_palette(qtbot, window)
    after = next(c for c in palette.commands() if c.text == "Connect All")
    assert after.enabled is False

    palette.dismiss()
    connect_all.setEnabled(True)
    palette = _open_palette(qtbot, window)
    assert next(c for c in palette.commands() if c.text == "Connect All").enabled is True
    del window


def test_ctrl_k_still_opens_the_palette_and_it_still_omits_itself(qtbot, main_window_factory):
    window = _builder(qtbot, main_window_factory)

    palette = _open_palette(qtbot, window)

    assert palette.isVisible()
    assert "Command Palette" not in palette.shown_texts()
    del window


# ----------------------------------------------------------------- shortcuts


def test_every_relocated_shortcut_is_associated_with_the_window(qtbot, main_window_factory):
    """A menu off the bar is not a visible widget, and Qt will not dispatch the
    shortcuts of actions whose only home is one. Association with the window is
    what puts F5 back."""
    window = _builder(qtbot, main_window_factory)

    on_the_window = {id(action) for action in window.actions()}
    orphaned = [
        _plain(action.text())
        for action in _relocated_actions(window)
        if not action.shortcut().isEmpty() and id(action) not in on_the_window
    ]
    assert orphaned == []
    del window


def test_an_action_in_a_submenu_off_the_bar_is_adopted_too(qtbot, main_window_factory):
    """The palette and the adoption must mean the same thing by "the actions of
    a menu". The palette walks submenus; the adoption read one flat level, so an
    action one level down would be *listed* with a shortcut that did nothing --
    the exact silent failure the adoption exists to prevent.
    """
    window = _builder(qtbot, main_window_factory)
    # Hardware rather than Tools: the menu has to be one that is actually off
    # the bar, or the adoption this checks is not the thing being exercised.
    hardware = _menu(window, "Hardware")
    assert hardware is not None
    submenu = hardware.addMenu("&Diagnostics")
    buried = QAction("&Ping Board", window)
    buried.setShortcut("Ctrl+Alt+P")
    submenu.addAction(buried)

    window._adopt_relocated_shortcuts()

    assert buried in window.actions()
    assert "Ping Board" in [command.text for command in window.commands()]
    del window


def test_an_action_in_a_submenu_on_the_bar_is_left_alone(qtbot, main_window_factory):
    """Two associations for one shortcut is Qt's ambiguous-shortcut warning, and
    the action already dispatches through the bar."""
    window = _builder(qtbot, main_window_factory)
    file_menu = _menu(window, "File")
    submenu = file_menu.addMenu("&Recent")
    buried = QAction("&Yesterday", window)
    buried.setShortcut("Ctrl+Alt+Y")
    submenu.addAction(buried)

    window._adopt_relocated_shortcuts()

    assert buried not in window.actions()
    del window


def test_f5_still_starts_a_run_from_the_keyboard(qtbot, main_window_factory):
    """The structural claim above, pressed.

    The real handler is disconnected first: it schedules a coroutine, and there
    is no running loop under the test. What is being asserted is Qt's dispatch,
    not what Start does.
    """
    window = _builder(qtbot, main_window_factory)
    start = next(
        action for action in _relocated_actions(window) if _plain(action.text()) == "Start"
    )
    assert start.shortcut().toString() == "F5"

    start.disconnect()
    fired: list[bool] = []
    start.triggered.connect(lambda _checked=False: fired.append(True))

    qtbot.keyClick(window, Qt.Key.Key_F5)

    assert fired == [True]
    del window


# -------------------------------------------------------------- the panels


def test_the_hardware_add_actions_are_also_buttons_in_the_hardware_panel(
    qtbot, main_window_factory
):
    """Two of the sixteen went somewhere better than the palette: the panel that
    already owned the handler."""
    window = _builder(qtbot, main_window_factory)
    panel = window._builder_view.left.widget_for("hardware")
    assert panel is window._hardware_panel

    labels = {button.text() for button in panel.findChildren(QPushButton)}
    assert "+ Board" in labels
    assert "+ Device" in labels
    del window
