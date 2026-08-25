"""An instruction that names a menu must name one the user can see.

When Experiment, Hardware, Run and Tools came off the menu bar, five strings
elsewhere in the application went on telling people to use them -- "Tools ▸
Batch Pose Tracking", "Experiment → Lab Setup", "Hardware → Add Device". Each
one sent a researcher looking along a menu bar for a menu that was no longer on
it. Nothing raised, no test failed, and the text was still perfectly accurate
about where the action lived; it had just stopped being somewhere anyone could
go.

That is the class this file exists for, and it is a class rather than an
incident: the strings live in six files that have no reason to know the bar
changed, so the next menu that moves breaks whichever of them mention it.

The check is deliberately crude -- a regex over the source for ``Menu ▸ Item``
and ``Menu -> Item`` -- because the alternative is a registry of documented
paths that drifts from the strings it claims to describe, which is the same
failure one level up.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from glider.gui.main_window import MENU_BAR_TITLES, RELOCATED_MENU_TITLES

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "src" / "glider"

#: "Tools ▸ Batch Pose Tracking", "Experiment → Lab Setup", "Hardware -> Add
#: Device". The separator is what marks it as a *path* rather than a mention:
#: prose says "the Hardware panel", an instruction says "Hardware ▸ something".
MENU_PATH = re.compile(r"\b(?P<menu>[A-Z][A-Za-z]+)\s*(?:▸|→|->)\s*(?P<item>[A-Z][\w /]+)")

#: Words that pass the pattern without being a menu. `Sequence -> Timer` in a
#: docstring about the flow graph is not an instruction to anybody.
NOT_MENUS = frozenset(
    {
        "Builder",
        "Dashboard",
        "Runner",
        "Sequence",
        "Input",
        "Output",
        "Data",
        "Note",
        "Returns",
        "Raises",
        "Args",
        "Example",
        "True",
        "False",
        "None",
    }
)

#: Every menu the window builds, on the bar or not.
ALL_MENUS = frozenset(MENU_BAR_TITLES) | frozenset(RELOCATED_MENU_TITLES)


def _menu_paths() -> list[tuple[Path, int, str, str]]:
    """Every ``Menu ▸ Item`` written anywhere under ``src/glider``."""
    found: list[tuple[Path, int, str, str]] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in MENU_PATH.finditer(line):
                menu = match.group("menu")
                if menu in NOT_MENUS or menu not in ALL_MENUS:
                    continue
                found.append((path, number, menu, match.group("item").strip()))
    return found


def test_the_scan_finds_something():
    """A guard on the guard. If the pattern stops matching -- a separator
    changes, the source moves -- every other test here passes by finding
    nothing, which is the failure mode a regex-based check actually has."""
    assert _menu_paths(), f"no menu paths found under {SOURCE_ROOT}; the scan is broken"


def test_every_menu_path_names_a_menu_on_the_bar():
    """The check itself.

    A menu deliberately off the bar is not a thing to send someone to. Either
    put it back, or rewrite the instruction to name where the action actually
    is -- "Add Device in the Hardware panel" rather than "Hardware → Add
    Device".
    """
    unfollowable = [
        f"{path.relative_to(SOURCE_ROOT)}:{number} says {menu} ▸ {item}, "
        f"but {menu} is not on the menu bar"
        for path, number, menu, item in _menu_paths()
        if menu not in MENU_BAR_TITLES
    ]

    assert not unfollowable, "\n" + "\n".join(unfollowable)


@pytest.mark.parametrize("menu", ["Experiment", "Tools"])
def test_the_menus_with_no_other_home_are_on_the_bar(menu):
    """Named outright, because the derived checks above cannot see this one.

    Every action in these two has exactly one call site and it is the menu
    action -- no panel button, no toolbar button. Taking either off the bar
    leaves the command palette as the only route, and a palette is something
    you have to already know about. Hardware and Run are off the bar precisely
    because they do not have that problem: the Hardware panel carries Add Board
    and Add Device, and the toolbar carries Connect, Start and Stop.
    """
    assert menu in MENU_BAR_TITLES
    assert menu not in RELOCATED_MENU_TITLES


def test_the_plugin_manager_is_reachable_without_the_palette(qtbot, mock_core):
    """The report that started this: 'how are people supposed to get the plugin
    manager without using the control search'."""
    from glider.gui.main_window import MainWindow

    window = MainWindow(mock_core)
    qtbot.addWidget(window)
    window.show()

    # Down one level from the bar: the titles on it, then the items inside.
    # Walking the real QMenuBar rather than the window's registry is the point
    # -- the registry holds menus that are not on it.
    reachable = {
        f"{bar_action.text()} > {item.text()}".replace("&", "")
        for bar_action in window.menuBar().actions()
        if bar_action.menu() is not None
        for item in bar_action.menu().actions()
        if not item.isSeparator()
    }

    assert any("Plugins" in path for path in reachable), sorted(reachable)


def test_the_recorded_palette_only_gap_still_describes_real_actions(qtbot, mock_core):
    """``PALETTE_ONLY_ACTIONS`` names the three actions that are still reachable
    only through Ctrl+K. It is a known gap, deliberately written down.

    What is checked is that the list still *describes something*: an action
    renamed or deleted would leave the note quietly describing nothing, which is
    how a recorded gap turns into a forgotten one. Whether these three should
    still be on the list is a question for whoever gives them a home -- taking
    one off is the point, and nothing here objects to that.
    """
    from glider.gui.main_window import PALETTE_ONLY_ACTIONS, MainWindow

    window = MainWindow(mock_core)
    qtbot.addWidget(window)

    by_menu = {
        menu.title()
        .replace("&", "")
        .strip(): {
            action.text().replace("&", "").strip()
            for action in menu.actions()
            if not action.isSeparator()
        }
        for menu in window.menus()
    }

    missing = [
        f"{menu} has no {name!r}"
        for menu, names in PALETTE_ONLY_ACTIONS.items()
        for name in names
        if name not in by_menu.get(menu, set())
    ]

    assert not missing, "\n".join(missing)


def test_nothing_on_the_bar_is_recorded_as_palette_only(qtbot):
    """The list describes menus that are off the bar. One naming a menu that is
    on it would be describing a gap that has already closed."""
    from glider.gui.main_window import PALETTE_ONLY_ACTIONS

    on_the_bar = [menu for menu in PALETTE_ONLY_ACTIONS if menu in MENU_BAR_TITLES]

    assert on_the_bar == [], f"{on_the_bar} are on the bar; their actions are reachable"
