"""The Builder shell: one content surface, a panel each side, one strip.

Kept out of ``main_window.py`` on purpose -- that file is already the largest
in the GUI, and the shell's pieces are worth being able to test without
constructing a window.
"""

from __future__ import annotations

from glider.gui.shell.app_shell import (
    AppShell,
    fit_window_to_screen,
    primary_available_geometry,
)
from glider.gui.shell.command_palette import (
    Command,
    CommandPalette,
    commands_from_menu_bar,
    commands_from_menus,
    menu_actions,
)
from glider.gui.shell.landing import (
    LandingPage,
    forget_experiment,
    recent_experiments,
    remember_experiment,
)
from glider.gui.shell.side_panel import DEFAULT_WIDTH, RAIL_WIDTH, SidePanel
from glider.gui.shell.status_strip import STRIP_HEIGHT, StatusStrip
from glider.gui.shell.tab_bar import TAB_BAR_HEIGHT, TAB_KEYS, ShellTabBar

__all__ = [
    "DEFAULT_WIDTH",
    "RAIL_WIDTH",
    "STRIP_HEIGHT",
    "TAB_BAR_HEIGHT",
    "TAB_KEYS",
    "AppShell",
    "Command",
    "CommandPalette",
    "LandingPage",
    "ShellTabBar",
    "SidePanel",
    "StatusStrip",
    "commands_from_menu_bar",
    "commands_from_menus",
    "fit_window_to_screen",
    "forget_experiment",
    "menu_actions",
    "primary_available_geometry",
    "recent_experiments",
    "remember_experiment",
]
