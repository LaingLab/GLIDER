"""The Builder shell: one content surface, a panel each side, one strip.

Kept out of ``main_window.py`` on purpose -- that file is already the largest
in the GUI, and the shell's pieces are worth being able to test without
constructing a window.
"""

from __future__ import annotations

from glider.gui.shell.side_panel import DEFAULT_WIDTH, RAIL_WIDTH, SidePanel
from glider.gui.shell.status_strip import STRIP_HEIGHT, StatusStrip

__all__ = ["DEFAULT_WIDTH", "RAIL_WIDTH", "STRIP_HEIGHT", "SidePanel", "StatusStrip"]
