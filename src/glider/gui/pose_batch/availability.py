"""Detect whether batch pose inference can run, so the Tools menu item can
disable itself gracefully.

Mirrors :mod:`glider.gui.behavior.availability`: an ``importlib.util.find_spec``
probe, memoized so it is cheap to call while building menus.
"""

from __future__ import annotations

import importlib.util

# (import name, pip/extra name shown to the user)
_REQUIRED = [("ultralytics", "ultralytics")]
_CACHE: bool | None = None


def missing_pose_batch_deps() -> list[str]:
    """Return the user-facing names of any missing dependencies."""
    return [pip for mod, pip in _REQUIRED if importlib.util.find_spec(mod) is None]


def pose_batch_available() -> bool:
    """True when batch pose inference can run (memoized)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = not missing_pose_batch_deps()
    return _CACHE
