"""Detect whether the optional ``[behavior]`` dependency stack is installed,
so the Behavior Analysis menu item can disable itself gracefully.

Mirrors the ``importlib.util.find_spec`` availability pattern used by
:mod:`glider.vision.yolo_install`, memoizing the result so the check is
cheap to call while building menus.
"""

from __future__ import annotations

import importlib.util

# (import name, pip/extra name shown to the user)
_REQUIRED = [
    ("umap", "umap-learn"),
    ("hdbscan", "hdbscan"),
    ("sklearn", "scikit-learn"),
    ("yaml", "pyyaml"),  # project.py save/load — don't report available if this is missing
]
_CACHE: bool | None = None


def missing_behavior_deps() -> list[str]:
    """Return the user-facing names of any missing ``[behavior]`` deps."""
    return [pip for mod, pip in _REQUIRED if importlib.util.find_spec(mod) is None]


def behavior_available() -> bool:
    """True when every ``[behavior]`` dependency can be imported (memoized)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = not missing_behavior_deps()
    return _CACHE
