"""
GLIDER GUI Shell

The PyQt6 application that provides the visual interface.
Acts as the View and Controller in the MVC paradigm.

Supports two distinct modes:
- Desktop Mode: Full IDE with dockable widgets and node graph editing
- Runner Mode: Touch-optimized dashboard for Raspberry Pi (480x800)
"""

__all__ = [
    "MainWindow",
    "ViewManager",
    "ViewMode",
]


def __getattr__(name: str):
    # Lazy, like glider/__init__: the splash process imports glider.gui.splash
    # and must not pay for the main window (and the vision stack under it).
    if name == "MainWindow":
        from glider.gui.main_window import MainWindow

        return MainWindow
    if name in ("ViewManager", "ViewMode"):
        from glider.gui import view_manager

        return getattr(view_manager, name)
    raise AttributeError(f"module 'glider.gui' has no attribute {name!r}")
