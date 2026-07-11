"""Tests for the Tools → Behavior Analysis menu entry in MainWindow.

A full ``MainWindow`` build against a mock core blocks (panel construction
spins up async/mocked machinery), so these tests exercise the real
``_setup_menu`` code path in isolation: bypass the heavy ``__init__`` with
``__new__`` + ``QMainWindow.__init__``, stub the one attribute
``_setup_menu`` needs (``_view_manager.is_runner_mode``), then call it. Every
``_on_*`` handler it connects to is a class method, so it resolves as a bound
method on the bare instance.

Each test takes the ``qtbot`` fixture purely to guarantee a ``QApplication``
exists (constructing a ``QMainWindow`` without one hangs on offscreen
Windows) — but we deliberately do NOT ``qtbot.addWidget`` the window: its
``closeEvent`` touches panel attributes the bypassed ``__init__`` never
created, so letting qtbot auto-close it at teardown would raise.
"""

from __future__ import annotations

import types

from PyQt6.QtWidgets import QMainWindow


def _menu_only_window():
    """A MainWindow with only _setup_menu run (no _setup_ui panels)."""
    from glider.gui.main_window import MainWindow

    win = MainWindow.__new__(MainWindow)  # skip the heavy __init__
    QMainWindow.__init__(win)  # real Qt base so menuBar() works
    win._view_manager = types.SimpleNamespace(is_runner_mode=False)
    win._setup_menu()
    return win


def _find_menu(win, title):
    for action in win.menuBar().actions():
        if action.text().replace("&", "") == title:
            return action.menu()
    return None


def test_tools_menu_has_behavior_action(qtbot):
    win = _menu_only_window()
    try:
        tools = _find_menu(win, "Tools")
        assert tools is not None, "Tools menu missing"
        labels = [a.text().replace("&", "") for a in tools.actions()]
        assert any("Behavior Analysis" in label for label in labels)
    finally:
        win.deleteLater()


def test_behavior_action_disabled_when_deps_missing(qtbot, monkeypatch):
    # Force the availability probe to report the stack absent, then rebuild
    # the menu and confirm the action is disabled with an install tooltip.
    from glider.gui.behavior import availability

    monkeypatch.setattr(availability, "behavior_available", lambda: False)
    monkeypatch.setattr(availability, "missing_behavior_deps", lambda: ["umap-learn", "hdbscan"])

    win = _menu_only_window()
    try:
        tools = _find_menu(win, "Tools")
        behavior = next(
            a for a in tools.actions() if "Behavior Analysis" in a.text().replace("&", "")
        )
        assert behavior.isEnabled() is False
        assert "pip install glider[behavior]" in behavior.toolTip()
        assert "umap-learn" in behavior.toolTip()
    finally:
        win.deleteLater()


def test_behavior_action_enabled_when_deps_present(qtbot, monkeypatch):
    from glider.gui.behavior import availability

    monkeypatch.setattr(availability, "behavior_available", lambda: True)

    win = _menu_only_window()
    try:
        tools = _find_menu(win, "Tools")
        behavior = next(
            a for a in tools.actions() if "Behavior Analysis" in a.text().replace("&", "")
        )
        assert behavior.isEnabled() is True
    finally:
        win.deleteLater()


def test_tools_menu_has_gpu_check_action(qtbot):
    win = _menu_only_window()
    try:
        tools = _find_menu(win, "Tools")
        assert tools is not None, "Tools menu missing"
        gpu = next(
            (a for a in tools.actions() if "GPU / Device Check" in a.text().replace("&", "")),
            None,
        )
        assert gpu is not None, "GPU / Device Check action missing"
        # Always enabled — it reports missing torch/GPU rather than being gated.
        assert gpu.isEnabled() is True
    finally:
        win.deleteLater()
