"""The GLIDER theme must reach the unparented tool windows.

Behavior Analysis and Batch Pose Tracking are deliberately top-level with
``parent=None``, so a stylesheet set on MainWindow never reaches them and they
render in the OS default theme. The sheet therefore belongs on the
QApplication.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def test_the_stylesheet_is_applied_to_the_application_not_the_window():
    """Pinned at the source: a sheet on MainWindow cannot reach a top-level."""
    import inspect

    from glider import __main__ as entry

    source = inspect.getsource(entry)
    assert "app.setStyleSheet(" in source
    assert (
        "window.setStyleSheet(" not in source
    ), "a sheet on MainWindow does not reach unparented tool windows"


def test_the_tool_windows_are_opened_unparented():
    """Records why the app-level sheet is required, not merely preferred."""
    import inspect

    from glider.gui import main_window

    source = inspect.getsource(main_window)
    assert "BehaviorAnalysisWindow(parent=None)" in source
    assert "PoseBatchWindow(parent=None)" in source


def test_an_application_stylesheet_reaches_an_unparented_window(qtbot):
    """The mechanism itself, so the fix is verified rather than assumed."""
    from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton

    app = QApplication.instance()
    previous = app.styleSheet()
    try:
        app.setStyleSheet("QPushButton { color: rgb(1, 2, 3); }")
        orphan = QMainWindow(None)
        qtbot.addWidget(orphan)
        button = QPushButton("x", orphan)
        orphan.ensurePolished()
        button.ensurePolished()
        from PyQt6.QtGui import QPalette

        assert button.palette().color(QPalette.ColorRole.ButtonText).getRgb()[:3] == (1, 2, 3)
    finally:
        app.setStyleSheet(previous)


def test_the_new_dialogs_use_the_palette_not_hardcoded_light_colours(qtbot):
    """A light background would be unreadable against the Deep Navy theme."""
    import inspect

    from glider.gui.behavior import keypoint_confirm, keypoint_editor

    for module in (keypoint_editor, keypoint_confirm):
        source = inspect.getsource(module)
        for offender in ("#fafafa", "#e0e0e0", "#212121", "#c0392b"):
            assert offender not in source, f"{module.__name__} still hardcodes {offender}"
