"""Smoke tests for the top-level Behavior Analysis window.

Just enough to prove the window builds its three tabs and can default
its project directory from config — the tab bodies (Annotate/Train/Apply)
get their own focused tests once the window exists.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def test_window_has_three_tabs(qtbot, tmp_path):
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles == ["Annotate", "Train", "Apply"]


def test_window_defaults_project_dir_from_config(qtbot):
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(parent=None)
    qtbot.addWidget(win)
    assert win.project_dir is not None
