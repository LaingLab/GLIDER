"""Shared fixtures for GUI panel unit tests.

The ``tiny_behavior_model`` fixture now lives in ``tests/unit/conftest.py`` so
both the GUI panel tests and the live/offline behavior-classify parity test can
share a single, byte-for-byte-consistent model definition. This module is kept
so pytest still collects the ``tests/unit/gui`` package cleanly.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _isolate_dashboard_layout(tmp_path, monkeypatch):
    """Redirect the per-user config dir to a tmp path so tests that build
    MainWindow/DashboardView never read or write the developer's real
    ~/.glider/dashboard_layout.json."""
    from glider.core.config import get_config

    monkeypatch.setattr(get_config().paths, "user_config_dir", tmp_path)
    yield


@pytest.fixture
def main_window_factory(qtbot):
    """Build a real MainWindow backed by a real GliderCore + asyncio loop.

    Mirrors the helper pattern in ``tests/unit/gui/test_dashboard_integration.py``
    but as a reusable factory. Each created (core, loop) is torn down after the test.
    """
    from PyQt6.QtWidgets import QApplication

    from glider.core.glider_core import GliderCore
    from glider.gui.main_window import MainWindow
    from glider.gui.view_manager import ViewManager, ViewMode

    created = []

    def _make(desktop_mode: bool = True):
        app = QApplication.instance()
        core = GliderCore()
        loop = asyncio.new_event_loop()
        created.append((core, loop))  # register before anything below can raise
        asyncio.set_event_loop(loop)
        loop.run_until_complete(core.initialize())
        vm = ViewManager(app)
        vm.mode = ViewMode.DESKTOP if desktop_mode else ViewMode.RUNNER
        w = MainWindow(core, view_manager=vm)
        if not desktop_mode:
            w.switch_to_runner()
        qtbot.addWidget(w)
        return w

    yield _make

    for core, loop in created:
        try:
            loop.run_until_complete(core.shutdown())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
