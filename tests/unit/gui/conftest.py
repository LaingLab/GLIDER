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
def main_window_factory(qtbot, tmp_path):
    """Build a real MainWindow backed by a real GliderCore + asyncio loop.

    Mirrors the helper pattern in ``tests/unit/gui/test_dashboard_integration.py``
    but as a reusable factory. Each created (core, loop) is torn down after the test.

    Every window gets a throwaway ``QSettings`` file unless the test supplies
    one. MainWindow reads ``first_run/*`` at construction to decide whether to
    offer the one-time Lab Setup form, so a default-constructed ``QSettings``
    here would read the developer's real state and could pop a modal dialog in
    the middle of the suite -- which blocks, rather than fails.
    """
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication

    from glider.core.glider_core import GliderCore
    from glider.gui.main_window import MainWindow
    from glider.gui.view_manager import ViewManager, ViewMode

    created = []
    made = 0

    def _make(desktop_mode: bool = True, settings: QSettings | None = None):
        nonlocal made
        app = QApplication.instance()
        core = GliderCore()
        loop = asyncio.new_event_loop()
        created.append((core, loop))  # register before anything below can raise
        asyncio.set_event_loop(loop)
        loop.run_until_complete(core.initialize())
        vm = ViewManager(app)
        vm.mode = ViewMode.DESKTOP if desktop_mode else ViewMode.RUNNER
        if settings is None:
            made += 1
            settings = QSettings(
                str(tmp_path / f"window_settings_{made}.ini"), QSettings.Format.IniFormat
            )
        w = MainWindow(core, view_manager=vm, settings=settings)
        if not desktop_mode:
            w.switch_to_runner()

        def _forget_unsaved_work(_widget) -> None:
            """Let a test end while the session is dirty.

            ``MainWindow.closeEvent`` asks ``_check_save``, which puts up a
            *modal* "Save changes?" box, and pytest-qt closes every registered
            widget from a teardown hook that runs before any fixture finalizer
            -- so nothing a fixture or a test body does afterwards can get in
            front of it. A test that leaves unsaved work behind (a graph edit
            is one) would otherwise hang the whole run instead of ending, and
            it would hang hardest on the run where it *failed*.
            """
            if getattr(core, "session", None) is not None:
                core.session._mark_clean()

        qtbot.addWidget(w, before_close_func=_forget_unsaved_work)
        return w

    yield _make

    for core, loop in created:
        try:
            loop.run_until_complete(core.shutdown())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
