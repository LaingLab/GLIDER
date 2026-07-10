"""Integration test: RunnerShell wired into MainWindow.

Builds a real GliderCore + MainWindow in runner mode to verify the
single-tab Runner shell boots on Setup and that switching to desktop mode
reuses (rather than duplicates) the shared HardwarePanel/CameraPanel
instances built by ``_create_runner_view``.

CameraPanel starts a QThread unconditionally in its constructor and only
stops it from ``closeEvent``/the widget's ``destroyed`` signal. Because it is
now nested several levels deep inside ``RunnerShell``'s QStackedWidget rather
than owned directly by a dock, qtbot's automatic teardown (which only closes
the widgets explicitly passed to ``addWidget``) does not reach it in time —
the thread is still alive when the interpreter starts finalizing, which
crashes the process. Each test therefore explicitly closes
``w._camera_panel`` (synchronously joining its CV thread) before returning.
This is a pre-existing CameraPanel lifecycle quirk, reproducible on main
in desktop mode too; it is unrelated to the runner-shell wiring itself.
"""

import asyncio

import pytest

pytestmark = pytest.mark.usefixtures("qtbot")


def _runner_window(qtbot):
    from PyQt6.QtWidgets import QApplication

    from glider.core.glider_core import GliderCore
    from glider.gui.main_window import MainWindow
    from glider.gui.view_manager import ViewManager, ViewMode

    app = QApplication.instance()
    core = GliderCore()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(core.initialize())
    vm = ViewManager(app)
    vm.mode = ViewMode.RUNNER
    w = MainWindow(core, view_manager=vm)
    w.switch_to_runner()
    qtbot.addWidget(w)
    return w, core, loop


def _teardown(w, core, loop):
    # Synchronously stop CameraPanel's CV thread before shutdown/loop
    # teardown — see module docstring.
    w._camera_panel.close()
    try:
        loop.run_until_complete(core.shutdown())
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_boots_on_setup_tab(qtbot):
    w, core, loop = _runner_window(qtbot)
    try:
        assert w._runner_shell._stack.currentIndex() == 0
    finally:
        _teardown(w, core, loop)


def test_desktop_switch_no_duplicate_hardware_panel(qtbot):
    w, core, loop = _runner_window(qtbot)
    try:
        hp_id = id(w._hardware_panel)
        w._switch_to_desktop_mode()
        assert id(w._hardware_panel) == hp_id
        assert w._hardware_dock.widget() is w._hardware_panel
    finally:
        _teardown(w, core, loop)
