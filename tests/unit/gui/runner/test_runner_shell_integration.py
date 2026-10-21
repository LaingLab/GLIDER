"""Integration test: RunnerShell wired into MainWindow.

Builds a real GliderCore + MainWindow in runner mode to verify the
single-tab Runner shell boots on Setup. The desktop-switch reuse of the shared
HardwarePanel/CameraPanel, and the manual controls' home, are covered in
``test_pi_dashboard_window.py`` now that the Pi dashboard hosts them.

CameraPanel starts a CV QThread in its constructor and only stops it from
``closeEvent``/``destroyed``. Since it is now nested inside RunnerShell's
QStackedWidget (not owned directly by the window), Qt does not fire those on a
MainWindow teardown, which used to leave the thread alive at interpreter
finalization and crash the process. ``MainWindow.closeEvent`` now stops the
CV thread deterministically, and pytest-qt's ``addWidget`` teardown fires
that ``closeEvent`` — so no manual camera-panel close is needed here.
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


def _teardown(core, loop):
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
        _teardown(core, loop)
