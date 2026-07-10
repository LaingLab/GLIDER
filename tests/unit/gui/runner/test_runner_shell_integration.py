"""Integration test: RunnerShell wired into MainWindow.

Builds a real GliderCore + MainWindow in runner mode to verify the
single-tab Runner shell boots on Setup and that switching to desktop mode
reuses (rather than duplicates) the shared HardwarePanel/CameraPanel
instances built by ``_create_runner_view``.

CameraPanel starts a CV QThread in its constructor and only stops it from
``closeEvent``/``destroyed``. Since it is now nested inside RunnerShell's
QStackedWidget (not owned directly by a dock), Qt does not fire those on a
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


def test_desktop_switch_no_duplicate_hardware_panel(qtbot):
    w, core, loop = _runner_window(qtbot)
    try:
        hp_id = id(w._hardware_panel)
        cam_id = id(w._camera_panel)
        w._switch_to_desktop_mode()
        assert id(w._hardware_panel) == hp_id
        assert w._hardware_dock.widget() is w._hardware_panel
        assert id(w._camera_panel) == cam_id
        assert w._camera_dock.widget() is w._camera_panel
    finally:
        _teardown(core, loop)
