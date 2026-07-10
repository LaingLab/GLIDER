from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QWidget

from glider.core.experiment_session import FlowConfig, NodeConfig

pytestmark = pytest.mark.usefixtures("qtbot")


def _core(board, has_start, name="Exp"):
    flow = FlowConfig(nodes=[NodeConfig(id="s", node_type="StartExperiment")] if has_start else [])
    session = SimpleNamespace(flow=flow, metadata=SimpleNamespace(name=name))
    hw = SimpleNamespace(
        is_any_board_connected=lambda: board, connected_board_description=lambda: "Uno"
    )
    return SimpleNamespace(session=session, hardware_manager=hw)


def _page(qtbot, core, hw=None):
    from glider.gui.runner.runner_setup_page import RunnerSetupPage

    p = RunnerSetupPage(core, hardware_widget=hw if hw is not None else QWidget())
    qtbot.addWidget(p)
    p.refresh()
    return p


def test_open_button_emits(qtbot):
    p = _page(qtbot, _core(False, False))
    with qtbot.waitSignal(p.open_requested):
        p._open_btn.click()


def test_new_and_save_buttons_emit(qtbot):
    p = _page(qtbot, _core(False, False))
    with qtbot.waitSignal(p.new_requested):
        p._new_btn.click()
    with qtbot.waitSignal(p.save_requested):
        p._save_btn.click()


def test_save_as_and_board_settings_emit(qtbot):
    p = _page(qtbot, _core(True, False))
    with qtbot.waitSignal(p.save_as_requested):
        p._save_as_btn.click()
    with qtbot.waitSignal(p.board_settings_requested):
        p._connect_btn.click()


def test_status_line_reflects_readiness(qtbot):
    p = _page(qtbot, _core(board=True, has_start=True, name="My Exp"))
    assert "✓" in p._board_status.text()
    assert "✓" in p._exp_status.text()
    p2 = _page(qtbot, _core(board=False, has_start=False))
    assert "✗" in p2._board_status.text()
    assert "✗" in p2._exp_status.text()


def test_hardware_widget_embedded(qtbot):
    hw = QWidget()
    p = _page(qtbot, _core(True, True), hw=hw)
    assert p.isAncestorOf(hw)


def test_housekeeping_signals(qtbot):
    # Drive the menu-backed signals directly (menus are hard to click in tests):
    p = _page(qtbot, _core(True, True))
    with qtbot.waitSignal(p.help_requested):
        p.help_requested.emit()
    with qtbot.waitSignal(p.switch_to_desktop_requested):
        p.switch_to_desktop_requested.emit()
    with qtbot.waitSignal(p.close_requested):
        p.close_requested.emit()
