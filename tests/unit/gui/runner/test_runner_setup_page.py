from types import SimpleNamespace

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QGridLayout, QWidget

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


def test_scroll_stored_and_no_hscroll(qtbot):
    p = _page(qtbot, _core(False, False))
    assert p._scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_file_buttons_in_grid(qtbot):
    p = _page(qtbot, _core(False, False))
    grid = p._new_btn.parentWidget().layout()
    assert isinstance(grid, QGridLayout)
    new_pos = grid.getItemPosition(grid.indexOf(p._new_btn))
    open_pos = grid.getItemPosition(grid.indexOf(p._open_btn))
    save_pos = grid.getItemPosition(grid.indexOf(p._save_btn))
    save_as_pos = grid.getItemPosition(grid.indexOf(p._save_as_btn))
    assert new_pos[0] == 0
    assert open_pos[0] == 0
    assert save_pos[0] == 1
    assert save_as_pos[0] == 1


def test_menu_button_small_and_in_status_row(qtbot):
    p = _page(qtbot, _core(False, False))
    assert p._menu_btn.size().width() <= 48
    assert p._menu_btn.size().height() <= 48
    # Lives alongside the status labels rather than as its own full-width row.
    assert p._menu_btn.parentWidget() is p._board_status.parentWidget()


def test_content_capped_to_viewport(qtbot):
    p = _page(qtbot, _core(False, False))
    p.resize(480, 800)
    p.show()
    QApplication.processEvents()
    assert p._content.maximumWidth() <= p._scroll.viewport().width() + 1


def test_menu_button_is_square(qtbot):
    p = _page(qtbot, _core(False, False))
    assert p._menu_btn.maximumHeight() == 40
    assert p._menu_btn.maximumWidth() == 40


def test_housekeeping_signals(qtbot):
    # Drive the menu-backed signals directly (menus are hard to click in tests):
    p = _page(qtbot, _core(True, True))
    with qtbot.waitSignal(p.help_requested):
        p.help_requested.emit()
    with qtbot.waitSignal(p.switch_to_desktop_requested):
        p.switch_to_desktop_requested.emit()
    with qtbot.waitSignal(p.close_requested):
        p.close_requested.emit()
