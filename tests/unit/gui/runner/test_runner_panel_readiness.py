from unittest.mock import MagicMock

import pytest

from glider.core.experiment_session import FlowConfig, NodeConfig

pytestmark = pytest.mark.usefixtures("qtbot")


def _panel(qtbot, mock_core, *, board, has_start, name="Exp"):
    from glider.gui.panels.runner_panel import RunnerPanel

    mock_core.hardware_manager.is_any_board_connected = lambda: board
    nodes = [NodeConfig(id="s", node_type="StartExperiment")] if has_start else []
    mock_core.session.flow = FlowConfig(nodes=nodes)
    mock_core.session.metadata.name = name
    p = RunnerPanel(mock_core, MagicMock())
    qtbot.addWidget(p)
    p.refresh_readiness()
    return p


def test_start_disabled_until_all_ready(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=False, has_start=False)
    assert p._start_btn.isEnabled() is False
    assert p._readiness_strip.isVisibleTo(p) is True


def test_start_enabled_when_all_ready(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=True, has_start=True)
    assert p._start_btn.isEnabled() is True


def test_board_row_tap_emits_board_settings(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=False, has_start=True)
    with qtbot.waitSignal(p.board_settings_requested, timeout=1000):
        p._board_row.click()


def test_experiment_row_tap_emits_open(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=True, has_start=False)
    with qtbot.waitSignal(p.open_requested, timeout=1000):
        p._exp_row.click()


def test_strip_hidden_while_running(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=True, has_start=True)
    assert p._readiness_strip.isVisibleTo(p) is True
    p.update_state("RUNNING")
    assert p._readiness_strip.isVisibleTo(p) is False


def test_strip_hidden_while_paused(qtbot, mock_core):
    # PAUSED runs the timer-snap path (else branch); None takes its early return
    # instead of painting a MagicMock into the timer label.
    mock_core.last_flow_duration_s = None
    p = _panel(qtbot, mock_core, board=True, has_start=True)
    assert p._readiness_strip.isVisibleTo(p) is True
    p.update_state("PAUSED")
    assert p._readiness_strip.isVisibleTo(p) is False


def test_experiment_row_blocked_rendering(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=True, has_start=False)
    assert p._reload_btn.isVisibleTo(p) is False
    assert "No experiment loaded" in p._exp_row.text()
    assert p._exp_row.property("readinessRow") == "blocked"


def test_experiment_row_ready_rendering(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=True, has_start=True, name="My Exp")
    assert p._reload_btn.isVisibleTo(p) is True
    assert p._exp_row.text().startswith("✓")
    assert p._exp_row.property("readinessRow") == "ok"


def test_board_row_ready_rendering(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=True, has_start=False)
    assert p._board_row.property("readinessRow") == "ok"
    assert p._board_row.text().startswith("✓")


def test_board_row_blocked_rendering(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=False, has_start=True)
    assert p._board_row.property("readinessRow") == "blocked"
    assert "Board not connected" in p._board_row.text()
