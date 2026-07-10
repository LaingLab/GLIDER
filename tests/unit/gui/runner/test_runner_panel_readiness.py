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
    p._refresh_run_readiness()
    return p


def test_start_disabled_until_all_ready(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=False, has_start=False)
    assert p._start_btn.isEnabled() is False


def test_start_enabled_when_all_ready(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=True, has_start=True)
    assert p._start_btn.isEnabled() is True


def test_not_ready_hint_shown_when_blocked(qtbot, mock_core):
    p = _panel(qtbot, mock_core, board=False, has_start=False)
    assert p._not_ready_hint.isVisibleTo(p) is True

    p2 = _panel(qtbot, mock_core, board=True, has_start=True)
    assert p2._not_ready_hint.isVisibleTo(p2) is False


def test_header_timer_hidden_while_running(qtbot, mock_core):
    # IDLE (after RUNNING) runs the timer-snap path (else branch); None takes
    # its early return instead of painting a MagicMock into the timer label.
    mock_core.last_flow_duration_s = None
    p = _panel(qtbot, mock_core, board=True, has_start=True)
    p.update_state("RUNNING")
    assert p._runner_timer.isVisibleTo(p) is False
    p.update_state("IDLE")
    assert p._runner_timer.isVisibleTo(p) is True


def test_header_timer_hidden_while_paused(qtbot, mock_core):
    # PAUSED runs the timer-snap path (else branch); None takes its early return
    # instead of painting a MagicMock into the timer label.
    mock_core.last_flow_duration_s = None
    p = _panel(qtbot, mock_core, board=True, has_start=True)
    p.update_state("PAUSED")
    assert p._runner_timer.isVisibleTo(p) is False
    p.update_state("IDLE")
    assert p._runner_timer.isVisibleTo(p) is True


def test_no_emergency_button(qtbot, mock_core):
    # EMERGENCY STOP was removed from the Run tab: it overlapped START/STOP
    # when the controls block was too short with hardware disconnected.
    # Emergency stop remains available from the desktop menu only.
    p = _panel(qtbot, mock_core, board=True, has_start=True)
    assert not hasattr(p, "_emergency_btn")
    assert not hasattr(type(p), "emergency_stop_requested")
