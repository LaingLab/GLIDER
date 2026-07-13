import pytest

from glider.core.experiment_session import FlowConfig, NodeConfig
from glider.gui.dashboard.panels.run_control_panel import RunControlPanel

pytestmark = pytest.mark.usefixtures("qtbot")


def _panel(qtbot, mock_core, *, board=False, has_start=False, name="Exp"):
    # A bare mock_core.session.flow is a MagicMock whose .nodes cannot be
    # iterated, so compute_readiness() raises unless we set a real FlowConfig.
    mock_core.hardware_manager.is_any_board_connected = lambda: board
    nodes = [NodeConfig(id="s", node_type="StartExperiment")] if has_start else []
    mock_core.session.flow = FlowConfig(nodes=nodes)
    mock_core.session.metadata.name = name
    mock_core.last_flow_duration_s = None
    p = RunControlPanel(mock_core)
    qtbot.addWidget(p)
    return p


def test_start_disabled_until_ready(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=False, has_start=False)
    assert panel._start_btn.isEnabled() is False


def test_start_button_emits_start_requested(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=True, has_start=True)
    with qtbot.waitSignal(panel.start_requested, timeout=1000):
        panel._start_btn.click()


def test_stop_button_emits_stop_requested(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    with qtbot.waitSignal(panel.stop_requested, timeout=1000):
        panel._stop_btn.click()


def test_update_state_sets_state_pill(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    panel.update_state("RUNNING")
    assert panel._status_label.text() == "RUNNING"


def test_elapsed_updated_emitted_on_timer_display(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    with qtbot.waitSignal(panel.elapsed_updated, timeout=1000) as blocker:
        panel._set_timer_display(12.34)
    assert blocker.args == ["00:12.34"]


def test_no_stall_instrument_attribute(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    assert not hasattr(panel, "_stall_timer")
