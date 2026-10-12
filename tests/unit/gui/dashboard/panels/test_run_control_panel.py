import pytest

from glider.core.experiment_session import ExperimentSession, FlowConfig, NodeConfig
from glider.gui.dashboard.panels.run_control_panel import RunControlPanel

pytestmark = pytest.mark.usefixtures("qtbot")


def _panel(qtbot, mock_core, *, board=False, has_start=False, name="Exp"):
    mock_core.hardware_manager.is_any_board_connected = lambda: board
    session = ExperimentSession()
    nodes = [NodeConfig(id="s", node_type="StartExperiment")] if has_start else []
    session._flow = FlowConfig(nodes=nodes)
    session.metadata.name = name
    mock_core.session = session
    mock_core.last_flow_duration_s = None
    p = RunControlPanel(mock_core)
    qtbot.addWidget(p)
    return p


def test_start_disabled_until_ready(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=False, has_start=False)
    assert panel._start_btn.isEnabled() is False
    assert not panel._stop_btn.isVisibleTo(panel)
    assert panel._not_ready_hint.isVisibleTo(panel)


def test_start_and_stop_gated_on_the_run(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=True, has_start=True)
    assert panel._start_btn.isEnabled() and panel._start_btn.isVisibleTo(panel)
    assert not panel._stop_btn.isVisibleTo(panel) and not panel._pause_btn.isVisibleTo(panel)

    panel.update_state("RUNNING")
    assert not panel._start_btn.isVisibleTo(panel)
    assert panel._stop_btn.isVisibleTo(panel) and panel._stop_btn.isEnabled()
    assert panel._pause_btn.text() == "Pause"

    panel.update_state("PAUSED")
    assert panel._pause_btn.text() == "Resume"

    panel.update_state("READY")
    assert panel._start_btn.isVisibleTo(panel) and panel._start_btn.isEnabled()


def test_pause_and_resume_emit(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=True, has_start=True)
    panel.update_state("RUNNING")
    with qtbot.waitSignal(panel.pause_requested, timeout=1000):
        panel._pause_btn.click()
    panel.update_state("PAUSED")
    with qtbot.waitSignal(panel.resume_requested, timeout=1000):
        panel._pause_btn.click()


def test_start_button_emits_start_requested(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=True, has_start=True)
    with qtbot.waitSignal(panel.start_requested, timeout=1000):
        panel._start_btn.click()


def test_stop_button_emits_stop_requested(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    panel.update_state("RUNNING")
    with qtbot.waitSignal(panel.stop_requested, timeout=1000):
        panel._stop_btn.click()


def test_elapsed_updated_emitted_on_timer_display(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    with qtbot.waitSignal(panel.elapsed_updated, timeout=1000) as blocker:
        panel._set_timer_display(12.34)
    assert blocker.args == ["00:12.34"]


def test_timer_snaps_to_flow_duration_on_stop(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    panel.update_state("RUNNING")
    mock_core.last_flow_duration_s = 10.0
    with qtbot.waitSignal(panel.elapsed_updated, timeout=1000) as blocker:
        panel.update_state("STOPPED")
    assert blocker.args == ["00:10.00"]


def test_elapsed_shows_on_own_timer(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    panel._set_timer_display(12.34)
    assert panel._timer.text() == "00:12.34"


def test_board_label_in_sub_line(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=False, has_start=True)
    assert "None needed" in panel._sub.text()


def test_missing_field_blocks_start_with_hint(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=True, has_start=True)
    panel.set_missing_field("Dam ID")
    assert not panel._start_btn.isEnabled()
    assert panel._not_ready_hint.text() == "Fill in Dam ID to start."
    assert panel._not_ready_hint.isVisibleTo(panel)
    panel.set_missing_field(None)
    assert panel._start_btn.isEnabled()


def _clock(monkeypatch, now):
    from glider.gui.dashboard.panels import run_control_panel as mod

    monkeypatch.setattr(mod.time, "time", lambda: now[0])


def test_pause_freezes_the_clock_without_snapping(qtbot, mock_core, monkeypatch):
    now = [1000.0]
    _clock(monkeypatch, now)
    panel = _panel(qtbot, mock_core)
    panel.update_state("RUNNING")
    panel._experiment_start_time = now[0] - 10
    panel._update_elapsed_time()
    mock_core.last_flow_duration_s = 99.0  # must not be shown while paused
    panel.update_state("PAUSED")
    assert panel._timer.text() == "00:10.00"
    assert not panel._elapsed_timer.isActive()


def test_resume_keeps_the_elapsed_time(qtbot, mock_core, monkeypatch):
    now = [1000.0]
    _clock(monkeypatch, now)
    panel = _panel(qtbot, mock_core)
    panel.update_state("RUNNING")
    panel._experiment_start_time = now[0] - 10
    panel.update_state("PAUSED")
    now[0] += 30  # time spent paused is not run time
    panel.update_state("RUNNING")
    assert panel._timer.text() == "00:10.00"
    assert panel._elapsed_timer.isActive()
    now[0] += 5
    panel._update_elapsed_time()
    assert panel._timer.text() == "00:15.00"


def test_stop_after_a_pause_shows_run_time_not_wall_time(qtbot, mock_core, monkeypatch):
    now = [1000.0]
    _clock(monkeypatch, now)
    panel = _panel(qtbot, mock_core)
    panel.update_state("RUNNING")
    now[0] += 50
    panel.update_state("PAUSED")
    now[0] += 10
    panel.update_state("PAUSED")  # a repeat callback must not restart the pause
    panel.update_state("RUNNING")
    now[0] += 10
    mock_core.last_flow_duration_s = 70.0  # wall time, pause included
    panel.update_state("STOPPING")
    assert panel._timer.text() == "01:00.00"
    panel.update_state("READY")
    assert panel._timer.text() == "01:00.00"
