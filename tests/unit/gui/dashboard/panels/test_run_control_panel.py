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
    assert panel._stop_btn.isEnabled() is False
    assert panel._not_ready_hint.isVisibleTo(panel)


def test_start_and_stop_gated_on_the_run(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=True, has_start=True)
    assert panel._start_btn.isEnabled() and not panel._stop_btn.isEnabled()

    panel.update_state("RUNNING")
    assert not panel._start_btn.isEnabled() and panel._stop_btn.isEnabled()

    panel.update_state("PAUSED")
    assert not panel._start_btn.isEnabled() and panel._stop_btn.isEnabled()

    panel.update_state("READY")
    assert panel._start_btn.isEnabled() and not panel._stop_btn.isEnabled()


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


def test_status_line_reports_no_board_needed(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, board=False, has_start=True)
    assert panel._board_status.text() == "Board: ✓ None needed"


def test_metadata_edits_write_to_session_and_mark_dirty(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    session = mock_core.session
    session._mark_clean()

    panel._name_edit.setText("Open field 3")
    panel._meta_edits["experimenter"].setText("GB")
    panel._meta_edits["protocol"].setText("P-17")
    panel._notes_edit.setPlainText("line one\nline two")

    md = session.metadata
    assert (md.name, md.experimenter, md.protocol) == ("Open field 3", "GB", "P-17")
    assert md.notes == "line one\nline two"
    assert session.is_dirty


def test_metadata_locked_while_live(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    fields = (panel._name_edit, panel._notes_edit, *panel._meta_edits.values())
    for state, locked in (("RUNNING", True), ("PAUSED", True), ("STOPPED", False)):
        panel.update_state(state)
        assert all(f.isReadOnly() is locked for f in fields), state


def test_refresh_loads_new_session(qtbot, mock_core):
    panel = _panel(qtbot, mock_core, name="First")
    assert panel._name_edit.text() == "First"
    new = ExperimentSession()
    new.metadata.name = "Second"
    new.metadata.experimenter = "AB"
    mock_core.session = new
    panel.refresh()
    assert panel._name_edit.text() == "Second"
    assert panel._meta_edits["experimenter"].text() == "AB"
    assert not new.is_dirty  # loading the fields is not an edit


def test_file_action_buttons_emit(qtbot, mock_core):
    panel = _panel(qtbot, mock_core)
    for label, signal in (
        ("New", panel.new_requested),
        ("Open", panel.open_requested),
        ("Save", panel.save_requested),
        ("Save As", panel.save_as_requested),
        ("Connect / Ports", panel.board_settings_requested),
    ):
        with qtbot.waitSignal(signal, timeout=1000):
            panel._file_buttons[label].click()
        assert panel._file_buttons[label].minimumHeight() >= 48
