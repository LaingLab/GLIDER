from glider.core.experiment_session import ExperimentSession
from glider.core.graph_functions import GraphFunctionInfo
from glider.gui.runner.manual_control_panel import (
    ManualControlPanel,
    build_picker_labels,
)


class _HW:
    def __init__(self, connected):
        self._c = connected

    def is_any_board_connected(self):
        return self._c


class _Core:
    def __init__(self, session, connected=True):
        self.session = session
        self.hardware_manager = _HW(connected)


def test_build_picker_labels_disambiguates_duplicate_names():
    infos = [
        GraphFunctionInfo("node_aaaa1111", "MyFunction", True),
        GraphFunctionInfo("node_bbbb2222", "MyFunction", True),
        GraphFunctionInfo("node_cccc3333", "Purge", True),
    ]
    labels = build_picker_labels(infos)
    display = [d for d, _ in labels]
    assert display[2] == "Purge"
    assert display[0] != display[1]
    assert display[0].endswith("1111") and display[1].endswith("2222")
    assert [nid for _, nid in labels] == [
        "node_aaaa1111",
        "node_bbbb2222",
        "node_cccc3333",
    ]


def test_assign_function_persists_entry(qtbot):
    session = ExperimentSession()
    panel = ManualControlPanel(_Core(session))
    qtbot.addWidget(panel)
    panel.assign_function(slot=0, start_node_id="s1", label="Purge")
    assert session.manual_controls == [{"slot": 0, "start_node_id": "s1", "label": "Purge"}]


def test_clear_repacks_slots(qtbot):
    session = ExperimentSession()
    session.set_manual_controls(
        [
            {"slot": 0, "start_node_id": "s1", "label": "A"},
            {"slot": 1, "start_node_id": "s2", "label": "B"},
        ]
    )
    panel = ManualControlPanel(_Core(session))
    qtbot.addWidget(panel)
    panel.clear_slot(0)
    assert session.manual_controls == [{"slot": 0, "start_node_id": "s2", "label": "B"}]


def test_run_requested_emitted_on_activate(qtbot):
    session = ExperimentSession()
    session.set_manual_controls([{"slot": 0, "start_node_id": "s1", "label": "A"}])
    panel = ManualControlPanel(_Core(session))
    qtbot.addWidget(panel)
    with qtbot.waitSignal(panel.function_run_requested, timeout=500) as blocker:
        panel.activate_slot(0)
    assert blocker.args == ["s1"]


def test_buttons_disabled_when_no_hardware(qtbot):
    session = ExperimentSession()
    session.set_manual_controls([{"slot": 0, "start_node_id": "s1", "label": "A"}])
    panel = ManualControlPanel(_Core(session, connected=False))
    qtbot.addWidget(panel)
    panel.refresh()
    assert panel.is_slot_enabled(0) is False


def test_buttons_disabled_in_blocked_state(qtbot):
    session = ExperimentSession()
    session.set_manual_controls([{"slot": 0, "start_node_id": "s1", "label": "A"}])
    panel = ManualControlPanel(_Core(session, connected=True))
    qtbot.addWidget(panel)
    panel.update_state("ERROR")
    assert panel.is_slot_enabled(0) is False
    panel.update_state("IDLE")
    assert panel.is_slot_enabled(0) is True
