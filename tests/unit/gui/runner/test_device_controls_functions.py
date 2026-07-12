"""Functions section on the Runner Manual surface (Slice 3).

The generated device controls also surface each complete graph function
(StartFunction→EndFunction) as a large run button that binds by start_node_id,
emits ``function_run_requested`` on tap, disambiguates duplicate names, hides
when there are none, and shows a per-button running affordance.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton

from glider.core.experiment_session import ExperimentSession
from glider.gui.runner.device_controls import RunnerDeviceControls

pytestmark = pytest.mark.usefixtures("qtbot")


def _session_with_flow(nodes, connections):
    session = ExperimentSession()
    data = session.to_dict()
    data["flow"] = {"nodes": nodes, "connections": connections}
    return ExperimentSession.from_dict(data)


def _node(node_id, node_type, state=None):
    return {
        "id": node_id,
        "node_type": node_type,
        "position": [0, 0],
        "state": state or {},
        "device_id": None,
    }


def _conn(from_node, to_node):
    return {
        "id": f"{from_node}->{to_node}",
        "from_node": from_node,
        "from_output": 0,
        "to_node": to_node,
        "to_input": 0,
        "connection_type": "exec",
    }


def _hw(devices=None):
    return SimpleNamespace(devices=devices or {})


def _controls(session):
    w = RunnerDeviceControls(_hw(), session_fn=lambda: session)
    w.refresh()
    return w


def test_complete_function_renders_a_run_button(qtbot):
    session = _session_with_flow(
        nodes=[
            _node("s1", "StartFunction", {"function_name": "Purge"}),
            _node("e1", "EndFunction"),
        ],
        connections=[_conn("s1", "e1")],
    )
    w = _controls(session)
    qtbot.addWidget(w)
    btn, label = w._function_buttons["s1"]
    assert isinstance(btn, QPushButton)
    assert label == "Purge"


def test_incomplete_function_is_not_offered(qtbot):
    session = _session_with_flow(
        nodes=[_node("s1", "StartFunction", {"function_name": "Dangling"})],
        connections=[],
    )
    w = _controls(session)
    qtbot.addWidget(w)
    assert w._function_buttons == {}


def test_duplicate_names_are_disambiguated_but_bind_by_id(qtbot):
    session = _session_with_flow(
        nodes=[
            _node("start-aaaa", "StartFunction"),
            _node("e1", "EndFunction"),
            _node("start-bbbb", "StartFunction"),
            _node("e2", "EndFunction"),
        ],
        connections=[_conn("start-aaaa", "e1"), _conn("start-bbbb", "e2")],
    )
    w = _controls(session)
    qtbot.addWidget(w)
    assert set(w._function_buttons) == {"start-aaaa", "start-bbbb"}
    assert w._function_buttons["start-aaaa"][1] == "MyFunction aaaa"
    assert w._function_buttons["start-bbbb"][1] == "MyFunction bbbb"


def test_tapping_a_function_emits_start_node_id(qtbot):
    session = _session_with_flow(
        nodes=[_node("s1", "StartFunction", {"function_name": "Go"}), _node("e1", "EndFunction")],
        connections=[_conn("s1", "e1")],
    )
    w = _controls(session)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.function_run_requested) as sig:
        w._function_buttons["s1"][0].click()
    assert sig.args == ["s1"]


def test_running_affordance_disables_then_restores(qtbot):
    session = _session_with_flow(
        nodes=[_node("s1", "StartFunction", {"function_name": "Go"}), _node("e1", "EndFunction")],
        connections=[_conn("s1", "e1")],
    )
    w = _controls(session)
    qtbot.addWidget(w)
    btn = w._function_buttons["s1"][0]

    w.set_function_running("s1", True)
    assert not btn.isEnabled()
    assert "Running" in btn.text()

    w.set_function_running("s1", False)
    assert btn.isEnabled()
    assert btn.text() == "Go"


def test_no_session_fn_means_no_functions_section(qtbot):
    w = RunnerDeviceControls(_hw())  # device-only construction
    w.refresh()
    qtbot.addWidget(w)
    assert w._function_buttons == {}
