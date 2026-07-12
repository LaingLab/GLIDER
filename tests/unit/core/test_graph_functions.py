# tests/unit/core/test_graph_functions.py
from glider.core.experiment_session import ExperimentSession
from glider.core.graph_functions import (
    GraphFunctionInfo,
    build_picker_labels,
    find_run_param,
    list_graph_functions,
)


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


def test_lists_one_entry_per_start_function_keyed_by_node_id():
    session = _session_with_flow(
        nodes=[
            _node("s1", "StartFunction", {"function_name": "Purge"}),
            _node("e1", "EndFunction"),
        ],
        connections=[_conn("s1", "e1")],
    )
    result = list_graph_functions(session)
    assert result == [GraphFunctionInfo(start_node_id="s1", name="Purge", has_end=True)]


def test_start_function_without_reachable_end_has_end_false():
    session = _session_with_flow(
        nodes=[_node("s1", "StartFunction", {"function_name": "Dangling"})],
        connections=[],
    )
    [info] = list_graph_functions(session)
    assert info.has_end is False


def test_two_start_functions_sharing_name_are_distinct_entries():
    session = _session_with_flow(
        nodes=[
            _node("s1", "StartFunction"),
            _node("e1", "EndFunction"),
            _node("s2", "StartFunction"),
            _node("e2", "EndFunction"),
        ],
        connections=[_conn("s1", "e1"), _conn("s2", "e2")],
    )
    result = list_graph_functions(session)
    ids = {info.start_node_id for info in result}
    assert ids == {"s1", "s2"}
    assert all(info.name == "MyFunction" for info in result)


def test_zero_functions_returns_empty():
    session = _session_with_flow(nodes=[_node("n1", "Delay")], connections=[])
    assert list_graph_functions(session) == []


def test_find_run_param_revolution():
    session = _session_with_flow(
        nodes=[
            _node("s1", "StartFunction"),
            _node("o1", "Output"),
            _node("w1", "WaitForInput", {"threshold_mode": "revolution", "turns_target": 3}),
            _node("e1", "EndFunction"),
        ],
        connections=[_conn("s1", "o1"), _conn("o1", "w1"), _conn("w1", "e1")],
    )
    rp = find_run_param("s1", session.flow)
    assert rp is not None
    assert (rp.node_id, rp.state_key, rp.value, rp.label) == (
        "w1",
        "turns_target",
        3,
        "Revolutions",
    )


def test_find_run_param_counts():
    session = _session_with_flow(
        nodes=[
            _node("s1", "StartFunction"),
            _node("w1", "WaitForInput", {"threshold_mode": "counts", "counts_target": 400}),
            _node("e1", "EndFunction"),
        ],
        connections=[_conn("s1", "w1"), _conn("w1", "e1")],
    )
    rp = find_run_param("s1", session.flow)
    assert (rp.node_id, rp.state_key, rp.value, rp.label) == ("w1", "counts_target", 400, "Counts")


def test_find_run_param_none_when_not_parameterizable():
    session = _session_with_flow(
        nodes=[
            _node("s1", "StartFunction"),
            _node("w1", "WaitForInput", {"threshold_mode": "digital"}),
            _node("e1", "EndFunction"),
        ],
        connections=[_conn("s1", "w1"), _conn("w1", "e1")],
    )
    assert find_run_param("s1", session.flow) is None


def test_build_picker_labels_uses_plain_name_when_unique():
    infos = [
        GraphFunctionInfo(start_node_id="s1", name="Purge", has_end=True),
        GraphFunctionInfo(start_node_id="s2", name="Fill", has_end=True),
    ]
    assert build_picker_labels(infos) == [("Purge", "s1"), ("Fill", "s2")]


def test_build_picker_labels_disambiguates_duplicate_names_by_id_suffix():
    infos = [
        GraphFunctionInfo(start_node_id="node-aaaa", name="Run", has_end=True),
        GraphFunctionInfo(start_node_id="node-bbbb", name="Run", has_end=True),
    ]
    labels = build_picker_labels(infos)
    # Disambiguated by the last 4 chars of the id; still binds by full id.
    assert labels == [("Run aaaa", "node-aaaa"), ("Run bbbb", "node-bbbb")]
