"""Discover graph functions (StartFunction -> EndFunction chains) from a session.

Extracted from node_library_panel so non-GUI code (the Runner manual-control
page) can list functions without a graph view, working off the serialized
session model rather than a live flow engine.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

_DEFAULT_FUNCTION_NAME = "MyFunction"


@dataclass(frozen=True)
class GraphFunctionInfo:
    """A graph function discovered in the session flow.

    start_node_id is the unique, stable binding key. name is display-only and
    may be duplicated across functions.
    """

    start_node_id: str
    name: str
    has_end: bool


@dataclass(frozen=True)
class RunParam:
    """A parameterizable WaitForInput found in a function chain.

    A touchscreen prompt sets ``state_key`` on node ``node_id`` before running;
    ``value`` is the current target (the prompt default) and ``label`` is what
    to ask for (e.g. "Revolutions" or "Counts").
    """

    node_id: str
    state_key: str
    value: int
    label: str


# Modes whose target a touchscreen prompt can set: mode -> (state_key, label).
_PARAM_MODES = {
    "revolution": ("turns_target", 1, "Revolutions"),
    "counts": ("counts_target", 400, "Counts"),
}


def find_run_param(start_id: str, flow) -> RunParam | None:
    """Trace exec connections from start_id for a parameterizable WaitForInput.

    Returns the first revolution- or counts-mode WaitForInput so a touchscreen
    prompt can set its target before running, or None if the function has none.
    """
    by_id = {node.id: node for node in flow.nodes}
    visited: set[str] = set()
    to_visit = [start_id]
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        node = by_id.get(current)
        if node is not None and node.node_type == "WaitForInput":
            mode = (node.state or {}).get("threshold_mode")
            if mode in _PARAM_MODES:
                state_key, default, label = _PARAM_MODES[mode]
                value = int((node.state or {}).get(state_key, default) or default)
                return RunParam(node_id=current, state_key=state_key, value=value, label=label)
        for conn in flow.connections:
            if conn.from_node == current:
                to_visit.append(conn.to_node)
    return None


def list_graph_functions(session) -> list[GraphFunctionInfo]:
    """Return one entry per StartFunction node in the session's flow."""
    if session is None:
        return []

    flow = session.flow
    result: list[GraphFunctionInfo] = []
    for node in flow.nodes:
        if node.node_type != "StartFunction":
            continue
        name = _DEFAULT_FUNCTION_NAME
        if node.state:
            name = node.state.get("function_name", _DEFAULT_FUNCTION_NAME)
        result.append(
            GraphFunctionInfo(
                start_node_id=node.id,
                name=name,
                has_end=_reaches_end_function(node.id, flow),
            )
        )
    return result


def build_picker_labels(infos: list[GraphFunctionInfo]) -> list[tuple[str, str]]:
    """Build ``(display_label, start_node_id)`` pairs, disambiguating dup names.

    Functions sharing a display name get a suffix of the last 4 characters of
    their StartFunction id so an operator can tell them apart; the button still
    binds by the stable ``start_node_id``, never the label.
    """
    name_counts = Counter(info.name for info in infos)
    labels: list[tuple[str, str]] = []
    for info in infos:
        if name_counts[info.name] > 1:
            display = f"{info.name} {info.start_node_id[-4:]}"
        else:
            display = info.name
        labels.append((display, info.start_node_id))
    return labels


def _reaches_end_function(start_id: str, flow) -> bool:
    """Trace exec connections from start_id to see if an EndFunction is reachable."""
    visited: set[str] = set()
    to_visit = [start_id]
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
        visited.add(current)
        for node in flow.nodes:
            if node.id == current and node.node_type == "EndFunction":
                return True
        for conn in flow.connections:
            if conn.from_node == current:
                to_visit.append(conn.to_node)
    return False
