"""Discover graph functions (StartFunction -> EndFunction chains) from a session.

Extracted from node_library_panel so non-GUI code (the Runner manual-control
page) can list functions without a graph view, working off the serialized
session model rather than a live flow engine.
"""

from __future__ import annotations

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
class RevolutionParam:
    """A revolution-mode WaitForInput found in a function chain.

    node_id is the WaitForInput node whose ``turns_target`` a touchscreen prompt
    can set; turns is its current target (used as the prompt's default).
    """

    node_id: str
    turns: int


def find_revolution_node(start_id: str, flow) -> RevolutionParam | None:
    """Trace exec connections from start_id for a revolution-mode WaitForInput.

    Returns the first such node (so a 'Run N revolutions' prompt can set its
    ``turns_target``), or None if the function has none.
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
        if (
            node is not None
            and node.node_type == "WaitForInput"
            and (node.state or {}).get("threshold_mode") == "revolution"
        ):
            turns = int((node.state or {}).get("turns_target", 1) or 1)
            return RevolutionParam(node_id=current, turns=turns)
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
