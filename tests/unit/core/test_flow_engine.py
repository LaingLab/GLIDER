import pytest

from glider.core.flow_engine import FlowEngine, FlowState
from glider.nodes.base_node import (
    GliderNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)


class _NoOpNode(GliderNode):
    """Minimal node used for FlowEngine connection tests."""

    definition = NodeDefinition(
        name="noop",
        category=NodeCategory.LOGIC,
        inputs=[PortDefinition(name="in", port_type=PortType.EXEC)],
        outputs=[PortDefinition(name="out", port_type=PortType.EXEC)],
    )

    def update_event(self) -> None:  # pragma: no cover - trivial
        pass


def test_flow_engine_init():
    """Test that FlowEngine initializes correctly."""
    engine = FlowEngine()
    assert engine.state == FlowState.STOPPED
    assert not engine.is_running
    assert len(engine.nodes) == 0


def test_node_registration():
    """Test that nodes can be registered."""

    class MockNode:
        pass

    FlowEngine.register_node("MockNode", MockNode)
    assert "MockNode" in FlowEngine.get_available_nodes()
    assert FlowEngine.get_node_class("MockNode") == MockNode


@pytest.mark.asyncio
async def test_flow_engine_start_stop():
    """Test starting and stopping the engine."""
    engine = FlowEngine()
    await engine.start()
    assert engine.state == FlowState.RUNNING
    assert engine.is_running

    await engine.stop()
    assert engine.state == FlowState.STOPPED
    assert not engine.is_running


def _build_connected_pair() -> tuple[FlowEngine, GliderNode, GliderNode, str, int]:
    """Build a two-node engine with one exec connection.

    Returns the engine, source node, dest node, connection id, and the count
    of callbacks the source node had *before* the connection was created
    (``create_node`` installs its own bookkeeping callback unrelated to
    connections, which we have to factor out).
    """
    engine = FlowEngine()
    engine.register_node("noop", _NoOpNode)
    engine.create_node("a", "noop")
    engine.create_node("b", "noop")
    src = engine._nodes["a"]
    baseline = len(src._update_callbacks)
    conn_id = "conn-a-b"
    engine.create_connection(conn_id, "a", 0, "b", 0, connection_type="exec")
    return engine, src, engine._nodes["b"], conn_id, baseline


def test_remove_connection_uninstalls_exec_callback():
    """Removing an exec connection must drop the callback it installed.

    Regression test for a leak where ``remove_connection`` forgot to remove
    the callback it added to the source node's ``_update_callbacks`` list,
    so after many add/remove cycles the source would fire stale callbacks.
    """
    engine, src, _dst, conn_id, baseline = _build_connected_pair()
    assert len(src._update_callbacks) == baseline + 1

    engine.remove_connection(conn_id)
    assert len(src._update_callbacks) == baseline
    assert conn_id not in engine._connection_callbacks


def test_remove_node_cascades_exec_callbacks():
    """Removing the source node should also uninstall its exec callbacks."""
    engine, src, _dst, conn_id, baseline = _build_connected_pair()
    assert len(src._update_callbacks) == baseline + 1

    engine.remove_node("a")
    # Connection record must be gone so a later remove_connection is a no-op.
    assert conn_id not in engine._connection_callbacks


def test_clear_uninstalls_all_exec_callbacks():
    """Wholesale clear() must not leak callbacks on retained node references."""
    engine, src, _dst, _conn_id, baseline = _build_connected_pair()
    assert len(src._update_callbacks) == baseline + 1

    engine.clear()
    # Only the connection-installed callback must be removed; the baseline
    # callback belongs to the node and is out of scope for clear().
    assert len(src._update_callbacks) == baseline
    assert engine._connection_callbacks == {}
