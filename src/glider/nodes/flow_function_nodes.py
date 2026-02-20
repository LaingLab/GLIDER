"""
Flow Function Nodes - Nodes for defining reusable functions in the graph.

StartFunction and EndFunction nodes allow users to define functions
directly in the node graph. Once connected, they become callable
nodes that can be reused throughout the flow.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from glider.nodes.base_node import (
    GliderNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

if TYPE_CHECKING:
    from glider.core.flow_engine import FlowEngine

logger = logging.getLogger(__name__)


class FlowFunctionRunner:
    """
    Executes a flow function's sub-graph.

    When a FunctionCall node invokes this runner, it triggers the
    StartFunction node and waits until the EndFunction node is reached.
    """

    def __init__(self, start_node_id: str, flow_engine: "FlowEngine"):
        """
        Initialize the function runner.

        Args:
            start_node_id: ID of the StartFunction node
            flow_engine: FlowEngine instance to execute nodes
        """
        self._start_node_id = start_node_id
        self._flow_engine = flow_engine
        self._completion_event = None
        self._end_node_ids: list[str] = []

    def _find_end_nodes(self) -> None:
        """Find all EndFunction nodes connected to this function's StartFunction."""
        visited = set()
        to_visit = [self._start_node_id]

        while to_visit:
            current_id = to_visit.pop()
            if current_id in visited:
                continue
            visited.add(current_id)

            node = self._flow_engine.get_node(current_id)
            if node is not None:
                node_name = (
                    getattr(getattr(node, "definition", None), "name", None) or type(node).__name__
                )
                if node_name in ("EndFunction", "EndFunctionNode"):
                    self._end_node_ids.append(current_id)

            # Follow connections
            for conn in self._flow_engine._connections:
                if conn["from_node"] == current_id:
                    to_visit.append(conn["to_node"])

    def _on_function_complete(self) -> None:
        """Called when EndFunction is reached."""
        logger.info(f"FlowFunctionRunner: function complete (start={self._start_node_id})")
        if self._completion_event:
            self._completion_event.set()

    async def execute(self) -> None:
        """
        Execute the function by triggering the StartFunction node.

        Waits until an EndFunction node is reached.
        """
        # Find EndFunction nodes if not already found
        if not self._end_node_ids:
            self._find_end_nodes()

        # Create completion event
        self._completion_event = asyncio.Event()

        # Register completion callback on EndFunction nodes
        for end_node_id in self._end_node_ids:
            end_node = self._flow_engine.get_node(end_node_id)
            if end_node and hasattr(end_node, "set_completion_callback"):
                end_node.set_completion_callback(self._on_function_complete)

        # Trigger the StartFunction node
        start_node = self._flow_engine.get_node(self._start_node_id)
        if start_node is None:
            logger.error(f"StartFunction node not found: {self._start_node_id}")
            return

        logger.info(f"FlowFunctionRunner: executing StartFunction {self._start_node_id}")
        if hasattr(start_node, "execute"):
            if asyncio.iscoroutinefunction(start_node.execute):
                await start_node.execute()
            else:
                start_node.execute()

        # Wait for function completion with timeout
        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=60.0)
            logger.info("FlowFunctionRunner: function execution complete")
        except asyncio.TimeoutError:
            logger.warning("FlowFunctionRunner: function timed out")
        finally:
            # Clear completion callbacks
            for end_node_id in self._end_node_ids:
                end_node = self._flow_engine.get_node(end_node_id)
                if end_node and hasattr(end_node, "set_completion_callback"):
                    end_node.set_completion_callback(None)


class StartFunctionNode(GliderNode):
    """
    Entry point for a user-defined function.

    Set the function name in the properties panel. Connect this to
    other nodes and end with an EndFunction node to create a
    reusable function.
    """

    definition = NodeDefinition(
        name="StartFunction",
        category=NodeCategory.LOGIC,
        description="Entry point for a user-defined function",
        inputs=[],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers the function body"),
        ],
    )

    def __init__(self):
        super().__init__()
        self._function_name = "MyFunction"

    def update_event(self) -> None:
        pass

    def get_function_name(self) -> str:
        """Get the function name from state."""
        return self._state.get("function_name", "MyFunction")

    async def start(self) -> None:
        """Called when this function is invoked."""
        logger.info(f"StartFunction '{self.get_function_name()}' triggered")
        await self._fire_exec_output("next")

    async def execute(self) -> None:
        """Execute the function start."""
        logger.info(f"StartFunction '{self.get_function_name()}' executing")
        await self._fire_exec_output("next")

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        for callback in self._update_callbacks:
            callback("next", True)


class EndFunctionNode(GliderNode):
    """
    Exit point for a user-defined function.

    Connect this to the end of your function flow. When reached,
    the function completes and returns control to the caller.
    """

    definition = NodeDefinition(
        name="EndFunction",
        category=NodeCategory.LOGIC,
        description="Exit point for a user-defined function",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
        ],
        outputs=[],
    )

    def __init__(self):
        super().__init__()
        self._completion_callback = None

    def set_completion_callback(self, callback):
        """Set callback to invoke when function completes."""
        self._completion_callback = callback

    def update_event(self) -> None:
        pass

    async def execute(self) -> None:
        """Called when the function completes."""
        logger.info("EndFunction reached - function complete")
        if self._completion_callback:
            self._completion_callback()


class FunctionCallNode(GliderNode):
    """
    Calls a user-defined function.

    This node is dynamically created when a function (StartFunction -> EndFunction)
    is detected in the graph. When executed, it runs the function's internal nodes.
    """

    definition = NodeDefinition(
        name="FunctionCall",
        category=NodeCategory.LOGIC,
        description="Call a user-defined function",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
        ],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers after function completes"),
        ],
    )

    def __init__(self):
        super().__init__()
        self._function_id = None
        self._function_runner = None

    def set_function_context(self, function_id: str, runner):
        """Set the function ID and runner."""
        self._function_id = function_id
        self._function_runner = runner

    def update_event(self) -> None:
        pass

    async def execute(self) -> None:
        """Execute the function."""
        function_name = self._state.get("function_name", "Unknown")
        logger.info(f"FunctionCall: invoking '{function_name}'")

        if self._function_runner is not None:
            try:
                await self._function_runner.execute()
                logger.info(f"FunctionCall: '{function_name}' complete")
            except Exception as e:
                logger.error(f"FunctionCall error: {e}")
                self._error = str(e)
        else:
            logger.warning(f"FunctionCall: no runner for function '{function_name}'")

        await self._fire_exec_output("next")

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        for callback in self._update_callbacks:
            callback("next", True)


def register_flow_function_nodes(flow_engine) -> None:
    """Register flow function nodes with the flow engine."""
    flow_engine.register_node("StartFunction", StartFunctionNode)
    flow_engine.register_node("EndFunction", EndFunctionNode)
    flow_engine.register_node("FunctionCall", FunctionCallNode)
    logger.info("Registered flow function nodes")
