"""
Flow Function Nodes - Nodes for defining reusable functions in the graph.

StartFunction and EndFunction nodes allow users to define functions
directly in the node graph. Once connected, they become callable
nodes that can be reused throughout the flow.
"""

import asyncio
import contextlib
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
        self._completion_event: asyncio.Event | None = None
        self._end_node_ids: list[str] = []
        # Serializes execution of THIS function across every caller (an in-graph
        # FunctionCall and a Runner button tap share one runner, hence one lock),
        # so two invocations can never clobber each other's completion signal.
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        """True while an invocation of this function is in flight."""
        return self._lock.locked()

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

            # Follow execution flow only — a data wire does not carry execution,
            # so an End reached solely by data is not a real completion point.
            # Exclude explicit data connections; untyped/exec ones are followed.
            for conn in self._flow_engine._connections:
                if conn["from_node"] == current_id and conn.get("type") != "data":
                    to_visit.append(conn["to_node"])

    def _on_function_complete(self) -> None:
        """Called when EndFunction is reached."""
        logger.info(f"FlowFunctionRunner: function complete (start={self._start_node_id})")
        if self._completion_event:
            self._completion_event.set()

    def _register_completion(self, callback) -> None:
        for end_node_id in self._end_node_ids:
            end_node = self._flow_engine.get_node(end_node_id)
            if end_node and hasattr(end_node, "set_completion_callback"):
                end_node.set_completion_callback(callback)

    async def execute(self, timeout: float = 60.0, on_timeout=None) -> bool:
        """Run the function until an EndFunction node is reached.

        Only one invocation runs at a time (per-function lock); a second caller
        waits its turn rather than clobbering the first's completion signal.

        Returns ``True`` once the function has actually ended. If it exceeds
        ``timeout`` it is reported unresponsive (``on_timeout`` fired), the run is
        **cancelled**, and ``False`` is returned — a hung chain (a node that
        raised and stopped propagation, or hardware that went away mid-run) can
        never leave the caller awaiting forever. Also returns ``False`` when the
        start node is missing.
        """
        async with self._lock:
            if not self._end_node_ids:
                self._find_end_nodes()
            self._completion_event = asyncio.Event()

            start_node = self._flow_engine.get_node(self._start_node_id)
            if start_node is None:
                logger.error(f"StartFunction node not found: {self._start_node_id}")
                return False

            self._register_completion(self._on_function_complete)
            run = asyncio.ensure_future(self._run_body(start_node))
            # Register so a graph reset (New/Open -> FlowEngine.clear()) can
            # cancel this run instead of leaving it driving a torn-down graph.
            self._flow_engine.track_task(run)
            try:
                logger.info(f"FlowFunctionRunner: executing StartFunction {self._start_node_id}")
                # The timeout wraps the WHOLE run (the body awaits the chain, so
                # a slow body must count against the timeout too).
                done, _ = await asyncio.wait({run}, timeout=timeout)
                if not done:
                    logger.warning(
                        "FlowFunctionRunner: '%s' unresponsive after %ss — cancelling",
                        self._start_node_id,
                        timeout,
                    )
                    if on_timeout is not None:
                        on_timeout()
                    return False  # cancelled in finally; caller regains control
                if run.cancelled():
                    # A graph reset cancelled the run out from under us.
                    logger.info("FlowFunctionRunner: '%s' run cancelled", self._start_node_id)
                    return False
                run.result()  # surface a body exception rather than swallow it
                logger.info("FlowFunctionRunner: function execution complete")
                return True
            finally:
                if not run.done():
                    run.cancel()  # timeout / cancelled caller (New/Open) stops the body
                    with contextlib.suppress(asyncio.CancelledError):
                        await run  # let cancellation unwind before releasing the lock
                self._register_completion(None)

    async def _run_body(self, start_node) -> None:
        if hasattr(start_node, "execute"):
            if asyncio.iscoroutinefunction(start_node.execute):
                await start_node.execute()
            else:
                start_node.execute()
        # The completion callback is the definitive "EndFunction reached" signal.
        if self._completion_event is not None:
            await self._completion_event.wait()


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
