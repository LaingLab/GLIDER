"""
Flow Engine - Manages the logic graph execution.

Wraps the ryvencore session and handles the execution of the experiment
flow. Supports both Data Flow (reactive) and Execution Flow (imperative).
"""

import asyncio
import logging
from collections.abc import Callable
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FlowState(Enum):
    """State of the flow engine."""

    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()
    ERROR = auto()


class FlowEngine:
    """
    Manages the execution of the experiment flow graph.

    Uses ryvencore for graph state management. Supports:
    - Data Flow: Reactive propagation when values change
    - Execution Flow: Imperative sequence of actions
    """

    # Registry of available node types (intentionally class-level: shared across all
    # FlowEngine instances so that node registration is global and persistent)
    _node_registry: dict[str, type] = {}

    def __init__(self, hardware_manager=None):
        """
        Initialize the flow engine.

        Args:
            hardware_manager: HardwareManager instance for device access
        """
        self._hardware_manager = hardware_manager
        self._session = None  # ryvencore Session
        self._flow = None  # Current flow
        self._state = FlowState.STOPPED
        self._nodes: dict[str, Any] = {}  # node_id -> node instance
        self._connections: list[dict[str, Any]] = []  # Connection list for standalone mode
        self._running_tasks: set[asyncio.Task] = set()

        # Callbacks
        self._state_callbacks: list[Callable[[FlowState], None]] = []
        self._node_update_callbacks: list[Callable[[str, str, Any], None]] = []
        self._error_callbacks: list[Callable[[str, Exception], None]] = []
        self._complete_callbacks: list[Callable[[], None]] = []

        # Try to import ryvencore
        self._ryvencore_available = False
        try:
            import ryvencore

            self._ryvencore_available = True
            self._ryvencore = ryvencore
        except ImportError:
            logger.warning("ryvencore not available - flow engine will operate in limited mode")

    @classmethod
    def register_node(cls, node_type: str, node_class: type) -> None:
        """Register a node type."""
        cls._node_registry[node_type] = node_class
        logger.debug(f"Registered node type: {node_type}")

    @classmethod
    def get_available_nodes(cls) -> list[str]:
        """Get list of available node type names."""
        return list(cls._node_registry.keys())

    @classmethod
    def get_node_class(cls, node_type: str) -> type | None:
        """Get a node class by type name."""
        return cls._node_registry.get(node_type)

    @property
    def state(self) -> FlowState:
        """Current flow state."""
        return self._state

    @state.setter
    def state(self, value: FlowState) -> None:
        if value != self._state:
            old_state = self._state
            self._state = value
            logger.debug(f"Flow state changed: {old_state} -> {value}")
            for callback in self._state_callbacks:
                try:
                    callback(value)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    @property
    def is_running(self) -> bool:
        """Whether the flow is currently running."""
        return self._state == FlowState.RUNNING

    @property
    def nodes(self) -> dict[str, Any]:
        """Dictionary of node instances."""
        return self._nodes.copy()

    def on_state_change(self, callback: Callable[[FlowState], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    def on_node_update(self, callback: Callable[[str, str, Any], None]) -> None:
        """Register callback for node output updates (node_id, output_name, value)."""
        self._node_update_callbacks.append(callback)

    def on_error(self, callback: Callable[[str, Exception], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def on_flow_complete(self, callback: Callable[[], None]) -> None:
        """Register callback for flow completion (EndExperiment reached)."""
        self._complete_callbacks.append(callback)

    def _notify_node_update(self, node_id: str, output_name: str, value: Any) -> None:
        """Notify node update callbacks."""
        for callback in self._node_update_callbacks:
            try:
                callback(node_id, output_name, value)
            except Exception as e:
                logger.error(f"Node update callback error: {e}")

    def _notify_error(self, source: str, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(source, error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _notify_complete(self) -> None:
        """Notify flow completion callbacks."""
        logger.info("Flow completed - EndExperiment reached")
        for callback in self._complete_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Complete callback failed: {e}")

    def initialize(self) -> None:
        """Initialize the flow engine with ryvencore."""
        if self._ryvencore_available:
            self._session = self._ryvencore.Session()
            # Register our custom nodes with ryvencore
            self._register_nodes_with_ryvencore()
            # Create a new flow
            self._flow = self._session.create_flow("Main Flow")
            logger.info("Flow engine initialized with ryvencore")
        else:
            logger.info("Flow engine initialized in standalone mode")

    def _register_nodes_with_ryvencore(self) -> None:
        """Register node types with ryvencore session."""
        if not self._session:
            return

        for node_type, node_class in self._node_registry.items():
            # Only register classes that are actual ryvencore Node subclasses
            # GliderNode classes work in standalone mode and don't need ryvencore registration
            if not hasattr(node_class, "_build_identifier"):
                logger.debug(
                    f"Skipping ryvencore registration for {node_type} (standalone GliderNode)"
                )
                continue
            try:
                self._session.register_node_type(node_class)
            except Exception as e:
                logger.warning(f"Failed to register node type {node_type}: {e}")

    def _bind_custom_device_runner(self, node, definition_id: str, session) -> None:
        """
        Create and bind a CustomDeviceRunner to a CustomDevice node.

        Args:
            node: The CustomDeviceNode instance
            definition_id: ID of the custom device definition
            session: ExperimentSession containing the definition
        """
        from glider.core.custom_device import CustomDeviceDefinition, CustomDeviceRunner

        # Get the definition from the session
        def_dict = session.get_custom_device_definition(definition_id)
        if not def_dict:
            logger.warning(f"Custom device definition not found: {definition_id}")
            return

        # Create the definition object
        definition = CustomDeviceDefinition.from_dict(def_dict)

        # Get the first available board from hardware manager
        board = None
        if self._hardware_manager:
            boards = self._hardware_manager.boards
            if boards:
                # Get the first connected board
                for board_id, b in boards.items():
                    if b.is_connected:
                        board = b
                        logger.info(f"Using board '{board_id}' for custom device")
                        break
                if board is None and boards:
                    # No connected board, use first one anyway
                    board = next(iter(boards.values()))

        if board is None:
            logger.warning("No board available for custom device - using mock mode")
            # Create a mock board for testing without hardware
            from glider.hal.mock_board import MockBoard

            board = MockBoard()

        # Create the runner
        runner = CustomDeviceRunner(definition, board)

        # Store runner for later initialization
        if not hasattr(self, "_custom_device_runners"):
            self._custom_device_runners = {}
        self._custom_device_runners[definition_id] = runner

        # Bind to the node
        if hasattr(node, "set_custom_device_context"):
            node.set_custom_device_context(runner, definition_id)
            logger.info(f"Bound CustomDeviceRunner for '{definition.name}' to node")
        else:
            logger.warning("Node does not support set_custom_device_context")

    def _bind_function_runner(self, node, start_node_id: str) -> None:
        """
        Create and bind a FlowFunctionRunner to a FunctionCall node.

        Args:
            node: The FunctionCallNode instance
            start_node_id: ID of the StartFunction node to invoke
        """
        from glider.nodes.flow_function_nodes import FlowFunctionRunner

        # Create the runner
        runner = FlowFunctionRunner(start_node_id, self)

        # Bind to the node
        if hasattr(node, "set_function_context"):
            node.set_function_context(start_node_id, runner)
            logger.info(f"Bound FlowFunctionRunner for StartFunction '{start_node_id}' to node")
        else:
            logger.warning("Node does not support set_function_context")

    def create_node(
        self,
        node_id: str,
        node_type: str,
        position: tuple = (0, 0),
        state: dict[str, Any] | None = None,
        device_id: str | None = None,
        session=None,
    ) -> Any:
        """
        Create a node instance.

        Args:
            node_id: Unique node ID
            node_type: Type of node to create
            position: (x, y) position in the graph
            state: Initial state data
            device_id: Associated device ID (for hardware nodes)
            session: ExperimentSession for accessing custom device definitions

        Returns:
            Created node instance
        """
        node_class = self._node_registry.get(node_type)
        if node_class is None:
            raise ValueError(f"Unknown node type: {node_type}")

        # Create node instance
        if self._ryvencore_available and self._flow:
            # Create through ryvencore
            node = self._flow.create_node(node_class)
            if node is None:
                # Node not registered with ryvencore - fall back to standalone
                logger.warning(
                    f"Node class {node_class} not found in ryvencore session, using standalone mode"
                )
                node = node_class()
            node._glider_id = node_id
        else:
            # Create standalone node
            node = node_class()
            node._glider_id = node_id

        # Apply initial state
        if state:
            if hasattr(node, "set_state"):
                node.set_state(state)
            elif hasattr(node, "_state"):
                node._state = state
            logger.info(f"Applied state to node {node_id}: {state}")

        # Handle CustomDevice nodes - create and bind the runner
        logger.info(f"create_node: type={node_type}, state={state}, session={session is not None}")
        if node_type in ("CustomDevice", "CustomDeviceAction"):
            logger.info("CustomDevice node detected, binding runner...")
            if state and session:
                definition_id = state.get("definition_id")
                if definition_id:
                    self._bind_custom_device_runner(node, definition_id, session)
                else:
                    logger.warning("CustomDevice node has no definition_id in state")
            else:
                logger.warning(
                    f"CustomDevice node missing state ({state}) or session ({session is not None})"
                )

        # Handle FunctionCall nodes - bind the runner
        if node_type == "FunctionCall":
            logger.info("FunctionCall node detected, binding runner...")
            if state:
                # Check both key names for compatibility
                start_node_id = state.get("function_start_id") or state.get("start_node_id")
                if start_node_id:
                    self._bind_function_runner(node, start_node_id)
                else:
                    logger.warning("FunctionCall node has no function_start_id in state")

        # Bind to device if specified
        if device_id and self._hardware_manager:
            device = self._hardware_manager.get_device(device_id)
            if device and hasattr(node, "bind_device"):
                node.bind_device(device)
                logger.info(f"Bound device '{device_id}' to node {node_id}")
            else:
                logger.warning(
                    f"Could not bind device '{device_id}' to node {node_id} (device={device})"
                )

        # Register update callback
        if hasattr(node, "on_output_update"):
            node.on_output_update(
                lambda output, value, n=node: self._notify_node_update(n._glider_id, output, value)
            )

        self._nodes[node_id] = node
        logger.debug(f"Created node: {node_type} (ID: {node_id})")
        return node

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the flow."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return

        if self._ryvencore_available and self._flow:
            try:
                self._flow.remove_node(node)
            except Exception as e:
                logger.warning(f"Error removing node from ryvencore: {e}")

        logger.debug(f"Removed node: {node_id}")

    def create_connection(
        self,
        connection_id: str,
        from_node_id: str,
        from_output: int,
        to_node_id: str,
        to_input: int,
        connection_type: str = "data",
    ) -> None:
        """
        Create a connection between nodes.

        Args:
            connection_id: Unique connection ID
            from_node_id: Source node ID
            from_output: Source output index
            to_node_id: Target node ID
            to_input: Target input index
            connection_type: "data" or "exec"
        """
        from_node = self._nodes.get(from_node_id)
        to_node = self._nodes.get(to_node_id)

        if from_node is None:
            raise ValueError(f"Source node not found: {from_node_id}")
        if to_node is None:
            raise ValueError(f"Target node not found: {to_node_id}")

        # Store connection for standalone execution
        self._connections.append(
            {
                "id": connection_id,
                "from_node": from_node_id,
                "from_output": from_output,
                "to_node": to_node_id,
                "to_input": to_input,
                "type": connection_type,
            }
        )

        # Wire up the execution flow callback on the source node
        # Map output index to expected output name for filtering
        from_node_obj = self._nodes.get(from_node_id)
        expected_output_name = None
        if hasattr(from_node_obj, "definition") and hasattr(from_node_obj.definition, "outputs"):
            outputs = from_node_obj.definition.outputs
            if from_output < len(outputs):
                expected_output_name = outputs[from_output].name
                logger.debug(
                    f"Connection {from_node_id}:{from_output} expects output name: '{expected_output_name}'"
                )

        def on_exec_output(
            output_name,
            value,
            fn=from_node_id,
            fo=from_output,
            tn=to_node_id,
            expected=expected_output_name,
        ):
            # Only propagate if this is the correct output for this connection
            if expected is not None and output_name != expected:
                logger.debug(f"Skipping callback - output '{output_name}' != expected '{expected}'")
                return None
            logger.debug(f"Exec callback fired: {fn}:{fo} -> {tn}, flow state: {self._state}")
            if self._state == FlowState.RUNNING:
                logger.info(f"Propagating execution: {fn} -> {tn}")
                # Return the task so callers can await it if needed
                return asyncio.create_task(self._propagate_execution(fn, fo, tn))
            else:
                logger.warning(f"Skipping propagation - flow not running (state: {self._state})")
                return None

        if hasattr(from_node, "_update_callbacks"):
            from_node._update_callbacks.append(on_exec_output)
            logger.debug(
                f"Registered exec callback on {from_node_id}, total callbacks: {len(from_node._update_callbacks)}"
            )

        if self._ryvencore_available and self._flow:
            try:
                # Get output and input ports
                output_port = from_node.outputs[from_output]
                input_port = to_node.inputs[to_input]
                self._flow.connect(output_port, input_port)
            except Exception as e:
                logger.warning(f"Error creating ryvencore connection: {e}")

        logger.debug(f"Created connection: {from_node_id}:{from_output} -> {to_node_id}:{to_input}")

    async def _propagate_execution(
        self, from_node_id: str, from_output: int, to_node_id: str
    ) -> None:
        """Propagate execution to the target node."""
        logger.info(f"_propagate_execution called: {from_node_id}:{from_output} -> {to_node_id}")
        to_node = self._nodes.get(to_node_id)
        if to_node is None:
            logger.error(f"Target node not found: {to_node_id}")
            return

        try:
            if hasattr(to_node, "execute") and callable(to_node.execute):
                logger.info(f"Executing node: {to_node_id} (type: {type(to_node).__name__})")
                try:
                    if asyncio.iscoroutinefunction(to_node.execute):
                        await to_node.execute()
                    else:
                        to_node.execute()
                    logger.info(f"Node {to_node_id} execution complete")
                except Exception as e:
                    # User-facing error handling for node execution failures
                    error_msg = (
                        f"Error executing node '{to_node_id}' ({type(to_node).__name__}): {str(e)}"
                    )
                    logger.error(error_msg, exc_info=True)
                    self._notify_error(to_node_id, e)
                    # Don't propagate execution further on error
                    return

                # Check if this was EndExperiment - signal flow completion
                if hasattr(to_node, "definition") and to_node.definition.name == "EndExperiment":
                    self._notify_complete()
            else:
                logger.warning(f"Node {to_node_id} has no execute method")
        except Exception as e:
            # Catch-all for unexpected framework errors
            logger.critical(
                f"Critical framework error during execution propagation: {e}", exc_info=True
            )
            self._notify_error("Framework", e)

    def remove_connection(self, connection_id: str) -> None:
        """Remove a connection."""
        # Connection removal is handled by ryvencore when nodes are connected
        logger.debug(f"Removed connection: {connection_id}")

    def get_node(self, node_id: str) -> Any | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_node_output(self, node_id: str, output_index: int) -> Any:
        """Get the current value of a node output."""
        node = self._nodes.get(node_id)
        if node is None:
            return None

        if hasattr(node, "outputs") and output_index < len(node.outputs):
            return node.outputs[output_index].val
        elif hasattr(node, "get_output"):
            return node.get_output(output_index)

        return None

    def set_node_input(self, node_id: str, input_index: int, value: Any) -> None:
        """Set the value of a node input."""
        node = self._nodes.get(node_id)
        if node is None:
            return

        if hasattr(node, "inputs") and input_index < len(node.inputs):
            node.inputs[input_index].update(value)
        elif hasattr(node, "set_input"):
            node.set_input(input_index, value)

    async def start(self) -> None:
        """Start flow execution."""
        if self._state == FlowState.RUNNING:
            return

        logger.info("Starting flow execution")

        # Initialize custom device runners
        if hasattr(self, "_custom_device_runners"):
            for def_id, runner in self._custom_device_runners.items():
                try:
                    if not runner.is_initialized:
                        logger.info(f"Initializing custom device runner for definition {def_id}")
                        await runner.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize custom device runner: {e}")

        self.state = FlowState.RUNNING

        # Start any continuous nodes (timers, sensors, etc.)
        for node_id, node in self._nodes.items():
            if hasattr(node, "start") and callable(node.start):
                try:
                    task = asyncio.create_task(self._run_node_start(node_id, node))
                    self._running_tasks.add(task)
                    task.add_done_callback(self._running_tasks.discard)
                except Exception as e:
                    logger.error(f"Error starting node {node_id}: {e}")
                    self._notify_error(node_id, e)

    async def _run_node_start(self, node_id: str, node: Any) -> None:
        """Run a node's start method."""
        try:
            if asyncio.iscoroutinefunction(node.start):
                await node.start()
            else:
                node.start()
        except Exception as e:
            logger.error(f"Error in node {node_id} start: {e}")
            self._notify_error(node_id, e)

    async def stop(self) -> None:
        """Stop flow execution."""
        if self._state == FlowState.STOPPED:
            return

        logger.info("Stopping flow execution")

        # Cancel all running tasks
        for task in self._running_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        self._running_tasks.clear()

        # Stop all nodes
        for node_id, node in self._nodes.items():
            if hasattr(node, "stop") and callable(node.stop):
                try:
                    if asyncio.iscoroutinefunction(node.stop):
                        await node.stop()
                    else:
                        node.stop()
                except Exception as e:
                    logger.error(f"Error stopping node {node_id}: {e}")

        self.state = FlowState.STOPPED

    async def pause(self) -> None:
        """Pause flow execution."""
        if self._state != FlowState.RUNNING:
            return

        logger.info("Pausing flow execution")
        self.state = FlowState.PAUSED

        # Pause all nodes
        for node_id, node in self._nodes.items():
            if hasattr(node, "pause") and callable(node.pause):
                try:
                    if asyncio.iscoroutinefunction(node.pause):
                        await node.pause()
                    else:
                        node.pause()
                except Exception as e:
                    logger.error(f"Error pausing node {node_id}: {e}")

    async def resume(self) -> None:
        """Resume flow execution."""
        if self._state != FlowState.PAUSED:
            return

        logger.info("Resuming flow execution")
        self.state = FlowState.RUNNING

        # Resume all nodes
        for node_id, node in self._nodes.items():
            if hasattr(node, "resume") and callable(node.resume):
                try:
                    if asyncio.iscoroutinefunction(node.resume):
                        await node.resume()
                    else:
                        node.resume()
                except Exception as e:
                    logger.error(f"Error resuming node {node_id}: {e}")

    def trigger_exec(self, node_id: str, exec_output: int = 0) -> None:
        """
        Trigger an execution flow from a node.

        Args:
            node_id: Source node ID
            exec_output: Execution output index
        """
        node = self._nodes.get(node_id)
        if node is None:
            return

        if hasattr(node, "exec_output"):
            try:
                node.exec_output(exec_output)
            except Exception as e:
                logger.error(f"Error triggering exec on node {node_id}: {e}")
                self._notify_error(node_id, e)

    def clear(self) -> None:
        """Clear all nodes and connections and reset state."""
        self._nodes.clear()
        self._connections.clear()
        self._running_tasks.clear()
        if hasattr(self, "_custom_device_runners"):
            self._custom_device_runners.clear()
        self.state = FlowState.STOPPED

        if self._ryvencore_available and self._session:
            # Re-register nodes with ryvencore (ensures new flow can create them)
            self._register_nodes_with_ryvencore()
            # Remove old flow and create new one
            self._flow = self._session.create_flow("Main Flow")

        logger.info("Flow cleared")

    async def shutdown(self) -> None:
        """Shutdown the flow engine."""
        logger.info("Shutting down flow engine")
        await self.stop()
        self._nodes.clear()
        self._session = None
        self._flow = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize flow state to dictionary."""
        nodes = []
        for node_id, node in self._nodes.items():
            node_data = {
                "id": node_id,
                "node_type": type(node).__name__,
            }
            if hasattr(node, "get_state"):
                node_data["state"] = node.get_state()
            nodes.append(node_data)

        return {"nodes": nodes}

    # =========================================================================
    # Agent Tool Helper Methods
    # =========================================================================

    def get_nodes(self) -> list[dict[str, Any]]:
        """Get list of all nodes with their metadata."""
        result = []
        for node_id, node in self._nodes.items():
            node_info = {
                "id": node_id,
                "type": type(node).__name__.replace("Node", ""),
                "name": getattr(node, "name", node_id),
            }
            if hasattr(node, "get_state"):
                node_info["state"] = node.get_state()
            if hasattr(node, "_device") and node._device:
                node_info["device_id"] = node._device.id
            result.append(node_info)
        return result

    def get_connections(self) -> list[dict[str, Any]]:
        """Get list of all connections."""
        return self._connections.copy()

    def delete_node(self, node_id: str) -> bool:
        """Delete a node by ID. Returns True if deleted."""
        if node_id in self._nodes:
            self.remove_node(node_id)
            # Also remove related connections
            self._connections = [
                c
                for c in self._connections
                if c["from_node"] != node_id and c["to_node"] != node_id
            ]
            return True
        return False

    def connect_nodes(
        self,
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str,
    ) -> str:
        """
        Connect nodes by port name.

        Args:
            from_node: Source node ID or name
            from_port: Source port name (e.g., 'exec', 'value')
            to_node: Target node ID or name
            to_port: Target port name

        Returns:
            Connection ID
        """
        # Resolve node IDs if names given
        from_id = self._resolve_node_id(from_node)
        to_id = self._resolve_node_id(to_node)

        if from_id is None:
            raise ValueError(f"Source node not found: {from_node}")
        if to_id is None:
            raise ValueError(f"Target node not found: {to_node}")

        # Get port indices
        from_idx = self._get_output_index(from_id, from_port)
        to_idx = self._get_input_index(to_id, to_port)

        if from_idx is None:
            raise ValueError(f"Output port not found: {from_port}")
        if to_idx is None:
            raise ValueError(f"Input port not found: {to_port}")

        # Generate connection ID
        import uuid

        conn_id = f"conn_{uuid.uuid4().hex[:8]}"

        # Determine connection type
        conn_type = "exec" if from_port == "exec" or to_port == "exec" else "data"

        self.create_connection(conn_id, from_id, from_idx, to_id, to_idx, conn_type)

        return conn_id

    def disconnect_nodes(
        self,
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str,
    ) -> bool:
        """Disconnect nodes. Returns True if disconnected."""
        from_id = self._resolve_node_id(from_node)
        to_id = self._resolve_node_id(to_node)

        if from_id is None or to_id is None:
            return False

        from_idx = self._get_output_index(from_id, from_port)
        to_idx = self._get_input_index(to_id, to_port)

        # Find and remove connection
        for i, conn in enumerate(self._connections):
            if (
                conn["from_node"] == from_id
                and conn["from_output"] == from_idx
                and conn["to_node"] == to_id
                and conn["to_input"] == to_idx
            ):
                self._connections.pop(i)
                return True

        return False

    def set_node_property(self, node_id: str, property_name: str, value: Any) -> bool:
        """Set a property on a node. Returns True if successful."""
        node_id = self._resolve_node_id(node_id)
        if node_id is None:
            return False

        node = self._nodes.get(node_id)
        if node is None:
            return False

        # Reject private/internal property names for safety
        if property_name.startswith("_"):
            logger.warning(f"Rejected attempt to set private property '{property_name}'")
            return False

        # Try different ways to set property
        if hasattr(node, "set_property"):
            node.set_property(property_name, value)
            return True
        elif hasattr(node, "_state") and isinstance(node._state, dict):
            node._state[property_name] = value
            return True
        elif hasattr(node, property_name):
            setattr(node, property_name, value)
            return True

        return False

    def validate(self) -> list[str]:
        """Validate the flow graph. Returns list of error messages."""
        errors = []

        def get_node_type_name(node) -> str:
            """Get node type name from definition or class name."""
            if hasattr(node, "definition") and hasattr(node.definition, "name"):
                return node.definition.name
            return type(node).__name__.replace("Node", "")

        # Check for StartExperiment node
        has_start = any(get_node_type_name(n) == "StartExperiment" for n in self._nodes.values())
        if not has_start:
            errors.append("Missing StartExperiment node")

        # Check for disconnected nodes
        connected_nodes = set()
        for conn in self._connections:
            connected_nodes.add(conn["from_node"])
            connected_nodes.add(conn["to_node"])

        for node_id in self._nodes:
            if node_id not in connected_nodes and len(self._nodes) > 1:
                node = self._nodes[node_id]
                node_type = get_node_type_name(node)
                # Some nodes don't need connections
                if node_type not in ("StartExperiment",):
                    errors.append(f"Node '{node_id}' is not connected")

        return errors

    def _resolve_node_id(self, node_ref: str) -> str | None:
        """Resolve a node reference (ID or name) to node ID."""
        # Direct ID match
        if node_ref in self._nodes:
            return node_ref

        # Search by name
        for node_id, node in self._nodes.items():
            if getattr(node, "name", None) == node_ref:
                return node_id

        return None

    def _get_output_index(self, node_id: str, port_name: str) -> int | None:
        """Get output port index by name."""
        node = self._nodes.get(node_id)
        if node is None:
            return None

        # Common port mappings
        port_map = {"exec": 0, "value": 0, "result": 0, "next": 0, "body": 0, "complete": 1}

        if hasattr(node, "definition") and hasattr(node.definition, "outputs"):
            for i, out in enumerate(node.definition.outputs):
                if out.name == port_name:
                    return i

        return port_map.get(port_name, 0)

    def _get_input_index(self, node_id: str, port_name: str) -> int | None:
        """Get input port index by name."""
        node = self._nodes.get(node_id)
        if node is None:
            return None

        # Common port mappings
        port_map = {"exec": 0, "value": 0, "condition": 1, "duration": 1, "count": 1}

        if hasattr(node, "definition") and hasattr(node.definition, "inputs"):
            for i, inp in enumerate(node.definition.inputs):
                if inp.name == port_name:
                    return i

        return port_map.get(port_name, 0)

    def load_from_session(self, session) -> None:
        """
        Load flow from an ExperimentSession.

        Args:
            session: ExperimentSession to load from
        """
        # Clear existing
        self.clear()

        # Create nodes
        for node_config in session.flow.nodes:
            try:
                self.create_node(
                    node_id=node_config.id,
                    node_type=node_config.node_type,
                    position=node_config.position,
                    state=node_config.state,
                    device_id=node_config.device_id,
                    session=session,
                )
            except Exception as e:
                logger.error(f"Error creating node {node_config.id}: {e}")

        # Create connections
        logger.info(f"Loading {len(session.flow.connections)} connections...")
        for conn_config in session.flow.connections:
            try:
                logger.debug(
                    f"Creating connection: {conn_config.from_node}:{conn_config.from_output} -> {conn_config.to_node}:{conn_config.to_input}"
                )
                self.create_connection(
                    connection_id=conn_config.id,
                    from_node_id=conn_config.from_node,
                    from_output=conn_config.from_output,
                    to_node_id=conn_config.to_node,
                    to_input=conn_config.to_input,
                    connection_type=conn_config.connection_type,
                )
            except Exception as e:
                logger.error(f"Error creating connection {conn_config.id}: {e}")

        logger.info(
            f"Loaded flow with {len(self._nodes)} nodes and {len(self._connections)} connections"
        )
