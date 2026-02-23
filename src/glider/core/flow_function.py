"""
Flow Function System - User-definable sub-flows that become callable nodes.

Flow functions allow users to create reusable sequences of nodes
that appear as single callable nodes in the main flow graph.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glider.core.flow_engine import FlowEngine
    from glider.nodes.base_node import GliderNode

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """Types of parameters for flow functions."""

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"


@dataclass
class FlowFunctionParameter:
    """Definition of a flow function parameter."""

    name: str
    param_type: ParameterType
    default_value: Any = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "param_type": self.param_type.value,
            "default_value": self.default_value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowFunctionParameter":
        return cls(
            name=data["name"],
            param_type=ParameterType(data["param_type"]),
            default_value=data.get("default_value"),
            description=data.get("description", ""),
        )

    def convert_value(self, value: Any) -> Any:
        """Convert a value to this parameter's type."""
        if value is None:
            return self.default_value
        if self.param_type == ParameterType.INT:
            return int(value)
        elif self.param_type == ParameterType.FLOAT:
            return float(value)
        elif self.param_type == ParameterType.BOOL:
            return bool(value)
        elif self.param_type == ParameterType.STRING:
            return str(value)
        return value


@dataclass
class FlowFunctionOutput:
    """Definition of a flow function output value."""

    name: str
    output_type: ParameterType = ParameterType.STRING
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "output_type": self.output_type.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowFunctionOutput":
        return cls(
            name=data["name"],
            output_type=ParameterType(data.get("output_type", "string")),
            description=data.get("description", ""),
        )


@dataclass
class InternalNodeConfig:
    """Configuration for a node within a flow function."""

    id: str
    node_type: str
    position: tuple = (0, 0)
    state: dict[str, Any] = field(default_factory=dict)
    device_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "position": list(self.position),
            "state": self.state,
            "device_id": self.device_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InternalNodeConfig":
        return cls(
            id=data["id"],
            node_type=data["node_type"],
            position=tuple(data.get("position", [0, 0])),
            state=data.get("state", {}),
            device_id=data.get("device_id"),
        )


@dataclass
class InternalConnectionConfig:
    """Configuration for a connection within a flow function."""

    id: str
    from_node: str
    from_output: int
    to_node: str
    to_input: int
    connection_type: str = "exec"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_node": self.from_node,
            "from_output": self.from_output,
            "to_node": self.to_node,
            "to_input": self.to_input,
            "connection_type": self.connection_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InternalConnectionConfig":
        return cls(
            id=data["id"],
            from_node=data["from_node"],
            from_output=data["from_output"],
            to_node=data["to_node"],
            to_input=data["to_input"],
            connection_type=data.get("connection_type", "exec"),
        )


@dataclass
class FlowFunctionDefinition:
    """Complete definition of a flow function."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Function"
    description: str = ""
    parameters: list[FlowFunctionParameter] = field(default_factory=list)
    outputs: list[FlowFunctionOutput] = field(default_factory=list)
    nodes: list[InternalNodeConfig] = field(default_factory=list)
    connections: list[InternalConnectionConfig] = field(default_factory=list)
    entry_node_id: str | None = None
    exit_node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "outputs": [o.to_dict() for o in self.outputs],
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
            "entry_node_id": self.entry_node_id,
            "exit_node_ids": self.exit_node_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlowFunctionDefinition":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", "Untitled Function"),
            description=data.get("description", ""),
            parameters=[FlowFunctionParameter.from_dict(p) for p in data.get("parameters", [])],
            outputs=[FlowFunctionOutput.from_dict(o) for o in data.get("outputs", [])],
            nodes=[InternalNodeConfig.from_dict(n) for n in data.get("nodes", [])],
            connections=[
                InternalConnectionConfig.from_dict(c) for c in data.get("connections", [])
            ],
            entry_node_id=data.get("entry_node_id"),
            exit_node_ids=data.get("exit_node_ids", []),
        )

    def get_parameter(self, name: str) -> FlowFunctionParameter | None:
        """Get a parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None


class FlowFunctionRunner:
    """
    Executes a flow function's internal graph.

    Creates a temporary sub-flow from the definition and executes it,
    passing parameters and collecting outputs.
    """

    def __init__(
        self,
        definition: FlowFunctionDefinition,
        flow_engine: "FlowEngine",
        hardware_manager=None,
    ):
        """
        Initialize the flow function runner.

        Args:
            definition: The flow function definition
            flow_engine: The main flow engine (for node registry access)
            hardware_manager: Hardware manager for device access
        """
        self._definition = definition
        self._flow_engine = flow_engine
        self._hardware_manager = hardware_manager
        self._internal_nodes: dict[str, Any] = {}
        self._completion_event: asyncio.Event | None = None
        self._output_values: dict[str, Any] = {}

    @property
    def definition(self) -> FlowFunctionDefinition:
        return self._definition

    async def execute(self, parameters: dict[str, Any] = None) -> dict[str, Any]:
        """
        Execute the flow function.

        Args:
            parameters: Parameter values to pass to the function

        Returns:
            Dictionary of output values
        """
        parameters = parameters or {}
        self._output_values = {}
        self._completion_event = asyncio.Event()

        logger.info(f"Executing flow function '{self._definition.name}' with params: {parameters}")

        try:
            # Create internal nodes
            await self._create_internal_nodes(parameters)

            # Wire up connections
            self._create_internal_connections()

            # Find and execute from entry point
            if self._definition.entry_node_id:
                entry_node = self._internal_nodes.get(self._definition.entry_node_id)
                if entry_node and hasattr(entry_node, "start"):
                    await entry_node.start()
                elif entry_node and hasattr(entry_node, "execute"):
                    await entry_node.execute()

            # Wait for completion (exit node reached) with timeout
            try:
                await asyncio.wait_for(self._completion_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                logger.warning(f"Flow function '{self._definition.name}' timed out")

        except Exception as e:
            logger.error(f"Error executing flow function: {e}")
            raise

        finally:
            # Cleanup internal nodes
            await self._cleanup()

        return self._output_values

    async def _create_internal_nodes(self, parameters: dict[str, Any]) -> None:
        """Create all internal nodes for this execution."""
        for node_config in self._definition.nodes:
            node_class = self._flow_engine.get_node_class(node_config.node_type)
            if node_class is None:
                # Check for special flow function nodes
                if node_config.node_type == "FlowFunctionEntry":
                    node = self._create_entry_node(parameters)
                elif node_config.node_type == "FlowFunctionExit":
                    node = self._create_exit_node()
                elif node_config.node_type == "Parameter":
                    param_name = node_config.state.get("parameter_name")
                    node = self._create_parameter_node(param_name, parameters)
                else:
                    logger.warning(f"Unknown node type: {node_config.node_type}")
                    continue
            else:
                node = node_class()
                node._glider_id = f"ff_{self._definition.id}_{node_config.id}"

                # Apply state
                if node_config.state and hasattr(node, "set_state"):
                    node.set_state(node_config.state)

                # Bind device if specified
                if node_config.device_id and self._hardware_manager:
                    device = self._hardware_manager.get_device(node_config.device_id)
                    if device and hasattr(node, "bind_device"):
                        node.bind_device(device)

            self._internal_nodes[node_config.id] = node

    def _create_entry_node(self, parameters: dict[str, Any]):
        """Create an entry node that triggers the flow."""
        from glider.nodes.base_node import (
            GliderNode,
            NodeCategory,
            NodeDefinition,
            PortDefinition,
            PortType,
        )

        class EntryNode(GliderNode):
            definition = NodeDefinition(
                name="FlowFunctionEntry",
                category=NodeCategory.LOGIC,
                description="Flow function entry point",
                outputs=[PortDefinition("next", PortType.EXEC)],
            )

            def __init__(self, params):
                super().__init__()
                self._params = params

            def update_event(self):
                pass

            async def start(self):
                self.exec_output(0)

            def exec_output(self, index=0):
                for callback in self._update_callbacks:
                    callback("next", True)

        node = EntryNode(parameters)
        node._glider_id = f"ff_{self._definition.id}_entry"
        return node

    def _create_exit_node(self):
        """Create an exit node that signals completion."""
        from glider.nodes.base_node import (
            GliderNode,
            NodeCategory,
            NodeDefinition,
            PortDefinition,
            PortType,
        )

        runner = self

        class ExitNode(GliderNode):
            definition = NodeDefinition(
                name="FlowFunctionExit",
                category=NodeCategory.LOGIC,
                description="Flow function exit point",
                inputs=[PortDefinition("exec", PortType.EXEC)],
            )

            def update_event(self):
                pass

            async def execute(self):
                logger.info("Flow function exit reached")
                runner._completion_event.set()

        node = ExitNode()
        node._glider_id = f"ff_{self._definition.id}_exit"
        return node

    def _create_parameter_node(self, param_name: str, parameters: dict[str, Any]):
        """Create a node that exposes a parameter value."""
        from glider.nodes.base_node import (
            GliderNode,
            NodeCategory,
            NodeDefinition,
            PortDefinition,
            PortType,
        )

        param_def = self._definition.get_parameter(param_name)
        value = parameters.get(param_name)
        if param_def:
            value = param_def.convert_value(value)

        class ParameterNode(GliderNode):
            definition = NodeDefinition(
                name="Parameter",
                category=NodeCategory.LOGIC,
                description="Exposes a parameter value",
                outputs=[PortDefinition("value", PortType.DATA)],
            )

            def __init__(self, val):
                super().__init__()
                self._value = val

            def update_event(self):
                pass

            def get_output(self, index=0):
                return self._value

        node = ParameterNode(value)
        node._glider_id = f"ff_{self._definition.id}_param_{param_name}"
        node._outputs = [value]
        return node

    def _create_internal_connections(self) -> None:
        """Wire up connections between internal nodes."""
        for conn in self._definition.connections:
            from_node = self._internal_nodes.get(conn.from_node)
            to_node = self._internal_nodes.get(conn.to_node)

            if from_node is None or to_node is None:
                logger.warning(f"Missing node for connection: {conn.from_node} -> {conn.to_node}")
                continue

            # Create execution callback
            def make_exec_callback(target):
                async def propagate():
                    if hasattr(target, "execute"):
                        await target.execute()

                return lambda name, val: asyncio.create_task(propagate())

            if hasattr(from_node, "_update_callbacks"):
                from_node._update_callbacks.append(make_exec_callback(to_node))

    async def _cleanup(self) -> None:
        """Clean up internal nodes."""
        for node_id, node in self._internal_nodes.items():
            if hasattr(node, "stop"):
                try:
                    await node.stop()
                except Exception as e:
                    logger.warning(f"Error stopping internal node {node_id}: {e}")

        self._internal_nodes.clear()


def create_flow_function_node_class(definition: FlowFunctionDefinition) -> type["GliderNode"]:
    """
    Dynamically create a GliderNode class from a FlowFunctionDefinition.

    This creates a node class that, when executed, runs the flow function's
    internal graph.

    Args:
        definition: The flow function definition

    Returns:
        A GliderNode subclass that executes the flow function
    """
    from glider.nodes.base_node import (
        GliderNode,
        NodeCategory,
        NodeDefinition,
        PortDefinition,
        PortType,
    )

    # Build input ports from parameters
    inputs = [PortDefinition("exec", PortType.EXEC)]
    for param in definition.parameters:
        inputs.append(
            PortDefinition(
                param.name,
                PortType.DATA,
                _param_type_to_python(param.param_type),
                param.default_value,
                param.description,
            )
        )

    # Build output ports
    outputs = [PortDefinition("next", PortType.EXEC)]
    for output in definition.outputs:
        outputs.append(
            PortDefinition(
                output.name,
                PortType.DATA,
                _param_type_to_python(output.output_type),
                description=output.description,
            )
        )

    # Create the node definition
    node_def = NodeDefinition(
        name=f"FlowFunction:{definition.name}",
        category=NodeCategory.LOGIC,
        description=definition.description or f"Flow function: {definition.name}",
        inputs=inputs,
        outputs=outputs,
    )

    # Store definition ID for serialization
    def_id = definition.id
    def_dict = definition.to_dict()

    class FlowFunctionNode(GliderNode):
        definition = node_def
        _flow_function_id = def_id
        _flow_function_def = def_dict

        def __init__(self):
            super().__init__()
            self._runner: FlowFunctionRunner | None = None
            self._flow_engine = None
            self._hardware_manager = None

        def set_flow_context(self, flow_engine, hardware_manager=None):
            """Set the flow engine and hardware manager for execution."""
            self._flow_engine = flow_engine
            self._hardware_manager = hardware_manager

        def update_event(self):
            pass

        async def execute(self):
            """Execute the flow function."""
            if self._flow_engine is None:
                logger.error("Flow function node has no flow engine context")
                return

            # Rebuild definition from stored dict
            ff_def = FlowFunctionDefinition.from_dict(self._flow_function_def)

            # Collect parameter values from inputs
            parameters = {}
            for i, param in enumerate(ff_def.parameters):
                # Parameter inputs start at index 1 (after exec input)
                if i + 1 < len(self._inputs):
                    parameters[param.name] = self._inputs[i + 1]

            # Create and run the flow function
            self._runner = FlowFunctionRunner(
                ff_def,
                self._flow_engine,
                self._hardware_manager,
            )

            try:
                output_values = await self._runner.execute(parameters)

                # Set output values
                for i, output in enumerate(ff_def.outputs):
                    if output.name in output_values:
                        # Output values start at index 1 (after exec output)
                        if i + 1 < len(self._outputs):
                            self._outputs[i + 1] = output_values[output.name]

            except Exception as e:
                logger.error(f"Flow function execution error: {e}")
                self._error = str(e)

            # Trigger execution output
            self.exec_output(0)

        def exec_output(self, index=0):
            for callback in self._update_callbacks:
                callback("next", True)

    return FlowFunctionNode


def _param_type_to_python(param_type: ParameterType) -> type:
    """Convert ParameterType to Python type."""
    if param_type == ParameterType.INT:
        return int
    elif param_type == ParameterType.FLOAT:
        return float
    elif param_type == ParameterType.BOOL:
        return bool
    elif param_type == ParameterType.STRING:
        return str
    return object
