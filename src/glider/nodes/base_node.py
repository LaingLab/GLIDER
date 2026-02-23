"""
Base Node Class for GLIDER

Extends ryvencore.Node with GLIDER-specific functionality:
- Error handling state
- Hardware device binding
- Serialization helpers
- Async operation support
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from glider.hal.base_device import BaseDevice

logger = logging.getLogger(__name__)


class NodeCategory(Enum):
    """Categories of nodes in GLIDER."""

    HARDWARE = "hardware"  # Green border
    LOGIC = "logic"  # Blue border
    INTERFACE = "interface"  # Orange border


class PortType(Enum):
    """Types of node ports."""

    DATA = auto()  # Data flow
    EXEC = auto()  # Execution flow


@dataclass
class PortDefinition:
    """Definition of a node port."""

    name: str
    port_type: PortType = PortType.DATA
    data_type: type = object
    default_value: Any = None
    description: str = ""


@dataclass
class NodeDefinition:
    """Definition of a node type."""

    name: str
    category: NodeCategory
    description: str = ""
    inputs: list[PortDefinition] = field(default_factory=list)
    outputs: list[PortDefinition] = field(default_factory=list)
    color: str = "#444444"


class GliderNode(ABC):
    """
    Base class for all GLIDER nodes.

    Provides common functionality for nodes including:
    - Input/output management
    - State serialization
    - Device binding for hardware nodes
    - Async operation support
    - Error handling
    """

    # Class-level definition (override in subclasses)
    definition: NodeDefinition = NodeDefinition(
        name="GliderNode",
        category=NodeCategory.LOGIC,
        description="Base node class",
    )

    def __init__(self):
        """Initialize the node."""
        self._glider_id: str = ""
        self._inputs: list[Any] = []
        self._outputs: list[Any] = []
        self._state: dict[str, Any] = {}
        self._device: Optional[BaseDevice] = None
        self._error: Optional[str] = None
        self._enabled = True
        self._visible_in_runner = False

        # Callbacks
        self._update_callbacks: list[Callable[[str, Any], None]] = []
        self._error_callbacks: list[Callable[[Exception], None]] = []

        # Initialize ports from definition
        self._init_ports()

    def _init_ports(self) -> None:
        """Initialize input and output ports from definition."""
        for port_def in self.definition.inputs:
            self._inputs.append(port_def.default_value)
        for _port_def in self.definition.outputs:
            self._outputs.append(None)

    @property
    def id(self) -> str:
        """Node ID."""
        return self._glider_id

    @property
    def name(self) -> str:
        """Node type name."""
        return self.definition.name

    @property
    def category(self) -> NodeCategory:
        """Node category."""
        return self.definition.category

    @property
    def inputs(self) -> list[Any]:
        """Current input values."""
        return self._inputs

    @property
    def outputs(self) -> list[Any]:
        """Current output values."""
        return self._outputs

    @property
    def device(self) -> Optional["BaseDevice"]:
        """Bound hardware device."""
        return self._device

    @property
    def error(self) -> Optional[str]:
        """Current error message, if any."""
        return self._error

    @property
    def is_enabled(self) -> bool:
        """Whether the node is enabled."""
        return self._enabled

    @property
    def visible_in_runner(self) -> bool:
        """Whether visible in runner dashboard."""
        return self._visible_in_runner

    @visible_in_runner.setter
    def visible_in_runner(self, value: bool) -> None:
        self._visible_in_runner = value

    def on_output_update(self, callback: Callable[[str, Any], None]) -> None:
        """Register callback for output updates."""
        self._update_callbacks.append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def get_input(self, index: int) -> Any:
        """Get input value by index."""
        if 0 <= index < len(self._inputs):
            return self._inputs[index]
        return None

    def get_input_by_name(self, name: str) -> Any:
        """Get input value by name."""
        for i, port_def in enumerate(self.definition.inputs):
            if port_def.name == name:
                return self._inputs[i]
        return None

    def set_input(self, index: int, value: Any) -> None:
        """Set input value and trigger update."""
        if 0 <= index < len(self._inputs):
            self._inputs[index] = value
            self.update_event()

    def get_output(self, index: int) -> Any:
        """Get output value by index."""
        if 0 <= index < len(self._outputs):
            return self._outputs[index]
        return None

    def set_output(self, index: int, value: Any) -> None:
        """Set output value and notify listeners."""
        if 0 <= index < len(self._outputs):
            self._outputs[index] = value
            # Notify callbacks
            output_name = (
                self.definition.outputs[index].name
                if index < len(self.definition.outputs)
                else str(index)
            )
            for callback in self._update_callbacks:
                try:
                    callback(output_name, value)
                except Exception as e:
                    logger.error(f"Output callback error: {e}")

    def set_output_val(self, index: int, value: Any) -> None:
        """Alias for set_output (ryvencore compatibility)."""
        self.set_output(index, value)

    def exec_output(self, index: int = 0) -> None:
        """
        Trigger execution flow output.

        Override in subclasses that support execution flow.
        """
        pass

    async def _fire_exec_output(self, output_name: str, value=True):
        """
        Fire callbacks for an exec output and await downstream propagation tasks.

        Unlike the synchronous exec_output(), this method collects any asyncio tasks
        returned by callbacks and awaits them, ensuring the full downstream execution
        chain completes before returning.
        """
        tasks = []
        for callback in self._update_callbacks:
            try:
                result = callback(output_name, value)
                if result is not None and asyncio.isfuture(result):
                    tasks.append(result)
            except Exception as e:
                logger.error(f"Error triggering callback for {output_name}: {e}", exc_info=True)

        if tasks:
            # Use return_exceptions=True so that one failing branch doesn't crash the others
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Error in downstream execution branch {i} from {output_name}: {result}",
                        exc_info=True,
                    )

    def bind_device(self, device: "BaseDevice") -> None:
        """
        Bind a hardware device to this node.

        Args:
            device: Device to bind
        """
        self._device = device

    def unbind_device(self) -> None:
        """Unbind the current device."""
        self._device = None

    def enable(self) -> None:
        """Enable the node."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the node."""
        self._enabled = False

    def set_error(self, error: Optional[str]) -> None:
        """Set error state."""
        self._error = error
        if error:
            logger.error(f"Node {self._glider_id} error: {error}")

    def clear_error(self) -> None:
        """Clear error state."""
        self._error = None

    @abstractmethod
    def update_event(self) -> None:
        """
        Called when input values change.

        Override in subclasses to implement reactive behavior.
        """
        pass

    def get_state(self) -> dict[str, Any]:
        """Get serializable state."""
        return self._state.copy()

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore state from dictionary."""
        self._state = state.copy()

    async def start(self) -> None:
        """
        Called when flow execution starts.

        Override for continuous nodes (timers, sensors).
        """
        pass

    async def stop(self) -> None:
        """
        Called when flow execution stops.

        Override for cleanup.
        """
        pass

    async def pause(self) -> None:
        """Called when flow is paused."""
        pass

    async def resume(self) -> None:
        """Called when flow is resumed."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self._glider_id,
            "node_type": self.name,
            "state": self.get_state(),
            "enabled": self._enabled,
            "visible_in_runner": self._visible_in_runner,
            "device_id": self._device.id if self._device else None,
        }


class DataNode(GliderNode):
    """Base class for data processing nodes."""

    def update_event(self) -> None:
        """Process inputs and update outputs."""
        if not self._enabled:
            return

        try:
            self.process()
            self.clear_error()
        except Exception as e:
            self.set_error(str(e))

    @abstractmethod
    def process(self) -> None:
        """Process inputs and set outputs. Override in subclasses."""
        pass


class ExecNode(GliderNode):
    """Base class for execution flow nodes."""

    def __init__(self):
        super().__init__()
        self._exec_callbacks: list[Callable[[int], None]] = []

    def on_exec(self, callback: Callable[[int], None]) -> None:
        """Register callback for execution output triggers."""
        self._exec_callbacks.append(callback)

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution flow output."""
        for callback in self._exec_callbacks:
            try:
                callback(index)
            except Exception as e:
                logger.error(f"Exec callback error: {e}")

    def update_event(self) -> None:
        """Data update event - may not trigger execution."""
        pass

    @abstractmethod
    async def execute(self) -> None:
        """Execute the node's action. Override in subclasses."""
        pass


class HardwareNode(ExecNode):
    """Base class for hardware interaction nodes."""

    definition = NodeDefinition(
        name="HardwareNode",
        category=NodeCategory.HARDWARE,
        description="Base hardware node",
        color="#2d5a2d",  # Green
    )

    async def execute(self) -> None:
        """Execute hardware operation."""
        if self._device is None:
            self.set_error("No device bound")
            return

        if not self._enabled:
            return

        try:
            await self.hardware_operation()
            self.clear_error()
        except Exception as e:
            self.set_error(str(e))

    @abstractmethod
    async def hardware_operation(self) -> None:
        """Perform hardware operation. Override in subclasses."""
        pass


class LogicNode(DataNode):
    """Base class for logic/math nodes."""

    definition = NodeDefinition(
        name="LogicNode",
        category=NodeCategory.LOGIC,
        description="Base logic node",
        color="#2d4a5a",  # Blue
    )


class InterfaceNode(GliderNode):
    """Base class for UI interface nodes."""

    definition = NodeDefinition(
        name="InterfaceNode",
        category=NodeCategory.INTERFACE,
        description="Base interface node",
        color="#5a4a2d",  # Orange
    )

    def __init__(self):
        super().__init__()
        self._visible_in_runner = True  # Interface nodes visible by default
        self._widget_callbacks: list[Callable[[Any], None]] = []

    def on_widget_update(self, callback: Callable[[Any], None]) -> None:
        """Register callback for widget value updates."""
        self._widget_callbacks.append(callback)

    def notify_widget(self, value: Any) -> None:
        """Notify widget of value change."""
        for callback in self._widget_callbacks:
            try:
                callback(value)
            except Exception as e:
                logger.error(f"Widget callback error: {e}")

    def update_event(self) -> None:
        """Update widget when inputs change."""
        if self._inputs:
            self.notify_widget(self._inputs[0])
