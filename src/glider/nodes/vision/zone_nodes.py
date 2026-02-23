"""
Zone Input Nodes - Nodes for zone-based triggering in the node graph.

Each zone can have its own ZoneInputNode that outputs:
- Occupied: bool indicating if any object is in the zone
- Object Count: int number of objects in the zone
- On Enter: exec triggered when an object enters the zone
- On Exit: exec triggered when an object exits the zone
"""

import asyncio
import logging
from typing import Any, Callable

from glider.nodes.base_node import (
    InterfaceNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)


class ZoneInputNode(InterfaceNode):
    """
    Node that monitors a specific zone for object occupancy.

    Provides outputs for zone state that can be connected to other nodes
    for zone-based automation.
    """

    definition = NodeDefinition(
        name="Zone Input",
        category=NodeCategory.INTERFACE,
        description="Monitors zone occupancy and triggers events on enter/exit",
        inputs=[],
        outputs=[
            PortDefinition(
                name="Occupied", data_type=bool, description="True when any object is in the zone"
            ),
            PortDefinition(
                name="Object Count",
                data_type=int,
                description="Number of objects currently in the zone",
            ),
            PortDefinition(
                name="On Enter",
                port_type=PortType.EXEC,
                description="Triggered when an object enters the zone",
            ),
            PortDefinition(
                name="On Exit",
                port_type=PortType.EXEC,
                description="Triggered when an object exits the zone",
            ),
        ],
        color="#5a4a2d",  # Orange - interface color
    )

    def __init__(self):
        super().__init__()
        self._zone_id: str = ""
        self._zone_name: str = "Unnamed Zone"
        self._occupied = False
        self._object_count = 0
        self._exec_callbacks: list[Callable[[int], None]] = []

        # Set outputs to initial values
        self._outputs = [False, 0, None, None]

    @property
    def zone_id(self) -> str:
        """ID of the zone this node monitors."""
        return self._zone_id

    @zone_id.setter
    def zone_id(self, value: str) -> None:
        self._zone_id = value

    @property
    def zone_name(self) -> str:
        """Display name of the zone."""
        return self._zone_name

    @zone_name.setter
    def zone_name(self, value: str) -> None:
        self._zone_name = value

    @property
    def occupied(self) -> bool:
        """Whether the zone is currently occupied."""
        return self._occupied

    @property
    def object_count(self) -> int:
        """Number of objects in the zone."""
        return self._object_count

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

    def update_zone_state(
        self, occupied: bool, object_count: int, entered: bool, exited: bool
    ) -> None:
        """
        Update the zone state from CV processor.

        This is called each frame with the current zone state.

        Args:
            occupied: Whether any objects are in the zone
            object_count: Number of objects in the zone
            entered: True if an object entered this frame
            exited: True if an object exited this frame
        """
        # Update state
        self._occupied = occupied
        self._object_count = object_count

        # Update data outputs
        self.set_output(0, occupied)  # Occupied
        self.set_output(1, object_count)  # Object Count

        # Trigger exec outputs on events (use _fire_exec_output to reach flow engine
        # callbacks registered on _update_callbacks; wrap in create_task since
        # update_zone_state is called synchronously from the CV thread)
        if entered:
            try:
                asyncio.create_task(self._fire_exec_output("On Enter"))
            except RuntimeError:
                logger.debug("No event loop for On Enter exec output")
            logger.debug(f"Zone '{self._zone_name}': On Enter triggered")

        if exited:
            try:
                asyncio.create_task(self._fire_exec_output("On Exit"))
            except RuntimeError:
                logger.debug("No event loop for On Exit exec output")
            logger.debug(f"Zone '{self._zone_name}': On Exit triggered")

    def update_event(self) -> None:
        """Zone inputs update from CV processor, not from node inputs."""
        pass

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["zone_id"] = self._zone_id
        state["zone_name"] = self._zone_name
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._zone_id = state.get("zone_id", "")
        self._zone_name = state.get("zone_name", "Unnamed Zone")

    def get_display_name(self) -> str:
        """Get the display name for this node instance."""
        return f"Zone: {self._zone_name}" if self._zone_name else "Zone Input"


def register_zone_nodes(flow_engine) -> None:
    """Register all zone nodes with the flow engine."""
    flow_engine.register_node("ZoneInput", ZoneInputNode)
    logger.info("Registered zone nodes")
