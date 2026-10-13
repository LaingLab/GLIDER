"""
Zone Input Nodes - Nodes for zone-based triggering in the node graph.

Each zone can have its own ZoneInputNode that outputs:
- Occupied: bool indicating if any object is in the zone
- Object Count: int number of objects in the zone
- On Enter: exec triggered when the zone goes from empty to occupied
- On Exit: exec triggered when the zone goes from occupied to empty
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

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
                description="Triggered when the zone goes from empty to occupied",
            ),
            PortDefinition(
                name="On Exit",
                port_type=PortType.EXEC,
                description="Triggered when the zone goes from occupied to empty",
            ),
        ],
        color="#5a4a2d",  # Orange - interface color
    )

    PROPERTIES_SCHEMA = [
        {
            "key": "zone_id",
            "label": "Zone id",
            "type": "str",
            "default": "",
            "help": "Id of the arena zone this node watches.",
        },
        {
            "key": "zone_name",
            "label": "Zone name",
            "type": "str",
            "default": "Unnamed Zone",
            "help": "Display name of the zone.",
        },
    ]

    def __init__(self):
        super().__init__()
        self._zone_id: str = self.setting("zone_id")
        self._zone_name: str = self.setting("zone_name")
        self._occupied = False
        self._object_count = 0
        self._exec_callbacks: list[Callable[[int], None]] = []
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._bus: Any = None
        self._subscribed = False

        # Set outputs to initial values
        self._outputs = [False, 0, None, None]

    async def start(self) -> None:
        """Capture the main event loop and subscribe to the live signal bus.

        The loop has to be captured here: zone updates arrive on the CV thread,
        where ``asyncio.get_event_loop()`` returns a fresh, non-running loop.
        """
        self._main_loop = asyncio.get_running_loop()
        if self._bus is None:
            logger.warning(
                "Zone Input '%s': no live signal bus attached; the node will not "
                "fire. The flow engine injects one via set_live_signals.",
                self._zone_name,
            )
            return
        self._subscribe()

    async def stop(self) -> None:
        """Unsubscribe so a stopped flow stops receiving zone updates."""
        self._unsubscribe()

    def set_live_signals(self, bus: Any) -> None:
        """Injected by the flow engine at node creation (as for Behavior Input)."""
        if bus is self._bus:
            return
        was_subscribed = self._subscribed
        self._unsubscribe()
        self._bus = bus
        if was_subscribed or self._main_loop is not None:
            self._subscribe()

    def _subscribe(self) -> None:
        if self._bus is not None and not self._subscribed:
            self._bus.subscribe_zones(self.handle_zone_states)
            self._subscribed = True

    def _unsubscribe(self) -> None:
        if self._bus is not None and self._subscribed:
            self._bus.unsubscribe_zones(self.handle_zone_states)
        self._subscribed = False

    def handle_zone_states(self, states: dict[str, Any]) -> None:
        """One frame's ``{zone_id: ZoneState}`` from the CV processor."""
        state = states.get(self._zone_id)
        if state is not None:
            self.update_zone_state(state.occupied, state.object_count)

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

    def update_zone_state(self, occupied: bool, object_count: int) -> None:
        """
        Update the zone state from CV processor.

        This is called each frame with the current zone state.

        On Enter / On Exit fire on the zone's occupancy changing, not on the
        tracker's per-object enter/exit: its IDs are not stable. A duplicate
        detection vanishing, or the animal being re-IDed (old ID exits, new ID
        enters, same frame), reported an exit while the mouse sat inside the
        zone -- and turned a stimulator off.

        Args:
            occupied: Whether any objects are in the zone
            object_count: Number of objects in the zone
        """
        was_occupied = self._occupied
        self._occupied = occupied
        self._object_count = object_count

        # Update data outputs
        self.set_output(0, occupied)  # Occupied
        self.set_output(1, object_count)  # Object Count

        # Trigger exec outputs on events.  update_zone_state() is called from the
        # CV thread, so we need a reference to the main event loop captured at
        # start().  asyncio.get_event_loop() from a non-main thread returns a new,
        # non-running loop on Python >=3.12, silently dropping exec outputs.
        loop = self._main_loop
        if loop is None:
            logger.debug(
                "Zone '%s': main loop not captured, skipping exec outputs", self._zone_name
            )
            return

        if occupied and not was_occupied:
            asyncio.run_coroutine_threadsafe(self._fire_exec_output("On Enter"), loop)
            logger.debug(f"Zone '{self._zone_name}': On Enter triggered")

        if was_occupied and not occupied:
            asyncio.run_coroutine_threadsafe(self._fire_exec_output("On Exit"), loop)
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
        self._zone_id = self.setting("zone_id")
        self._zone_name = self.setting("zone_name")

    def get_display_name(self) -> str:
        """Get the display name for this node instance."""
        return f"Zone: {self._zone_name}" if self._zone_name else "Zone Input"


def register_zone_nodes(flow_engine) -> None:
    """Register all zone nodes with the flow engine."""
    flow_engine.register_node("ZoneInput", ZoneInputNode)
    logger.info("Registered zone nodes")
