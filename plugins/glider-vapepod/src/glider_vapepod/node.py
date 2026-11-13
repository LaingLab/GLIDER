"""VapePod Node

Drives the vape box side pod from the flow graph: a timed puff, the pump on or
off, or everything off. The generic Device Action node can do the same, but this
one offers Mode / Duration / Power instead of action names and arguments.
"""

from typing import Any

from glider.nodes.base_node import (
    HardwareNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)
from glider_vapepod.device import MAX_PUFF_MS

# Mode -> the device action it calls.
MODE_PUFF = "puff"
MODE_PUMP_ON = "pump_on"
MODE_PUMP_OFF = "pump_off"
MODE_OFF = "off"
MODES = (MODE_PUFF, MODE_PUMP_ON, MODE_PUMP_OFF, MODE_OFF)

DEFAULT_DURATION_MS = 3000
DEFAULT_POWER = 100
DEFAULT_PUMP_PERCENT = 100


class VapePodNode(HardwareNode):
    """Send one VapePod command: a puff, the pump on or off, or all off.

    Settings come from the node's properties, not input ports, for the reason
    MaimuNode gives: a data input on an exec-driven node never receives a value
    in GLIDER today.

    A puff is fire-and-continue: ``exec`` out fires once the command has landed
    and the pod times the puff itself. Put a Delay node after this one to hold
    the flow for the puff.
    """

    PROPERTIES_SCHEMA = [
        {
            "key": "mode",
            "label": "Mode",
            "type": "enum",
            "default": MODE_PUFF,
            "choices": [
                [MODE_PUFF, "Puff"],
                [MODE_PUMP_ON, "Pump on"],
                [MODE_PUMP_OFF, "Pump off"],
                [MODE_OFF, "All off"],
            ],
            "help": "Puff runs the atomizer and stops on its own; the pump latches.",
        },
        {
            "key": "duration_ms",
            "label": "Duration (ms)",
            "type": "int",
            "default": DEFAULT_DURATION_MS,
            "min": 1,
            "max": MAX_PUFF_MS,
            "help": (
                f"Puff mode only. How long the atomizer runs, at most {MAX_PUFF_MS} ms. "
                "The flow continues at once; add a Delay node to hold it."
            ),
        },
        {
            "key": "power",
            "label": "Power (%)",
            "type": "int",
            "default": DEFAULT_POWER,
            "min": 1,
            "max": 100,
            "help": "Puff mode only. Atomizer duty cycle; 100 is the full 12 V.",
        },
        {
            "key": "pump_percent",
            "label": "Pump speed (%)",
            "type": "int",
            "default": DEFAULT_PUMP_PERCENT,
            "min": 1,
            "max": 100,
            "help": "Pump on mode only. Below 100 the pump is switched at 1 kHz and whines.",
        },
    ]

    definition = NodeDefinition(
        name="VapePod",
        category=NodeCategory.HARDWARE,
        description="Drive the vape side pod: a timed puff, or its pump on or off",
        inputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Send the configured command",
            ),
        ],
        outputs=[
            PortDefinition(
                name="exec",
                port_type=PortType.EXEC,
                description="Triggered once the command has been sent",
            ),
        ],
        color="#2d5a2d",
    )

    def __init__(self):
        super().__init__()
        self._mode = MODE_PUFF
        self._duration_ms = DEFAULT_DURATION_MS
        self._power = DEFAULT_POWER
        self._pump_percent = DEFAULT_PUMP_PERCENT

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        text = str(value).strip().lower()
        if text not in MODES:
            raise ValueError(f"VapePod mode must be one of {MODES}, got {value!r}")
        self._mode = text

    @property
    def duration_ms(self) -> int:
        return self._duration_ms

    @duration_ms.setter
    def duration_ms(self, value: Any) -> None:
        self._duration_ms = int(value)

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, value: Any) -> None:
        self._power = int(value)

    @property
    def pump_percent(self) -> int:
        return self._pump_percent

    @pump_percent.setter
    def pump_percent(self, value: Any) -> None:
        self._pump_percent = int(value)

    async def hardware_operation(self) -> None:
        """Send the configured command, then trigger the exec output."""
        if self._mode == MODE_PUFF:
            await self._device.execute_action(MODE_PUFF, self._duration_ms, self._power)
        elif self._mode == MODE_PUMP_ON:
            await self._device.execute_action("pump", self._pump_percent)
        else:
            await self._device.execute_action(self._mode)

        await self._fire_exec_output("exec")

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["mode"] = self._mode
        state["duration_ms"] = self._duration_ms
        state["power"] = self._power
        state["pump_percent"] = self._pump_percent
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        # A saved file is not a trusted source of a valid mode: an unknown one
        # falls back to the default rather than reaching the device.
        mode = str(state.get("mode", MODE_PUFF)).strip().lower()
        self._mode = mode if mode in MODES else MODE_PUFF
        self._duration_ms = int(state.get("duration_ms", DEFAULT_DURATION_MS))
        self._power = int(state.get("power", DEFAULT_POWER))
        self._pump_percent = int(state.get("pump_percent", DEFAULT_PUMP_PERCENT))
