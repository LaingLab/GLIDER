"""Maimu Node

Drives a Maimu BLE stimulator from the flow graph without the researcher having
to know the wire protocol. The generic Device Action node can do the same job --
action ``write``, argument ``"500,10"`` -- but only if you remember both. This
node offers Mode / Period / Duration instead.
"""

from typing import Any

from glider.nodes.base_node import (
    HardwareNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

# Mode -> the device action it calls.
MODE_ON = "on"
MODE_OFF = "off"
MODE_PULSE = "pulse"
MODES = (MODE_ON, MODE_OFF, MODE_PULSE)

DEFAULT_PERIOD_MS = 500
DEFAULT_DURATION_S = 10


class MaimuNode(HardwareNode):
    """Send one Maimu command: on, off, or a timed pulse.

    Period and duration come from the node's properties, not from input ports.
    That is deliberate: a data input port on an exec-driven node cannot receive
    a value in GLIDER today -- ``FlowEngine._propagate_execution`` calls
    ``execute()`` on the target and never resolves its data connections, and
    ``GliderNode.get_input`` reads a plain list that is only ever populated with
    port defaults. (``DeviceActionNode`` documents the same limitation and falls
    back to typed properties for the same reason.) Ports that render as wireable
    and then silently do nothing would be worse than no ports at all.

    Pulse is fire-and-continue: ``exec`` out fires as soon as the write lands
    and the stimulator runs the pattern on its own. That is what ``exec`` out
    means everywhere else in GLIDER -- "the command was sent". Put a Delay node
    after this one to hold the flow for the duration.
    """

    # Rendered by the properties panel via schema_form. The hardcoded
    # Mode/Period/Duration widgets this replaces could grey out period and
    # duration outside Pulse mode; a declared schema has no notion of one field
    # depending on another, so that cue is carried in the help text instead.
    PROPERTIES_SCHEMA = [
        {
            "key": "mode",
            "label": "Mode",
            "type": "enum",
            "default": MODE_PULSE,
            "choices": [["on", "On"], ["off", "Off"], ["pulse", "Pulse"]],
            "help": "On and Off latch; Pulse runs a train and stops on its own.",
        },
        {
            "key": "period_ms",
            "label": "Period (ms)",
            "type": "int",
            "default": DEFAULT_PERIOD_MS,
            "min": 1,
            "max": 3_600_000,
            "help": (
                "Pulse mode only. On/off toggle period in milliseconds -- a "
                "period, not a frequency. 500 ms toggles about once a second."
            ),
        },
        {
            "key": "duration_s",
            "label": "Duration (s)",
            "type": "int",
            "default": DEFAULT_DURATION_S,
            "min": 1,
            "max": 86_400,
            "help": (
                "Pulse mode only. How long the train runs. The stimulator "
                "times this itself, so the flow continues immediately -- add a "
                "Delay node to hold it."
            ),
        },
    ]

    definition = NodeDefinition(
        name="Maimu",
        category=NodeCategory.HARDWARE,
        description="Drive a Maimu BLE stimulator: on, off, or a timed pulse",
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
        self._mode = MODE_PULSE
        self._period_ms = DEFAULT_PERIOD_MS
        self._duration_s = DEFAULT_DURATION_S

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        text = str(value).strip().lower()
        if text not in MODES:
            raise ValueError(f"Maimu mode must be one of {MODES}, got {value!r}")
        self._mode = text

    @property
    def period_ms(self) -> int:
        return self._period_ms

    @period_ms.setter
    def period_ms(self, value: Any) -> None:
        self._period_ms = int(value)

    @property
    def duration_s(self) -> int:
        return self._duration_s

    @duration_s.setter
    def duration_s(self, value: Any) -> None:
        self._duration_s = int(value)

    async def hardware_operation(self) -> None:
        """Send the configured command, then trigger the exec output."""
        if self._mode == MODE_PULSE:
            await self._device.execute_action(MODE_PULSE, self._period_ms, self._duration_s)
        else:
            await self._device.execute_action(self._mode)

        await self._fire_exec_output("exec")

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["mode"] = self._mode
        state["period_ms"] = self._period_ms
        state["duration_s"] = self._duration_s
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        # A saved file is not a trusted source of a valid mode -- a hand-edited
        # or future-version value falls back to the default rather than sending
        # an action name the device has never heard of.
        mode = str(state.get("mode", MODE_PULSE)).strip().lower()
        self._mode = mode if mode in MODES else MODE_PULSE
        self._period_ms = int(state.get("period_ms", DEFAULT_PERIOD_MS))
        self._duration_s = int(state.get("duration_s", DEFAULT_DURATION_S))
