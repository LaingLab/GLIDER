"""Maimu Node

Drives a Maimu BLE stimulator from the flow graph without the researcher having
to know the wire protocol. The generic Device Action node can do the same job --
action ``write``, argument ``"50,5,20,100"`` -- but only if you remember all
four fields in order. This node offers Mode / Period / Pulse width / Pulses /
Intensity instead.
"""

import logging
from typing import Any

from glider.nodes.base_node import (
    HardwareNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)

# Mode -> the device action it calls.
MODE_ON = "on"
MODE_OFF = "off"
MODE_PULSE = "pulse"
MODES = (MODE_ON, MODE_OFF, MODE_PULSE)

DEFAULT_PERIOD_MS = 50
DEFAULT_PULSE_WIDTH_MS = 5
DEFAULT_COUNT = 20
DEFAULT_INTENSITY_PCT = 100


class MaimuNode(HardwareNode):
    """Send one Maimu command: on, off, or a timed pulse.

    Period, pulse width, count and intensity come from the node's properties,
    not from input ports. That is deliberate: a data input port on an
    exec-driven node cannot receive a value in GLIDER today --
    ``FlowEngine._propagate_execution`` calls ``execute()`` on the target and
    never resolves its data connections, and ``GliderNode.get_input`` reads a
    plain list that is only ever populated with port defaults.
    (``DeviceActionNode`` documents the same limitation and falls back to
    typed properties for the same reason.) Ports that render as wireable and
    then silently do nothing would be worse than no ports at all.

    Pulse is fire-and-continue: ``exec`` out fires as soon as the write lands
    and the stimulator runs the pattern on its own. That is what ``exec`` out
    means everywhere else in GLIDER -- "the command was sent". Put a Delay node
    after this one to hold the flow for the duration of the train.
    """

    # Rendered by the properties panel via schema_form. The hardcoded
    # Mode/Period/Pulse width/Pulses/Intensity widgets this replaces could grey
    # out the pulse fields outside Pulse mode; a declared schema has no notion
    # of one field depending on another, so that cue is carried in the help
    # text instead. Every key/default/min/max below matches
    # MaimuDevice.ACTION_ARGS_SCHEMA["pulse"] exactly -- test_pulse_schema.py
    # asserts they agree.
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
                "Pulse mode only. Full cycle period in milliseconds -- a "
                "period, not a frequency. 50 ms is 20 Hz."
            ),
        },
        {
            "key": "pulse_width_ms",
            "label": "Pulse width (ms)",
            "type": "int",
            "default": DEFAULT_PULSE_WIDTH_MS,
            "min": 1,
            "max": 3_600_000,
            "help": (
                "Pulse mode only. On-time within each cycle; cannot exceed "
                "the period. Equal to the period means continuous light."
            ),
        },
        {
            "key": "count",
            "label": "Pulses",
            "type": "int",
            "default": DEFAULT_COUNT,
            "min": 0,
            "max": 65_535,
            "help": (
                "Pulse mode only. How many pulses to deliver. 0 runs until "
                "stopped -- use an Off node or Emergency Stop to end it."
            ),
        },
        {
            "key": "intensity_pct",
            "label": "Intensity (%)",
            "type": "int",
            "default": DEFAULT_INTENSITY_PCT,
            "min": 0,
            "max": 100,
            "help": (
                "Pulse mode only. Relative light output, not calibrated "
                "optical power. Per-unit brightness is trimmed in firmware."
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
        self._pulse_width_ms = DEFAULT_PULSE_WIDTH_MS
        self._count = DEFAULT_COUNT
        self._intensity_pct = DEFAULT_INTENSITY_PCT

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
    def pulse_width_ms(self) -> int:
        return self._pulse_width_ms

    @pulse_width_ms.setter
    def pulse_width_ms(self, value: Any) -> None:
        self._pulse_width_ms = int(value)

    @property
    def count(self) -> int:
        return self._count

    @count.setter
    def count(self, value: Any) -> None:
        self._count = int(value)

    @property
    def intensity_pct(self) -> int:
        return self._intensity_pct

    @intensity_pct.setter
    def intensity_pct(self, value: Any) -> None:
        self._intensity_pct = int(value)

    async def hardware_operation(self) -> None:
        """Send the configured command, then trigger the exec output."""
        if self._mode == MODE_PULSE:
            await self._device.execute_action(
                MODE_PULSE,
                self._period_ms,
                self._pulse_width_ms,
                self._count,
                self._intensity_pct,
            )
        else:
            await self._device.execute_action(self._mode)

        await self._fire_exec_output("exec")

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["mode"] = self._mode
        state["period_ms"] = self._period_ms
        state["pulse_width_ms"] = self._pulse_width_ms
        state["count"] = self._count
        state["intensity_pct"] = self._intensity_pct
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        # A saved file is not a trusted source of a valid mode -- a hand-edited
        # or future-version value falls back to the default rather than sending
        # an action name the device has never heard of.
        mode = str(state.get("mode", MODE_PULSE)).strip().lower()
        self._mode = mode if mode in MODES else MODE_PULSE
        self._period_ms = int(state.get("period_ms", DEFAULT_PERIOD_MS))
        self._pulse_width_ms = int(state.get("pulse_width_ms", DEFAULT_PULSE_WIDTH_MS))
        self._count = int(state.get("count", DEFAULT_COUNT))
        self._intensity_pct = int(state.get("intensity_pct", DEFAULT_INTENSITY_PCT))

        # Graphs saved before the four-field grammar carry duration_s and no
        # width, count or intensity. They load -- but they load as a *different*
        # stimulus, and a researcher who is not told will find that out from
        # data that does not look right.
        if "duration_s" in state:
            logger.warning(
                "Maimu node %s: this graph was saved with the old duration-based "
                "grammar. 'duration_s'=%s was dropped; the train now ends after "
                "Pulses (%s) instead. Check the stimulus before running.",
                getattr(self, "node_id", "?"),
                state["duration_s"],
                self._count,
            )
