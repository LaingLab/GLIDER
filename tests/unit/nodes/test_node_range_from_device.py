"""Hardware nodes read their range from the bound device (D-4), not a hardcode.

Before this change PWMWriteNode capped every write at 255; now a 12-bit PWM
device driven through the node reaches 4095.
"""

from __future__ import annotations

from glider.hal.value_spec import KIND_WHOLE, ActionValueSpec
from glider.nodes.hardware.analog_nodes import PWMWriteNode


class _RecordingPWM:
    device_type = "PWMOutput"

    def __init__(self, spec):
        self._spec = spec
        self.writes: list = []

    def value_spec(self, action):
        return self._spec if action == "set" else None

    async def execute_action(self, action, value):
        self.writes.append((action, value))


def _pwm_node(max_value, input_value):
    node = PWMWriteNode()
    node._device = _RecordingPWM(ActionValueSpec(KIND_WHOLE, 0, max_value))
    node.get_input = lambda i: input_value
    return node


async def test_pwm_write_reaches_12bit_range_not_capped_at_255():
    node = _pwm_node(max_value=4095, input_value=3000)
    await node.hardware_operation()
    assert node._device.writes == [("set", 3000)]  # would have been 255 before


async def test_pwm_write_clamps_to_the_devices_own_max():
    node = _pwm_node(max_value=4095, input_value=99999)
    await node.hardware_operation()
    assert node._device.writes == [("set", 4095)]
