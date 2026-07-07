# tests/unit/nodes/test_wait_for_input_behaviors.py

from glider.hal.input_behavior import InputBehavior
from glider.nodes.control_nodes import WaitForInputNode


class _AtLeast(InputBehavior):
    key = "at_least"
    label = "At least"
    settings = [{"key": "limit", "label": "Limit", "type": "int", "default": 5}]

    def check(self, value, settings, ctx):
        return value >= settings["limit"]


class _BehaviorDevice:
    device_type = "Fake"

    def __init__(self, values):
        self._values, self._i = list(values), 0

    @property
    def input_behaviors(self):
        return [_AtLeast()]

    async def read(self):
        v = self._values[self._i]
        if self._i < len(self._values) - 1:
            self._i += 1
        return v


def _fired(node):
    fired = []
    node._update_callbacks.append(lambda name, val: fired.append((name, val)) or None)
    return fired


async def test_new_path_fires_triggered_with_value():
    node = WaitForInputNode()
    node.bind_device(_BehaviorDevice([0, 3, 6, 9]))
    node.set_state(
        {
            "input_behavior": "at_least",
            "behavior_settings": {"at_least": {"limit": 5}},
            "poll_interval": 0.0,
        }
    )
    fired = _fired(node)
    await node.execute()
    assert ("triggered", True) in fired
    assert node._outputs[2] == 6  # value output port


async def test_new_path_fires_timeout():
    node = WaitForInputNode()
    node.bind_device(_BehaviorDevice([0, 1, 2]))
    node.set_state(
        {
            "input_behavior": "at_least",
            "behavior_settings": {"at_least": {"limit": 9}},
            "poll_interval": 0.0,
            "timeout": 0.15,
        }
    )
    fired = _fired(node)
    await node.execute()
    assert ("timeout", True) in fired


async def test_falls_back_to_legacy_when_no_behavior_selected():
    # No input_behavior in state -> legacy digital mode still works.
    node = WaitForInputNode()
    node.bind_device(_BehaviorDevice([False, True]))
    node.set_state({"threshold_mode": "digital", "poll_interval": 0.0})
    fired = _fired(node)
    await node.execute()
    assert ("triggered", True) in fired


async def test_unknown_behavior_key_falls_back_to_legacy():
    node = WaitForInputNode()
    node.bind_device(_BehaviorDevice([True]))
    node.set_state({"input_behavior": "nope", "threshold_mode": "digital", "poll_interval": 0.0})
    fired = _fired(node)
    await node.execute()
    assert ("triggered", True) in fired  # legacy path ran, not an error
