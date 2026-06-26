"""
Tests for DeviceActionNode's constant-arguments field.

The flow engine doesn't deliver wired values into a node's action arg ports,
so DeviceActionNode falls back to a comma-separated "arguments" string typed in
the node properties (e.g. a BLE write of "on" / "off" / "20,10").
"""

from __future__ import annotations

from glider.nodes.hardware.device_nodes import DeviceActionNode


class _RecordingDevice:
    def __init__(self):
        self.calls: list[tuple] = []

    async def execute_action(self, name, *args):
        self.calls.append((name, args))
        return "ok"


def _node(action="write", arguments=""):
    node = DeviceActionNode()
    node._device = _RecordingDevice()
    node.action_name = action
    node.arguments = arguments
    return node


async def test_numeric_args_split_and_coerced():
    node = _node(arguments="20,10")
    await node.hardware_operation()
    assert node._device.calls == [("write", (20, 10))]


async def test_string_command_argument():
    node = _node(arguments="on")
    await node.hardware_operation()
    assert node._device.calls == [("write", ("on",))]


async def test_mixed_args_keep_type():
    node = _node(arguments="1.5, off")
    await node.hardware_operation()
    assert node._device.calls == [("write", (1.5, "off"))]


async def test_wired_inputs_take_precedence_over_constant():
    node = _node(arguments="on")
    node.set_input(1, "wired")
    await node.hardware_operation()
    assert node._device.calls == [("write", ("wired",))]


async def test_no_args_calls_action_bare():
    node = _node(arguments="")
    await node.hardware_operation()
    assert node._device.calls == [("write", ())]


def test_arguments_state_round_trip():
    node = DeviceActionNode()
    node.set_state({"action_name": "write", "arguments": "20,10"})
    assert node.arguments == "20,10"
    assert node.action_name == "write"
    state = node.get_state()
    assert state["arguments"] == "20,10"
    assert state["action_name"] == "write"
