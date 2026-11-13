"""Tests for VapePodNode: Mode / Duration / Power into the right device action."""

from __future__ import annotations

import pytest

from glider_vapepod.node import VapePodNode


class _RecordingDevice:
    def __init__(self, error: Exception | None = None):
        self.calls: list[tuple] = []
        self.error = error

    async def execute_action(self, name, *args):
        self.calls.append((name, args))
        if self.error:
            raise self.error
        return None


def _node(mode="puff", device=None, **props):
    node = VapePodNode()
    node._device = _RecordingDevice() if device is None else device
    node.mode = mode
    for key, value in props.items():
        setattr(node, key, value)
    return node


def test_node_is_named_vapepod():
    assert VapePodNode.definition.name == "VapePod"


def test_has_only_exec_ports():
    from glider.nodes.base_node import PortType

    assert [p.port_type for p in VapePodNode.definition.inputs] == [PortType.EXEC]
    assert [p.port_type for p in VapePodNode.definition.outputs] == [PortType.EXEC]


def test_defaults_are_a_usable_puff():
    node = VapePodNode()
    assert (node.mode, node.duration_ms, node.power) == ("puff", 3000, 50)


@pytest.mark.parametrize(
    "mode, props, expected",
    [
        ("puff", {"duration_ms": 2500, "power": 15}, ("puff", (2500, 15))),
        ("pump_on", {"pump_percent": 80}, ("pump", (80,))),
        ("pump_off", {}, ("pump_off", ())),
        ("off", {}, ("off", ())),
    ],
)
async def test_each_mode_calls_its_action(mode, props, expected):
    node = _node(mode, **props)
    await node.hardware_operation()
    assert node._device.calls == [expected]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        VapePodNode().mode = "vape_forever"


def test_mode_is_case_insensitive():
    node = VapePodNode()
    node.mode = "PUMP_ON"
    assert node.mode == "pump_on"


async def test_exec_output_fires_after_the_command():
    node = _node("puff")
    fired: list[tuple] = []
    node._update_callbacks.append(lambda name, value: fired.append((name, value)))
    await node.hardware_operation()
    assert fired == [("exec", True)]


async def test_exec_does_not_fire_when_the_puff_is_not_delivered():
    """A step after a puff must not run as though the animal had been dosed."""
    node = _node("puff", device=_RecordingDevice(error=ConnectionError("link is down")))
    fired: list[tuple] = []
    node._update_callbacks.append(lambda name, value: fired.append((name, value)))

    await node.execute()  # HardwareNode.execute traps and records the error

    assert fired == []
    assert "link is down" in (node.error or "")


async def test_unbound_device_sets_an_error_instead_of_raising():
    node = VapePodNode()
    await node.execute()
    assert node.error == "No device bound"


def test_state_round_trips():
    node = _node("pump_on", duration_ms=1200, power=60, pump_percent=45)
    restored = VapePodNode()
    restored.set_state(node.get_state())
    assert (restored.mode, restored.duration_ms, restored.power, restored.pump_percent) == (
        "pump_on",
        1200,
        60,
        45,
    )


def test_unknown_saved_mode_falls_back_to_puff():
    restored = VapePodNode()
    restored.set_state({"mode": "strobe", "duration_ms": 500})
    assert (restored.mode, restored.duration_ms) == ("puff", 500)


def test_registered_with_the_flow_engine():
    from glider.core.flow_engine import FlowEngine

    assert FlowEngine.get_node_class("VapePod") is VapePodNode
