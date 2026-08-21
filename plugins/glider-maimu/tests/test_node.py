"""Tests for MaimuNode.

The node's whole job is turning Mode / Period / Duration into the right device
action, so these check the dispatch, the exec handoff, and that a saved file
can't push an unknown mode through to the device.
"""

from __future__ import annotations

import pytest

from glider_maimu.node import MaimuNode


class _RecordingDevice:
    def __init__(self, error: Exception | None = None):
        self.calls: list[tuple] = []
        self.error = error

    async def execute_action(self, name, *args):
        self.calls.append((name, args))
        if self.error:
            raise self.error
        return None


def _node(mode="pulse", period_ms=500, duration_s=10, device=None):
    node = MaimuNode()
    node._device = _RecordingDevice() if device is None else device
    node.mode = mode
    node.period_ms = period_ms
    node.duration_s = duration_s
    return node


# --- definition ---------------------------------------------------------------


def test_node_is_named_maimu():
    assert MaimuNode.definition.name == "Maimu"


def test_has_no_data_ports():
    """Data input ports on an exec node can't receive values in GLIDER today,
    so shipping period/duration as ports would be shipping dead UI."""
    from glider.nodes.base_node import PortType

    inputs = MaimuNode.definition.inputs
    assert [port.port_type for port in inputs] == [PortType.EXEC]
    assert [port.port_type for port in MaimuNode.definition.outputs] == [PortType.EXEC]


def test_defaults_are_a_usable_pulse():
    node = MaimuNode()
    assert (node.mode, node.period_ms, node.duration_s) == ("pulse", 500, 10)


# --- dispatch -----------------------------------------------------------------


async def test_pulse_passes_period_and_duration():
    node = _node(mode="pulse", period_ms=250, duration_s=5)
    await node.hardware_operation()
    assert node._device.calls == [("pulse", (250, 5))]


async def test_on_calls_bare_action():
    node = _node(mode="on")
    await node.hardware_operation()
    assert node._device.calls == [("on", ())]


async def test_off_calls_bare_action():
    node = _node(mode="off")
    await node.hardware_operation()
    assert node._device.calls == [("off", ())]


def test_unknown_mode_is_rejected():
    node = MaimuNode()
    with pytest.raises(ValueError, match="mode"):
        node.mode = "strobe"


def test_mode_is_case_insensitive():
    node = MaimuNode()
    node.mode = "ON"
    assert node.mode == "on"


# --- exec handoff -------------------------------------------------------------


async def test_exec_output_fires_after_the_command():
    node = _node(mode="on")
    fired: list[tuple] = []
    node._update_callbacks.append(lambda name, value: fired.append((name, value)))

    await node.hardware_operation()

    assert node._device.calls == [("on", ())]
    assert fired == [("exec", True)]


async def test_exec_does_not_fire_when_the_write_fails():
    """A downstream step must not run as though the stimulator had been driven."""
    device = _RecordingDevice(error=RuntimeError("link down"))
    node = _node(mode="on", device=device)
    fired: list[tuple] = []
    node._update_callbacks.append(lambda name, value: fired.append((name, value)))

    await node.execute()  # HardwareNode.execute traps and records the error

    assert fired == []
    assert "link down" in (node.error or "")


async def test_unbound_device_sets_an_error_instead_of_raising():
    node = MaimuNode()
    await node.execute()
    assert node.error == "No device bound"


# --- state --------------------------------------------------------------------


def test_state_round_trips():
    node = _node(mode="pulse", period_ms=125, duration_s=30)
    restored = MaimuNode()
    restored.set_state(node.get_state())
    assert (restored.mode, restored.period_ms, restored.duration_s) == ("pulse", 125, 30)


def test_state_restores_on_and_off():
    for mode in ("on", "off"):
        restored = MaimuNode()
        restored.set_state(_node(mode=mode).get_state())
        assert restored.mode == mode


def test_unknown_saved_mode_falls_back_to_the_default():
    """A hand-edited or future-version file must not push an unknown action name
    through to the device."""
    restored = MaimuNode()
    restored.set_state({"mode": "strobe", "period_ms": 200, "duration_s": 2})
    assert restored.mode == "pulse"
    assert (restored.period_ms, restored.duration_s) == (200, 2)


def test_missing_state_keys_fall_back_to_defaults():
    restored = MaimuNode()
    restored.set_state({})
    assert (restored.mode, restored.period_ms, restored.duration_s) == ("pulse", 500, 10)


# --- registration -------------------------------------------------------------


def test_registered_with_the_flow_engine():
    from glider.core.flow_engine import FlowEngine
    from glider.nodes.hardware import register_hardware_nodes

    register_hardware_nodes(FlowEngine)
    assert FlowEngine.get_node_class("Maimu") is MaimuNode
