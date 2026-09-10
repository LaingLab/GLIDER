"""Tests for MaimuNode.

The node's whole job is turning Mode / Period / Pulse width / Pulses /
Intensity into the right device action, so these check the dispatch, the exec
handoff, and that a saved file can't push an unknown mode -- or a silently
different stimulus -- through to the device.
"""

from __future__ import annotations

import pytest

from glider_maimu.node import (
    DEFAULT_COUNT,
    DEFAULT_INTENSITY_PCT,
    DEFAULT_PERIOD_MS,
    DEFAULT_PULSE_WIDTH_MS,
    MaimuNode,
)


class _RecordingDevice:
    def __init__(self, error: Exception | None = None):
        self.calls: list[tuple] = []
        self.error = error

    async def execute_action(self, name, *args):
        self.calls.append((name, args))
        if self.error:
            raise self.error
        return None


def _node(
    mode="pulse",
    period_ms=DEFAULT_PERIOD_MS,
    pulse_width_ms=DEFAULT_PULSE_WIDTH_MS,
    count=DEFAULT_COUNT,
    intensity_pct=DEFAULT_INTENSITY_PCT,
    device=None,
):
    node = MaimuNode()
    node._device = _RecordingDevice() if device is None else device
    node.mode = mode
    node.period_ms = period_ms
    node.pulse_width_ms = pulse_width_ms
    node.count = count
    node.intensity_pct = intensity_pct
    return node


# --- definition ---------------------------------------------------------------


def test_node_is_named_maimu():
    assert MaimuNode.definition.name == "Maimu"


def test_has_no_data_ports():
    """Data input ports on an exec node can't receive values in GLIDER today,
    so shipping the pulse fields as ports would be shipping dead UI."""
    from glider.nodes.base_node import PortType

    inputs = MaimuNode.definition.inputs
    assert [port.port_type for port in inputs] == [PortType.EXEC]
    assert [port.port_type for port in MaimuNode.definition.outputs] == [PortType.EXEC]


def test_defaults_are_a_usable_pulse():
    node = MaimuNode()
    assert (
        node.mode,
        node.period_ms,
        node.pulse_width_ms,
        node.count,
        node.intensity_pct,
    ) == ("pulse", DEFAULT_PERIOD_MS, DEFAULT_PULSE_WIDTH_MS, DEFAULT_COUNT, DEFAULT_INTENSITY_PCT)


# --- dispatch -----------------------------------------------------------------


async def test_pulse_passes_all_four_arguments_in_order():
    node = _node(mode="pulse", period_ms=25, pulse_width_ms=3, count=7, intensity_pct=80)
    await node.hardware_operation()
    assert node._device.calls == [("pulse", (25, 3, 7, 80))]


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
    node = _node(mode="pulse", period_ms=125, pulse_width_ms=10, count=30, intensity_pct=75)
    restored = MaimuNode()
    restored.set_state(node.get_state())
    assert (
        restored.mode,
        restored.period_ms,
        restored.pulse_width_ms,
        restored.count,
        restored.intensity_pct,
    ) == ("pulse", 125, 10, 30, 75)


def test_state_restores_on_and_off():
    for mode in ("on", "off"):
        restored = MaimuNode()
        restored.set_state(_node(mode=mode).get_state())
        assert restored.mode == mode


def test_unknown_saved_mode_falls_back_to_the_default():
    """A hand-edited or future-version file must not push an unknown action name
    through to the device."""
    restored = MaimuNode()
    restored.set_state(
        {
            "mode": "strobe",
            "period_ms": 200,
            "pulse_width_ms": 8,
            "count": 2,
            "intensity_pct": 50,
        }
    )
    assert restored.mode == "pulse"
    assert (
        restored.period_ms,
        restored.pulse_width_ms,
        restored.count,
        restored.intensity_pct,
    ) == (
        200,
        8,
        2,
        50,
    )


def test_missing_state_keys_fall_back_to_defaults():
    restored = MaimuNode()
    restored.set_state({})
    assert (
        restored.mode,
        restored.period_ms,
        restored.pulse_width_ms,
        restored.count,
        restored.intensity_pct,
    ) == ("pulse", DEFAULT_PERIOD_MS, DEFAULT_PULSE_WIDTH_MS, DEFAULT_COUNT, DEFAULT_INTENSITY_PCT)


def test_an_old_graph_loads_and_says_what_it_dropped(caplog):
    """Graphs saved before the four-field grammar carry duration_s. They must
    load rather than crash -- but silently becoming a different stimulus is
    worse than not loading at all, so it is reported."""
    import logging

    from glider_maimu.node import DEFAULT_COUNT, DEFAULT_PULSE_WIDTH_MS, MaimuNode

    node = MaimuNode()
    with caplog.at_level(logging.WARNING, logger="glider_maimu.node"):
        node.set_state({"mode": "pulse", "period_ms": 500, "duration_s": 10})

    assert node.period_ms == 500
    assert node.pulse_width_ms == DEFAULT_PULSE_WIDTH_MS
    assert node.count == DEFAULT_COUNT
    assert any("duration_s" in r.message for r in caplog.records)


def test_a_current_graph_loads_silently(caplog):
    import logging

    from glider_maimu.node import MaimuNode

    node = MaimuNode()
    with caplog.at_level(logging.WARNING, logger="glider_maimu.node"):
        node.set_state(
            {
                "mode": "pulse",
                "period_ms": 50,
                "pulse_width_ms": 4,
                "count": 4,
                "intensity_pct": 80,
            }
        )

    assert (node.period_ms, node.pulse_width_ms, node.count, node.intensity_pct) == (
        50,
        4,
        4,
        80,
    )
    assert caplog.records == []


# --- registration -------------------------------------------------------------


def test_registered_with_the_flow_engine():
    from glider.core.flow_engine import FlowEngine
    from glider.nodes.hardware import register_hardware_nodes

    register_hardware_nodes(FlowEngine)
    assert FlowEngine.get_node_class("Maimu") is MaimuNode
