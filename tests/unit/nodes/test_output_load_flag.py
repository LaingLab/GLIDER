"""D11: a saved Output value now outside its device's range is flagged at load."""

from __future__ import annotations

from types import SimpleNamespace

from glider.core.flow_engine import FlowEngine
from glider.hal.value_spec import KIND_WHOLE, ActionValueSpec
from glider.nodes.experiment_nodes import OutputNode


def _pwm(name, max_value):
    return SimpleNamespace(
        name=name,
        device_type="PWMOutput",
        value_spec=lambda a: ActionValueSpec(KIND_WHOLE, 0, max_value) if a == "set" else None,
    )


def test_output_flags_value_now_out_of_range():
    node = OutputNode()
    node._state = {"value": 4000}
    node._device = _pwm("pump", 100)
    warn = node.out_of_range_at_bind()
    assert warn is not None
    assert "4000" in warn and "0–100" in warn


def test_output_no_warning_when_in_range():
    node = OutputNode()
    node._state = {"value": 50}
    node._device = _pwm("pump", 100)
    assert node.out_of_range_at_bind() is None


def test_output_no_warning_without_device_or_value():
    node = OutputNode()
    node._device = None
    node._state = {"value": 4000}
    assert node.out_of_range_at_bind() is None
    node._device = _pwm("p", 100)
    node._state = {}
    assert node.out_of_range_at_bind() is None


def test_flow_engine_consume_load_warnings_returns_and_clears():
    fe = FlowEngine()
    fe._load_warnings = ["a", "b"]
    assert fe.consume_load_warnings() == ["a", "b"]
    assert fe.consume_load_warnings() == []
