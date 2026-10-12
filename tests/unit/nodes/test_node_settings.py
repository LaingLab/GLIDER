"""Node settings schemas: lookups, value checks, and GliderNode.setting()."""

import asyncio
import math

import pytest

from glider.core.flow_engine import FlowEngine
from glider.nodes.base_node import ExecNode, NodeCategory, NodeDefinition
from glider.nodes.settings import check_settings, check_value, find_field

INT = {"key": "count", "type": "int", "default": 0, "min": 0}
FLOAT = {"key": "delay", "type": "float", "default": 1.0, "min": 0, "max": 10}
NULLABLE = {"key": "limit", "type": "float", "default": None}
BOOL = {"key": "on", "type": "bool", "default": False}
ENUM = {
    "key": "unit",
    "type": "enum",
    "default": "seconds",
    "choices": [["seconds", "sec"], ["milliseconds", "ms"]],
}
STR = {"key": "label", "type": "str", "default": ""}
OBJ = {"key": "extra", "type": "object", "default": {}}


@pytest.mark.parametrize(
    "field, value",
    [
        (INT, 5),
        (INT, 500.0),
        (FLOAT, 2),
        (FLOAT, 2.5),
        (NULLABLE, None),
        (NULLABLE, 3.0),
        (BOOL, True),
        (ENUM, "milliseconds"),
        (STR, "x"),
        (OBJ, {"a": 1}),
    ],
)
def test_acceptable_values(field, value):
    assert check_value(field, value) is None


@pytest.mark.parametrize(
    "field, value, fragment",
    [
        (INT, 2.5, "whole number"),
        (INT, True, "whole number"),
        (INT, -1, "at least 0"),
        (FLOAT, "500", "number"),
        (FLOAT, 11, "at most 10"),
        (FLOAT, math.nan, "finite"),
        (FLOAT, None, "null"),
        (BOOL, "yes", "true or false"),
        (ENUM, "msec", "'seconds'"),
        (STR, 3, "string"),
        (OBJ, [], "object"),
    ],
)
def test_unacceptable_values(field, value, fragment):
    problem = check_value(field, value)
    assert problem is not None and fragment in problem


def test_check_settings_splits_errors_and_warnings():
    errors, warnings = check_settings([INT, ENUM], {"count": -2, "unti": "seconds"})
    assert [e[0] for e in errors] == ["count"]
    assert warnings[0][0] == "unti"
    assert "did you mean 'unit'" in warnings[0][2]


def test_find_field():
    assert find_field([INT, ENUM], "unit") is ENUM
    assert find_field([INT], "nope") is None
    assert find_field(None, "x") is None


class _Node(ExecNode):
    definition = NodeDefinition(name="Probe", category=NodeCategory.LOGIC)
    PROPERTIES_SCHEMA = [INT, OBJ]

    async def execute(self) -> None:
        pass


def test_setting_returns_saved_value_or_default():
    node = _Node()
    assert node.setting("count") == 0
    node.set_state({"count": 7})
    assert node.setting("count") == 7


def test_setting_default_is_a_copy():
    a, b = _Node(), _Node()
    a.setting("extra")["x"] = 1
    assert b.setting("extra") == {}


def test_setting_rejects_undeclared_keys():
    node = _Node()
    node.set_state({"typo": 1})
    with pytest.raises(KeyError, match="typo"):
        node.setting("typo")


# Every built-in setting with the default the code used before schemas existed.
# A schema default or node.setting() that disagrees is a behaviour change.
PINNED = [
    ("Delay", "duration", 1.0),
    ("Delay", "unit", "seconds"),
    ("Timer", "interval", 1.0),
    ("Timer", "unit", "seconds"),
    ("Timer", "count", 0),
    ("Loop", "count", 0),
    ("Loop", "delay", 1.0),
    ("Loop", "prompt_count", False),
    ("WaitForInput", "input_behavior", None),
    ("WaitForInput", "timeout", 0.0),
    ("WaitForInput", "poll_interval", 0.05),
    ("WaitForInput", "behavior_settings", {}),
    ("WaitForInput", "threshold_mode", "digital"),
    ("WaitForInput", "threshold", 512),
    ("WaitForInput", "threshold_direction", "above"),
    ("WaitForInput", "turns_target", 1),
    ("WaitForInput", "counts_per_turn", 4096),
    ("WaitForInput", "counts_target", 400),
    ("WaitForInput", "ramp_down", False),
    ("WaitForInput", "ramp_device_id", None),
    ("WaitForInput", "drive_pwm", 100),
    ("WaitForInput", "creep_pwm", 30),
    ("WaitForInput", "ramp_zone", 512),
    ("WaitForInput", "land_tolerance", 0),
    ("Output", "value", 1),
    ("MotorGovernor", "action", "stop"),
    ("StartFunction", "function_name", "MyFunction"),
    ("FunctionCall", "function_name", "Unknown"),
    ("PID", "integral", 0.0),
    ("PID", "last_error", 0.0),
    ("PID", "output_min", -255.0),
    ("PID", "output_max", 255.0),
    ("Toggle", "toggle_state", False),
]

FIELD_TYPES = {"int", "hex", "float", "bool", "str", "enum", "device_ref", "ble_address", "object"}


@pytest.fixture(scope="module")
def node_class():
    from glider.core.glider_core import GliderCore

    asyncio.run(GliderCore().initialize(load_plugins=False))
    return FlowEngine.get_node_class


@pytest.mark.parametrize("node_type, key, default", PINNED, ids=lambda v: str(v))
def test_pinned_default(node_class, node_type, key, default):
    cls = node_class(node_type)
    field = find_field(cls.PROPERTIES_SCHEMA, key)
    assert field is not None, f"{node_type} does not declare '{key}'"
    assert field["default"] == default
    node = cls()
    assert node.setting(key) == default
    node.set_state({})
    state = node.get_state()
    if key in state:  # nodes that write attributes back into state
        assert state[key] == default


def _schema_classes(node_class):
    names = sorted({t for t, _, _ in PINNED})
    return [(n, node_class(n)) for n in names]


def test_schemas_are_well_formed(node_class):
    for name, cls in _schema_classes(node_class):
        keys = [f["key"] for f in cls.PROPERTIES_SCHEMA]
        assert len(keys) == len(set(keys)), f"{name} has duplicate keys"
        for f in cls.PROPERTIES_SCHEMA:
            for attr in ("key", "label", "type", "default"):
                assert attr in f, f"{name}.{f.get('key')} lacks {attr}"
            assert f["type"] in FIELD_TYPES, f"{name}.{f['key']} has unknown type {f['type']!r}"
            if f["type"] == "enum":
                assert f.get("choices"), f"{name}.{f['key']} enum has no choices"
            if f["default"] is not None:
                assert check_value(f, f["default"]) is None, f"{name}.{f['key']} default invalid"


def test_every_declared_setting_is_pinned(node_class):
    pinned = {(t, k) for t, k, _ in PINNED}
    for name, cls in _schema_classes(node_class):
        for f in cls.PROPERTIES_SCHEMA:
            assert (name, f["key"]) in pinned, f"{name}.{f['key']} has no pinned default"
