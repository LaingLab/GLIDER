"""Node settings schemas: lookups, value checks, and GliderNode.setting()."""

import asyncio
import math
import re
from pathlib import Path

import pytest

import glider.nodes
from glider.core.flow_engine import FlowEngine
from glider.nodes.base_node import ExecNode, NodeCategory, NodeDefinition
from glider.nodes.logic.flow_nodes import TimerNode
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
HEX = {"key": "address", "type": "hex", "default": 0}
SERIAL_ENUM = {"key": "stopbits", "type": "enum", "default": 1, "choices": [[1, "1"], [2, "2"]]}
FLAG_ENUM = {
    "key": "flag",
    "type": "enum",
    "default": False,
    "choices": [[True, "on"], [False, "off"]],
}
STR = {"key": "label", "type": "str", "default": ""}
OBJ = {"key": "extra", "type": "object", "default": {}}


@pytest.mark.parametrize(
    "field, value",
    [
        (INT, 5),
        (FLOAT, 2),
        (FLOAT, 2.5),
        (NULLABLE, None),
        (NULLABLE, 3.0),
        (BOOL, True),
        (ENUM, "milliseconds"),
        (SERIAL_ENUM, 2),
        (FLAG_ENUM, True),
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
        (INT, 500.0, "whole number"),  # NumericInput rounds with round(value, n): a float breaks it
        (HEX, 2.0, "whole number"),
        (SERIAL_ENUM, True, "one of"),  # True == 1 would otherwise pass a numeric choice list
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


def test_setting_default_ignores_the_saved_value():
    node = _Node()
    node.set_state({"count": 7})
    assert node.setting_default("count") == 0
    assert node.setting("count") == 7


def test_setting_default_method_returns_a_copy():
    a, b = _Node(), _Node()
    a.setting_default("extra")["x"] = 1
    assert b.setting_default("extra") == {}


def test_setting_default_rejects_undeclared_keys():
    node = _Node()
    node.set_state({"typo": 1})
    with pytest.raises(KeyError, match="typo"):
        node.setting_default("typo")


def test_timer_with_a_null_interval_falls_back_to_the_default():
    timer = TimerNode()
    timer.set_state({"interval": None})
    assert timer._effective_interval() == 1.0


# Every built-in setting with the default the code used before schemas existed.
# A schema default or node.setting() that disagrees is a behaviour change.
PINNED = [
    ("Delay", "duration", 1.0),
    ("Delay", "unit", "seconds"),
    ("Delay", "count_from", "schedule"),
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
    ("FunctionCall", "function_start_id", None),
    ("PID", "integral", 0.0),
    ("PID", "last_error", 0.0),
    ("PID", "output_min", -255.0),
    ("PID", "output_max", 255.0),
    ("Toggle", "toggle_state", False),
]

PINNED += [
    ("AnalogRead", "pin", 0),
    ("AnalogRead", "reference_voltage", 5.0),
    ("AnalogRead", "resolution", 10),
    ("AnalogRead", "continuous", False),
    ("AnalogRead", "poll_interval", 0.05),
    ("AnalogRead", "threshold", 512),
    ("AnalogRead", "threshold_enabled", False),
    ("AnalogRead", "visible_in_runner", False),
    ("PWMWrite", "pin", 0),
    ("DigitalWrite", "pin", 0),
    ("DigitalRead", "pin", 0),
    ("DigitalRead", "continuous", False),
    ("DigitalRead", "poll_interval", 0.1),
    ("DeviceAction", "action_name", ""),
    ("DeviceAction", "arguments", ""),
    ("DeviceRead", "read_action", "read"),
    ("Label", "label", "Value"),
    ("Gauge", "label", "Value"),
    ("Gauge", "unit", ""),
    ("Gauge", "warning_threshold", None),
    ("Gauge", "danger_threshold", None),
    ("Chart", "label", "Chart"),
    ("Chart", "max_points", 100),
    ("Chart", "y_min", None),
    ("Chart", "y_max", None),
    ("LEDIndicator", "label", "LED"),
    ("LEDIndicator", "on_color", "#00ff00"),
    ("LEDIndicator", "off_color", "#333333"),
    ("Button", "label", "Button"),
    ("ToggleSwitch", "label", "Switch"),
    ("ToggleSwitch", "switch_state", False),
    ("Slider", "label", "Slider"),
    ("Slider", "slider_value", 0.0),
    ("Slider", "min_value", 0.0),
    ("Slider", "max_value", 100.0),
    ("Slider", "step", 1.0),
    ("NumericInput", "label", "Input"),
    ("NumericInput", "numeric_value", 0.0),
    ("NumericInput", "min_value", None),
    ("NumericInput", "max_value", None),
    ("NumericInput", "decimal_places", 2),
    ("AudioPlayback", "file_path", ""),
    ("AudioPlayback", "device_index", None),
    ("AudioPlayback", "device_name", ""),
    ("AudioPlayback", "volume", 1.0),
    ("VideoPlayback", "file_path", ""),
    ("VideoPlayback", "monitor_index", -1),
    ("ZoneInput", "zone_id", ""),
    ("ZoneInput", "zone_name", "Unnamed Zone"),
    ("BehaviorInput", "target_behavior", ""),
    ("BehaviorInput", "min_frames", 5),
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
    # == alone lets 1 pass for 1.0 and 0 for False; get_state writes these into saved files.
    assert field["default"] == default
    assert type(field["default"]) is type(default)
    node = cls()
    assert node.setting(key) == default
    assert type(node.setting(key)) is type(default)
    node.set_state({})
    state = node.get_state()
    if key in state:  # nodes that write attributes back into state
        assert state[key] == default
        assert type(state[key]) is type(default)


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


_READ = re.compile(r"""(?:_state|\bstate)\.(?:get|setdefault)\(\s*["']""")


def test_no_node_reads_a_setting_inline():
    """Every setting read goes through self.setting(), so defaults live in one place."""
    root = Path(glider.nodes.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for path in sorted(root.rglob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if _READ.search(line)
    ]
    assert offenders == [], "convert to self.setting(): " + ", ".join(offenders)


def test_gui_written_keys_are_declared(node_class):
    """Files the GUI saves must not raise unknown-setting warnings."""
    from glider.gui.panels import node_editor_controller as nec

    source = Path(nec.__file__).read_text()
    written = set(re.findall(r'_on_node_property_changed\(\s*\w+\s*,\s*"(\w+)"', source))
    # Keys a node is created with: initial_state["key"] = ... and initial_state = {"key": ...}.
    written |= set(re.findall(r'initial_state\["(\w+)"\]', source))
    for literal in re.findall(r"initial_state\s*=\s*\{([^}]+)\}", source):
        written |= set(re.findall(r'"(\w+)"\s*:', literal))
    declared = {
        f["key"]
        for name in FlowEngine.get_available_nodes()
        for f in getattr(node_class(name), "PROPERTIES_SCHEMA", [])
    }
    assert written - declared == set()
