"""Node settings schemas: lookups, value checks, and GliderNode.setting()."""

import math

import pytest

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
