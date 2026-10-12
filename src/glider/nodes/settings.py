"""Settings schemas: one place for a setting's default and the values it accepts.

A schema is a list of fields -- ``key``, ``label``, ``type``, ``default`` and
optional ``choices``, ``min``, ``max``, ``help`` -- the format the properties
panel already renders for plugin nodes and device settings. Types: ``int``,
``hex``, ``float``, ``bool``, ``str``, ``enum``, ``device_ref``,
``ble_address`` and ``object`` (a JSON object). A field whose default is
``None`` also accepts ``None``.

Qt-free on purpose: nodes read their defaults from here at run time and the
MCP validator checks agent-written files against the same fields.
"""

from __future__ import annotations

import difflib
import math
from typing import Any

_WHOLE = ("int", "hex")
_NUMERIC = ("int", "hex", "float")
_TEXT = ("str", "device_ref", "ble_address")


def find_field(schema: list[dict] | None, key: str) -> dict | None:
    """The field declaring ``key``, or None."""
    return next((f for f in schema or [] if f.get("key") == key), None)


def check_value(field: dict, value: Any) -> str | None:
    """Why ``value`` is not acceptable for ``field``, or None if it is."""
    ftype = field.get("type", "str")
    if value is None:
        return None if "default" in field and field["default"] is None else "must not be null"
    if ftype in _WHOLE:
        whole = (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
        if not whole:
            return "must be a whole number"
    elif ftype == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "must be a number"
        if not math.isfinite(value):
            return "must be a finite number"
    elif ftype == "bool":
        if not isinstance(value, bool):
            return "must be true or false"
    elif ftype == "enum":
        allowed = [choice[0] for choice in field.get("choices", [])]
        if value not in allowed:
            return "must be one of " + ", ".join(repr(a) for a in allowed)
    elif ftype == "object":
        if not isinstance(value, dict):
            return "must be an object"
    elif ftype in _TEXT:
        if not isinstance(value, str):
            return "must be a string"
    if ftype in _NUMERIC:
        if "min" in field and value < field["min"]:
            return f"must be at least {field['min']}"
        if "max" in field and value > field["max"]:
            return f"must be at most {field['max']}"
    return None


def check_settings(
    schema: list[dict], values: dict
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Check saved settings against a schema: (errors, warnings) of (key, message, hint)."""
    errors: list[tuple[str, str, str]] = []
    warnings: list[tuple[str, str, str]] = []
    keys = [f["key"] for f in schema]
    for key, value in values.items():
        field = find_field(schema, key)
        if field is None:
            close = difflib.get_close_matches(str(key), keys, n=1)
            hint = (f"did you mean '{close[0]}'? " if close else "") + (
                "valid settings: " + ", ".join(keys)
            )
            warnings.append((key, f"unknown setting '{key}'", hint))
            continue
        problem = check_value(field, value)
        if problem:
            errors.append((key, f"'{key}' {problem}", field.get("help", "")))
    return errors, warnings
