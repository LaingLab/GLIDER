"""Structural checks: content that would be dropped, misspelled fields, settings.

Each catches a way an agent's experiment loses content without GLIDER saying
so: a top-level ``nodes`` list, a node's settings under ``settings``, a field
named ``type`` where ``node_type`` belongs, a setting typed ``"msec"``.
"""

from __future__ import annotations

import difflib
from typing import Any

from glider.nodes.settings import check_settings

#: Keys an ExperimentSession file may have at the top level.
TOP_LEVEL = frozenset(
    {"metadata", "hardware", "flow", "dashboard", "camera", "zones", "manual_controls", "vision"}
)
#: Sections agents put at the top level that belong one level down.
_MISPLACED = {
    "nodes": "flow.nodes",
    "connections": "flow.connections",
    "boards": "hardware.boards",
    "devices": "hardware.devices",
}
#: Fields each list item may have (ExperimentSession's to_dict).
FIELDS = {
    "hardware.boards": frozenset(
        {"id", "driver_type", "port", "board_type", "auto_reconnect", "settings"}
    ),
    "hardware.devices": frozenset({"id", "device_type", "name", "board_id", "pins", "settings"}),
    "flow.nodes": frozenset(
        {"id", "node_type", "position", "state", "device_id", "visible_in_runner"}
    ),
    "flow.connections": frozenset(
        {"id", "from_node", "from_output", "to_node", "to_input", "connection_type"}
    ),
}
#: Names agents use for a required field.
_ALIASES = {
    "node_type": ("type", "nodeType", "kind"),
    "driver_type": ("type", "driver"),
    "device_type": ("type",),
    "from_output": ("from_port", "from_index", "source_port"),
    "to_input": ("to_port", "to_index", "target_port"),
    "from_node": ("source", "from"),
    "to_node": ("target", "to"),
}
#: Names agents use for a node's state; ignored by GLIDER, so an error.
_STATE_ALIASES = ("settings", "properties", "params", "config")


def finding(path: str, message: str, hint: str = "") -> dict[str, str]:
    return {"path": path, "message": message, "hint": hint}


def check_top_level(data: dict, errors: list, warnings: list) -> None:
    for key in data:
        if key in _MISPLACED:
            errors.append(
                finding(
                    key,
                    f"'{key}' belongs under {_MISPLACED[key]}",
                    f"move it there; a top-level '{key}' is ignored",
                )
            )
        elif key not in TOP_LEVEL:
            close = difflib.get_close_matches(str(key), sorted(TOP_LEVEL), n=1)
            warnings.append(
                finding(
                    key,
                    f"unknown top-level key '{key}'",
                    f"did you mean '{close[0]}'?" if close else "GLIDER ignores it",
                )
            )


def did_you_mean(missing: str, item: dict, known: frozenset) -> str:
    """A suggestion for a missing field, from the item's keys that are not fields themselves.

    A key that is a legitimate field of the item kind is never offered: a
    missing ``to_node`` must not be answered with "found 'from_node'".
    """
    extra = [k for k in item if isinstance(k, str) and k not in known]
    present = [k for k in _ALIASES.get(missing, ()) if k in extra]
    if not present:
        present = difflib.get_close_matches(missing, extra, n=1)
    return f" (found '{present[0]}'; did you mean '{missing}'?)" if present else ""


def check_item_fields(dotted: str, path: str, item: dict, errors: list, warnings: list) -> None:
    known = FIELDS[dotted]
    for key in item:
        if key in known:
            continue
        if dotted == "flow.nodes" and key in _STATE_ALIASES and "state" not in item:
            errors.append(
                finding(
                    f"{path}.{key}",
                    "node settings belong in 'state'",
                    f"rename '{key}' to 'state'; GLIDER ignores '{key}'",
                )
            )
            continue
        close = difflib.get_close_matches(str(key), sorted(known), n=1)
        warnings.append(
            finding(
                f"{path}.{key}",
                f"unknown field '{key}'",
                f"did you mean '{close[0]}'?" if close else "known: " + ", ".join(sorted(known)),
            )
        )


def check_settings_at(
    path: str, schema: list[dict], values: Any, errors: list, warnings: list
) -> None:
    """Check one node's state or device's settings; `path` ends in .state/.settings."""
    if not isinstance(values, dict):
        errors.append(finding(path, "must be an object"))
        return
    bad, unknown = check_settings(schema, values)
    errors.extend(finding(f"{path}.{key}", msg, hint) for key, msg, hint in bad)
    warnings.extend(finding(f"{path}.{key}", msg, hint) for key, msg, hint in unknown)
