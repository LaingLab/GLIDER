"""Experiment design tools for the GLIDER MCP server.

Agents edit ``.glider`` files as whole JSON documents in the session format,
the one File > Save writes. The catalog comes from the live registries, so
plugin drivers and devices are included. Nothing here connects a board.
"""

from __future__ import annotations

import json
from typing import Any

from glider.core.flow_engine import FlowEngine
from glider.core.glider_core import GliderCore
from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import DEVICE_REGISTRY, create_device_from_dict
from glider.hal.mock_board import MockBoard

_core: GliderCore | None = None


async def get_core() -> GliderCore:
    """The server's one headless core, initialized on first use.

    Lazy so ``tools/list`` answers instantly; plugins load here because their
    node types, drivers and devices must be in the catalog and pass validation.
    """
    global _core
    if _core is None:
        core = GliderCore()
        await core.initialize(load_plugins=True)
        _core = core
    return _core


def _type_name(t: Any) -> str:
    return getattr(t, "__name__", str(t))


def _ports(ports: list) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "name": p.name,
            "kind": p.port_type.name.lower(),
            "data_type": _type_name(p.data_type),
            "description": p.description,
        }
        for i, p in enumerate(ports)
    ]


async def list_node_types(category: str | None = None) -> dict[str, Any]:
    """Every node type an experiment's flow may use, with its ports.

    Args:
        category: Optional filter: "hardware", "logic", "interface", ...
    """
    await get_core()
    types = []
    for name in sorted(FlowEngine.get_available_nodes()):
        cls = FlowEngine.get_node_class(name)
        definition = getattr(cls, "definition", None)
        if definition is None:  # a registered class with no node definition
            continue
        if category and definition.category.value != category:
            continue
        try:
            defaults = cls().get_state()
        except Exception:
            defaults = {}
        types.append(
            {
                "node_type": name,
                "category": definition.category.value,
                "description": definition.description,
                "inputs": _ports(definition.inputs),
                "outputs": _ports(definition.outputs),
                "default_state": json.loads(json.dumps(defaults, default=str)),
            }
        )
    return {
        "summary": f"{len(types)} node types",
        "notes": (
            "A connection joins a source node's output index to a target node's "
            "input index. exec ports carry control flow and connect only to exec "
            "ports with connection_type 'exec'; data ports connect only to data "
            "ports with connection_type 'data'. A flow starts at StartExperiment. "
            "Hardware nodes are bound to a device with device_id."
        ),
        "node_types": types,
    }


async def list_device_types() -> dict[str, Any]:
    """Device types (with the pin names each needs) and board drivers."""
    await get_core()
    devices = []
    for name in sorted(DEVICE_REGISTRY):
        try:
            probe = create_device_from_dict(
                {
                    "id": "probe",
                    "device_type": name,
                    "name": name,
                    "board_id": "probe",
                    "config": {"pins": {}, "settings": {}},
                },
                MockBoard(),
            )
            pins: list[str] | None = list(probe.required_pins)
        except Exception:
            pins = None
        devices.append({"device_type": name, "required_pins": pins})
    drivers = sorted(HardwareManager.get_available_drivers())
    return {
        "summary": f"{len(devices)} device types, {len(drivers)} board drivers",
        "device_types": devices,
        "board_drivers": drivers,
    }
