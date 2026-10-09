"""Experiment design tools for the GLIDER MCP server.

Agents edit ``.glider`` files as whole JSON documents in the session format,
the one File > Save writes. The catalog comes from the live registries, so
plugin drivers and devices are included. Nothing here connects a board.
"""

from __future__ import annotations

import copy
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from glider.core.experiment_session import ConnectionConfig, ExperimentSession, NodeConfig
from glider.core.flow_engine import FlowEngine
from glider.core.glider_core import GliderCore
from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import DEVICE_REGISTRY, create_device_from_dict
from glider.hal.mock_board import MockBoard
from glider.mcp.paths import writable_path
from glider.serialization import is_schema_format

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


# Fields each list item must have before anything else can be checked.
_REQUIRED = {
    "hardware.boards": ("id", "driver_type"),
    "hardware.devices": ("id", "device_type", "name", "board_id", "pins"),
    "flow.nodes": ("id", "node_type"),
    "flow.connections": ("id", "from_node", "from_output", "to_node", "to_input"),
}


# Required fields that must be strings (they are hashed and compared by id).
_STRINGS = {
    "hardware.boards": ("id", "driver_type"),
    "hardware.devices": ("id", "device_type", "name", "board_id"),
    "flow.nodes": ("id", "node_type"),
    "flow.connections": ("id", "from_node", "to_node"),
}


def _finding(path: str, message: str, hint: str = "") -> dict[str, str]:
    return {"path": path, "message": message, "hint": hint}


def _result(errors: list, warnings: list) -> dict[str, Any]:
    if errors:
        summary = f"invalid: {len(errors)} error(s), {len(warnings)} warning(s)"
    else:
        summary = f"valid ({len(warnings)} warning(s))"
    return {"valid": not errors, "errors": errors, "warnings": warnings, "summary": summary}


def _parse(content: dict | str) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON at line {e.lineno}, column {e.colno}: {e.msg}") from e


def _items(data: dict, dotted: str, errors: list) -> list[tuple[str, dict]]:
    """(json path, item) for every item of one list that has its required fields."""
    section, key = dotted.split(".")
    body = data.get(section)
    if body is not None and not isinstance(body, dict):
        errors.append(_finding(section, f"{section} must be an object"))
        return []
    raw = (body or {}).get(key, [])
    if not isinstance(raw, list):
        errors.append(_finding(dotted, f"{dotted} must be a list"))
        return []
    good = []
    for i, item in enumerate(raw):
        path = f"{dotted}[{i}]"
        if not isinstance(item, dict):
            errors.append(_finding(path, "must be an object"))
            continue
        missing = [f for f in _REQUIRED[dotted] if f not in item]
        if missing:
            errors.append(_finding(path, f"missing {', '.join(missing)}"))
            continue
        bad = [f for f in _STRINGS[dotted] if not isinstance(item[f], str)]
        for f in bad:
            errors.append(_finding(f"{path}.{f}", "must be a string"))
        if dotted == "hardware.devices" and not isinstance(item["pins"], dict):
            bad.append("pins")
            errors.append(_finding(f"{path}.pins", "must be an object"))
        if dotted == "flow.nodes" and not isinstance(item.get("device_id", ""), (str, type(None))):
            bad.append("device_id")
            errors.append(_finding(f"{path}.device_id", "must be a string"))
        if bad:
            continue
        good.append((path, item))
    return good


def _check_port(path, index, ports, owner, direction, errors) -> bool:
    if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(ports):
        return True
    have = f"{direction}s 0..{len(ports) - 1}" if ports else f"no {direction}s"
    errors.append(_finding(path, f"{owner} has {have}", "list_node_types shows each node's ports"))
    return False


def _check_graph(data: dict, errors: list, warnings: list) -> None:
    boards = _items(data, "hardware.boards", errors)
    devices = _items(data, "hardware.devices", errors)
    nodes = _items(data, "flow.nodes", errors)
    connections = _items(data, "flow.connections", errors)

    drivers = set(HardwareManager.get_available_drivers())
    board_ids = {b["id"] for _, b in boards}
    for path, b in boards:
        if b["driver_type"] not in drivers:
            errors.append(
                _finding(
                    f"{path}.driver_type",
                    f"unknown board driver '{b['driver_type']}'",
                    f"one of: {', '.join(sorted(drivers))}",
                )
            )

    device_ids = {d["id"] for _, d in devices}
    for path, d in devices:
        if d["device_type"] not in DEVICE_REGISTRY:
            errors.append(
                _finding(
                    f"{path}.device_type",
                    f"unknown device type '{d['device_type']}'",
                    "list_device_types shows the available types",
                )
            )
        if d["board_id"] not in board_ids:
            errors.append(_finding(f"{path}.board_id", f"no board with id '{d['board_id']}'"))

    by_id: dict[str, dict] = {}
    for path, n in nodes:
        if n["id"] in by_id:
            errors.append(_finding(f"{path}.id", f"duplicate node id '{n['id']}'"))
        by_id[n["id"]] = n
        if FlowEngine.get_node_class(n["node_type"]) is None:
            errors.append(
                _finding(
                    f"{path}.node_type",
                    f"unknown node type '{n['node_type']}'",
                    "list_node_types shows the available types",
                )
            )
        device_id = n.get("device_id")
        if device_id is not None and device_id not in device_ids:
            errors.append(_finding(f"{path}.device_id", f"no device with id '{device_id}'"))

    types = {n["node_type"] for n in by_id.values()}
    if "StartExperiment" not in types:
        errors.append(
            _finding("flow.nodes", "no StartExperiment node", "the flow starts there; add one")
        )
    if "EndExperiment" not in types:
        warnings.append(_finding("flow.nodes", "no EndExperiment node; the run never ends itself"))

    connected: set[str] = set()
    for path, c in connections:
        src, dst = by_id.get(c["from_node"]), by_id.get(c["to_node"])
        if src is None:
            errors.append(_finding(f"{path}.from_node", f"no node with id '{c['from_node']}'"))
        if dst is None:
            errors.append(_finding(f"{path}.to_node", f"no node with id '{c['to_node']}'"))
        if src is None or dst is None:
            continue
        connected |= {src["id"], dst["id"]}
        src_cls = FlowEngine.get_node_class(src["node_type"])
        dst_cls = FlowEngine.get_node_class(dst["node_type"])
        if src_cls is None or dst_cls is None:
            continue
        outs, ins = src_cls.definition.outputs, dst_cls.definition.inputs
        ok_out = _check_port(
            f"{path}.from_output", c["from_output"], outs, src["node_type"], "output", errors
        )
        ok_in = _check_port(
            f"{path}.to_input", c["to_input"], ins, dst["node_type"], "input", errors
        )
        if not (ok_out and ok_in):
            continue
        kind = outs[c["from_output"]].port_type.name.lower()
        other = ins[c["to_input"]].port_type.name.lower()
        if kind != other:
            errors.append(_finding(path, f"cannot connect a {kind} output to a {other} input"))
        elif c.get("connection_type", "data") != kind:
            errors.append(
                _finding(
                    f"{path}.connection_type",
                    f"must be '{kind}' for {kind} ports",
                    (
                        "exec connections that are typed 'data' never fire"
                        if kind == "exec"
                        else "set connection_type to match the ports"
                    ),
                )
            )

    for path, n in nodes:
        if n["id"] not in connected:
            warnings.append(_finding(path, f"node '{n['id']}' is not connected to anything"))
    bound = {n.get("device_id") for n in by_id.values()}
    for path, d in devices:
        if d["id"] not in bound:
            warnings.append(_finding(path, f"no node is bound to device '{d['id']}'"))


class _ErrorCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


async def _real_load(data: dict) -> list[dict[str, str]]:
    """Open the experiment exactly as File > Open does, minus any connection.

    populate_hardware_from_session builds drivers and devices without opening
    a port, and only logs a board or device it could not build -- so those log
    records are captured. load_failures is the list the app's Start refuses on.
    """
    core = await get_core()
    capture = _ErrorCapture()
    core_log = logging.getLogger("glider.core.glider_core")
    core_log.addHandler(capture)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate.glider"
            candidate.write_text(json.dumps(data), encoding="utf-8")
            core.load_session(str(candidate))
            core.populate_hardware_from_session()
            core.setup_flow()
    except Exception as e:
        return [_finding("$", f"GLIDER could not load this experiment: {e}")]
    finally:
        core_log.removeHandler(capture)
    problems = dict.fromkeys(capture.messages + core.flow_engine.load_failures)
    return [
        _finding("$", m, "reported by GLIDER's own loader; the app would refuse to start this")
        for m in problems
    ]


async def validate(data: Any) -> dict[str, Any]:
    """Validate an experiment dict. Graph checks first; the real load only if they pass."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if not isinstance(data, dict):
        return _result([_finding("$", "an experiment must be a JSON object")], warnings)
    if is_schema_format(data):
        errors.append(
            _finding(
                "$",
                "this file is in GLIDER's legacy serializer format",
                "open it in GLIDER and re-save it to convert it, or use the session "
                "format: nodes have 'node_type', boards have 'driver_type'",
            )
        )
        return _result(errors, warnings)
    await get_core()
    _check_graph(data, errors, warnings)
    if not errors:
        errors.extend(await _real_load(data))
    return _result(errors, warnings)


async def validate_experiment(
    path: str | None = None, content: dict | str | None = None
) -> dict[str, Any]:
    """Check an experiment for anything that would stop GLIDER opening or starting it.

    Pass exactly one of ``path`` (an existing .glider file) or ``content`` (the
    experiment JSON). Returns ``valid``, plus ``errors`` and ``warnings``, each
    with a JSON ``path``, a ``message`` and a ``hint``. Never touches hardware.
    """
    if (path is None) == (content is None):
        raise ValueError("pass exactly one of path or content")
    if path is not None:
        content = Path(path).read_text(encoding="utf-8")
    return await validate(_parse(content))


_COLUMN_WIDTH = 250.0
_ROW_HEIGHT = 150.0


def _describe(data: dict) -> str:
    flow, hardware = data.get("flow") or {}, data.get("hardware") or {}
    name = (data.get("metadata") or {}).get("name", "experiment")
    return (
        f"{name}: {len(flow.get('nodes') or [])} nodes, "
        f"{len(flow.get('connections') or [])} connections, "
        f"{len(hardware.get('devices') or [])} devices"
    )


def new_experiment(path: str, name: str, description: str = "") -> dict[str, Any]:
    """Create a minimal valid experiment (Start -> End) to build on.

    Args:
        path: Absolute path ending in .glider. Must not exist yet.
        name: Experiment name.
        description: Optional description.
    """
    target = writable_path(path, ".glider")
    session = ExperimentSession()
    session.name = name
    session.metadata.description = description
    session.add_node(NodeConfig(id="start", node_type="StartExperiment", position=(0.0, 0.0)))
    session.add_node(NodeConfig(id="end", node_type="EndExperiment", position=(_COLUMN_WIDTH, 0.0)))
    session.add_connection(
        ConnectionConfig(
            id="start_to_end",
            from_node="start",
            from_output=0,
            to_node="end",
            to_input=0,
            connection_type="exec",
        )
    )
    session.save(str(target))
    data = session.to_dict()
    return {"path": str(target), "experiment": data, "summary": f"created {_describe(data)}"}


def read_experiment(path: str) -> dict[str, Any]:
    """Return an experiment file's JSON so it can be edited and saved back."""
    p = Path(path).expanduser()
    if p.suffix.lower() != ".glider":
        raise ValueError(f"{p.name} is not a .glider file")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {"path": str(p), "experiment": data, "summary": _describe(data)}


def _layout(nodes: list[dict], connections: list[dict]) -> None:
    """Put nodes that have no position in a column by their depth from the start.

    So an agent's experiment opens readable in the node graph instead of every
    new node stacked at the origin.
    """
    if all("position" in n for n in nodes):
        return
    depth = {n["id"]: 0 for n in nodes}
    limit = len(nodes)
    # ponytail: repeated relaxation, O(nodes * connections); bounded so Loop cycles terminate
    for _ in range(limit):
        changed = False
        for c in connections:
            a, b = c.get("from_node"), c.get("to_node")
            if a in depth and b in depth and depth[a] + 1 < limit and depth[b] < depth[a] + 1:
                depth[b] = depth[a] + 1
                changed = True
        if not changed:
            break
    taken = {tuple(float(v) for v in n["position"]) for n in nodes if "position" in n}
    for n in nodes:
        if "position" in n:
            continue
        x, row = depth[n["id"]] * _COLUMN_WIDTH, 0
        while (x, row * _ROW_HEIGHT) in taken:
            row += 1
        n["position"] = [x, row * _ROW_HEIGHT]
        taken.add((x, row * _ROW_HEIGHT))


async def save_experiment(
    path: str, content: dict | str, overwrite: bool = False
) -> dict[str, Any]:
    """Validate an experiment and write it as a .glider file GLIDER can open.

    Refuses to write anything that fails validate_experiment. Nodes without a
    "position" are laid out left to right by flow depth.

    Args:
        path: Absolute path ending in .glider.
        content: The experiment JSON (object or string).
        overwrite: Replace an existing file. Off by default.
    """
    target = writable_path(path, ".glider", overwrite=overwrite)
    data = _parse(content)
    report = await validate(data)
    if not report["valid"]:
        return {"saved": False, **report, "summary": f"not saved; {report['summary']}"}
    data = copy.deepcopy(data)
    flow = data.setdefault("flow", {})
    _layout(flow.setdefault("nodes", []), flow.get("connections") or [])
    session = ExperimentSession.from_dict(data)
    session.metadata.modified_at = datetime.now().isoformat()
    session.save(str(target))
    return {
        "saved": True,
        "path": str(target),
        "warnings": report["warnings"],
        "summary": f"saved {_describe(data)} to {target.name}",
    }
