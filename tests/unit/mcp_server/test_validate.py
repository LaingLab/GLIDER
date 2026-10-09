"""validate_experiment: catches what would stop GLIDER opening or starting a file."""

import copy

import pytest

pytest.importorskip("mcp")

from glider.core.glider_core import GliderCore  # noqa: E402
from glider.mcp import experiments  # noqa: E402

VALID = {
    "metadata": {"name": "LED blink"},
    "hardware": {
        "boards": [{"id": "uno", "driver_type": "arduino", "board_type": "uno"}],
        "devices": [
            {
                "id": "led",
                "device_type": "DigitalOutput",
                "name": "LED",
                "board_id": "uno",
                "pins": {"output": 13},
            }
        ],
    },
    "flow": {
        "nodes": [
            {"id": "start", "node_type": "StartExperiment", "position": [0, 0]},
            {"id": "on", "node_type": "Output", "position": [250, 0], "device_id": "led"},
            {"id": "end", "node_type": "EndExperiment", "position": [500, 0]},
        ],
        "connections": [
            {
                "id": "c1",
                "from_node": "start",
                "from_output": 0,
                "to_node": "on",
                "to_input": 0,
                "connection_type": "exec",
            },
            {
                "id": "c2",
                "from_node": "on",
                "from_output": 0,
                "to_node": "end",
                "to_input": 0,
                "connection_type": "exec",
            },
        ],
    },
}


def broken(edit):
    data = copy.deepcopy(VALID)
    edit(data)
    return data


def error_paths(report):
    return [e["path"] for e in report["errors"]]


async def test_valid_experiment_passes():
    report = await experiments.validate(VALID)
    assert report["valid"], report
    assert report["errors"] == []


@pytest.mark.parametrize(
    "edit, path",
    [
        (lambda d: d["flow"]["nodes"][1].update(node_type="Blink"), "flow.nodes[1].node_type"),
        (lambda d: d["flow"]["connections"][0].update(to_input=5), "flow.connections[0].to_input"),
        (
            lambda d: d["flow"]["connections"][0].update(from_node="nope"),
            "flow.connections[0].from_node",
        ),
        (
            lambda d: d["flow"]["connections"][0].update(connection_type="data"),
            "flow.connections[0].connection_type",
        ),
        (lambda d: d["flow"]["nodes"][1].update(device_id="ghost"), "flow.nodes[1].device_id"),
        (
            lambda d: d["hardware"]["devices"][0].update(board_id="mega"),
            "hardware.devices[0].board_id",
        ),
        (
            lambda d: d["hardware"]["devices"][0].update(device_type="Laser"),
            "hardware.devices[0].device_type",
        ),
        (
            lambda d: d["hardware"]["boards"][0].update(driver_type="esp32"),
            "hardware.boards[0].driver_type",
        ),
        (lambda d: d["flow"]["nodes"].append(dict(d["flow"]["nodes"][2])), "flow.nodes[3].id"),
        (lambda d: d["flow"]["nodes"].pop(0) and d["flow"]["connections"].pop(0), "flow.nodes"),
        (lambda d: d["flow"]["connections"][0].pop("to_node"), "flow.connections[0]"),
    ],
)
async def test_each_rule_reports_its_path(edit, path):
    report = await experiments.validate(broken(edit))
    assert not report["valid"]
    assert path in error_paths(report), report["errors"]
    assert all(e["message"] for e in report["errors"])


async def test_duplicate_pin_caught_by_real_load():
    def edit(d):
        twin = dict(d["hardware"]["devices"][0], id="led2", name="LED 2")
        d["hardware"]["devices"].append(twin)

    report = await experiments.validate(broken(edit))
    assert not report["valid"]
    assert any("led2" in e["message"] for e in report["errors"]), report["errors"]


async def test_missing_required_pin_caught_by_real_load():
    report = await experiments.validate(
        broken(lambda d: d["hardware"]["devices"][0].update(pins={}))
    )
    assert not report["valid"]
    assert any("led" in e["message"] for e in report["errors"]), report["errors"]


async def test_warnings_do_not_invalidate():
    report = await experiments.validate(
        broken(
            lambda d: d["flow"]["nodes"].append(
                {"id": "lonely", "node_type": "Delay", "position": [0, 200]}
            )
        )
    )
    assert report["valid"]
    assert any(w["path"] == "flow.nodes[3]" for w in report["warnings"])


async def test_legacy_format_is_rejected_with_hint():
    legacy = {
        "flow": {"nodes": [{"id": "a", "type": "x", "title": "x", "position": {"x": 0, "y": 0}}]}
    }
    report = await experiments.validate(legacy)
    assert not report["valid"]
    assert "re-save" in report["errors"][0]["hint"]


async def test_validate_experiment_reads_path_or_content(tmp_path):
    import json

    path = tmp_path / "x.glider"
    path.write_text(json.dumps(VALID))
    assert (await experiments.validate_experiment(path=str(path)))["valid"]
    assert (await experiments.validate_experiment(content=json.dumps(VALID)))["valid"]
    with pytest.raises(ValueError, match="exactly one"):
        await experiments.validate_experiment()
    with pytest.raises(ValueError, match="line 1"):
        await experiments.validate_experiment(content="{nope")


async def test_validation_never_touches_hardware(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError("validation must not connect hardware")

    monkeypatch.setattr(GliderCore, "connect_hardware", forbidden)
    monkeypatch.setattr(GliderCore, "setup_hardware", forbidden)
    assert (await experiments.validate(VALID))["valid"]
    core = await experiments.get_core()
    assert not any(b.is_connected for b in core.hardware_manager.boards.values())
