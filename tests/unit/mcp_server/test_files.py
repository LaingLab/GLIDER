"""new/read/save: the files an agent writes open in GLIDER like any other."""

import json

import pytest

pytest.importorskip("mcp")

from glider.core.glider_core import GliderCore  # noqa: E402
from glider.mcp import experiments  # noqa: E402


async def test_new_experiment_is_valid_and_refuses_to_overwrite(tmp_path):
    path = tmp_path / "blink.glider"
    result = experiments.new_experiment(str(path), "Blink", "toggle an LED")
    assert path.exists()
    assert result["experiment"]["metadata"]["name"] == "Blink"
    assert (await experiments.validate_experiment(path=str(path)))["valid"]
    with pytest.raises(ValueError, match="already exists"):
        experiments.new_experiment(str(path), "Again")


def test_read_experiment_round_trips(tmp_path):
    path = tmp_path / "a.glider"
    experiments.new_experiment(str(path), "A")
    result = experiments.read_experiment(str(path))
    assert result["experiment"] == json.loads(path.read_text())
    assert "2 nodes" in result["summary"]
    with pytest.raises(ValueError, match=r"\.glider"):
        experiments.read_experiment(str(tmp_path / "a.json"))


async def test_agent_edit_saves_and_opens_like_the_gui(tmp_path):
    path = tmp_path / "delay.glider"
    data = experiments.new_experiment(str(path), "Delay")["experiment"]
    flow = data["flow"]
    flow["connections"] = []
    flow["nodes"].append({"id": "wait", "node_type": "Delay", "state": {}})  # no position
    for i, (a, b) in enumerate([("start", "wait"), ("wait", "end")]):
        flow["connections"].append(
            {
                "id": f"c{i}",
                "from_node": a,
                "from_output": 0,
                "to_node": b,
                "to_input": 0,
                "connection_type": "exec",
            }
        )
    result = await experiments.save_experiment(str(path), data, overwrite=True)
    assert result["saved"], result

    saved = json.loads(path.read_text())
    positions = {n["id"]: n["position"] for n in saved["flow"]["nodes"]}
    # depth 1 from start is x=250, but End already sits at (250, 0): next row down
    assert positions["wait"] == [250.0, 150.0]

    core = GliderCore()
    await core.initialize(load_plugins=False)
    await core.load_experiment(path)
    assert core.flow_engine.load_failures == []
    assert set(core.flow_engine.nodes) == {"start", "wait", "end"}


async def test_save_refuses_invalid_and_existing(tmp_path):
    path = tmp_path / "x.glider"
    bad = {"flow": {"nodes": [], "connections": []}}
    result = await experiments.save_experiment(str(path), bad)
    assert result["saved"] is False
    assert not path.exists()
    assert result["errors"]

    experiments.new_experiment(str(path), "X")
    good = experiments.read_experiment(str(path))["experiment"]
    with pytest.raises(ValueError, match="overwrite"):
        await experiments.save_experiment(str(path), good)
    with pytest.raises(ValueError, match=r"\.glider"):
        await experiments.save_experiment(str(tmp_path / "x.json"), good)


def test_layout_survives_cycles():
    nodes = [{"id": "a"}, {"id": "b"}]
    conns = [{"from_node": "a", "to_node": "b"}, {"from_node": "b", "to_node": "a"}]
    experiments._layout(nodes, conns)
    assert all("position" in n for n in nodes)


async def test_save_refuses_bad_position_without_writing(tmp_path):
    path = tmp_path / "p.glider"
    experiments.new_experiment(str(path), "P")
    data = experiments.read_experiment(str(path))["experiment"]
    data["flow"]["nodes"][0]["position"] = "abc"
    data["flow"]["nodes"].append({"id": "wait", "node_type": "Delay", "state": {}})
    other = tmp_path / "q.glider"
    result = await experiments.save_experiment(str(other), data)
    assert result["saved"] is False
    assert result["errors"][0]["path"] == "flow.nodes[0].position"
    assert not other.exists()


def test_read_experiment_rejects_relative_path_and_bad_json(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        experiments.read_experiment("a.glider")
    bad = tmp_path / "bad.glider"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="line 1"):
        experiments.read_experiment(str(bad))


async def test_save_keeps_the_vision_block(tmp_path):
    from tests.unit.mcp_server.test_validate import VALID

    data = {
        **VALID,
        "vision": {"backend": "POSE_MODEL", "model_path": "/nonexistent/weights.pt"},
    }
    path = tmp_path / "v.glider"
    result = await experiments.save_experiment(str(path), data)
    assert result["saved"], result
    assert json.loads(path.read_text())["vision"]["model_path"] == "/nonexistent/weights.pt"
