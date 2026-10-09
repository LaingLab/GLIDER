"""Catalog tools: what an agent may legally write into a .glider file."""

from pathlib import Path

import pytest

pytest.importorskip("mcp")

from glider.mcp import experiments  # noqa: E402
from glider.mcp.paths import writable_path  # noqa: E402


async def test_node_types_describe_ports():
    result = await experiments.list_node_types()
    by_name = {t["node_type"]: t for t in result["node_types"]}
    start = by_name["StartExperiment"]
    assert start["inputs"] == []
    assert start["outputs"][0] == {
        "index": 0,
        "name": "next",
        "kind": "exec",
        "data_type": "object",
        "description": "Triggers the next node",
    }
    delay = by_name["Delay"]
    assert [p["kind"] for p in delay["inputs"]] == ["exec", "data"]
    assert "exec" in result["notes"]


async def test_node_types_filter_by_category():
    result = await experiments.list_node_types(category="hardware")
    assert result["node_types"]
    assert {t["category"] for t in result["node_types"]} == {"hardware"}


async def test_device_types_list_required_pins_and_drivers():
    result = await experiments.list_device_types()
    by_name = {d["device_type"]: d for d in result["device_types"]}
    assert by_name["DigitalOutput"]["required_pins"] == ["output"]
    assert by_name["HX711"]["required_pins"] == ["dout", "sck"]
    assert {"arduino", "raspberry_pi"} <= set(result["board_drivers"])


async def test_plugin_driver_appears_in_catalog():
    pytest.importorskip("glider_harp")
    result = await experiments.list_device_types()
    assert "harp" in result["board_drivers"]


def test_writable_path_rules(tmp_path: Path):
    good = tmp_path / "a.glider"
    assert writable_path(str(good), ".glider") == good
    with pytest.raises(ValueError, match="absolute"):
        writable_path("a.glider", ".glider")
    with pytest.raises(ValueError, match=r"\.glider"):
        writable_path(str(tmp_path / "a.json"), ".glider")
    good.write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        writable_path(str(good), ".glider")
    assert writable_path(str(good), ".glider", overwrite=True) == good
    with pytest.raises(ValueError, match="does not exist"):
        writable_path(str(tmp_path / "missing" / "a.glider"), ".glider")
