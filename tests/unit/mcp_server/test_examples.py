"""Every shipped example must validate: a rule that rejects a real lab file is wrong."""

from pathlib import Path

import pytest

pytest.importorskip("mcp")

from glider.mcp import experiments  # noqa: E402

EXAMPLES = sorted((Path(__file__).parents[3] / "examples").glob("*.glider"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
async def test_example_validates(path):
    report = await experiments.validate_experiment(path=str(path))
    assert report["errors"] == [], report["errors"]


def test_examples_found():
    assert len(EXAMPLES) >= 5


def test_list_examples_lists_every_shipped_example():
    result = experiments.list_examples()
    names = {e["name"] for e in result["examples"]}
    assert names == {p.stem for p in EXAMPLES}
    dispense = next(e for e in result["examples"] if e["name"] == "dispense")
    assert "DigitalOutput" in dispense["devices"]
    assert "StartExperiment" in dispense["node_types"]
