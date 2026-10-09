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
