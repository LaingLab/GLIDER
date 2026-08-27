"""The installed plugin must be reachable through core, not just importable.

The equivalent of glider-sleap's suite, and it carries one extra burden: two
SLEAP plugins now exist, and the whole design depends on them staying distinct.
If either shadowed the other, or if both claimed one folder, the operator would
get whichever the registry happened to iterate first.

Requires the plugin to be installed (``uv pip install -e ./plugins/glider-sleap-nn``),
which is what CI does.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from glider_sleap_nn.converter import SleapNNConverter

from glider.plugins.plugin_manager import PluginManager, plugin_components
from glider.vision.pose.converters import POSE_CONVERTERS, find_converter, needs_conversion

EXPECTED_NAME = "sleap_nn"


@pytest.fixture
async def loaded():
    """Discover and load plugins the way the application does at startup."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("glider-sleap-nn")
    except PackageNotFoundError:
        pytest.skip("not installed; run `uv pip install -e ./plugins/glider-sleap-nn`")

    manager = PluginManager()
    await manager.discover_plugins()
    await manager.load_plugins()
    return manager


def _sleap_nn_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "training_config.yaml").write_text("model_config: {}\n")
    (d / "best.ckpt").write_bytes(b"PK\x03\x04not really a checkpoint")
    return d


def _classic_sleap_dir(tmp_path: Path) -> Path:
    d = tmp_path / "classic"
    d.mkdir()
    (d / "training_config.json").write_text("{}")
    (d / "best_model.h5").write_bytes(b"not really a model")
    return d


async def test_the_converter_registers(loaded):
    assert POSE_CONVERTERS.get(EXPECTED_NAME) is SleapNNConverter


async def test_no_stray_aliases(loaded):
    strays = [
        name
        for name, cls in POSE_CONVERTERS.items()
        if cls is SleapNNConverter and name != EXPECTED_NAME
    ]
    assert strays == [], f"SleapNNConverter registered under extra names: {strays}"


async def test_the_converter_is_attributed_to_this_plugin(loaded):
    """Provenance is what names the plugin in the conversion prompt."""
    assert plugin_components("pose").get(EXPECTED_NAME) == "glider_sleap_nn"


async def test_core_recognises_a_sleap_nn_folder(loaded, tmp_path):
    """The end of the chain: installing the plugin is the whole difference
    between core seeing a sleap-nn folder and core seeing a folder."""
    folder = _sleap_nn_dir(tmp_path)

    assert isinstance(find_converter(folder), SleapNNConverter)
    assert isinstance(needs_conversion(folder), SleapNNConverter)


async def test_the_two_sleap_plugins_do_not_claim_each_other(loaded, tmp_path):
    """The point of the split.

    `find_converter` warns and picks the first when two converters claim one
    folder, so an overlap here would silently hand a PyTorch checkpoint to the
    TensorFlow converter on whichever machine iterated the registry the other
    way round.
    """
    assert not SleapNNConverter().claims(_classic_sleap_dir(tmp_path))

    classic = POSE_CONVERTERS.get("sleap")
    if classic is None:
        pytest.skip("glider-sleap is not installed alongside")
    assert not classic().claims(_sleap_nn_dir(tmp_path))
