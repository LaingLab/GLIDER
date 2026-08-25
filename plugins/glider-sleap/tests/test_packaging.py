"""The installed plugin must be reachable through core, not just importable.

glider-maimu shipped once with class-style entry points, which PluginManager
registers under the *entry point's own name* while the module's own tables
never run. Everything logged "Successfully loaded" and nothing worked. This
suite is the equivalent check for the pose group: discover and load the way
startup does, then ask core -- not this package -- whether a SLEAP folder is
recognised.

Requires the plugin to be installed (``uv pip install -e ./plugins/glider-sleap``),
which is what CI does and what the README tells a developer to do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glider.plugins.plugin_manager import PluginManager, plugin_components
from glider.vision.pose.converters import POSE_CONVERTERS, find_converter, needs_conversion
from glider_sleap.converter import SleapConverter

EXPECTED_NAME = "sleap"


@pytest.fixture
async def loaded():
    """Discover and load plugins the way the application does at startup.

    The skip guard checks that the *distribution* is installed, deliberately
    not that a plugin of some expected name was discovered. Keying it on the
    name would make these tests skip -- silently, and therefore green -- in
    exactly the case they exist to catch.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("glider-sleap")
    except PackageNotFoundError:
        pytest.skip("glider-sleap is not installed; run `uv pip install -e ./plugins/glider-sleap`")

    manager = PluginManager()
    await manager.discover_plugins()
    await manager.load_plugins()
    return manager


def _sleap_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "training_config.json").write_text(json.dumps({"model": {"heads": {}}}))
    (d / "best_model.h5").write_bytes(b"not really a model")
    return d


async def test_the_converter_registers(loaded):
    assert POSE_CONVERTERS.get(EXPECTED_NAME) is SleapConverter


async def test_no_stray_aliases(loaded):
    """``glider_sleap`` -- the entry point's own name -- is what a class-style
    entry point would have produced. A stray alias means the same converter is
    asked twice about every folder and reported as two plugins."""
    strays = [n for n in POSE_CONVERTERS if "sleap" in n.lower() and n != EXPECTED_NAME]

    assert strays == [], f"unexpected registrations: {strays}"


async def test_the_converter_is_attributed_to_this_plugin(loaded):
    """Provenance is what names the plugin in the conversion prompt."""
    assert plugin_components("pose").get(EXPECTED_NAME) == "glider_sleap"


async def test_core_recognises_a_sleap_folder_once_the_plugin_is_installed(loaded, tmp_path):
    """The end of the chain: installing the plugin is the whole difference
    between core seeing a SLEAP folder and core seeing a folder."""
    folder = _sleap_dir(tmp_path)

    assert isinstance(find_converter(folder), SleapConverter)
    assert isinstance(needs_conversion(folder), SleapConverter)


async def test_core_leaves_other_vendors_alone(loaded, tmp_path):
    (tmp_path / "best.pt").write_bytes(b"yolo")

    assert find_converter(tmp_path) is None
