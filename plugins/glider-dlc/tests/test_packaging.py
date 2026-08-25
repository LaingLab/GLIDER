"""The installed plugin must be reachable through core, not just importable.

glider-maimu shipped once with class-style entry points, which PluginManager
registers under the *entry point's own name* while the module's own tables
never run. Everything logged "Successfully loaded" and nothing worked. This is
the equivalent check for glider-dlc: discover and load the way startup does,
then ask core -- not this package -- whether a DeepLabCut folder is recognised.

Requires the plugin to be installed (``uv pip install -e ./plugins/glider-dlc``),
which is what CI does and what the README tells a developer to do.
"""

from __future__ import annotations

import pytest

from glider.plugins.plugin_manager import PluginManager, plugin_components
from glider.vision.pose.converters import POSE_CONVERTERS, find_converter, needs_conversion
from glider_dlc.converter import DlcConverter

EXPECTED_NAME = "deeplabcut"


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
        distribution("glider-dlc")
    except PackageNotFoundError:
        pytest.skip("glider-dlc is not installed; run `uv pip install -e ./plugins/glider-dlc`")

    manager = PluginManager()
    await manager.discover_plugins()
    await manager.load_plugins()
    return manager


async def test_the_converter_registers(loaded):
    assert POSE_CONVERTERS.get(EXPECTED_NAME) is DlcConverter


async def test_no_stray_aliases(loaded):
    """``glider_dlc`` -- the entry point's own name -- is what a class-style
    entry point would have produced. A stray alias means the same converter is
    asked twice about every folder and reported as two plugins."""
    strays = [n for n in POSE_CONVERTERS if "dlc" in n.lower() and n != EXPECTED_NAME]

    assert strays == [], f"unexpected registrations: {strays}"


async def test_the_converter_is_attributed_to_this_plugin(loaded):
    assert plugin_components("pose").get(EXPECTED_NAME) == "glider_dlc"


async def test_core_recognises_a_deeplabcut_folder(loaded, dlc_dir):
    """The end of the chain: installing the plugin is the whole difference
    between core seeing a DeepLabCut model and core seeing a folder."""
    folder = dlc_dir()

    assert isinstance(find_converter(folder), DlcConverter)
    assert isinstance(needs_conversion(folder), DlcConverter)


async def test_two_pose_plugins_do_not_fight_over_one_folder(loaded, dlc_dir, tmp_path):
    """glider-sleap may well be installed alongside this. Each converter has to
    recognise only its own vendor's folder, or the wrong one runs."""
    sleap_like = tmp_path / "sleap"
    sleap_like.mkdir()
    (sleap_like / "training_config.json").write_text("{}")
    (sleap_like / "best_model.h5").write_bytes(b"keras")

    assert not isinstance(find_converter(sleap_like), DlcConverter)
    assert isinstance(find_converter(dlc_dir()), DlcConverter)
