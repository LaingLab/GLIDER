"""The installed plugin must register under the names experiments already use.

This is the test that would have caught it. The plugin originally declared
class-style entry points::

    [project.entry-points."glider.device"]
    maimu = "glider_maimu.device:MaimuDevice"

PluginManager registers an entry point that names a class under the *entry
point's own name*, not under any name the class picks -- so that produced a
device type called ``maimu`` and a node called ``maimu_node``, while the
``DEVICE_TYPES`` / ``NODE_TYPES`` tables never ran, because the class branch
returns before reaching them. Everything logged "Successfully loaded".

The names are not cosmetic. Maimu was a built-in before it was a plugin, so
every ``.glider`` file written to date names the device type and the node type
``Maimu``. Registering anything else silently breaks those files.

Requires the plugin to be installed (``uv pip install -e ./plugins/glider-maimu``),
which is what CI does and what the README tells a developer to do.
"""

from __future__ import annotations

import pytest

from glider.core.flow_engine import FlowEngine
from glider.hal.base_device import DEVICE_REGISTRY
from glider.plugins.plugin_manager import PluginManager, plugin_components
from glider_maimu.device import MaimuDevice
from glider_maimu.node import MaimuNode

# The name every experiment file written while Maimu was a built-in refers to.
EXPECTED_NAME = "Maimu"

pytestmark = pytest.mark.real_registration


@pytest.fixture
async def loaded():
    """Discover and load plugins the way the application does at startup.

    The skip guard checks that the *distribution* is installed, deliberately
    not that a plugin of some expected name was discovered. Keying it on the
    name would make these tests skip -- silently, and therefore green -- in
    exactly the case they exist to catch, which is the plugin registering under
    a name nobody meant.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("glider-maimu")
    except PackageNotFoundError:
        pytest.skip("glider-maimu is not installed; run `uv pip install -e ./plugins/glider-maimu`")

    manager = PluginManager()
    await manager.discover_plugins()
    await manager.load_plugins()
    return manager


async def test_the_device_registers_under_the_name_saved_files_use(loaded):
    assert DEVICE_REGISTRY.get(EXPECTED_NAME) is MaimuDevice


async def test_the_node_registers_under_the_name_saved_files_use(loaded):
    assert FlowEngine.get_node_class(EXPECTED_NAME) is MaimuNode


async def test_no_stray_lowercase_aliases(loaded):
    """`maimu` and `maimu_node` were what the class-style entry points produced.
    A stray alias is not harmless: it shows up as a second entry in the Add
    Device list and invites saving a file against a name the next version will
    not have."""
    strays = [n for n in DEVICE_REGISTRY if "aimu" in n.lower() and n != EXPECTED_NAME]
    strays += [n for n in FlowEngine._node_registry if "aimu" in n.lower() and n != EXPECTED_NAME]

    assert strays == [], f"unexpected registrations: {strays}"


async def test_both_components_are_attributed_to_this_plugin(loaded):
    """Provenance is what puts the node in the library's Plugins section and
    names the plugin on its tooltip."""
    assert plugin_components("device").get(EXPECTED_NAME) == "glider_maimu"
    assert plugin_components("node").get(EXPECTED_NAME) == "glider_maimu"


async def test_a_saved_experiment_can_still_build_the_device(loaded):
    """The whole point of the name: create_device_from_dict is what loading a
    .glider file goes through."""
    from glider.hal.base_device import create_device_from_dict

    device = create_device_from_dict(
        {
            "id": "stim1",
            "device_type": EXPECTED_NAME,
            "name": "Stimulator",
            "config": {"pins": {}, "settings": {"address": "AA:BB:CC:DD:EE:FF"}},
        },
        board=None,
    )

    assert isinstance(device, MaimuDevice)
    assert device.device_type == EXPECTED_NAME
