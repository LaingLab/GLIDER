"""The installed plugin registers under the name experiment files use.

Requires the plugin to be installed (``uv pip install -e ./plugins/glider-vapepod``),
which is what CI does; skips otherwise. glider-maimu's test_packaging.py has the
story of the class-style entry points this guards against.
"""

from __future__ import annotations

import pytest

from glider.core.flow_engine import FlowEngine
from glider.hal.base_device import DEVICE_REGISTRY
from glider.plugins.plugin_manager import PluginManager, plugin_components
from glider_vapepod.device import VapePodDevice
from glider_vapepod.node import VapePodNode
from glider_vapepod.wired import VapePodUSBDevice

EXPECTED_NAME = "VapePod"
USB_NAME = "VapePodUSB"

pytestmark = pytest.mark.real_registration


@pytest.fixture
async def loaded():
    """Discover and load plugins the way the application does at startup.

    The skip keys on the distribution being installed, not on a plugin of the
    expected name being found, so a plugin registering under the wrong name
    fails here rather than skipping green.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        distribution("glider-vapepod")
    except PackageNotFoundError:
        pytest.skip(
            "glider-vapepod is not installed; run `uv pip install -e ./plugins/glider-vapepod`"
        )

    manager = PluginManager()
    await manager.discover_plugins()
    await manager.load_plugins()
    return manager


async def test_the_device_registers_as_vapepod(loaded):
    assert DEVICE_REGISTRY.get(EXPECTED_NAME) is VapePodDevice


async def test_the_usb_device_registers_as_vapepodusb(loaded):
    assert DEVICE_REGISTRY.get(USB_NAME) is VapePodUSBDevice


async def test_the_node_registers_as_vapepod(loaded):
    assert FlowEngine.get_node_class(EXPECTED_NAME) is VapePodNode


async def test_no_stray_aliases(loaded):
    strays = [
        n for n in DEVICE_REGISTRY if "vapepod" in n.lower() and n not in (EXPECTED_NAME, USB_NAME)
    ]
    strays += [
        n for n in FlowEngine._node_registry if "vapepod" in n.lower() and n != EXPECTED_NAME
    ]
    assert strays == [], f"unexpected registrations: {strays}"


async def test_both_components_are_attributed_to_this_plugin(loaded):
    assert plugin_components("device").get(EXPECTED_NAME) == "glider_vapepod"
    assert plugin_components("device").get(USB_NAME) == "glider_vapepod"
    assert plugin_components("node").get(EXPECTED_NAME) == "glider_vapepod"


async def test_a_saved_experiment_can_build_the_device(loaded):
    from glider.hal.base_device import create_device_from_dict

    device = create_device_from_dict(
        {
            "id": "pod1",
            "device_type": EXPECTED_NAME,
            "name": "Side pod",
            "config": {"pins": {}, "settings": {"name": "VapePod"}},
        },
        board=None,
    )
    assert isinstance(device, VapePodDevice)


async def test_a_saved_experiment_can_build_the_usb_device(loaded):
    from glider.hal.base_device import create_device_from_dict

    device = create_device_from_dict(
        {
            "id": "pod1",
            "device_type": USB_NAME,
            "name": "Side pod",
            "config": {"pins": {}, "settings": {"port": "/dev/ttyACM0"}},
        },
        board=None,
    )
    assert isinstance(device, VapePodUSBDevice)
    assert device.port == "/dev/ttyACM0"
