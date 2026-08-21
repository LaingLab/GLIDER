"""Fixtures shared by the whole ``glider-maimu`` suite.

The tests exercise this plugin the way GLIDER runs it -- through the core
registries -- so they need the components registered. In the app that is
PluginManager's job; here it is done directly, which also keeps the suite
honest about what registration actually has to accomplish.
"""

import pytest

from glider_maimu import DEVICE_TYPES, NODE_TYPES


@pytest.fixture(autouse=True)
def registered_plugin(request, monkeypatch):
    """Register the plugin's device and node, and unregister afterwards.

    Autouse and unconditional: every test here reaches the plugin through a
    core registry (the device factory, the flow engine, the Add Device dialog),
    and a leaked registration would let a later test pass for the wrong reason.
    monkeypatch.setitem restores the previous state either way.
    """
    if "real_registration" in request.keywords:
        # test_packaging exercises the installed entry points instead, and a
        # hand-registered "Maimu" would mask exactly what it is checking.
        return

    from glider.core.flow_engine import FlowEngine
    from glider.hal.base_device import DEVICE_REGISTRY
    from glider.plugins import plugin_manager as pm

    for name, cls in DEVICE_TYPES.items():
        monkeypatch.setitem(DEVICE_REGISTRY, name, cls)
        monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("device", name), "glider-maimu")
    for name, cls in NODE_TYPES.items():
        monkeypatch.setitem(FlowEngine._node_registry, name, cls)
        monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("node", name), "glider-maimu")
