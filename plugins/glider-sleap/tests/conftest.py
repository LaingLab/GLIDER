"""Fixtures shared by the whole ``glider-sleap`` suite.

Most tests here call the converter directly and need nothing. The ones that go
through core -- ``find_converter``, ``needs_conversion``, the model-selection
path -- need it in the registry, which is PluginManager's job in the running
application and is done by hand here.
"""

import pytest

from glider_sleap import POSE_CONVERTERS


@pytest.fixture
def registered_plugin(monkeypatch):
    """Register this plugin's converter, and unregister afterwards.

    Not autouse: unlike a device or a node, a converter is reachable without
    the registry, and most of this suite is better off calling it directly.
    Ask for the fixture when the test is about what core does with it.

    ``monkeypatch.setitem`` restores the previous state either way, so a
    registration cannot leak into a later test and let it pass for the wrong
    reason.
    """
    from glider.plugins import plugin_manager as pm
    from glider.vision.pose import converters as core

    for name, cls in POSE_CONVERTERS.items():
        monkeypatch.setitem(core.POSE_CONVERTERS, name, cls)
        monkeypatch.setitem(pm._PLUGIN_COMPONENTS, ("pose", name), "glider-sleap")
