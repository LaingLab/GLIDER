"""Fixtures shared by the whole ``glider-sleap`` suite.

Most tests here call the converter directly and need nothing. The ones that go
through core -- ``find_converter``, ``needs_conversion``, the model-selection
path -- need it in the registry, which is PluginManager's job in the running
application and is done by hand here.
"""

import pytest

# When the plugin is not installed, ignore this directory instead of failing
# collection for the whole repository. glider-sleap declares tensorflow-cpu,
# which publishes wheels only for manylinux x86_64 and win_amd64 -- so on an
# Apple Silicon machine it cannot be installed at all, and the module-scope
# import that used to sit here took `pytest` down before a single unrelated
# test in any other directory ran.
#
# collect_ignore_glob rather than pytest.importorskip: raising Skipped while a
# conftest is being imported is a collection *error*, not a skip. This is the
# supported way to say "there is nothing to collect here".
#
# Nothing is being papered over. These tests are not passing, they are absent,
# and the plugin's own workflow installs it on a platform that has the wheels
# and runs them there.
try:
    from glider_sleap import POSE_CONVERTERS
except ImportError:  # pragma: no cover - depends on the host's wheels
    POSE_CONVERTERS = None
    collect_ignore_glob = ["*"]


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
