"""A12: plugin discovery, unload and directory problems (review 2026-09-28)."""

from pathlib import Path

import pytest

from glider.plugins import plugin_manager as pm
from glider.plugins.installer import MalformedEntryError, installer_command
from glider.plugins.plugin_manager import PluginInfo, PluginManager


def _manager(tmp_path, monkeypatch) -> PluginManager:
    monkeypatch.setattr(PluginManager, "DEFAULT_PLUGIN_DIR", tmp_path / "plugins")
    return PluginManager()


async def test_one_name_in_two_groups_keeps_both(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)

    async def found():
        return [
            PluginInfo(name="harp", entry_point="h:Board", plugin_type="driver"),
            PluginInfo(name="harp", entry_point="h:Device", plugin_type="device"),
        ]

    monkeypatch.setattr(manager, "_discover_from_entry_points", found)
    await manager.discover_plugins()
    await manager.discover_plugins()  # idempotent

    assert sorted(manager.plugins) == ["device:harp", "harp"]
    assert manager.plugins["device:harp"].name == "harp"


async def test_unload_takes_components_out_of_the_registry(tmp_path, monkeypatch):
    manager = _manager(tmp_path, monkeypatch)

    class Thing:
        pass

    pm._register_component("node", "ThingNode", Thing, "thing")
    manager._plugins["thing"] = PluginInfo(name="thing", loaded=True)

    await manager.unload_plugin("thing")

    assert "ThingNode" not in pm._registry_for("node")
    assert "ThingNode" not in pm.plugin_components("node")


def test_an_unwritable_plugin_dir_does_not_raise(tmp_path, monkeypatch):
    def refuse(self, *a, **k):
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "mkdir", refuse)
    _manager(tmp_path, monkeypatch)  # must not raise


@pytest.mark.parametrize(
    "package", ["git+https://evil.example/repo.git", "x @ https://evil.example/x.whl"]
)
def test_a_url_or_direct_reference_package_is_refused(package):
    with pytest.raises(MalformedEntryError, match="malformed"):
        installer_command(package, pip_available=lambda: True, uv_path=lambda: None)


async def test_no_plugins_means_no_plugin_manager():
    from glider.core.glider_core import GliderCore

    core = GliderCore()
    await core.initialize(load_plugins=False)
    assert core.plugin_manager is None


async def test_unload_leaves_a_built_in_the_plugin_only_re_declared(tmp_path, monkeypatch):
    # GLIDER declares its own drivers as entry points too.
    manager = _manager(tmp_path, monkeypatch)
    registry = pm._registry_for("driver")
    builtin = registry["arduino"]
    pm._register_component("driver", "arduino", builtin, "arduino")
    manager._plugins["arduino"] = PluginInfo(name="arduino", loaded=True)

    await manager.unload_plugin("arduino")

    assert registry["arduino"] is builtin
