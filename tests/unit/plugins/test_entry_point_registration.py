"""What a `module:Class` entry point must do.

These tests own the contract that a plugin author gets what the entry-point
syntax appears to promise. They are deliberately free of Qt and of the real
plugin discovery machinery: a PluginInfo is constructed directly, because the
behaviour under test is what `load_plugin` does with one, not how one is found.
"""

import sys
import types

import pytest

from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import DEVICE_REGISTRY
from glider.plugins.plugin_manager import PluginInfo, PluginManager


class _Board:
    """Stands in for a board class. Deliberately requires no constructor args,
    so that a test failure means the class was *called*, not that calling it
    happened to raise."""


class _OtherBoard:
    pass


@pytest.fixture
def fake_module(monkeypatch):
    """Install a throwaway module that tests can point entry points at."""
    module = types.ModuleType("fake_plugin_mod")
    module.Board = _Board
    module.OtherBoard = _OtherBoard
    calls = []
    module.setup = lambda: calls.append("setup")
    module.calls = calls
    monkeypatch.setitem(sys.modules, "fake_plugin_mod", module)
    return module


@pytest.fixture
def table_modules(monkeypatch):
    """Two distinct plugins whose ``BOARD_DRIVERS`` claim the same driver name."""
    first = types.ModuleType("fake_plugin_a")
    first.BOARD_DRIVERS = {"shared": _Board}
    second = types.ModuleType("fake_plugin_b")
    second.BOARD_DRIVERS = {"shared": _OtherBoard}
    monkeypatch.setitem(sys.modules, "fake_plugin_a", first)
    monkeypatch.setitem(sys.modules, "fake_plugin_b", second)
    return first, second


@pytest.fixture
def restore_sys_path():
    """`load_plugin` prepends a directory plugin's parent to ``sys.path``."""
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    yield
    sys.path[:] = saved_path
    for name in set(sys.modules) - saved_modules:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def clean_registries():
    """Registries are class-level and leak between tests otherwise."""
    drivers = dict(HardwareManager._driver_registry)
    devices = dict(DEVICE_REGISTRY)
    yield
    HardwareManager._driver_registry.clear()
    HardwareManager._driver_registry.update(drivers)
    DEVICE_REGISTRY.clear()
    DEVICE_REGISTRY.update(devices)


async def _load(manager, info):
    manager._plugins[info.name] = info
    return await manager.load_plugin(info.name)


async def test_a_class_entry_point_registers_into_its_group(fake_module):
    manager = PluginManager()
    info = PluginInfo(name="fakeboard", entry_point="fake_plugin_mod:Board", plugin_type="driver")

    assert await _load(manager, info) is True
    assert HardwareManager._driver_registry["fakeboard"] is _Board


async def test_a_device_class_lands_in_the_device_registry(fake_module):
    manager = PluginManager()
    info = PluginInfo(name="fakedev", entry_point="fake_plugin_mod:Board", plugin_type="device")

    assert await _load(manager, info) is True
    assert DEVICE_REGISTRY["fakedev"] is _Board


async def test_the_class_is_registered_not_instantiated(fake_module):
    """The old behaviour called the attribute. Registering the *class* is the
    whole point, so assert on identity rather than on truthiness."""
    manager = PluginManager()
    info = PluginInfo(name="fakeboard", entry_point="fake_plugin_mod:Board", plugin_type="driver")

    await _load(manager, info)
    registered = HardwareManager._driver_registry["fakeboard"]
    assert registered is _Board
    assert not isinstance(registered, _Board)


async def test_a_function_entry_point_is_still_called(fake_module):
    """Regression guard: the existing contract must not change."""
    manager = PluginManager()
    info = PluginInfo(name="fakesetup", entry_point="fake_plugin_mod:setup", plugin_type="driver")

    assert await _load(manager, info) is True
    assert fake_module.calls == ["setup"]
    assert "fakesetup" not in HardwareManager._driver_registry


async def test_a_missing_attribute_records_an_error(fake_module):
    manager = PluginManager()
    info = PluginInfo(name="ghost", entry_point="fake_plugin_mod:NoSuchThing", plugin_type="driver")

    assert await _load(manager, info) is False
    assert info.error is not None
    assert "NoSuchThing" in info.error


async def test_registering_the_same_class_twice_is_a_no_op(fake_module, caplog):
    """glider-harp declares both a module:Class and a module-only entry point,
    so the same class arrives twice. That must be silent."""
    manager = PluginManager()
    for _ in range(2):
        info = PluginInfo(name="dup", entry_point="fake_plugin_mod:Board", plugin_type="driver")
        info.loaded = False
        manager._plugins["dup"] = info
        await manager.load_plugin("dup")

    assert HardwareManager._driver_registry["dup"] is _Board
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


async def test_a_class_in_an_unmappable_group_fails_the_load(fake_module):
    """A class that registers nowhere must not report success.

    ``generic`` is the default ``plugin_type``, so this is the easy accident,
    and reporting True while registering nothing is the exact silent failure
    this module exists to remove. The error has to reach ``info.error``, since
    that is what the plugin row can show; a log warning is not a substitute.
    """
    manager = PluginManager()
    info = PluginInfo(name="orphan", entry_point="fake_plugin_mod:Board")

    assert info.plugin_type == "generic"
    assert await _load(manager, info) is False
    assert info.error is not None
    assert "generic" in info.error
    assert "orphan" not in HardwareManager._driver_registry


async def test_a_conflicting_class_is_logged_and_the_first_wins(table_modules, caplog):
    """Two *different* plugins claiming one driver name.

    Reachable as written: ``discover_plugins`` keys plugins by their own name,
    so the collision cannot come from one name loaded twice -- it comes from
    two distinct plugins whose ``BOARD_DRIVERS`` share a key.
    """
    manager = PluginManager()
    await _load(manager, PluginInfo(name="plugin_a", entry_point="fake_plugin_a"))
    await _load(manager, PluginInfo(name="plugin_b", entry_point="fake_plugin_b"))

    assert HardwareManager._driver_registry["shared"] is _Board
    assert any("shared" in r.message for r in caplog.records if r.levelname == "WARNING")


async def test_a_table_collides_with_a_class_entry_point(fake_module, table_modules, caplog):
    """The two registration routes share one namespace, so they can collide."""
    manager = PluginManager()
    first = PluginInfo(name="shared", entry_point="fake_plugin_mod:Board", plugin_type="driver")
    await _load(manager, first)
    await _load(manager, PluginInfo(name="plugin_b", entry_point="fake_plugin_b"))

    assert HardwareManager._driver_registry["shared"] is _Board
    assert any("shared" in r.message for r in caplog.records if r.levelname == "WARNING")


async def test_a_directory_plugin_without_setup_still_loads(tmp_path, restore_sys_path, caplog):
    """`setup` is optional, and directory plugins are how that gets broken.

    Discovery synthesizes the entry point for a bare package directory. If it
    synthesizes one *with* a colon, `load_plugin` cannot tell the difference
    between an author naming an attribute (missing => error) and discovery
    defaulting to `setup` (missing => fine), and every directory plugin that
    registers through tables instead of a setup function stops loading.
    """
    pkg = tmp_path / "dirplug"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "class Board:\n    pass\n\n\nBOARD_DRIVERS = {'dirboard': Board}\n",
        encoding="utf-8",
    )

    manager = PluginManager()
    discovered = await manager._discover_from_directory(tmp_path)
    assert [d.name for d in discovered] == ["dirplug"]

    info = discovered[0]
    assert await _load(manager, info) is True, info.error
    assert HardwareManager._driver_registry["dirboard"].__name__ == "Board"
    # Asserted last so that a regression fails on the behaviour above rather
    # than on this, which is only the mechanism.
    assert ":" not in info.entry_point
