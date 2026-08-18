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


async def test_a_conflicting_class_is_logged_and_the_first_wins(fake_module, caplog):
    manager = PluginManager()
    first = PluginInfo(name="clash", entry_point="fake_plugin_mod:Board", plugin_type="driver")
    await _load(manager, first)

    second = PluginInfo(
        name="clash", entry_point="fake_plugin_mod:OtherBoard", plugin_type="driver"
    )
    await _load(manager, second)

    assert HardwareManager._driver_registry["clash"] is _Board
    assert any("clash" in r.message for r in caplog.records if r.levelname == "WARNING")
