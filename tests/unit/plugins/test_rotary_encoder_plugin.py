"""
Tests for the rotary_encoder example device plugin.

Covers the device's revolution-tracking/conversion logic (hardware-free, by
driving the accumulator directly) and the plugin-loading path (PluginManager
discovers the directory plugin and registers RotaryEncoder in DEVICE_REGISTRY).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from glider.hal.base_device import DeviceConfig

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "plugins"
_ENCODER_INIT = _EXAMPLES / "rotary_encoder" / "__init__.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rotary_encoder_under_test", _ENCODER_INIT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_device(**settings):
    mod = _load_module()
    cfg = DeviceConfig(pins={}, settings=settings)
    return mod.RotaryEncoderDevice(board=None, config=cfg, name="enc")


def _feed(dev, angles):
    for a in angles:
        dev._accumulate(a)


def test_one_full_forward_turn_is_one_revolution():
    dev = _make_device(counts_per_turn=4096)
    _feed(dev, [0, 1024, 2048, 3072, 4095, 0])  # 0 -> 4095 -> wrap to 0
    assert dev.read_revolutions() == 1.0
    assert dev.read_total_counts() == 4096


def test_reverse_turn_is_negative():
    dev = _make_device(counts_per_turn=4096)
    # Decreasing through the wrap: 0 -> 4095(wrap back) -> 3072 -> ... -> 0
    _feed(dev, [0, 4095, 3072, 2048, 1024, 0])
    assert dev.read_revolutions() == -1.0


def test_gear_ratio_scales_output_revolutions():
    dev = _make_device(counts_per_turn=4096, gear_ratio=10.0, decimals=3)
    _feed(dev, [0, 1024, 2048, 3072, 4095, 0])  # one encoder turn
    assert dev.read_revolutions() == 0.1  # 1 / gear_ratio


def test_degrees_from_current_angle():
    dev = _make_device(counts_per_turn=4096, decimals=1)
    _feed(dev, [1024])  # quarter turn
    assert dev.read_degrees() == 90.0
    assert dev.read_angle() == 1024


def test_rounding_respects_decimals():
    dev = _make_device(counts_per_turn=4096, decimals=2)
    _feed(dev, [0, 100])  # 100 counts ~ 0.0244 rev
    assert dev.read_revolutions() == 0.02


def test_reset_zeroes_the_count():
    dev = _make_device(counts_per_turn=4096)
    _feed(dev, [0, 1024, 2048])
    assert dev.read_total_counts() > 0
    dev.reset()
    assert dev.read_revolutions() == 0.0
    assert dev.read_total_counts() == 0


def test_settings_schema_present():
    mod = _load_module()
    schema = mod.RotaryEncoderDevice.SETTINGS_SCHEMA
    keys = {f["key"] for f in schema}
    assert {"i2c_bus", "i2c_address", "counts_per_turn", "gear_ratio", "decimals"} <= keys
    assert mod.DEVICE_TYPES["RotaryEncoder"] is mod.RotaryEncoderDevice


async def test_plugin_manager_loads_and_registers_device():
    from glider.hal.base_device import DEVICE_REGISTRY
    from glider.plugins.plugin_manager import PluginManager

    pm = PluginManager(plugin_dirs=[_EXAMPLES], enable_directory_plugins=True)
    registered_before = "RotaryEncoder" in DEVICE_REGISTRY
    try:
        await pm.discover_plugins()
        await pm.load_plugins()
        assert "RotaryEncoder" in DEVICE_REGISTRY
        info = pm.get_plugin("rotary_encoder")
        assert info is not None and info.loaded
    finally:
        if not registered_before:
            DEVICE_REGISTRY.pop("RotaryEncoder", None)
