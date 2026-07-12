"""Declarative-device value semantics: fallback, declared override, clamp-not-wrap.

Covers D8 backward-compat (old definitions with no `value` block fall back to
today's op-inferred ranges), the declared-spec override, the clamp-not-wrap fix
for over-range writes, NaN rejection, and the extended save-time validation.
"""

from __future__ import annotations

import math
import types

import pytest

from glider.hal.base_device import DeviceConfig
from glider.hal.declarative_device import build_device_class, validate_definition
from glider.hal.value_spec import KIND_WHOLE


class _Board:
    def __init__(self, pwm_resolution=8):
        self.calls: list = []
        self.capabilities = types.SimpleNamespace(pwm_resolution=pwm_resolution)

    async def set_pin_mode(self, pin, mode, ptype):
        pass

    async def write_analog(self, pin, val):
        self.calls.append(("write_analog", pin, val))

    async def write_digital(self, pin, val):
        self.calls.append(("write_digital", pin, val))


def _pwm_device(value_block=None, board=None):
    action = {"name": "set", "op": "write_pwm", "runtime_args": ["value"]}
    if value_block is not None:
        action["value"] = value_block
    defn = {"name": "Dimmer", "transport": "gpio", "actions": [action]}
    dev = build_device_class(defn)(
        board or _Board(), DeviceConfig(settings={"pin": 7}), name="Dimmer"
    )
    dev._initialized = True
    return dev


# --- D8 fallback (old definitions with no value block) -----------------------


def test_pwm_fallback_matches_board_resolution():
    assert _pwm_device().value_spec("set") == _pwm_device().value_spec("set")
    spec = _pwm_device(board=_Board(pwm_resolution=8)).value_spec("set")
    assert (spec.kind, spec.min, spec.max) == (KIND_WHOLE, 0, 255)
    spec12 = _pwm_device(board=_Board(pwm_resolution=12)).value_spec("set")
    assert spec12.max == 4095


def test_i2c_write_byte_word_fallbacks():
    defn = {
        "name": "Reg",
        "transport": "i2c",
        "actions": [
            {"name": "b", "op": "write_byte", "params": {"register": 1}, "runtime_args": ["value"]},
            {"name": "w", "op": "write_word", "params": {"register": 2}, "runtime_args": ["value"]},
        ],
    }
    dev = build_device_class(defn)(_Board(), DeviceConfig(), name="Reg")
    assert dev.value_spec("b").max == 0xFF
    assert dev.value_spec("w").max == 0xFFFF


def test_no_value_action_has_no_spec():
    defn = {"name": "Sw", "transport": "gpio", "actions": [{"name": "on", "op": "set_high"}]}
    dev = build_device_class(defn)(_Board(), DeviceConfig(settings={"pin": 3}), name="Sw")
    assert dev.value_spec("on") is None


# --- declared override -------------------------------------------------------


def test_declared_block_overrides_fallback():
    dev = _pwm_device(
        {"kind": "whole", "min": 0, "max": 100, "unit": "mL/min", "label": "Set Rate"}
    )
    spec = dev.value_spec("set")
    assert (spec.min, spec.max, spec.unit, spec.label) == (0, 100, "mL/min", "Set Rate")


# --- clamp, not wrap ---------------------------------------------------------


async def test_over_range_pwm_write_clamps_to_declared_max():
    board = _Board()
    dev = _pwm_device({"kind": "whole", "min": 0, "max": 100}, board=board)
    await dev.execute_action("set", 5000)
    assert [c[2] for c in board.calls] == [100]  # clamped to 100, not 5000


async def test_massive_value_does_not_wrap():
    # The old bug: a huge value masked to an unrelated number. With a 0..4095
    # declaration it must land on 4095, never 70000 & anything.
    board = _Board()
    dev = _pwm_device({"kind": "whole", "min": 0, "max": 4095}, board=board)
    await dev.execute_action("set", 70000)
    assert [c[2] for c in board.calls] == [4095]  # not 70000 & 0xFFFF == 4464


async def test_nan_write_is_rejected_and_sends_nothing():
    board = _Board()
    dev = _pwm_device({"kind": "whole", "min": 0, "max": 100}, board=board)
    with pytest.raises(ValueError, match="non-finite"):
        await dev.execute_action("set", math.nan)
    assert board.calls == []  # nothing reached hardware


# --- extended validation -----------------------------------------------------


@pytest.mark.parametrize(
    "action, needle",
    [
        (
            {"name": "on", "op": "set_high", "value": {"kind": "whole", "min": 0, "max": 5}},
            "value-writing op",
        ),
        (
            {
                "name": "xy",
                "op": "write_pwm",
                "runtime_args": ["x", "y"],
                "value": {"kind": "whole", "min": 0, "max": 5},
            },
            "multi-argument",
        ),
        (
            {
                "name": "p",
                "op": "write_pwm",
                "runtime_args": ["value"],
                "value": {"kind": "whole", "min": 5, "max": 5},
            },
            "less than",
        ),
        (
            {
                "name": "p",
                "op": "write_pwm",
                "runtime_args": ["value"],
                "value": {"kind": "whole", "min": 0},
            },
            "missing 'max'",
        ),
    ],
)
def test_validate_rejects_bad_value_blocks(action, needle):
    errors = validate_definition({"name": "D", "transport": "gpio", "actions": [action]})
    assert any(needle in e for e in errors), errors


def test_validate_accepts_good_value_block():
    defn = {
        "name": "D",
        "transport": "gpio",
        "actions": [
            {
                "name": "p",
                "op": "write_pwm",
                "runtime_args": ["value"],
                "value": {"kind": "whole", "min": 0, "max": 4095, "unit": "counts"},
            }
        ],
    }
    # No value-related errors (a valid device may still need pin settings, but
    # validate_definition doesn't require those).
    assert not any("value" in e for e in validate_definition(defn))
