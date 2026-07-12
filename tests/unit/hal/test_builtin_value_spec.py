"""Built-in devices expose value_spec() so nodes/runner read one authority.

Confirms the op/kind-inferred defaults match today's ranges and that the servo's
per-instance angle bounds feed its declared spec.
"""

from __future__ import annotations

import types

from glider.hal.base_device import (
    AnalogInputDevice,
    DeviceConfig,
    DigitalOutputDevice,
    PWMOutputDevice,
    ServoDevice,
)
from glider.hal.value_spec import KIND_SWITCH, KIND_WHOLE


def _board(pwm=8, analog=10):
    return types.SimpleNamespace(
        capabilities=types.SimpleNamespace(pwm_resolution=pwm, analog_resolution=analog)
    )


def test_digital_output_set_is_a_switch():
    dev = DigitalOutputDevice(_board(), DeviceConfig(pins={"output": 5}))
    spec = dev.value_spec("set")
    assert (spec.kind, spec.min, spec.max) == (KIND_SWITCH, 0, 1)
    assert dev.value_spec("on") is None  # no-value action


def test_pwm_output_set_matches_resolution():
    assert (
        PWMOutputDevice(_board(pwm=8), DeviceConfig(pins={"output": 6})).value_spec("set").max
        == 255
    )
    assert (
        PWMOutputDevice(_board(pwm=12), DeviceConfig(pins={"output": 6})).value_spec("set").max
        == 4095
    )


def test_pwm_output_set_percent_is_zero_to_hundred():
    spec = PWMOutputDevice(_board(), DeviceConfig(pins={"output": 6})).value_spec("set_percent")
    assert (spec.min, spec.max, spec.unit) == (0, 100, "%")


def test_analog_input_read_matches_resolution():
    dev = AnalogInputDevice(_board(analog=10), DeviceConfig(pins={"input": 0}))
    spec = dev.value_spec("read")
    assert (spec.kind, spec.min, spec.max) == (KIND_WHOLE, 0, 1023)


def test_servo_angle_uses_instance_bounds():
    dev = ServoDevice(_board(), DeviceConfig(settings={"min_angle": 10, "max_angle": 170}))
    spec = dev.value_spec("set_angle")
    assert (spec.min, spec.max, spec.unit) == (10, 170, "deg")
    assert dev.value_spec("center") is None


def test_servo_inverted_bounds_are_normalized_to_a_valid_range():
    # min >= max would collapse both the angle clamp and the slider; the device
    # normalizes to 0..180 (and warns) so it stays controllable and its spec is
    # valid rather than silently ignoring every commanded angle.
    dev = ServoDevice(_board(), DeviceConfig(settings={"min_angle": 170, "max_angle": 10}))
    spec = dev.value_spec("set_angle")
    assert (spec.min, spec.max) == (0, 180)
    assert spec.validate() == []
