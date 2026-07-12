"""Built-in PWM/servo writes clamp to the device's declared range and reject
non-finite values (B4) — matching the declarative-device write path, instead of
silently turning NaN into 0 / min_angle.
"""

from __future__ import annotations

import math
import types

import pytest

from glider.hal.base_device import DeviceConfig, PWMOutputDevice, ServoDevice


class _RecordingBoard:
    def __init__(self, pwm=8):
        self.capabilities = types.SimpleNamespace(pwm_resolution=pwm, analog_resolution=10)
        self.writes: list = []

    async def write_analog(self, pin, value):
        self.writes.append(("analog", pin, value))

    async def write_servo(self, pin, angle):
        self.writes.append(("servo", pin, angle))


async def test_pwm_set_value_clamps_to_range():
    board = _RecordingBoard(pwm=8)  # 0..255
    dev = PWMOutputDevice(board, DeviceConfig(pins={"output": 6}))

    await dev.set_value(300)  # over max -> clamps to 255
    assert board.writes[-1] == ("analog", 6, 255)
    await dev.set_value(-5)  # under min -> clamps to 0
    assert board.writes[-1] == ("analog", 6, 0)
    await dev.set_value(128)  # in range -> unchanged
    assert board.writes[-1] == ("analog", 6, 128)


async def test_pwm_set_value_rejects_non_finite():
    board = _RecordingBoard(pwm=8)
    dev = PWMOutputDevice(board, DeviceConfig(pins={"output": 6}))
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            await dev.set_value(bad)
    assert board.writes == []  # nothing reached the wire


async def test_servo_set_angle_clamps_and_rejects_non_finite():
    board = _RecordingBoard()
    dev = ServoDevice(
        board, DeviceConfig(pins={"signal": 9}, settings={"min_angle": 10, "max_angle": 170})
    )

    await dev.set_angle(200)  # over max -> clamps to 170
    assert board.writes[-1] == ("servo", 9, 170)
    await dev.set_angle(0)  # under min -> clamps to 10
    assert board.writes[-1] == ("servo", 9, 10)

    board.writes.clear()
    with pytest.raises(ValueError):
        await dev.set_angle(math.nan)
    assert board.writes == []
