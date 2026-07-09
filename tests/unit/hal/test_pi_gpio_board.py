"""Tests for PiGPIOBoard emergency_stop (no real GPIO hardware needed)."""

import logging

from glider.hal.base_board import PinMode
from glider.hal.boards.pi_gpio_board import PiGPIOBoard


class _FakeInput:
    """gpiozero-style input: read-only `value` property, no `off()`."""

    @property
    def value(self):
        return 0


class _FakeOutput:
    def __init__(self):
        self.off_called = False

    def off(self):
        self.off_called = True


async def test_emergency_stop_skips_input_devices(caplog):
    board = PiGPIOBoard(auto_reconnect=False)
    out = _FakeOutput()
    board._devices = {4: _FakeInput(), 5: out}
    board._pin_modes = {4: PinMode.INPUT, 5: PinMode.OUTPUT}

    with caplog.at_level(logging.ERROR):
        await board.emergency_stop()

    assert out.off_called is True
    assert "Error during emergency stop" not in caplog.text
