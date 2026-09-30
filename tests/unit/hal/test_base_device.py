"""Tests for BaseDevice.execute_action lifecycle guards."""

import pytest

from glider.hal.base_device import DeviceConfig, DigitalOutputDevice
from glider.hal.mock_board import MockBoard


async def test_execute_action_refused_after_shutdown():
    board = MockBoard()
    device = DigitalOutputDevice(board, DeviceConfig(pins={"output": 5}), name="relay")
    await device.initialize()
    await device.execute_action("on")
    assert board.get_pin_state(5) is True

    await device.shutdown()
    assert board.get_pin_state(5) is False

    # After shutdown (e.g. an e-stop), actions must refuse to run rather than
    # drive the pin HIGH again.
    with pytest.raises(RuntimeError, match="not initialized"):
        await device.execute_action("on")
    assert board.get_pin_state(5) is False


async def test_execute_action_rearmed_by_reinitialize():
    board = MockBoard()
    device = DigitalOutputDevice(board, DeviceConfig(pins={"output": 5}), name="relay")
    await device.initialize()
    await device.shutdown()
    await device.initialize()
    await device.execute_action("on")
    assert board.get_pin_state(5) is True


async def test_analog_read_clamps_to_the_boards_resolution_not_10_bits():
    # B-note: a 12-bit board's 4000 is a valid reading, not "out of range".
    from types import SimpleNamespace

    from glider.hal.base_device import AnalogInputDevice

    class Board12:
        capabilities = SimpleNamespace(analog_resolution=12)

        async def read_analog(self, pin):
            return self.value

    board = Board12()
    device = AnalogInputDevice(board, DeviceConfig(pins={"input": 0}))
    board.value = 4000
    assert await device.read() == 4000
    board.value = 5000
    assert await device.read() == 4095
