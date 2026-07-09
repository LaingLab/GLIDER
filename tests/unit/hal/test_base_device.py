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
