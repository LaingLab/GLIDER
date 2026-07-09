"""HardwareManager teardown paths must bound device.shutdown() with a timeout.

BLEWriteDevice.shutdown() (and any device shutdown doing real I/O) can hang on
a wedged peripheral. emergency_stop() already wraps shutdown in
asyncio.wait_for(DEVICE_IO_TIMEOUT_S); disconnect_board() and shutdown_device()
must do the same so teardown continues past a stuck device.
"""

import asyncio
import logging

import glider.core.hardware_manager as hm_module
from glider.core.hardware_manager import HardwareManager
from glider.hal.mock_board import MockBoard


class _HungDevice:
    """Device whose shutdown never returns (wedged I/O)."""

    def __init__(self, board, device_id="hung1"):
        self.id = device_id
        self.board = board
        self.shutdown_started = False

    async def shutdown(self):
        self.shutdown_started = True
        await asyncio.sleep(60)


async def test_shutdown_device_times_out_on_wedged_device(monkeypatch, caplog):
    monkeypatch.setattr(hm_module, "DEVICE_IO_TIMEOUT_S", 0.05)
    hm = HardwareManager()
    board = MockBoard()
    device = _HungDevice(board)
    hm._devices["hung1"] = device

    with caplog.at_level(logging.WARNING):
        # Must return promptly (bounded by the patched timeout), not hang 60s.
        await asyncio.wait_for(hm.shutdown_device("hung1"), timeout=1.0)

    assert device.shutdown_started is True
    assert "TIMED OUT" in caplog.text


async def test_disconnect_board_times_out_on_wedged_device_and_continues(monkeypatch, caplog):
    monkeypatch.setattr(hm_module, "DEVICE_IO_TIMEOUT_S", 0.05)
    hm = HardwareManager()
    board = MockBoard()
    board._id = "b1"
    hm._boards["b1"] = board
    device = _HungDevice(board)
    hm._devices["hung1"] = device

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(hm.disconnect_board("b1"), timeout=1.0)

    # Teardown continued past the wedged device: the board was disconnected.
    assert device.shutdown_started is True
    assert board.is_connected is False
    assert "TIMED OUT" in caplog.text
