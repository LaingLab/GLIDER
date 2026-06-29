"""
Tests for BLEBoard -- the host Bluetooth LE adapter "board".

bleak is mocked via sys.modules so these run with no Bluetooth hardware.
"""

from __future__ import annotations

import sys
import types

import pytest

from glider.hal.boards.ble_board import BLEBoard


class _FakeBLEDevice:
    def __init__(self, name, address):
        self.name = name
        self.address = address


class _FakeAdv:
    def __init__(self, local_name):
        self.local_name = local_name
        self.service_uuids = []


@pytest.fixture
def fake_bleak(monkeypatch):
    mod = types.ModuleType("bleak")

    class _Scanner:
        @staticmethod
        async def discover(timeout=5.0, return_adv=False, **kwargs):
            # "Opto-A" advertises its name only in the scan response, so
            # device.name is empty but the advertisement's local_name has it.
            entries = {
                "AA:BB": (_FakeBLEDevice(None, "AA:BB"), _FakeAdv("Opto-A")),
                "CC:DD": (_FakeBLEDevice(None, "CC:DD"), _FakeAdv(None)),
            }
            if return_adv:
                return entries
            return [dev for dev, _ in entries.values()]

    mod.BleakScanner = _Scanner
    monkeypatch.setitem(sys.modules, "bleak", mod)
    return mod


async def test_connect_marks_ready(fake_bleak):
    board = BLEBoard()
    assert await board.connect() is True
    assert board.is_connected


async def test_scan_returns_name_address_pairs(fake_bleak):
    board = BLEBoard()
    await board.connect()
    results = await board.scan(timeout=0.1)
    assert ("Opto-A", "AA:BB") in results
    # Unnamed peripheral falls back to "(unknown)".
    assert ("(unknown)", "CC:DD") in results


async def test_pin_operations_raise(fake_bleak):
    board = BLEBoard()
    with pytest.raises(NotImplementedError):
        await board.write_digital(1, True)
    with pytest.raises(NotImplementedError):
        await board.read_analog(1)


def test_board_identity():
    board = BLEBoard()
    assert board.board_type == "bluetooth"
    assert board.capabilities.pins == {}
