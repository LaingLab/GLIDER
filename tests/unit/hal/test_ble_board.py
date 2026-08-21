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
    def __init__(self, local_name, service_uuids=None, rssi=None):
        self.local_name = local_name
        self.service_uuids = list(service_uuids or [])
        self.rssi = rssi


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


async def test_scan_reports_each_peripheral(fake_bleak):
    board = BLEBoard()
    await board.connect()

    results = await board.scan(timeout=0.1)

    by_address = {p.address: p for p in results}
    # "Opto-A" advertises its name only in the scan response.
    assert by_address["AA:BB"].name == "Opto-A"
    # A peripheral that advertised no name keeps an empty one rather than a
    # placeholder, so callers can tell "nameless" from "named '(unknown)'".
    assert by_address["CC:DD"].name == ""


async def test_an_unnamed_peripheral_is_labelled_by_address(fake_bleak):
    """A Zephyr device whose scan response was dropped is a bare MAC in a list
    of bare MACs; the label is what a human has to pick from."""
    board = BLEBoard()
    await board.connect()

    results = await board.scan(timeout=0.1)

    assert next(p for p in results if p.address == "CC:DD").label == "CC:DD"
    assert next(p for p in results if p.address == "AA:BB").label == "Opto-A"


async def test_the_label_carries_signal_strength_when_known(monkeypatch, fake_bleak):
    """Which of several identical peripherals is the one on the bench in front
    of you is answered by RSSI and nothing else."""

    class _Scanner:
        @staticmethod
        async def discover(timeout=5.0, return_adv=False, **kwargs):
            return {"AA:BB": (_FakeBLEDevice(None, "AA:BB"), _FakeAdv("Stim", rssi=-42))}

    fake_bleak.BleakScanner = _Scanner
    board = BLEBoard()
    await board.connect()

    assert (await board.scan(timeout=0.1))[0].label == "Stim (-42 dBm)"


async def test_results_are_sorted_strongest_first(monkeypatch, fake_bleak):
    class _Scanner:
        @staticmethod
        async def discover(timeout=5.0, return_adv=False, **kwargs):
            return {
                "FAR": (_FakeBLEDevice(None, "FAR"), _FakeAdv("far", rssi=-90)),
                "NEAR": (_FakeBLEDevice(None, "NEAR"), _FakeAdv("near", rssi=-30)),
            }

    fake_bleak.BleakScanner = _Scanner
    board = BLEBoard()
    await board.connect()

    assert [p.address for p in await board.scan(timeout=0.1)] == ["NEAR", "FAR"]


async def test_advertised_services_are_kept_and_matchable(monkeypatch, fake_bleak):
    """The sturdiest identifier a nameless peripheral has."""
    service = "12345678-1234-5678-1234-56789ABCDEF0"

    class _Scanner:
        @staticmethod
        async def discover(timeout=5.0, return_adv=False, **kwargs):
            return {
                "AA:BB": (_FakeBLEDevice(None, "AA:BB"), _FakeAdv(None, service_uuids=[service]))
            }

    fake_bleak.BleakScanner = _Scanner
    board = BLEBoard()
    await board.connect()

    found = (await board.scan(timeout=0.1))[0]
    # Case-insensitive: advertisements and configuration disagree on case
    # routinely, and a UUID that matched only sometimes would be worse than one
    # that never did.
    assert found.advertises(service.lower())
    assert found.advertises(service.upper())
    assert not found.advertises("00000000-0000-0000-0000-000000000000")


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
