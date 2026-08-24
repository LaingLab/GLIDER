"""A Maimu comes back off.

The firmware runs a pulse autonomously, so a link that died mid-train left a
stimulator running with nothing attached to stop it. The reconnect is the
first chance anyone has had to say otherwise, and it takes it -- the device
never silently resumes whatever it was doing.

Same reasoning as shutdown(), which writes 'off' before it disconnects.
"""

import asyncio
import sys

import pytest

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice
from glider_maimu.device import MaimuDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _FakeClient:
    def __init__(self, address, disconnected_callback=None):
        self.address = address
        self.is_connected = False
        self.written = []
        self.disconnected_callback = disconnected_callback

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append(bytes(data))

    def drop(self):
        self.is_connected = False
        if self.disconnected_callback is not None:
            self.disconnected_callback(self)


@pytest.fixture
def fake_bleak(monkeypatch):
    from unittest.mock import MagicMock

    created = {"clients": []}

    def make_client(address, *a, disconnected_callback=None, **k):
        client = _FakeClient(address, disconnected_callback)
        created["client"] = client
        created["clients"].append(client)
        return client

    module = MagicMock(name="bleak")
    module.BleakClient = make_client
    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    monkeypatch.setattr(BLEDevice, "RECONNECT_BASE_S", 0.001)
    monkeypatch.setattr(BLEDevice, "RECONNECT_MAX_BACKOFF_S", 0.001)


async def _initialized():
    config = DeviceConfig(settings={"address": "11:22:33:44:55:66"})
    device = MaimuDevice(_FakeBoard(), config, name="maimu")
    await device.initialize()
    return device


async def _settle(device):
    for _ in range(500):
        if device.link_state is ConnectionState.CONNECTED:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0)
    raise AssertionError("never reconnected")


async def test_reconnect_writes_off(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        if device._client.written:
            break
        await asyncio.sleep(0)
    assert device._client.written == [b"off"]
    await device.shutdown()


async def test_off_is_written_exactly_once(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    for _ in range(100):
        await asyncio.sleep(0)
    assert device._client.written.count(b"off") == 1
    await device.shutdown()


async def test_off_is_the_first_thing_on_the_new_link(fake_bleak):
    """A pulse issued after the reconnect must land after the safe state."""
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        if device._client.written:
            break
        await asyncio.sleep(0)
    await device.pulse(500, 10)
    assert device._client.written == [b"off", b"500,10"]
    await device.shutdown()


async def test_a_write_retry_does_not_write_off(fake_bleak):
    """The retry path carries the caller's own command; 'off' would cancel it."""
    _module, created = fake_bleak
    device = await _initialized()
    original = device._client
    failing = {"first": True}
    calls = {"n": 0}

    async def _flaky(char, data, response=False):
        calls["n"] += 1
        if failing["first"]:
            failing["first"] = False
            raise OSError("link dropped mid-write")
        original.written.append(bytes(data))

    device._client.write_gatt_char = _flaky
    # original.is_connected is still True here, so _ensure_connected()'s first,
    # unconditional call short-circuits and op() runs straight into _flaky,
    # which raises -- that is what actually drives _with_retry into its
    # except branch, the reconnect-inside-a-write path this test polices.
    clients_before = len(created["clients"])

    await device.pulse(500, 10)
    await asyncio.sleep(0)

    # Prove the retry path was actually taken, not skipped: _flaky was entered
    # (and raised) exactly once, and a fresh client was built to carry the
    # retried write -- so the write that landed did not go through _flaky
    # again.
    assert calls["n"] == 1
    assert len(created["clients"]) == clients_before + 1
    assert device._client is not original
    assert b"off" not in device._client.written
    assert device._client.written == [b"500,10"]
    await device.shutdown()
