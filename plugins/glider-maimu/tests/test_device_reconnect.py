"""A Maimu comes back doing what the experiment last asked for.

The firmware turns the light off the moment the link drops, so a train cut off
by a drop has been interrupted, not continued. The reconnect puts the last
command back -- a train for whatever is left of it -- and with nothing asked,
or the train run out, it writes 'off', so the device is always in a known state.
Commands that arrive while the link is down must not hold the flow up; the
newest one is what the reconnect sends.
"""

import asyncio
import sys
import types

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
        if getattr(self, "connect_error", None):
            raise self.connect_error
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
        client.connect_error = created.get("connect_error")
        created["client"] = client
        created["clients"].append(client)
        return client

    module = MagicMock(name="bleak")
    module.BleakClient = make_client

    async def _present(address, timeout=10.0):
        return address  # stands in for the BLEDevice bleak returns; BleakClient takes either

    module.BleakScanner.find_device_by_address = _present
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
            original.is_connected = False  # a real drop: retries only fire when the link is down
            raise OSError("link dropped mid-write")
        original.written.append(bytes(data))

    device._client.write_gatt_char = _flaky
    # original.is_connected is still True here, so _ensure_connected()'s first,
    # unconditional call short-circuits and op() runs straight into _flaky,
    # which drops the link and raises -- that drives _with_retry into its
    # reconnect branch, the reconnect-inside-a-write path this test polices.
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


# --- putting the last command back ---------------------------------------------


@pytest.fixture
def clock(monkeypatch):
    """The device's view of time, moved by hand: [now] in seconds."""
    now = [1000.0]
    monkeypatch.setattr(
        "glider_maimu.device.time", types.SimpleNamespace(perf_counter=lambda: now[0])
    )
    return now


async def _first_write_on_the_new_link(device):
    for _ in range(200):
        if device._client.written:
            return device._client.written
        await asyncio.sleep(0)
    raise AssertionError("nothing was written after the reconnect")


async def test_a_train_cut_off_by_a_drop_resumes_for_what_is_left(fake_bleak, clock):
    _module, created = fake_bleak
    device = await _initialized()
    await device.train(10, 40, 1200)
    assert created["client"].written == [b"p10,40,1200"]

    clock[0] += 200.4
    created["client"].drop()
    await _settle(device)

    assert await _first_write_on_the_new_link(device) == [b"p10,40,999"]
    await device.shutdown()


async def test_a_symmetric_train_resumes_in_the_original_form(fake_bleak, clock):
    _module, created = fake_bleak
    device = await _initialized()
    await device.pulse(500, 10)

    clock[0] += 3.5
    created["client"].drop()
    await _settle(device)

    assert await _first_write_on_the_new_link(device) == [b"500,6"]
    await device.shutdown()


@pytest.mark.parametrize("elapsed", [1200.0, 1199.5, 5000.0])
async def test_a_train_with_under_a_second_left_comes_back_off(fake_bleak, clock, elapsed):
    _module, created = fake_bleak
    device = await _initialized()
    await device.train(10, 40, 1200)

    clock[0] += elapsed
    created["client"].drop()
    await _settle(device)

    assert await _first_write_on_the_new_link(device) == [b"off"]
    await device.shutdown()


@pytest.mark.parametrize("command, expected", [("on", b"on"), ("off", b"off")])
async def test_on_and_off_are_put_back_as_they_were(fake_bleak, command, expected):
    _module, created = fake_bleak
    device = await _initialized()
    await getattr(device, command)()

    created["client"].drop()
    await _settle(device)

    assert await _first_write_on_the_new_link(device) == [expected]
    await device.shutdown()


async def test_commands_during_a_drop_return_at_once_and_the_newest_wins(fake_bleak, clock):
    """Zone enter/exit kept firing while the link was down. Each used to block
    the flow behind the reconnect and then land in a burst, oldest first."""
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("out of range")
    created["client"].drop()
    for _ in range(20):
        await asyncio.sleep(0)
    assert device.link_state is ConnectionState.RECONNECTING

    await asyncio.wait_for(device.train(10, 40, 1200), timeout=0.5)  # mouse enters
    await asyncio.wait_for(device.off(), timeout=0.5)  # mouse leaves
    await asyncio.wait_for(device.train(10, 40, 1200), timeout=0.5)  # and back in

    assert all(client.written == [] for client in created["clients"][1:])

    created["connect_error"] = None  # back in range
    await _settle(device)
    assert await _first_write_on_the_new_link(device) == [b"p10,40,1200"]
    await device.shutdown()


async def test_shutdown_forgets_the_last_command(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    await device.train(10, 40, 1200)
    await device.shutdown()
    assert device._desired is None
