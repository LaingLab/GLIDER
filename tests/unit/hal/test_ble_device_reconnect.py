"""BLEDevice's bounded reconnect.

Two behaviours carry the weight here.

The first is that a reconnect re-subscribes. _with_retry rebuilt the client on
a failed GATT op but only initialize() ever called start_notify, so a notify
device silently lost its subscription on the first blip and get_state()
returned None for the rest of the session with nothing logged.

The second is the _on_reconnected hook, and specifically that it fires on the
supervised reconnect and NOT on the reconnect inside a write's retry. A Maimu
uses it to write 'off'; running it on the retry path would cancel the exact
command the caller had just issued.
"""

import asyncio
import sys

import pytest

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice


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
        self._handler = None
        self.connect_error = None

    async def connect(self):
        if self.connect_error:
            raise self.connect_error
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append(bytes(data))

    async def read_gatt_char(self, char):
        return bytearray(b"1")

    async def start_notify(self, char, handler):
        self._handler = handler

    async def stop_notify(self, char):
        self._handler = None

    @property
    def subscribed(self):
        return self._handler is not None

    def push(self, data: bytes):
        assert self._handler is not None, "not subscribed"
        self._handler(object(), bytearray(data))

    def drop(self, *, notify=True):
        self.is_connected = False
        if notify and self.disconnected_callback is not None:
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
    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    """Collapse the backoff so the suite runs in milliseconds, not minutes.

    The delays are asserted separately in test_backoff_doubles_to_the_cap,
    which reads them off a recorded sleep log instead of living through them.
    """
    monkeypatch.setattr(BLEDevice, "RECONNECT_BASE_S", 0.001)
    monkeypatch.setattr(BLEDevice, "RECONNECT_MAX_BACKOFF_S", 0.001)


async def _initialized(**settings):
    settings.setdefault("address", "11:22:33:44:55:66")
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings=settings), name="ble")
    await device.initialize()
    return device


async def _settle(device, tries=200):
    """Let the reconnect task run to a resting state."""
    for _ in range(tries):
        if device.link_state is not ConnectionState.RECONNECTING:
            return
        await asyncio.sleep(0)
    raise AssertionError("reconnect never settled")


# --- the loop ----------------------------------------------------------------


async def test_a_drop_starts_a_reconnect(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    assert device.link_state is ConnectionState.CONNECTED
    await device.shutdown()


async def test_reconnect_builds_a_fresh_client(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    first = created["client"]
    first.drop()
    await _settle(device)
    assert device._client is not first
    assert device._client.is_connected
    await device.shutdown()


async def test_giving_up_lands_in_error(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("peripheral is gone")
    created["client"].drop()
    for _ in range(5000):
        if device.link_state is ConnectionState.ERROR:
            break
        await asyncio.sleep(0)
    assert device.link_state is ConnectionState.ERROR
    await device.shutdown()


async def test_attempts_are_bounded(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("gone")
    before = len(created["clients"])
    created["client"].drop()
    for _ in range(5000):
        if device.link_state is ConnectionState.ERROR:
            break
        await asyncio.sleep(0)
    attempts = len(created["clients"]) - before
    assert attempts == BLEDevice.MAX_RECONNECT_ATTEMPTS
    await device.shutdown()


async def test_backoff_doubles_to_the_cap(monkeypatch):
    """5 -> 10 -> 20 -> 40 -> 60 -> 60, matching BaseBoard._attempt_reconnect."""
    monkeypatch.setattr(BLEDevice, "RECONNECT_BASE_S", 5.0)
    monkeypatch.setattr(BLEDevice, "RECONNECT_MAX_BACKOFF_S", 60.0)
    delays = [BLEDevice._backoff_for(BLEDevice, attempt) for attempt in range(6)]
    assert delays == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]


async def test_shutdown_during_backoff_cancels_the_task(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("gone")
    created["client"].drop()
    await asyncio.sleep(0)
    await device.shutdown()
    task = device._reconnect_task
    assert task is None or task.done()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_only_one_reconnect_task_at_a_time(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("gone")
    client = created["client"]
    client.drop()
    first = device._reconnect_task
    client.drop()
    assert device._reconnect_task is first
    await device.shutdown()


async def test_no_reconnect_after_shutdown(fake_bleak):
    """A drop reported during teardown must not resurrect the device."""
    _module, created = fake_bleak
    device = await _initialized()
    client = created["client"]
    await device.shutdown()
    client.drop()
    assert device._reconnect_task is None
    assert device.link_state is ConnectionState.DISCONNECTED


# --- the subscription --------------------------------------------------------


async def test_reconnect_restores_the_notify_subscription(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized(read_char_uuid="beef", notify=True)
    created["client"].drop()
    await _settle(device)
    assert device._client.subscribed
    device._client.push(b"7")
    assert await device.get_state() == "7"
    await device.shutdown()


async def test_a_non_notify_device_does_not_subscribe(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized(write_char_uuid="cafe")
    created["client"].drop()
    await _settle(device)
    assert not device._client.subscribed
    await device.shutdown()


# --- the hook ----------------------------------------------------------------


async def test_the_hook_runs_on_a_supervised_reconnect(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    ran = []
    device._on_reconnected = lambda: _record(ran)
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        if ran:
            break
        await asyncio.sleep(0)
    assert ran == ["hook"]
    await device.shutdown()


def _record(log):
    async def _run():
        log.append("hook")

    return _run()


async def test_the_hook_does_not_run_on_a_write_retry(fake_bleak):
    """The retry path is a caller's own command; an 'off' there would cancel it."""
    _module, created = fake_bleak
    device = await _initialized(write_char_uuid="cafe")
    ran = []
    device._on_reconnected = lambda: _record(ran)

    original = created["client"]
    failing = {"first": True}

    async def _flaky(char, data, response=False):
        if failing["first"]:
            failing["first"] = False
            raise OSError("link dropped mid-write")
        original.written.append(bytes(data))

    device._client.write_gatt_char = _flaky
    device._client.is_connected = False  # force _ensure_connected to rebuild

    await device.write("hello")
    await asyncio.sleep(0)
    assert ran == []
    assert device._client.written == [b"hello"]
    await device.shutdown()


async def test_a_failing_hook_leaves_the_link_up(fake_bleak):
    """The link genuinely reconnected; a failed safe-state write does not undo that."""
    _module, created = fake_bleak
    device = await _initialized()

    async def _boom():
        raise OSError("could not send off")

    device._on_reconnected = _boom
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        await asyncio.sleep(0)
    assert device.link_state is ConnectionState.CONNECTED
    await device.shutdown()
