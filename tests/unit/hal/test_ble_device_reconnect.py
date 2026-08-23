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

    The 5/10/20/40/60 schedule itself is asserted separately by
    test_backoff_doubles_to_the_cap, which calls _backoff_for directly --
    unbound, passing the class in for self -- and reads the computed values
    off undisturbed RECONNECT_BASE_S/RECONNECT_MAX_BACKOFF_S. It never lives
    through an actual sleep, so there is no recorded log to read either way.
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
    task = device._reconnect_task
    await asyncio.wait_for(task, timeout=5)
    assert device.link_state is ConnectionState.ERROR
    await device.shutdown()


async def test_attempts_are_bounded(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("gone")
    before = len(created["clients"])
    created["client"].drop()
    task = device._reconnect_task
    await asyncio.wait_for(task, timeout=5)
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
    task = device._reconnect_task
    await device.shutdown()
    assert task.done()
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


async def test_poll_link_recovers_a_drop_that_lands_mid_hook(fake_bleak):
    """A drop that lands while _on_reconnected() is still awaiting must not
    strand the device.

    _start_reconnect() correctly no-ops while the current loop task is still
    (briefly) alive, but once that loop returns and its own `finally` clears
    the task handle, the device is left resting at DISCONNECTED with nothing
    left to retry it -- poll_link()'s backstop is the only thing that can
    still notice and re-arm it.
    """
    _module, created = fake_bleak
    device = await _initialized()
    dropped_once = {"done": False}

    async def _drop_mid_hook():
        await asyncio.sleep(0)  # yield so the drop below lands mid-await
        if not dropped_once["done"]:
            dropped_once["done"] = True
            created["client"].drop()

    device._on_reconnected = _drop_mid_hook
    created["client"].drop()
    task = device._reconnect_task
    await task  # run the whole loop, including the hook's own drop, to completion

    assert device.link_state is ConnectionState.DISCONNECTED
    assert device._reconnect_task is None

    await device.poll_link()
    await _settle(device)
    assert device.link_state is ConnectionState.CONNECTED
    await device.shutdown()


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


async def test_reconnect_publishes_connected_only_after_resubscribing(fake_bleak):
    """Success order: re-subscribe, then CONNECTED, then the hook.

    _ensure_connected() publishes a transient CONNECTED of its own the moment
    the socket is up, before the subscription is restored; the loop must
    revert that so the CONNECTED a caller ultimately sees is the one where
    get_state() is already live, not one where _latest is still cleared and
    nothing is subscribed.
    """
    _module, created = fake_bleak
    device = await _initialized(read_char_uuid="beef", notify=True)
    seen = []
    device.set_link_state_callback(
        lambda dev: seen.append(
            (dev.link_state, dev._client.subscribed if dev._client is not None else False)
        )
    )
    created["client"].drop()
    await _settle(device)

    states = [state for state, _subscribed in seen]
    assert states[-2:] == [ConnectionState.RECONNECTING, ConnectionState.CONNECTED]
    # The subscription was already live at the moment the final CONNECTED published.
    assert seen[-1] == (ConnectionState.CONNECTED, True)
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
    # original.is_connected is still True here, so _ensure_connected()'s first,
    # unconditional call short-circuits and op() runs straight into _flaky,
    # which raises -- that is what actually drives _with_retry into its
    # except branch, the reconnect-inside-a-write path this test polices.
    clients_before = len(created["clients"])

    await device.write("hello")
    await asyncio.sleep(0)

    assert ran == []
    # Prove the retry path was actually taken, not skipped: a fresh client was
    # built and the write that landed went through it, not through _flaky again.
    assert len(created["clients"]) == clients_before + 1
    assert device._client is not original
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


# --- the three reconnect actors ----------------------------------------------
#
# There are three, not two: bleak's disconnected_callback (_on_disconnected),
# the supervised _reconnect_loop, and _with_retry's reconnect-inside-a-write.
# The tests above drive the first two. These drive the third, and the case
# where it collides with the second -- one blip mid-write arms *both*, and
# nothing above fires the callback and a write together.


async def test_a_write_retry_keeps_the_notify_subscription(fake_bleak):
    """The in-band repair re-subscribes, like the supervised loop does.

    start_notify binds to one BleakClient, so the client _with_retry builds
    inside a caller's write is deaf unless someone re-subscribes it -- and
    nothing notices: _ensure_connected publishes CONNECTED, so poll_link's
    backstop sees a live client and never fires, no reconnect task ever starts,
    and get_state() returns None for the rest of the session while the GUI
    reads Ready.
    """
    _module, created = fake_bleak
    device = await _initialized(read_char_uuid="beef", write_char_uuid="cafe", notify=True)
    original = created["client"]
    assert original.subscribed

    async def _flaky(char, data, response=False):
        raise OSError("link dropped mid-write")

    original.write_gatt_char = _flaky

    await device.write("hello")

    assert device._client is not original  # the retry did rebuild the client
    assert device._client.subscribed  # ... and the subscription moved with it
    device._client.push(b"7")
    assert await device.get_state() == "7"
    assert device.link_state is ConnectionState.CONNECTED
    await device.shutdown()


async def test_a_live_client_is_not_resubscribed_by_a_retry(fake_bleak):
    """The happy path must not churn the subscription (or drop _latest).

    _ensure_connected short-circuits when the existing client is still up, so
    a write over a healthy link has no reason to re-run start_notify -- and
    _resubscribe clears the cached sample, which would make an unnecessary
    call visible as a get_state() that briefly returns None.
    """
    _module, created = fake_bleak
    device = await _initialized(read_char_uuid="beef", write_char_uuid="cafe", notify=True)
    original = created["client"]
    original.push(b"42")
    calls = []
    original.start_notify = lambda char, handler: calls.append(char)

    await device.write("hello")

    assert calls == []
    assert device._client is original
    assert await device.get_state() == "42"
    await device.shutdown()


async def test_the_loop_keeps_a_link_a_write_already_repaired(fake_bleak):
    """One blip mid-write arms both recovery paths; only one may rebuild.

    bleak's callback starts the supervised loop and the in-flight write repairs
    the link in-band. Without an early-out the loop then discards the working
    client -- without disconnecting it, so the orphan is still holding the
    peripheral -- and runs _on_reconnected(), which for a stimulator writes
    'off' over the train the operator just started. Operator-visible as: press
    Pulse, GUI says Ready throughout, stimulation stops ~5s in for no stated
    reason.
    """
    _module, created = fake_bleak
    device = await _initialized(write_char_uuid="cafe")
    ran = []
    device._on_reconnected = lambda: _record(ran)

    original = created["client"]

    async def _drop_mid_write(char, data, response=False):
        # Fires bleak's callback (arming the loop) and fails the write that is
        # already in flight (arming _with_retry) -- both, from one event.
        original.drop()
        raise OSError("link dropped mid-write")

    original.write_gatt_char = _drop_mid_write

    await device.write(500, 10)
    repaired = device._client
    assert repaired is not original
    assert repaired.written == [b"500,10"]

    task = device._reconnect_task
    assert task is not None, "the disconnect callback should have armed the supervised loop"
    await asyncio.wait_for(task, timeout=5)

    assert device._client is repaired  # not torn down and rebuilt behind the write
    assert len(created["clients"]) == 2  # no third client, so no orphan left connected
    assert repaired.written == [b"500,10"]  # and no 'off' cancelling the train
    assert ran == []  # the safe-state hook is for a link *this loop* brought back
    assert device.link_state is ConnectionState.CONNECTED
    await device.shutdown()


async def test_the_loops_early_out_leaves_a_subscribed_link_not_just_a_live_one(fake_bleak):
    """The ordering dependency between the two fixes, pinned.

    The loop's early-out trusts _with_retry to have re-subscribed. A client
    that is connected but deaf is exactly as useless as a disconnected one --
    get_state() returns None forever -- and returning CONNECTED for one would
    reintroduce the bug through the other door.
    """
    _module, created = fake_bleak
    device = await _initialized(read_char_uuid="beef", write_char_uuid="cafe", notify=True)
    original = created["client"]

    async def _drop_mid_write(char, data, response=False):
        original.drop()
        raise OSError("link dropped mid-write")

    original.write_gatt_char = _drop_mid_write

    await device.write("go")
    task = device._reconnect_task
    assert task is not None
    await asyncio.wait_for(task, timeout=5)

    assert device.link_state is ConnectionState.CONNECTED
    assert device._client.subscribed
    device._client.push(b"9")
    assert await device.get_state() == "9"
    await device.shutdown()
