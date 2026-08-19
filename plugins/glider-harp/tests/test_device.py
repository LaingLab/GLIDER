"""``HarpDevice`` -- the four layers composed, and the order they run in.

Everything here runs against ``MockHarpDevice``, which substitutes the serial
handle and nothing else: the framing, decoding, register cache and reader
thread under test are the ones that talk to real hardware.

The failures being pinned are all silent ones. A device left in Standby is
connected, answers commands, and records nothing. A ``read()`` wired to
``snapshot()`` eats counts out of the CSV twenty times a second with no symptom
anywhere. A second ``initialize()`` leaves a daemon thread on the old handle
eating the frames the new one is waiting for. A ``shutdown()`` that ignores
``stop()``'s return writes a register while that thread is still consuming the
reply. None of these raise; all of them come back as an empty or wrong CSV.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import pytest
import yaml

from glider.hal.base_board import BoardConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.value_spec import KIND_WHOLE
from glider_harp import derivation
from glider_harp import device as device_module
from glider_harp.board import HarpBoard
from glider_harp.device import (
    OPERATION_CONTROL,
    ROUND_TRIP_TIMEOUT_S,
    SHUTDOWN_ROUND_TRIP_TIMEOUT_S,
)
from glider_harp.frames import MESSAGE_EVENT, MESSAGE_WRITE, decode, encode
from glider_harp.mock import DEFAULT_OPERATION_CONTROL, MockHarpDevice

# The shipped profile records LickState as "lick" and is written for WhoAmI
# 1400, so a schema used with it has to agree on both.
LICKETYSPLIT_WHO_AM_I = 1400

SCHEMA = {
    "device": "LicketySplit",
    "whoAmI": LICKETYSPLIT_WHO_AM_I,
    "registers": {
        # Recorded by the shipped profile.
        "LickState": {"address": 32, "type": "U8", "access": "Event"},
        # Writable, so it becomes an action.
        "StimulusOn": {"address": 33, "type": "U8", "access": "Write"},
        # Both, so it becomes an action too and can be read back.
        "Threshold": {"address": 34, "type": "U16", "access": ["Read", "Write"]},
    },
}

LICK_COLUMNS = ["lick_state", "lick_count", "lick_last_ms"]


def event(address: int, value: int, timestamp: float | None = None) -> bytes:
    """One Event frame, as the device would put it on the wire."""
    from harp.protocol import HarpMessage, MessageType, PayloadType

    return HarpMessage(
        MessageType.Event, address, PayloadType.U8, bytes([value]), timestamp=timestamp
    ).bytes


def _dead_read(_size=1):
    """A port whose device has gone: every read raises, as an unplugged one does."""
    raise OSError("the device has been disconnected")


def _wait_until_dead(reader, timeout: float = 2.0) -> bool:
    """Block until the reader thread has actually exited.

    A thread boundary, so a ``threading`` wait rather than a sleep-and-hope:
    the reader notices the dead port inside its own read, records the failure,
    and exits, and none of that is visible to the event loop until it has.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not reader.is_alive():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def schema_path(tmp_path):
    path = tmp_path / "device.yml"
    path.write_text(yaml.safe_dump(SCHEMA), encoding="utf-8")
    return path


@pytest.fixture
def board():
    return HarpBoard()


def make_device(board, schema_path, *, profile="licketysplit", frames=(), **kwargs):
    config = DeviceConfig(
        settings={
            "port": "COM-fake",
            "baudrate": 115200,
            "device_yml": str(schema_path),
            "profile": profile,
        }
    )
    return MockHarpDevice(board, config, "harp1", frames=frames, **kwargs)


@pytest.fixture
async def device(board, schema_path):
    """An initialized device with the shipped profile and no events queued."""
    dev = make_device(board, schema_path)
    await dev.initialize()
    try:
        yield dev
    finally:
        if dev.is_initialized:
            await dev.shutdown()


# --- what the recorder sees ---------------------------------------------


async def test_state_columns_match_the_profile(device):
    """The profile names one column base; the cache expands it into three.

    Separated by ``_``, not ``:``: the recorder builds its header as
    ``{device_id}:{sub_column}``, so a colon here would produce
    ``harp1:lick:state`` and nothing downstream could tell which colon was the
    separator.
    """
    assert device.state_columns() == LICK_COLUMNS


async def test_get_state_is_keyed_by_exactly_those_columns(device):
    """A header that disagrees with its rows writes a CSV that opens cleanly and lies."""
    state = await device.get_state()
    assert list(state) == device.state_columns()


async def test_replayed_events_arrive_with_the_right_counts(board, schema_path):
    """Three licks in one interval must be reported as three, not as "a lick"."""
    frames = [event(32, 1, 0.5), event(32, 0, 0.75), event(32, 1, 1.0)]
    dev = make_device(board, schema_path, frames=frames)
    await dev.initialize()
    try:
        assert dev.wait_for_replay(), "the replayed frames never reached the cache"
        state = await dev.get_state()
    finally:
        await dev.shutdown()

    assert state["lick_count"] == 3
    # The last event's value and device time, in ms, sub-millisecond kept.
    assert state["lick_state"] == 1
    assert state["lick_last_ms"] == pytest.approx(1000.0)


async def test_events_for_unrecorded_registers_are_ignored(board, schema_path):
    """A Harp board emits registers no experiment asked for; counting them
    would inflate a column that is not theirs."""
    frames = [event(32, 1, 0.5), event(33, 1, 0.6), event(99, 1, 0.7)]
    dev = make_device(board, schema_path, frames=frames)
    await dev.initialize()
    try:
        assert dev.wait_for_replay()
        state = await dev.get_state()
    finally:
        await dev.shutdown()
    assert state["lick_count"] == 1


async def test_an_untimestamped_event_reports_no_device_time(board, schema_path):
    """``last_ms`` of None beside a non-zero count is the normal representation
    of an untimestamped event, not an error. The CSV writes it as a blank."""
    dev = make_device(board, schema_path, frames=[event(32, 1, None)])
    await dev.initialize()
    try:
        assert dev.wait_for_replay()
        state = await dev.get_state()
    finally:
        await dev.shutdown()
    assert state["lick_count"] == 1
    assert state["lick_last_ms"] is None
    assert state["lick_state"] == 1


# --- read() must not consume; get_state() must ---------------------------


async def test_read_does_not_clear_the_counters(board, schema_path):
    """WaitForInput polls read() every 50 ms and the Input node prefers it.

    If that path consumed, an Input node dropped onto this device would eat
    counts out of the recording twenty times a second, and the CSV would be
    wrong with no symptom anywhere.
    """
    dev = make_device(board, schema_path, frames=[event(32, 1, 0.5), event(32, 1, 0.6)])
    await dev.initialize()
    try:
        assert dev.wait_for_replay()
        first = await dev.read()
        second = await dev.read()
        after = await dev.get_state()
    finally:
        await dev.shutdown()

    assert first["lick_count"] == 2
    assert second["lick_count"] == 2, "read() consumed the counters"
    assert after["lick_count"] == 2, "read() consumed the counters before get_state() saw them"


async def test_get_state_clears_the_counters(board, schema_path):
    """The counter is what makes the record honest: every event is reported in
    exactly one row, which requires the row that reported it to clear it."""
    dev = make_device(board, schema_path, frames=[event(32, 1, 0.5)])
    await dev.initialize()
    try:
        assert dev.wait_for_replay()
        first = await dev.get_state()
        second = await dev.get_state()
    finally:
        await dev.shutdown()

    assert first["lick_count"] == 1
    assert second["lick_count"] == 0
    # State and device time persist across the read; only the counter clears.
    assert second["lick_state"] == 1
    assert second["lick_last_ms"] == pytest.approx(500.0)


# --- Standby / Active ----------------------------------------------------


async def test_initialize_takes_the_device_out_of_standby(device):
    """A Harp device boots in Standby, where it answers commands and emits no
    events. Skip this and the device looks connected and is permanently
    silent -- an empty recording, with no error anywhere."""
    assert device.operation_control & 0x03 == 0x01


async def test_shutdown_returns_the_device_to_standby(board, schema_path):
    dev = make_device(board, schema_path)
    await dev.initialize()
    await dev.shutdown()
    assert dev.operation_control & 0x03 == 0x00


async def test_the_mode_change_preserves_the_rest_of_the_register(board, schema_path):
    """OperationControl also carries the heartbeat and LED flags. Writing a
    bare 0x01 would switch the operator's operation LED off because GLIDER
    connected."""
    dev = make_device(board, schema_path, operation_control=DEFAULT_OPERATION_CONTROL)
    await dev.initialize()
    try:
        assert dev.operation_control == DEFAULT_OPERATION_CONTROL | 0x01
    finally:
        await dev.shutdown()
    assert dev.operation_control == DEFAULT_OPERATION_CONTROL


async def test_initialize_fails_if_the_device_does_not_enter_active(board, schema_path):
    """The readback is the point of the write. A device that acknowledges a
    mode it did not enter is indistinguishable from one that entered it, right
    up until the recording comes back empty."""
    dev = make_device(board, schema_path)
    handle = dev.port_handle
    answer = handle._handle_request

    def stubborn(raw):
        frame = decode(raw)
        if frame.message_type == MESSAGE_WRITE and frame.address == OPERATION_CONTROL:
            return  # accepted, and quietly not applied
        answer(raw)

    handle._handle_request = stubborn

    with pytest.raises(RuntimeError, match="not in the mode"):
        await dev.initialize()
    assert not dev.is_initialized
    # And the failure left nothing behind, so a retry is possible.
    assert dev.port_handle.closed


async def test_initialize_rejects_a_schema_for_another_board(board, tmp_path):
    """A schema whose register names happen to exist on the connected board
    builds, derives and records -- the wrong registers."""
    other = dict(SCHEMA, whoAmI=1234)
    path = tmp_path / "other.yml"
    path.write_text(yaml.safe_dump(other), encoding="utf-8")
    config = DeviceConfig(settings={"port": "COM-fake", "device_yml": str(path), "profile": ""})
    dev = MockHarpDevice(board, config, "harp1", who_am_i=LICKETYSPLIT_WHO_AM_I)
    with pytest.raises(RuntimeError, match="another board"):
        await dev.initialize()
    assert dev.port_handle.closed


# --- initialize() twice --------------------------------------------------


async def test_a_second_initialize_is_refused(device):
    """Pinned as a refusal rather than a silent re-connect.

    The hazard is that a second initialize() opens a second handle and leaves
    the first reader -- a daemon thread -- running on the old one, consuming
    the frames the new one waits for, for the rest of the process.

    Refusing is chosen over shutting down and reconnecting because the quiet
    alternative is worse: a re-connect mid-recording stops the reader, rebuilds
    the cache, and resets every counter, so the CSV loses events and no row
    says so. A refusal is loud, reversible, and leaves the recording running.
    """
    with pytest.raises(RuntimeError, match="call shutdown"):
        await device.initialize()

    # And the running device is untouched by the refusal.
    assert device.is_initialized
    assert device.reader is not None and device.reader.is_alive()
    assert not device.port_handle.closed


async def test_initialize_works_again_after_shutdown(board, schema_path):
    """The refusal is about a port still held, not a one-shot device: a
    reconnect after a clean shutdown has to work, and has to get a *fresh*
    reader, since HarpReader.start() refuses after a stop()."""
    dev = make_device(board, schema_path, frames=[event(32, 1, 0.5)])
    await dev.initialize()
    first_reader = dev.reader
    await dev.shutdown()

    dev.port_handle._frames = [event(32, 1, 0.9)]
    dev.port_handle._replayed = threading.Event()
    await dev.initialize()
    try:
        assert dev.reader is not None
        assert dev.reader is not first_reader
        assert dev.wait_for_replay()
        assert (await dev.get_state())["lick_count"] == 1
    finally:
        await dev.shutdown()


# --- shutdown ordering ---------------------------------------------------


async def test_shutdown_does_not_touch_the_port_when_the_reader_will_not_stop(device):
    """``stop()`` returning False means the thread is still inside a read.

    Writing OperationControl then would race it for the reply and see only a
    timeout; closing the handle under an in-flight read is indistinguishable
    from an unplugged cable. So neither happens, and the port stays held --
    which is the truth, and is also what keeps initialize() refused.
    """
    reader = device.reader
    real_stop = reader.stop
    reader.stop = lambda *args, **kwargs: False
    before = len(device.port_handle.writes)

    await device.shutdown()

    assert len(device.port_handle.writes) == before, "wrote a register past a refused join"
    assert not device.port_handle.closed, "closed the port under a live reader"
    assert device.operation_control & 0x03 == 0x01, "left Active, as it must"
    assert device.reader is reader, "forgot the thread it left running"
    with pytest.raises(RuntimeError, match="call shutdown"):
        await device.initialize()

    # stop() keeps asking, so a later shutdown() still succeeds -- which is
    # also how this test hands the thread back rather than leaking it.
    reader.stop = real_stop
    await device.shutdown()
    assert device.port_handle.closed
    assert device.operation_control & 0x03 == 0x00


async def test_shutdown_stops_the_reader_before_writing_the_register(board, schema_path):
    """The order is stop, then Standby, then close -- and the first is what
    makes the second's reply reachable at all."""
    dev = make_device(board, schema_path)
    await dev.initialize()
    reader = dev.reader
    order: list[str] = []

    real_stop = reader.stop

    def watched_stop(*args, **kwargs):
        order.append("stop")
        return real_stop(*args, **kwargs)

    reader.stop = watched_stop

    real_write = dev.port_handle.write

    def watched_write(data):
        frame = decode(data)
        if frame.address == OPERATION_CONTROL and frame.message_type == MESSAGE_WRITE:
            order.append("write")
        return real_write(data)

    dev.port_handle.write = watched_write

    real_close = dev.port_handle.close

    def watched_close():
        order.append("close")
        return real_close()

    dev.port_handle.close = watched_close

    await dev.shutdown()

    assert order[0] == "stop"
    assert order[-1] == "close"
    assert "write" in order
    assert order.index("stop") < order.index("write") < order.index("close")


async def test_shutdown_joins_the_reader_off_the_event_loop(device):
    """``stop()`` blocks for up to 2 s waiting on the join.

    Run on the loop that is a two-second freeze of the GUI, the recorder and
    every other device, in the middle of a recording -- and nothing about it
    looks like an error afterwards. Asserted on the *thread* rather than on
    elapsed time, which would only be a slow test that sometimes passes.
    """
    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}
    real_stop = device.reader.stop

    def watched_stop(*args, **kwargs):
        seen["thread"] = threading.get_ident()
        return real_stop(*args, **kwargs)

    device.reader.stop = watched_stop
    await device.shutdown()

    assert seen["thread"] != loop_thread, "joined the reader on the event loop"


async def test_the_port_lock_is_held_across_the_close(board, schema_path):
    """Released before the close, an action already waiting on the lock wakes
    up and writes into a handle that is closing underneath it -- the exact
    interleaving the lock is documented to prevent, and it surfaces as an
    ``OSError`` that reads like a hardware fault rather than like a shutdown.

    Asserted on the lock while the close is in flight, which is the invariant
    itself; racing an action against it would only be a test that usually
    passes.
    """
    dev = make_device(board, schema_path)
    await dev.initialize()

    closing = threading.Event()
    release = threading.Event()
    real_close = dev.port_handle.close

    def slow_close():
        closing.set()
        release.wait(2.0)
        return real_close()

    dev.port_handle.close = slow_close
    shutdown = asyncio.create_task(dev.shutdown())
    try:
        assert await asyncio.to_thread(closing.wait, 2.0), "close was never reached"
        assert dev._port_lock.locked(), "released the port lock before closing the port"
    finally:
        release.set()
        await shutdown
    assert dev.port_handle.closed


async def test_settings_cannot_be_edited_while_a_refusing_reader_holds_the_port(device):
    """``apply_settings`` is keyed on the port, exactly as initialize() is.

    The two differ in one reachable case: a shutdown whose join was refused
    clears ``_initialized`` while the thread still owns the handle. Keyed on
    the flag, an edit there renames the port this device is still reading.
    """
    reader = device.reader
    real_stop = reader.stop
    reader.stop = lambda *args, **kwargs: False
    await device.shutdown()
    assert not device.is_initialized

    device.apply_settings({"port": "COM-somewhere-else"})
    assert device.port == "COM-fake", "renamed a port the reader is still holding"
    assert device.config.settings["port"] == "COM-somewhere-else", "the edit must still be saved"

    reader.stop = real_stop
    await device.shutdown()


async def test_a_vanished_device_does_not_strand_its_port(board, schema_path):
    """The whole chain a pulled cable used to take.

    pyserial's ``timeout`` setter reconfigures the open port, so it raises once
    the device is gone -- and ``HarpReader.stop()`` runs it on every shutdown to
    hand the borrowed timeout back. Unguarded, that exception left ``stop()``
    before ``shutdown()`` could close the handle, so ``_serial`` stayed set,
    ``initialize()`` refused forever, and re-plugging the cable did not help:
    only restarting the application recovered the port.
    """
    dev = make_device(board, schema_path)
    await dev.initialize()
    dev.port_handle.raise_on_timeout_set = True  # the cable is pulled here

    await dev.shutdown()

    assert dev.port_handle.closed, "the port was never released"
    assert dev.reader is None
    # And the device can be brought back up once the cable is replaced.
    dev.port_handle.raise_on_timeout_set = False
    dev.port_handle._replayed = threading.Event()
    await dev.initialize()
    assert dev.is_initialized
    await dev.shutdown()


async def test_shutdown_gives_the_standby_courtesy_a_small_budget(device):
    """``HardwareManager`` bounds shutdown() at 2 s, of which the reader's join
    may already have taken most. Two full-length round-trips on top overrun it,
    the caller's ``wait_for`` cancels part-way, and the port leaks.

    Returning the device to Standby is a courtesy; releasing the port is not.
    """
    seen: list[float] = []
    real = device._round_trip

    async def watched(address, payload_type, timeout=None, *args, **kwargs):
        seen.append(timeout)
        return await real(address, payload_type, timeout)

    device._round_trip = watched
    await device.shutdown()

    assert seen, "shutdown did not round-trip at all"
    assert all(t == SHUTDOWN_ROUND_TRIP_TIMEOUT_S for t in seen), seen
    assert sum(seen) < 2.0, "the Standby courtesy alone could exceed the caller's budget"


async def test_a_cancelled_shutdown_still_releases_the_port(board, schema_path):
    """The caller cancels at its own deadline. A shutdown that gives up
    without closing leaves a handle nothing in the process can reopen --
    the same dead end as the stranded port above, by a different route.

    ``CancelledError`` is a ``BaseException``, so this is exactly what a bare
    ``except Exception`` with no ``finally`` would have missed.
    """
    dev = make_device(board, schema_path)
    await dev.initialize()

    entered = threading.Event()
    release = threading.Event()
    answer = dev.port_handle._handle_request

    def slow(raw):
        frame = decode(raw)
        if frame.address == OPERATION_CONTROL:
            entered.set()
            release.wait(2.0)
        answer(raw)

    dev.port_handle._handle_request = slow
    task = asyncio.create_task(dev.shutdown())
    try:
        assert await asyncio.to_thread(entered.wait, 2.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()

    # The shielded close outlives the cancellation; give it the loop.
    for _ in range(200):
        if dev.port_handle.closed:
            break
        await asyncio.sleep(0.01)
    assert dev.port_handle.closed, "a cancelled shutdown leaked the port"


# --- a link that breaks mid-recording ------------------------------------


async def test_a_dead_reader_is_noticed_and_reported(board, schema_path, caplog):
    """The worst-shaped failure this device has.

    A cable pulled twenty minutes into a four-hour unattended run kills the
    reader thread. Without this the cache simply stops changing, so every row
    for the next three hours carries the last state, a count of zero and a
    frozen device time -- byte for byte what an animal that stopped licking
    looks like. A plausible result is worse than a broken one; nobody
    investigates it.
    """
    errors: list[Exception] = []
    board.register_error_callback(errors.append)
    dev = make_device(board, schema_path)
    await dev.initialize()
    try:
        assert dev.recording_warnings() == []
        # The port dies under the reader thread, which records why and exits.
        dev.port_handle.read = _dead_read
        assert await asyncio.to_thread(_wait_until_dead, dev.reader), "the reader never noticed"

        with caplog.at_level(logging.ERROR):
            await dev.get_state()

        warnings = dev.recording_warnings()
        assert len(warnings) == 1
        assert "reader thread stopped" in warnings[0]
        assert "is not a reading" in warnings[0]
        assert "reader thread stopped" in caplog.text
        assert board.state is BoardConnectionState.ERROR
        assert errors, "the board's error listeners were never told"
    finally:
        await dev.shutdown()


async def test_the_link_failure_is_reported_once_not_once_per_row(board, schema_path, caplog):
    """The recorder calls get_state() every row. A four-hour run at 30 fps is
    430,000 rows; one line each buries the log and 430,000 warning rows buries
    the CSV."""
    dev = make_device(board, schema_path)
    await dev.initialize()
    try:
        dev.port_handle.read = _dead_read
        assert await asyncio.to_thread(_wait_until_dead, dev.reader)
        with caplog.at_level(logging.ERROR):
            for _ in range(5):
                await dev.get_state()
            await dev.read()
        assert len(dev.recording_warnings()) == 1
        assert caplog.text.count("reader thread stopped") == 1
    finally:
        await dev.shutdown()


async def test_a_second_cable_pull_is_caught_after_a_reconnect(board, schema_path):
    """The detector has to be re-armed by initialize(), not merely emptied.

    ``_check_link`` returns early while the latch is set, so a device that is
    reconnected after a cable is fixed gets a fresh reader that nothing is
    watching. The second pull then goes unnoticed and the CSV goes back to
    looking exactly like a subject that stopped responding -- C2 regressing on
    every run after the first.

    Asserted on the *second* detection rather than on the warning list being
    empty after re-init: an implementation that clears the list and leaves the
    latch set passes the emptiness check and fails here, which is the whole
    point of the test.
    """
    dev = make_device(board, schema_path)
    await dev.initialize()
    real_read = dev.port_handle.read

    # First failure, detected.
    dev.port_handle.read = _dead_read
    assert await asyncio.to_thread(_wait_until_dead, dev.reader)
    await dev.get_state()
    first = dev.recording_warnings()
    assert len(first) == 1
    await dev.shutdown()

    # The cable is replaced and the device brought back up.
    dev.port_handle.read = real_read
    dev.port_handle.closed = False
    dev.port_handle._replayed = threading.Event()
    board._set_state(BoardConnectionState.CONNECTED)
    await dev.initialize()
    try:
        assert dev.recording_warnings() == [], "carried the dead link's warning into a new one"
        await dev.get_state()

        # Second failure, on the new reader.
        dev.port_handle.read = _dead_read
        assert await asyncio.to_thread(_wait_until_dead, dev.reader)
        await dev.get_state()

        second = dev.recording_warnings()
        assert len(second) == 1, "the second cable pull was never noticed"
        assert board.state is BoardConnectionState.ERROR
    finally:
        dev.port_handle.read = real_read
        await dev.shutdown()


async def test_a_healthy_reader_reports_nothing(device):
    """The check must not cry wolf on the ordinary case, or the warning stops
    meaning anything."""
    for _ in range(5):
        await device.get_state()
    assert device.recording_warnings() == []
    assert device.board.state is not BoardConnectionState.ERROR


async def test_shutdown_closes_the_port_even_if_standby_fails(board, schema_path):
    """A device that will not answer is still a device whose port has to be
    released; an unreleased handle blocks the next session."""
    dev = make_device(board, schema_path)
    await dev.initialize()

    def deaf(_raw):
        return None

    dev.port_handle._handle_request = deaf

    await dev.shutdown()
    assert dev.port_handle.closed
    assert not dev.is_initialized


# --- no profile ----------------------------------------------------------


async def test_a_device_with_no_profile_records_nothing_but_still_acts(board, schema_path):
    """``derive``'s third rule: widening the record is always deliberate, so
    an unrecognised board is silent until somebody writes a profile -- but its
    whole control surface is still reachable from the flow graph."""
    dev = make_device(board, schema_path, profile="")
    assert set(dev.actions) == {"StimulusOn", "Threshold"}

    await dev.initialize()
    try:
        assert dev.state_columns() is None
        assert await dev.get_state() is None
        assert await dev.read() is None
        assert dev.reader is None
        assert set(dev.actions) == {"StimulusOn", "Threshold"}
        # And left in Standby, which is the one configuration that wants it.
        # Active is what makes events flow; with no reader draining them they
        # would stream into a port nobody reads.
        assert dev.operation_control & 0x03 == 0x00
    finally:
        await dev.shutdown()


async def test_actions_are_listed_before_the_device_is_initialized(board, schema_path):
    """The node editor asks for these while the hardware is still in a box."""
    dev = make_device(board, schema_path)
    assert set(dev.actions) == {"StimulusOn", "Threshold"}
    # An Event-only register is the device talking to us; there is nothing to
    # call, and offering it would be an action that times out.
    assert "LickState" not in dev.actions


async def test_an_unreadable_schema_costs_actions_but_not_the_experiment(board, tmp_path):
    """A saved experiment must still open on a machine where the device.yml
    has moved; initialize() is where that becomes an error."""
    config = DeviceConfig(
        settings={"port": "COM-fake", "device_yml": str(tmp_path / "gone.yml"), "profile": ""}
    )
    dev = MockHarpDevice(board, config, "harp1")
    assert dev.actions == {}
    with pytest.raises(OSError):
        await dev.initialize()


# --- actions -------------------------------------------------------------


async def test_a_write_action_puts_the_register_on_the_wire(device):
    """Writes are legal during a recording: they read nothing back, so the
    reader consuming the echo costs nothing."""
    await device.execute_action("Threshold", 4096)
    written = [decode(frame) for frame in device.port_handle.writes]
    threshold = [f for f in written if f.address == 34 and f.message_type == MESSAGE_WRITE]
    assert threshold, "no write reached the port"
    # U16, little-endian, from the width the schema declared -- not a byte.
    assert int.from_bytes(threshold[-1].payload, "little") == 4096
    assert len(threshold[-1].payload) == 2


async def test_a_read_action_is_refused_while_the_reader_owns_the_port(device):
    """Not a timeout, and not a silent None: the reply is decoded by the
    reader thread and dropped, so this call could only ever wait for something
    that never arrives."""
    with pytest.raises(RuntimeError, match="while recording"):
        await device.execute_action("Threshold")


async def test_a_read_action_works_when_nothing_is_recording(board, schema_path):
    dev = make_device(board, schema_path, profile="", registers={34: (1234).to_bytes(2, "little")})
    await dev.initialize()
    try:
        assert await dev.execute_action("Threshold") == 1234
    finally:
        await dev.shutdown()


async def test_a_value_too_wide_for_the_register_is_refused(device):
    """Silently truncating would write a number the experimenter never asked
    for and log nothing."""
    with pytest.raises(ValueError, match="does not fit"):
        await device.execute_action("StimulusOn", 300)


# --- access modes: which half of an action a caller actually gets ---------


async def test_reading_a_write_only_register_is_refused_at_once(board, schema_path):
    """A Read sent to a write-only register gets no reply, so without the
    schema check the caller waits out the whole round-trip timeout and is then
    told the device did not answer -- which reads as broken hardware rather
    than as an action that was never readable.

    ``DeviceReadNode`` calls ``execute_action`` with no args, so this is the
    dedicated read node pointed at a write-only register.
    """
    dev = make_device(board, schema_path, profile="")
    await dev.initialize()
    try:
        quiet = len(dev.port_handle.writes)
        started = time.monotonic()
        with pytest.raises(ValueError, match="not readable"):
            await dev.execute_action("StimulusOn")
        assert time.monotonic() - started < ROUND_TRIP_TIMEOUT_S / 2, "waited for the wire"
        assert len(dev.port_handle.writes) == quiet, "sent a request nothing could answer"
    finally:
        await dev.shutdown()


async def test_writing_a_read_only_register_is_refused(board, tmp_path):
    """A device answers a write to a read-only register by ignoring it, so the
    alternative is an action that reports success and does nothing all
    session."""
    schema = {
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"Serial": {"address": 35, "type": "U16", "access": "Read"}},
    }
    path = tmp_path / "readonly.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    dev = make_device(board, path, profile="", registers={35: (7).to_bytes(2, "little")})
    await dev.initialize()
    try:
        quiet = len(dev.port_handle.writes)
        with pytest.raises(ValueError, match="not writable"):
            await dev.execute_action("Serial", 3)
        assert len(dev.port_handle.writes) == quiet, "sent a write the device would ignore"
        # And the readable half still works.
        assert await dev.execute_action("Serial") == 7
    finally:
        await dev.shutdown()


async def test_a_recorded_register_is_read_from_the_cache(board, tmp_path):
    """The register the reader already owns is answerable without the wire.

    This is what makes ``DeviceReadNode`` usable against a recording device at
    all: the round-trip cannot work while the reader consumes every reply, and
    the value it would have fetched is already in the cache. It also works for
    a Write+Event register, which the wire would refuse to read.
    """
    schema = {
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"LickState": {"address": 32, "type": "U8", "access": ["Write", "Event"]}},
    }
    path = tmp_path / "both.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    dev = make_device(board, path, frames=[event(32, 1, 0.5), event(32, 7, 0.9)])
    await dev.initialize()
    try:
        assert dev.wait_for_replay()
        assert await dev.execute_action("LickState") == 7
        # Non-consuming: a read node polling this must not eat the record.
        assert (await dev.get_state())["lick_count"] == 2
    finally:
        await dev.shutdown()


# --- value specs: how the runner tells a command from a measurement -------


async def test_a_writable_register_declares_its_width_as_its_range(device):
    """``DeviceControlsPanel`` classifies every control by ``value_spec``.

    Answering None for everything makes each Harp action a bare button that is
    invoked with no value -- i.e. every button on the runner panel becomes a
    read, which is wrong for every writable register and impossible for most.
    """
    spec = device.value_spec("Threshold")
    assert spec is not None
    assert spec.kind == KIND_WHOLE
    assert (spec.min, spec.max) == (0, 0xFFFF), "U16's own range, not a byte's"
    assert spec.validate() == []

    narrow = device.value_spec("StimulusOn")
    assert (narrow.min, narrow.max) == (0, 0xFF)


async def test_a_read_only_register_declares_no_value(board, tmp_path):
    """None is the answer that makes the runner draw a button that reads."""
    schema = {
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"Serial": {"address": 35, "type": "U16", "access": "Read"}},
    }
    path = tmp_path / "readonly.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    assert make_device(board, path, profile="").value_spec("Serial") is None


async def test_a_wide_register_declares_a_range_a_control_can_hold(board, tmp_path):
    """A U64's true range is meaningless to a Qt slider, whose bounds are
    32-bit. ``_pack`` stays the authority on what actually fits, so clamping
    the declared range cannot admit a bad write."""
    schema = {
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"Big": {"address": 36, "type": "U64", "access": "Write"}},
    }
    path = tmp_path / "wide.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    dev = make_device(board, path, profile="")

    spec = dev.value_spec("Big")
    assert spec.validate() == []
    assert spec.max == (1 << 31) - 1

    # The register itself still takes its full width.
    await dev.initialize()
    try:
        await dev.execute_action("Big", 1 << 40)
        assert len(decode(dev.port_handle.writes[-1]).payload) == 8
    finally:
        await dev.shutdown()


async def test_value_specs_are_available_before_initialize(board, schema_path):
    """The runner builds its panel from a hardware map, not from live ports."""
    assert make_device(board, schema_path).value_spec("Threshold") is not None


# --- warnings that have to outlive the log -------------------------------


async def test_recording_a_non_event_register_is_reported_for_the_csv(board, tmp_path):
    """``derive`` logs this, and a log line during an unattended overnight run
    reaches nobody. The recording is the artefact that survives."""
    schema = {
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"LickState": {"address": 32, "type": "U8", "access": "Write"}},
    }
    path = tmp_path / "quiet.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    dev = make_device(board, path)

    warnings = dev.recording_warnings()
    assert len(warnings) == 1
    assert "LickState" in warnings[0]
    assert "lick" in warnings[0]


async def test_an_event_register_produces_no_warning(device):
    assert device.recording_warnings() == []


async def test_the_warning_is_derive_s_own_finding(board, tmp_path):
    """The device passes on what ``derive`` reported rather than deciding
    again. One predicate, one wording, one place -- the second copy this
    replaced got the list-of-access-modes case wrong."""
    from glider_harp.derivation import derive, load_profile

    schema = {
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"LickState": {"address": 32, "type": "U8", "access": "Write"}},
    }
    path = tmp_path / "quiet.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")

    expected = derive(schema, load_profile("licketysplit")).warnings
    assert expected, "derive should have found this itself"
    assert make_device(board, path).recording_warnings() == expected


async def test_a_failed_device_still_names_the_columns_it_was_configured_for(board, schema_path):
    """A device that never came up has no cache and still has a profile.

    Answered from the cache alone it would collapse to one unnamed column
    while ``recording_warnings()`` went on describing columns no longer in the
    header. Empty cells say "not recording"; a missing column says nothing.
    """
    dev = make_device(board, schema_path)
    assert not dev.is_initialized
    assert dev.state_columns() == LICK_COLUMNS
    assert await dev.get_state() is None


async def test_a_list_of_access_modes_is_read_as_a_list(board, tmp_path):
    """``access: [Write, Event]`` is ordinary. Read as one opaque value it
    matches nothing, and every recorded register would be reported as silent
    -- a warning in every CSV is a warning nobody reads."""
    schema = {
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"LickState": {"address": 32, "type": "U8", "access": ["Write", "Event"]}},
    }
    path = tmp_path / "both.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    assert make_device(board, path).recording_warnings() == []


async def test_a_schema_that_names_no_board_is_taken_at_its_word(board, tmp_path):
    """A hand-written schema routinely omits ``whoAmI``.

    Rejecting one would make every schema not copied from the vendor unusable,
    which is the failure mode opposite to the one the check exists for --
    ``derive`` treats a missing declaration the same way.
    """
    schema = {"registers": dict(SCHEMA["registers"])}
    path = tmp_path / "anonymous.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")
    dev = make_device(board, path, profile="", who_am_i=9999)

    await dev.initialize()
    try:
        assert dev.who_am_i == 9999
        assert dev.is_initialized
    finally:
        await dev.shutdown()


# --- the fake's own contract ---------------------------------------------


async def test_wait_for_replay_reports_a_replay_that_never_arrived(board, schema_path):
    """The helper has to be able to say no, or every assertion behind it is
    conditional on timing nobody checked."""
    dev = make_device(board, schema_path, frames=[event(32, 1, 0.5)])
    await dev.initialize()
    try:
        assert dev.wait_for_replay(timeout=1.0)
        # A second wait on an exhausted replay stays satisfied; the event is
        # latched, not a one-shot signal.
        assert dev.wait_for_replay(timeout=0.01)
    finally:
        await dev.shutdown()

    stalled = make_device(board, schema_path, frames=[event(32, 1, 0.5)])
    assert not stalled.wait_for_replay(timeout=0.05), "reported a replay that never started"


async def test_the_fake_answers_who_am_i_from_the_schema(device):
    """So a test never has to state the board's identity in two places."""
    assert device.who_am_i == LICKETYSPLIT_WHO_AM_I


async def test_frames_round_trip_through_the_real_codec():
    """Both directions go through the shipped codec, so a test that passes
    here is not agreeing with a private frame format of its own."""
    event_frame = decode(event(32, 1, 1.5))
    assert event_frame.message_type == MESSAGE_EVENT
    assert event_frame.address == 32
    assert event_frame.timestamp == pytest.approx(1.5)

    built = decode(encode(MESSAGE_WRITE, OPERATION_CONTROL, "U8", b"\x01"))
    assert built.message_type == MESSAGE_WRITE
    assert built.address == OPERATION_CONTROL
    assert built.payload == b"\x01"
    assert built.timestamp is None


async def test_shutdown_before_initialize_is_harmless(board, schema_path):
    """Emergency stop calls shutdown() on everything it can reach, including
    devices that never came up."""
    dev = make_device(board, schema_path)
    await dev.shutdown()
    assert not dev.is_initialized


async def test_concurrent_reads_and_the_reader_do_not_deadlock(device):
    """The recorder polls get_state() while the reader thread ingests; neither
    may hold the other up."""
    results = await asyncio.gather(*(device.read() for _ in range(20)))
    assert all(list(r) == LICK_COLUMNS for r in results)


async def test_a_vanished_device_surfaces_the_read_failure_not_the_timeout_restore():
    """`_exchange` borrows the handle's read timeout and puts it back in a
    `finally`. On pyserial that assignment reconfigures the open port, and it
    raises when the device has been unplugged -- which is exactly when the read
    it wraps has just failed. Unguarded, the operator sees "cannot reconfigure
    a port" instead of the disconnect. Same rationale as
    ``HarpReader._restore_timeout``."""
    from glider_harp.device import _exchange

    class VanishingHandle:
        in_waiting = 0

        def __init__(self) -> None:
            self._timeout_sets = 0

        @property
        def timeout(self) -> float:
            return 1.0

        @timeout.setter
        def timeout(self, value: float) -> None:
            self._timeout_sets += 1
            if self._timeout_sets > 1:  # the restore, after the device is gone
                raise OSError("could not reconfigure port: device disconnected")

        def write(self, data: bytes) -> None:
            pass

        def read(self, size: int) -> bytes:
            raise OSError("device vanished mid-exchange")

    with pytest.raises(OSError, match="vanished mid-exchange"):
        _exchange(VanishingHandle(), b"\x01", 32, timeout=0.2, device_name="test")


# --- a profile the lab wrote, and the types it can now record -------------
#
# Two features that meet in one place: a second Harp device needs a profile
# that does not live inside an installed package, and the device it describes
# is unlikely to report only unsigned bytes.

SIGNED_SCHEMA = {
    "device": "SignedRig",
    "whoAmI": LICKETYSPLIT_WHO_AM_I,
    "registers": {
        # Written *and* reported back, which is what makes a round trip
        # observable at all.
        "Offset": {"address": 40, "type": "S16", "access": ["Write", "Event"]},
    },
}


@pytest.fixture
def signed_rig(tmp_path, user_profiles):
    """A signed Write+Event register, described by a profile in the user directory."""
    path = tmp_path / "signed.yml"
    path.write_text(yaml.safe_dump(SIGNED_SCHEMA), encoding="utf-8")
    (user_profiles / "signedrig.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "name": "SignedRig",
                "who_am_i": LICKETYSPLIT_WHO_AM_I,
                "record": [{"register": "Offset", "as": "offset"}],
            }
        ),
        encoding="utf-8",
    )
    return path


async def test_a_signed_register_written_as_minus_one_records_as_minus_one(board, signed_rig):
    """The asymmetry the recordable gate existed to prevent, now closed.

    ``_pack`` has always sent -1 out as ``ff ff``, correctly signed for an S16.
    The cache read it back as 65535, so a value written and reported back
    inside one program came out as a different number -- and the CSV that held
    it opened cleanly. Both halves are the point: the bytes on the wire are the
    device's own, and the number that reaches the record is the one that was
    written.
    """
    dev = make_device(board, signed_rig, profile="signedrig")
    await dev.initialize()
    try:
        await dev.actions["Offset"](-1)
        written = decode(dev.port_handle.writes[-1])
        assert written.payload == b"\xff\xff", "the write was not packed as a signed S16"

        # Replayed as the device reporting the register it was just given.
        dev.port_handle.queue(encode(MESSAGE_EVENT, 40, "S16", written.payload))
        assert dev.wait_for_replay(), "the replayed frame never reached the cache"
        state = await dev.get_state()
    finally:
        await dev.shutdown()

    assert state["offset_state"] == -1


async def test_a_user_profile_drives_a_real_device(board, signed_rig):
    """The end the directory exists for: a profile nobody shipped, naming the
    columns of a board nobody shipped a profile for."""
    dev = make_device(board, signed_rig, profile="signedrig")
    await dev.initialize()
    try:
        assert dev.state_columns() == ["offset_state", "offset_count", "offset_last_ms"]
    finally:
        await dev.shutdown()


async def test_a_profile_name_that_names_nothing_still_fails_at_initialize(board, schema_path):
    """A user directory does not turn a typo into a device that records
    nothing; it is still an error, and it still names what is available."""
    dev = make_device(board, schema_path, profile="notaprofile")

    with pytest.raises(FileNotFoundError, match="notaprofile"):
        await dev.initialize()


# --- the dropdown a person picks a profile from --------------------------


def test_the_dropdown_offers_shipped_and_user_profiles(user_profiles):
    (user_profiles / "ourrig.json").write_text("{}", encoding="utf-8")

    values = [value for value, _ in device_module._profile_choices()]

    assert values[0] == "", "the empty choice -- record nothing -- must stay first"
    assert "licketysplit" in values and "ourrig" in values


def test_the_dropdown_says_which_profiles_are_the_labs_own(user_profiles):
    """Free text was rejected here because a typo surfaces on a bench with the
    animal already in the rig. A user profile silently replacing a shipped one
    is the same class of surprise, so the label carries the provenance."""
    (user_profiles / "ourrig.json").write_text("{}", encoding="utf-8")
    (user_profiles / "licketysplit.json").write_text("{}", encoding="utf-8")

    labels = dict(device_module._profile_choices())

    assert "user" in labels["ourrig"].lower()
    assert "overrides" in labels["licketysplit"].lower()


def test_the_dropdown_leaves_a_purely_shipped_profile_unadorned():
    labels = dict(device_module._profile_choices())

    assert labels["licketysplit"] == "licketysplit"


def test_an_unreadable_user_profile_directory_does_not_break_the_dropdown(monkeypatch):
    """``_profile_choices`` runs while the class body is executing, so an
    exception here does not misconfigure a device -- it stops the plugin from
    importing at all."""

    class Exploding:
        def glob(self, _pattern):
            raise OSError("permission denied")

    monkeypatch.setattr(derivation, "user_profile_dir", Exploding)

    assert "licketysplit" in dict(device_module._profile_choices())


async def test_a_signed_register_read_over_the_wire_comes_back_signed(board, tmp_path):
    """The read path that does not go through the cache at all.

    A register nothing records is answered by a round trip, and it has to
    decode by the same rule the record does: a DeviceRead node returning -1
    while the same register recorded returns 65535 would be a contradiction
    with nothing in the graph able to show its cause.
    """
    schema = {
        "device": "SignedRig",
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {"Offset": {"address": 40, "type": "S16", "access": ["Read", "Write"]}},
    }
    path = tmp_path / "device.yml"
    path.write_text(yaml.safe_dump(schema), encoding="utf-8")

    # No profile: nothing is recorded, so no reader owns the port and the read
    # is a real round trip.
    dev = make_device(board, path, profile="")
    await dev.initialize()
    try:
        await dev.actions["Offset"](-1)
        assert await dev.actions["Offset"]() == -1
    finally:
        await dev.shutdown()
