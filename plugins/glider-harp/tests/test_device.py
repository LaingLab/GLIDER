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
import threading

import pytest
import yaml

from glider.hal.base_device import DeviceConfig
from glider_harp.board import HarpBoard
from glider_harp.device import OPERATION_CONTROL
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
