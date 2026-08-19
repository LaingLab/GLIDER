"""Per-register cache backing the CSV columns."""

import struct
import threading

import pytest

from glider_harp.frames import HarpFrame
from glider_harp.reader import RegisterCache


def _event(address, value, timestamp=1.0):
    return HarpFrame(3, address, 0xFF, 0x11, bytes([value]), timestamp)


def test_snapshot_reports_latest_state():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    assert cache.snapshot()["lick_state"] == 1


def test_count_accumulates_between_snapshots():
    cache = RegisterCache({32: "lick"})
    for _ in range(3):
        cache.ingest(_event(32, 1))
    assert cache.snapshot()["lick_count"] == 3


def test_count_clears_on_read():
    """Counts mean 'since last poll' -- this is what makes the column honest."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    cache.snapshot()
    assert cache.snapshot()["lick_count"] == 0


def test_state_persists_across_snapshots():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    cache.snapshot()
    assert cache.snapshot()["lick_state"] == 1


def test_last_ms_records_device_time_in_milliseconds():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=2.5))
    assert cache.snapshot()["lick_last_ms"] == 2500.0


def test_unmapped_register_is_ignored():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(99, 1))
    assert cache.snapshot()["lick_count"] == 0


def test_columns_match_snapshot_keys():
    cache = RegisterCache({32: "lick"})
    assert set(cache.columns()) == set(cache.snapshot().keys())


# --- what an empty cache reports before anything arrives ---


def test_snapshot_reports_every_column_before_any_event():
    """A row written before the device says anything still needs all its columns."""
    cache = RegisterCache({32: "lick", 33: "poke"})
    assert cache.snapshot() == {
        "lick_state": None,
        "lick_count": 0,
        "lick_last_ms": None,
        "poke_state": None,
        "poke_count": 0,
        "poke_last_ms": None,
    }


def test_columns_are_ordered_by_register_and_field():
    """The CSV header comes from here, so its order is part of the contract."""
    cache = RegisterCache({32: "lick", 33: "poke"})
    assert cache.columns() == [
        "lick_state",
        "lick_count",
        "lick_last_ms",
        "poke_state",
        "poke_count",
        "poke_last_ms",
    ]


def test_columns_agree_with_snapshot_order_too():
    cache = RegisterCache({32: "lick", 33: "poke"})
    assert cache.columns() == list(cache.snapshot().keys())


def test_each_snapshot_is_a_fresh_dict():
    """A caller batching rows holds onto them; a shared dict would rewrite history.

    Reusing one dict makes row N silently change when row N+1 is taken -- a
    corruption that looks like nothing at all, since every value in it is one
    the cache really did report at some point.
    """
    cache = RegisterCache({32: "lick"})
    first = cache.snapshot()
    cache.ingest(_event(32, 1, timestamp=5.0))
    second = cache.snapshot()
    assert first is not second
    assert first == {"lick_state": None, "lick_count": 0, "lick_last_ms": None}
    assert second["lick_count"] == 1


def test_columns_do_not_change_when_the_caller_mutates_its_map():
    """The CSV header is written once; a later edit must not reshape the rows."""
    registers = {32: "lick"}
    cache = RegisterCache(registers)
    registers[33] = "poke"
    assert cache.columns() == ["lick_state", "lick_count", "lick_last_ms"]
    assert list(cache.snapshot()) == cache.columns()


def test_no_column_contains_a_colon():
    """These are sub-columns: the recorder prefixes '{device_id}:', so a colon
    here would yield 'harp1:lick:state' and nothing could tell which colon
    separated the pair. BaseDevice.state_columns forbids it."""
    cache = RegisterCache({32: "lick", 33: "poke"})
    assert all(":" not in column for column in cache.columns())


def test_duplicate_columns_are_rejected():
    """Two addresses under one name would collide in the CSV header."""
    with pytest.raises(ValueError, match="lick_state"):
        RegisterCache({32: "lick", 33: "lick"})


def test_a_colon_in_a_register_name_is_rejected():
    with pytest.raises(ValueError, match=":"):
        RegisterCache({32: "lick:left"})


def test_an_empty_register_name_is_rejected():
    """'_state' passes every column rule and still names nothing."""
    with pytest.raises(ValueError, match="non-empty"):
        RegisterCache({32: ""})


def test_an_empty_register_map_is_rejected():
    """columns() == [] reads to DataRecorder as single-column behaviour.

    It emits one header, '{device_id}:{device_type}', then looks the device
    *type* up in the dict get_state() returns -- a key that is never there --
    and writes an empty cell in every row. The device records nothing at all,
    and nothing raises.
    """
    with pytest.raises(ValueError, match="at least one register"):
        RegisterCache({})


# --- only Event frames count ---


@pytest.mark.parametrize("message_type", [1, 2], ids=["read", "write"])
def test_non_event_frames_do_not_count(message_type):
    """Read=1 and Write=2 are our own traffic echoed back, not device activity."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(HarpFrame(message_type, 32, 0xFF, 0x11, bytes([1]), 1.0))
    values = cache.snapshot()
    assert values["lick_count"] == 0
    assert values["lick_state"] is None
    assert values["lick_last_ms"] is None


def test_non_event_frame_does_not_overwrite_a_recorded_event():
    """A Write echo carries the value we sent, not one the device reported."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=2.0))
    cache.ingest(HarpFrame(2, 32, 0xFF, 0x11, bytes([0]), 9.0))
    values = cache.snapshot()
    assert values["lick_state"] == 1
    assert values["lick_count"] == 1
    assert values["lick_last_ms"] == 2000.0


# --- state ---


def test_state_reports_the_most_recent_value_not_the_first():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    cache.ingest(_event(32, 0))
    assert cache.snapshot()["lick_state"] == 0


@pytest.mark.parametrize("value", [0, 1, 2, 7, 128, 255])
def test_state_reports_the_value_it_was_given(value):
    """Swept, so a passing assertion cannot be an artefact of one chosen constant."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, value))
    assert cache.snapshot()["lick_state"] == value


def test_multi_byte_payload_is_read_little_endian():
    """U16/U32 registers are what the counter registers report."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(HarpFrame(3, 32, 0xFF, 0x12, (513).to_bytes(2, "little"), 1.0))
    assert cache.snapshot()["lick_state"] == 513


def test_empty_payload_counts_the_event_but_reports_no_value():
    """The event happened; there is no value in it to report, so state says so."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(HarpFrame(3, 32, 0xFF, 0x11, b"", 1.0))
    values = cache.snapshot()
    assert values["lick_count"] == 1
    assert values["lick_state"] is None
    assert values["lick_last_ms"] == 1000.0


def test_empty_payload_clears_a_previously_known_state():
    """Carrying the old value forward would date-stamp it to the event just counted."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    cache.ingest(HarpFrame(3, 32, 0xFF, 0x11, b"", 2.0))
    assert cache.snapshot()["lick_state"] is None


# --- count ---


@pytest.mark.parametrize("events", [1, 2, 5, 17])
def test_count_reports_exactly_the_events_ingested(events):
    cache = RegisterCache({32: "lick"})
    for _ in range(events):
        cache.ingest(_event(32, 1))
    assert cache.snapshot()["lick_count"] == events


def test_count_resumes_from_zero_after_a_read():
    """Every event lands in exactly one row: none double-counted, none dropped."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    cache.ingest(_event(32, 1))
    assert cache.snapshot()["lick_count"] == 2
    cache.ingest(_event(32, 1))
    assert cache.snapshot()["lick_count"] == 1


def test_repeated_events_at_the_same_level_still_count():
    """A 20 ms lick inside one 33 ms poll is invisible in state; count is why."""
    cache = RegisterCache({32: "lick"})
    for _ in range(4):
        cache.ingest(_event(32, 1))
    values = cache.snapshot()
    assert values["lick_count"] == 4
    assert values["lick_state"] == 1


# --- last_ms ---


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [(0.0, 0.0), (0.001, 1.0), (1.0, 1000.0), (2.5, 2500.0), (12.345, 12345.0)],
)
def test_last_ms_converts_seconds_to_milliseconds(timestamp, expected):
    """Swept: seconds, microseconds and identity all differ across these rows."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=timestamp))
    assert cache.snapshot()["lick_last_ms"] == pytest.approx(expected)


def test_last_ms_keeps_sub_millisecond_precision():
    """Device time ticks in 32 us units, so sub-millisecond digits are real data.

    A Harp timestamp is Seconds(U32) + Micros(U16) at 32 us per tick. Rounding
    to whole milliseconds would throw away resolution the device actually has,
    which is the resolution a lick-to-reward latency is measured in.
    """
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=7 + 501 * 32e-6))
    assert cache.snapshot()["lick_last_ms"] == pytest.approx(7016.032)


def test_last_ms_tracks_the_most_recent_event():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=1.0))
    cache.ingest(_event(32, 0, timestamp=4.0))
    assert cache.snapshot()["lick_last_ms"] == 4000.0


def test_last_ms_persists_across_snapshots():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=2.5))
    cache.snapshot()
    assert cache.snapshot()["lick_last_ms"] == 2500.0


def test_untimestamped_event_reports_no_device_time():
    """Frames without the timestamp flag are real; they carry no device clock."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(HarpFrame(3, 32, 0xFF, 0x01, bytes([1]), None))
    values = cache.snapshot()
    assert values["lick_count"] == 1
    assert values["lick_state"] == 1
    assert values["lick_last_ms"] is None


def test_untimestamped_event_clears_an_earlier_device_time():
    """Keeping 1000.0 here would time the event just counted to an older one.

    A row would then read count=1 with a last_ms that never moved, which is
    indistinguishable from a row where nothing happened -- exactly the silent
    loss the counter exists to prevent.
    """
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=1.0))
    cache.ingest(HarpFrame(3, 32, 0xFF, 0x01, bytes([1]), None))
    assert cache.snapshot()["lick_last_ms"] is None


# --- peek: the non-consuming read, for callers that do not own the record ---


def test_peek_reports_what_snapshot_would():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1, timestamp=2.5))
    assert cache.peek() == {"lick_state": 1, "lick_count": 1, "lick_last_ms": 2500.0}


def test_peek_does_not_clear_the_counters():
    """The whole point: a second poller must not consume the record's counts.

    WaitForInput and the Input node both try device.read() before get_state(),
    the latter on a 50 ms loop, so this is what stops an Input node dropped
    onto a Harp device from eating counts twenty times a second.
    """
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    cache.ingest(_event(32, 1))
    cache.peek()
    assert cache.snapshot()["lick_count"] == 2


def test_repeated_peeks_report_the_same_counts():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    assert cache.peek()["lick_count"] == 1
    assert cache.peek()["lick_count"] == 1
    assert cache.peek()["lick_count"] == 1


def test_peek_reports_the_interval_since_the_last_snapshot():
    """An unsynchronised observer sees a partial interval; that is correct."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(32, 1))
    cache.snapshot()
    assert cache.peek()["lick_count"] == 0
    cache.ingest(_event(32, 1))
    assert cache.peek()["lick_count"] == 1


def test_peek_keys_match_columns():
    cache = RegisterCache({32: "lick", 33: "poke"})
    assert list(cache.peek()) == cache.columns()


def test_each_peek_is_a_fresh_dict():
    cache = RegisterCache({32: "lick"})
    assert cache.peek() is not cache.peek()


def test_a_second_poller_peeking_steals_no_counts():
    """A 50 ms Input-node poll running against the recorder's snapshot loop.

    Asserted on the total across snapshots: peeks are interleaved with ingests
    and snapshots, and not one of them may consume an event.
    """
    cache = RegisterCache({32: "lick"})
    events = 400
    done = threading.Event()
    total = 0

    def peeker():
        while not done.is_set():
            cache.peek()

    poller = threading.Thread(target=peeker)
    poller.start()
    try:
        for _ in range(events):
            cache.ingest(_event(32, 1))
            total += cache.snapshot()["lick_count"]
    finally:
        done.set()
        poller.join()

    total += cache.snapshot()["lick_count"]
    assert total == events


# --- registers are independent ---


def test_registers_do_not_share_counts():
    cache = RegisterCache({32: "lick", 33: "poke"})
    cache.ingest(_event(32, 1))
    cache.ingest(_event(32, 1))
    cache.ingest(_event(33, 1))
    values = cache.snapshot()
    assert values["lick_count"] == 2
    assert values["poke_count"] == 1


def test_registers_do_not_share_state_or_time():
    cache = RegisterCache({32: "lick", 33: "poke"})
    cache.ingest(_event(32, 1, timestamp=1.0))
    cache.ingest(_event(33, 7, timestamp=3.0))
    values = cache.snapshot()
    assert values["lick_state"] == 1
    assert values["lick_last_ms"] == 1000.0
    assert values["poke_state"] == 7
    assert values["poke_last_ms"] == 3000.0


def test_reading_one_register_does_not_clear_another():
    """One snapshot clears every counter, so no register lags a row behind."""
    cache = RegisterCache({32: "lick", 33: "poke"})
    cache.ingest(_event(32, 1))
    cache.ingest(_event(33, 1))
    cache.snapshot()
    values = cache.snapshot()
    assert values["lick_count"] == 0
    assert values["poke_count"] == 0


def test_unmapped_register_does_not_leak_into_a_mapped_one():
    """Two mapped registers, so 'ignored' cannot pass by landing in the only one."""
    cache = RegisterCache({32: "lick", 33: "poke"})
    for _ in range(5):
        cache.ingest(_event(99, 1))
    values = cache.snapshot()
    assert values["lick_count"] == 0
    assert values["poke_count"] == 0
    assert values["lick_state"] is None
    assert values["poke_state"] is None


def test_unmapped_register_adds_no_columns():
    cache = RegisterCache({32: "lick"})
    cache.ingest(_event(99, 1))
    assert set(cache.snapshot()) == {"lick_state", "lick_count", "lick_last_ms"}


# --- thread safety ---


def test_concurrent_ingest_and_snapshot_lose_no_events():
    """The reader thread writes while the event loop reads.

    Asserted on the total across every snapshot, never on one snapshot: which
    events land in which row is genuinely nondeterministic, but the sum is not.
    A lost update -- a snapshot clearing an increment it did not report -- makes
    the total fall short.
    """
    cache = RegisterCache({32: "lick", 33: "poke"})
    writers, per_writer = 8, 250
    start = threading.Barrier(writers + 1)
    done = threading.Event()
    totals = {"lick_count": 0, "poke_count": 0}

    def write(address):
        start.wait()
        for _ in range(per_writer):
            cache.ingest(_event(address, 1))

    def read():
        start.wait()
        while not done.is_set():
            for column, value in cache.snapshot().items():
                if column in totals:
                    totals[column] += value

    threads = [
        threading.Thread(target=write, args=(32 if index % 2 == 0 else 33,))
        for index in range(writers)
    ]
    reader = threading.Thread(target=read)
    reader.start()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    done.set()
    reader.join()

    for column, value in cache.snapshot().items():
        if column in totals:
            totals[column] += value

    assert totals["lick_count"] == writers // 2 * per_writer
    assert totals["poke_count"] == writers // 2 * per_writer


def test_concurrent_ingest_leaves_a_consistent_final_state():
    """Every event carries the same value, so no interleaving can produce another."""
    cache = RegisterCache({32: "lick"})
    threads = [
        threading.Thread(target=lambda: [cache.ingest(_event(32, 3)) for _ in range(200)])
        for _ in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    values = cache.snapshot()
    assert values["lick_count"] == 6 * 200
    assert values["lick_state"] == 3


# --- decoding by declared type -------------------------------------------
#
# The cache used to read every payload as one unsigned little-endian integer,
# and ``derivation`` refused to record anything else because of it. These pin
# the half that had to move first.


def _payload_event(address, payload, timestamp=1.0):
    """One Event frame carrying exactly these payload bytes."""
    return HarpFrame(3, address, 0xFF, 0x11, bytes(payload), timestamp)


@pytest.mark.parametrize(
    "declared, payload, expected",
    [
        ("S8", b"\xff", -1),
        ("S8", b"\x7f", 127),
        ("S16", b"\xff\xff", -1),
        ("S16", (-1234).to_bytes(2, "little", signed=True), -1234),
        ("S32", (-70000).to_bytes(4, "little", signed=True), -70000),
        ("S64", (-(2**40)).to_bytes(8, "little", signed=True), -(2**40)),
    ],
)
def test_a_signed_register_decodes_as_signed(declared, payload, expected):
    """An S16 of -1 used to be recorded as 65535, in a CSV that opened cleanly."""
    cache = RegisterCache({32: "offset"}, {32: declared})
    cache.ingest(_payload_event(32, payload))

    assert cache.snapshot()["offset_state"] == expected


@pytest.mark.parametrize("declared", ["U8", "U16", "U32", "U64"])
def test_an_unsigned_register_still_decodes_as_unsigned(declared):
    """The top bit is a value, not a sign: 255 in a U8 is 255."""
    cache = RegisterCache({32: "level"}, {32: declared})
    cache.ingest(_payload_event(32, b"\xff"))

    assert cache.snapshot()["level_state"] == 255


def test_a_float_register_decodes_as_a_float():
    """1.5 used to be recorded as 1069547520."""
    cache = RegisterCache({32: "temp"}, {32: "Float"})
    cache.ingest(_payload_event(32, struct.pack("<f", 1.5)))

    assert cache.snapshot()["temp_state"] == pytest.approx(1.5)


def test_a_float_register_reports_a_negative_value_as_negative():
    cache = RegisterCache({32: "temp"}, {32: "Float"})
    cache.ingest(_payload_event(32, struct.pack("<f", -0.25)))

    assert cache.snapshot()["temp_state"] == pytest.approx(-0.25)


def test_a_float_payload_of_the_wrong_width_reports_no_value():
    """A Float register whose device sends three bytes is a schema that
    disagrees with the hardware. Reported as unknown -- a blank column is a
    visible gap; a number invented from three bytes is not -- and the event is
    still counted, because it did happen.
    """
    cache = RegisterCache({32: "temp"}, {32: "Float"})
    cache.ingest(_payload_event(32, b"\x00\x00\x80"))

    values = cache.snapshot()
    assert values["temp_state"] is None
    assert values["temp_count"] == 1


def test_a_cache_given_no_types_reads_every_payload_unsigned():
    """The default is what the cache has always done, so a caller that has no
    schema to hand -- ``_columns_for_recorded``, and every test written before
    types existed -- keeps working."""
    cache = RegisterCache({32: "lick"})
    cache.ingest(_payload_event(32, b"\xff\xff"))

    assert cache.snapshot()["lick_state"] == 65535


def test_types_are_per_register_not_per_cache():
    cache = RegisterCache({32: "lick", 33: "offset"}, {33: "S16"})
    cache.ingest(_payload_event(32, b"\xff\xff"))
    cache.ingest(_payload_event(33, b"\xff\xff"))

    values = cache.snapshot()
    assert values["lick_state"] == 65535
    assert values["offset_state"] == -1


def test_a_type_the_cache_cannot_decode_is_rejected_when_the_cache_is_built():
    """The last line of defence behind ``derivation``'s gate. A type nobody
    decodes has to raise while a device is being configured, not silently
    become an unsigned column halfway through a trial."""
    with pytest.raises(ValueError, match="Uint16"):
        RegisterCache({32: "lick"}, {32: "Uint16"})


def test_a_type_for_an_unknown_address_is_rejected():
    """``recorded`` and its types are two dicts that have to agree; a type for
    an address the cache was not given means they have drifted."""
    with pytest.raises(ValueError, match="33"):
        RegisterCache({32: "lick"}, {33: "S16"})


def test_an_empty_payload_still_reports_no_value_whatever_the_type():
    for declared in ("U16", "S16", "Float"):
        cache = RegisterCache({32: "reg"}, {32: declared})
        cache.ingest(_payload_event(32, b""))
        values = cache.snapshot()
        assert values["reg_state"] is None, declared
        assert values["reg_count"] == 1, declared
