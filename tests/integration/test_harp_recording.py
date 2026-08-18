"""End-to-end: a Harp device recorded to CSV by the real ``DataRecorder``.

Every layer below this has been proved on its own -- framing, the register
cache, the reader thread, the device lifecycle, the recorder's multi-column
support. This is the test that they compose: a ``MockHarpDevice`` replaying
event frames, attached to a real ``HardwareManager``, sampled by a real
``DataRecorder``, producing a real CSV that the real analysis reader parses.

Only the serial handle is fake. The frames are decoded by the shipping codec,
counted by the shipping cache, drained by the shipping reader thread, named by
the shipping profile, and written by the shipping recorder.

Rows are driven through ``record_at_frame`` rather than the sampling timer.
The timer would work, but the questions here are all about *which row* a value
lands in -- "the count cleared between rows" is not a statement a test can make
about rows it did not decide the boundaries of.

Two things about the fixture are load-bearing and easy to undo:

* ``wait_for_replay()`` is how the event loop learns that the reader thread
  has ingested the replay. Its precondition is that nothing but the reader
  touches the fake handle, so no test here reads or drains it.
* ``get_state()`` clears the event counters and ``DataRecorder`` calls it once
  per row. A test that calls it itself takes the counts out of the row it was
  about to assert on.
"""

from __future__ import annotations

import csv
import time

import pytest
import yaml

from glider.analysis._io import parse_csv
from glider.core.data_recorder import DataRecorder
from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import DeviceConfig, DigitalOutputDevice
from glider.hal.mock_board import MockBoard

# The plugin lives outside ``src`` and is not on the path for the setup CLAUDE.md
# documents (``PYTHONPATH=src``). A hard import here aborts *collection* of the
# whole run, so someone who has not installed the plugin gets zero tests instead
# of one skip. Same convention as the sklearn/torch/lightgbm suites.
pytest.importorskip("glider_harp", reason="the glider-harp plugin is not installed")

from glider_harp.board import HarpBoard  # noqa: E402
from glider_harp.mock import MockHarpDevice  # noqa: E402

# The shipped ``licketysplit`` profile records LickState as "lick" and is
# written for WhoAmI 1400, so any schema used with it has to agree on both.
LICKETYSPLIT_WHO_AM_I = 1400
LICK_STATE_ADDRESS = 32

# What the profile's one recorded register expands into, in order.
LICK_COLUMNS = ["lick_state", "lick_count", "lick_last_ms"]


def _schema(lick_access: str | list[str] = "Event") -> dict:
    """A LicketySplit-shaped ``device.yml`` mapping.

    ``lick_access`` is a parameter because the non-Event case is a whole test:
    a profile may name a register the schema does not declare as reporting,
    and the column then never changes.
    """
    return {
        "device": "LicketySplit",
        "whoAmI": LICKETYSPLIT_WHO_AM_I,
        "registers": {
            "LickState": {"address": LICK_STATE_ADDRESS, "type": "U8", "access": lick_access},
            "StimulusOn": {"address": 33, "type": "U8", "access": "Write"},
        },
    }


def _write_schema(tmp_path, name: str = "device.yml", **kwargs) -> str:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(_schema(**kwargs)), encoding="utf-8")
    return str(path)


def event(address: int, value: int, timestamp: float | None = None) -> bytes:
    """One Event frame, as the device would put it on the wire."""
    from harp.protocol import HarpMessage, MessageType, PayloadType

    return HarpMessage(
        MessageType.Event, address, PayloadType.U8, bytes([value]), timestamp=timestamp
    ).bytes


# --- fixture assembly ---------------------------------------------------


def _make_harp(
    board: HarpBoard,
    device_yml: str,
    name: str,
    *,
    profile: str = "licketysplit",
    frames=(),
) -> MockHarpDevice:
    config = DeviceConfig(
        settings={
            "port": f"COM-fake-{name}",
            "baudrate": 115200,
            "device_yml": device_yml,
            "profile": profile,
        }
    )
    return MockHarpDevice(board, config, name, frames=frames)


def _hardware_manager(*boards) -> HardwareManager:
    """A real HardwareManager holding the given boards and nothing else.

    Populated through the private dicts, as ``tests/unit/core/
    test_data_recorder_frame.py`` does: ``add_board``/``add_device`` build
    from serializable configs and would need a registered driver factory for
    a board type this test constructs directly.
    """
    hw = HardwareManager()
    for board in boards:
        hw._boards[board.id] = board
    return hw


def _attach(hw: HardwareManager, device) -> None:
    hw._devices[device.id] = device


def _read_csv_rows(path) -> tuple[list[str], list[list[str]]]:
    """Return (header_row, data_rows), skipping metadata comment rows."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header_idx = next(i for i, r in enumerate(rows) if r and not r[0].startswith("#") and r[0])
    data_rows = [r for r in rows[header_idx + 1 :] if r and not r[0].startswith("#")]
    return rows[header_idx], data_rows


def _frame_rows(path) -> tuple[list[str], list[list[str]]]:
    """Only the rows written by ``record_at_frame``.

    ``stop()`` writes one final timer-style row (empty ``frame`` cell) after
    cancelling the sampling task, and that row samples the devices again --
    so it is a real row with a real, cleared count in it. Every assertion
    here is about rows this test decided the boundaries of.
    """
    header, rows = _read_csv_rows(path)
    return header, [r for r in rows if r and r[0].isdigit()]


def _cell(header: list[str], row: list[str], column: str) -> str:
    return row[header.index(column)]


def _comment_lines(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f if line.startswith("#")]


# --- 1. columns reach the header ----------------------------------------


async def test_harp_columns_reach_the_csv_header(tmp_path):
    """The profile's one register becomes three CSV columns, device-prefixed.

    Note the two separators: ``_`` inside the base name (the cache's doing)
    and ``:`` between device id and sub-column (the recorder's). They differ
    on purpose -- the recorder recovers the sub-column by partitioning on the
    first colon, so a colon in the base name would be unrecoverable.
    """
    board = HarpBoard()
    harp = _make_harp(board, _write_schema(tmp_path), "harp1")
    hw = _hardware_manager(board)
    _attach(hw, harp)
    await harp.initialize()

    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)
    await recorder.start("harp_header")
    try:
        header, _ = _read_csv_rows(recorder.file_path)
    finally:
        await recorder.stop()
        await harp.shutdown()

    expected = [f"{harp.id}:{name}" for name in LICK_COLUMNS]
    assert header[-3:] == expected, header
    # The fixed prefix is untouched by a multi-column device.
    assert header[:4] == ["frame", "timestamp", "elapsed_ms", "flow_elapsed_ms"]


# --- 2. values reach the rows -------------------------------------------


async def test_replayed_events_reach_the_row(tmp_path):
    """Three licks replayed before the row is written must be three in it."""
    board = HarpBoard()
    frames = [
        event(LICK_STATE_ADDRESS, 1, 0.5),
        event(LICK_STATE_ADDRESS, 0, 0.75),
        event(LICK_STATE_ADDRESS, 1, 1.0),
    ]
    harp = _make_harp(board, _write_schema(tmp_path), "harp1", frames=frames)
    hw = _hardware_manager(board)
    _attach(hw, harp)
    await harp.initialize()
    assert harp.wait_for_replay(), "the replayed frames never reached the cache"

    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)
    await recorder.start("harp_values")
    try:
        await recorder.record_at_frame(1, time.time())
    finally:
        await recorder.stop()
        await harp.shutdown()

    header, rows = _frame_rows(recorder.file_path)
    assert len(rows) == 1
    row = rows[0]
    assert _cell(header, row, f"{harp.id}:lick_count") == "3"
    # The value of the *last* event, not of the first and not a boolean "some
    # licking happened".
    assert _cell(header, row, f"{harp.id}:lick_state") == "1"
    # Device time of that last event, in ms.
    assert float(_cell(header, row, f"{harp.id}:lick_last_ms")) == pytest.approx(1000.0)


# --- 3. count clears between rows ---------------------------------------


async def test_count_clears_between_rows(tmp_path):
    """A count is events *since the previous row*, so it must not repeat.

    This is the property that makes the record honest at 30 fps: rows are
    written faster than licks last, so a count carried forward would report
    the same three events in every row for the rest of the session.
    """
    board = HarpBoard()
    frames = [event(LICK_STATE_ADDRESS, 1, 0.1), event(LICK_STATE_ADDRESS, 0, 0.2)]
    harp = _make_harp(board, _write_schema(tmp_path), "harp1", frames=frames)
    hw = _hardware_manager(board)
    _attach(hw, harp)
    await harp.initialize()
    assert harp.wait_for_replay(), "the replayed frames never reached the cache"

    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)
    await recorder.start("harp_clearing")
    try:
        t0 = time.time()
        await recorder.record_at_frame(1, t0)
        # No further events in between.
        await recorder.record_at_frame(2, t0 + 0.033)
    finally:
        await recorder.stop()
        await harp.shutdown()

    header, rows = _frame_rows(recorder.file_path)
    assert len(rows) == 2
    counts = [_cell(header, r, f"{harp.id}:lick_count") for r in rows]
    assert counts == ["2", "0"], counts
    # `state` and `last_ms` persist across the read; only `count` is consumed.
    states = [_cell(header, r, f"{harp.id}:lick_state") for r in rows]
    assert states == ["0", "0"], states


# --- 4. the metadata warning, and a CSV that still parses ----------------


async def test_non_event_register_warns_in_metadata_and_still_parses(tmp_path):
    """A profile may record a register the schema says never reports.

    That is not fatal -- a hand-written schema is often incomplete -- but the
    column would sit at its initial value all session, and nothing downstream
    can tell that apart from an animal that never licked. So the finding is
    written into the CSV itself, where it outlives the run's log.
    """
    board = HarpBoard()
    harp = _make_harp(
        board,
        _write_schema(tmp_path, lick_access="Read"),
        "harp1",
        frames=[event(LICK_STATE_ADDRESS, 1, 0.5)],
    )
    hw = _hardware_manager(board)
    _attach(hw, harp)
    await harp.initialize()
    assert harp.wait_for_replay()

    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)
    await recorder.start("harp_warning")
    try:
        await recorder.record_at_frame(1, time.time())
    finally:
        await recorder.stop()
        await harp.shutdown()

    warnings = [line for line in _comment_lines(recorder.file_path) if line.startswith("# WARNING")]
    assert len(warnings) == 1, warnings
    assert harp.id in warnings[0]
    assert "not an Event register" in warnings[0]

    # The warning row lives in the comment block, so the analysis reader has
    # to walk straight past it. A metadata row that broke the parse would
    # cost the recording rather than annotate it.
    metadata, df = parse_csv(recorder.file_path)
    assert metadata["Experiment Name"] == "harp_warning"
    assert f"{harp.id}:lick_count" in df.columns
    frame_rows = df[df["frame"].notna()]
    assert len(frame_rows) == 1
    assert int(frame_rows.iloc[0][f"{harp.id}:lick_count"]) == 1


# --- 5. a device with no profile does not break the recording ------------


async def test_device_without_a_profile_does_not_break_the_recording(tmp_path):
    """No profile means record nothing -- for that device, and only it.

    A Harp device with no profile is left in Standby with no reader running,
    so it has no columns and no state. It must still take its place in the
    header (as a single, empty column) without disturbing the device beside
    it, because "we have not written a profile for this board yet" is the
    ordinary state of a new rig.
    """
    board = HarpBoard()
    device_yml = _write_schema(tmp_path)
    silent = _make_harp(board, device_yml, "harp_silent", profile="")
    recording = _make_harp(
        board,
        device_yml,
        "harp_recording",
        frames=[event(LICK_STATE_ADDRESS, 1, 0.5), event(LICK_STATE_ADDRESS, 1, 0.6)],
    )
    hw = _hardware_manager(board)
    _attach(hw, silent)
    _attach(hw, recording)
    await silent.initialize()
    await recording.initialize()
    assert recording.wait_for_replay()

    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)
    await recorder.start("harp_no_profile")
    try:
        await recorder.record_at_frame(1, time.time())
    finally:
        await recorder.stop()
        await recording.shutdown()
        await silent.shutdown()

    header, rows = _frame_rows(recorder.file_path)
    assert len(rows) == 1
    row = rows[0]

    # The profile-less device falls back to the historical single column and
    # writes an empty cell -- not a crash, and not a missing column.
    assert f"{silent.id}:MockHarp" in header
    assert _cell(header, row, f"{silent.id}:MockHarp") == ""
    assert not any(h.startswith(f"{silent.id}:lick") for h in header), header

    # Its neighbour is unaffected.
    assert _cell(header, row, f"{recording.id}:lick_count") == "2"
    assert _cell(header, row, f"{recording.id}:lick_state") == "1"


# --- 6. mixed devices in one CSV ----------------------------------------


async def test_harp_and_an_ordinary_device_share_one_csv(tmp_path):
    """A three-column Harp device beside a one-column LED, in declaration order.

    The recorder builds a single header from every device it holds, and the
    two kinds are distinguished per row by what ``get_state()`` returned --
    a dict for the Harp, a scalar for the LED. Getting that wrong shifts every
    cell after the Harp's columns by two, which opens cleanly in a spreadsheet
    and is wrong everywhere.
    """
    mock_board = MockBoard()
    harp_board = HarpBoard()
    led = DigitalOutputDevice(
        board=mock_board, config=DeviceConfig(pins={"output": 13}), name="led"
    )
    harp = _make_harp(
        harp_board,
        _write_schema(tmp_path),
        "harp1",
        frames=[event(LICK_STATE_ADDRESS, 1, 0.5)],
    )
    hw = _hardware_manager(mock_board, harp_board)
    # Declaration order: LED first, Harp second.
    _attach(hw, led)
    _attach(hw, harp)
    await led.initialize()
    await harp.initialize()
    assert harp.wait_for_replay()

    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)
    await recorder.start("harp_mixed")
    try:
        await led.turn_on()
        await recorder.record_at_frame(1, time.time())
    finally:
        await recorder.stop()
        await harp.shutdown()

    header, rows = _frame_rows(recorder.file_path)
    assert header == [
        "frame",
        "timestamp",
        "elapsed_ms",
        "flow_elapsed_ms",
        f"{led.id}:{led.device_type}",
        *(f"{harp.id}:{name}" for name in LICK_COLUMNS),
    ]

    assert len(rows) == 1
    row = rows[0]
    assert _cell(header, row, f"{led.id}:{led.device_type}") == "1"
    assert _cell(header, row, f"{harp.id}:lick_count") == "1"
    assert _cell(header, row, f"{harp.id}:lick_state") == "1"
