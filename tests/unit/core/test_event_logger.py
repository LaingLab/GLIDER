"""Tests for DeviceEventLogger.

Exercises both event streams end-to-end against the real MockBoard +
real Device classes + real HardwareManager. MockBoard fires both the
legacy ``_notify_callbacks`` (per-pin) and the new
``_notify_output_change`` (per-board) paths from its write methods, so
output writes produce one ``output_write`` row in the events CSV via
the board-level subscription.
"""

from __future__ import annotations

import csv

import pytest

from glider.core.event_logger import DeviceEventLogger
from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import (
    DeviceConfig,
    DigitalInputDevice,
    DigitalOutputDevice,
)
from glider.hal.mock_board import MockBoard


def _make_hw(output_pin: int = 13, input_pin: int = 7):
    hw = HardwareManager()
    board = MockBoard()
    hw._boards[board.id] = board

    led = DigitalOutputDevice(
        board=board, config=DeviceConfig(pins={"output": output_pin}), name="led"
    )
    beam = DigitalInputDevice(
        board=board, config=DeviceConfig(pins={"input": input_pin}), name="beam"
    )
    hw._devices[led.id] = led
    hw._devices[beam.id] = beam
    return hw, board, led, beam


def _read_event_rows(path) -> list[dict[str, str]]:
    """Return event rows as a list of dicts keyed by the CSV header."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header_idx = next(i for i, r in enumerate(rows) if r and r and r[0] == "frame")
    header = rows[header_idx]
    out: list[dict[str, str]] = []
    for r in rows[header_idx + 1 :]:
        if not r or r[0].startswith("#"):
            continue
        if len(r) != len(header):
            continue
        out.append(dict(zip(header, r, strict=False)))
    return out


@pytest.mark.asyncio
async def test_output_writes_are_captured(tmp_path):
    hw, board, led, _beam = _make_hw()

    logger = DeviceEventLogger(hw, output_dir=tmp_path)
    await logger.start("out_test")
    try:
        await led.initialize()  # initialize() also writes pin LOW
        await led.turn_on()  # write True
        await led.turn_off()  # write False
    finally:
        await logger.stop()

    rows = _read_event_rows(logger.file_path)
    output_rows = [r for r in rows if r["source"] == "output_write"]
    assert [r["value"] for r in output_rows[-2:]] == ["1", "0"]
    assert all(r["device_type"] == "DigitalOutput" for r in output_rows[-2:])
    assert all(r["device_id"] == led.id for r in output_rows[-2:])
    assert all(r["pin"] == "13" for r in output_rows[-2:])
    assert all(r["pin_type"] == "DIGITAL" for r in output_rows[-2:])
    assert all(r["board_id"] == board.id for r in output_rows[-2:])


@pytest.mark.asyncio
async def test_input_changes_are_captured(tmp_path):
    """Simulate an input-side state change via MockBoard's pin callbacks."""
    hw, board, _led, beam = _make_hw()

    logger = DeviceEventLogger(hw, output_dir=tmp_path)
    await logger.start("in_test")
    try:
        # Directly fire a pin callback as if the board reported an edge.
        # MockBoard's write_digital fires _notify_callbacks too, but here
        # we want to simulate an actual input edge on a pin owned by an
        # input device, so we use the board's internal hook directly.
        board._notify_callbacks(7, True)
        board._notify_callbacks(7, False)
    finally:
        await logger.stop()

    rows = _read_event_rows(logger.file_path)
    input_rows = [r for r in rows if r["source"] == "input_change"]
    assert [r["value"] for r in input_rows] == ["1", "0"]
    assert all(r["device_type"] == "DigitalInput" for r in input_rows)
    assert all(r["device_id"] == beam.id for r in input_rows)
    assert all(r["pin"] == "7" for r in input_rows)


@pytest.mark.asyncio
async def test_current_frame_stamps_subsequent_events(tmp_path):
    hw, _board, led, _beam = _make_hw()
    logger = DeviceEventLogger(hw, output_dir=tmp_path)
    await logger.start("frame_stamp_test")
    try:
        # No frame seen yet — first event should have empty frame cell.
        await led.initialize()
        await led.turn_on()

        # Tick the camera frame counter and write again.
        logger.set_current_frame(42, 1_700_000_000.0)
        await led.turn_off()

        logger.set_current_frame(43, 1_700_000_000.033)
        await led.turn_on()
    finally:
        await logger.stop()

    rows = _read_event_rows(logger.file_path)
    out_rows = [r for r in rows if r["source"] == "output_write"]
    # First write happened before set_current_frame: frame must be empty.
    assert out_rows[0]["frame"] == ""
    # After frame=42 is set, the next event row carries it.
    forty_two = [r for r in out_rows if r["frame"] == "42"]
    forty_three = [r for r in out_rows if r["frame"] == "43"]
    assert len(forty_two) >= 1
    assert len(forty_three) >= 1


@pytest.mark.asyncio
async def test_session_epoch_anchors_event_elapsed_ms(tmp_path):
    """
    When ``set_session_epoch`` is called before ``start()``, an event
    fires after the override epoch should report elapsed_ms measured
    against the shared epoch, not the logger's own start_time.

    Set the epoch ~5 seconds in the past; an event fired now should
    therefore report elapsed_ms ~ 5000, not ~0 (which is what we'd see
    if the logger used its own start_time).
    """
    import time as _time

    hw, _board, led, _beam = _make_hw()

    epoch = _time.time() - 5.0
    elog = DeviceEventLogger(hw, output_dir=tmp_path)
    elog.set_session_epoch(epoch)

    await elog.start("epoch_anchor_test")
    try:
        await led.initialize()
        await led.turn_on()
    finally:
        await elog.stop()

    rows = _read_event_rows(elog.file_path)
    out_rows = [r for r in rows if r["source"] == "output_write"]
    assert out_rows, "no output_write rows captured"
    elapsed = float(out_rows[-1]["elapsed_ms"])
    # ~5000 ms is the floor since the event fires after start(). It will be
    # slightly higher (test overhead). The point is that it's >> 0, which
    # would be the case if the override hadn't been respected.
    assert elapsed >= 4990.0, f"event logger ignored session epoch override; got {elapsed} ms"
    assert elapsed < 10000.0, f"unexpectedly large elapsed_ms ({elapsed} ms); test probably hung"


@pytest.mark.asyncio
async def test_session_epoch_shared_across_recorder_and_event_log(tmp_path):
    """
    Same physical wall-clock instant should produce equal elapsed_ms in
    both the device-state CSV and the events CSV when both recorders
    share the same epoch (modulo small per-call dispatch overhead).
    """
    import time as _time

    from glider.core.data_recorder import DataRecorder

    hw, _board, led, _beam = _make_hw()
    epoch = _time.time()

    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)
    recorder.set_session_epoch(epoch)

    elog = DeviceEventLogger(hw, output_dir=tmp_path)
    elog.set_session_epoch(epoch)

    await recorder.start("shared_epoch_test")
    await elog.start("shared_epoch_test")
    try:
        await led.initialize()
        now = _time.time()
        # Push the same wall-clock timestamp into the recorder's frame
        # path AND into the event logger's current-frame state. Then fire
        # an event a few microseconds later. Both rows should be anchored
        # to the same shared epoch.
        await recorder.record_at_frame(42, now)
        elog.set_current_frame(42, now)
        await led.turn_on()
    finally:
        await elog.stop()
        await recorder.stop()

    import csv as _csv

    with open(recorder.file_path, encoding="utf-8") as f:
        rec_rows = list(_csv.reader(f))
    rec_frame_rows = [r for r in rec_rows if r and r[0] == "42"]
    assert rec_frame_rows
    rec_elapsed = float(rec_frame_rows[0][2])

    ev_rows = [
        r
        for r in _read_event_rows(elog.file_path)
        if r["frame"] == "42" and r["source"] == "output_write" and r["value"] == "1"
    ]
    assert ev_rows
    ev_elapsed = float(ev_rows[0]["elapsed_ms"])

    # The two are anchored to the same epoch and measured a few ms apart,
    # so the difference should be small. If the event logger were using
    # its own start_time instead of the shared epoch the difference would
    # be larger because the two recorders started ~ms apart.
    assert (
        abs(ev_elapsed - rec_elapsed) < 50.0
    ), f"shared epoch not respected: rec={rec_elapsed}, ev={ev_elapsed}"


@pytest.mark.asyncio
async def test_unsubscribes_cleanly_on_stop(tmp_path):
    """After stop(), further writes must not append rows to a closed file."""
    hw, board, led, _beam = _make_hw()
    logger = DeviceEventLogger(hw, output_dir=tmp_path)
    await logger.start("unsubscribe_test")
    await led.initialize()
    await led.turn_on()
    path = logger.file_path
    await logger.stop()

    # Capture file size after stop, then write a few more times via the LED.
    size_before = path.stat().st_size
    await led.turn_off()
    await led.turn_on()
    size_after = path.stat().st_size

    assert (
        size_after == size_before
    ), "DeviceEventLogger appended rows to the events CSV after stop()"
