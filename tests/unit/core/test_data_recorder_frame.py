"""Tests for the frame-aligned DataRecorder mode.

The recorder now supports two modes:

* timer-driven (default) — periodic ``asyncio.sleep`` loop, frame
  column empty
* camera-driven — one row per call to ``record_at_frame``, frame
  column populated, no timer task created

These tests exercise both paths against the real MockBoard +
DigitalOutputDevice + HardwareManager.
"""

from __future__ import annotations

import asyncio
import csv

import pytest

from glider.core.data_recorder import DataRecorder
from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import DeviceConfig, DigitalOutputDevice
from glider.hal.mock_board import MockBoard


def _make_hw_with_led(pin: int = 13) -> tuple[HardwareManager, DigitalOutputDevice]:
    hw = HardwareManager()
    board = MockBoard()
    hw._boards[board.id] = board
    led = DigitalOutputDevice(board=board, config=DeviceConfig(pins={"output": pin}), name="led")
    hw._devices[led.id] = led
    return hw, led


def _read_csv_rows(path) -> tuple[list[str], list[list[str]]]:
    """Return (header_row, data_rows) skipping metadata comment rows."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # Header is the first non-blank, non-`#`-prefixed row.
    header_idx = next(
        i for i, r in enumerate(rows) if r and not r[0].startswith("#") and r[0]
    )
    data_rows = [
        r for r in rows[header_idx + 1 :] if r and not r[0].startswith("#") and r != []
    ]
    return rows[header_idx], data_rows


@pytest.mark.asyncio
async def test_header_contains_frame_column_at_position_zero(tmp_path):
    hw, _ = _make_hw_with_led()
    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)  # long
    recorder.set_camera_driven(True)  # disable timer entirely

    await recorder.start("frame_header_test")
    try:
        header, _ = _read_csv_rows(recorder.file_path)
        assert header[0] == "frame", f"first column must be 'frame', got {header!r}"
        assert header[1] == "timestamp"
        assert header[2] == "elapsed_ms"
    finally:
        await recorder.stop()


@pytest.mark.asyncio
async def test_record_at_frame_writes_row_with_frame_index(tmp_path):
    hw, led = _make_hw_with_led()
    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)

    await recorder.start("record_at_frame_test")
    try:
        # Simulate three camera frames at known timestamps. The recorder
        # uses time.time() inside, so we just need monotonic floats.
        import time as _time

        t0 = _time.time()
        await recorder.record_at_frame(1, t0 + 0.000)
        await led.initialize()
        await led.turn_on()
        await recorder.record_at_frame(2, t0 + 0.033)  # ~30 fps
        await led.turn_off()
        await recorder.record_at_frame(3, t0 + 0.066)
    finally:
        await recorder.stop()

    _, data_rows = _read_csv_rows(recorder.file_path)

    # `stop()` writes a final timer-style row with frame="", so we filter to
    # only the rows that have a numeric frame.
    frame_rows = [r for r in data_rows if r and r[0].isdigit()]
    assert [int(r[0]) for r in frame_rows] == [1, 2, 3]

    # elapsed_ms should be monotonic and roughly the deltas we passed in.
    elapsed = [float(r[2]) for r in frame_rows]
    assert elapsed[0] < elapsed[1] < elapsed[2]
    assert 25 <= (elapsed[2] - elapsed[0]) <= 100, elapsed


@pytest.mark.asyncio
async def test_camera_driven_does_not_create_sample_task(tmp_path):
    """In camera-driven mode the periodic sampling task must not exist."""
    hw, _ = _make_hw_with_led()
    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=0.05)
    recorder.set_camera_driven(True)

    await recorder.start("no_task_test")
    try:
        assert recorder._sample_task is None
        # Sleep longer than several sample intervals; no rows should appear.
        await asyncio.sleep(0.2)
        _, data_rows = _read_csv_rows(recorder.file_path)
        numeric_frames = [r for r in data_rows if r and r[0].isdigit()]
        assert numeric_frames == [], (
            "camera-driven mode wrote rows without record_at_frame calls"
        )
    finally:
        await recorder.stop()


@pytest.mark.asyncio
async def test_session_epoch_overrides_start_time(tmp_path):
    """When a session epoch is set, elapsed_ms is computed against it."""
    import time as _time

    hw, _ = _make_hw_with_led()
    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=10.0)
    recorder.set_camera_driven(True)

    # Set the epoch 1.0 seconds before this recorder's start() runs.
    pre_start_epoch = _time.time() - 1.0
    recorder.set_session_epoch(pre_start_epoch)

    await recorder.start("epoch_test")
    try:
        # Use a frame_ts ~exactly at the epoch + 2.0s.
        await recorder.record_at_frame(1, pre_start_epoch + 2.0)
    finally:
        await recorder.stop()

    _, rows = _read_csv_rows(recorder.file_path)
    frame_rows = [r for r in rows if r and r[0].isdigit()]
    assert len(frame_rows) == 1
    elapsed = float(frame_rows[0][2])
    # 2.0 seconds since the shared epoch.
    assert 1990.0 <= elapsed <= 2010.0, (
        f"expected ~2000 ms against shared epoch, got {elapsed}"
    )


@pytest.mark.asyncio
async def test_timer_driven_default_still_writes_rows(tmp_path):
    """Backward-compat: with no camera attached, the timer loop still works."""
    hw, _ = _make_hw_with_led()
    recorder = DataRecorder(hw, output_dir=tmp_path, sample_interval=0.02)
    # Note: set_camera_driven NOT called -> default timer-driven.

    await recorder.start("timer_test")
    try:
        await asyncio.sleep(0.12)  # enough for ~6 samples
        # Force a sample to make the test deterministic against scheduler jitter.
    finally:
        await recorder.stop()

    _, data_rows = _read_csv_rows(recorder.file_path)
    timer_rows = [r for r in data_rows if r and r[0] == ""]
    assert len(timer_rows) >= 3, f"expected at least 3 timer rows, got {len(timer_rows)}"
