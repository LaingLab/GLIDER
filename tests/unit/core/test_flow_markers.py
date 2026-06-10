"""
Regression test: flow boundaries must be locatable in the recorded output
files so post-hoc analysis (ethogram / raster plot / video sync) can
trim, align, and synchronize cleanly without ad-hoc detective work.

Two contracts under test:

  1. ``DeviceEventLogger.record_flow_marker(marker)`` writes a row to the
     event log with ``source="flow_marker"`` and the marker name (``"start"``
     / ``"end"``) in the ``value`` cell. Analysts can grep this to find
     the exact wall-clock time the flow began and ended.

  2. The tracking logger and data recorder both write a ``flow_elapsed_ms``
     column populated relative to a flow-start wall-clock anchor set via
     ``set_flow_anchor(timestamp)``. Rows before the anchor get an empty
     cell (frames captured during pre-flow setup); rows after get
     ``(row_timestamp - anchor) * 1000``. This gives ethogram/raster
     scripts a column they can plot directly without computing offsets.

These are unit tests against the loggers in isolation; a separate
integration test in ``test_flow_duration.py`` verifies the wiring from
``GliderCore`` calls both at the right moments.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from glider.core.data_recorder import DataRecorder
from glider.core.event_logger import DeviceEventLogger
from glider.core.hardware_manager import HardwareManager
from glider.vision.tracking_logger import TrackingDataLogger


@pytest.mark.asyncio
async def test_event_logger_records_flow_markers(tmp_path: Path):
    """Flow markers are written to the event log as ``source=flow_marker``
    rows; analysts can grep these to find flow boundaries.
    """
    hm = HardwareManager()
    logger = DeviceEventLogger(hm)
    logger.set_output_directory(tmp_path)

    epoch = time.time()
    logger.set_session_epoch(epoch)

    path = await logger.start("flow_marker_test")
    try:
        # Simulate flow start a bit later, then end.
        logger.record_flow_marker("start")
        logger.record_flow_marker("end")
    finally:
        await logger.stop()

    content = path.read_text()
    assert "flow_marker" in content, (
        "record_flow_marker did not write a row with source=flow_marker. "
        f"Event log content:\n{content}"
    )
    # Match the *source* column specifically — commas on both sides — so we
    # don't accidentally catch a metadata line like "# Experiment,flow_marker_test".
    lines = content.splitlines()
    marker_rows = [line for line in lines if ",flow_marker," in line]
    assert (
        len(marker_rows) == 2
    ), f"Expected 2 flow_marker rows, got {len(marker_rows)}: {marker_rows}"
    # The value cell is the last column; markers are ",start" / ",end" at end-of-line.
    assert any(
        row.endswith(",start") for row in marker_rows
    ), f"No 'start' marker found in rows: {marker_rows}"
    assert any(
        row.endswith(",end") for row in marker_rows
    ), f"No 'end' marker found in rows: {marker_rows}"


@pytest.mark.asyncio
async def test_tracking_logger_flow_elapsed_ms_column(tmp_path: Path):
    """Tracking CSV has a ``flow_elapsed_ms`` column.

    Empty before ``set_flow_anchor`` is called (pre-flow frames), then
    populated as ``(timestamp - anchor) * 1000`` for each subsequent
    frame logged.
    """
    logger = TrackingDataLogger(output_dir=tmp_path)

    epoch = time.time()
    logger.set_session_epoch(epoch)
    path = await logger.start("flow_elapsed_test")
    try:
        # Log a "pre-flow" frame (no anchor yet)
        logger.log_frame(timestamp=epoch + 0.1, tracked_objects=[])

        # Anchor at epoch + 0.5 — simulates the flow starting at that wall-clock
        anchor = epoch + 0.5
        logger.set_flow_anchor(anchor)

        # Log frames after anchor — should get positive flow_elapsed_ms
        logger.log_frame(timestamp=epoch + 0.5, tracked_objects=[])  # flow_elapsed=0
        logger.log_frame(timestamp=epoch + 1.5, tracked_objects=[])  # flow_elapsed=1000
    finally:
        await logger.stop()

    content = path.read_text()
    header_row = next(line for line in content.splitlines() if line.startswith("frame,"))
    assert (
        "flow_elapsed_ms" in header_row
    ), f"flow_elapsed_ms column missing from tracking CSV header: {header_row}"


@pytest.mark.asyncio
async def test_data_recorder_flow_elapsed_ms_column(tmp_path: Path):
    """Data CSV has a ``flow_elapsed_ms`` column with the same semantics."""
    hm = HardwareManager()
    recorder = DataRecorder(hm)
    recorder.set_output_directory(tmp_path)

    epoch = time.time()
    recorder.set_session_epoch(epoch)
    path = await recorder.start("flow_elapsed_data_test")
    try:
        anchor = epoch + 0.5
        recorder.set_flow_anchor(anchor)
        # Force a sample so we get at least one data row (no devices configured,
        # so the row will be header-shape with empty device cells).
        await recorder._record_sample()
    finally:
        await recorder.stop()

    content = path.read_text()
    header_row = next(line for line in content.splitlines() if line.startswith("frame,"))
    assert (
        "flow_elapsed_ms" in header_row
    ), f"flow_elapsed_ms column missing from data CSV header: {header_row}"
