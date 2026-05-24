"""TrackingDataLogger contract: frame_count must advance on every call,
including when no objects were tracked.

This is the post-fix invariant the CameraPanel relies on: the device-state
recorder reads ``tracking_logger.frame_count`` after each ``log_frame``
call to label its own row, so the counter must advance once per camera
frame regardless of CV output.
"""

from __future__ import annotations

import pytest

from glider.vision.tracking_logger import TrackingDataLogger


@pytest.mark.asyncio
async def test_frame_count_advances_with_empty_tracked_list(tmp_path):
    logger = TrackingDataLogger(output_dir=tmp_path)
    await logger.start("empty_tracked_test")
    try:
        import time as _time

        ts = _time.time()
        for i in range(5):
            logger.log_frame(ts + i * 0.033, [], motion_detected=False, motion_area=0.0)
        assert logger.frame_count == 5
    finally:
        await logger.stop()


@pytest.mark.asyncio
async def test_frame_count_advances_with_motion_only(tmp_path):
    logger = TrackingDataLogger(output_dir=tmp_path)
    await logger.start("motion_only_test")
    try:
        import time as _time

        ts = _time.time()
        logger.log_frame(ts, [], motion_detected=True, motion_area=0.05)
        logger.log_frame(ts + 0.033, [], motion_detected=True, motion_area=0.07)
        assert logger.frame_count == 2
    finally:
        await logger.stop()


@pytest.mark.asyncio
async def test_session_epoch_anchors_elapsed_ms(tmp_path):
    """When a session epoch is set, elapsed_ms is measured against it."""
    import time as _time

    logger = TrackingDataLogger(output_dir=tmp_path)
    epoch = _time.time() - 0.5
    logger.set_session_epoch(epoch)
    await logger.start("epoch_test")
    try:
        # Frame ts is +1.5s after the shared epoch.
        logger.log_frame(epoch + 1.5, [], motion_detected=False)
    finally:
        await logger.stop()

    # Read the file to verify the elapsed_ms column.
    import csv as _csv

    with open(logger.file_path, encoding="utf-8") as f:
        rows = list(_csv.reader(f))
    # The heartbeat path fires on the first call when no tracked objects.
    data_rows = [r for r in rows if r and r[0].isdigit() and not r[0].startswith("#")]
    # Either a heartbeat row was written, or the row is the motion row;
    # either way we expect elapsed_ms ~ 1500 against the shared epoch.
    assert data_rows, "expected at least one numeric-frame row from log_frame"
    elapsed = float(data_rows[0][2])
    assert 1490.0 <= elapsed <= 1510.0, f"got {elapsed} ms; expected ~1500"
