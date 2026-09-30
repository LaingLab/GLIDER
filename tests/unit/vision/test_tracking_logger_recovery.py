"""A TrackingDataLogger that hit its write-error cap must recover next session."""

from __future__ import annotations

import pytest

from glider.vision.tracking_logger import TrackingDataLogger


class _FailingFlush:
    """Wraps the real file so flush() raises like a full disk."""

    def __init__(self, f):
        self._f = f

    def flush(self):
        raise OSError(28, "No space left on device")

    def __getattr__(self, name):
        return getattr(self._f, name)


@pytest.mark.asyncio
async def test_flush_failures_reach_the_cap_then_next_session_is_healthy(tmp_path):
    log = TrackingDataLogger(output_dir=tmp_path)
    await log.start("first")
    real_file = log._file
    log._file = _FailingFlush(real_file)
    for i in range(TrackingDataLogger._MAX_CONSECUTIVE_WRITE_ERRORS):
        log.log_frame(1000.0 + i, [], False, 0.0)
    assert log.is_failed and not log.is_recording

    # stop() used to return early here and leak the handle.
    log._file = real_file
    await log.stop()
    assert log._file is None and real_file.closed

    await log.start("second")
    try:
        assert log.is_recording and not log.is_failed
        log.log_frame(2000.0, [], False, 0.0)
        assert log.frame_count == 1
    finally:
        await log.stop()
