"""
Tests for glider.vision.frame_writer module.

Tests the FrameWriterThread that decouples disk I/O from the capture thread.
"""

import time
from unittest.mock import MagicMock

import numpy as np

from glider.vision.frame_writer import FrameWriterThread


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestFrameWriterThread:
    """Tests for FrameWriterThread."""

    def test_basic_write_through(self):
        """All enqueued frames are written after graceful stop."""
        mock_writer = MagicMock()
        fwt = FrameWriterThread(mock_writer, max_queue_size=100)
        fwt.start()

        frames = [_make_frame() for _ in range(10)]
        for frame in frames:
            assert fwt.enqueue(frame) is True

        fwt.stop(timeout=5.0)

        assert fwt.frames_written == 10
        assert fwt.frames_dropped == 0
        assert mock_writer.write.call_count == 10

    def test_queue_overflow_drops(self):
        """Frames are dropped and counted when queue is full."""
        # Use a slow writer to fill the queue
        mock_writer = MagicMock()
        mock_writer.write.side_effect = lambda f: time.sleep(0.05)

        fwt = FrameWriterThread(mock_writer, max_queue_size=3)
        fwt.start()

        # Give the writer thread time to start blocking on the first write
        time.sleep(0.01)

        # Enqueue more frames than the buffer can hold
        results = []
        for _ in range(10):
            results.append(fwt.enqueue(_make_frame()))

        fwt.stop(timeout=10.0)

        # Some frames should have been dropped
        assert fwt.frames_dropped > 0
        assert False in results
        # Written + dropped should equal total attempted
        assert fwt.frames_written + fwt.frames_dropped == 10

    def test_drain_on_stop(self):
        """All buffered frames are flushed on stop."""
        mock_writer = MagicMock()

        # Use a large buffer so nothing gets dropped
        fwt = FrameWriterThread(mock_writer, max_queue_size=100)

        # Enqueue frames BEFORE starting (they queue up)
        for _ in range(20):
            fwt.enqueue(_make_frame())

        assert fwt.queue_depth == 20

        fwt.start()
        fwt.stop(timeout=5.0)

        assert fwt.frames_written == 20
        assert fwt.frames_dropped == 0
        assert fwt.queue_depth == 0

    def test_high_fps_stress(self):
        """Simulate 600 frames at 60fps with 5ms write latency, expect zero drops."""
        mock_writer = MagicMock()
        mock_writer.write.side_effect = lambda f: time.sleep(0.005)

        fwt = FrameWriterThread(mock_writer, max_queue_size=300)
        fwt.start()

        frame = _make_frame()
        for _ in range(600):
            assert fwt.enqueue(frame.copy()) is True
            time.sleep(1.0 / 60)  # ~16.7ms between frames

        fwt.stop(timeout=30.0)

        assert fwt.frames_written == 600
        assert fwt.frames_dropped == 0

    def test_properties_initial_state(self):
        """Properties return correct values before start."""
        mock_writer = MagicMock()
        fwt = FrameWriterThread(mock_writer, max_queue_size=50)

        assert fwt.frames_written == 0
        assert fwt.frames_dropped == 0
        assert fwt.queue_depth == 0
        assert fwt.max_queue_size == 50

    def test_writer_exception_aborts_writer(self):
        """A write error marks the writer as failed and stops further writes.

        Once cv2.VideoWriter fails (disk full, codec fault, closed handle) it
        will not recover — draining the rest of the queue just masks the
        failure. The thread aborts on the first exception, marks itself
        failed, counts remaining frames as dropped, and invokes the error
        callback exactly once.
        """
        mock_writer = MagicMock()
        call_count = 0

        def side_effect(frame):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated disk error")

        mock_writer.write.side_effect = side_effect

        errors: list[BaseException] = []

        def on_error(exc: BaseException) -> None:
            errors.append(exc)

        fwt = FrameWriterThread(mock_writer, max_queue_size=100, error_callback=on_error)
        fwt.start()

        # Enqueue enough frames that the third write raises, leaving frames
        # 4 and 5 either dropped from the queue or rejected at enqueue time.
        for _ in range(5):
            fwt.enqueue(_make_frame())
            time.sleep(0.005)  # small gap so frames are processed in order

        fwt.stop(timeout=5.0)

        # Frames 1 and 2 made it to disk; frame 3 failed and aborted the loop.
        assert fwt.frames_written == 2
        assert fwt.failed is True
        assert isinstance(fwt.error, RuntimeError)
        # Callback fires exactly once with the original exception.
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        # Any frame enqueued after the abort is counted as dropped; the writer
        # never gets called more than the 3 attempts it made before aborting.
        assert mock_writer.write.call_count <= 3

    def test_enqueue_after_failure_returns_false(self):
        """Enqueue after the writer has aborted should fast-fail."""
        mock_writer = MagicMock()
        mock_writer.write.side_effect = RuntimeError("fail")

        fwt = FrameWriterThread(mock_writer, max_queue_size=10)
        fwt.start()
        fwt.enqueue(_make_frame())
        # Give the writer thread a beat to hit the exception.
        time.sleep(0.2)
        assert fwt.failed is True
        # Any subsequent enqueue should be rejected (counted as dropped).
        assert fwt.enqueue(_make_frame()) is False
        assert fwt.frames_dropped >= 1
        fwt.stop(timeout=2.0)

    def test_stop_without_start(self):
        """Calling stop without start should not raise."""
        mock_writer = MagicMock()
        fwt = FrameWriterThread(mock_writer, max_queue_size=10)
        fwt.stop(timeout=1.0)  # Should not raise

    def test_start_idempotent(self):
        """Calling start twice should not create a second thread."""
        mock_writer = MagicMock()
        fwt = FrameWriterThread(mock_writer, max_queue_size=10)
        fwt.start()
        thread1 = fwt._thread
        fwt.start()
        thread2 = fwt._thread

        assert thread1 is thread2
        fwt.stop(timeout=2.0)


class TestVideoRecorderIntegration:
    """Integration-level tests for VideoRecorder using FrameWriterThread."""

    def test_frames_written_after_stop(self):
        """Frames sent via _on_frame are all written after stop."""
        mock_writer = MagicMock()
        mock_writer.isOpened.return_value = True

        fwt = FrameWriterThread(mock_writer, max_queue_size=100)
        fwt.start()

        # Simulate 10 frames from _on_frame
        for _ in range(10):
            frame = _make_frame()
            fwt.enqueue(frame.copy())

        fwt.stop(timeout=5.0)

        assert fwt.frames_written == 10
        assert mock_writer.write.call_count == 10
