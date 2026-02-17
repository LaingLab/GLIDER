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

    def test_writer_exception_does_not_crash_thread(self):
        """A write error on one frame doesn't prevent subsequent frames."""
        mock_writer = MagicMock()
        call_count = 0

        def side_effect(frame):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated disk error")

        mock_writer.write.side_effect = side_effect

        fwt = FrameWriterThread(mock_writer, max_queue_size=100)
        fwt.start()

        for _ in range(5):
            fwt.enqueue(_make_frame())

        fwt.stop(timeout=5.0)

        # Frame 3 failed, but frames 1,2,4,5 should have been written
        assert fwt.frames_written == 4
        assert mock_writer.write.call_count == 5

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
