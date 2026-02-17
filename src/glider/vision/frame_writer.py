"""
Frame Writer Thread - Dedicated thread for writing video frames to disk.

Decouples disk I/O from the camera capture thread by buffering frames
in a queue and writing them from a separate thread. This prevents
cv2.VideoWriter.write() blocking from causing frame drops in the
capture pipeline.
"""

import logging
import platform
import queue
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default buffer: 300 frames (~5s at 60fps, ~270MB at 640x480).
# On aarch64 (Raspberry Pi), reduce to 60 to limit memory usage.
_DEFAULT_MAX_QUEUE = 60 if platform.machine().startswith("aarch") else 300


class FrameWriterThread:
    """
    A dedicated thread that drains a frame queue and writes to a cv2.VideoWriter.

    Usage::

        writer = cv2.VideoWriter(...)
        fwt = FrameWriterThread(writer)
        fwt.start()
        for frame in frames:
            fwt.enqueue(frame)
        fwt.stop()   # drains remaining frames, then joins
        writer.release()
    """

    def __init__(
        self,
        writer: cv2.VideoWriter,
        max_queue_size: int = _DEFAULT_MAX_QUEUE,
    ):
        self._writer = writer
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_written = 0
        self._frames_dropped = 0
        self._lock = threading.Lock()  # guards counters

    # -- public API --

    def start(self) -> None:
        """Launch the writer thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="FrameWriter", daemon=True)
        self._thread.start()

    def enqueue(self, frame: np.ndarray) -> bool:
        """
        Add a frame to the write queue (non-blocking).

        Returns:
            True if the frame was queued, False if the queue was full (frame dropped).
        """
        try:
            self._queue.put_nowait(frame)
            return True
        except queue.Full:
            with self._lock:
                self._frames_dropped += 1
            return False

    def stop(self, timeout: float = 30.0) -> None:
        """
        Signal the writer to stop, drain remaining frames, then join.

        Args:
            timeout: Maximum seconds to wait for the thread to finish.
        """
        self._stop_event.set()
        if self._thread is not None:
            remaining = self._queue.qsize()
            if remaining > 0:
                logger.info(f"FrameWriterThread: draining {remaining} buffered frames...")
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("FrameWriterThread: thread did not finish within timeout")
        with self._lock:
            if self._frames_dropped > 0:
                logger.warning(
                    f"FrameWriterThread: {self._frames_dropped} frames dropped "
                    f"due to full buffer (max={self._max_queue_size})"
                )
            logger.info(f"FrameWriterThread: {self._frames_written} frames written")

    # -- properties --

    @property
    def frames_written(self) -> int:
        with self._lock:
            return self._frames_written

    @property
    def frames_dropped(self) -> int:
        with self._lock:
            return self._frames_dropped

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def max_queue_size(self) -> int:
        return self._max_queue_size

    # -- internal --

    def _run(self) -> None:
        """Writer loop: drain queue until stopped and queue is empty."""
        while True:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            try:
                self._writer.write(frame)
                with self._lock:
                    self._frames_written += 1
            except Exception:
                logger.exception("FrameWriterThread: error writing frame")
