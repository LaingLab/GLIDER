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
from collections.abc import Callable

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
        error_callback: Callable[[BaseException], None] | None = None,
    ):
        self._writer = writer
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_size)
        self._max_queue_size = max_queue_size
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._frames_written = 0
        self._frames_dropped = 0
        self._failed = False
        self._error: BaseException | None = None
        self._error_callback = error_callback
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
            True if the frame was queued, False if the queue was full (frame dropped)
            or the writer has already failed.
        """
        if self._failed:
            with self._lock:
                self._frames_dropped += 1
            return False
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

    @property
    def failed(self) -> bool:
        """True if the writer aborted due to an unrecoverable write error."""
        return self._failed

    @property
    def error(self) -> BaseException | None:
        """The exception that caused the writer to abort, if any."""
        return self._error

    # -- internal --

    def _run(self) -> None:
        """Writer loop: drain queue until stopped and queue is empty.

        On a write exception the loop aborts: once cv2.VideoWriter fails
        (disk full, codec fault, closed file handle) it will not recover,
        so draining the remaining buffered frames just wastes CPU and masks
        the failure. We mark the writer as failed, count the queued frames
        as dropped, and invoke the error callback if one was supplied.
        """
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
            except Exception as exc:
                logger.exception("FrameWriterThread: error writing frame; aborting writer")
                self._failed = True
                self._error = exc
                # Count the frame we couldn't write plus any still in the queue as dropped.
                with self._lock:
                    remaining = self._queue.qsize()
                    self._frames_dropped += 1 + remaining
                # Best-effort drain so stop() doesn't wait on a stale queue.
                while True:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
                if self._error_callback is not None:
                    try:
                        self._error_callback(exc)
                    except Exception:
                        logger.exception("FrameWriterThread: error callback raised; swallowing")
                break
