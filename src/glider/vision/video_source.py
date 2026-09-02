"""
VideoFileSource — pull-based wrapper over cv2.VideoCapture for offline video.

Unlike CameraManager (a real-time, threaded, callback-driven live source),
this is synchronous and on-demand: seek to any frame for scrubbing, or
iterate sequentially with no dropped frames for a tracking pass. No threads,
no callbacks. Used by the Camera panel's video-tracking mode.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_FPS = 30.0


def video_resolution(path: Path | str) -> tuple[int, int] | None:
    """``(width, height)`` from a video's header, or None if unreadable.

    A header read, so it costs nothing next to decoding. None is meaningful
    and must not be papered over with a default: callers use this to size an
    arena, and a wrong size is worse than an absent one.
    """
    source = VideoFileSource()
    if not source.load(path):
        return None
    try:
        width, height = source.resolution
    finally:
        source.release()
    return (width, height) if width > 0 and height > 0 else None


class VideoFileSource:
    """Open a video file and read frames by index or sequentially."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._path: Path | None = None
        self._frame_count = 0
        self._fps = _DEFAULT_FPS
        self._resolution = (0, 0)
        # Exact index-based access; None until a video is loaded.
        self._reader: ExactFrameReader | None = None

    def load(self, path: Path | str) -> bool:
        """Open ``path``. Returns False (and stays unloaded) if it cannot be
        opened or reports zero frames."""
        self.release()
        path = Path(path)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            logger.warning("VideoFileSource: cannot open %s", path)
            return False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            logger.warning("VideoFileSource: %s reports zero frames", path)
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Guard against 0 / NaN fps from some containers.
        if not fps or math.isnan(fps) or fps <= 0:
            fps = _DEFAULT_FPS
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        self._path = path
        self._frame_count = frame_count
        self._fps = float(fps)
        self._resolution = (width, height)
        self._reader = ExactFrameReader(cap)
        return True

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def is_loaded(self) -> bool:
        return self._cap is not None

    def read_frame(self, n: int) -> np.ndarray | None:
        """Return frame ``n`` exactly (BGR), or None.

        Exact, not approximate. ``cap.set(CAP_PROP_POS_FRAMES, n)`` lands
        several frames off on long-GOP video and this is how pose gets paired
        with a frame, so an approximate answer puts the skeleton on the wrong
        image — invisible while the animal is still, obvious while it runs.
        :class:`ExactFrameReader` counts from frame 0 instead; walking forward
        costs nothing, and only a backwards jump re-decodes.
        """
        if self._cap is None or self._reader is None:
            return None
        n = max(0, min(int(n), self._frame_count - 1))
        return self._reader.read(n)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, frame)`` sequentially from frame 0. Exact, no-drop —
        use this for the tracking pass, not read_frame in a loop."""
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # This generator drives the decoder, so read_frame's tracked position
        # is no longer true; make it rewind rather than trust a stale count.
        if self._reader is not None:
            self._reader.invalidate()
        n = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield n, frame
            n += 1

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._path = None
        self._reader = None
        self._frame_count = 0
        self._fps = _DEFAULT_FPS
        self._resolution = (0, 0)


class ExactFrameReader:
    """Read frames by index, counting from 0 instead of trusting the seek.

    ``cap.set(CAP_PROP_POS_FRAMES, n)`` is exact only on all-keyframe codecs.
    On a long-GOP mp4 it lands near ``n`` — measured at -5 to +8 frames on a
    30 fps session — and ``get(CAP_PROP_POS_FRAMES)`` afterwards still returns
    ``n``, so the decoder cannot be asked where it really is. Seeking to an
    earlier frame and grabbing forward inherits the same error, and seeking by
    ``CAP_PROP_POS_MSEC`` misses by exactly as much.

    Only one seek is trustworthy: to frame 0. So this counts decoded frames
    from there, which is exact, and keeps the count as it walks forward. A
    request behind the current position rewinds and re-walks; a request ahead
    grabs on. ``grab`` skips the decode of frames nobody asked for, and a full
    pass over a 21,700-frame session runs in about six seconds, so the rewind
    is affordable even from the end of a video.

    Why it matters: pose is paired with frames by index. A two-frame error is
    invisible while the animal is still and throws the skeleton clean off it
    while the animal runs — which is how it went unnoticed.
    """

    def __init__(self, cap):
        self._cap = cap
        # Index the next grab() will decode. -1 = unknown, so rewind first.
        self._next = -1

    def read(self, n: int) -> np.ndarray | None:
        """Frame ``n`` exactly, or None past the end of the video."""
        if n < 0:
            raise ValueError(f"frame index must be >= 0, got {n}")
        if self._next < 0 or n < self._next:
            if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                self._next = -1
                return None
            self._next = 0
        for _ in range(n - self._next):
            if not self._cap.grab():
                self._next = -1
                return None
        if not self._cap.grab():
            self._next = -1
            return None
        ok, frame = self._cap.retrieve()
        if not ok:
            self._next = -1
            return None
        self._next = n + 1
        return frame

    def invalidate(self) -> None:
        """Forget the tracked position — call after anyone else moves the cap."""
        self._next = -1
