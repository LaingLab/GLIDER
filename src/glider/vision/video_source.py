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


#: A jump further ahead than this seeks, rather than decoding every frame between.
_WALK_LIMIT = 150
#: How far before its target a seek aims, so one that lands a few frames late
#: (as real long-GOP captures do) still lands at or before the target.
_SEEK_MARGIN = 16


class ExactFrameReader:
    """Read frames by index, exactly, without trusting where a seek lands.

    ``cap.set(CAP_PROP_POS_FRAMES, n)`` is exact only on all-keyframe codecs.
    On a long-GOP mp4 it lands near ``n`` -- measured at -5 to +8 frames on a
    30 fps session -- and ``get(CAP_PROP_POS_FRAMES)`` afterwards still returns
    ``n``, so it cannot say where the decoder really is. Seeking to an earlier
    frame and grabbing forward inherits the same error, and seeking by
    ``CAP_PROP_POS_MSEC`` misses by exactly as much.

    What *can* be trusted is the frame itself: FFmpeg stamps every decoded
    frame with its own time (``CAP_PROP_POS_MSEC`` after a grab). So a jump
    seeks to just before its target, reads the landed frame's timestamp to
    learn exactly which frame it is, and walks the rest -- the way video
    players seek exactly. Walking forward costs nothing, and a short hop ahead
    just walks.

    The timestamps are checked against the frame asked for on every read. A
    file whose timestamps do not count its frames (variable frame rate, or a
    capture that reports none) is caught on the first mismatch, and from then
    on the reader counts from frame 0 -- the one seek every capture gets
    right -- as it always used to. A frame reached through a seek that failed
    the check is fetched again that way, so the answer is exact either way.

    Why it matters: pose is paired with frames by index. A two-frame error is
    invisible while the animal is still and throws the skeleton clean off it
    while the animal runs -- which is how it went unnoticed. And counting from
    0 on every backward step made scrubbing a long session crawl: one step
    back at frame 40,000 re-decoded 40,000 frames.
    """

    def __init__(self, cap):
        self._cap = cap
        # Index the next grab() will decode. -1 = unknown, so rewind first.
        self._next = -1
        # Index of the frame grab() last decoded, which retrieve() returns.
        self._held: int | None = None
        # Frame 0's timestamp, read on the first rewind.
        self._t0_ms: float | None = None
        # Whether the decoder was put where it is by a seek (not by counting).
        self._from_seek = False
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and not math.isnan(fps) and fps > 0 else 0.0
        # Trusted until one disagrees with the frame count.
        self._stamps_ok = self._fps > 0

    def read(self, n: int) -> np.ndarray | None:
        """Frame ``n`` exactly, or None past the end of the video."""
        if n < 0:
            raise ValueError(f"frame index must be >= 0, got {n}")
        if not (self._held == n and self._next == n + 1):
            far = self._next >= 0 and n - self._next > _WALK_LIMIT
            if self._next < 0 or n < self._next or (far and self._stamps_ok):
                self._position(n)
                if self._next < 0:
                    return None
            for _ in range(n - self._next + 1):
                if not self._cap.grab():
                    self._next, self._held = -1, None
                    return None
                self._held, self._next = self._next, self._next + 1
                if self._held == 0 and self._t0_ms is None:
                    self._t0_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        ok, frame = self._cap.retrieve()
        if not ok:
            self._next, self._held = -1, None
            return None
        if self._stamps_ok and self._t0_ms is not None and self._stamp_index() != n:
            # This file's timestamps do not name its frames. Stop trusting
            # them; a frame reached by a seek may be the wrong one, so fetch
            # it again by counting.
            self._stamps_ok = False
            if self._from_seek:
                self._next, self._held = -1, None
                return self.read(n)
        return frame

    def _stamp_index(self) -> int:
        """Which frame the decoder holds, by its own timestamp."""
        ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        return round((ms - self._t0_ms) * self._fps / 1000.0)

    def _position(self, n: int) -> None:
        """Put the decoder at or before ``n``, knowing exactly which frame it is at."""
        if self._stamps_ok and self._t0_ms is None:
            self._rewind()  # the walk that follows reads frame 0's timestamp
            if self._next == 0 and n <= _WALK_LIMIT:
                return
            if self._next == 0 and self._cap.grab():
                self._held, self._next = 0, 1
                self._t0_ms = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
        if self._stamps_ok and self._t0_ms is not None and self._seek(n):
            return
        self._rewind()

    def _seek(self, n: int) -> bool:
        """Land at or before ``n`` by a seek, and learn where from the timestamp."""
        target = n - _SEEK_MARGIN
        for _ in range(3):
            if target <= 0:
                return False
            if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, target) or not self._cap.grab():
                return False
            landed = self._stamp_index()
            if landed < target - _WALK_LIMIT:
                # Nowhere near where it was sent: these stamps are not frame times.
                self._stamps_ok = False
                return False
            if landed <= n:
                self._held, self._next, self._from_seek = landed, landed + 1, True
                return True
            target -= 4 * _SEEK_MARGIN  # landed past n: aim earlier
        return False

    def _rewind(self) -> None:
        """Back to frame 0, the one seek every capture gets right."""
        self._held, self._from_seek = None, False
        self._next = 0 if self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0) else -1

    def invalidate(self) -> None:
        """Forget the tracked position — call after anyone else moves the cap."""
        self._next, self._held, self._from_seek = -1, None, False
