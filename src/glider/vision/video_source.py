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


class VideoFileSource:
    """Open a video file and read frames by index or sequentially."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._path: Path | None = None
        self._frame_count = 0
        self._fps = _DEFAULT_FPS
        self._resolution = (0, 0)

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
        """Seek to frame ``n`` and return it (BGR), or None. For scrubbing;
        seeking is frame-approximate on some codecs, which is fine here."""
        if self._cap is None:
            return None
        n = max(0, min(int(n), self._frame_count - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = self._cap.read()
        return frame if ok else None

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, frame)`` sequentially from frame 0. Exact, no-drop —
        use this for the tracking pass, not read_frame in a loop."""
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        self._frame_count = 0
        self._fps = _DEFAULT_FPS
        self._resolution = (0, 0)
