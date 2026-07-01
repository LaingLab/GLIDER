"""LRU cache of `cv2.VideoCapture` handles for the multi-video annotator.

The single-video flow is fine with one open capture forever. The multi-video
flow needs to switch between videos per clip; reopening every time is slow
and seeking can be flaky on H.264. This cache keeps a small number of
handles open and evicts the least-recently-used one when the cap is full.

Capped at 3 by default - small enough to stay well under OS file-handle
limits even if a labeler powers through 30 videos in one session; large
enough to absorb the typical "jump between 2-3 hot videos" working set.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any


def _open_capture(path: str | Path) -> Any:
    """Open a `cv2.VideoCapture` for `path`. Indirected so tests can monkeypatch."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cv2.VideoCapture failed to open {path}")
    return cap


class VideoCaptureCache:
    """LRU cache for cv2.VideoCapture handles keyed by absolute path string."""

    def __init__(self, max_open: int = 3):
        if max_open < 1:
            raise ValueError(f"max_open must be >= 1, got {max_open}")
        self._max_open = int(max_open)
        self._caps: OrderedDict[str, Any] = OrderedDict()

    def get(self, video_path: str | Path) -> Any:
        """Return a (possibly cached) `cv2.VideoCapture` for `video_path`.

        Marks the entry as most-recently-used. Evicts the LRU entry if the
        cache would exceed `max_open` after this insertion.
        """
        key = str(video_path)
        if key in self._caps:
            self._caps.move_to_end(key)
            return self._caps[key]
        cap = _open_capture(key)
        self._caps[key] = cap
        if len(self._caps) > self._max_open:
            _evicted_key, evicted_cap = self._caps.popitem(last=False)
            try:
                evicted_cap.release()
            except Exception:  # pragma: no cover - defensive
                pass
        return cap

    def close_all(self) -> None:
        """Release every cached capture and clear the cache."""
        while self._caps:
            _k, cap = self._caps.popitem(last=False)
            try:
                cap.release()
            except Exception:  # pragma: no cover - defensive
                pass
