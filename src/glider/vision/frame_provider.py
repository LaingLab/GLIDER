"""A single frame-source interface over the live camera and recorded video.

:class:`glider.gui.dialogs.calibration_dialog.CalibrationDialog` needs one
still frame to draw measurement lines on. It used to reach into CameraManager
directly, which made it unusable for a batch over recorded files. These
adapters expose the two members it actually needs, plus an optional seek
capability so a video source can be scrubbed for a frame worth measuring.

The Protocols are for typing only — deliberately not runtime_checkable.
Consumers test for seekability with ``hasattr(provider, "seek")``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from glider.vision.video_source import VideoFileSource

if TYPE_CHECKING:  # pragma: no cover - typing only
    from glider.vision.camera_manager import CameraManager

logger = logging.getLogger(__name__)

__all__ = [
    "CameraFrameProvider",
    "FrameProvider",
    "SeekableFrameProvider",
    "VideoFrameProvider",
]


class FrameProvider(Protocol):
    """Something that can hand over one still frame on demand."""

    @property
    def is_connected(self) -> bool: ...

    def get_frame(self) -> tuple[np.ndarray, float] | None:
        """``(BGR frame, timestamp)``, or None when no frame is available."""
        ...


class SeekableFrameProvider(FrameProvider, Protocol):
    """A frame source with a timeline, so callers can offer a scrubber."""

    @property
    def frame_count(self) -> int: ...

    @property
    def position(self) -> int: ...

    def seek(self, frame_index: int) -> None: ...


class CameraFrameProvider:
    """Adapts the live :class:`CameraManager`. Not seekable: no timeline."""

    def __init__(self, camera_manager: CameraManager) -> None:
        self._camera = camera_manager

    @property
    def is_connected(self) -> bool:
        return self._camera.is_connected

    def get_frame(self) -> tuple[np.ndarray, float] | None:
        return self._camera.get_frame()


class VideoFrameProvider:
    """Adapts a video file, exposing its timeline for scrubbing.

    Thin by design: VideoFileSource.read_frame already seeks. Holds an open
    capture, so call :meth:`release` when the dialog closes.
    """

    def __init__(self, video_path: Path | str) -> None:
        self._source = VideoFileSource()
        self._loaded = self._source.load(video_path)
        self._index = 0
        if not self._loaded:
            logger.warning("VideoFrameProvider: cannot open %s", video_path)

    @property
    def is_connected(self) -> bool:
        return self._loaded

    @property
    def frame_count(self) -> int:
        return self._source.frame_count

    @property
    def position(self) -> int:
        return self._index

    def seek(self, frame_index: int) -> None:
        last = max(0, self._source.frame_count - 1)
        self._index = max(0, min(int(frame_index), last))

    def get_frame(self) -> tuple[np.ndarray, float] | None:
        frame = self._source.read_frame(self._index)
        if frame is None:
            return None
        # Video timeline, matching VideoTrackingRunner's convention.
        return frame, self._index / self._source.fps

    def release(self) -> None:
        self._source.release()
        self._loaded = False
