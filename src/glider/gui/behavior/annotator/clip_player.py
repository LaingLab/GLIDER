"""Looping clip viewer for the active-learning UI.

Unlike a general video player, this widget plays exactly one short
clip on repeat. ``set_clip(path, start_frame, end_frame, fps)`` seeks
to ``start_frame``, plays forward to ``end_frame``, and snaps back to
``start_frame``. The user makes a decision; the UI then calls
``set_clip`` again with the next clip and the loop continues.

Implementation
--------------

* ``cv2.VideoCapture`` for frame-accurate decoding (same trade-off as
  the deleted BORIS player — QMediaPlayer can't reliably land on
  arbitrary frames over H.264).
* A ``QTimer`` ticks at ``1000 / fps`` ms during playback. On each
  tick we read one frame and ``setPixmap`` on the underlying QLabel.
* When the current frame reaches ``end_frame`` we seek back to
  ``start_frame``. We DON'T re-open the capture between loops — that
  would be slow — we just call ``cap.set(POS_FRAMES, start_frame)``.

The widget is a ``QLabel`` so the parent can drop it into a layout
and apply size policies without an extra wrapping container.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

if TYPE_CHECKING:
    from glider.gui.behavior.annotator.capture_cache import VideoCaptureCache


class ClipPlayer(QLabel):
    """Looping clip viewer. One clip at a time, plays on repeat."""

    #: The frame index just put on screen. Emitted on every displayed frame,
    #: including the seek back to the loop start. Anything drawn alongside the
    #: video (the speed trace's playhead) has to follow the frame the viewer
    #: is actually looking at, not the decoder's read-ahead position -- cv2
    #: advances POS_FRAMES past the frame it just handed back, so those two
    #: differ by one and the wrong choice puts the playhead permanently off.
    frame_changed = pyqtSignal(int)

    def __init__(
        self,
        parent=None,
        capture_cache: VideoCaptureCache | None = None,
    ):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #111; color: #888; border-radius: 8px;")
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setText("(no clip loaded)")

        self._capture_cache = capture_cache  # may be None for legacy single-clip use
        self._cap = None  # only used when capture_cache is None
        self._cap_path: Path | None = None
        self._start_frame: int = 0
        self._end_frame: int = 0
        self._current_frame: int = 0
        self._fps: float = 30.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_clip(
        self,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        fps: float = 30.0,
    ) -> bool:
        """Switch to a new clip and start playing it on loop.

        Returns True on success. Returns False (and shows a placeholder
        message) if the video can't be opened.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            self._stop_timer()
            self.setText("OpenCV not installed.\n\npip install opencv-python")
            return False

        path = Path(video_path)
        if not path.exists():
            self._stop_timer()
            self.setText(f"clip not found:\n{path.name}")
            return False

        # Resolve the capture: cache first (multi-video), fall back to the
        # single-instance private capture (legacy single-video flow).
        if self._capture_cache is not None:
            try:
                self._cap = self._capture_cache.get(path)
                self._cap_path = path
            except OSError:
                self._stop_timer()
                self.setText(f"couldn't open\n{path.name}")
                return False
        else:
            # Re-open the capture only when the source file changes; for
            # back-to-back clips from the same video, just seek.
            if self._cap is None or self._cap_path != path:
                self._open_capture(path)
                if self._cap is None:
                    return False

        self._start_frame = max(0, int(start_frame))
        self._end_frame = max(self._start_frame + 1, int(end_frame))
        self._fps = float(max(fps, 1e-3))
        self._seek_to_start()

        interval = max(int(round(1000.0 / self._fps)), 1)
        self._timer.start(interval)
        return True

    def set_loop_bounds(self, start_frame: int, end_frame: int) -> None:
        """Change the loop range ``[start_frame, end_frame)`` in place.

        Used by the trim editor: dragging a handle updates the loop so the
        preview always shows the current trim, without re-opening the
        capture. ``end_frame`` is forced above ``start_frame``. Seeks to
        the new start when a capture is open; otherwise just records the
        bounds (the next :meth:`set_clip` / tick will honour them).
        """
        self._start_frame = max(0, int(start_frame))
        self._end_frame = max(self._start_frame + 1, int(end_frame))
        if self._cap is not None:
            self._seek_to_start()

    def loop_bounds(self) -> tuple[int, int]:
        return (self._start_frame, self._end_frame)

    def stop(self) -> None:
        """Stop playback and clear the surface."""
        self._stop_timer()
        self.setText("(stopped)")

    def release(self) -> None:
        """Release the privately-owned capture, if any. No-op when the capture
        comes from an injected VideoCaptureCache — the cache owns lifecycle."""
        self._stop_timer()
        if self._capture_cache is None and self._cap is not None:
            self._cap.release()
        self._cap = None
        self._cap_path = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _open_capture(self, path: Path) -> None:
        import cv2

        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.setText(f"failed to open\n{path.name}")
            self._cap = None
            self._cap_path = None
            return
        self._cap = cap
        self._cap_path = path

    def _seek_to_start(self) -> None:
        if self._cap is None:
            return
        import cv2

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start_frame)
        self._current_frame = self._start_frame
        ok, frame_bgr = self._cap.read()
        if ok and frame_bgr is not None:
            self._display_bgr(frame_bgr)
            # cv2 advances POS_FRAMES on read; track that. The frame now on
            # screen is _start_frame, which is what listeners are told.
            self._current_frame = self._start_frame + 1
            self.frame_changed.emit(self._start_frame)

    def _on_tick(self) -> None:
        if self._cap is None:
            self._stop_timer()
            return
        if self._current_frame >= self._end_frame:
            # End of clip → loop back to start.
            self._seek_to_start()
            return
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            # Reached end of file unexpectedly; treat as end-of-clip.
            self._seek_to_start()
            return
        shown = self._current_frame
        self._current_frame += 1
        self._display_bgr(frame_bgr)
        self.frame_changed.emit(shown)

    def _stop_timer(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    def _display_bgr(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        qimg = QImage(
            frame_rgb.data,
            w,
            h,
            frame_rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pix)

    def __del__(self):  # pragma: no cover
        try:
            if self._capture_cache is None and self._cap is not None:
                self._cap.release()
        except Exception:
            pass
