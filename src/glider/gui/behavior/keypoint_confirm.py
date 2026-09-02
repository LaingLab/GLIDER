"""Confirm keypoint labels against a real frame before an Apply run starts.

The keypoint-names field is positional: each name is attached to the pose
model's Nth output keypoint, and the behavior model then looks its features up
by those names. Get the order wrong and the features it needs are simply never
produced — they arrive as NaN, every prediction is blank, and *nothing raises*.
A wrong order costs a full inference pass per video before the emptiness shows.

So show the operator what the names actually landed on, on a frame from their
own video, and make them say yes. A picture settles in a second what a text
field cannot express at all.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from glider.gui.styles import colors
from glider.vision.video_source import ExactFrameReader

logger = logging.getLogger(__name__)

# Okabe-Ito, so adjacent keypoints stay distinguishable for colourblind users.
_PALETTE = [
    (186, 114, 0),
    (0, 158, 230),
    (115, 158, 0),
    (0, 114, 213),
    (167, 121, 204),
    (233, 180, 86),
    (66, 228, 240),
]
_UPSCALE_TO = 720  # small arena videos are unreadable at native size


def annotate_keypoints(frame: np.ndarray, points: np.ndarray, names: list[str]) -> np.ndarray:
    """Return a copy of *frame* with each keypoint dotted and named.

    Pure and Qt-free so the labelling can be tested without a display. Points
    are ``(K, 2)``; NaN rows are skipped (an undetected keypoint), and any
    surplus name is ignored rather than raising — the caller has already
    checked the counts and a drawing helper is the wrong place to fail a run.
    """
    out = frame.copy()
    scale = max(1.0, _UPSCALE_TO / max(out.shape[0], out.shape[1]))
    if scale > 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    for i, (x, y) in enumerate(np.asarray(points, dtype=float)):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        px, py = int(round(x * scale)), int(round(y * scale))
        color = _PALETTE[i % len(_PALETTE)]
        label = f"{i}:{names[i]}" if i < len(names) else str(i)
        cv2.circle(out, (px, py), 6, color, -1)
        cv2.circle(out, (px, py), 6, (255, 255, 255), 1)
        # Outline then fill, so the text stays legible over fur or bedding.
        for thickness, ink in ((3, (0, 0, 0)), (1, color)):
            cv2.putText(
                out,
                label,
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ink,
                thickness,
                cv2.LINE_AA,
            )
    return out


def first_detected_frame(video, yolo_path, *, max_frames: int = 90, stride: int = 10):
    """``(frame, keypoints)`` for the first frame with a detection, or None.

    Samples forward rather than taking frame 0, because the animal is often
    out of view or being placed at the very start of a recording.
    """
    from ultralytics import YOLO

    model = YOLO(str(yolo_path))
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        reader = ExactFrameReader(cap)
        for n in range(0, max_frames * stride, stride):
            frame = reader.read(n)
            if frame is None:
                break
            result = model.predict(frame, verbose=False)[0]
            kp = result.keypoints
            if kp is None or kp.xy.shape[0] == 0:
                continue
            return frame, kp.xy[0].cpu().numpy()
    finally:
        cap.release()
    return None


def first_tracked_frame(video, pose_csv):
    """``(frame, keypoints)`` from an already-tracked pose CSV, or None.

    The same picture as :func:`first_detected_frame` without loading torch,
    and it is what the run will actually score: these are the coordinates the
    ethogram is built from, rather than a second opinion from a fresh pose
    pass. It is also the only source available when a run supplies poses and
    no weights.

    Prefers the first frame where every keypoint was found — a partial
    detection labels only some of the names, which is the least useful frame
    to check an ordering against — and settles for the fullest frame if no
    frame is complete.
    """
    from pathlib import Path

    from glider.vision.pose.dlc import from_dlc_csv

    xy = np.asarray(from_dlc_csv(Path(pose_csv)).xy, dtype=float)  # (frames, K, 2)
    if xy.ndim != 3 or xy.shape[0] == 0:
        return None
    found = np.isfinite(xy).all(axis=2)  # (frames, K)
    counts = found.sum(axis=1)
    if not counts.any():
        return None
    complete = np.flatnonzero(counts == xy.shape[1])
    index = int(complete[0]) if complete.size else int(np.argmax(counts))

    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None
        # Exact: the frame is handed back paired with xy[index], so an
        # approximate seek silently shows the keypoints of a different frame.
        frame = ExactFrameReader(cap).read(index)
    finally:
        cap.release()
    return (frame, xy[index]) if frame is not None else None


class _PreviewWorker(QObject):
    """Finds the preview frame off the UI thread; loading torch takes seconds."""

    done = pyqtSignal(object)  # (frame, keypoints) | None
    failed = pyqtSignal(str)

    def __init__(self, video, yolo_path, pose_csv=None):
        super().__init__()
        self._video, self._yolo = video, yolo_path
        self._pose_csv = pose_csv

    def run(self) -> None:
        try:
            if self._pose_csv is not None:
                self.done.emit(first_tracked_frame(self._video, self._pose_csv))
            else:
                self.done.emit(first_detected_frame(self._video, self._yolo))
        except Exception as e:  # never let it kill the thread
            logger.warning("keypoint preview failed", exc_info=True)
            self.failed.emit(str(e))


class KeypointConfirmDialog(QDialog):
    """Shows a labelled frame and asks the operator to confirm the mapping.

    ``accept()`` means the operator looked and agreed. Rejecting cancels the
    run — which is the point: a wrong mapping wastes an inference pass per
    video and produces an empty ethogram with no error to explain it.
    """

    def __init__(
        self,
        video,
        yolo_path,
        names,
        parent=None,
        *,
        warning: str | None = None,
        pose_csv=None,
    ):
        """``pose_csv`` labels a frame from poses already on disk.

        Preferred when the run has them: it shows the very coordinates that
        will be scored, skips a torch load, and works for a run that supplied
        poses and no weights — which has no pose model to ask.
        """
        super().__init__(parent)
        self.setWindowTitle("Confirm keypoint labels")
        self._names = list(names)
        self._thread: QThread | None = None
        self._worker: _PreviewWorker | None = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Each name below is attached to the pose model's keypoint at that "
            "position. Check every label sits on the body part it names — a "
            "wrong order produces an empty ethogram with no error."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        if warning:
            self._warning = QLabel(f"⚠ {warning}")
            self._warning.setWordWrap(True)
            self._warning.setStyleSheet(f"color: {colors.ERROR}; font-weight: bold;")
            layout.addWidget(self._warning)

        self._image = QLabel("Loading a frame and running the pose model…")
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(480, 360)
        layout.addWidget(self._image, 1)

        self._legend = QLabel(", ".join(f"{i}:{n}" for i, n in enumerate(self._names)))
        self._legend.setWordWrap(True)
        layout.addWidget(self._legend)

        buttons = QHBoxLayout()
        self._ok = QPushButton("Labels are correct — run")
        self._ok.setEnabled(False)  # nothing to confirm until a frame is shown
        self._ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(self._ok)
        layout.addLayout(buttons)

        self._video, self._yolo_path = video, yolo_path
        self._pose_csv = pose_csv
        self._started = False

    # ------------------------------------------------------------------

    def showEvent(self, event):
        """Start the preview when the dialog is actually shown.

        Deliberately not in ``__init__``: constructing a dialog should not
        spawn a thread, so this stays testable without an event loop.
        """
        super().showEvent(event)
        if not self._started:
            self._started = True
            self._start(self._video, self._yolo_path)

    def _start(self, video, yolo_path) -> None:
        # The thread is intentionally unparented and retires itself once the
        # worker finishes. Parenting it to the dialog means closing the dialog
        # mid-inference destroys a running QThread, which aborts the process —
        # and quit()/wait() cannot help, because quit only ends an event loop
        # and the worker is inside a blocking predict() call.
        self._thread = QThread()
        self._worker = _PreviewWorker(video, yolo_path, self._pose_csv)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_done(self, result) -> None:
        if result is None:
            self._on_failed("no animal was detected in the first frames of this video")
            return
        frame, points = result
        self.show_frame(annotate_keypoints(frame, points, self._names))

    def _on_failed(self, message: str) -> None:
        self._image.setText(
            f"Could not preview keypoints:\n{message}\n\n"
            "You can still run, but the labels have not been checked."
        )
        # Let them proceed deliberately rather than blocking on a preview.
        self._ok.setText("Run without checking")
        self._ok.setEnabled(True)

    def show_frame(self, annotated: np.ndarray) -> None:
        """Display an already-annotated BGR frame. Separated for testing."""
        rgb = np.ascontiguousarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._image.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._image.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._ok.setEnabled(True)

    def closeEvent(self, event):
        """Let the preview thread retire on its own.

        Nothing is waited on and nothing is dropped. The worker may be inside a
        multi-second model load, so joining it would freeze the UI; and clearing
        the Python references would let the QThread wrapper be collected while
        C++ still runs it, which aborts the process rather than raising.
        """
        super().closeEvent(event)


__all__ = [
    "KeypointConfirmDialog",
    "annotate_keypoints",
    "first_detected_frame",
    "first_tracked_frame",
]
