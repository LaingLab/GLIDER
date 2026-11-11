"""Live 3-D lift as one of the live pipeline's frame consumers.

Hand :func:`frame_consumer` to ``LiveInferenceConfig.frame_consumers``::

    from glider_lift3d import Lift3DConfig
    from glider_lift3d.live import frame_consumer

    config.frame_consumers = [frame_consumer(Lift3DConfig(ppc=11.7), csv_path=out)]

The tracker fans each ``(frame_idx, frame_bgr, keypoints, confidences)`` out to this
thread's queue as well as the feature engine and display. The lift runs at its own pace
(dropping frames under back-pressure like every other stage), writes a per-frame CSV,
and draws its newest rear weight and nose height on the live display.
"""

from __future__ import annotations

import csv
import logging
import queue
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

END_OF_STREAM = None


class LatestLift:
    """Newest lift result, shared with the display (lock-guarded)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame_idx = -1
        self._rear = 0.0
        self._nose = 0.0

    def update(self, frame_idx: int, rear_weight: float, nose_cm: float) -> None:
        with self._lock:
            self._frame_idx, self._rear, self._nose = frame_idx, rear_weight, nose_cm

    def get(self) -> tuple[int, float, float]:
        with self._lock:
            return self._frame_idx, self._rear, self._nose


def draw_lift_readout(frame, rear_weight: float, nose_cm: float):
    """Rear weight and estimated nose height in the bottom-left corner."""
    import cv2

    text = f"rear {rear_weight:4.2f}  nose {nose_cm:4.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 12, frame.shape[0] - 12
    colour = (80, 200, 255) if rear_weight >= 0.5 else (220, 220, 220)
    cv2.putText(frame, text, (x + 1, y + 1), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, 0.5, colour, 1, cv2.LINE_AA)
    return frame


class Lift3DConsumer(threading.Thread):
    def __init__(
        self,
        lift_queue: queue.Queue,
        stop_event: threading.Event,
        keypoint_names: list[str],
        config,
        csv_path: Path | None = None,
        lift=None,
    ):
        super().__init__(name="Lift3DConsumer", daemon=True)
        self.queue = lift_queue
        self.stop_event = stop_event
        self.keypoint_names = list(keypoint_names)
        self.config = config
        self.csv_path = Path(csv_path) if csv_path else None
        self.latest = LatestLift()
        self._lift = lift  # injected for tests
        self.error: str | None = None
        self.n_lifted = 0

    def draw_overlay(self, frame_bgr) -> None:
        idx, rear, nose = self.latest.get()
        if idx >= 0:
            draw_lift_readout(frame_bgr, rear, nose)

    def run(self) -> None:
        from .batch import CSV_COLUMNS, frame_row
        from .pipeline import Lift3D

        try:
            lift = self._lift or Lift3D(self.keypoint_names, self.config)
        except Exception as e:  # noqa: BLE001 - surfaced through self.error
            self.error = f"3-D lift unavailable: {e}"
            logger.warning(self.error)
            return
        fh = open(self.csv_path, "w", newline="") if self.csv_path else None  # noqa: SIM115
        writer = csv.writer(fh) if fh else None
        if writer:
            writer.writerow(CSV_COLUMNS)
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    return
                frame_idx, frame_bgr, keypoints, _conf = item
                try:
                    r = lift.process(frame_bgr, keypoints, frame_idx=frame_idx)
                except Exception as e:  # noqa: BLE001
                    self.error = f"3-D lift failed on frame {frame_idx}: {e}"
                    logger.exception(self.error)
                    return
                if writer:
                    writer.writerow(frame_row(frame_idx, self.config.fps, r))
                if r is not None:
                    self.n_lifted += 1
                    self.latest.update(frame_idx, r.rear_weight, float(r.anchors_cm[-1]))
        finally:
            if fh:
                fh.close()
            lift.close()


def frame_consumer(config, csv_path: Path | None = None):
    """A ``LiveInferenceConfig.frame_consumers`` factory for the 3-D lift."""

    def make(frame_queue, stop_event, keypoint_names, fps):
        config.fps = fps
        return Lift3DConsumer(frame_queue, stop_event, keypoint_names, config, csv_path)

    return make
