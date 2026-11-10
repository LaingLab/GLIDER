"""Live 3-D lift as an extra consumer of the pose tracker's output.

The tracker fans each ``(frame_idx, frame_bgr, keypoints, confidences)`` out to this
thread's queue as well as the feature engine and display. The lift runs at its own pace
(dropping frames under back-pressure like every other stage), writes a per-frame CSV,
and publishes its newest result for the display overlay.
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


class Lift3DConsumer(threading.Thread):
    def __init__(
        self,
        lift_queue: queue.Queue,
        stop_event: threading.Event,
        keypoint_names: list[str],
        config,
        csv_path: Path | None = None,
        latest: LatestLift | None = None,
        lift=None,
    ):
        super().__init__(name="Lift3DConsumer", daemon=True)
        self.lift_queue = lift_queue
        self.stop_event = stop_event
        self.keypoint_names = list(keypoint_names)
        self.config = config
        self.csv_path = Path(csv_path) if csv_path else None
        self.latest = latest
        self._lift = lift  # injected for tests
        self.error: str | None = None
        self.n_lifted = 0

    def run(self) -> None:
        from glider.vision.lift3d.batch import CSV_COLUMNS, frame_row
        from glider.vision.lift3d.pipeline import Lift3D

        try:
            lift = self._lift or Lift3D(self.keypoint_names, self.config)
        except Exception as e:  # noqa: BLE001 - surfaced through self.error
            self.error = f"3-D lift unavailable: {e}"
            logger.warning(self.error)
            self._drain()
            return
        fh = open(self.csv_path, "w", newline="") if self.csv_path else None  # noqa: SIM115
        writer = csv.writer(fh) if fh else None
        if writer:
            writer.writerow(CSV_COLUMNS)
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.lift_queue.get(timeout=0.1)
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
                    self._drain()
                    return
                if writer:
                    writer.writerow(frame_row(frame_idx, self.config.fps, r))
                if r is not None:
                    self.n_lifted += 1
                    if self.latest is not None:
                        self.latest.update(frame_idx, r.rear_weight, float(r.anchors_cm[-1]))
        finally:
            if fh:
                fh.close()
            lift.close()

    def _drain(self) -> None:
        """Keep emptying the queue after a failure so the tracker never blocks on it."""
        while not self.stop_event.is_set():
            try:
                if self.lift_queue.get(timeout=0.1) is END_OF_STREAM:
                    return
            except queue.Empty:
                continue
