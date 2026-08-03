"""The threads that make the live pipeline.

Five always run; a sixth (the EmbeddingProjector, in ``embedding.py``) is
added when the 3D embedding view is active.

Each class is a :class:`threading.Thread` subclass with a stop event
and bounded queues. None values on queues are the "end of stream"
sentinel that each thread propagates downstream before exiting.

Why bounded queues? Without a maxsize, a fast producer (camera) feeding
a slow consumer (the classifier) would blow memory in seconds. With a
small maxsize, the slow consumer applies back-pressure: the producer
blocks on ``put()`` and we naturally drop frames at the source — which
is the right behaviour for live video.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer
from glider.analysis.behavior.classify.features_stream import StreamingFeatureExtractor
from glider.analysis.behavior.classify.overlay import (
    color_for_behavior,
    draw_fps,
    draw_label_badge,
    draw_skeleton,
)
from glider.analysis.behavior.classify.pose_extract import extract_keypoints
from glider.analysis.behavior.features import FeatureSpec
from glider.analysis.behavior.model import BehaviorModel

logger = logging.getLogger(__name__)

# Sentinel on every queue meaning "no more items will come".
END_OF_STREAM = None


# ---------------------------------------------------------------------------
# Shared latest-label state (atomic-ish via a lock)
# ---------------------------------------------------------------------------


class LatestLabel:
    """Thread-safe holder for the most recently predicted behavior.

    The :class:`BehaviorClassifier` writes; the :class:`DisplayConsumer`
    reads. We use a lock rather than a queue because the display always
    wants the *latest* label, not a backlog.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._label: str = ""
        self._frame_idx: int = -1
        self._updated_at: float = 0.0

    def update(self, frame_idx: int, label: str) -> None:
        with self._lock:
            self._label = label
            self._frame_idx = int(frame_idx)
            self._updated_at = time.monotonic()

    def get(self) -> tuple[int, str, float]:
        """Return ``(frame_idx, label, seconds_since_update)``."""
        with self._lock:
            return (
                self._frame_idx,
                self._label,
                time.monotonic() - self._updated_at,
            )


# ---------------------------------------------------------------------------
# Thread 1: VideoProducer
# ---------------------------------------------------------------------------


class VideoProducer(threading.Thread):
    """Reads frames from a file or webcam and pushes them on ``raw_queue``.

    For file inputs we don't pace ourselves to real-time FPS by
    default — downstream applies back-pressure via the bounded queue,
    so the pipeline naturally runs as fast as the slowest stage allows.
    Set ``realtime=True`` to sleep between frames so a video file plays
    at its native rate (useful for "live demo" scenarios on recorded
    video).
    """

    def __init__(
        self,
        source: str | int,
        raw_queue: queue.Queue,
        stop_event: threading.Event,
        realtime: bool = False,
        fps_override: float | None = None,
    ):
        super().__init__(name="VideoProducer", daemon=True)
        self.source = source
        self.raw_queue = raw_queue
        self.stop_event = stop_event
        self.realtime = realtime
        self.fps_override = fps_override
        # Filled once the capture is open.
        self.fps: float = 30.0
        self.frame_size: tuple[int, int] = (0, 0)  # (width, height)
        self.n_frames: int = 0
        self.error: str | None = None

    def run(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.error = f"failed to open video source {self.source!r}"
            self.raw_queue.put(END_OF_STREAM)
            return

        self.fps = self.fps_override or float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 0.5:
            self.fps = 30.0
        self.frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_interval = 1.0 / self.fps if self.realtime else 0.0

        frame_idx = 0
        try:
            while not self.stop_event.is_set():
                t0 = time.monotonic()
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                # Put with a timeout so we can respond to stop_event
                # even if downstream is jammed.
                while not self.stop_event.is_set():
                    try:
                        self.raw_queue.put((frame_idx, frame_bgr), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                frame_idx += 1
                if frame_interval > 0:
                    elapsed = time.monotonic() - t0
                    sleep = frame_interval - elapsed
                    if sleep > 0:
                        time.sleep(sleep)
        finally:
            cap.release()
            self.raw_queue.put(END_OF_STREAM)


# ---------------------------------------------------------------------------
# Thread 2: PoseTracker (YOLO)
# ---------------------------------------------------------------------------


class PoseTracker(threading.Thread):
    """Pulls raw frames from ``raw_queue``, runs YOLO, fans out to two queues.

    ``tracked_queue`` feeds the :class:`FeatureEngine`. ``display_queue``
    feeds the :class:`DisplayConsumer` so the display gets the frame +
    keypoints without having to wait for the classifier to finish.
    """

    def __init__(
        self,
        raw_queue: queue.Queue,
        tracked_queue: queue.Queue,
        display_queue: queue.Queue,
        stop_event: threading.Event,
        yolo_model_path: str | Path,
        keypoint_names: list[str],
        conf_threshold: float = 0.25,
        device: str | None = None,
        undetected_dir: Path | None = None,
        pose_csv_out: Path | None = None,
        fps: float = 30.0,
    ):
        super().__init__(name="PoseTracker", daemon=True)
        # Poses are computed anyway; writing them costs one CSV and saves the
        # whole inference pass next time they are wanted -- for percentile or
        # cohort thresholds, or any downstream analysis.
        self.pose_csv_out = Path(pose_csv_out) if pose_csv_out else None
        self.fps = float(fps)
        self._pose_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
        # Frame size, learned from the first frame. Recorded with the poses so
        # the analysis viewer can draw them without the video.
        self._frame_size: tuple[int, int] | None = None
        self.raw_queue = raw_queue
        self.tracked_queue = tracked_queue
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.yolo_model_path = str(yolo_model_path)
        self.keypoint_names = list(keypoint_names)
        self.conf_threshold = float(conf_threshold)
        self.device = device
        # When set, raw frames where YOLO found no confident mouse (all
        # keypoints NaN) are written here as PNGs for re-labeling.
        self.undetected_dir = Path(undetected_dir) if undetected_dir else None
        self.n_undetected_saved = 0
        self.error: str | None = None

    def run(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            self.error = "ultralytics isn't installed; install with `pip install ultralytics`"
            self._propagate_eos()
            return

        try:
            model = YOLO(self.yolo_model_path)
        except Exception as e:
            self.error = f"failed to load YOLO model {self.yolo_model_path}: {e}"
            self._propagate_eos()
            return

        k = len(self.keypoint_names)
        if self.undetected_dir is not None:
            self.undetected_dir.mkdir(parents=True, exist_ok=True)
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.raw_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    self._propagate_eos()
                    return
                frame_idx, frame_bgr = item

                # YOLO model.predict on a single frame. verbose=False
                # silences per-frame progress chatter that floods the
                # terminal at 30 fps.
                kwargs = {"conf": self.conf_threshold, "verbose": False}
                if self.device is not None:
                    kwargs["device"] = self.device
                try:
                    results = model.predict(frame_bgr, **kwargs)
                except Exception as e:  # noqa: BLE001
                    self.error = f"YOLO predict failed on frame {frame_idx}: {e}"
                    self._propagate_eos()
                    return

                keypoints, confidences = extract_keypoints(
                    results[0] if results else None, self.conf_threshold, k
                )

                # No confident mouse this frame → optionally save it for
                # re-labeling (all keypoints came back NaN).
                if self.undetected_dir is not None and bool(np.isnan(keypoints).all()):
                    self._save_undetected(frame_idx, frame_bgr)

                if self._frame_size is None and frame_bgr is not None:
                    self._frame_size = (frame_bgr.shape[1], frame_bgr.shape[0])
                if self.pose_csv_out is not None:
                    # Recorded before the queues, so a frame dropped under
                    # back-pressure downstream is still in the pose record.
                    self._pose_rows.append((frame_idx, keypoints, confidences))

                payload = (frame_idx, frame_bgr, keypoints, confidences)
                # Push to both downstream queues; ignore Full timeouts
                # because we'd rather drop a frame than block the YOLO
                # forward pass.
                _put_or_drop(self.tracked_queue, payload, self.stop_event)
                _put_or_drop(self.display_queue, payload, self.stop_event)
        finally:
            self._write_pose_csv()
            self._propagate_eos()

    def _write_pose_csv(self) -> None:
        """Write the tracked poses as a DeepLabCut CSV, if one was asked for.

        Frames the producer dropped leave gaps in the index, so the array is
        laid out over the full range and missing frames stay NaN — DLC CSVs are
        positional, and silently compacting them would shift every timestamp.
        """
        if self.pose_csv_out is None or not self._pose_rows:
            return
        try:
            from glider.vision.pose.core import PoseData
            from glider.vision.pose.dlc import to_dlc_csv

            n_frames = max(idx for idx, _, _ in self._pose_rows) + 1
            k = len(self.keypoint_names)
            xy = np.full((n_frames, k, 2), np.nan)
            conf = np.zeros((n_frames, k))
            for idx, keypoints, confidences in self._pose_rows:
                xy[idx] = keypoints
                conf[idx] = confidences

            self.pose_csv_out.parent.mkdir(parents=True, exist_ok=True)
            to_dlc_csv(
                PoseData(
                    xy=xy,
                    confidence=conf,
                    keypoint_names=self.keypoint_names,
                    fps=self.fps,
                    metadata={"resolution": self._frame_size},
                ),
                self.pose_csv_out,
            )
        except Exception:  # noqa: BLE001 - a failed save must not fail the run
            logger.warning("could not write pose CSV to %s", self.pose_csv_out, exc_info=True)

    def _save_undetected(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        """Write a frame with no confident detection to the undetected dir."""
        import cv2

        path = self.undetected_dir / f"undetected_{frame_idx:07d}.png"
        try:
            if cv2.imwrite(str(path), frame_bgr):
                self.n_undetected_saved += 1
        except Exception:  # noqa: BLE001 - never let disk I/O kill the pipeline
            pass

    def _propagate_eos(self) -> None:
        # Make sure both downstream consumers see end-of-stream.
        try:
            self.tracked_queue.put(END_OF_STREAM, timeout=1.0)
        except queue.Full:
            pass
        try:
            self.display_queue.put(END_OF_STREAM, timeout=1.0)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Thread 3: FeatureEngine
# ---------------------------------------------------------------------------


class PoseReplay(threading.Thread):
    """A :class:`PoseTracker` that reads poses from a CSV instead of running YOLO.

    Batch Pose Tracking has usually already tracked these videos, and running
    the pose model a second time to produce the same numbers is the single
    most expensive thing the apply path does. This is a drop-in for the
    tracker: same queues, same payload shape, so the feature engine, the
    annotated video and the ethogram are all unaffected.

    Frames still flow through ``raw_queue`` because the annotated video needs
    them, and pairing is by frame index -- a pose CSV is positional, so row N
    describes frame N. A frame with no row (the CSV is shorter, or the row is
    all-NaN) is passed on with NaN keypoints, exactly as an undetected frame
    would be.
    """

    def __init__(
        self,
        raw_queue: queue.Queue,
        tracked_queue: queue.Queue,
        display_queue: queue.Queue,
        stop_event: threading.Event,
        pose_csv: str | Path,
        keypoint_names: list[str],
    ):
        super().__init__(name="PoseReplay", daemon=True)
        self.raw_queue = raw_queue
        self.tracked_queue = tracked_queue
        self.display_queue = display_queue
        self.stop_event = stop_event
        self.pose_csv = Path(pose_csv)
        self.keypoint_names = list(keypoint_names)
        self.error: str | None = None
        self.n_frames_without_pose = 0

    def run(self) -> None:
        try:
            xy, conf = self._load()
        except Exception as e:  # noqa: BLE001 - surfaced through self.error
            self.error = f"could not read {self.pose_csv.name}: {e}"
            self._propagate_eos()
            return

        n_rows = xy.shape[0]
        k = len(self.keypoint_names)
        blank = np.full((k, 2), np.nan)
        blank_conf = np.zeros(k)
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.raw_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    return
                frame_idx, frame_bgr = item
                if 0 <= frame_idx < n_rows:
                    keypoints, confidences = xy[frame_idx], conf[frame_idx]
                else:
                    # The CSV ran out before the video did.
                    self.n_frames_without_pose += 1
                    keypoints, confidences = blank, blank_conf
                payload = (frame_idx, frame_bgr, keypoints, confidences)
                _put_or_drop(self.tracked_queue, payload, self.stop_event)
                _put_or_drop(self.display_queue, payload, self.stop_event)
        finally:
            self._propagate_eos()

    def _load(self):
        from glider.vision.pose.dlc import from_dlc_csv

        pose = from_dlc_csv(self.pose_csv)
        if len(pose.keypoint_names) != len(self.keypoint_names):
            raise ValueError(
                f"it has {len(pose.keypoint_names)} keypoints "
                f"({', '.join(pose.keypoint_names)}) but {len(self.keypoint_names)} "
                "names are configured; the features would not line up"
            )
        return pose.xy, pose.confidence

    def _propagate_eos(self) -> None:
        for q in (self.tracked_queue, self.display_queue):
            try:
                q.put(END_OF_STREAM, timeout=1.0)
            except queue.Full:
                pass


class FeatureEngine(threading.Thread):
    """Per-frame features → sliding buffer → emit rolling row to classifier.

    Velocity/acceleration come from ``np.gradient``, which is *centered* in
    the interior — so to match the training features we keep a 5-frame
    keypoint history and emit the MIDDLE frame's row (centered velocity needs
    the ±1 neighbours, centered acceleration the ±2). This costs a 2-frame
    (~66 ms) latency but makes the live features identical to training, so
    live accuracy tracks the cross-validated numbers instead of drifting
    (a one-sided ``iloc[-1]`` row diverged on ~9% of frames, worst on dig).

    The classifier doesn't need a prediction on *every* frame — for
    live display, 10 Hz is plenty. ``predict_every`` controls the
    cadence (default 3 frames @ 30 fps = 10 Hz).
    """

    def __init__(
        self,
        tracked_queue: queue.Queue,
        classifier_queue: queue.Queue,
        stop_event: threading.Event,
        spec: FeatureSpec,
        keypoint_names: list[str],
        window: int,
        stats: tuple[str, ...],
        per_frame_feature_names: list[str],
        predict_every: int = 3,
        spectral_features: list[str] | None = None,
        freeze_threshold: float | None = None,
        dart_threshold: float | None = None,
        freeze_min_frames: int = 30,
        dart_min_frames: int = 3,
    ):
        super().__init__(name="FeatureEngine", daemon=True)
        self.tracked_queue = tracked_queue
        self.classifier_queue = classifier_queue
        self.stop_event = stop_event
        self.spec = spec
        self.keypoint_names = list(keypoint_names)
        self.window = int(window)
        self.stats = tuple(stats)
        self.predict_every = max(1, int(predict_every))
        self.per_frame_feature_names = list(per_frame_feature_names)
        # Pure per-frame feature core: a 5-frame keypoint ring that emits the
        # MIDDLE frame's features (centered velocity ±1, acceleration ±2),
        # matching training's whole-session np.gradient. Extracted so a future
        # live classifier can reuse the EXACT same math (live == offline).
        self._extractor = StreamingFeatureExtractor(
            spec=self.spec,
            keypoint_names=self.keypoint_names,
            fps=30.0,
        )
        # Rolling buffer of per-frame features.
        self.buffer = SlidingFeatureBuffer(
            feature_names=self.per_frame_feature_names,
            window=self.window,
            stats=self.stats,
            spectral_features=spectral_features,
        )
        self._tick: int = 0
        # Centered-feature lag: the emitted windowed row describes the MIDDLE
        # frame, which trails the current frame by this many frames.
        self._lag = self._extractor.lag
        # Optional speed axis (freeze/dart heuristic), active only when both
        # absolute thresholds are given. Its per-frame label is buffered by
        # `_lag` so it aligns with the middle (posture) frame on emit.
        self._speed_axis = freeze_threshold is not None and dart_threshold is not None
        if self._speed_axis:
            from glider.analysis.behavior.classify.speed_state import (
                CausalSpeed,
                FreezeDartDetector,
            )

            self._causal_speed = CausalSpeed()
            self._freeze_dart = FreezeDartDetector(
                freeze_threshold,
                dart_threshold,
                freeze_min_frames=freeze_min_frames,
                dart_min_frames=dart_min_frames,
            )
            self._speed_labels: deque[str] = deque(maxlen=self._lag + 1)
            # The numeric speed behind each label, carried alongside it so the
            # ethogram can report a real per-frame value. The label alone is
            # blank whenever the animal sits between the thresholds, which is
            # most of a recording — a number is what an analysis actually wants.
            self._speed_values: deque[float] = deque(maxlen=self._lag + 1)

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.tracked_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    self._end()
                    return
                frame_idx, _frame_bgr, keypoints, _confidences = item
                if self._speed_axis:
                    spd = self._causal_speed.push(keypoints)
                    self._speed_labels.append(self._freeze_dart.push(spd))
                    self._speed_values.append(float(spd))

                # Compute per-frame features from the keypoint history.
                feats = self._extractor.push(keypoints)
                if feats is not None:
                    self.buffer.push_features(feats)

                self._tick += 1
                # Emit a windowed feature row at the configured cadence.
                if self._tick % self.predict_every != 0:
                    continue
                if len(self.buffer) == 0:
                    continue
                column_names, row = self.buffer.rolling_features()
                # The newest per-frame row in the buffer is the MIDDLE of the
                # kp history, so this windowed row describes a frame lagging the
                # current one by self._lag. Tag it with that index so the
                # ethogram timeline stays aligned.
                mid_idx = max(0, frame_idx - self._lag)
                if self._speed_axis:
                    # Speed label for that same middle frame = the oldest entry
                    # in the lag-sized buffer (once it has filled).
                    filled = len(self._speed_labels) == self._speed_labels.maxlen
                    speed_label = self._speed_labels[0] if filled else ""
                    speed_px = self._speed_values[0] if filled else float("nan")
                    out = (mid_idx, column_names, row, speed_label, speed_px)
                else:
                    out = (mid_idx, column_names, row)
                _put_or_drop(self.classifier_queue, out, self.stop_event)
        finally:
            self._end()

    def _end(self) -> None:
        try:
            self.classifier_queue.put(END_OF_STREAM, timeout=1.0)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Thread 4: BehaviorClassifier
# ---------------------------------------------------------------------------


class BehaviorClassifier(threading.Thread):
    """Pulls windowed feature rows, runs ``model.predict_one``, updates LatestLabel.

    ``confidence_threshold`` is forwarded to the model: when the top-
    class probability is below it, the classifier emits "" so the
    overlay shows "unknown" instead of force-classifying. Use this
    with models trained without ``--with-background``.
    """

    def __init__(
        self,
        classifier_queue: queue.Queue,
        latest_label: LatestLabel,
        stop_event: threading.Event,
        model: BehaviorModel,
        ethogram_path: Path | None = None,
        confidence_threshold: float = 0.0,
        smooth_window: int = 1,
        class_thresholds: dict[str, float] | None = None,
        embedding_queue: queue.Queue | None = None,
        speed_axis: bool = False,
        cm_s_per_px_frame: float | None = None,
    ):
        super().__init__(name="BehaviorClassifier", daemon=True)
        # Set before anything can write the ethogram; None = no pixel scale
        # was supplied, so speed_cm_s stays blank rather than guessed.
        self.cm_s_per_px_frame = cm_s_per_px_frame
        self.classifier_queue = classifier_queue
        self.latest_label = latest_label
        self.stop_event = stop_event
        self.model = model
        self.ethogram_path = ethogram_path
        self.confidence_threshold = float(confidence_threshold)
        self.class_thresholds = dict(class_thresholds) if class_thresholds else None
        # Optional tap for the 3D embedding view: the model-ordered
        # feature vector + predicted label are dropped here for the
        # EmbeddingProjector to transform off this hot path. None = no
        # embedding view this run.
        self.embedding_queue = embedding_queue
        # Majority-vote smoother over the last `smooth_window` predictions;
        # 1 = off. Stabilises the overlay, video, and ethogram identically.
        from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

        self._smoother = MajorityVoteSmoother(window=smooth_window)
        # Two-axis output: when on, write a (posture, speed) ethogram and let
        # the speed axis win the displayed label.
        self.speed_axis = bool(speed_axis)
        # Buffered (frame_idx, posture, speed label, speed px/frame) rows.
        self._ethogram: list[tuple[int, str, str, float]] = []

    def _to_cm_s(self, px_per_frame: float) -> float:
        """Pixels/frame -> cm/s, or NaN when no pixel scale was supplied.

        One precomputed factor rather than carrying fps and px_per_mm down
        here separately; the caller knows both and neither varies per frame.
        """
        factor = getattr(self, "cm_s_per_px_frame", None)
        if not factor:
            return float("nan")
        return float(px_per_frame) * float(factor)

    def run(self) -> None:

        feature_names = self.model.feature_names
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.classifier_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    return
                if len(item) == 5:
                    frame_idx, column_names, row, speed_label, speed_px = item
                elif len(item) == 4:
                    frame_idx, column_names, row, speed_label = item
                    speed_px = float("nan")
                else:
                    frame_idx, column_names, row = item
                    speed_label, speed_px = "", float("nan")
                # Map the row into the model's expected column order.
                # The FeatureEngine builds column_names in its own
                # configured order; if it matches, this is a no-op.
                if column_names == feature_names:
                    aligned = row
                else:
                    pos = {n: i for i, n in enumerate(column_names)}
                    aligned = np.full(len(feature_names), np.nan)
                    for i, name in enumerate(feature_names):
                        j = pos.get(name)
                        if j is not None:
                            aligned[i] = row[j]
                raw_label = self.model.predict_one(
                    aligned,
                    confidence_threshold=self.confidence_threshold,
                    class_thresholds=self.class_thresholds,
                )
                label = self._smoother.push(raw_label)
                # Speed axis (freezing/darting) takes precedence on the display.
                self.latest_label.update(frame_idx, speed_label or label)
                if self.embedding_queue is not None:
                    # Tap the model-ordered feature vector + label for the
                    # 3D embedding view. Drop-on-full so the projector
                    # falling behind never stalls predictions.
                    _put_or_drop(self.embedding_queue, (aligned, label), self.stop_event)
                if self.ethogram_path is not None:
                    self._ethogram.append((int(frame_idx), label, speed_label, speed_px))
        finally:
            if self.ethogram_path is not None and self._ethogram:
                self._write_ethogram()

    def _write_ethogram(self) -> None:
        import csv

        try:
            self.ethogram_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ethogram_path.open("w", newline="") as f:
                w = csv.writer(f)
                if self.speed_axis:
                    # behavior       -- one label per frame. An animal cannot be
                    #                   darting and digging at once, so the
                    #                   speed axis wins where it fired.
                    # speed_px_frame -- always present, the raw measured signal
                    # speed_cm_s     -- only when a pixel scale was supplied
                    w.writerow(["frame", "behavior", "speed_px_frame", "speed_cm_s"])
                    for fidx, lab, spd, px in self._ethogram:
                        w.writerow([fidx, spd or lab, _fmt(px), _fmt(self._to_cm_s(px))])
                else:
                    w.writerow(["frame", "behavior"])
                    for fidx, lab, _spd, _px in self._ethogram:
                        w.writerow([fidx, lab])
        except OSError:
            # Don't crash the pipeline on a save failure; the user will
            # see the error in stdout via the orchestrator.
            pass


# ---------------------------------------------------------------------------
# Thread 3+4 (CNN): SequenceClassifier
# ---------------------------------------------------------------------------


class SequenceClassifier(threading.Thread):
    """Live inference for CNN :class:`~glider.analysis.behavior.sequence.SequenceModel`.

    Replaces the FeatureEngine + BehaviorClassifier pair on the CNN path:
    there are no tabular features to compute, so this thread just buffers
    the last ``model.window`` raw keypoint frames and calls
    ``model.predict_window`` at the configured cadence, updating
    :class:`LatestLabel` (and an optional ethogram). Frames before the
    buffer fills, or with missing keypoints, yield a blank label — the
    same "unknown" contract as the tabular path.
    """

    def __init__(
        self,
        tracked_queue: queue.Queue,
        latest_label: LatestLabel,
        stop_event: threading.Event,
        model,
        predict_every: int = 3,
        smooth_window: int = 1,
        ethogram_path=None,
        gate=None,
    ):
        super().__init__(name="SequenceClassifier", daemon=True)
        from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

        self.tracked_queue = tracked_queue
        self.latest_label = latest_label
        self.stop_event = stop_event
        self.model = model
        self.gate = gate
        self.predict_every = max(1, int(predict_every))
        self.window = int(model.window)
        self._buf: deque[np.ndarray] = deque(maxlen=self.window)
        self._smoother = MajorityVoteSmoother(window=smooth_window)
        self.ethogram_path = ethogram_path
        self._ethogram: list[tuple[int, str]] = []
        self._tick = 0

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    item = self.tracked_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    return
                frame_idx, _frame_bgr, keypoints, _confidences = item
                self._buf.append(np.asarray(keypoints, dtype=np.float64))
                self._tick += 1
                if self._tick % self.predict_every != 0:
                    continue
                if len(self._buf) < self.window:
                    continue
                window_xy = np.stack(self._buf)  # (window, K, 2)
                raw_label = self.model.predict_window(window_xy, gate=self.gate)
                label = self._smoother.push(raw_label)
                self.latest_label.update(frame_idx, label)
                if self.ethogram_path is not None:
                    self._ethogram.append((int(frame_idx), label))
        finally:
            if self.ethogram_path is not None and self._ethogram:
                self._write_ethogram()

    def _write_ethogram(self) -> None:
        import csv

        try:
            self.ethogram_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ethogram_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame", "behavior"])
                for fidx, lab in self._ethogram:
                    w.writerow([fidx, lab])
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Thread 5: DisplayConsumer
# ---------------------------------------------------------------------------


class DisplayConsumer(threading.Thread):
    """Renders frame + skeleton + latest behavior label to screen / file.

    Exactly one of ``display_window`` or ``video_writer_path`` must be
    set (both is also fine — same frame goes to both sinks).
    """

    def __init__(
        self,
        display_queue: queue.Queue,
        latest_label: LatestLabel,
        stop_event: threading.Event,
        keypoint_names: list[str],
        edges: list[tuple[int, int]] | None = None,
        display_window: bool = True,
        video_writer_path: Path | None = None,
        fps: float = 30.0,
        frame_size: tuple[int, int] | None = None,
        vocab_order: list[str] | None = None,
    ):
        super().__init__(name="DisplayConsumer", daemon=True)
        self.display_queue = display_queue
        self.latest_label = latest_label
        self.stop_event = stop_event
        self.keypoint_names = list(keypoint_names)
        self.edges = list(edges) if edges else None
        self.display_window = bool(display_window)
        self.video_writer_path = video_writer_path
        self.fps = float(fps)
        self.frame_size = frame_size
        self.vocab_order = list(vocab_order) if vocab_order else None
        self._writer = None
        self._last_render_time = time.monotonic()
        self._render_fps = 0.0

    def run(self) -> None:
        import cv2

        try:
            while not self.stop_event.is_set():
                try:
                    item = self.display_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is END_OF_STREAM:
                    return
                _frame_idx, frame_bgr, keypoints, confidences = item

                # Update render FPS estimate with EMA so the overlay
                # doesn't flicker. Quick coefficient = 0.1.
                now = time.monotonic()
                dt = now - self._last_render_time
                if dt > 0:
                    inst = 1.0 / dt
                    self._render_fps = (
                        0.9 * self._render_fps + 0.1 * inst if self._render_fps > 0 else inst
                    )
                self._last_render_time = now

                draw_skeleton(
                    frame_bgr,
                    keypoints=keypoints,
                    confidences=confidences,
                    edges=self.edges,
                )
                _, label, age = self.latest_label.get()
                if label:
                    color = color_for_behavior(label, self.vocab_order)
                else:
                    color = (60, 60, 60)
                draw_label_badge(frame_bgr, label or "(waiting...)", color)
                draw_fps(frame_bgr, self._render_fps)

                if self.video_writer_path is not None:
                    self._ensure_writer(frame_bgr)
                    if self._writer is not None:
                        self._writer.write(frame_bgr)

                if self.display_window:
                    cv2.imshow("GLIDER · live", frame_bgr)
                    # 1 ms wait so cv2 actually pumps its event loop;
                    # Q or Esc quits.
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self.stop_event.set()
                        return
        finally:
            if self.display_window:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            if self._writer is not None:
                self._writer.release()

    def _ensure_writer(self, frame_bgr: np.ndarray) -> None:
        if self._writer is not None:
            return
        import cv2

        h, w = frame_bgr.shape[:2]
        # mp4v: broad compatibility, modest file size. The user can
        # transcode if they need H.264.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.video_writer_path),
            fourcc,
            self.fps,
            (w, h),
        )
        if not self._writer.isOpened():
            self._writer = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    """CSV cell for a float: blank for NaN, so an unmeasured frame reads as
    missing rather than as a real zero."""
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.4f}"


def _put_or_drop(
    q: queue.Queue,
    item,
    stop_event: threading.Event,
    block_timeout: float = 0.05,
) -> None:
    """Put with a brief block; on Full, drop. Always polls stop_event.

    Drop-on-full is the right behaviour for live video: a backed-up
    consumer means we're falling behind. Better to silently skip the
    frame than to grow memory unboundedly.
    """
    if stop_event.is_set():
        return
    try:
        q.put(item, timeout=block_timeout)
    except queue.Full:
        # Caller's frame is dropped. The downstream stage will catch up
        # on subsequent ticks.
        return
