"""Orchestrator: starts the pipeline threads, manages lifecycle, exposes a CLI-friendly API.

Five worker threads always run (producer → tracker → feature engine →
classifier → display).

Typical usage::

    config = LiveInferenceConfig(
        source="my_video.mp4",
        keypoint_names=["snout", "left_ear", ..., "tail_base"],
        yolo_model_path="best.pt",
        behavior_model_path="behavior_model.pkl",
        display=True,
        output_video="annotated.mp4",
        ethogram_csv="ethogram.csv",
    )
    pipeline = LiveInferencePipeline(config)
    pipeline.run()  # blocks until video ends or user presses Q

Press ``q`` or ``ESC`` in the display window to stop early.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from glider.analysis.behavior.classify.threads import (
    BehaviorClassifier,
    DisplayConsumer,
    FeatureEngine,
    LatestLabel,
    PoseReplay,
    PoseTracker,
    SequenceClassifier,
    VideoProducer,
)
from glider.analysis.behavior.model import BehaviorModel


def _is_hybrid_bundle(path) -> bool:
    """Whether ``path`` is a HybridModel save rather than a plain bundle."""
    try:
        import joblib

        payload = joblib.load(path)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("kind") == "hybrid"


def _load_behavior_model(path):
    """Load either a CNN sequence bundle or a tabular BehaviorModel.

    CNN bundles (``train --classifier cnn``) are torch saves with a
    ``format`` marker; everything else is a joblib BehaviorModel. Try the
    CNN loader first and fall back — so the live command transparently
    accepts both kinds of model.

    A HybridModel bundle is rejected by name. The live path has no way to
    supply the windowed columns its kinematic prior needs, so it could only
    ever run the base classifier — and silently doing that would leave the
    operator believing the prior was in play when it was not.
    """
    try:
        from glider.analysis.behavior.sequence import SequenceModel

        return SequenceModel.load(path)
    except Exception:
        pass
    try:
        return BehaviorModel.load(path)
    except Exception:
        if _is_hybrid_bundle(path):
            raise ValueError(
                f"{path} is a hybrid model bundle, which the apply/live path cannot run: "
                "its kinematic prior needs windowed features this pipeline does not "
                "produce, so only the base classifier would apply and the prior would "
                "be silently ignored. Train and save a plain model for apply, or set "
                "the freeze/dart thresholds directly (freeze_mm_s / dart_mm_s)."
            ) from None
        raise


def _unstreamable_feature_families(feature_names) -> list[str]:
    """Feature-name stems the live FeatureEngine can't reproduce in real time.

    ``motion_*`` needs the source video (egocentric frame differencing) and
    ``traj_*``'s live path isn't wired. A model trained with either emits NaN
    for those columns on every frame, so :meth:`BehaviorModel.predict_one`
    returns ``""`` every frame and the overlay sticks on "(waiting...)".
    Returning the offending stems lets the pipeline fail loudly at startup
    instead of silently producing blank predictions forever.
    """
    return sorted(
        {
            c.split("__")[0]
            for c in feature_names
            if c.startswith("motion_") or c.startswith("traj_")
        }
    )


# Queue capacities. Small enough that a slow downstream stage applies
# back-pressure within a few frames; large enough to absorb jitter.
RAW_QUEUE_MAX = 4
TRACKED_QUEUE_MAX = 4
DISPLAY_QUEUE_MAX = 4
CLASSIFIER_QUEUE_MAX = 8


@dataclass
class LiveInferenceConfig:
    """All knobs for one live-inference run."""

    source: str | int  # video path or camera index
    keypoint_names: list[str]
    # Both are None-able because the vectorised apply path in `.batch` needs
    # neither: poses already on disk make the tracker redundant, and a
    # speed-only run (freezing/darting from the speed trace alone) has no
    # behaviour model to load. The streaming pipeline below still requires
    # both, and says so when it is handed neither.
    yolo_model_path: str | Path | None
    behavior_model_path: str | Path | None
    # Optional sinks. At least one of display / output_video should be on
    # for there to be a point in running, but the orchestrator doesn't
    # enforce that — running headless with just an ethogram CSV is fine.
    display: bool = True
    output_video: Path | None = None
    ethogram_csv: Path | None = None
    # When set, raw frames where YOLO found no confident mouse are written
    # here as PNGs (one per undetected frame) so they can be re-labeled to
    # improve the pose model.
    undetected_dir: Path | None = None
    # Real-time pacing on file inputs (sleep between frames to match
    # video FPS). Default: True for cameras (they self-pace), False for
    # files (run as fast as YOLO allows).
    realtime: bool = False
    fps_override: float | None = None
    # Classifier cadence (every N tracked frames). 3 ≈ 10 Hz at 30 fps.
    predict_every: int = 3
    # Optional edge list for the skeleton overlay
    # (list of (kp_index_a, kp_index_b)). Defaults to a consecutive chain.
    edges: list[tuple[int, int]] | None = None
    # YOLO knobs.
    conf_threshold: float = 0.25
    device: str | None = None
    # Optional MovementGate (CNN models only): relabel a translation-
    # requiring class when the body didn't actually move.
    move_gate: object = None
    # Behavior-classifier confidence threshold: when the trained model's
    # top-class probability falls below this, the live overlay shows
    # "unknown" instead of force-classifying. 0.0 = always emit the top
    # class (legacy). Most useful with models trained WITHOUT
    # --with-background, where confidence_threshold gives you the
    # "nothing of interest" signal at inference time instead.
    behavior_confidence_threshold: float = 0.0
    # Optional per-behavior firing thresholds {behavior: τ}. A frame fires
    # the highest-probability behavior that clears its OWN τ; unlisted
    # behaviors fall back to behavior_confidence_threshold. Lets a clean
    # minority class fire low while a noisy one sits high. None = use the
    # single global threshold above.
    behavior_class_thresholds: dict[str, float] | None = None
    # Majority-vote smoothing over the last N classifier predictions to
    # stabilise the label stream (overlay, video, ethogram). 1 = off.
    smooth_window: int = 1
    # Offline-only: a CENTRED vote of this many frames, replacing the causal
    # one above. Ignored by the live path, which cannot read frames it has not
    # reached. Scoring a recording can, and it is worth 0.780 -> 0.823 macro F1
    # on held-out data; a causal vote of the same width instead lags every bout
    # boundary by half its window. 0 = keep the causal behaviour.
    offline_smooth_window: int = 0
    # Realtime 3D behavior-embedding window: removed (Phase-2, Qt-dependent).
    # Kept as a config field for compatibility with persisted configs; always
    # treated as off.
    embedding: bool = False
    # Optional live speed axis (freeze/dart heuristic). With both absolute
    # thresholds set, the FeatureEngine runs the detector, the ethogram gains a
    # `speed` column, and the displayed label shows freezing/darting over the
    # postural label. Calibrate once per rig (see the `calibrate-speed` command).
    freeze_threshold: float | None = None
    dart_threshold: float | None = None
    freeze_min_frames: int = 30
    dart_min_frames: int = 3
    # Multiply a px/frame speed by this to get cm/s in the ethogram. None
    # leaves speed_cm_s blank -- the scale is a calibration the operator may
    # simply not have, and a guessed number in real units is worse than none.
    cm_s_per_px_frame: float | None = None
    # Where to write the tracked poses as a DeepLabCut CSV. None = discard
    # them, which is the old behaviour; the poses are computed either way.
    pose_csv_out: Path | None = None
    # Read poses from this CSV instead of running the pose model. The single
    # biggest cost in an apply run is re-deriving poses that Batch Pose
    # Tracking already computed.
    pose_csv_in: Path | None = None


def _make_tracker(config, raw_queue, tracked_queue, display_queue, stop_event):
    """PoseReplay when a pose CSV was supplied, else the YOLO PoseTracker.

    Both present the same queues and payload shape, so nothing downstream --
    features, annotated video, ethogram -- can tell which one ran.
    """
    if config.pose_csv_in:
        return PoseReplay(
            raw_queue=raw_queue,
            tracked_queue=tracked_queue,
            display_queue=display_queue,
            stop_event=stop_event,
            pose_csv=config.pose_csv_in,
            keypoint_names=config.keypoint_names,
        )
    return PoseTracker(
        raw_queue=raw_queue,
        tracked_queue=tracked_queue,
        display_queue=display_queue,
        stop_event=stop_event,
        yolo_model_path=config.yolo_model_path,
        keypoint_names=config.keypoint_names,
        conf_threshold=config.conf_threshold,
        device=config.device,
        undetected_dir=config.undetected_dir,
        # Never re-write a CSV we just read from.
        pose_csv_out=None if config.pose_csv_in else config.pose_csv_out,
        fps=config.fps_override or 30.0,
    )


class LiveInferencePipeline:
    """Owns the queues + threads + shared state for one run."""

    def __init__(self, config: LiveInferenceConfig, model=None):
        """``model`` lets a caller that already loaded the bundle hand it over.

        Loading some bundles twice in one process has been observed to corrupt
        native memory (the umap/pynndescent index in a bundle that cannot be
        rebuilt fails *during* unpickling, and the second attempt takes the
        process down in unrelated C code). Reusing the loaded object avoids
        that entirely, and saves several seconds per video besides.
        """
        self.config = config
        self.stop_event = threading.Event()
        self.latest_label = LatestLabel()

        # Queues.
        self.raw_queue: queue.Queue = queue.Queue(maxsize=RAW_QUEUE_MAX)
        self.tracked_queue: queue.Queue = queue.Queue(maxsize=TRACKED_QUEUE_MAX)
        self.display_queue: queue.Queue = queue.Queue(maxsize=DISPLAY_QUEUE_MAX)
        self.classifier_queue: queue.Queue = queue.Queue(maxsize=CLASSIFIER_QUEUE_MAX)

        # Load the behavior model up front so we fail loudly if it's
        # missing — before we spin up any threads. Streaming a video without
        # one is not a mode: every frame here goes through the classifier.
        # Speed-only runs are served by `.batch`, which never gets this far.
        if model is None and config.behavior_model_path is None:
            raise ValueError(
                "the streaming pipeline classifies every frame, so it needs a "
                "behaviour model; a speed-only run scores poses on disk through "
                "glider.analysis.behavior.classify.batch instead"
            )
        self.model = (
            model if model is not None else _load_behavior_model(config.behavior_model_path)
        )

        # CNN sequence models take a separate, simpler thread path (raw
        # keypoint window → egocentric → predict_window); no tabular feature
        # engine or classifier.
        from glider.analysis.behavior.sequence import SequenceModel

        self.is_sequence = isinstance(self.model, SequenceModel)
        if self.is_sequence:
            if config.freeze_threshold is not None or config.dart_threshold is not None:
                print(
                    "  note: the freeze/dart speed axis isn't wired for CNN "
                    "models yet; --freeze/--dart-threshold ignored."
                )
            self.embedding_active = False
            self.embedding_queue = None
            self.feature_engine = None
            self.classifier = None
            self._build_sequence_threads(config)
            return

        # The realtime 3D embedding view has been removed (Phase-2,
        # Qt-dependent). embedding_queue stays defined as None so it can
        # still be forwarded to BehaviorClassifier unconditionally below.
        self.embedding_active = False
        self.embedding_queue: queue.Queue | None = None

        # The model's windowed feature names include suffixes like
        # __mean / __std / __max. To stream them, the FeatureEngine
        # needs the *base* feature names — derived by stripping the
        # suffix off the model's first stat's columns.
        first_stat = self.model.stats[0]
        suffix = f"__{first_stat}"
        per_frame_feature_names = [
            c[: -len(suffix)] for c in self.model.feature_names if c.endswith(suffix)
        ]
        if not per_frame_feature_names:
            raise ValueError(
                f"could not derive per-frame feature names from the model; "
                f"first stat is {first_stat!r} but no columns end with "
                f"{suffix!r}"
            )
        self.per_frame_feature_names = per_frame_feature_names

        # Derive which kinematic base features carry rolling spectral
        # columns (__domfreq) so the live buffer reproduces them. Empty
        # for models trained without --freq-features — no bundle schema
        # change needed; presence of the columns is the signal.
        self.spectral_features = [
            c[: -len("__domfreq")] for c in self.model.feature_names if c.endswith("__domfreq")
        ]

        # Some feature families can't be computed in the live path: motion_*
        # needs the source video (frame differencing) and traj_* isn't wired.
        # A model using them emits NaN for those columns every frame, so every
        # prediction is blank and the overlay stays on "(waiting...)". Fail
        # loudly here instead of letting the user stare at a stuck overlay.
        unstreamable = _unstreamable_feature_families(self.model.feature_names)
        if unstreamable:
            raise RuntimeError(
                "this behavior model needs features the live pipeline can't "
                f"compute in real time: {unstreamable}. Every frame would be "
                "NaN, so the overlay would stay on '(waiting...)'. Train "
                "without --motion-features/--traj-features (a pose-only or "
                "--freq-features model runs live) for live inference."
            )

        # Build the FeatureSpec the model was trained with (so feature
        # extraction matches at training and inference time).
        self.spec = self.model.spec

        # Pre-build the threads (started later via run()).
        self.producer = VideoProducer(
            source=config.source,
            raw_queue=self.raw_queue,
            stop_event=self.stop_event,
            realtime=config.realtime,
            fps_override=config.fps_override,
        )
        self.tracker = _make_tracker(
            config, self.raw_queue, self.tracked_queue, self.display_queue, self.stop_event
        )
        self.feature_engine = FeatureEngine(
            tracked_queue=self.tracked_queue,
            classifier_queue=self.classifier_queue,
            stop_event=self.stop_event,
            spec=self.spec,
            keypoint_names=config.keypoint_names,
            window=self.model.window,
            stats=self.model.stats,
            per_frame_feature_names=self.per_frame_feature_names,
            predict_every=config.predict_every,
            spectral_features=self.spectral_features,
            freeze_threshold=config.freeze_threshold,
            dart_threshold=config.dart_threshold,
            freeze_min_frames=config.freeze_min_frames,
            dart_min_frames=config.dart_min_frames,
        )
        self.classifier = BehaviorClassifier(
            classifier_queue=self.classifier_queue,
            latest_label=self.latest_label,
            stop_event=self.stop_event,
            model=self.model,
            ethogram_path=(Path(config.ethogram_csv) if config.ethogram_csv else None),
            confidence_threshold=config.behavior_confidence_threshold,
            class_thresholds=config.behavior_class_thresholds,
            smooth_window=config.smooth_window,
            embedding_queue=self.embedding_queue,
            speed_axis=(config.freeze_threshold is not None and config.dart_threshold is not None),
            cm_s_per_px_frame=config.cm_s_per_px_frame,
        )
        self.display = DisplayConsumer(
            display_queue=self.display_queue,
            latest_label=self.latest_label,
            stop_event=self.stop_event,
            keypoint_names=config.keypoint_names,
            edges=config.edges,
            display_window=config.display,
            video_writer_path=(Path(config.output_video) if config.output_video else None),
            fps=config.fps_override or 30.0,
            vocab_order=self.model.classes,
        )

    def _build_sequence_threads(self, config) -> None:
        """Thread set for a CNN SequenceModel: producer → tracker →
        SequenceClassifier → display (no feature engine)."""
        self.producer = VideoProducer(
            source=config.source,
            raw_queue=self.raw_queue,
            stop_event=self.stop_event,
            realtime=config.realtime,
            fps_override=config.fps_override,
        )
        self.tracker = _make_tracker(
            config, self.raw_queue, self.tracked_queue, self.display_queue, self.stop_event
        )
        self.seq_classifier = SequenceClassifier(
            tracked_queue=self.tracked_queue,
            latest_label=self.latest_label,
            stop_event=self.stop_event,
            model=self.model,
            predict_every=config.predict_every,
            smooth_window=config.smooth_window,
            ethogram_path=(Path(config.ethogram_csv) if config.ethogram_csv else None),
            gate=config.move_gate,
        )
        self.display = DisplayConsumer(
            display_queue=self.display_queue,
            latest_label=self.latest_label,
            stop_event=self.stop_event,
            keypoint_names=config.keypoint_names,
            edges=config.edges,
            display_window=config.display,
            video_writer_path=(Path(config.output_video) if config.output_video else None),
            fps=config.fps_override or 30.0,
            vocab_order=list(self.model.classes),
        )

    def run(self) -> None:
        """Start all threads and block until the stream ends.

        Press ``q`` / ``Esc`` in the display window to stop early. On
        non-display runs we run until the producer hits end-of-file or
        the user hits ``Ctrl+C``.
        """
        # Probe cv2.imshow up front when display was requested. Without
        # this, a missing GUI backend (e.g. opencv-python-headless
        # shadowing opencv-python) crashes inside the DisplayConsumer
        # *after* threads have started, with a cryptic
        # "rebuild the library with Windows / GTK+ / Cocoa support"
        # message. Catch it here and surface the fix.
        if self.config.display:
            ok, reason = _probe_display()
            if not ok:
                raise RuntimeError(
                    "OpenCV's GUI backend is not available — cv2.imshow "
                    "would crash mid-stream.\n\n"
                    "Most common cause: opencv-python-headless is installed "
                    "(or is shadowing opencv-python).\n"
                    "Fix:\n"
                    "  pip uninstall -y opencv-python-headless opencv-python\n"
                    "  pip install opencv-python\n\n"
                    "Or run with --no-display + --output / --ethogram to "
                    "produce files headlessly.\n\n"
                    f"Underlying cv2 error: {reason}"
                )
        if self.is_sequence:
            threads = [
                self.producer,
                self.tracker,
                self.seq_classifier,
                self.display,
            ]
        else:
            threads = [
                self.producer,
                self.tracker,
                self.feature_engine,
                self.classifier,
                self.display,
            ]
        for t in threads:
            t.start()

        try:
            # Wait for the display thread first because pressing Q in
            # its window is what signals the user-initiated stop. If
            # display is off, wait for producer instead.
            anchor = self.display if self.config.display else self.producer
            while anchor.is_alive():
                anchor.join(timeout=0.2)
                if self.stop_event.is_set():
                    break
        except KeyboardInterrupt:
            self.stop_event.set()
        finally:
            self.stop_event.set()
            for t in threads:
                t.join(timeout=2.0)
            # Surface any per-thread errors so the user knows what
            # broke instead of staring at a silent exit.
            for t in (self.producer, self.tracker):
                err = getattr(t, "error", None)
                if err:
                    print(f"[{t.name}] error: {err}")

    def stop(self) -> None:
        """Signal all threads to exit cleanly. Safe to call from any thread."""
        self.stop_event.set()


# ---------------------------------------------------------------------------
# cv2 GUI capability probe
# ---------------------------------------------------------------------------


def _probe_display() -> tuple[bool, str]:
    """Return ``(ok, reason)`` describing whether ``cv2.imshow`` works.

    Creates and immediately destroys a 10×10 throwaway window. If that
    raises (typical signature: opencv-python-headless installed without
    a real GUI backend), we catch it and return the message so callers
    can surface a friendly error before they've spawned any threads.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        return False, str(e)
    name = "__glider_display_probe__"
    try:
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(name, np.zeros((10, 10, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow(name)
        return True, ""
    except Exception as e:  # noqa: BLE001 - cv2.error or anything else
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return False, str(e)
