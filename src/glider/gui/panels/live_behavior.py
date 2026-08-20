"""Pure-logic live behavior classifier (no Qt, no threads).

This is the composable core the Runner's live-inference panel drives: it
wraps a YOLO pose net + a trained :class:`BehaviorModel` and turns one BGR
frame into a :class:`LiveResult` (``label`` + decoded ``keypoints``). It
reuses the exact shared streaming cores the offline/threaded pipeline uses
so live predictions are bit-for-bit identical to offline ones:

* :class:`~glider.vision.pose.backend.PoseBackend` — one BGR frame → ``(K, 2)``
  keypoints, whether the pose net is YOLO, DeepLabCut or SLEAP.
* :class:`StreamingFeatureExtractor` — centered per-frame kinematics.
* :class:`SlidingFeatureBuffer` — rolling-stat window → the model's row.
* :meth:`BehaviorModel.predict_one` — one label per full window.

Construction fails loudly for models the live path can't serve (CNN
sequence models, or tabular models needing ``motion_*`` / ``traj_*``
features) and for keypoint-count mismatches between the entered names, the
model, and the YOLO head.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PyQt6.QtCore import QObject, pyqtSignal

from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer
from glider.analysis.behavior.classify.features_stream import (
    StreamingFeatureExtractor,
    derive_stream_columns,
    expected_keypoint_order,
)
from glider.analysis.behavior.classify.pipeline import (
    _load_behavior_model,
    _unstreamable_feature_families,
)
from glider.analysis.behavior.sequence import SequenceModel

logger = logging.getLogger(__name__)


class UnsupportedModelError(Exception):
    """The model can't be served by the live path (sequence / unstreamable)."""


class KeypointMismatchError(Exception):
    """Entered keypoint names disagree with the model or YOLO head arity."""


@dataclass
class LiveResult:
    """One frame's classification: the behavior ``label`` + decoded keypoints."""

    label: str
    keypoints: np.ndarray


def model_keypoint_count(model) -> int:
    """Number of distinct ``speed_*`` base names in ``model.feature_names``.

    Each per-keypoint speed feature contributes exactly one base name
    (``speed_<kp>``) regardless of how many ``__{stat}`` windowed columns it
    fans out into. Deduping across those suffixes recovers ``K`` — the number
    of keypoints the model was trained on.
    """
    bases = {col.split("__")[0] for col in model.feature_names}
    return sum(1 for base in bases if base.startswith("speed_"))


class LiveBehaviorClassifier:
    """Compose YOLO + a trained BehaviorModel into ``classify_frame``.

    Parameters
    ----------
    model
        A loaded :class:`~glider.analysis.behavior.model.BehaviorModel`.
    yolo
        An ultralytics YOLO (or a double) exposing ``predict(bgr, conf=,
        verbose=)``.
    keypoint_names
        The keypoint names, in model order. Length must equal the model's
        keypoint count (and the YOLO head's, when readable).
    conf_threshold
        Per-keypoint confidence gate passed through to YOLO + keypoint decode.
    """

    def __init__(self, model, backend, conf_threshold: float = 0.25):
        if isinstance(model, SequenceModel):
            raise UnsupportedModelError(
                "CNN sequence models aren't supported by the live tabular "
                "classifier; use the sequence live path instead."
            )
        unstreamable = _unstreamable_feature_families(model.feature_names)
        if unstreamable:
            raise UnsupportedModelError(
                "this behavior model needs features the live path can't compute "
                f"in real time: {unstreamable}. Train a pose-only (or "
                "--freq-features) model for live inference."
            )

        # The backend owns the names: for DeepLabCut and SLEAP they come from
        # the model's own config, so accepting them separately would let two
        # sources of truth disagree about the order that matters.
        names = list(backend.keypoint_names)

        model_k = model_keypoint_count(model)
        entered_k = len(names)
        if entered_k != model_k:
            raise KeypointMismatchError(
                f"entered {entered_k} keypoint names but the model expects "
                f"{model_k} (distinct speed_* features)."
            )

        native_k = getattr(backend, "native_keypoint_count", None)
        if native_k is not None:
            if native_k != model_k:
                raise KeypointMismatchError(
                    f"the pose model emits {native_k} keypoints but the "
                    f"behavior model expects {model_k}."
                )
        else:
            warnings.warn(
                "could not read the pose model's keypoint count; proceeding on "
                "the entered-vs-model keypoint-count check only.",
                stacklevel=2,
            )

        # Names agreeing in *count* is not enough. With FeatureSpec.auto_angles
        # the angle columns are named after the keypoints at given indices, so a
        # re-ordering yields columns that never appear: they arrive as NaN,
        # every prediction comes back blank, and nothing raises.
        expected = expected_keypoint_order(model)
        if expected and names != expected:
            raise KeypointMismatchError(
                "keypoint names are in a different order than the behavior "
                f"model was trained on.\n  model:    {expected}\n"
                f"  provided: {names}"
            )

        self._model = model
        self.backend = backend
        self.keypoint_names = names
        self.conf_threshold = float(conf_threshold)

        per_frame, spectral = derive_stream_columns(model)
        self._ext = StreamingFeatureExtractor(model.spec, self.keypoint_names, model.fps)
        self._buf = SlidingFeatureBuffer(per_frame, model.window, model.stats, spectral)

        self.window = model.window
        self.warmup = self._ext.lag

    def classify_frame(self, bgr) -> LiveResult:
        """Classify one BGR frame; never resets on NaN / no-detection."""
        kps, _ = self.backend.predict(bgr)
        row = self._ext.push(kps)
        if row is not None:
            self._buf.push_features(row)
        if self._buf.is_full():
            label = self._model.predict_one(pd.Series(self._buf.rolling_dict()))
        else:
            label = ""
        return LiveResult(label, kps)

    def reset(self) -> None:
        """Clear the extractor history + rolling buffer (re-arm warm-up)."""
        self._ext.reset()
        self._buf.clear()


def _load_backend(model_path, keypoint_names, conf_threshold: float = 0.25):
    """Build a pose backend for ``model_path``.

    Isolated as a module-level function (with the heavy imports deferred inside
    :func:`~glider.vision.pose.backend.load_pose_backend`, so importing this
    module never requires torch or onnxruntime) so tests can monkeypatch it
    with a lightweight double.

    Accepts anything :func:`identify_pose_model` understands: a YOLO ``.pt``, or
    a DeepLabCut/SLEAP exported-model folder. ``keypoint_names`` is used only
    for the YOLO case; the other formats carry their own.
    """
    from glider.vision.pose.backend import load_pose_backend

    return load_pose_backend(model_path, keypoint_names, conf_threshold)


class BehaviorInferenceWorker(QObject):
    """Loads the models and runs :class:`LiveBehaviorClassifier` off the UI thread.

    Mirrors the ``CVWorker`` pattern in ``camera_panel``: a plain ``QObject``
    that is ``moveToThread``-ed onto a worker ``QThread``; slots are invoked via
    queued connections and results come back over signals. Model loading (YOLO +
    the behavior bundle) happens inside :meth:`initialize` so the (potentially
    slow, torch-importing) work never blocks the GUI thread.
    """

    ready = pyqtSignal()
    load_failed = pyqtSignal(str)
    result_ready = pyqtSignal(str, object)  # label, keypoints ndarray

    def __init__(self, conf_threshold: float = 0.25, parent=None):
        super().__init__(parent)
        self._conf_threshold = float(conf_threshold)
        self._classifier: LiveBehaviorClassifier | None = None
        # The model's ordered class vocabulary, published to the panel (for
        # stable label colors) once :meth:`initialize` succeeds.
        self.classes: list[str] = []

    def initialize(self, behavior_pkl: str, pose_model: str, keypoint_names: list[str]) -> None:
        """Load the models + build the classifier (runs on the worker thread).

        ``pose_model`` is a YOLO ``.pt`` or a DeepLabCut/SLEAP exported-model
        folder; ``keypoint_names`` applies to the YOLO case only, since the
        other formats carry their own names in training order.

        Emits :attr:`ready` on success or :attr:`load_failed` (with the error
        message) on ANY failure — unsupported/mismatched models, missing files,
        or a missing ``ultralytics``/``onnxruntime`` install. Never raises out
        of the slot.
        """
        try:
            backend = _load_backend(pose_model, keypoint_names, self._conf_threshold)
            model = _load_behavior_model(behavior_pkl)
            self._classifier = LiveBehaviorClassifier(model, backend, self._conf_threshold)
            self.classes = list(model.classes)
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            self._classifier = None
            self.classes = []
            self.load_failed.emit(str(exc))
            return
        self.ready.emit()

    def process_frame(self, frame_data) -> None:
        """Classify one frame and emit :attr:`result_ready` (label, keypoints).

        Ignores frames that arrive before :meth:`initialize` has succeeded.
        Like ``CVWorker``, each frame is classified immediately on the worker
        thread; because the classifier is stateful (rolling window) frames are
        processed in order rather than being dropped.
        """
        if self._classifier is None:
            return
        try:
            result = self._classifier.classify_frame(frame_data.frame)
        except Exception:
            logger.exception("Error in behavior inference worker")
            return
        self.result_ready.emit(result.label, result.keypoints)
