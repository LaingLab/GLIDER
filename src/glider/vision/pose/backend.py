"""Pose inference backends behind one small protocol.

Every backend turns one BGR frame into ``(K, 2)`` xy in source-frame pixels plus
``(K,)`` confidence in [0, 1], with undetected keypoints as NaN xy and zero
confidence. That is exactly the contract
:func:`glider.analysis.behavior.classify.pose_extract.extract_keypoints` already
returns, which is why nothing downstream — overlays, the streaming feature
extractor, the DeepLabCut CSV writer — needs to change.

Both concrete backends take their heavy object by injection (a YOLO net, an
onnxruntime session) rather than constructing it inline. Loading is factored
into the module-level ``_load_yolo`` / ``_make_session`` seams, which keeps the
whole surface testable with lightweight doubles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from glider.vision.pose.decode import decode_dlc_locref, decode_sleap_confmaps
from glider.vision.pose.spec import (
    PoseModelError,
    PoseModelSpec,
    ensure_onnxruntime,
    identify_pose_model,
)


@runtime_checkable
class PoseBackend(Protocol):
    """One BGR frame in, keypoints out."""

    keypoint_names: list[str]

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(K, 2)`` xy in source-frame pixels, ``(K,)`` confidence in [0, 1]."""
        ...

    def close(self) -> None: ...


class UltralyticsBackend:
    """Today's YOLO path, wearing the protocol.

    A deliberate pass-through: it calls ultralytics exactly as the old code did
    and reuses ``extract_keypoints`` verbatim, so results are unchanged. That is
    what makes the existing offline behaviour suite a valid regression guard for
    this refactor.
    """

    def __init__(self, yolo, keypoint_names, conf_threshold: float = 0.25, device=None):
        self.yolo = yolo
        self.keypoint_names = list(keypoint_names)
        self.conf_threshold = float(conf_threshold)
        # Forwarded to ultralytics only when set, so the kwargs match what the
        # offline PoseTracker sent before this backend existed.
        self.device = device

    @property
    def native_keypoint_count(self) -> int | None:
        """Keypoints the YOLO head emits, or ``None`` when it cannot be read.

        Ultralytics records this as ``model.kpt_shape``. It is the only check
        that can catch an operator typing the wrong number of names, since the
        checkpoint carries no body-part names to compare against.
        """
        shape = getattr(getattr(self.yolo, "model", None), "kpt_shape", None)
        return int(shape[0]) if shape else None

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Imported here, not at module scope: glider.analysis.behavior.classify
        # imports back into glider.vision.pose, so a top-level import makes the
        # two packages circular depending on which one is entered first.
        from glider.analysis.behavior.classify.pose_extract import extract_keypoints

        kwargs = {"conf": self.conf_threshold, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.yolo.predict(bgr, **kwargs)
        return extract_keypoints(
            results[0] if results else None,
            self.conf_threshold,
            len(self.keypoint_names),
        )

    def close(self) -> None:
        self.yolo = None


def preprocess_frame(bgr: np.ndarray, spec: PoseModelSpec) -> tuple[np.ndarray, float]:
    """Turn a BGR frame into the model's input tensor.

    Returns the tensor and the scale factor applied, which the caller divides
    out of the decoded coordinates to get back to source-frame pixels.

    Padding goes to the bottom and right only, so it never shifts the coordinate
    origin and needs no un-mapping on the way back.
    """
    import cv2

    if spec.color_mode == "gray":
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[:, :, None]
    else:
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    scale = float(spec.scale)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = img[:, :, None]

    pad = max(int(spec.pad_to_stride), 1)
    if pad > 1:
        h, w = img.shape[:2]
        pad_h = (-h) % pad
        pad_w = (-w) % pad
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    # Normalisation runs while the array is still HWC, so the channel slice
    # below indexes the right axis for both colour modes.
    tensor = img.astype(np.float32)
    if spec.divide_by_255:
        tensor /= 255.0
    if spec.mean is not None:
        tensor -= np.asarray(spec.mean, dtype=np.float32)[: tensor.shape[2]]
    if spec.std is not None:
        tensor /= np.asarray(spec.std, dtype=np.float32)[: tensor.shape[2]]

    if spec.input_layout == "NCHW":
        tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor[None, ...]), scale


class OnnxPoseBackend:
    """Run a DeepLabCut or SLEAP model through onnxruntime.

    One class serves both families: the differences live in the *spec*
    (preprocessing, stride, normalisation) and in which decoder gets selected —
    a location-refinement stdev means DeepLabCut, its absence means SLEAP.

    ``session`` is injected rather than constructed here, so the whole class is
    testable with a stub returning canned tensors and CI never needs
    onnxruntime installed.
    """

    def __init__(self, session, spec: PoseModelSpec):
        self.session = session
        self.spec = spec
        self.keypoint_names = list(spec.keypoint_names)

    @property
    def native_keypoint_count(self) -> int:
        """Keypoints the model emits, per its own config. Always known here."""
        return len(self.spec.keypoint_names)

    def _as_kchw(self, arr: np.ndarray) -> np.ndarray:
        """Drop the batch axis and put channels first.

        Output layout is assumed to match input layout, which holds for the
        exporters in play: torch graphs are NCHW throughout, TensorFlow graphs
        NHWC throughout.
        """
        arr = np.asarray(arr)
        if arr.ndim == 4:
            arr = arr[0]
        if self.spec.input_layout == "NHWC":
            arr = np.transpose(arr, (2, 0, 1))
        return arr

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, scale = preprocess_frame(bgr, self.spec)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: tensor})

        maps = self._as_kchw(outputs[0])
        k = len(self.keypoint_names)
        if maps.shape[0] != k:
            raise ValueError(
                f"model emitted {maps.shape[0]} keypoint channels but the spec "
                f"names {k} keypoints — check the model against its "
                "glider_pose.json."
            )

        if self.spec.locref_stdev is not None:
            if len(outputs) < 2:
                raise ValueError(
                    "spec declares locref_stdev, so a DeepLabCut "
                    "location-refinement output was expected, but the model "
                    f"returned {len(outputs)} output(s). Re-export with the "
                    "locref head, or clear locref_stdev in glider_pose.json."
                )
            xy, conf = decode_dlc_locref(
                maps,
                self._as_kchw(outputs[1]),
                stride=self.spec.output_stride,
                locref_stdev=self.spec.locref_stdev,
                apply_sigmoid=self.spec.apply_sigmoid,
            )
        else:
            xy, conf = decode_sleap_confmaps(
                maps,
                stride=self.spec.output_stride,
                window=self.spec.refine_window,
                apply_sigmoid=self.spec.apply_sigmoid,
            )

        if scale != 1.0:
            xy = xy / scale
        return xy, conf

    def close(self) -> None:
        self.session = None


def _load_yolo(path):
    """Load an ultralytics YOLO net. Isolated so tests can double it."""
    from ultralytics import YOLO

    return YOLO(str(path))


def _make_session(spec: PoseModelSpec):
    """Build an onnxruntime session for *spec*. Isolated so tests can double it."""
    ensure_onnxruntime()
    import onnxruntime as ort

    return ort.InferenceSession(str(spec.model_path), providers=["CPUExecutionProvider"])


def load_pose_backend(
    path_or_spec,
    keypoint_names: list[str] | None = None,
    conf_threshold: float = 0.25,
    device=None,
) -> PoseBackend:
    """Build the right backend for a model path (or an already-built spec).

    When the model knows its own keypoint names — DeepLabCut and SLEAP always
    do — those win over anything the caller passes, because the model's training
    order is the only order that is correct. YOLO checkpoints do not carry
    body-part names, so there the caller must supply them.

    ``device`` applies to the ultralytics path only. The ONNX backend is
    CPU-only by design; GPU execution providers are out of scope.
    """
    spec = (
        path_or_spec
        if isinstance(path_or_spec, PoseModelSpec)
        else identify_pose_model(path_or_spec)
    )

    if spec.kind == "yolo":
        if not keypoint_names:
            raise PoseModelError(
                f"{Path(spec.model_path).name} is a YOLO checkpoint, which does "
                "not record body-part names. Enter the keypoint names in the "
                "model's training order."
            )
        return UltralyticsBackend(
            _load_yolo(spec.model_path), keypoint_names, conf_threshold, device
        )

    return OnnxPoseBackend(_make_session(spec), spec)
