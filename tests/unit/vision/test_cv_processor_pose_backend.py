"""``POSE_MODEL`` as a detection backend.

A SLEAP/DeepLabCut model previously had nowhere to plug into tracking:
``DetectionBackend`` had no pose member, so the operator's model was consulted
by the behaviour classifier only, and tracking fell back to whatever backend
was configured (background subtraction, by default). See
docs/superpowers/specs/2026-08-25-pose-tracking-backend-design.md.

These tests use a fake backend returning known keypoints, per that spec's
testing section — no real SLEAP/DLC model is loaded here.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.cv_processor import (
    _MODEL_BACKED_BACKENDS,
    POSE_BBOX_MARGIN,
    POSE_BBOX_MIN_MARGIN_PX,
    CVProcessor,
    CVSettings,
    DetectionBackend,
)
from glider.vision.pose.spec import PoseModelError


class _FakePoseBackend:
    """A ``PoseBackend`` double returning fixed keypoints for every frame."""

    def __init__(self, xy, conf, keypoint_names=None):
        self._xy = np.asarray(xy, dtype=np.float64)
        self._conf = np.asarray(conf, dtype=np.float64)
        self.keypoint_names = list(keypoint_names or [f"kp{i}" for i in range(len(self._conf))])

    def predict(self, bgr):
        return self._xy, self._conf

    def close(self):
        pass


def _proc_with_backend(backend: _FakePoseBackend, **settings_kwargs) -> CVProcessor:
    """A CVProcessor wired directly to *backend*, bypassing real model I/O."""
    settings = CVSettings(backend=DetectionBackend.POSE_MODEL, **settings_kwargs)
    proc = CVProcessor(settings)
    proc._active_backend = DetectionBackend.POSE_MODEL
    proc._pose_backend = backend
    return proc


def test_pose_model_is_in_model_backed_backends():
    """Changing model_path on POSE_MODEL must force a reload, like YOLO."""
    assert DetectionBackend.POSE_MODEL in _MODEL_BACKED_BACKENDS


def test_initialize_builds_a_pose_backend_via_load_pose_backend(monkeypatch):
    """``initialize()`` loads the configured model through load_pose_backend."""
    import glider.vision.pose.backend as pose_backend_mod

    fake = _FakePoseBackend(xy=[[0, 0], [1, 1]], conf=[0.9, 0.9])
    captured = {}

    def _load(path, keypoint_names=None, conf_threshold=0.25, device=None):
        captured["path"] = path
        captured["conf_threshold"] = conf_threshold
        return fake

    monkeypatch.setattr(pose_backend_mod, "load_pose_backend", _load)

    settings = CVSettings(
        backend=DetectionBackend.POSE_MODEL,
        model_path="/models/mouse/model.onnx",
        keypoint_min_confidence=0.42,
        tracking_enabled=False,
    )
    proc = CVProcessor(settings)

    assert proc.initialize() is True
    assert captured["path"] == "/models/mouse/model.onnx"
    assert captured["conf_threshold"] == 0.42
    assert proc._pose_backend is fake
    assert proc.active_backend == DetectionBackend.POSE_MODEL
    assert proc.degradation is None


def test_initialize_raises_rather_than_degrading_on_a_bad_pose_model(monkeypatch):
    """A pose model that cannot load must raise, not fall back silently.

    This is the whole point of the spec: a quiet fallback to background
    subtraction produced plausible-looking results from the wrong algorithm,
    which is exactly the bug being fixed. Contrast with a bad YOLO checkpoint,
    which _load_yolo_model degrades from -- deliberately not mirrored here.
    """
    import glider.vision.pose.backend as pose_backend_mod

    def _boom(path, keypoint_names=None, conf_threshold=0.25, device=None):
        raise PoseModelError("model.onnx has no glider_pose.json beside it")

    monkeypatch.setattr(pose_backend_mod, "load_pose_backend", _boom)

    settings = CVSettings(
        backend=DetectionBackend.POSE_MODEL,
        model_path="/models/mouse/model.onnx",
        tracking_enabled=False,
    )
    proc = CVProcessor(settings)

    with pytest.raises(PoseModelError):
        proc.initialize()

    # Not degraded: the requested backend is still what's recorded as active,
    # and no BackendDegradation was fabricated for it.
    assert proc.active_backend == DetectionBackend.POSE_MODEL
    assert proc.degradation is None
    assert proc.is_initialized is False


def test_detect_emits_one_detection_with_padded_bbox_and_mean_confidence():
    backend = _FakePoseBackend(
        xy=[[10, 10], [50, 10], [10, 50], [50, 50]],
        conf=[0.9, 0.8, 0.7, 0.6],
    )
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    detections = proc._detect(frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.class_name == "animal"
    # extent (10,10)-(50,50) -> span 40, margin = max(40*0.10, 4) = 4
    margin = max(40 * POSE_BBOX_MARGIN, POSE_BBOX_MIN_MARGIN_PX)
    assert margin == 4.0
    assert det.bbox == (6, 6, 48, 48)
    assert det.confidence == pytest.approx((0.9 + 0.8 + 0.7 + 0.6) / 4)


def test_margin_floors_at_min_pixels_for_a_tight_cluster():
    """10% of a small span would be under the 4px floor -- the floor wins."""
    backend = _FakePoseBackend(xy=[[10, 10], [12, 10]], conf=[0.9, 0.9])
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    det = proc._detect(frame)[0]

    assert det.bbox == (6, 6, 10, 8)


def test_fewer_than_two_confident_keypoints_emits_no_detection():
    backend = _FakePoseBackend(xy=[[10, 10], [50, 10], [90, 90]], conf=[0.9, 0.1, 0.1])
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    assert proc._detect(frame) == []


def test_a_single_confident_keypoint_emits_no_detection():
    """One point has no extent to box -- the floor of two, not zero."""
    backend = _FakePoseBackend(xy=[[25, 25]], conf=[0.99])
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    assert proc._detect(frame) == []


def test_zero_confident_keypoints_emits_no_detection():
    backend = _FakePoseBackend(xy=[[25, 25], [30, 30]], conf=[0.0, 0.0])
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    assert proc._detect(frame) == []


def test_keypoints_attribute_carries_every_keypoint_not_only_confident_ones():
    backend = _FakePoseBackend(
        xy=[[10, 10], [50, 50], [99, 99]],
        conf=[0.9, 0.8, 0.05],
        keypoint_names=["nose", "tail", "whisker"],
    )
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    det = proc._detect(frame)[0]

    kps = det._keypoints
    assert kps.shape == (3, 3)
    # The low-confidence third keypoint is present, unfiltered.
    assert kps[2, 0] == pytest.approx(99.0)
    assert kps[2, 1] == pytest.approx(99.0)
    assert kps[2, 2] == pytest.approx(0.05)


def test_bbox_is_clipped_to_the_frame_near_an_edge():
    backend = _FakePoseBackend(xy=[[5, 5], [95, 95]], conf=[0.9, 0.9])
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)  # h=100, w=100

    det = proc._detect(frame)[0]

    # span=90, margin=9 -> would extend to (-4,-4)-(104,104); clipped to frame.
    assert det.bbox == (0, 0, 100, 100)


def test_dispatch_does_not_run_pose_detection_for_other_backends():
    """Sanity check that the new branch is gated on POSE_MODEL, not always run."""
    backend = _FakePoseBackend(xy=[[10, 10], [50, 50]], conf=[0.9, 0.9])
    proc = _proc_with_backend(backend, keypoint_min_confidence=0.3)
    proc._active_backend = DetectionBackend.MOTION_ONLY

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert proc._detect(frame) == []
