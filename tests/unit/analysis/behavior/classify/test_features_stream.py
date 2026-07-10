"""Unit tests for the pure streaming feature extractor.

``StreamingFeatureExtractor`` is the non-threaded, non-Qt core lifted out of
``FeatureEngine`` so a future live classifier can reuse the EXACT same
per-frame feature math (live == offline). These tests pin two contracts:

  * warm-up + centered-middle-frame equivalence with ``compute_features``, and
  * the strip-suffix column derivation used to configure the live buffer.
"""

from __future__ import annotations

import numpy as np

from glider.analysis.behavior.classify.features_stream import (
    StreamingFeatureExtractor,
    derive_stream_columns,
)
from glider.analysis.behavior.classify.pose_extract import extract_keypoints
from glider.analysis.behavior.features import FeatureSpec, compute_features
from glider.vision.pose.core import PoseData


def test_extractor_warms_up_then_matches_centered_gradient():
    spec = FeatureSpec(body_axis=(0, 3))
    keypoint_names = ["nose", "neck", "body", "tailbase"]
    fps = 30.0
    extractor = StreamingFeatureExtractor(spec=spec, keypoint_names=keypoint_names, fps=fps)

    rng = np.random.default_rng(0)
    frames = [rng.random((4, 2)) for _ in range(5)]

    outputs = [extractor.push(f) for f in frames]

    # First four frames warm up the 5-frame history → no output yet.
    assert outputs[:4] == [None, None, None, None]
    assert isinstance(outputs[4], dict)

    # The 5th push must equal the MIDDLE row of the offline features
    # computed over the same 5-frame window (centered gradients).
    xy = np.stack(frames, axis=0)
    conf = np.where(np.isnan(xy).any(axis=-1), 0.0, 1.0)
    pose = PoseData(xy=xy, confidence=conf, keypoint_names=keypoint_names, fps=fps)
    df = compute_features(pose, spec=spec)
    expected = df.iloc[len(df) // 2].to_dict()

    got = outputs[4]
    assert got.keys() == expected.keys()
    for name, val in expected.items():
        np.testing.assert_allclose(got[name], val, equal_nan=True)


# ---------------------------------------------------------------------------
# extract_keypoints (shared YOLO -> keypoints decode)
# ---------------------------------------------------------------------------


class _Torchy:
    """Minimal stand-in for a torch tensor: indexing + .cpu().numpy() + .shape."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def __getitem__(self, i):
        return _Torchy(self._a[i])

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape


class FakeKeypoints:
    def __init__(self, xy, conf=None):
        self.xy = _Torchy(np.asarray(xy)[None]) if xy is not None else _Torchy(np.zeros((0, 0, 2)))
        self.conf = _Torchy(np.asarray(conf)[None]) if conf is not None else None


class FakeResult:
    def __init__(self, kp):
        self.keypoints = kp


def test_extract_keypoints_nan_masks_low_confidence():
    k = 4
    xy = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    conf = np.array([0.9, 0.1, 0.8, 0.7])  # index 1 below threshold 0.5

    kps, confs = extract_keypoints(FakeResult(FakeKeypoints(xy, conf)), 0.5, k)

    assert kps.shape == (k, 2)
    assert confs.shape == (k,)
    # The low-confidence keypoint row is NaN'd.
    assert np.isnan(kps[1]).all()
    # The rest survive verbatim.
    np.testing.assert_array_equal(kps[0], [1.0, 2.0])
    np.testing.assert_array_equal(kps[2], [5.0, 6.0])
    np.testing.assert_array_equal(kps[3], [7.0, 8.0])
    # Confidences are preserved (not masked) for the overlay fade.
    np.testing.assert_array_equal(confs, conf)


def test_extract_keypoints_none_result_all_nan():
    k = 4
    kps, confs = extract_keypoints(None, 0.5, k)
    assert kps.shape == (k, 2)
    assert confs.shape == (k,)
    assert np.isnan(kps).all()
    np.testing.assert_array_equal(confs, np.zeros(k))


def test_extract_keypoints_empty_detection_all_nan():
    k = 4
    kps, confs = extract_keypoints(FakeResult(FakeKeypoints(None)), 0.5, k)
    assert kps.shape == (k, 2)
    assert confs.shape == (k,)
    assert np.isnan(kps).all()
    np.testing.assert_array_equal(confs, np.zeros(k))


def test_derive_stream_columns_strips_suffixes():
    class _StubModel:
        feature_names = [
            "dist_a_b__mean",
            "dist_a_b__std",
            "speed_nose__mean",
            "speed_nose__domfreq",
        ]
        stats = ("mean", "std", "max")

    per_frame, spectral = derive_stream_columns(_StubModel())

    assert "dist_a_b" in per_frame
    assert "speed_nose" in per_frame
    assert spectral == ["speed_nose"]
