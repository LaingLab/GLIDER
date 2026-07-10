"""Live vs offline feature-fidelity parity.

The live path (:class:`~glider.gui.panels.live_behavior.LiveBehaviorClassifier`)
and the offline path
(:class:`~glider.analysis.behavior.classify.threads.FeatureEngine`) share the
same stateful cores: a :class:`StreamingFeatureExtractor` (centered-gradient
kinematics over *consecutive* frames) feeding a :class:`SlidingFeatureBuffer`
(rolling stats over consecutive frames). For live predictions to match the
trained/offline model, **every** camera frame must reach the extractor in
order — exactly as the offline ``PoseTracker`` pushes every frame.

This module pins that contract two ways:

* :func:`test_live_matches_offline_windowed_rows` feeds one identical 40-frame
  keypoint stream through BOTH paths and asserts the windowed feature rows are
  numerically equal at every matching frame.
* :func:`test_decimated_intake_diverges_from_offline` feeds only every 3rd
  frame into a live classifier (simulating the panel's old 3:1 intake
  decimation) and asserts the windowed features drift OUT of parity — this is
  the regression the ``CameraPanel`` fan-out bug would have caused.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Minimal stateful ultralytics doubles: one queued keypoint array per predict()
# ---------------------------------------------------------------------------


class _Torchy:
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


class _FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = _Torchy(np.asarray(xy)[None])
        self.conf = _Torchy(np.asarray(conf)[None])


class _FakeResult:
    def __init__(self, kp):
        self.keypoints = kp


class _SeqYolo:
    """Return queued keypoint arrays, one per ``predict()`` call.

    Confidences are all 1.0 so :func:`extract_keypoints` round-trips the xy
    unchanged (nothing NaN-masked), letting both paths see identical keypoints.
    """

    def __init__(self, stream, k):
        self._it = iter(stream)
        # Expose kpt_shape so LiveBehaviorClassifier's arity check passes
        # without a warning.
        self.model = type("M", (), {"kpt_shape": (k, 3)})()

    def predict(self, bgr, conf=0.25, verbose=False):  # noqa: ARG002
        kps = next(self._it)
        return [_FakeResult(_FakeKeypoints(kps, np.ones(len(kps))))]


# ---------------------------------------------------------------------------
# Keypoint stream + offline reference path
# ---------------------------------------------------------------------------


def _keypoint_stream(k: int, n_frames: int) -> list[np.ndarray]:
    """``n_frames`` distinct, non-coincident ``(k, 2)`` keypoint arrays.

    Each frame is a base spread of keypoints translated + jittered so that
    velocities/accelerations are non-trivial and vary frame to frame — the
    signal that exposes any intake decimation.
    """
    rng = np.random.default_rng(3)
    base = np.stack([np.linspace(0.0, 10.0, k), np.linspace(0.0, 5.0, k)], axis=1)
    stream = []
    for t in range(n_frames):
        shift = np.array([0.7 * t, 0.4 * t])
        jitter = rng.normal(0.0, 0.5, size=(k, 2))
        stream.append(base + shift + jitter)
    return stream


def _offline_rows(model, keypoint_names, stream):
    """Windowed rows from an extractor+buffer built exactly as FeatureEngine does.

    Returns the per-frame ``rolling_dict()`` snapshot after pushing each frame,
    mirroring the offline :class:`FeatureEngine`, which pushes *every* frame into
    the extractor and reads the buffer's rolling row.
    """
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer
    from glider.analysis.behavior.classify.features_stream import (
        StreamingFeatureExtractor,
        derive_stream_columns,
    )

    per_frame, spectral = derive_stream_columns(model)
    ext = StreamingFeatureExtractor(model.spec, keypoint_names, fps=30.0)
    buf = SlidingFeatureBuffer(per_frame, model.window, model.stats, spectral)

    rows = []
    for kps in stream:
        feats = ext.push(kps)
        if feats is not None:
            buf.push_features(feats)
        rows.append(buf.rolling_dict())
    return rows


def _assert_rows_equal(live_rows, offline_rows):
    assert len(live_rows) == len(offline_rows)
    for i, (lr, orr) in enumerate(zip(live_rows, offline_rows, strict=True)):
        assert lr.keys() == orr.keys(), f"column set diverged at frame {i}"
        lv = np.array([lr[c] for c in orr])
        ov = np.array([orr[c] for c in orr])
        assert np.allclose(
            lv, ov, rtol=1e-9, atol=1e-9, equal_nan=True
        ), f"windowed features differ at frame {i}"


# ---------------------------------------------------------------------------
# Parity: live (every frame) == offline (every frame)
# ---------------------------------------------------------------------------


def test_live_matches_offline_windowed_rows(tiny_behavior_model):
    from glider.gui.panels.live_behavior import LiveBehaviorClassifier, model_keypoint_count

    model = tiny_behavior_model
    k = model_keypoint_count(model)
    names = [f"k{i}" for i in range(k)]
    stream = _keypoint_stream(k, n_frames=40)

    offline = _offline_rows(model, names, stream)

    clf = LiveBehaviorClassifier(model, _SeqYolo(stream, k), names)
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    live = []
    for _ in stream:
        clf.classify_frame(bgr)  # advances the classifier's internal extractor+buffer
        live.append(clf._buf.rolling_dict())

    # A genuine (non-all-NaN) window is exercised, so the assertion has teeth.
    assert any(not all(np.isnan(v) for v in row.values()) for row in live)
    _assert_rows_equal(live, offline)


# ---------------------------------------------------------------------------
# Regression: decimating intake (the old 3:1 panel fan-out) breaks parity
# ---------------------------------------------------------------------------


def test_decimated_intake_diverges_from_offline(tiny_behavior_model):
    """Feeding only every 3rd frame to the live extractor (the panel's former
    intake decimation) inflates the centered-gradient kinematics and stretches
    the rolling window, so the windowed features no longer match offline — the
    exact corruption the CameraPanel fan-out fix prevents."""
    from glider.gui.panels.live_behavior import LiveBehaviorClassifier, model_keypoint_count

    model = tiny_behavior_model
    k = model_keypoint_count(model)
    names = [f"k{i}" for i in range(k)]
    stream = _keypoint_stream(k, n_frames=60)

    offline = _offline_rows(model, names, stream)

    # Only every 3rd frame reaches the classifier — the decimated intake.
    decimated = stream[::3]
    clf = LiveBehaviorClassifier(model, _SeqYolo(decimated, k), names)
    bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in decimated:
        clf.classify_frame(bgr)
    decimated_row = clf._buf.rolling_dict()

    # Compare against the offline row once the offline buffer is full (its last
    # frame). The decimated features must NOT match — dropping frames corrupts
    # the stateful kinematics.
    offline_full = offline[-1]
    cols = list(offline_full.keys())
    dec = np.array([decimated_row[c] for c in cols])
    off = np.array([offline_full[c] for c in cols])
    assert not np.allclose(
        dec, off, rtol=1e-3, atol=1e-3, equal_nan=True
    ), "decimated intake unexpectedly matched offline — the parity guard is toothless"
