"""Shared fixtures for vision unit tests: synthesize a tiny deterministic clip."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

CLIP_FRAMES = 12
CLIP_W = 64
CLIP_H = 48
CLIP_FPS = 10.0


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """A short MJPG/AVI clip: a white square sliding left-to-right on black.

    MJPG in an .avi container gives exact, portable frame counts across
    OpenCV builds. Skips the test if this build cannot open the writer.
    """
    path = tmp_path / "clip.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, CLIP_FPS, (CLIP_W, CLIP_H))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot open an MJPG writer")
    for i in range(CLIP_FRAMES):
        frame = np.zeros((CLIP_H, CLIP_W, 3), dtype=np.uint8)
        x = 4 + i * 4
        cv2.rectangle(frame, (x, 18), (x + 8, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def _distinct_kps(k: int) -> np.ndarray:
    """Non-coincident ``(k, 2)`` keypoints (no zero body length / NaN)."""
    return np.stack([np.linspace(0.0, 10.0, k), np.linspace(0.0, 5.0, k)], axis=1)


@pytest.fixture
def tiny_behavior_model():
    """A fitted RandomForest-backed BehaviorModel with consistent columns.

    Builds a *tiny but internally consistent*
    :class:`~glider.analysis.behavior.model.BehaviorModel`. The tricky part is
    feature-name alignment: the model's ``feature_names`` must match, byte-for-
    byte, the windowed column names that ``SlidingFeatureBuffer.rolling_dict()``
    produces when fed the same per-frame feature dicts the
    ``StreamingFeatureExtractor`` emits. We build the model by running the exact
    live-path plumbing once and reading the column names back out, so
    ``predict_one(pd.Series(rolling_dict()))`` reindexes cleanly onto the
    trained columns.
    """
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer
    from glider.analysis.behavior.classify.features_stream import StreamingFeatureExtractor
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.model import BehaviorModel

    names = ["k0", "k1", "k2", "k3"]
    spec = FeatureSpec(body_axis=(0, 3))
    window = 5
    stats = ("mean", "std", "max")

    # 1) Run the extractor to get a real per-frame feature dict.
    ext = StreamingFeatureExtractor(spec, names, 30.0)
    per_frame_dict = None
    for _ in range(6):
        row = ext.push(_distinct_kps(len(names)))
        if row is not None:
            per_frame_dict = row
    assert per_frame_dict is not None, "extractor never emitted a per-frame row"
    per_frame_names = list(per_frame_dict.keys())

    # 2) Feed the buffer to read back the exact windowed feature names.
    buf = SlidingFeatureBuffer(per_frame_names, window, stats, [])
    for _ in range(window):
        buf.push_features(per_frame_dict)
    cols = list(buf.rolling_dict().keys())

    # 3) Fit a tiny classifier over random rows shaped like ``cols`` with
    #    both classes present.
    rng = np.random.default_rng(0)
    n = 12
    x = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
    y = np.array(["A", "B"] * (n // 2))
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(x, y)

    return BehaviorModel(
        classifier=rf,
        feature_names=cols,
        spec=spec,
        window=window,
        stats=stats,
        fps=30.0,
        classes=["A", "B"],
    )
