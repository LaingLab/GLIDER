"""Shared pytest fixtures.

Synthetic PoseData with smooth, noisy, and gappy variants — enough to exercise
all converters and filters without requiring a real model or video, plus the
Ultralytics stand-ins that let infer_video's streaming loop run with no torch.
"""

from __future__ import annotations

import sys
import types
from typing import TYPE_CHECKING

import numpy as np
import pytest

# NOTE: torch is preloaded in the root tests/conftest.py to avoid a Windows
# DLL-ordering conflict with PyQt6 (see comment there). It must happen before
# any Qt import, so the leaf conftest is too late to own that workaround.

# NOTE: PoseData is imported lazily inside the fixtures below (not at module top)
# because it isn't exported from glider.vision.pose until the core.py port (Task
# 1.5). A top-level runtime import here would fail conftest collection for the
# whole vision/pose/ test dir before then. The TYPE_CHECKING guard keeps the
# `-> PoseData` return annotations resolvable to ruff (F821) without importing
# at runtime; `from __future__ import annotations` keeps them as lazy strings.
if TYPE_CHECKING:
    from glider.vision.pose import PoseData

KEYPOINTS = ["snout", "left_ear", "right_ear", "neck", "tail_base"]


@pytest.fixture
def kpt_names() -> list[str]:
    return list(KEYPOINTS)


@pytest.fixture
def synthetic_pose(kpt_names) -> PoseData:
    """Smooth synthetic mouse: each keypoint follows a translated sinusoid."""
    from glider.vision.pose import PoseData

    rng = np.random.default_rng(42)
    n_frames = 200
    n_kpts = len(kpt_names)

    t = np.linspace(0, 4 * np.pi, n_frames)
    base_x = 320 + 40 * np.sin(t)
    base_y = 240 + 30 * np.cos(t)

    xy = np.empty((n_frames, n_kpts, 2))
    offsets = np.array(
        [
            [0, -20],  # snout
            [-10, -10],  # left_ear
            [10, -10],  # right_ear
            [0, 0],  # neck
            [0, 40],  # tail_base
        ],
        dtype=float,
    )
    for k in range(n_kpts):
        xy[:, k, 0] = base_x + offsets[k, 0] + rng.normal(0, 0.5, size=n_frames)
        xy[:, k, 1] = base_y + offsets[k, 1] + rng.normal(0, 0.5, size=n_frames)

    confidence = np.full((n_frames, n_kpts), 0.95)
    return PoseData(xy=xy, confidence=confidence, keypoint_names=kpt_names, fps=30.0)


@pytest.fixture
def gappy_pose(synthetic_pose) -> PoseData:
    """Same as synthetic but with low-confidence dropouts on body part 0."""
    pose = synthetic_pose.copy()
    pose.confidence[10:14, 0] = 0.1  # 4-frame gap
    pose.confidence[50:60, 0] = 0.2  # 10-frame gap (longer than default max_gap)
    return pose


# ---------------------------------------------------------------------------
# Ultralytics stand-ins
#
# infer_video imports YOLO inside the function body, which is what lets a fake
# module in sys.modules drive the whole streaming loop with no torch, no GPU
# and no real video file. Shared here because the callback tests and the
# arena-aware candidate-selection tests need the same Results shape.
# ---------------------------------------------------------------------------


class FakeTensor:
    """Minimal stand-in for a torch tensor: supports indexing, .cpu(), .numpy()."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def __getitem__(self, index):
        return FakeTensor(self._array[index])

    @property
    def shape(self):
        return self._array.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class FakeKeypoints:
    """``.xy`` is ``(n_detections, n_keypoints, 2)``; ``.conf`` matches it.

    ``conf=None`` models a checkpoint that emits no keypoint confidences at
    all, which infer_video has to survive.
    """

    def __init__(self, xy, conf=None):
        self.xy = FakeTensor(np.asarray(xy, dtype=float))
        self.conf = None if conf is None else FakeTensor(np.asarray(conf, dtype=float))


class FakeBoxes:
    def __init__(self, conf):
        self.conf = FakeTensor(np.asarray(conf, dtype=float))


class FakeResult:
    """One frame of Ultralytics ``Results``.

    ``boxes_conf`` is one box confidence per detection, or None for a result
    carrying no boxes — the branch where infer_video falls back to detection 0.
    """

    def __init__(self, keypoints, *, boxes_conf=None, keypoint_conf=None):
        self.keypoints = FakeKeypoints(keypoints, keypoint_conf)
        self.boxes = None if boxes_conf is None else FakeBoxes(boxes_conf)


def fake_yolo_streaming(results):
    """A YOLO class whose ``predict`` streams ``results``, one per frame."""

    class _FakeYOLO:
        def __init__(self, path):
            self.path = path

        def to(self, device):
            return self

        def predict(self, **kwargs):
            yield from results

    return _FakeYOLO


@pytest.fixture
def stub_ultralytics(monkeypatch):
    """Install a fake ``ultralytics`` module and neutralize device/fps probing.

    The caller assigns ``.YOLO`` — different tests need different streams.
    """
    import glider.vision.pose.device as device_mod
    from glider.vision.pose import core

    module = types.ModuleType("ultralytics")
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(core, "_video_fps", lambda path: 30.0)
    monkeypatch.setattr(device_mod, "resolve_device", lambda d, require_gpu=False: "cpu")
    return module
