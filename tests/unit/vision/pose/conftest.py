"""Shared pytest fixtures.

Synthetic PoseData with smooth, noisy, and gappy variants — enough to exercise
all converters and filters without requiring a real model or video.
"""

from __future__ import annotations

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
