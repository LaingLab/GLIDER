"""Shared fixtures for the behavior-analysis test suite.

The :func:`hybrid_sessions` fixture writes a small two-session dataset with
two behaviors whose *kinematics* are separable — ``rest`` = near-still
keypoints, ``locomote`` = fast-moving keypoints — so the freeze/dart
:class:`~glider.analysis.behavior.prior.KinematicPrior` has real structure to
key off. Tasks 9-10 (the hybrid trainer + benchmark harness) consume it.

Kept deliberately small (a few hundred frames across several alternating
bouts) so it fits LightGBM after 30-frame windowing yet still runs in ~1 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glider.vision.pose.core import PoseData

_KEYPOINT_NAMES = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
# Egocentric keypoint offsets from the body centroid (px). Fixed across
# frames, so every keypoint shares the centroid's motion.
_OFFSETS = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)


def _make_hybrid_pose(
    *,
    seed: int,
    base_x: float,
    base_y: float,
    n_bouts: int = 8,
    bout_len: int = 60,
    fps: float = 30.0,
) -> tuple[PoseData, list[tuple[str, int, int]]]:
    """Build a pose that alternates ``rest`` and ``locomote`` bouts.

    ``rest`` bouts hold the body still (tiny jitter → near-zero speed);
    ``locomote`` bouts sweep it in a fast circle (large frame-to-frame
    displacement → high speed). Returns the pose plus the annotation zones
    ``[(behavior, start_frame, end_frame), ...]`` that cover it exactly.
    """
    rng = np.random.default_rng(seed)
    n_kpts = len(_KEYPOINT_NAMES)
    n_frames = n_bouts * bout_len
    xy = np.empty((n_frames, n_kpts, 2))
    zones: list[tuple[str, int, int]] = []
    for b in range(n_bouts):
        start = b * bout_len
        end = start + bout_len
        t = np.arange(bout_len)
        if b % 2 == 0:
            # rest — hold position, minimal jitter → speed ~ 0.
            cx = np.full(bout_len, base_x)
            cy = np.full(bout_len, base_y)
            jitter = 0.15
            zones.append(("rest", start, end))
        else:
            # locomote — fast circular sweep → large per-frame velocity.
            cx = base_x + 15.0 * np.sin(0.8 * t)
            cy = base_y + 15.0 * np.cos(0.8 * t)
            jitter = 1.0
            zones.append(("locomote", start, end))
        for k in range(n_kpts):
            xy[start:end, k, 0] = cx + _OFFSETS[k, 0] + rng.normal(0, jitter, bout_len)
            xy[start:end, k, 1] = cy + _OFFSETS[k, 1] + rng.normal(0, jitter, bout_len)
    confidence = np.full((n_frames, n_kpts), 0.95)
    pose = PoseData(xy=xy, confidence=confidence, keypoint_names=list(_KEYPOINT_NAMES), fps=fps)
    return pose, zones


def _write_session(
    tmp_path: Path, name: str, pose: PoseData, zones: list[tuple[str, int, int]]
) -> tuple[Path, Path]:
    """Write a DLC pose CSV + annotations CSV for one session; return the pair."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.vision.pose.dlc import to_dlc_csv

    pose_csv = tmp_path / f"{name}.csv"
    ann_csv = tmp_path / f"{name}_annotations.csv"
    to_dlc_csv(pose, pose_csv)
    store = AnnotationStore()
    for behavior, start, end in zones:
        store.add(BehaviorZone(behavior=behavior, start_frame=start, end_frame=end))
    store.save_csv(ann_csv)
    return pose_csv, ann_csv


@pytest.fixture
def hybrid_sessions(tmp_path):
    """A 2-session (rest / locomote) dataset for the hybrid model tests.

    Returns ``(sessions, tag_map)`` where ``sessions`` is the
    ``[(pose_csv, annotations_csv), ...]`` list :func:`train_model` /
    :func:`train_hybrid_model` consume, and ``tag_map`` binds each behavior to
    the semantic tag the freeze/dart prior keys off.
    """
    pose_a, zones_a = _make_hybrid_pose(seed=11, base_x=100.0, base_y=200.0)
    pose_b, zones_b = _make_hybrid_pose(seed=29, base_x=140.0, base_y=230.0)
    sessions = [
        _write_session(tmp_path, "session_a", pose_a, zones_a),
        _write_session(tmp_path, "session_b", pose_b, zones_b),
    ]
    tag_map = {
        "rest": frozenset({"stationary"}),
        "locomote": frozenset({"locomotory"}),
    }
    return sessions, tag_map
