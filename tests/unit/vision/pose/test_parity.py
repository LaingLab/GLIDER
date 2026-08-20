"""Decode parity against real DeepLabCut / SLEAP output.

**This is the test that decides whether the decode is actually correct.**

Every other test in this directory proves the maths is *self-consistent* with
the conventions written down in ``decode.py``. None of them prove those
conventions match what a real DeepLabCut or SLEAP export emits. A wrong stride,
a missing sigmoid, or the wrong sub-pixel refinement produces a
plausible-looking skeleton that is quietly offset by a constant — which
corrupts every downstream behaviour feature instead of failing loudly.

Only this file closes that gap, and it cannot run in CI because it needs real
model weights. Until it has run green, "the decode is correct" is an assertion,
not a verified fact.

Fixture layout
--------------
Point ``GLIDER_POSE_FIXTURES`` at a directory holding one subdirectory per
format::

    $GLIDER_POSE_FIXTURES/
        dlc/
            model/              <- exported folder: model.onnx + glider_pose.json
            clip.mp4            <- the video both tools ran on
            reference.csv       <- what DeepLabCut itself produced for clip.mp4
        sleap/
            model/
            clip.mp4
            reference.csv

``reference.csv`` is a standard DeepLabCut-format CSV (the 3-row header), which
is what ``deeplabcut.analyze_videos`` writes and what SLEAP can be exported to.

Run with::

    GLIDER_POSE_FIXTURES=/path/to/fixtures pytest -m pose_parity tests/unit/vision/pose/test_parity.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.pose_parity

#: Mean absolute error, in pixels, tolerated per keypoint. Deliberately loose
#: to start: tighten it once real numbers exist rather than inventing a
#: threshold that has never been measured.
DEFAULT_TOLERANCE_PX = 2.0

FIXTURE_ENV = "GLIDER_POSE_FIXTURES"


def _fixture_root() -> Path:
    root = os.environ.get(FIXTURE_ENV)
    if not root:
        pytest.skip(
            f"{FIXTURE_ENV} is not set. This test needs a real model plus the "
            "reference CSV that DeepLabCut/SLEAP produced for the same clip; "
            "see this module's docstring for the expected layout."
        )
    path = Path(root)
    if not path.is_dir():
        pytest.skip(f"{FIXTURE_ENV}={root} is not a directory")
    return path


def _case(kind: str):
    root = _fixture_root() / kind
    if not root.is_dir():
        pytest.skip(f"no {kind} fixtures under {root}")

    model = root / "model"
    clip = next(iter(sorted(root.glob("clip.*"))), None)
    reference = root / "reference.csv"

    missing = [str(p) for p in (model, clip, reference) if p is None or not Path(p).exists()]
    if missing:
        pytest.skip(f"incomplete {kind} fixture; missing: {missing}")
    return model, clip, reference


def _worst_offender(ours: np.ndarray, theirs: np.ndarray, names: list[str]):
    """(name, frame, error) for the single worst keypoint-frame pair.

    Reported instead of a bare aggregate because one systematically-offset body
    part is exactly the failure this test hunts, and a mean over all keypoints
    hides it behind the ones that are fine.
    """
    err = np.linalg.norm(ours - theirs, axis=2)  # (frames, keypoints)
    err = np.where(np.isfinite(err), err, -np.inf)
    frame, kp = np.unravel_index(np.argmax(err), err.shape)
    return names[kp], int(frame), float(err[frame, kp])


def _compare(kind: str, tolerance: float = DEFAULT_TOLERANCE_PX):
    from glider.vision.pose.core import infer_video
    from glider.vision.pose.dlc import from_dlc_csv

    model, clip, reference = _case(kind)

    ours = infer_video(model, clip, progress=False)
    theirs = from_dlc_csv(reference)

    assert ours.keypoint_names == theirs.keypoint_names, (
        "keypoint names disagree — the comparison would be meaningless.\n"
        f"  ours:   {ours.keypoint_names}\n  theirs: {theirs.keypoint_names}"
    )
    n = min(ours.n_frames, theirs.n_frames)
    assert n > 0, "no overlapping frames between our run and the reference"

    a = ours.xy[:n]
    b = theirs.xy[:n]
    both = np.isfinite(a).all(axis=2) & np.isfinite(b).all(axis=2)
    assert both.any(), "no frame had a detection in both runs"

    err = np.linalg.norm(a - b, axis=2)[both]
    mae = float(err.mean())

    name, frame, worst = _worst_offender(a, b, ours.keypoint_names)
    assert mae <= tolerance, (
        f"{kind} decode disagrees with the reference by {mae:.2f} px mean "
        f"absolute error (tolerance {tolerance} px).\n"
        f"  worst: keypoint {name!r} on frame {frame}, off by {worst:.2f} px\n"
        f"  compared {int(both.sum())} keypoint-frames over {n} frames.\n"
        "A roughly constant offset points at the stride/half-cell convention "
        "in decode.py; a scale factor points at spec.scale or output_stride."
    )


def test_dlc_decode_matches_deeplabcut():
    _compare("dlc")


def test_sleap_decode_matches_sleap():
    _compare("sleap")
