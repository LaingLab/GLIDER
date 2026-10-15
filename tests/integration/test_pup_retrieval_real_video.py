"""A real retrieval video, end to end: batch pup pass, then Session Review scoring.

The repository is public and retrieval videos belong to the labs that recorded
them, so nothing about one is committed -- not the video, the models, the nest
outline nor its hand-checked latencies. Everything comes from the environment:

  GLIDER_PUP_VIDEO=/path/to/video.avi
  GLIDER_PUP_MODEL=/path/to/pups.pt
  GLIDER_POSE_CSV=/path/to/videoDLC_model.csv     (adult pose with a 'nose' keypoint)
  GLIDER_NEST=x,y;x,y;...                         (the nest outline, video pixels)
  GLIDER_PX_PER_MM=2.0
  GLIDER_EXPECTED_LATENCIES=s1,s2,...             (hand-checked, seconds from trial start)
  GLIDER_LITTER_SIZE=5                            (optional; default: one per latency)

    QT_QPA_PLATFORM=offscreen uv run --no-sync pytest \\
        tests/integration/test_pup_retrieval_real_video.py -q -m slow
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from glider.analysis.behavior.retrieval_session import score_session
from glider.analysis.behavior.retrieval_setup import RetrievalProtocol, RetrievalSetup
from glider.analysis.behavior.session_view import SessionView
from glider.vision.pose.pups import track_video_pups
from glider.vision.zones import Zone, ZoneShape

NEEDED = (
    "GLIDER_PUP_VIDEO",
    "GLIDER_PUP_MODEL",
    "GLIDER_POSE_CSV",
    "GLIDER_NEST",
    "GLIDER_PX_PER_MM",
    "GLIDER_EXPECTED_LATENCIES",
)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all(os.environ.get(k) for k in NEEDED),
        reason="set " + ", ".join(NEEDED) + " to run against a real retrieval video",
    ),
]

# Hand-checked latencies are written to 0.1 s, and a nose crossing a nest line
# traced by eye can land a frame either side; allow both, no more.
ROUNDING_S = 0.05


def test_latencies_match_a_hand_check(tmp_path):
    expected = sorted(float(s) for s in os.environ["GLIDER_EXPECTED_LATENCIES"].split(","))
    litter = int(os.environ.get("GLIDER_LITTER_SIZE", len(expected)))
    video = tmp_path / Path(os.environ["GLIDER_PUP_VIDEO"]).name
    shutil.copy(os.environ["GLIDER_PUP_VIDEO"], video)  # never write beside the original
    shutil.copy(os.environ["GLIDER_POSE_CSV"], tmp_path / f"{video.stem}DLC_adult.csv")
    track_video_pups(Path(os.environ["GLIDER_PUP_MODEL"]), video, litter_size=litter)

    view = SessionView.from_video(video)
    assert view.pups is not None, "the pup pass wrote no _pups.npz"
    assert "nose" in view.keypoint_names, f"no 'nose' keypoint in {view.keypoint_names}"
    w, h = view.resolution
    poly = [tuple(map(float, p.split(","))) for p in os.environ["GLIDER_NEST"].split(";")]
    setup = RetrievalSetup(trial_start_s=0.0, px_per_mm=float(os.environ["GLIDER_PX_PER_MM"]))
    setup.set_nest(0.0, Zone("n", "nest", ZoneShape.POLYGON, [(x / w, y / h) for x, y in poly]))
    nose = view.xy[:, view.keypoint_names.index("nose"), :]

    r = score_session(
        view.pups, nose, fps=view.fps, size=(w, h), setup=setup, protocol=RetrievalProtocol()
    )

    # Slot numbers differ between runs, so compare the sorted latencies.
    latencies = sorted(p.retrieval_latency_s for p in r.pups.values())
    retrieved = [t for t in latencies if np.isfinite(t)]
    assert retrieved == pytest.approx(expected, abs=ROUNDING_S + 1.0 / view.fps)
    assert all(p.carries >= 1 for p in r.pups.values() if np.isfinite(p.retrieval_latency_s))
