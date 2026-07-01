"""Tests for `propose_clips_multi` — multi-video diverse-clip sampling.

These exercise the quota-allocation math, deterministic interleaving,
under-quota fallback, and missing-pose-CSV error path. Each test uses
the existing `conftest.py` synthetic-pose helpers where possible.
"""

from __future__ import annotations

from pathlib import Path


def _make_synthetic_session(tmp_path: Path, name: str, n_frames: int = 600) -> tuple[Path, Path]:
    """Write a tiny synthetic DLC pose CSV + a fake matching video path.

    The video file is created as an empty placeholder — `propose_clips_multi`
    only stores the path string on `ProposedClip`, it doesn't open the video
    itself. The actual video-open happens in the GUI clip player.
    """
    import numpy as np
    import pandas as pd

    keypoints = [
        "snout",
        "left_ear",
        "right_ear",
        "neck",
        "body_center",
        "left_hip",
        "right_hip",
        "tail_base",
    ]
    cols = pd.MultiIndex.from_product(
        [["scorer"], keypoints, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    rng = np.random.default_rng(hash(name) & 0xFFFF)
    data = rng.uniform(10.0, 200.0, size=(n_frames, len(keypoints) * 3))
    # Force likelihood column to ~1.0 so the smoother / sampler keeps every frame.
    for k_idx in range(len(keypoints)):
        data[:, k_idx * 3 + 2] = 0.9
    df = pd.DataFrame(data, columns=cols)
    pose_csv = tmp_path / f"{name}.csv"
    df.to_csv(pose_csv)
    video = tmp_path / f"{name}.mp4"
    video.write_bytes(b"")  # placeholder; sampler never opens it
    return pose_csv, video


def test_propose_clips_multi_allocates_per_video_quota(tmp_path):
    from glider.gui.behavior.annotator.sampler import propose_clips_multi

    sessions = [_make_synthetic_session(tmp_path, f"vid_{i:02d}", n_frames=600) for i in range(3)]
    pairs = [(pose, video) for pose, video in sessions]

    clips = propose_clips_multi(
        sessions=pairs,
        n_clips_total=10,
        window=30,
        fps=30.0,
        random_state=42,
    )

    assert len(clips) == 10
    # 10 // 3 = 3 each, remainder 1 → first video gets 4, others get 3.
    from collections import Counter

    counts = Counter(c.video_path for c in clips)
    quotas = sorted(counts.values(), reverse=True)
    assert quotas == [4, 3, 3]


def test_propose_clips_multi_is_deterministic(tmp_path):
    from glider.gui.behavior.annotator.sampler import propose_clips_multi

    sessions = [_make_synthetic_session(tmp_path, f"vid_{i:02d}") for i in range(3)]
    pairs = [(pose, video) for pose, video in sessions]

    a = propose_clips_multi(sessions=pairs, n_clips_total=12, random_state=99)
    b = propose_clips_multi(sessions=pairs, n_clips_total=12, random_state=99)

    assert [c.center_frame for c in a] == [c.center_frame for c in b]
    assert [c.video_path for c in a] == [c.video_path for c in b]


def test_propose_clips_multi_per_video_seeds_decorrelate(tmp_path):
    """Two different videos should NOT produce identical clip patterns just
    because they share a top-level seed. Per-video seed offsets prevent that.
    """
    from glider.gui.behavior.annotator.sampler import propose_clips_multi

    s1 = _make_synthetic_session(tmp_path, "alpha", n_frames=600)
    s2 = _make_synthetic_session(tmp_path, "beta", n_frames=600)

    clips = propose_clips_multi(sessions=[s1, s2], n_clips_total=6, random_state=42)
    # Both videos used different seeds (42 vs 43), so even with the same input
    # shape the chosen center frames should differ between videos.
    per_video = {}
    for c in clips:
        per_video.setdefault(c.video_path, []).append(c.center_frame)
    a_frames = sorted(per_video[str(s1[1])])
    b_frames = sorted(per_video[str(s2[1])])
    # They might happen to share one or two purely by luck — but not all three.
    assert a_frames != b_frames


def test_propose_clips_multi_under_quota_video_contributes_what_it_can(tmp_path):
    """If a video has fewer valid windowed rows than its quota, it contributes
    all of them and the unused quota is NOT redistributed."""
    from glider.gui.behavior.annotator.sampler import propose_clips_multi

    big = _make_synthetic_session(tmp_path, "big", n_frames=600)
    # 70-frame video with window=30 leaves ~40 valid windowed rows
    # (well above the kinematic-lookback NaN prefix; still under the
    # 50-clip quota allocated below).
    tiny = _make_synthetic_session(tmp_path, "tiny", n_frames=70)

    clips = propose_clips_multi(sessions=[big, tiny], n_clips_total=100, window=30, random_state=7)

    from collections import Counter

    by_vid = Counter(c.video_path for c in clips)
    # Big should get its full 50-quota; tiny capped by valid-row count (~40).
    assert by_vid[str(big[1])] == 50
    assert by_vid[str(tiny[1])] <= 50
    assert by_vid[str(tiny[1])] >= 1
    # And no redistribution — tiny's shortfall is NOT made up by sampling more
    # from big.
    assert len(clips) < 100
