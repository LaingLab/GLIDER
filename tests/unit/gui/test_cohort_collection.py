"""Collecting a cohort's pose CSVs for threshold derivation.

The behavior tool derives its cohort percentiles from every pose CSV under a
folder. Which files count as pose CSVs is decided in two places — the
annotate/train path calls ``find_pose_csv``, the cohort scan globs for itself —
and they drifted: the scan admitted the ``_raw`` companion alongside the
primary, so every session was pooled twice and each animal's weight in the
percentiles was silently halved.
"""

from __future__ import annotations


def test_cohort_collection_pools_each_session_once(tmp_path):
    """Regression: a folder with primaries and _raw companions pooled every
    session twice, silently halving each animal's weight in the percentiles."""
    from glider.gui.behavior.window import _unique_pose_csvs

    for stem in ("t1_d1", "t1_d2"):
        (tmp_path / f"{stem}DLC_exp-7.csv").write_text("x")
        (tmp_path / f"{stem}DLC_exp-7_raw.csv").write_text("x")
        (tmp_path / f"{stem}DLC_exp-7_ungated.csv").write_text("x")

    found = _unique_pose_csvs(tmp_path)
    assert len(found) == 2
    assert all("_raw" not in p.stem and "_ungated" not in p.stem for p in found)


def test_both_discovery_paths_share_one_exclusion_list(tmp_path):
    """They drifted apart once already. One list, two readers."""
    from glider.vision.pose.dlc import NOT_POSE_SUFFIXES

    video = tmp_path / "t1_d1.mp4"
    video.touch()
    primary = tmp_path / "t1_d1DLC_exp-7.csv"
    primary.write_text("x")
    for suffix in NOT_POSE_SUFFIXES:
        (tmp_path / f"t1_d1DLC_exp-7{suffix}.csv").write_text("x")

    from glider.gui.behavior.window import _unique_pose_csvs
    from glider.vision.pose.batch import find_pose_csv

    assert find_pose_csv(video) == primary
    assert _unique_pose_csvs(tmp_path) == [primary]
