"""Tests for the Sturman OFT benchmark adapter.

Fixtures synthesise the two real inputs: a DeepLabCut pose CSV that mixes animal
keypoints with the five static arena markers, and a Sturman episode table
(semicolon-separated, ``from``/``to`` in seconds, a ``type`` column, and a
``CSVname`` video link). No real dataset is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.annotations import AnnotationStore
from glider.analysis.behavior.benchmarks import oft
from glider.analysis.behavior.features import compute_features
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import from_dlc_csv, to_dlc_csv

# Sturman-like skeleton: three animal keypoints + the five arena markers.
ANIMAL_KP = ["nose", "tailbase", "earr"]
ALL_KP = ANIMAL_KP + ["tl", "tr", "br", "bl", "centre"]


def _make_pose(n_frames: int = 60, keypoints=ALL_KP, fps: float = oft.OFT_FPS) -> PoseData:
    rng = np.random.default_rng(0)
    k = len(keypoints)
    xy = rng.uniform(0, 900, size=(n_frames, k, 2))
    conf = np.full((n_frames, k), 0.99)
    return PoseData(xy=xy, confidence=conf, keypoint_names=list(keypoints), fps=fps, source="dlc")


def _write_pose_csv(path: Path, **kw) -> Path:
    return to_dlc_csv(_make_pose(**kw), path)


def _write_label_table(path: Path, rows: list[dict], sep: str = ";") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep=sep, index=False)
    return path


# --- Pose cleaning ------------------------------------------------------------


def test_drop_keypoints_removes_arena_markers():
    pose = _make_pose()
    cleaned = oft.drop_keypoints(pose)
    assert cleaned.keypoint_names == ANIMAL_KP
    assert cleaned.n_keypoints == 3
    assert cleaned.n_frames == pose.n_frames


def test_drop_keypoints_preserves_order_and_data():
    pose = _make_pose()
    cleaned = oft.drop_keypoints(pose)
    # nose is index 0 in both; its coordinates are untouched.
    np.testing.assert_array_equal(cleaned.xy[:, 0, :], pose.xy[:, 0, :])


def test_drop_all_keypoints_raises():
    pose = _make_pose(keypoints=["tl", "tr", "bl", "br", "centre"])
    with pytest.raises(ValueError, match="every keypoint"):
        oft.drop_keypoints(pose)


def test_load_sturman_pose_is_animal_only(tmp_path):
    csv = _write_pose_csv(tmp_path / "mouseA.csv")
    pose = oft.load_sturman_pose(csv)
    assert pose.keypoint_names == ANIMAL_KP
    assert pose.fps == oft.OFT_FPS


def test_resolve_body_axis_by_name():
    pose = oft.drop_keypoints(_make_pose())
    assert oft.resolve_body_axis(pose) == (0, 1)  # nose, tailbase


def test_oft_feature_spec_uses_resolved_axis():
    pose = oft.drop_keypoints(_make_pose())
    spec = oft.oft_feature_spec(pose)
    assert spec.body_axis == (0, 1)


# --- Label conversion ---------------------------------------------------------


def test_labels_to_store_seconds_to_frames():
    store = oft.labels_to_store([("Supported", 1.0, 2.0)], fps=25.0)
    (zone,) = store.zones()
    assert zone.behavior == "Supported"
    assert (zone.start_frame, zone.end_frame) == (25, 50)


def test_labels_to_store_frames_unit():
    store = oft.labels_to_store([("Grooming", 10, 20)], time_unit="frames")
    (zone,) = store.zones()
    assert (zone.start_frame, zone.end_frame) == (10, 20)


def test_labels_to_store_normalises_casing():
    store = oft.labels_to_store([("supported", 0, 5)], time_unit="frames")
    assert store.zones()[0].behavior == "Supported"


def test_labels_to_store_skips_unknown_behavior():
    store = oft.labels_to_store([("Walking", 0, 5), ("Grooming", 5, 10)], time_unit="frames")
    assert [z.behavior for z in store] == ["Grooming"]


def test_labels_to_store_clamps_to_n_frames():
    store = oft.labels_to_store([("Supported", 1.0, 2.0)], fps=25.0, n_frames=30)
    (zone,) = store.zones()
    assert (zone.start_frame, zone.end_frame) == (25, 30)


def test_labels_to_store_drops_bout_starting_past_end():
    store = oft.labels_to_store([("Supported", 100, 110)], time_unit="frames", n_frames=30)
    assert len(store) == 0


def test_labels_to_store_zero_length_bout_gets_one_frame():
    store = oft.labels_to_store([("Grooming", 5, 5)], time_unit="frames")
    (zone,) = store.zones()
    assert (zone.start_frame, zone.end_frame) == (5, 6)


def test_labels_to_store_coalesces_same_behavior_overlap():
    store = oft.labels_to_store([("Supported", 10, 20), ("Supported", 15, 25)], time_unit="frames")
    (zone,) = store.zones()  # would raise OverlapError if not coalesced
    assert (zone.start_frame, zone.end_frame) == (10, 25)


def test_labels_to_store_rejects_bad_time_unit():
    with pytest.raises(ValueError, match="time_unit"):
        oft.labels_to_store([("Grooming", 0, 1)], time_unit="minutes")


# --- Label table reading ------------------------------------------------------


def test_read_label_table_sniffs_semicolon(tmp_path):
    path = _write_label_table(
        tmp_path / "labels.csv",
        [{"from": 1.0, "to": 2.0, "type": "Grooming", "CSVname": "mouseA"}],
        sep=";",
    )
    df = oft.read_label_table(path)
    assert list(df.columns) == ["from", "to", "type", "CSVname"]
    assert len(df) == 1


def test_read_label_table_reads_comma(tmp_path):
    path = _write_label_table(
        tmp_path / "labels.csv",
        [{"from": 1.0, "to": 2.0, "type": "Grooming", "CSVname": "mouseA"}],
        sep=",",
    )
    df = oft.read_label_table(path)
    assert len(df) == 1


def test_episodes_for_video_filters_by_link_column():
    df = pd.DataFrame(
        [
            {"from": 1.0, "to": 2.0, "type": "Grooming", "CSVname": "mouseA"},
            {"from": 3.0, "to": 4.0, "type": "Supported", "CSVname": "mouseB"},
        ]
    )
    eps = oft.episodes_for_video(df, oft._video_key("mouseA_DLC_resnet50.csv"))
    assert eps == [("Grooming", 1.0, 2.0)]


def test_episodes_for_video_no_filter_returns_all():
    df = pd.DataFrame([{"from": 1.0, "to": 2.0, "type": "Grooming", "CSVname": "x"}])
    assert len(oft.episodes_for_video(df, None)) == 1


def test_episodes_missing_column_raises():
    df = pd.DataFrame([{"from": 1.0, "to": 2.0}])  # no 'type'
    with pytest.raises(ValueError, match="type"):
        oft.episodes_for_video(df)


# --- End-to-end orchestration -------------------------------------------------


def test_build_oft_benchmark_end_to_end(tmp_path):
    pose_dir = tmp_path / "Output_DLC"
    pose_dir.mkdir()
    _write_pose_csv(pose_dir / "mouseA.csv", n_frames=60)
    _write_pose_csv(pose_dir / "mouseB.csv", n_frames=60)

    labels = _write_label_table(
        tmp_path / "labels.csv",
        [
            {"from": 0.0, "to": 1.0, "type": "Supported", "CSVname": "mouseA"},
            {"from": 1.0, "to": 2.0, "type": "Grooming", "CSVname": "mouseB"},
        ],
    )

    sessions = oft.build_oft_benchmark(pose_dir, labels, tmp_path / "out")
    assert [s.video_id for s in sessions] == ["mousea", "mouseb"]

    a = sessions[0]
    # Cleaned pose reads back animal-only.
    pose_back = from_dlc_csv(a.pose_csv, fps=oft.OFT_FPS)
    assert pose_back.keypoint_names == ANIMAL_KP
    # Annotations read back with the right frames (0-1s @ 25fps -> [0, 25)).
    store = AnnotationStore.load_csv(a.annotations_csv)
    (zone,) = store.zones()
    assert zone.behavior == "Supported"
    assert (zone.start_frame, zone.end_frame) == (0, 25)
    assert a.n_labeled_frames == 25


def test_build_oft_benchmark_output_feeds_feature_extraction(tmp_path):
    pose_dir = tmp_path / "Output_DLC"
    pose_dir.mkdir()
    _write_pose_csv(pose_dir / "mouseA.csv", n_frames=60)
    labels = _write_label_table(
        tmp_path / "labels.csv",
        [{"from": 0.0, "to": 1.0, "type": "Supported", "CSVname": "mouseA"}],
    )
    (session,) = oft.build_oft_benchmark(pose_dir, labels, tmp_path / "out")

    pose = from_dlc_csv(session.pose_csv, fps=oft.OFT_FPS)
    feats = compute_features(pose, oft.oft_feature_spec(pose))
    assert len(feats) == 60
    # No feature column should reference an arena marker.
    assert not any(m in col for col in feats.columns for m in ("tl", "tr", "centre"))


def test_build_oft_benchmark_skips_unlabeled_videos_by_default(tmp_path):
    # Two pose files; only one has labels. Default require_labels=True keeps
    # exactly the labeled subset (the real OFT set is 20 labeled of 59 videos).
    pose_dir = tmp_path / "Output_DLC"
    pose_dir.mkdir()
    _write_pose_csv(pose_dir / "labeled.csv", n_frames=30)
    _write_pose_csv(pose_dir / "unlabeled.csv", n_frames=30)
    labels = _write_label_table(
        tmp_path / "labels.csv",
        [{"from": 0.0, "to": 1.0, "type": "Supported", "DLCFile": "labeled"}],
    )
    sessions = oft.build_oft_benchmark(pose_dir, labels, tmp_path / "out")
    assert [s.video_id for s in sessions] == ["labeled"]


def test_build_oft_benchmark_require_labels_false_keeps_all(tmp_path):
    pose_dir = tmp_path / "Output_DLC"
    pose_dir.mkdir()
    _write_pose_csv(pose_dir / "mouseZ.csv", n_frames=30)
    labels = _write_label_table(
        tmp_path / "labels.csv",
        [{"from": 0.0, "to": 1.0, "type": "Supported", "DLCFile": "someone_else"}],
    )
    (session,) = oft.build_oft_benchmark(pose_dir, labels, tmp_path / "out", require_labels=False)
    assert session.n_labeled_frames == 0
    assert AnnotationStore.load_csv(session.annotations_csv).zones() == []


def test_build_oft_benchmark_all_unlabeled_raises(tmp_path):
    pose_dir = tmp_path / "Output_DLC"
    pose_dir.mkdir()
    _write_pose_csv(pose_dir / "mouseZ.csv", n_frames=30)
    labels = _write_label_table(
        tmp_path / "labels.csv",
        [{"from": 0.0, "to": 1.0, "type": "Supported", "DLCFile": "someone_else"}],
    )
    with pytest.raises(ValueError, match="no labeled videos"):
        oft.build_oft_benchmark(pose_dir, labels, tmp_path / "out")


# --- Multi-annotator handling -------------------------------------------------


def test_list_experimenters():
    df = pd.DataFrame(
        [
            {"from": 0, "to": 1, "type": "Grooming", "Experimenter": "Jin"},
            {"from": 1, "to": 2, "type": "Supported", "Experimenter": "Oliver"},
            {"from": 2, "to": 3, "type": "Grooming", "Experimenter": "Jin"},
        ]
    )
    assert oft.list_experimenters(df) == ["Jin", "Oliver"]


def test_episodes_for_video_filters_by_experimenter():
    df = pd.DataFrame(
        [
            {"from": 0.0, "to": 1.0, "type": "Grooming", "DLCFile": "m", "Experimenter": "Jin"},
            {"from": 5.0, "to": 6.0, "type": "Supported", "DLCFile": "m", "Experimenter": "Oliver"},
        ]
    )
    eps = oft.episodes_for_video(df, oft._video_key("m.csv"), experimenter="Jin")
    assert eps == [("Grooming", 0.0, 1.0)]


def test_experimenter_selection_avoids_rater_union(tmp_path):
    # Two raters both label Supported at overlapping-but-different times.
    # Selecting one rater must NOT union them.
    pose_dir = tmp_path / "Output_DLC"
    pose_dir.mkdir()
    _write_pose_csv(pose_dir / "m.csv", n_frames=300)
    labels = _write_label_table(
        tmp_path / "labels.csv",
        [
            {"from": 0.0, "to": 4.0, "type": "Supported", "DLCFile": "m", "Experimenter": "Jin"},
            {"from": 2.0, "to": 6.0, "type": "Supported", "DLCFile": "m", "Experimenter": "Oliver"},
        ],
    )
    (jin,) = oft.build_oft_benchmark(pose_dir, labels, tmp_path / "jin", experimenter="Jin")
    # Jin only: 0-4s @ 25fps = 100 frames (not the 0-6s=150 rater union).
    assert jin.n_labeled_frames == 100


def test_build_oft_benchmark_empty_pose_dir_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    labels = _write_label_table(tmp_path / "labels.csv", [{"from": 0, "to": 1, "type": "Grooming"}])
    with pytest.raises(FileNotFoundError, match="no pose CSVs"):
        oft.build_oft_benchmark(tmp_path / "empty", labels, tmp_path / "out")


def test_leave_one_out_splits():
    sessions = [oft.OFTSession(f"v{i}", Path(f"p{i}"), Path(f"a{i}"), 10, 5) for i in range(3)]
    folds = oft.leave_one_out_splits(sessions)
    assert len(folds) == 3
    train, test = folds[0]
    assert test.video_id == "v0"
    assert [s.video_id for s in train] == ["v1", "v2"]


def test_sessions_to_pairs():
    sessions = [oft.OFTSession("v0", Path("p0"), Path("a0"), 10, 5)]
    assert oft.sessions_to_pairs(sessions) == [(Path("p0"), Path("a0"))]
