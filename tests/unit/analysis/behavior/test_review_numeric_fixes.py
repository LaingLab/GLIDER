"""Regression tests for the behavior-analysis findings of the 2026-09-28 review.

One focused test per fix, each asserting the number the fixed code must
report (E1-E12, E-notes, G5, and the storage half of F10).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
from glider.vision.pose.core import PoseData
from glider.vision.pose.dlc import to_dlc_csv

NAMES = ["nose", "left_ear", "right_ear", "tail_base"]


def _pose(n=450, seed=0, names=NAMES, fps=30.0):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1.0, size=(n, 1, 2)), axis=0) + 300.0
    xy = base + rng.normal(0, 3.0, size=(n, len(names), 2))
    return PoseData(xy=xy, confidence=np.ones((n, len(names))), keypoint_names=list(names), fps=fps)


def _session(tmp_path, name, zones, *, n=450, seed=0):
    pose_csv = tmp_path / f"{name}.csv"
    to_dlc_csv(_pose(n=n, seed=seed), pose_csv)
    store = AnnotationStore()
    for behavior, start, end in zones:
        store.add(BehaviorZone(behavior, start, end))
    ann = store.save_csv(tmp_path / f"{name}_annotations.csv")
    return pose_csv, ann


# ---------------------------------------------------------------------------
# E1 / E6 / E12: training assembly
# ---------------------------------------------------------------------------


def test_e1_a_mirrored_copy_shares_its_originals_zone_ids(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.pipeline import _assemble_sessions

    pair = _session(tmp_path, "s", [("walk", 0, 150), ("rear", 150, 300), ("walk", 300, 450)])
    _x, _y, groups, _c = _assemble_sessions(
        [pair],
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        mirror_augment=True,
    )
    g = groups.to_numpy()
    original, mirrored = g[:450], g[450:]
    np.testing.assert_array_equal(original, mirrored)
    assert int(g.max()) == 2  # three zones, not six


def test_e6_mirroring_preserves_every_keypoint_speed():
    from glider.analysis.behavior.pipeline import _mirror_pose

    pose = _pose(n=200, names=["a", "b", "c"])
    mirrored = _mirror_pose(pose)
    speed = np.linalg.norm(np.diff(pose.xy, axis=0), axis=2)
    speed_m = np.linalg.norm(np.diff(mirrored.xy, axis=0), axis=2)
    np.testing.assert_allclose(speed_m, speed, atol=1e-9)


def test_e12_holdout_accuracy_is_scored_on_clean_windows_only(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.pipeline import train_model

    zones = [("walk", 0, 150), ("rear", 150, 300), ("walk", 300, 450)]
    train = _session(tmp_path, "train", zones, seed=1)
    hold = _session(tmp_path, "hold", zones, seed=2)
    result = train_model(
        [train],
        holdout_sessions=[hold],
        spec=FeatureSpec(),
        window=10,
        fps=30.0,
        n_estimators=5,
        classifier_type="rf",
    )
    # 450 labelled frames; the first 10 have no complete features, and the
    # two internal boundaries each lose the 9 frames whose window straddles
    # them (the holdout used to keep those 18, unlike CV and evaluation).
    assert result.summary["test_size"] == 450 - 10 - 2 * 9


# ---------------------------------------------------------------------------
# E2 / E9: cross-validation scoring
# ---------------------------------------------------------------------------


def test_e2_thinned_background_is_still_scoreable(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.pipeline import _assemble_for_cv

    pairs = [_session(tmp_path, f"s{i}", [("walk", 300, 400)], seed=i) for i in range(2)]
    data = _assemble_for_cv(
        pairs,
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        mirror_augment=False,
        merge_map=None,
        exclude=None,
        freq_features=False,
        traj_features=False,
        motion_features=False,
        include_background=True,
        background_class_name="background",
        background_ratio=1.0,
        random_state=0,
    )
    background = (data.y == "background").to_numpy()
    assert background.sum() == 200  # thinned to 1x the 2 x 100 walk rows
    # Before: ~0 survivors had window-1 consecutive thinned neighbours.
    assert data.clean[background].mean() > 0.9


def test_e9_clean_window_counts_over_every_frame_not_kept_rows(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.pipeline import _assemble_for_cv

    pair = _session(tmp_path, "s", [("walk", 0, 450)])
    data = _assemble_for_cv(
        [pair],
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        mirror_augment=False,
        merge_map=None,
        exclude=None,
        freq_features=False,
        traj_features=False,
        motion_features=False,
        include_background=False,
        background_class_name="background",
        background_ratio=5.0,
        random_state=0,
    )
    # The first featured frame's window is all "walk", counted over every
    # frame exactly as evaluate_model does. CV used to count runs over kept
    # rows only and drop 9 more rows here.
    assert bool(data.clean[0]) is True
    assert data.clean.all()


def test_e9_cv_macro_honours_the_floor_and_bouts_see_short_bouts(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.pipeline import cross_validate_sessions

    zones = [("walk", 0, 200), ("blip", 200, 205), ("rear", 205, 450)]
    pairs = [_session(tmp_path, f"s{i}", zones, seed=i) for i in range(3)]
    res = cross_validate_sessions(
        pairs,
        spec=FeatureSpec(),
        window=10,
        fps=30.0,
        n_estimators=5,
        classifier_type="rf",
        n_folds=3,
    )
    pcm = res["per_class_metrics"]
    kept = [c for c, m in pcm.items() if m["support"] >= res["support_floor"]]
    assert set(kept) == {"walk", "rear"}
    assert res["macro_f1"] == pytest.approx(np.mean([pcm[c]["f1"] for c in kept]))
    # A 5-frame bout is shorter than the 10-frame window, so the clean-window
    # filter erases every one of its frames. Bouts must still count it.
    assert res["bout_metrics"]["blip"]["n_bouts"] == 3


# ---------------------------------------------------------------------------
# E3 / G5 / E11: session view and spatial
# ---------------------------------------------------------------------------


def _strided_session(folder, *, n=300, stride=3, label="freezing", resolution=(640, 480)):
    folder.mkdir(parents=True, exist_ok=True)
    frames = np.arange(0, n, stride)
    pd.DataFrame({"frame": frames, "behavior": [label] * len(frames)}).to_csv(
        folder / "ethogram_raw.csv", index=False
    )
    xy = np.full((n, 2, 2), 100.0)
    to_dlc_csv(
        PoseData(
            xy=xy,
            confidence=np.ones((n, 2)),
            keypoint_names=["nose", "tail"],
            fps=30.0,
            metadata={"resolution": resolution},
        ),
        folder / "vDLC_x.csv",
    )
    return folder / "ethogram_raw.csv"


def test_e3_a_strided_all_freeze_window_reports_its_real_duration(tmp_path):
    from glider.analysis.behavior.session_view import SessionView

    view = SessionView.load(_strided_session(tmp_path / "v"))
    assert view.duration_s == pytest.approx(10.0)
    bouts = view.segment_stats(0, 299).bouts.set_index("state")
    assert bouts.loc["freezing", "total_s"] == pytest.approx(10.0)
    assert bouts.loc["freezing", "fraction"] == pytest.approx(1.0)


def test_e3_zone_occupancy_fraction_uses_real_time(tmp_path):
    from glider.analysis.behavior.session_view import SessionView
    from glider.analysis.behavior.spatial import zone_occupancy
    from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

    view = SessionView.load(_strided_session(tmp_path / "v"))
    zones = ZoneConfiguration()
    zones.add_zone(Zone("z", "arena", ZoneShape.RECTANGLE, [(0.0, 0.0), (1.0, 1.0)]))
    row = zone_occupancy(view, zones).set_index("zone").loc["arena"]
    assert row["total_s"] == pytest.approx(10.0)
    assert row["fraction"] == pytest.approx(1.0)  # was ~3.0


def test_g5_no_label_past_the_last_rows_stride(tmp_path):
    from glider.analysis.behavior.session_view import SessionView

    view = SessionView.load(_strided_session(tmp_path / "v"))
    assert view.label_at(299) == "freezing"  # inside the last row's stride
    assert view.label_at(300) == ""
    assert view.bout_at(5000) is None


def test_e11_a_circle_zone_is_a_pixel_circle_on_widescreen():
    from glider.analysis.behavior.spatial import _zone_ids
    from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

    zones = ZoneConfiguration()
    # Radius 0.1 of the width = 192 px on 1920x1080.
    zones.add_zone(Zone("c", "ring", ZoneShape.CIRCLE, [(0.5, 0.5), (0.6, 0.5)]))
    track = np.array([[960.0, 540.0 + 150.0], [960.0, 540.0 + 200.0]])
    # 150 px below centre is inside a 192 px circle; 200 px is outside.
    assert _zone_ids(track, zones, (1920, 1080)) == ["ring", ""]


# ---------------------------------------------------------------------------
# E4 / E5 / E7: speed axis and frame rate
# ---------------------------------------------------------------------------


def test_e4_reseed_after_dropout_is_nan_and_does_not_halve_the_next_speed():
    from glider.analysis.behavior.classify.speed_state import causal_speed_series

    frames = np.array([[[float(i), 0.0]] for i in range(10)])
    frames[5] = np.nan
    speeds = causal_speed_series(frames, coord_smooth=1, speed_smooth=3)
    assert np.isnan(speeds[6])  # re-seed: no predecessor
    assert speeds[7] == pytest.approx(1.0)  # was 0.5 (averaged with a fake 0)


def test_e5_a_model_fps_mismatch_warns():
    from glider.analysis.behavior.classify.batch import warn_on_fps_mismatch

    with pytest.warns(UserWarning, match="60 fps.*30 fps"):
        assert warn_on_fps_mismatch(60.0, 30.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not warn_on_fps_mismatch(29.97, 30.0)


def test_e5_an_unrecorded_pose_rate_does_not_override_the_run_fps(tmp_path):
    from glider.analysis.behavior.session_view import SessionView

    folder = tmp_path / "v"
    folder.mkdir()
    pd.DataFrame({"frame": range(60), "behavior": ["rest"] * 60}).to_csv(
        folder / "ethogram_raw.csv", index=False
    )
    (folder / "run.json").write_text(json.dumps({"fps": 60.0}))
    pose = _pose(n=60, names=["nose", "tail"])
    to_dlc_csv(pose, folder / "vDLC_x.csv", write_meta=False)
    view = SessionView.load(folder / "ethogram_raw.csv")
    assert view.xy is not None
    assert view.fps == pytest.approx(60.0)  # not the 30 fps default


def test_e7_streamed_freeze_bouts_are_relabelled_whole(tmp_path):
    from types import SimpleNamespace

    from glider.analysis.behavior.classify import _relabel_speed_axis_offline

    n = 120
    xy = np.zeros((n, 2, 2))
    xy[60:, :, 0] = np.arange(n - 60)[:, None] * 5.0  # still, then walking
    pose_csv = tmp_path / "p.csv"
    to_dlc_csv(PoseData(xy=xy, confidence=np.ones((n, 2)), keypoint_names=["a", "b"]), pose_csv)
    # What the live detector wrote: freezing confirmed only from frame 29.
    frames = np.arange(0, n, 3)
    labels = ["freezing" if 29 <= f < 60 else "rest" for f in frames]
    etho = tmp_path / "ethogram_raw.csv"
    pd.DataFrame({"frame": frames, "behavior": labels}).to_csv(etho, index=False)
    config = SimpleNamespace(
        freeze_threshold=0.5, dart_threshold=1000.0, freeze_min_frames=30, dart_min_frames=3
    )
    _relabel_speed_axis_offline(etho, pose_csv, config)
    out = pd.read_csv(etho)
    assert (out.loc[out["frame"] < 60, "behavior"] == "freezing").all()
    assert (out.loc[out["frame"] >= 63, "behavior"] == "rest").all()


# ---------------------------------------------------------------------------
# E8 / E10: sequence model and motion cache
# ---------------------------------------------------------------------------


def test_e8_sequence_assembly_refuses_a_multi_animal_csv(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.sequence import assemble_sequences

    pose_csv = tmp_path / "s.csv"
    to_dlc_csv(_pose(n=100), pose_csv)
    store = AnnotationStore()
    store.add(BehaviorZone("walk", 0, 50, individual=0))
    store.add(BehaviorZone("walk", 0, 50, individual=1))
    ann = store.save_csv(tmp_path / "s_annotations.csv")
    with pytest.raises(ValueError, match="more than one animal"):
        assemble_sequences([(pose_csv, ann)], spec=FeatureSpec(), window=8)


def test_e10_motion_cache_does_not_collide_on_a_repeated_stem(tmp_path, monkeypatch):
    from glider.analysis.behavior import motion

    def fake(video_path, xy, body_axis, **_kw):
        value = 1.0 if "a" in video_path.parent.name else 2.0
        return pd.DataFrame(np.full((len(xy), 4), value), columns=motion.MOTION_COLUMNS)

    monkeypatch.setattr(motion, "compute_motion_for_video", fake)
    xy = np.zeros((5, 2, 2))
    out = {}
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        video = tmp_path / d / "v.mp4"
        video.write_bytes(b"")
        out[d] = motion.load_or_compute_motion(
            tmp_path / d / "pose.csv", video, xy, (0, 1), cache_dir=tmp_path / "cache"
        )
    assert out["a"]["motion_total"].iloc[0] == 1.0
    assert out["b"]["motion_total"].iloc[0] == 2.0  # was a's cached features


# ---------------------------------------------------------------------------
# F10 (storage half) and E-notes
# ---------------------------------------------------------------------------


def test_f10_a_failed_save_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    import csv

    path = tmp_path / "v_annotations.csv"
    AnnotationStore([BehaviorZone("walk", 0, 10)]).save_csv(path)
    before = path.read_text()

    def boom(self, row):
        raise OSError("disk full")

    monkeypatch.setattr(csv.DictWriter, "writerow", boom)
    with pytest.raises(OSError):
        AnnotationStore([BehaviorZone("rear", 0, 10)]).save_csv(path)
    assert path.read_text() == before
    assert list(tmp_path.iterdir()) == [path]  # no stray temp file


def test_enote_event_triggered_sem_is_across_trials():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from glider.analysis.plots import plot_event_triggered

    offsets = np.tile(np.linspace(-1000, 1000, 200), 2)
    eta = pd.DataFrame(
        {
            "trial_id": np.repeat([0, 1], 200),
            "event_time_ms": 0.0,
            "time_offset_ms": offsets,
            "value": np.repeat([0.0, 2.0], 200),
        }
    )
    ax = plot_event_triggered(eta, n_bins=10)
    band = ax.collections[0].get_paths()[0].vertices[:, 1]
    # mean 1, SEM over the two trials = std([0, 2]) / sqrt(2) = 1.
    assert band.min() == pytest.approx(0.0)
    assert band.max() == pytest.approx(2.0)


def test_enote_a_hash_inside_a_field_survives_parsing(tmp_path):
    from glider.analysis._io import parse_csv

    path = tmp_path / "x.csv"
    path.write_text("# Experiment, demo\nzone,note\narena,cage #4\n", encoding="utf-8")
    metadata, df = parse_csv(path)
    assert metadata["Experiment"] == "demo"
    assert df.loc[0, "note"] == "cage #4"
