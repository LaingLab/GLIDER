"""The image lane: crop embedding beside the keypoint features.

The networks are not run here (they need model files and minutes of video);
every test that needs raw embeddings gets synthetic ones through the same
seam the pipeline uses, ``_image_lane_raw``. What is pinned is everything
GLIDER itself computes: the causal crop, the mask shape descriptors, the PCA
fitted on training sessions only, the windows (identical to the research
pandas recipe), the live stream reproducing the offline columns frame for
frame, and the NaN tolerance of the lane columns.
"""

from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior import FeatureSpec
from glider.analysis.behavior import image_lane as il
from glider.analysis.behavior.image_lane import (
    IMAGE_LANE_PREFIX,
    N_SHAPE,
    CropTracker,
    ImageLaneArtifact,
    ImageLaneConfig,
    ImageLaneEncoder,
    StreamingImageLane,
    required_columns,
    window_lane,
)

pytest.importorskip("lightgbm")

N_NECK = 16
RAW_WIDTH = N_SHAPE + 768 + N_NECK
CONFIG = ImageLaneConfig(seg_model="seg.pt", neck_model="neck.pt", dims=4)
SPEC = FeatureSpec(body_axis=(0, 4))
COMMON = {"window": 8, "n_estimators": 20, "random_state": 42, "fps": 30.0}


# ---------------------------------------------------------------------------
# Synthetic sessions + raw lane rows
# ---------------------------------------------------------------------------


def _pose(seed: int, n_frames: int = 300):
    from glider.vision.pose.core import PoseData

    rng = np.random.default_rng(seed)
    names = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
    offsets = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)
    xy = np.empty((n_frames, len(names), 2))
    per = n_frames // 3
    for regime, slc in enumerate([slice(0, per), slice(per, 2 * per), slice(2 * per, n_frames)]):
        n = slc.stop - slc.start
        t = np.arange(n)
        if regime == 0:
            cx, cy, jitter = 50 + 0.5 * t, np.full(n, 200.0), 0.4
        elif regime == 1:
            cx, cy, jitter = 350 + 3 * np.sin(0.5 * t), 200 + 3 * np.cos(0.5 * t), 1.0
        else:
            cx, cy, jitter = np.full(n, 360.0), np.full(n, 200.0), 0.15
        for k in range(len(names)):
            xy[slc, k, 0] = cx + offsets[k, 0] + rng.normal(0, jitter, n)
            xy[slc, k, 1] = cy + offsets[k, 1] + rng.normal(0, jitter, n)
    return PoseData(
        xy=xy, confidence=np.full((n_frames, len(names)), 0.95), keypoint_names=names, fps=30.0
    )


def _raw_for(n_frames: int, seed: int) -> np.ndarray:
    """Raw lane rows whose level tracks the regime, so the lane carries signal."""
    rng = np.random.default_rng(seed)
    regime = np.minimum(np.arange(n_frames) // (n_frames // 3), 2)
    raw = rng.normal(size=(n_frames, RAW_WIDTH)) + regime[:, None]
    raw[::17, :N_SHAPE] = np.nan  # the seg model misses the odd frame
    return raw.astype(np.float32)


@pytest.fixture
def sessions(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.vision.pose.dlc import to_dlc_csv

    pairs = []
    for i in range(4):
        pose = _pose(seed=i)
        pose_csv = tmp_path / f"s{i}.csv"
        to_dlc_csv(pose, pose_csv)
        store = AnnotationStore()
        third = pose.n_frames // 3
        store.add(BehaviorZone("walk", 0, third))
        store.add(BehaviorZone("mill", third, 2 * third))
        store.add(BehaviorZone("still", 2 * third, pose.n_frames))
        ann = tmp_path / f"s{i}_annotations.csv"
        store.save_csv(ann)
        pairs.append((pose_csv, ann))
    return pairs


@pytest.fixture
def fake_raw(monkeypatch):
    """Route the pipeline's raw-lane seam to synthetic rows; record who asked."""
    from glider.analysis.behavior import pipeline

    asked: list[str] = []

    def raw(pose_csv, pose, spec, config):
        asked.append(str(pose_csv))
        return _raw_for(pose.n_frames, seed=int(str(pose_csv)[-5]))

    monkeypatch.setattr(pipeline, "_image_lane_raw", raw)
    return asked


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("center", [False, True])
def test_windows_match_the_research_pandas_recipe(center):
    """value, rolling mean/std (min_periods=1) and diff, as pm_raster.win() builds them."""
    rng = np.random.default_rng(0)
    vals = rng.normal(size=(120, 5))
    vals[[3, 40, 41, 90]] = np.nan
    df = pd.DataFrame(vals)
    r = df.rolling(5, min_periods=1, center=center)
    ref = pd.concat([df, r.mean(), r.std(), df.diff()], axis=1).to_numpy()

    got = window_lane(vals, np.arange(120), np.arange(120), window=5, center=center)

    np.testing.assert_allclose(got, ref, atol=1e-12)
    assert (np.isnan(got) == np.isnan(ref)).all()


def test_absent_frames_count_as_missing_not_as_neighbours():
    """A sparse frame set must give what the dense, NaN-filled one does."""
    rng = np.random.default_rng(1)
    dense = rng.normal(size=(50, 3))
    keep = np.setdiff1d(np.arange(50), [10, 11, 30])
    dense_nan = dense.copy()
    dense_nan[[10, 11, 30]] = np.nan

    sparse = window_lane(dense[keep], keep, np.arange(50), window=5)
    full = window_lane(dense_nan, np.arange(50), np.arange(50), window=5)

    np.testing.assert_allclose(sparse, full, equal_nan=True)


def test_past_window_never_reads_a_later_frame():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(40, 3))
    b = a.copy()
    b[21:] += 100.0
    ra = window_lane(a, np.arange(40), np.arange(21), window=5)
    rb = window_lane(b, np.arange(40), np.arange(21), window=5)
    np.testing.assert_array_equal(ra, rb)


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def test_fit_projects_each_learned_block_to_dims_and_passes_shape_through():
    raws = [_raw_for(300, 0), _raw_for(300, 1)]
    art = ImageLaneArtifact.fit(raws, CONFIG)

    per_frame = art.transform(raws[0])
    assert per_frame.shape == (300, N_SHAPE + 2 * CONFIG.dims)
    np.testing.assert_array_equal(per_frame[:, :N_SHAPE], raws[0][:, :N_SHAPE])
    assert len(art.columns) == 4 * per_frame.shape[1]
    assert all(c.startswith(IMAGE_LANE_PREFIX) for c in art.columns)
    # No lane column may look like a keypoint rolling stat, or the live
    # engine would try to stream it as a per-frame keypoint feature.
    assert not any("__" in c for c in art.columns)


def test_fit_uses_only_the_sessions_it_is_given():
    """The held-out session must not shape the PCA that scores it."""
    a, b = _raw_for(300, 0), _raw_for(300, 1)
    held_out = _raw_for(300, 2) * 50.0
    art1 = ImageLaneArtifact.fit([a, b], CONFIG)
    art2 = ImageLaneArtifact.fit([a, b], CONFIG)
    art3 = ImageLaneArtifact.fit([a, b, held_out], CONFIG)
    np.testing.assert_array_equal(art1.transform(held_out), art2.transform(held_out))
    assert not np.allclose(art1.transform(a), art3.transform(a))


def test_fit_refuses_rows_it_cannot_split():
    with pytest.raises(ValueError, match="shape"):
        ImageLaneArtifact.fit([np.zeros((10, N_SHAPE + 5))], CONFIG)


# ---------------------------------------------------------------------------
# Crop + mask shape
# ---------------------------------------------------------------------------


def test_crop_tracker_carries_the_centre_and_takes_a_running_median_scale():
    tracker = CropTracker(CONFIG, body_axis=(0, 1))
    assert tracker.push(np.full((2, 2), np.nan)) is None  # nothing seen yet

    center, scale = tracker.push(np.array([[0.0, 0.0], [100.0, 0.0]]))
    np.testing.assert_allclose(center, [50.0, 0.0])
    assert scale == pytest.approx(CONFIG.crop_px / (CONFIG.crop_body_lengths * 100.0))

    # A missed frame keeps the last centre; a short (rearing) body barely moves the scale.
    center2, scale2 = tracker.push(np.full((2, 2), np.nan))
    np.testing.assert_allclose(center2, center)
    tracker.push(np.array([[0.0, 0.0], [100.0, 0.0]]))
    _c, scale3 = tracker.push(np.array([[0.0, 0.0], [40.0, 0.0]]))
    assert scale3 == pytest.approx(scale)


def test_crop_tracker_can_centre_on_a_named_keypoint():
    cfg = ImageLaneConfig(seg_model="s", neck_model="n", center_keypoint=1)
    center, _ = CropTracker(cfg, (0, 1)).push(np.array([[0.0, 0.0], [10.0, 4.0]]))
    np.testing.assert_allclose(center, [10.0, 4.0])


def test_shape_features_describe_an_ellipse_and_blank_an_empty_mask():
    cv2 = pytest.importorskip("cv2")
    mask = np.zeros((320, 320), np.uint8)
    cv2.ellipse(mask, (160, 160), (80, 30), 0, 0, 360, 1, -1)
    f = il.shape_features(mask.astype(bool))
    assert np.isfinite(f).all()
    pc = 320 / il.NOMINAL_CROP_CM
    assert f[0] == pytest.approx(np.pi * 80 * 30 / pc**2, rel=0.02)  # area, nominal cm^2
    assert f[4] == pytest.approx(30 / 80, rel=0.05)  # minor/major axis
    assert np.isnan(il.shape_features(np.zeros((320, 320), bool))).all()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_names_the_folder_when_a_model_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GLIDER_IMAGE_LANE_MODELS", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="mouse_seg.pt and mmv2.pt"):
        il.default_config()
    (tmp_path / "mouse_seg.pt").write_bytes(b"x")
    (tmp_path / "mmv2.pt").write_bytes(b"x")
    cfg = il.resolve_config(True)
    assert cfg.seg_model == str(tmp_path / "mouse_seg.pt")
    assert cfg.center is False  # the shipped lane is the live (past-only) one


def test_resolve_config_accepts_off_dicts_and_configs():
    assert il.resolve_config(None) is None
    assert il.resolve_config(False) is None
    assert il.resolve_config(CONFIG) is CONFIG
    assert il.resolve_config(CONFIG.to_dict()) == CONFIG
    with pytest.raises(TypeError):
        il.resolve_config("yes")


def test_cache_tag_changes_with_anything_that_changes_the_raw_rows():
    assert CONFIG.cache_tag() != ImageLaneConfig("seg.pt", "neck.pt", crop_px=256).cache_tag()
    # dims / window / center only change the reducer, so the cache is shared.
    assert CONFIG.cache_tag() == ImageLaneConfig("seg.pt", "neck.pt", dims=8, window=9).cache_tag()


# ---------------------------------------------------------------------------
# Training + cross-validation
# ---------------------------------------------------------------------------


def test_train_model_appends_the_lane_and_bundles_its_reducer(sessions, fake_raw, tmp_path):
    from glider.analysis.behavior.model import BehaviorModel
    from glider.analysis.behavior.pipeline import train_model

    result = train_model(sessions[:3], spec=SPEC, image_lane=CONFIG, **COMMON)
    model = result.model

    assert isinstance(model.image_lane, ImageLaneArtifact)
    lane_cols = [c for c in model.feature_names if c.startswith(IMAGE_LANE_PREFIX)]
    assert lane_cols == model.image_lane.columns
    assert result.summary["image_lane"]["dims"] == CONFIG.dims

    path = model.save(tmp_path / "m.pkl")
    loaded = BehaviorModel.load(path)
    np.testing.assert_array_equal(
        loaded.image_lane.transform(_raw_for(50, 9)), model.image_lane.transform(_raw_for(50, 9))
    )


def test_rows_with_a_missed_mask_still_train(sessions, fake_raw):
    """Lane NaN is routed by the classifier, not a reason to drop the frame."""
    from glider.analysis.behavior.pipeline import train_model

    with_lane = train_model(sessions[:3], spec=SPEC, image_lane=CONFIG, **COMMON)
    without = train_model(sessions[:3], spec=SPEC, **COMMON)
    assert with_lane.summary["n_rows_kept"] == without.summary["n_rows_kept"]


def test_the_holdout_session_is_kept_out_of_the_lane_fit(sessions, fake_raw, monkeypatch):
    from glider.analysis.behavior import pipeline

    fitted_on: list[int] = []
    real_fit = ImageLaneArtifact.fit.__func__

    def spy(cls, raws, config, **kw):
        fitted_on.append(len(raws))
        return real_fit(cls, raws, config, **kw)

    monkeypatch.setattr(ImageLaneArtifact, "fit", classmethod(spy))
    result = pipeline.train_model(
        sessions[:3], spec=SPEC, image_lane=CONFIG, holdout_sessions=[sessions[3]], **COMMON
    )
    assert fitted_on == [3]
    assert result.summary["test_size"] > 0


def test_image_lane_with_mirror_is_refused(sessions, fake_raw):
    from glider.analysis.behavior.pipeline import cross_validate_sessions, train_model

    with pytest.raises(ValueError, match="mirror"):
        train_model(sessions[:2], spec=SPEC, image_lane=CONFIG, mirror_augment=True, **COMMON)
    with pytest.raises(ValueError, match="mirror"):
        cross_validate_sessions(
            sessions, spec=SPEC, image_lane=CONFIG, mirror_augment=True, n_folds=2, **COMMON
        )


def test_each_cv_fold_fits_the_lane_without_its_held_out_session(sessions, fake_raw, monkeypatch):
    from glider.analysis.behavior import pipeline

    folds: list[set[int]] = []
    real = pipeline._CvLane.fit

    def spy(self, train_sessions):
        folds.append({int(i) for i in train_sessions})
        return real(self, train_sessions)

    monkeypatch.setattr(pipeline._CvLane, "fit", spy)
    cv, trained = pipeline.cross_validate_and_train(
        sessions, spec=SPEC, image_lane=CONFIG, n_folds=4, **COMMON
    )

    assert len(folds) == 5
    assert all(len(f) == 3 for f in folds[:4])  # one session held out per fold
    assert len({frozenset(f) for f in folds[:4]}) == 4  # a different one each time
    assert folds[4] == {0, 1, 2, 3}  # the deployed lane sees every session
    assert cv["n_folds"] == 4
    assert trained.model.image_lane is not None
    assert any(c.startswith(IMAGE_LANE_PREFIX) for c in trained.model.feature_names)


# ---------------------------------------------------------------------------
# Inference: NaN tolerance, offline scoring, live stream
# ---------------------------------------------------------------------------


@pytest.fixture
def lane_model(sessions, fake_raw):
    from glider.analysis.behavior.pipeline import train_model

    return train_model(sessions[:3], spec=SPEC, image_lane=CONFIG, **COMMON).model


def test_required_columns_leave_out_only_the_lane():
    cols = ["dist_0_1__mean", "img_shape00", "img_dino01_m", "speed__std"]
    assert required_columns(cols) == ["dist_0_1__mean", "speed__std"]


def test_predict_one_ignores_lane_nan_but_blanks_on_keypoint_nan(lane_model):
    row = np.zeros(len(lane_model.feature_names))
    lane = [i for i, c in enumerate(lane_model.feature_names) if c.startswith(IMAGE_LANE_PREFIX)]
    keypoint = [i for i in range(len(row)) if i not in set(lane)]

    row[lane] = np.nan
    assert lane_model.predict_one(row) != ""
    row[keypoint[0]] = np.nan
    assert lane_model.predict_one(row) == ""


def test_offline_scoring_needs_the_video(lane_model):
    from glider.analysis.behavior.classify.batch import classify_pose_data

    with pytest.raises(NotImplementedError, match="image lane"):
        classify_pose_data(_pose(seed=7), lane_model)


def test_offline_scoring_appends_the_lane_at_each_row_frame(lane_model, monkeypatch):
    from glider.analysis.behavior.classify import batch

    pose = _pose(seed=7)
    raw = _raw_for(pose.n_frames, 7)
    monkeypatch.setattr(batch, "_image_lane_raw", lambda *a, **k: raw)
    captured = {}
    real_predict = batch._predict

    def capture(model, frame, *a):
        captured["frame"] = frame
        return real_predict(model, frame, *a)

    monkeypatch.setattr(batch, "_predict", capture)
    rows = batch.classify_pose_data(pose, lane_model, video="v.mp4", predict_every=1)

    # Only the keypoint warm-up (a one-row std is NaN) blanks a row; the
    # frames whose mask was missed (every 17th) are still scored.
    blank = [f for f, label in zip(rows.frames, rows.labels, strict=True) if not label]
    assert len(rows) > 250 and all(f <= 3 for f in blank)
    lane_part = captured["frame"][lane_model.image_lane.columns].to_numpy()
    scored = [f for f, label in zip(rows.frames, rows.labels, strict=True) if label]
    expected = lane_model.image_lane.frame_features(raw, query=np.asarray(scored)).to_numpy()
    np.testing.assert_allclose(lane_part, expected, equal_nan=True)


def test_the_live_stream_reproduces_the_offline_lane_columns(lane_model):
    """Frame by frame, the streaming lane equals the offline one — causally."""
    art = lane_model.image_lane
    raw = _raw_for(120, 3)
    values = art.transform(raw)
    offline = art.frame_features(raw).to_numpy()

    stream = StreamingImageLane(art)
    lag = 2  # the keypoint lane emits the frame two behind the newest
    for f in range(120):
        stream.push(f, values[f])
        if f >= lag:
            np.testing.assert_allclose(stream.row(f - lag), offline[f - lag], equal_nan=True)


def test_the_stream_refuses_a_centred_lane():
    art = ImageLaneArtifact.fit([_raw_for(100, 0)], ImageLaneConfig("s", "n", center=True))
    with pytest.raises(ValueError, match="centred"):
        StreamingImageLane(art)


def test_feature_engine_appends_the_lane_to_every_emitted_row(lane_model):
    from glider.analysis.behavior.classify.features_stream import derive_stream_columns
    from glider.analysis.behavior.classify.threads import END_OF_STREAM, FeatureEngine

    pose = _pose(seed=5, n_frames=60)
    raw = _raw_for(60, 5)
    values = lane_model.image_lane.transform(raw)
    per_frame, spectral = derive_stream_columns(lane_model)
    tracked, out = queue.Queue(), queue.Queue()
    engine = FeatureEngine(
        tracked_queue=tracked,
        classifier_queue=out,
        stop_event=threading.Event(),
        spec=lane_model.spec,
        keypoint_names=pose.keypoint_names,
        window=lane_model.window,
        stats=lane_model.stats,
        per_frame_feature_names=per_frame,
        predict_every=1,
        spectral_features=spectral,
        image_lane=lane_model.image_lane,
    )
    for f in range(60):
        tracked.put((f, None, pose.xy[f], pose.confidence[f], values[f]))
    tracked.put(END_OF_STREAM)
    engine.run()

    offline = lane_model.image_lane.frame_features(raw)
    rows = []
    while (item := out.get_nowait()) is not END_OF_STREAM:
        rows.append(item)
    assert rows
    for frame_idx, names, row in rows:
        assert names[-len(lane_model.image_lane.columns) :] == lane_model.image_lane.columns
        np.testing.assert_allclose(
            row[-len(lane_model.image_lane.columns) :],
            offline.loc[frame_idx].to_numpy(),
            equal_nan=True,
        )


def test_lane_worker_encodes_each_fixed_frame_and_passes_none_before_one(lane_model, monkeypatch):
    from glider.analysis.behavior.classify.threads import END_OF_STREAM, ImageLaneWorker

    encoder = MagicMock(spec=ImageLaneEncoder)
    encoder.device = "cuda:0"
    encoder.encode.side_effect = lambda crops: _raw_for(len(crops) * 3, 1)[: len(crops)]
    monkeypatch.setattr(il, "get_encoder", lambda config, device=None: encoder)

    pose = _pose(seed=1, n_frames=6)
    inq, outq = queue.Queue(), queue.Queue()
    worker = ImageLaneWorker(
        inq, outq, threading.Event(), lane_model.image_lane, body_axis=(0, 4), device="cpu"
    )
    frame = np.zeros((480, 640, 3), np.uint8)
    inq.put((0, frame, np.full((5, 2), np.nan), np.zeros(5)))
    for f in range(1, 6):
        inq.put((f, frame, pose.xy[f], pose.confidence[f]))
    inq.put(END_OF_STREAM)
    worker.run()

    items = []
    while (item := outq.get_nowait()) is not END_OF_STREAM:
        items.append(item)
    assert worker.error is None
    assert [it[0] for it in items] == list(range(6))  # every frame, in order
    assert items[0][4] is None  # no keypoint fix yet: nothing to crop
    assert all(it[4].shape == (len(lane_model.image_lane.frame_columns),) for it in items[1:])
    # Everything was already queued, so it went through as one micro-batch.
    batches = [len(c.args[0]) for c in encoder.encode.call_args_list]
    assert batches == [5]
    crop = encoder.encode.call_args.args[0][0]
    assert crop.shape == (CONFIG.crop_px, CONFIG.crop_px, 3)
    assert len(worker.encode_ms) == 5


def test_lane_worker_batches_no_more_than_max_batch(lane_model, monkeypatch):
    from glider.analysis.behavior.classify.threads import END_OF_STREAM, ImageLaneWorker

    encoder = MagicMock(spec=ImageLaneEncoder)
    encoder.device = "cuda:0"
    encoder.encode.side_effect = lambda crops: _raw_for(len(crops) * 3, 1)[: len(crops)]
    monkeypatch.setattr(il, "get_encoder", lambda config, device=None: encoder)
    pose = _pose(seed=1, n_frames=12)
    inq, outq = queue.Queue(), queue.Queue()
    worker = ImageLaneWorker(
        inq, outq, threading.Event(), lane_model.image_lane, (0, 4), max_batch=4
    )
    frame = np.zeros((480, 640, 3), np.uint8)
    for f in range(12):
        inq.put((f, frame, pose.xy[f], pose.confidence[f]))
    inq.put(END_OF_STREAM)
    worker.run()
    assert [len(c.args[0]) for c in encoder.encode.call_args_list] == [4, 4, 4]


def test_the_display_draws_on_a_copy_not_the_frame_the_lane_crops():
    """The tracker shares one array between display and features; the lane reads its pixels."""
    from glider.analysis.behavior.classify.threads import (
        END_OF_STREAM,
        DisplayConsumer,
        LatestLabel,
    )

    frame = np.zeros((120, 160, 3), np.uint8)
    q: queue.Queue = queue.Queue()
    pose = _pose(seed=0, n_frames=1)
    q.put((0, frame, pose.xy[0] / 4, pose.confidence[0]))
    q.put(END_OF_STREAM)
    display = DisplayConsumer(
        display_queue=q,
        latest_label=LatestLabel(),
        stop_event=threading.Event(),
        keypoint_names=pose.keypoint_names,
        display_window=False,
    )
    display.run()
    assert not frame.any()


def test_live_pipeline_refuses_a_centred_lane(sessions, fake_raw):
    from glider.analysis.behavior.classify.pipeline import (
        LiveInferenceConfig,
        LiveInferencePipeline,
    )
    from glider.analysis.behavior.pipeline import train_model

    centred = ImageLaneConfig(seg_model="seg.pt", neck_model="neck.pt", dims=4, center=True)
    model = train_model(sessions[:2], spec=SPEC, image_lane=centred, **COMMON).model
    config = LiveInferenceConfig(
        source="x.mp4",
        keypoint_names=["snout", "left_ear", "right_ear", "neck", "tail_base"],
        yolo_model_path=None,
        behavior_model_path=None,
        display=False,
    )
    with pytest.raises(RuntimeError, match="centred"):
        LiveInferencePipeline(config, model=model)


def test_live_pipeline_puts_the_lane_worker_between_tracker_and_engine(lane_model):
    from glider.analysis.behavior.classify.pipeline import (
        LiveInferenceConfig,
        LiveInferencePipeline,
    )
    from glider.analysis.behavior.classify.threads import ImageLaneWorker

    config = LiveInferenceConfig(
        source="x.mp4",
        keypoint_names=["snout", "left_ear", "right_ear", "neck", "tail_base"],
        yolo_model_path="pose.pt",
        behavior_model_path=None,
        display=False,
    )
    pipe = LiveInferencePipeline(config, model=lane_model)
    assert isinstance(pipe.lane_worker, ImageLaneWorker)
    assert pipe.lane_worker.in_queue is pipe.tracked_queue
    assert pipe.feature_engine.tracked_queue is pipe.lane_worker.out_queue
    assert pipe.lane_worker.body_axis == (0, 4)
