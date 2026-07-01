"""Tests for the live inference pipeline.

Only covers pure-Python pieces (SlidingFeatureBuffer + LatestLabel +
the thread helper). Full threaded end-to-end testing would need a
mock YOLO model + OpenCV, which isn't worth the harness complexity at
this stage.

Ported from yolo2pose's ``tests/test_live.py``. The Qt/pyqtgraph-backed
embedding-view tests do not exist in the source file (it only ever
covered pure-Python pieces), so nothing was dropped on that front. The
``PoseTracker`` undetected-frame test is intentionally NOT ported here —
it belongs with a dedicated threads.py test module, not the pipeline
engine tests this file targets.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Live-unstreamable feature guard (motion_* / traj_* can't run live)
# ---------------------------------------------------------------------------


def test_unstreamable_feature_families_flags_motion_and_traj():
    """Models trained with --motion-features (needs the video) or
    --traj-features (live path unwired) emit NaN every frame -> blank
    predictions. The guard surfaces the offending feature stems."""
    from glider.analysis.behavior.classify.pipeline import _unstreamable_feature_families

    names = [
        "body_length__mean",
        "speed_nose__mean",
        "speed_nose__domfreq",
        "motion_total__mean",
        "motion_anterior__domfreq",
        "traj_straightness",
    ]
    assert _unstreamable_feature_families(names) == [
        "motion_anterior",
        "motion_total",
        "traj_straightness",
    ]


def test_unstreamable_feature_families_empty_for_pose_and_freq_model():
    """A pose-only or --freq-features model is fully streamable in live."""
    from glider.analysis.behavior.classify.pipeline import _unstreamable_feature_families

    assert (
        _unstreamable_feature_families(
            ["body_length__mean", "speed_nose__std", "accel_tail__domfreq"]
        )
        == []
    )


# ---------------------------------------------------------------------------
# SlidingFeatureBuffer
# ---------------------------------------------------------------------------


def test_buffer_starts_empty():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x", "y"], window=10)
    assert len(b) == 0
    assert not b.is_full()


def test_buffer_push_array_and_dict():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x", "y"], window=10)
    b.push_features(np.array([1.0, 2.0]))
    b.push_features({"x": 3.0, "y": 4.0})
    b.push_features(pd.Series({"x": 5.0, "y": 6.0}))
    assert len(b) == 3


def test_buffer_push_array_wrong_shape_raises():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x", "y"], window=10)
    with pytest.raises(ValueError):
        b.push_features(np.array([1.0]))  # length mismatch


def test_buffer_push_dict_handles_missing_and_extra_keys():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x", "y", "z"], window=10)
    # Missing 'z' becomes NaN. Extra 'q' is ignored.
    b.push_features({"x": 1.0, "y": 2.0, "q": 99.0})
    names, row = b.rolling_features()
    # First col (z) is NaN → its mean over the buffer is NaN.
    assert names == [
        "x__mean",
        "y__mean",
        "z__mean",
        "x__std",
        "y__std",
        "z__std",
        "x__max",
        "y__max",
        "z__max",
    ]
    assert row[0] == 1.0 and row[1] == 2.0 and np.isnan(row[2])


def test_buffer_emits_spectral_columns_for_kinematic_features():
    """With spectral_features set, the buffer appends __domfreq/__specflat
    columns whose values match the shared window_spectral function."""
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer
    from glider.analysis.behavior.spectral import window_spectral

    base = ["dist_a_b", "speed_nose", "accel_tail"]
    buf = SlidingFeatureBuffer(
        feature_names=base,
        window=30,
        stats=("mean", "std", "max"),
        spectral_features=["speed_nose", "accel_tail"],
    )
    rng = np.random.default_rng(1)
    speed = np.sin(2 * np.pi * 5 * np.arange(30) / 30)
    accel = rng.normal(size=30)
    for i in range(30):
        buf.push_features({"dist_a_b": float(i), "speed_nose": speed[i], "accel_tail": accel[i]})
    names, row = buf.rolling_features()
    col = dict(zip(names, row, strict=False))

    assert "speed_nose__domfreq" in col
    assert "accel_tail__specflat" in col
    # Values equal the shared function applied to the buffered window.
    exp_dom, exp_flat = window_spectral(speed)
    assert col["speed_nose__domfreq"] == pytest.approx(exp_dom)
    assert col["speed_nose__specflat"] == pytest.approx(exp_flat)


def test_train_and_live_spectral_columns_match_on_same_window():
    """Parity: apply_spectral_rolling (train) and SlidingFeatureBuffer
    (live) must produce identical spectral columns for one full window.
    This is the one place a silent train/live drift would wreck live
    accuracy, so it's pinned explicitly."""
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer
    from glider.analysis.behavior.windowing import apply_spectral_rolling

    window = 30
    rng = np.random.default_rng(7)
    # One full window of per-frame features.
    frames = pd.DataFrame(
        {
            "dist_a_b": rng.normal(size=window),
            "speed_nose": np.sin(2 * np.pi * 4 * np.arange(window) / window),
            "accel_tail": rng.normal(size=window),
            "body_angular_velocity": np.cos(2 * np.pi * 3 * np.arange(window) / window),
        }
    )

    # Train side: rolling apply, take the last (only full) row.
    train_out = apply_spectral_rolling(frames, window=window).iloc[-1]

    # Live side: push the same frames through the buffer.
    buf = SlidingFeatureBuffer(
        feature_names=list(frames.columns),
        window=window,
        stats=("mean", "std", "max"),
        spectral_features=["speed_nose", "accel_tail", "body_angular_velocity"],
    )
    for i in range(window):
        buf.push_features(frames.iloc[i].to_dict())
    names, row = buf.rolling_features()
    live = dict(zip(names, row, strict=False))

    for col in train_out.index:
        assert col in live, f"{col} missing from live output"
        np.testing.assert_allclose(
            live[col],
            train_out[col],
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"train/live mismatch for {col}",
        )


def test_sequence_classifier_predicts_from_keypoint_window():
    """The live CNN path buffers raw keypoints and calls predict_window
    once the trailing window is full, updating LatestLabel."""
    import queue

    from glider.analysis.behavior.classify.threads import (
        END_OF_STREAM,
        LatestLabel,
        SequenceClassifier,
    )

    class _StubModel:
        window = 3

        def predict_window(self, win, gate=None):
            # win is (3, K, 2); label by mean x of the last frame.
            return "move" if np.nanmean(win[-1, :, 0]) > 0 else "still"

    q: queue.Queue = queue.Queue()
    latest = LatestLabel()
    stop = threading.Event()
    clf = SequenceClassifier(
        tracked_queue=q,
        latest_label=latest,
        stop_event=stop,
        model=_StubModel(),
        predict_every=1,
    )
    clf.start()
    k = 4
    # 2 "still" frames (x<0) then 3 "move" frames (x>0).
    for i, sign in enumerate([-1, -1, 1, 1, 1]):
        kp = np.full((k, 2), float(sign))
        q.put((i, None, kp, None))
    q.put(END_OF_STREAM)
    clf.join(timeout=5)

    label = latest.get()[1]
    # Last full window ended on a "move" frame.
    assert label == "move"


def test_sequence_classifier_blank_during_warmup():
    """Before the buffer fills, no full window → label stays blank."""
    import queue

    from glider.analysis.behavior.classify.threads import (
        END_OF_STREAM,
        LatestLabel,
        SequenceClassifier,
    )

    class _StubModel:
        window = 10

        def predict_window(self, win, gate=None):
            return "x"

    q: queue.Queue = queue.Queue()
    latest = LatestLabel()
    stop = threading.Event()
    clf = SequenceClassifier(
        tracked_queue=q,
        latest_label=latest,
        stop_event=stop,
        model=_StubModel(),
        predict_every=1,
    )
    clf.start()
    for i in range(3):  # fewer than window=10
        q.put((i, None, np.zeros((4, 2)), None))
    q.put(END_OF_STREAM)
    clf.join(timeout=5)

    label = latest.get()[1]
    assert label == ""  # never reached a full window


def test_buffer_caps_at_window():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x"], window=3)
    for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
        b.push_features(np.array([v]))
    assert len(b) == 3
    assert b.is_full()
    # Mean over the kept window [30, 40, 50] = 40.
    _, row = b.rolling_features()
    # Columns: x__mean, x__std, x__max — that's the default stats order.
    assert row[0] == pytest.approx(40.0)
    assert row[2] == pytest.approx(50.0)


def test_buffer_rolling_stats_match_numpy():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    rng = np.random.default_rng(0)
    n = 30
    data = rng.normal(size=(n, 4))
    b = SlidingFeatureBuffer(
        feature_names=["a", "b", "c", "d"],
        window=n,
        stats=("mean", "std", "max", "min"),
    )
    for row in data:
        b.push_features(row)
    names, vals = b.rolling_features()
    expected_means = np.nanmean(data, axis=0)
    # ddof=1: the buffer matches pandas .rolling().std() used in training
    # (see test_live_parity), not numpy's ddof=0 default.
    expected_stds = np.nanstd(data, axis=0, ddof=1)
    expected_max = np.nanmax(data, axis=0)
    expected_min = np.nanmin(data, axis=0)
    np.testing.assert_allclose(vals[0:4], expected_means)
    np.testing.assert_allclose(vals[4:8], expected_stds)
    np.testing.assert_allclose(vals[8:12], expected_max)
    np.testing.assert_allclose(vals[12:16], expected_min)


def test_buffer_handles_all_nan_column_without_crash():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x", "y"], window=5)
    # x is always real; y is always NaN.
    for v in [1.0, 2.0, 3.0]:
        b.push_features({"x": v, "y": float("nan")})
    _, row = b.rolling_features()
    # x__mean = 2.0
    assert row[0] == pytest.approx(2.0)
    # y__mean over all-NaN should be NaN, not crash.
    assert np.isnan(row[1])


def test_buffer_empty_rolling_returns_nan():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x"], window=5)
    names, row = b.rolling_features()
    assert names == ["x__mean", "x__std", "x__max"]
    assert all(np.isnan(row))


def test_buffer_rejects_bad_stat():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    with pytest.raises(ValueError):
        SlidingFeatureBuffer(feature_names=["x"], stats=("not_a_stat",))


def test_buffer_rolling_dict_round_trip():
    from glider.analysis.behavior.classify.buffer import SlidingFeatureBuffer

    b = SlidingFeatureBuffer(feature_names=["x"], window=3, stats=("mean",))
    for v in [1.0, 2.0, 3.0]:
        b.push_features(np.array([v]))
    d = b.rolling_dict()
    assert d == {"x__mean": pytest.approx(2.0)}


# ---------------------------------------------------------------------------
# LatestLabel (shared state for the classifier → display link)
# ---------------------------------------------------------------------------


def test_latest_label_initial_state():
    from glider.analysis.behavior.classify.threads import LatestLabel

    ll = LatestLabel()
    idx, label, age = ll.get()
    assert idx == -1
    assert label == ""
    assert age >= 0.0


def test_latest_label_update_get():
    from glider.analysis.behavior.classify.threads import LatestLabel

    ll = LatestLabel()
    ll.update(42, "rearing")
    idx, label, age = ll.get()
    assert idx == 42
    assert label == "rearing"
    # Age should be ~0 (just set).
    assert age < 1.0


def test_latest_label_thread_safe_under_concurrent_writes():
    """Two writer threads racing — the lock should keep us from observing
    a torn (idx, label) tuple."""
    from glider.analysis.behavior.classify.threads import LatestLabel

    ll = LatestLabel()
    stop = threading.Event()

    def writer(prefix: str):
        i = 0
        while not stop.is_set():
            ll.update(i, f"{prefix}_{i}")
            i += 1

    t1 = threading.Thread(target=writer, args=("A",))
    t2 = threading.Thread(target=writer, args=("B",))
    t1.start()
    t2.start()
    for _ in range(200):
        idx, label, _ = ll.get()
        # The label always reflects whichever writer won the lock most
        # recently; the (idx, label) pair should be consistent — the
        # label string should always be one of the prefixes followed by
        # an integer matching idx.
        if label:
            prefix, _, n = label.partition("_")
            assert prefix in ("A", "B")
            assert int(n) == idx
    stop.set()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)


# ---------------------------------------------------------------------------
# _put_or_drop helper
# ---------------------------------------------------------------------------


def test_put_or_drop_drops_on_full():
    import queue

    from glider.analysis.behavior.classify.threads import _put_or_drop

    q = queue.Queue(maxsize=2)
    stop = threading.Event()
    _put_or_drop(q, 1, stop, block_timeout=0.01)
    _put_or_drop(q, 2, stop, block_timeout=0.01)
    # Third put can't fit — should be silently dropped, not raise.
    _put_or_drop(q, 3, stop, block_timeout=0.01)
    items = []
    while not q.empty():
        items.append(q.get())
    assert items == [1, 2]


def test_put_or_drop_skips_when_stop_event_set():
    import queue

    from glider.analysis.behavior.classify.threads import _put_or_drop

    q = queue.Queue(maxsize=10)
    stop = threading.Event()
    stop.set()
    _put_or_drop(q, "should not land", stop)
    assert q.empty()


# ---------------------------------------------------------------------------
# FeatureEngine — _compute_per_frame guard
# ---------------------------------------------------------------------------


def test_feature_engine_returns_none_until_history_full():
    """The engine emits the MIDDLE frame of a 5-frame history (centered
    gradients matching training), so it returns None until the history is
    full — and never lets np.gradient see too few frames."""
    import queue

    from glider.analysis.behavior.classify.threads import FeatureEngine
    from glider.analysis.behavior.features import FeatureSpec

    keypoint_names = ["snout", "neck", "tail_base"]
    spec = FeatureSpec(body_axis=(0, 2))
    eng = FeatureEngine(
        tracked_queue=queue.Queue(),
        classifier_queue=queue.Queue(),
        stop_event=threading.Event(),
        spec=spec,
        keypoint_names=keypoint_names,
        window=10,
        stats=("mean", "std", "max"),
        per_frame_feature_names=["dummy"],
        predict_every=1,
    )
    # Fewer than 5 frames -> None (and no np.gradient crash). snout↔tail is
    # always 20 apart, so the middle frame's body_length is 20.
    for t in range(4):
        eng._kp_history.append(np.array([[float(t), 0.0], [t + 10.0, 0.0], [t + 20.0, 0.0]]))
        assert eng._compute_per_frame() is None

    # 5th frame fills the history -> the middle frame's features.
    eng._kp_history.append(np.array([[4.0, 0.0], [14.0, 0.0], [24.0, 0.0]]))
    feats = eng._compute_per_frame()
    assert feats is not None
    assert "body_length" in feats
    assert feats["body_length"] == pytest.approx(20.0)
