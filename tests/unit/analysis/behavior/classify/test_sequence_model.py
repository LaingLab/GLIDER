"""Tests for the deployable CNN model wrapper (SequenceModel) + its
training entry point. Covers the predict contract, persistence, offline
per-frame scoring, and end-to-end training on a synthetic session.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from glider.vision.pose.core import PoseData  # noqa: E402


def _toy_pose(n_frames=400, k=7, seed=0) -> PoseData:
    """A pose with two distinguishable regimes: still vs translating."""
    rng = np.random.default_rng(seed)
    xy = np.zeros((n_frames, k, 2), dtype=np.float64)
    base = np.stack([np.linspace(-1, 1, k), np.zeros(k)], axis=1)  # body along x
    for f in range(n_frames):
        xy[f] = base + rng.normal(0, 0.01, size=(k, 2))
        if f >= n_frames // 2:  # second half: drift in +x (locomotion)
            xy[f, :, 0] += (f - n_frames // 2) * 0.1
    conf = np.ones((n_frames, k))
    return PoseData(xy=xy, confidence=conf, keypoint_names=[f"k{i}" for i in range(k)], fps=30.0)


def _build_model(classes=("a", "b", "c"), window=30, k=7, n_models=1):
    from glider.analysis.behavior.sequence import SequenceModel, _make_cnn_class

    net_cls = _make_cnn_class()
    nets = [net_cls(n_channels=k * 2, n_classes=len(classes)) for _ in range(n_models)]
    return SequenceModel(
        modules=nets,
        classes=list(classes),
        window=window,
        body_axis=(0, k - 1),
        fps=30.0,
        arch={"n_channels": k * 2, "n_classes": len(classes), "hidden": 64, "dropout": 0.3},
    )


def test_predict_window_returns_a_known_class():
    model = _build_model()
    win = np.zeros((30, 7, 2)) + np.stack([np.linspace(-1, 1, 7), np.zeros(7)], axis=1)
    label = model.predict_window(win)
    assert label in {"a", "b", "c"}


def test_predict_window_interpolates_gaps_like_training():
    """Training interpolates keypoint gaps before windowing, so inference
    must too — otherwise a single dropped keypoint (common during digs,
    when keypoints occlude) blanks the whole window. A brief interior gap
    must NOT blank, and must match the interpolated-window prediction."""
    from glider.analysis.behavior.sequence import _interpolate_xy

    model = _build_model(window=10, k=7)
    win = (
        np.stack([np.stack([np.linspace(-1, 1, 7), np.zeros(7)], axis=1)] * 10)
        + np.linspace(0, 1, 10)[:, None, None]
    )
    holed = win.copy()
    holed[5, 2, :] = np.nan  # one keypoint missing for one mid-window frame

    assert model.predict_window(holed) != ""  # gap filled, not blanked
    assert model.predict_window(holed) == model.predict_window(_interpolate_xy(holed))


def test_predict_window_blank_when_keypoint_missing_whole_window():
    """If a keypoint is absent for the ENTIRE window, interpolation can't
    fill it → blank (the honest 'unknown')."""
    model = _build_model(window=10, k=7)
    win = np.zeros((10, 7, 2))
    win[:, 2, :] = np.nan  # keypoint 2 gone for every frame
    assert model.predict_window(win) == ""


def test_save_load_roundtrip_preserves_predictions(tmp_path):
    from glider.analysis.behavior.sequence import SequenceModel

    model = _build_model()
    win = (
        np.stack([np.stack([np.linspace(-1, 1, 7), np.zeros(7)], axis=1)] * 30)
        + np.linspace(0, 1, 30)[:, None, None]
    )
    before = model.predict_window(win)

    path = tmp_path / "cnn.pt"
    model.save(path)
    reloaded = SequenceModel.load(path)

    assert reloaded.classes.tolist() == model.classes.tolist()
    assert reloaded.window == model.window
    assert tuple(reloaded.body_axis) == tuple(model.body_axis)
    assert reloaded.predict_window(win) == before


def test_score_pose_aligns_and_blanks_warmup():
    model = _build_model(window=30)
    pose = _toy_pose(n_frames=120)
    labels = model.score_pose(pose)
    assert len(labels) == 120
    # First window-1 frames have no full trailing window → blank.
    assert all(lbl == "" for lbl in labels[:29])
    assert all(lbl in {"a", "b", "c"} for lbl in labels[29:])


def test_ensemble_model_saves_and_loads_all_members(tmp_path):
    """A 5-net ensemble round-trips: all members persist and predictions
    are identical after reload."""
    from glider.analysis.behavior.sequence import SequenceModel

    model = _build_model(n_models=5)
    assert len(model.modules) == 5
    win = (
        np.stack([np.stack([np.linspace(-1, 1, 7), np.zeros(7)], axis=1)] * 30)
        + np.linspace(0, 1, 30)[:, None, None]
    )
    before = model.predict_window(win)

    path = tmp_path / "ens.pt"
    model.save(path)
    reloaded = SequenceModel.load(path)
    assert len(reloaded.modules) == 5
    assert reloaded.predict_window(win) == before


def test_live_loader_dispatches_cnn_vs_tabular(tmp_path):
    """The live pipeline's model loader returns a SequenceModel for a CNN
    bundle and a BehaviorModel for a tabular bundle."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    from glider.analysis.behavior import BehaviorModel, FeatureSpec
    from glider.analysis.behavior.classify.pipeline import _load_behavior_model
    from glider.analysis.behavior.sequence import SequenceModel

    # CNN bundle.
    cnn = _build_model()
    cnn_path = tmp_path / "cnn.pt"
    cnn.save(cnn_path)
    assert isinstance(_load_behavior_model(cnn_path), SequenceModel)

    # Tabular bundle.
    x_df = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], columns=["a", "b"])
    clf = RandomForestClassifier(n_estimators=3, random_state=0).fit(x_df, ["x", "y"])
    tab = BehaviorModel(
        classifier=clf,
        feature_names=["a", "b"],
        spec=FeatureSpec(),
        window=10,
        stats=("mean",),
        fps=30.0,
        classes=["x", "y"],
    )
    tab_path = tmp_path / "tab.pkl"
    tab.save(tab_path)
    assert isinstance(_load_behavior_model(tab_path), BehaviorModel)


def test_live_pipeline_builds_sequence_threads_for_cnn(tmp_path):
    """Constructing the live pipeline with a CNN model takes the sequence
    path (SequenceClassifier, no feature engine/classifier/embedding).
    YOLO + video are loaded lazily, so this builds without either."""
    from glider.analysis.behavior.classify.pipeline import (
        LiveInferenceConfig,
        LiveInferencePipeline,
    )

    model = _build_model(k=7)
    path = tmp_path / "cnn.pt"
    model.save(path)

    cfg = LiveInferenceConfig(
        source="nonexistent.mp4",
        keypoint_names=[f"k{i}" for i in range(7)],
        yolo_model_path="dummy.pt",
        behavior_model_path=path,
        display=False,
        ethogram_csv=tmp_path / "etho.csv",
    )
    pipe = LiveInferencePipeline(cfg)
    assert pipe.is_sequence is True
    assert pipe.seq_classifier is not None
    assert pipe.feature_engine is None
    assert pipe.classifier is None
    assert pipe.embedding_active is False


def test_movement_gate_relabels_stationary_locomote():
    """A frame called 'locomote' while the body barely moved is relabeled
    to the model's runner-up; a frame with real translation is kept."""
    from glider.analysis.behavior.sequence import MovementGate

    classes = np.array(["dig", "grooming", "locomote", "sniff_rear"])
    # proba favors locomote, runner-up sniff_rear.
    proba = np.array([0.05, 0.10, 0.55, 0.30])
    gate = MovementGate(gate_class="locomote", min_displacement=0.5)
    body_axis = (0, 6)

    w, k = 30, 7
    # Stationary window: keypoints jitter in place → tiny net displacement.
    rng = np.random.default_rng(0)
    still = np.tile(np.stack([np.linspace(-1, 1, k), np.zeros(k)], axis=1), (w, 1, 1)) + rng.normal(
        0, 0.01, size=(w, k, 2)
    )
    assert gate.relabel(proba, classes, still, body_axis) == "sniff_rear"

    # Moving window: body center translates ~3 body-lengths over the window.
    moving = still.copy()
    moving[:, :, 0] += np.linspace(0, 4, w)[:, None]
    assert gate.relabel(proba, classes, moving, body_axis) == "locomote"


def test_movement_gate_ignores_non_gated_classes():
    """A non-locomote prediction is never touched, regardless of movement."""
    from glider.analysis.behavior.sequence import MovementGate

    classes = np.array(["dig", "grooming", "locomote", "sniff_rear"])
    proba = np.array([0.1, 0.6, 0.2, 0.1])  # grooming wins
    gate = MovementGate(gate_class="locomote", min_displacement=0.5)
    still = np.zeros((30, 7, 2)) + np.stack([np.linspace(-1, 1, 7), np.zeros(7)], axis=1)
    assert gate.relabel(proba, classes, still, (0, 6)) == "grooming"


def test_predict_window_applies_movement_gate():
    from glider.analysis.behavior.sequence import MovementGate

    model = _build_model(classes=("locomote", "sniff_rear"), window=10, k=7)
    gate = MovementGate(gate_class="locomote", min_displacement=0.5)
    still = np.zeros((10, 7, 2)) + np.stack([np.linspace(-1, 1, 7), np.zeros(7)], axis=1)
    # Whatever the net predicts, a stationary 'locomote' must not survive.
    label = model.predict_window(still, gate=gate)
    assert label in {"locomote", "sniff_rear"}
    # Force the gate path deterministically: if it predicted locomote, the
    # stationary window means it should have been relabeled.
    raw = model.predict_window(still)
    if raw == "locomote":
        assert label == "sniff_rear"


def test_train_sequence_model_end_to_end(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.features import FeatureSpec
    from glider.analysis.behavior.sequence import SequenceModel, train_sequence_model
    from glider.vision.pose.dlc import to_dlc_csv

    def write_session(pose, stem):
        pc = tmp_path / f"{stem}.csv"
        ac = tmp_path / f"{stem}_annotations.csv"
        to_dlc_csv(pose, pc)
        store = AnnotationStore()
        # still first half = "still", drifting second half = "move"
        store.add(BehaviorZone(behavior="still", start_frame=0, end_frame=199))
        store.add(BehaviorZone(behavior="move", start_frame=200, end_frame=399))
        store.save_csv(ac)
        return (pc, ac)

    train = [write_session(_toy_pose(seed=i), f"s{i}") for i in range(2)]
    holdout = [write_session(_toy_pose(seed=99), "hold")]

    model, summary = train_sequence_model(
        train,
        spec=FeatureSpec(),
        window=20,
        fps=30.0,
        holdout_sessions=holdout,
        max_epochs=40,
        seed=0,
        n_ensemble=2,
    )
    assert isinstance(model, SequenceModel)
    assert len(model.modules) == 2  # ensemble path
    assert set(model.classes.tolist()) == {"still", "move"}
    # On this trivially separable toy task, holdout accuracy should be high.
    assert summary["holdout_accuracy"] > 0.8

    # And it scores a fresh pose end to end.
    labels = model.score_pose(_toy_pose(seed=7))
    assert {lbl for lbl in labels if lbl} <= {"still", "move"}
