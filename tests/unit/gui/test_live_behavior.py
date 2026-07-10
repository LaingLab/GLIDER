"""Unit tests for the pure-logic live behavior classifier.

``LiveBehaviorClassifier`` composes the shared streaming cores
(``StreamingFeatureExtractor`` + ``extract_keypoints``), the
``SlidingFeatureBuffer`` and ``BehaviorModel.predict_one`` into a single
``classify_frame(bgr) -> LiveResult`` call, with construction-time guards
against models the live path can't serve. These tests pin the guards and
the warm-up → label lifecycle using lightweight ultralytics doubles, so
nothing here touches torch or a real YOLO net.
"""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from glider.gui.panels.live_behavior import (
    KeypointMismatchError,
    LiveBehaviorClassifier,
    LiveResult,
    UnsupportedModelError,
    model_keypoint_count,
)

# ---------------------------------------------------------------------------
# ultralytics doubles
# ---------------------------------------------------------------------------


class _Torchy:
    """Minimal stand-in for a torch tensor: indexing + .cpu().numpy() + .shape."""

    def __init__(self, a):
        self._a = np.asarray(a)

    def __getitem__(self, i):
        return _Torchy(self._a[i])

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape


class FakeKeypoints:
    def __init__(self, xy, conf=None):
        self.xy = _Torchy(np.asarray(xy)[None]) if xy is not None else _Torchy(np.zeros((0, 0, 2)))
        self.conf = _Torchy(np.asarray(conf)[None]) if conf is not None else None


class FakeResult:
    def __init__(self, kp):
        self.keypoints = kp


class FakeYolo:
    """A YOLO stand-in whose ``predict`` returns ``[result]`` (or empty detection).

    Pass ``None`` to simulate a frame where nothing was detected.
    """

    def __init__(self, xy):
        self._xy = xy

    def predict(self, bgr, conf=0.25, verbose=False):  # noqa: ARG002
        if self._xy is None:
            return [FakeResult(FakeKeypoints(None))]
        conf_arr = np.ones(len(self._xy))
        return [FakeResult(FakeKeypoints(self._xy, conf_arr))]


def _distinct_kps(k: int) -> np.ndarray:
    return np.stack([np.linspace(0.0, 10.0, k), np.linspace(0.0, 5.0, k)], axis=1)


def _model_k(model) -> int:
    return model_keypoint_count(model)


# ---------------------------------------------------------------------------
# model_keypoint_count
# ---------------------------------------------------------------------------


def test_model_keypoint_count_dedups_stats():
    class _StubModel:
        feature_names = [
            "speed_nose__mean",
            "speed_nose__std",
            "speed_nose__max",
            "speed_tail__mean",
            "speed_tail__std",
            "dist_a_b__mean",
            "body_length__mean",
        ]

    assert model_keypoint_count(_StubModel()) == 2


# ---------------------------------------------------------------------------
# construction guards
# ---------------------------------------------------------------------------


def test_mismatched_keypoint_count_raises(tiny_behavior_model):
    yolo = FakeYolo(_distinct_kps(_model_k(tiny_behavior_model)))
    with pytest.raises(KeypointMismatchError):
        LiveBehaviorClassifier(tiny_behavior_model, yolo, ["a", "b"])


def test_unstreamable_features_raise(tiny_behavior_model):
    # A model whose feature_names carry a motion_* / traj_* column can't be
    # served live — the FeatureEngine can't reproduce those in real time.
    tiny_behavior_model.feature_names = tiny_behavior_model.feature_names + ["motion_energy__mean"]
    k = _model_k(tiny_behavior_model)
    yolo = FakeYolo(_distinct_kps(k))
    with pytest.raises(UnsupportedModelError):
        LiveBehaviorClassifier(tiny_behavior_model, yolo, [f"k{i}" for i in range(k)])


def test_sequence_model_rejected(tiny_behavior_model):
    from glider.analysis.behavior.sequence import SequenceModel

    class _FakeSeq(SequenceModel):
        def __init__(self):  # bypass the real SequenceModel constructor
            pass

    seq = _FakeSeq()
    seq.feature_names = []
    yolo = FakeYolo(_distinct_kps(4))
    with pytest.raises(UnsupportedModelError):
        LiveBehaviorClassifier(seq, yolo, ["k0", "k1", "k2", "k3"])


# ---------------------------------------------------------------------------
# classify lifecycle
# ---------------------------------------------------------------------------


def test_label_emitted_after_warmup(tiny_behavior_model):
    model = tiny_behavior_model
    k = _model_k(model)
    yolo = FakeYolo(_distinct_kps(k))
    clf = LiveBehaviorClassifier(model, yolo, [f"k{i}" for i in range(k)])

    labels = []
    for _ in range(clf.window + clf.warmup + 4):
        result = clf.classify_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        assert isinstance(result, LiveResult)
        labels.append(result.label)

    assert labels[0] == ""
    assert any(lbl in set(model.classes) for lbl in labels)


def test_no_detection_empty_keypoints(tiny_behavior_model):
    model = tiny_behavior_model
    k = _model_k(model)
    yolo = FakeYolo(None)
    clf = LiveBehaviorClassifier(model, yolo, [f"k{i}" for i in range(k)])

    result = clf.classify_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.label == ""
    assert result.keypoints is not None
    assert result.keypoints.shape == (k, 2)


def test_reset_rearms_warmup(tiny_behavior_model):
    model = tiny_behavior_model
    k = _model_k(model)
    yolo = FakeYolo(_distinct_kps(k))
    clf = LiveBehaviorClassifier(model, yolo, [f"k{i}" for i in range(k)])

    # Fill up so a label can fire.
    for _ in range(clf.window + clf.warmup + 4):
        clf.classify_frame(np.zeros((8, 8, 3), dtype=np.uint8))

    clf.reset()
    # Immediately after reset the buffer is empty → blank again.
    assert clf.classify_frame(np.zeros((8, 8, 3), dtype=np.uint8)).label == ""


# ---------------------------------------------------------------------------
# BehaviorInferenceWorker (Qt worker, run on a real QThread)
# ---------------------------------------------------------------------------


class _Driver(QObject):
    """Emits into the worker's slots so calls hop threads via queued connections."""

    init = pyqtSignal(str, str, list)
    frame = pyqtSignal(object)


def _make_frame_data():
    """A ``FrameData`` carrying a throwaway BGR frame (imported lazily: heavy Qt)."""
    from glider.gui.panels.camera_panel import FrameData

    return FrameData(frame=np.zeros((8, 8, 3), dtype=np.uint8), timestamp=0.0)


def test_worker_initialize_ready_then_classifies(qtbot, tiny_behavior_model, tmp_path, monkeypatch):
    from glider.gui.panels import live_behavior as lb

    pkl = tmp_path / "m.pkl"
    tiny_behavior_model.save(pkl)
    k = model_keypoint_count(tiny_behavior_model)
    monkeypatch.setattr(lb, "_load_yolo", lambda path: FakeYolo(_distinct_kps(k)))

    worker = lb.BehaviorInferenceWorker()
    thread = QThread()
    worker.moveToThread(thread)
    thread.start()

    driver = _Driver()
    driver.init.connect(worker.initialize)
    driver.frame.connect(worker.process_frame)

    results: list[tuple[str, object]] = []
    worker.result_ready.connect(lambda label, kps: results.append((label, kps)))

    try:
        with qtbot.waitSignal(worker.ready, timeout=5000):
            driver.init.emit(str(pkl), "yolo.pt", [f"k{i}" for i in range(k)])

        n_frames = worker._classifier.window + worker._classifier.warmup + 4
        for _ in range(n_frames):
            driver.frame.emit(_make_frame_data())

        qtbot.waitUntil(lambda: len(results) >= n_frames, timeout=5000)
    finally:
        thread.quit()
        thread.wait(5000)

    assert any(label in set(tiny_behavior_model.classes) for label, _ in results)


def test_worker_bad_model_emits_load_failed(qtbot, tmp_path, monkeypatch):
    from glider.gui.panels import live_behavior as lb

    # Isolate the failure to the (missing) behavior pkl: YOLO load succeeds.
    monkeypatch.setattr(lb, "_load_yolo", lambda path: FakeYolo(_distinct_kps(4)))

    worker = lb.BehaviorInferenceWorker()
    thread = QThread()
    worker.moveToThread(thread)
    thread.start()

    driver = _Driver()
    driver.init.connect(worker.initialize)

    try:
        with qtbot.waitSignal(worker.load_failed, timeout=5000):
            driver.init.emit(str(tmp_path / "does_not_exist.pkl"), "yolo.pt", ["k0", "k1"])
    finally:
        thread.quit()
        thread.wait(5000)
