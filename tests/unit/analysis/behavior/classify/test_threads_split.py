"""Integration: the two-axis live split — a speed axis (freeze/dart heuristic)
wired alongside the posture classifier.

  * FeatureEngine, given absolute thresholds, emits a per-frame speed label.
  * BehaviorClassifier writes a two-axis ethogram (frame, behavior, speed) and
    the display label arbitrates with speed taking precedence.
"""

from __future__ import annotations

import csv
import queue
import threading

import numpy as np
import pytest

from glider.analysis.behavior.classify.threads import (
    END_OF_STREAM,
    BehaviorClassifier,
    FeatureEngine,
    LatestLabel,
)
from glider.analysis.behavior.features import FeatureSpec


def test_feature_engine_emits_freezing_on_still_keypoints():
    """A motionless mouse → causal speed ~0 → freezing once the run is long
    enough; the FeatureEngine surfaces it as the 4th tuple element."""
    spec = FeatureSpec(body_axis=(0, 2))
    names = ["snout", "neck", "tail_base"]
    cq: queue.Queue = queue.Queue()
    tq: queue.Queue = queue.Queue()
    eng = FeatureEngine(
        tracked_queue=tq,
        classifier_queue=cq,
        stop_event=threading.Event(),
        spec=spec,
        keypoint_names=names,
        window=10,
        stats=("mean", "std", "max"),
        per_frame_feature_names=["body_length"],
        predict_every=1,
        freeze_threshold=1.0,
        dart_threshold=50.0,
        freeze_min_frames=30,
        dart_min_frames=3,
    )
    eng.start()
    still = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    for i in range(60):
        tq.put((i, None, still.copy(), None))
    tq.put(END_OF_STREAM)
    eng.join(timeout=10)

    speed_labels = []
    while not cq.empty():
        item = cq.get()
        if item is END_OF_STREAM:
            continue
        assert (
            len(item) == 5
        ), "speed axis on → 5-tuple (frame, cols, row, speed label, speed px/frame)"
        speed_labels.append(item[3])
        # The numeric speed rides alongside the label so the ethogram can
        # report a value on the frames the label leaves blank.
        assert isinstance(item[4], float)
    assert "freezing" in speed_labels


def test_behavior_classifier_two_axis_ethogram_and_arbitration(tmp_path):
    class _StubModel:
        feature_names = ["a__mean"]
        classes = ["groom"]

        def predict_one(self, row, confidence_threshold=0.0, class_thresholds=None):
            return "groom"

    eth = tmp_path / "etho.csv"
    cq: queue.Queue = queue.Queue()
    latest = LatestLabel()
    clf = BehaviorClassifier(
        classifier_queue=cq,
        latest_label=latest,
        stop_event=threading.Event(),
        model=_StubModel(),
        ethogram_path=eth,
        speed_axis=True,
    )
    clf.start()
    cq.put((0, ["a__mean"], np.array([1.0]), "", 0.5))  # posture only
    cq.put((1, ["a__mean"], np.array([1.0]), "freezing", 0.1))  # speed overrides
    cq.put((2, ["a__mean"], np.array([1.0]), "darting", 9.0))
    cq.put(END_OF_STREAM)
    clf.join(timeout=5)

    rows = list(csv.reader(eth.open(newline="")))
    assert rows[0] == ["frame", "behavior", "speed_px_frame", "speed_cm_s"]
    # No pixel scale was supplied, so cm/s stays blank rather than guessed --
    # but the raw px/frame is always recorded.
    #
    # `behavior` is resolved: an animal cannot be darting and digging at once,
    # so where the speed axis fired it wins, and the classifier's own label is
    # kept beside it rather than thrown away.
    assert rows[1] == ["0", "groom", "0.5000", ""]
    assert rows[2] == ["1", "freezing", "0.1000", ""]
    assert rows[3] == ["2", "darting", "9.0000", ""]
    # Display label arbitrates: speed wins on the last frame.
    assert latest.get()[1] == "darting"


def test_ethogram_reports_cm_per_second_when_a_scale_is_known(tmp_path):
    """The whole point of loading a calibration: real units per frame."""

    class _StubModel:
        feature_names = ["a__mean"]
        classes = ["groom"]

        def predict_one(self, row, confidence_threshold=0.0, class_thresholds=None):
            return "groom"

    eth = tmp_path / "etho.csv"
    cq: queue.Queue = queue.Queue()
    clf = BehaviorClassifier(
        classifier_queue=cq,
        latest_label=LatestLabel(),
        stop_event=threading.Event(),
        model=_StubModel(),
        ethogram_path=eth,
        speed_axis=True,
        # 30 fps at 3 px/mm -> 1 px/frame is 30 px/s = 10 mm/s = 1 cm/s.
        cm_s_per_px_frame=30.0 / 3.0 / 10.0,
    )
    clf.start()
    cq.put((0, ["a__mean"], np.array([1.0]), "", 1.0))
    cq.put((1, ["a__mean"], np.array([1.0]), "darting", 4.0))
    cq.put(END_OF_STREAM)
    clf.join(timeout=5)

    rows = list(csv.reader(eth.open(newline="")))
    assert rows[1][-1] == "1.0000"
    assert rows[2][-1] == "4.0000"


def test_a_dropout_frame_reads_as_missing_not_zero(tmp_path):
    """NaN speed must not become a real 0.0 in the CSV."""

    class _StubModel:
        feature_names = ["a__mean"]
        classes = ["groom"]

        def predict_one(self, row, confidence_threshold=0.0, class_thresholds=None):
            return "groom"

    eth = tmp_path / "etho.csv"
    cq: queue.Queue = queue.Queue()
    clf = BehaviorClassifier(
        classifier_queue=cq,
        latest_label=LatestLabel(),
        stop_event=threading.Event(),
        model=_StubModel(),
        ethogram_path=eth,
        speed_axis=True,
        cm_s_per_px_frame=1.0,
    )
    clf.start()
    cq.put((0, ["a__mean"], np.array([1.0]), "", float("nan")))
    cq.put(END_OF_STREAM)
    clf.join(timeout=5)

    rows = list(csv.reader(eth.open(newline="")))
    assert rows[1] == ["0", "groom", "", ""]


def test_behavior_classifier_stays_single_axis_without_speed(tmp_path):
    """Back-compat: no speed_axis → 3-tuples, original 2-column ethogram."""

    class _StubModel:
        feature_names = ["a__mean"]
        classes = ["groom"]

        def predict_one(self, row, confidence_threshold=0.0, class_thresholds=None):
            return "groom"

    eth = tmp_path / "etho.csv"
    cq: queue.Queue = queue.Queue()
    clf = BehaviorClassifier(
        classifier_queue=cq,
        latest_label=LatestLabel(),
        stop_event=threading.Event(),
        model=_StubModel(),
        ethogram_path=eth,
    )
    clf.start()
    cq.put((0, ["a__mean"], np.array([1.0])))  # legacy 3-tuple
    cq.put(END_OF_STREAM)
    clf.join(timeout=5)
    rows = list(csv.reader(eth.open(newline="")))
    assert rows[0] == ["frame", "behavior"]
    assert rows[1] == ["0", "groom"]


def test_pose_tracker_saves_undetected_frame(tmp_path):
    """A frame with no confident detection is written to undetected_dir.

    Ported from yolo2pose's tests/test_live.py; PoseTracker's undetected-frame
    path otherwise loses coverage in the split-out classify suite.
    """
    from glider.analysis.behavior.classify.threads import PoseTracker

    out = tmp_path / "undetected"
    pt = PoseTracker(
        queue.Queue(),
        queue.Queue(),
        queue.Queue(),
        threading.Event(),
        "model.pt",
        ["a", "b"],
        undetected_dir=out,
    )
    out.mkdir(parents=True, exist_ok=True)
    pt._save_undetected(42, np.zeros((8, 8, 3), dtype=np.uint8))
    assert (out / "undetected_0000042.png").exists()
    assert pt.n_undetected_saved == 1


# --------------------------------------------------------------------------
# Poses are computed anyway; keeping them saves the next inference pass
# --------------------------------------------------------------------------


def _tracker_with_poses(tmp_path, out, rows, frame_size=None):
    """Drive PoseTracker's pose buffer directly, without loading YOLO."""
    from glider.analysis.behavior.classify.threads import PoseTracker

    tracker = PoseTracker(
        queue.Queue(),
        queue.Queue(),
        queue.Queue(),
        threading.Event(),
        "model.pt",
        ["nose", "l_ear", "r_ear"],
        pose_csv_out=out,
        fps=30.0,
    )
    tracker._pose_rows = rows
    tracker._frame_size = frame_size
    tracker._write_pose_csv()
    return tracker


def test_tracked_poses_are_written_as_a_dlc_csv(tmp_path):
    from glider.vision.pose.dlc import from_dlc_csv

    out = tmp_path / "vDLC_exp-5.csv"
    rows = [(i, np.full((3, 2), float(i)), np.ones(3)) for i in range(5)]
    _tracker_with_poses(tmp_path, out, rows)

    assert out.exists()
    pose = from_dlc_csv(out)
    assert pose.xy.shape == (5, 3, 2)
    assert pose.keypoint_names == ["nose", "l_ear", "r_ear"]
    # Each frame was filled with its own index, so the rows stay in order.
    assert [pose.xy[i][0][0] for i in range(5)] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_the_frame_size_is_recorded_with_the_poses(tmp_path):
    """The review window draws these coordinates without the video, so it
    needs the canvas they were measured on."""
    from glider.vision.pose.dlc import resolution_for_csv

    out = tmp_path / "vDLC_exp-5.csv"
    tracker = _tracker_with_poses(
        tmp_path, out, [(0, np.zeros((3, 2)), np.ones(3))], frame_size=(1280, 960)
    )
    assert tracker._frame_size == (1280, 960)
    assert resolution_for_csv(out) == (1280, 960)


def test_an_unknown_frame_size_leaves_the_sidecar_silent(tmp_path):
    """A tracker that never saw a frame must not invent a resolution."""
    from glider.vision.pose.dlc import resolution_for_csv

    out = tmp_path / "vDLC_exp-5.csv"
    _tracker_with_poses(tmp_path, out, [(0, np.zeros((3, 2)), np.ones(3))])

    assert resolution_for_csv(out) is None


def test_dropped_frames_leave_a_gap_rather_than_shifting_the_timeline(tmp_path):
    """DLC CSVs are positional; compacting would move every later timestamp."""
    from glider.vision.pose.dlc import from_dlc_csv

    out = tmp_path / "vDLC_exp-5.csv"
    # Frame 1 never arrived (producer back-pressure).
    rows = [(0, np.zeros((3, 2)), np.ones(3)), (2, np.full((3, 2), 9.0), np.ones(3))]
    _tracker_with_poses(tmp_path, out, rows)

    pose = from_dlc_csv(out)
    assert pose.xy.shape[0] == 3
    assert np.isnan(pose.xy[1]).all()
    assert pose.xy[2][0][0] == 9.0


def test_no_csv_is_written_when_none_was_asked_for(tmp_path):
    from glider.analysis.behavior.classify.threads import PoseTracker

    tracker = PoseTracker(
        queue.Queue(),
        queue.Queue(),
        queue.Queue(),
        threading.Event(),
        "model.pt",
        ["a", "b"],
    )
    tracker._pose_rows = [(0, np.zeros((2, 2)), np.ones(2))]
    tracker._write_pose_csv()
    assert list(tmp_path.glob("*.csv")) == []


def test_a_failed_save_does_not_kill_the_run(tmp_path):
    """A pose CSV is a bonus artifact; losing it must not lose the analysis."""
    out = tmp_path / "nested" / "vDLC_exp-5.csv"
    out.parent.mkdir()
    out.parent.chmod(0o500)  # best-effort read-only
    try:
        _tracker_with_poses(tmp_path, out, [(0, np.zeros((3, 2)), np.ones(3))])
    finally:
        out.parent.chmod(0o700)
    # No exception escaped; whether the write succeeded is OS-dependent.


# --------------------------------------------------------------------------
# PoseTracker.run() loads through the pose-backend seam
# --------------------------------------------------------------------------


class _StubBackend:
    """A PoseBackend double recording what it was asked to predict."""

    def __init__(self, names, kps, confs):
        self.keypoint_names = list(names)
        self.native_keypoint_count = len(names)
        self._kps = kps
        self._confs = confs
        self.frames = []

    def predict(self, bgr):
        self.frames.append(bgr)
        return self._kps, self._confs

    def close(self):
        pass


def _run_tracker(monkeypatch, backend, *, names=("a", "b"), device=None, frames=1):
    """Drive PoseTracker.run() synchronously over `frames` frames, then EOS."""
    from glider.analysis.behavior.classify import threads as th
    from glider.vision.pose import backend as backend_mod

    seen = {}

    def _fake_load(path, keypoint_names=None, conf_threshold=0.25, dev=None):
        seen["path"] = path
        seen["names"] = keypoint_names
        seen["conf"] = conf_threshold
        seen["device"] = dev
        return backend

    monkeypatch.setattr(backend_mod, "load_pose_backend", _fake_load)

    raw, tracked, display = queue.Queue(), queue.Queue(), queue.Queue()
    for i in range(frames):
        raw.put((i, np.zeros((6, 8, 3), dtype=np.uint8)))
    raw.put(th.END_OF_STREAM)

    tracker = th.PoseTracker(
        raw,
        tracked,
        display,
        threading.Event(),
        "model.pt",
        list(names),
        conf_threshold=0.4,
        device=device,
    )
    tracker.run()
    return tracker, tracked, display, seen


def test_pose_tracker_run_uses_the_backend_seam(monkeypatch):
    kps = np.array([[1.0, 2.0], [3.0, 4.0]])
    confs = np.array([0.9, 0.8])
    backend = _StubBackend(["a", "b"], kps, confs)

    tracker, tracked, display, seen = _run_tracker(monkeypatch, backend)

    assert seen["path"] == "model.pt"
    assert seen["conf"] == pytest.approx(0.4)
    assert len(backend.frames) == 1

    frame_idx, _, out_kps, out_confs = tracked.get_nowait()
    assert frame_idx == 0
    assert out_kps == pytest.approx(kps)
    assert out_confs == pytest.approx(confs)
    # Both queues get the same payload, then the end-of-stream sentinel.
    assert display.get_nowait()[0] == 0
    assert tracked.get_nowait() is END_OF_STREAM


def test_pose_tracker_run_forwards_the_device(monkeypatch):
    backend = _StubBackend(["a", "b"], np.zeros((2, 2)), np.zeros(2))
    _, _, _, seen = _run_tracker(monkeypatch, backend, device="cuda:1")
    assert seen["device"] == "cuda:1"


def test_pose_tracker_run_adopts_the_backends_names(monkeypatch):
    """DLC/SLEAP carry their own names; the tracker must take them over."""
    backend = _StubBackend(["snout", "tailbase"], np.zeros((2, 2)), np.zeros(2))
    tracker, _, _, _ = _run_tracker(monkeypatch, backend, names=("wrong", "names"))
    assert tracker.keypoint_names == ["snout", "tailbase"]


def test_pose_tracker_run_reports_a_load_failure(monkeypatch):
    from glider.analysis.behavior.classify import threads as th
    from glider.vision.pose import backend as backend_mod

    def _boom(*a, **kw):
        raise RuntimeError("no such model")

    monkeypatch.setattr(backend_mod, "load_pose_backend", _boom)

    raw, tracked, display = queue.Queue(), queue.Queue(), queue.Queue()
    tracker = th.PoseTracker(raw, tracked, display, threading.Event(), "missing.pt", ["a", "b"])
    tracker.run()

    assert "failed to load pose model" in tracker.error
    assert tracked.get_nowait() is END_OF_STREAM
