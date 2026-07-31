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
    assert rows[0] == ["frame", "behavior", "speed", "speed_px_frame", "speed_cm_s"]
    # No pixel scale was supplied, so cm/s stays blank rather than guessed --
    # but the raw px/frame is always recorded.
    assert rows[1] == ["0", "groom", "", "0.5000", ""]
    assert rows[2] == ["1", "groom", "freezing", "0.1000", ""]
    assert rows[3] == ["2", "groom", "darting", "9.0000", ""]
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
    assert rows[1] == ["0", "groom", "", "", ""]


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
