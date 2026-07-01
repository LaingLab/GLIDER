"""Tests for the live-inference majority-vote label smoother."""

from __future__ import annotations


def test_window_one_is_passthrough():
    from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

    s = MajorityVoteSmoother(window=1)
    assert [s.push(x) for x in ("dig", "groom", "rear")] == ["dig", "groom", "rear"]


def test_brief_blip_is_outvoted():
    from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

    s = MajorityVoteSmoother(window=5)
    for _ in range(4):
        s.push("dig")
    # A single off-label among four 'dig' stays 'dig'.
    assert s.push("groom") == "dig"


def test_sustained_change_is_adopted():
    from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

    s = MajorityVoteSmoother(window=5)
    for _ in range(5):
        s.push("dig")
    # Once 'groom' becomes the majority of the window it takes over.
    out = [s.push("groom") for _ in range(5)]
    assert out[-1] == "groom"
    assert "groom" in out and out.index("groom") <= 3  # adopted within ~window/2


def test_tie_keeps_committed_label_hysteresis():
    from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

    s = MajorityVoteSmoother(window=4)
    s.push("dig")
    s.push("dig")  # committed = dig
    # Now 2 dig / 2 groom — a tie; hysteresis keeps the committed 'dig'.
    s.push("groom")
    assert s.push("groom") == "dig"


def test_unknown_run_commits_to_unknown():
    from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother

    s = MajorityVoteSmoother(window=3)
    s.push("dig")
    s.push("unknown")
    assert s.push("unknown") == "unknown"


def test_classifier_writes_smoothed_ethogram(tmp_path):
    """BehaviorClassifier with smooth_window>1 votes raw predictions into a
    stable ethogram — a one-frame blip is outvoted."""
    import queue
    import threading

    import numpy as np

    from glider.analysis.behavior.classify.threads import (
        END_OF_STREAM,
        BehaviorClassifier,
        LatestLabel,
    )

    class FakeModel:
        feature_names = ["f__mean"]

        def __init__(self, labels):
            self._labels = iter(labels)

        def predict_one(self, row, confidence_threshold=0.0, class_thresholds=None):
            return next(self._labels)

    raw = ["dig", "dig", "groom", "dig", "dig"]  # single 'groom' blip
    q: queue.Queue = queue.Queue()
    model = FakeModel(raw)
    eth = tmp_path / "ethogram.csv"
    clf = BehaviorClassifier(
        classifier_queue=q,
        latest_label=LatestLabel(),
        stop_event=threading.Event(),
        model=model,
        ethogram_path=eth,
        smooth_window=3,
    )
    for i in range(len(raw)):
        q.put((i, model.feature_names, np.array([0.0])))
    q.put(END_OF_STREAM)
    clf.run()  # processes the queue synchronously, writes the CSV at EOS

    text = eth.read_text()
    assert "dig" in text
    assert "groom" not in text  # the blip was smoothed away
