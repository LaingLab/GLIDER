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


# ---------------------------------------------------------------------------
# Offline centred vote
#
# The live smoother is causal because a live overlay has no choice. Scoring a
# recording does: the frames after each one are already on disk. Refusing to
# read them costs real accuracy -- a causal vote lags every bout boundary by
# half its window, and on held-out data it scored below no smoothing at all on
# transition frames.
# ---------------------------------------------------------------------------


def test_centred_window_one_is_passthrough():
    from glider.analysis.behavior.classify.smoothing import centered_majority_vote

    labels = ["dig", "groom", "rear"]
    assert centered_majority_vote(labels, 1) == labels


def test_a_single_flicker_is_outvoted():
    from glider.analysis.behavior.classify.smoothing import centered_majority_vote

    labels = ["dig"] * 5 + ["groom"] + ["dig"] * 5
    assert centered_majority_vote(labels, 5) == ["dig"] * 11


def test_it_looks_forward_as_well_as_back():
    """The whole reason this exists -- a causal vote cannot do this."""
    from glider.analysis.behavior.classify.smoothing import (
        MajorityVoteSmoother,
        centered_majority_vote,
    )

    # One stray 'dig' at the very start, then a long run of 'groom'.
    labels = ["dig"] + ["groom"] * 8
    centred = centered_majority_vote(labels, 5)
    causal = [MajorityVoteSmoother(window=5).push(x) for x in labels]

    assert centred[0] == "groom", "the frames ahead should have outvoted the stray"
    assert causal[0] == "dig", "the causal smoother cannot see them (guards the premise)"


def test_a_sustained_change_survives():
    """Smoothing must not erase real behavior, only flicker."""
    from glider.analysis.behavior.classify.smoothing import centered_majority_vote

    labels = ["dig"] * 20 + ["groom"] * 20
    out = centered_majority_vote(labels, 9)
    assert out[:16] == ["dig"] * 16
    assert out[-16:] == ["groom"] * 16


def test_blanks_stay_blank():
    """'' is the model declining, not a class -- it must not be voted into one."""
    from glider.analysis.behavior.classify.smoothing import centered_majority_vote

    labels = ["dig"] * 4 + ["", ""] + ["dig"] * 4
    out = centered_majority_vote(labels, 5)
    assert out[4] == "" and out[5] == ""


def test_blanks_do_not_vote():
    """Otherwise a run of blanks would drag a real label toward nothing."""
    from glider.analysis.behavior.classify.smoothing import centered_majority_vote

    labels = ["", "", "groom", "", ""]
    assert centered_majority_vote(labels, 5) == ["", "", "groom", "", ""]


def test_it_votes_on_raw_labels_not_its_own_output():
    """Reading back smoothed neighbours lets one decision run away down the
    recording; each frame must see what the model actually said."""
    from glider.analysis.behavior.classify.smoothing import centered_majority_vote

    labels = ["a", "a", "b", "b", "b", "a", "a"]
    out = centered_majority_vote(labels, 3)
    # Recomputing from `out` instead of `labels` would propagate the first
    # decision rightward and flatten the middle run.
    assert "b" in out


def test_empty_input_is_fine():
    from glider.analysis.behavior.classify.smoothing import centered_majority_vote

    assert centered_majority_vote([], 25) == []


def test_the_default_window_is_about_one_bout():
    """Tuned to bout length, not to a test-set score -- see the constant."""
    from glider.analysis.behavior.classify.smoothing import DEFAULT_OFFLINE_WINDOW

    assert 21 <= DEFAULT_OFFLINE_WINDOW <= 31
