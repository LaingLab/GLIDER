"""Scoring a saved model against annotated sessions.

The metric assembly is tested on hand-built label sequences so the numbers are
exact rather than plausible; the loading and feature-rebuilding path is tested
end-to-end against a model this suite trains itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.evaluation import (
    evaluate_model,
    summarise_predictions,
)

pytest.importorskip("sklearn")

NAMES = ["snout", "neck", "tail_base"]


# ---------------------------------------------------------------------------
# The pure metric assembly
# ---------------------------------------------------------------------------


def _flat(true, pred, sess=None):
    """One session of consecutive frames unless *sess* says otherwise."""
    true = np.asarray(true, dtype=object)
    pred = np.asarray(pred, dtype=object)
    sess = np.zeros(len(true), dtype=int) if sess is None else np.asarray(sess)
    return true, pred, sess, np.arange(len(true), dtype=int)


class TestSummarise:
    def test_per_class_numbers_are_exact(self):
        # walk: 3 true, 2 predicted correctly -> recall 2/3, precision 2/2
        # rear: 2 true, 1 predicted correctly -> recall 1/2, precision 1/2
        true = ["walk", "walk", "walk", "rear", "rear"]
        pred = ["walk", "walk", "rear", "rear", "walk"]
        result = summarise_predictions(*_flat(true, pred), support_floor=1)

        walk = result["per_class"]["walk"]
        assert walk["recall"] == pytest.approx(2 / 3)
        assert walk["precision"] == pytest.approx(2 / 3)
        assert walk["support"] == 3

        rear = result["per_class"]["rear"]
        assert rear["recall"] == pytest.approx(1 / 2)
        assert rear["precision"] == pytest.approx(1 / 2)
        assert rear["support"] == 2

    def test_accuracy_counts_every_scored_frame(self):
        true = ["a", "a", "b", "b"]
        pred = ["a", "b", "b", "b"]
        result = summarise_predictions(*_flat(true, pred), support_floor=1)
        assert result["accuracy"] == pytest.approx(0.75)

    def test_macro_f1_is_the_unweighted_mean_of_scored_classes(self):
        true = ["a", "a", "b", "b"]
        pred = ["a", "a", "b", "b"]
        result = summarise_predictions(*_flat(true, pred), support_floor=1)
        assert result["macro_f1"] == pytest.approx(1.0)
        assert result["macro_classes"] == ["a", "b"]

    def test_a_thin_class_is_reported_but_left_out_of_the_macro(self):
        """One 2-frame bout must not weigh as much as a 100-frame class.

        This is the cross-validation lesson: a rare class scored as an equal
        share of the macro average measures its luck, not the model.
        """
        true = ["a"] * 100 + ["b"] * 100 + ["rare", "rare"]
        pred = ["a"] * 100 + ["b"] * 100 + ["a", "a"]
        result = summarise_predictions(*_flat(true, pred), support_floor=100)

        assert "rare" in result["per_class"], "a thin class is still reported"
        assert result["per_class"]["rare"]["thin"] is True
        assert result["thin_classes"] == ["rare"]
        assert "rare" not in result["macro_classes"]
        # 'a' and 'b' are both perfect on their own frames; only 'a' is dented
        # by the two stray predictions, and 'rare' must not drag the average.
        assert result["macro_f1"] > 0.97

    def test_a_class_the_model_never_sees_still_appears(self):
        """A behaviour with no predictions is a finding, not an omission."""
        true = ["a"] * 100 + ["ghost"] * 100
        pred = ["a"] * 200
        result = summarise_predictions(*_flat(true, pred), support_floor=1)

        assert result["per_class"]["ghost"]["recall"] == pytest.approx(0.0)
        assert result["per_class"]["ghost"]["support"] == 100

    def test_a_class_only_predicted_is_charged_against_precision(self):
        """False alarms for an unannotated behaviour are not free."""
        true = ["a"] * 10
        pred = ["a"] * 8 + ["phantom", "phantom"]
        result = summarise_predictions(*_flat(true, pred), support_floor=1)

        assert result["per_class"]["phantom"]["support"] == 0
        assert result["per_class"]["phantom"]["precision"] == pytest.approx(0.0)

    def test_the_confusion_matrix_is_square_over_every_label(self):
        true = ["a", "a", "b"]
        pred = ["a", "b", "b"]
        cm = summarise_predictions(*_flat(true, pred), support_floor=1)["confusion"]
        assert cm["labels"] == ["a", "b"]
        assert cm["matrix"] == [[1, 1], [0, 1]]

    def test_bouts_are_counted_per_session_not_across_them(self):
        """Two sessions each ending and starting in 'a' is two bouts, not one."""
        true = ["a"] * 4 + ["a"] * 4
        pred = ["a"] * 8
        sess = np.array([0] * 4 + [1] * 4)
        result = summarise_predictions(*_flat(true, pred, sess), support_floor=1)
        assert result["bouts"]["a"]["n_bouts"] == 2

    def test_no_scored_frames_is_reported_not_raised(self):
        result = summarise_predictions(*_flat([], []), support_floor=1)
        assert result["per_class"] == {}
        assert result["macro_f1"] is None


# ---------------------------------------------------------------------------
# Loading a bundle and rebuilding its features
# ---------------------------------------------------------------------------


def _write_session(tmp_path, stem, *, seed, labels=("walk", "rear")):
    """A pose CSV plus annotations covering most of it."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    rng = np.random.default_rng(seed)
    n = 400
    xy = rng.normal(300, 20, size=(n, len(NAMES), 2))
    # Give the two behaviours genuinely different geometry so a model can
    # separate them; otherwise the end-to-end test measures noise.
    xy[n // 2 :, 0, :] += 120
    pose = PoseData(
        xy=xy,
        confidence=np.ones((n, len(NAMES))),
        keypoint_names=NAMES,
        fps=30.0,
    )
    pose_csv = tmp_path / f"{stem}.csv"
    to_dlc_csv(pose, pose_csv)

    store = AnnotationStore()
    store.add(BehaviorZone(behavior=labels[0], start_frame=20, end_frame=n // 2 - 1))
    store.add(BehaviorZone(behavior=labels[1], start_frame=n // 2, end_frame=n - 1))
    ann_csv = tmp_path / f"{stem}_annotations.csv"
    store.save_csv(ann_csv)
    return (pose_csv, ann_csv)


@pytest.fixture
def trained_model(tmp_path):
    """A saved bundle plus the sessions it was trained on."""
    from glider.analysis.behavior.pipeline import train_model

    sessions = [_write_session(tmp_path, f"train{i}", seed=i) for i in range(2)]
    result = train_model(sessions, window=10, classifier_type="rf", n_estimators=20)
    path = tmp_path / "model.pkl"
    result.model.save(path)
    return path, sessions


class TestEvaluateModel:
    def test_it_scores_a_saved_bundle_against_sessions(self, trained_model):
        path, sessions = trained_model
        result = evaluate_model(path, sessions, support_floor=1)

        assert set(result["per_class"]) >= {"walk", "rear"}
        assert result["accuracy"] > 0.8, "a model should recognise its own training data"
        assert result["n_scored"] > 0

    def test_frames_the_window_has_not_filled_are_counted_not_scored(self, trained_model):
        """Warm-up rows predict "" and must be declared, not hidden."""
        path, sessions = trained_model
        result = evaluate_model(path, sessions, support_floor=1)
        assert result["n_unscored"] >= 0
        assert "" not in result["per_class"]

    def test_the_model_and_session_provenance_is_recorded(self, trained_model):
        path, sessions = trained_model
        result = evaluate_model(path, sessions, support_floor=1)
        assert result["model_path"] == str(path)
        assert len(result["sessions"]) == len(sessions)
        assert result["window"] == 10

    def test_sessions_without_annotations_are_refused_clearly(self, tmp_path, trained_model):
        from glider.analysis.behavior.annotations import AnnotationStore

        path, _sessions = trained_model
        bare = _write_session(tmp_path, "bare", seed=99)
        AnnotationStore().save_csv(bare[1])  # valid file, no zones

        with pytest.raises(ValueError, match="no annotated frames"):
            evaluate_model(path, [bare], support_floor=1)

    def test_an_empty_session_list_is_refused(self, trained_model):
        path, _sessions = trained_model
        with pytest.raises(ValueError, match="at least one session"):
            evaluate_model(path, [], support_floor=1)
