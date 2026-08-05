"""Reading a training run's report back, and what it says about the run.

The interpretive rules live here rather than in the widget, so the thing that
decides whether a score is trustworthy is tested without a display.
"""

from __future__ import annotations

import json

import pytest

from glider.analysis.behavior.run_report import RunReportError, TrainingRun

CLASSES = ["groom", "locomote", "rear"]


def _summary(**overrides):
    base = {
        "n_sessions": 4,
        "n_rows_kept": 10_000,
        "kept_label_counts": {"locomote": 6_000, "groom": 3_000, "rear": 1_000},
        "test_size": 2_000,
        "train_accuracy": 0.99,
        "test_accuracy": 0.81,
        "split_strategy": "cross_session",
        "n_holdout_sessions": 2,
        "classifier_type": "lightgbm",
        "per_class_metrics": {
            "groom": {"precision": 0.9, "recall": 0.88, "f1": 0.89, "support": 900},
            "locomote": {"precision": 0.95, "recall": 0.96, "f1": 0.95, "support": 800},
            "rear": {"precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 300},
        },
        "confusion_matrix": {
            "labels": CLASSES,
            "matrix": [[80, 10, 10], [5, 90, 5], [30, 30, 40]],
        },
        "top_features": [
            {"feature": "speed_mean", "importance": 0.4},
            {"feature": "area_std", "importance": 0.1},
        ],
        "window": 30,
        "fps": 30.0,
        "n_features": 400,
    }
    base.update(overrides)
    return base


def _write(tmp_path, summary, name="run_report"):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps(summary))
    return folder


class TestLoading:
    def test_loads_a_report_folder(self, tmp_path):
        run = TrainingRun.load(_write(tmp_path, _summary()))
        assert run.summary["classifier_type"] == "lightgbm"
        assert run.path is not None

    def test_accepts_the_summary_file_itself(self, tmp_path):
        """A file dialog makes picking summary.json at least as likely as its folder."""
        folder = _write(tmp_path, _summary())
        assert TrainingRun.load(folder / "summary.json").summary["n_sessions"] == 4

    def test_a_folder_with_no_summary_says_where_to_look(self, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(RunReportError, match="next to the model bundle"):
            TrainingRun.load(empty)

    def test_unreadable_json_is_reported_not_raised_bare(self, tmp_path):
        folder = tmp_path / "broken"
        folder.mkdir()
        (folder / "summary.json").write_text("{not json")
        with pytest.raises(RunReportError, match="Could not read"):
            TrainingRun.load(folder)

    def test_charts_are_listed_when_present(self, tmp_path):
        folder = _write(tmp_path, _summary())
        (folder / "confusion.png").write_bytes(b"")
        assert [p.name for p in TrainingRun.load(folder).charts] == ["confusion.png"]

    def test_an_in_memory_summary_needs_no_folder(self, tmp_path):
        """A run whose report could not be written is still reviewable."""
        run = TrainingRun.from_summary(_summary())
        assert run.path is None
        assert run.headline.value == pytest.approx(0.81)


class TestHeadline:
    def test_a_cross_session_holdout_is_trustworthy(self):
        head = TrainingRun.from_summary(_summary()).headline
        assert head.value == pytest.approx(0.81)
        assert head.label == "Test accuracy"
        assert head.strength == "strong"
        assert head.is_trustworthy

    def test_no_holdout_falls_back_to_train_accuracy_and_says_so(self):
        """The most misleading number this pipeline can print must be labelled."""
        run = TrainingRun.from_summary(
            _summary(
                split_strategy="no_holdout",
                test_size=0,
                test_accuracy=None,
                n_holdout_sessions=0,
            )
        )
        head = run.headline
        assert head.value == pytest.approx(0.99)
        assert head.label == "Train accuracy"
        assert not head.is_trustworthy
        assert "fitted to" in head.caption

    def test_a_within_session_split_is_weaker_than_cross_session(self):
        head = TrainingRun.from_summary(_summary(split_strategy="group_shuffle")).headline
        assert head.strength == "fair"
        assert "same animals" in head.caption

    def test_cross_validation_leads_with_the_mean_and_its_spread(self):
        run = TrainingRun.from_summary(
            {
                "split_strategy": "cross_validated",
                "mean_macro_f1": 0.72,
                "fold_macro_f1": [0.60, 0.72, 0.84],
                "n_sessions": 6,
            }
        )
        head = run.headline
        assert head.value == pytest.approx(0.72)
        assert head.spread == pytest.approx(0.098, abs=0.005)
        assert head.is_trustworthy

    def test_a_missing_accuracy_does_not_crash_the_headline(self):
        head = TrainingRun.from_summary({"split_strategy": "no_holdout"}).headline
        assert head.value is None


class TestPerClass:
    def test_rows_come_back_worst_first(self):
        """The reason to open the table is to find the failure."""
        names = [c.name for c in TrainingRun.from_summary(_summary()).per_class]
        assert names[0] == "rear"
        assert names[-1] == "locomote"

    def test_the_weakest_class_is_the_lowest_f1(self):
        weakest = TrainingRun.from_summary(_summary()).weakest
        assert weakest.name == "rear"
        assert weakest.f1 == pytest.approx(0.44)

    def test_macro_f1_is_unweighted(self):
        """A support-weighted mean is exactly what hides a failing rare class."""
        value = TrainingRun.from_summary(_summary()).macro_f1
        assert value == pytest.approx((0.89 + 0.95 + 0.44) / 3)

    def test_no_metrics_means_no_rows_rather_than_a_crash(self):
        run = TrainingRun.from_summary(_summary(per_class_metrics={}))
        assert run.per_class == []
        assert run.weakest is None
        assert run.macro_f1 is None

    def test_cross_validated_macro_f1_pools_the_classes(self):
        """Cross-validation must not swap this for the mean-of-folds figure.

        They answer different questions, and they diverge precisely when a
        class is missing from some fold's test set: that class scores 0 in
        those folds, so the mean-of-folds drops while the pooled per-class
        F1 -- computed once over every fold's predictions -- does not. The
        headline already reports mean-of-folds and labels it as such; this
        property is the other number, and reporting mean-of-folds here made
        the two tiles duplicate each other and understate a good model.
        """
        run = TrainingRun.from_summary(
            _summary(split_strategy="cross_validated", mean_macro_f1=0.684)
        )
        assert run.is_cross_validated
        assert run.macro_f1 == pytest.approx((0.89 + 0.95 + 0.44) / 3)

    def test_scored_rows_counts_what_was_measured_not_what_was_trained(self):
        """``n_rows_kept`` includes mirror-augmented copies, which never score."""
        run = TrainingRun.from_summary(
            _summary(split_strategy="cross_validated", n_rows_kept=10_000)
        )
        assert run.scored_rows == 900 + 800 + 300

    def test_scored_rows_is_none_without_a_per_class_table(self):
        assert TrainingRun.from_summary(_summary(per_class_metrics={})).scored_rows is None

    def test_cross_validated_macro_f1_falls_back_when_unmeasured(self):
        """No per-class table, no pooled mean -- keep the only number there is."""
        run = TrainingRun.from_summary(
            _summary(
                split_strategy="cross_validated",
                mean_macro_f1=0.684,
                per_class_metrics={},
            )
        )
        assert run.macro_f1 == pytest.approx(0.684)


class TestConfusion:
    def test_rows_are_normalized_to_fractions(self):
        labels, matrix = TrainingRun.from_summary(_summary()).confusion_rows_normalized()
        assert labels == CLASSES
        assert matrix[0] == pytest.approx([0.8, 0.1, 0.1])
        for row in matrix:
            assert sum(row) == pytest.approx(1.0)

    def test_an_empty_row_does_not_divide_by_zero(self):
        run = TrainingRun.from_summary(
            _summary(confusion_matrix={"labels": ["a", "b"], "matrix": [[0, 0], [1, 1]]})
        )
        _labels, matrix = run.confusion_rows_normalized()
        assert matrix[0] == [0.0, 0.0]

    def test_a_mismatched_matrix_is_refused_rather_than_drawn_wrong(self):
        run = TrainingRun.from_summary(
            _summary(confusion_matrix={"labels": ["a", "b", "c"], "matrix": [[1, 2, 3]]})
        )
        assert run.confusion == ([], [])


class TestWarnings:
    def test_no_holdout_is_warned_about(self):
        run = TrainingRun.from_summary(
            _summary(split_strategy="no_holdout", test_size=0, test_accuracy=None)
        )
        assert any("No held-out data" in w for w in run.warnings)

    def test_a_balanced_well_scoring_run_warns_about_nothing(self):
        run = TrainingRun.from_summary(
            _summary(
                kept_label_counts={"a": 1000, "b": 1100, "c": 900},
                per_class_metrics={
                    "a": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 100},
                },
            )
        )
        assert run.warnings == []

    def test_class_imbalance_is_reported_with_the_ratio(self):
        run = TrainingRun.from_summary(_summary(kept_label_counts={"common": 50_000, "rare": 500}))
        assert run.imbalance_ratio == pytest.approx(100.0)
        assert any("100:1" in w for w in run.warnings)

    def test_a_collapsed_class_is_called_out_by_name(self):
        run = TrainingRun.from_summary(_summary())  # 'rear' sits at F1 0.44
        assert any("'rear'" in w and "0.44" in w for w in run.warnings)

    def test_a_wide_cross_validation_spread_is_warned_about(self):
        """A spread wider than most between-setting differences is a trap."""
        run = TrainingRun.from_summary(
            {
                "split_strategy": "cross_validated",
                "mean_macro_f1": 0.7,
                "fold_macro_f1": [0.5, 0.7, 0.9],
            }
        )
        assert any("spread" in w for w in run.warnings)


class TestBalanceAndFeatures:
    def test_label_counts_come_back_largest_first(self):
        counts = TrainingRun.from_summary(_summary()).label_counts
        assert [c[0] for c in counts] == ["locomote", "groom", "rear"]

    def test_one_class_has_no_imbalance_ratio(self):
        run = TrainingRun.from_summary(_summary(kept_label_counts={"only": 10}))
        assert run.imbalance_ratio is None

    def test_top_features_are_pairs(self):
        assert TrainingRun.from_summary(_summary()).top_features[0] == ("speed_mean", 0.4)

    def test_a_malformed_feature_entry_is_skipped(self):
        run = TrainingRun.from_summary(
            _summary(top_features=[{"feature": "a"}, {"feature": "b", "importance": 0.2}])
        )
        assert run.top_features == [("b", 0.2)]
