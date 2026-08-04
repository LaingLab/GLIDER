"""Rendering a training summary as something a person can read.

The Train tab used to pprint the raw dict. These tests pin the report's
content, not its exact pixels: what must appear, what must be suppressed
when the data isn't there, and what must never crash the pane.
"""

from __future__ import annotations

import pytest

from glider.analysis.behavior.summary_text import WIDTH, format_training_summary


def _summary(**overrides):
    base = {
        "n_sessions": 27,
        "n_rows_total": 19636,
        "n_rows_kept": 18432,
        "n_rows_dropped": 1204,
        "kept_label_counts": {"investigate": 8000, "grooming": 6000, "dig": 2432, "locomote": 2000},
        "per_session_label_counts": {},
        "train_size": 15000,
        "test_size": 3020,
        "train_accuracy": 0.9812,
        "test_accuracy": 0.8471,
        "classes": ["dig", "grooming", "investigate", "locomote"],
        "n_features": 204,
        "top_features": [
            {"feature": "nose_speed__mean", "importance": 0.0412},
            {"feature": "body_length__std", "importance": 0.0388},
        ],
        "window": 30,
        "stats": ["mean", "std"],
        "fps": 30.0,
        "background_subsampled_to": None,
        "class_weight": "balanced",
        "per_class_metrics": {
            "dig": {"precision": 0.771, "recall": 0.688, "f1": 0.727, "support": 412},
            "grooming": {"precision": 0.902, "recall": 0.934, "f1": 0.918, "support": 980},
            "investigate": {"precision": 0.881, "recall": 0.842, "f1": 0.861, "support": 1240},
            "locomote": {"precision": 0.812, "recall": 0.845, "f1": 0.828, "support": 388},
        },
        "confusion_matrix": {
            "labels": ["dig", "grooming", "investigate", "locomote"],
            "matrix": [[283, 14, 62, 53], [12, 915, 41, 12], [71, 68, 1044, 57], [13, 9, 38, 328]],
        },
        "split_strategy": "cross_session",
        "n_holdout_sessions": 3,
        "classifier_type": "lightgbm",
        "mirror_augment": True,
        "lgbm_reg": {"num_leaves": 31, "min_child_samples": 50, "learning_rate": 0.1},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# It stops looking like a dict
# ---------------------------------------------------------------------------


def test_output_is_not_a_python_dict_dump():
    text = format_training_summary(_summary())
    assert "{" not in text
    assert "'per_class_metrics'" not in text
    assert "n_rows_kept" not in text  # the key name, not the number


def test_headline_names_the_classifier_and_split():
    text = format_training_summary(_summary())
    assert "lightgbm" in text
    assert "cross_session" in text


def test_accuracies_are_shown_to_three_decimals():
    text = format_training_summary(_summary())
    assert "0.847" in text
    assert "0.981" in text


def test_row_counts_are_grouped_for_reading():
    text = format_training_summary(_summary())
    assert "18,432" in text
    assert "1,204" in text


# ---------------------------------------------------------------------------
# Per-class table
# ---------------------------------------------------------------------------


def test_every_class_gets_a_row():
    text = format_training_summary(_summary())
    for name in ("investigate", "grooming", "dig", "locomote"):
        assert name in text


def test_per_class_table_carries_precision_recall_f1_support():
    text = format_training_summary(_summary())
    assert "precision" in text
    assert "recall" in text
    assert "f1" in text
    assert "support" in text


def test_macro_average_is_computed_and_shown():
    """Support-weighted totals hide a class the model is failing on."""
    text = format_training_summary(_summary())
    assert "macro" in text.lower()
    # mean of .771 .902 .881 .812 = 0.8415
    assert "0.842" in text or "0.841" in text


def test_support_totals_to_the_test_set():
    text = format_training_summary(_summary())
    assert "3,020" in text or "3020" in text


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_is_rendered_with_its_labels():
    text = format_training_summary(_summary())
    assert "CONFUSION" in text.upper()
    assert "1044" in text  # the investigate/investigate diagonal
    assert "actual" in text.lower() and "predicted" in text.lower()


def test_confusion_matrix_is_skipped_when_absent():
    text = format_training_summary(_summary(confusion_matrix={}))
    assert "CONFUSION" not in text.upper()


# ---------------------------------------------------------------------------
# The no-holdout case, which is the one that misleads
# ---------------------------------------------------------------------------


def test_no_holdout_run_suppresses_the_empty_tables():
    text = format_training_summary(
        _summary(
            split_strategy="no_holdout",
            test_size=0,
            test_accuracy=None,
            per_class_metrics={},
            confusion_matrix={},
            n_holdout_sessions=0,
        )
    )
    assert "CONFUSION" not in text.upper()
    assert "precision" not in text


def test_no_holdout_run_says_the_accuracy_is_not_a_generalization_estimate():
    """0.98 train accuracy with no test set is the most misleading number here."""
    text = format_training_summary(
        _summary(
            split_strategy="no_holdout",
            test_size=0,
            test_accuracy=None,
            per_class_metrics={},
            confusion_matrix={},
            n_holdout_sessions=0,
        )
    )
    lowered = text.lower()
    assert "no held-out" in lowered or "no holdout" in lowered or "not a generalization" in lowered


# ---------------------------------------------------------------------------
# Settings and features
# ---------------------------------------------------------------------------


def test_top_features_are_listed_with_importances():
    text = format_training_summary(_summary())
    assert "nose_speed__mean" in text
    assert "0.041" in text


def test_lgbm_knobs_appear_only_for_lightgbm():
    text = format_training_summary(_summary())
    assert "num_leaves" in text

    rf = format_training_summary(_summary(classifier_type="rf", lgbm_reg=None))
    assert "num_leaves" not in rf


def test_settings_report_class_weight_and_mirror():
    text = format_training_summary(_summary())
    assert "balanced" in text
    assert "mirror" in text.lower()


# ---------------------------------------------------------------------------
# Robustness: this renders whatever the pipeline hands it
# ---------------------------------------------------------------------------


def test_dropped_rows_are_described_accurately():
    """Most dropped rows are simply unlabelled frames.

    keep_mask drops (y == "") | ambiguous | NaN features, and on a real
    session the unlabelled frames outnumber the rest a hundred to one.
    Captioning the whole figure "unclear / multi-behavior" told the operator
    their annotations were being thrown away.
    """
    text = format_training_summary(_summary())
    lowered = text.lower()
    assert "unlabelled" in lowered or "unlabeled" in lowered
    assert "multi-behavior" not in lowered


def test_split_count_importances_render_as_integers():
    """LightGBM reports split COUNTS, not fractions; 82.0000 reads as noise."""
    text = format_training_summary(
        _summary(top_features=[{"feature": "body_length__max", "importance": 82.0}])
    )
    assert "82" in text
    assert "82.0000" not in text


def test_fractional_importances_keep_their_precision():
    """RandomForest reports fractions, where 4 dp is the whole signal."""
    text = format_training_summary(
        _summary(top_features=[{"feature": "speed__max", "importance": 0.0225}])
    )
    assert "0.0225" in text


def test_confusion_headers_are_readable_when_there_is_room():
    """Four short class names must not be cut to 'gro…' inside a 76-col pane."""
    text = format_training_summary(_summary())
    confusion = text.split("CONFUSION")[1].splitlines()[1]
    assert "grooming" in confusion or "groomin" in confusion


def test_confusion_still_fits_with_many_classes():
    labels = [f"behaviour_number_{i}" for i in range(12)]
    matrix = [[i * j for j in range(12)] for i in range(12)]
    text = format_training_summary(_summary(confusion_matrix={"labels": labels, "matrix": matrix}))
    over = [line for line in text.splitlines() if len(line) > WIDTH]
    assert not over, f"lines wider than {WIDTH}: {over}"


def test_an_empty_summary_does_not_raise():
    assert isinstance(format_training_summary({}), str)


def test_a_none_summary_does_not_raise():
    assert isinstance(format_training_summary(None), str)


def test_a_non_dict_summary_does_not_raise():
    assert isinstance(format_training_summary(["not", "a", "dict"]), str)


def test_missing_accuracy_renders_a_placeholder_not_a_crash():
    text = format_training_summary(_summary(train_accuracy=None, test_accuracy=None))
    assert isinstance(text, str)
    assert "None" not in text


def test_long_class_names_do_not_wrap_the_table():
    long_name = "investigating_the_novel_object_at_length"
    text = format_training_summary(
        _summary(
            per_class_metrics={
                long_name: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 10}
            },
            confusion_matrix={"labels": [long_name], "matrix": [[10]]},
        )
    )
    assert max(len(line) for line in text.splitlines()) <= WIDTH


def test_every_line_is_within_the_pane_width():
    """The pane is a fixed-width card; a wrapped table row stops lining up."""
    text = format_training_summary(_summary())
    over = [line for line in text.splitlines() if len(line) > WIDTH]
    assert not over, f"lines wider than {WIDTH}: {over}"


@pytest.mark.parametrize("strategy", ["cross_session", "group_shuffle", "no_holdout", "weird"])
def test_any_split_strategy_renders(strategy):
    assert isinstance(format_training_summary(_summary(split_strategy=strategy)), str)
