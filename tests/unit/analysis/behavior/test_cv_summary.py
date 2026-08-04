"""Rendering a cross-validation result as a readable report.

cross_validate_sessions returns a different dict from train_model's summary —
per-fold arrays, pooled metrics, bout recall — so it gets its own formatter
rather than being forced through the training one.
"""

from __future__ import annotations

import pytest

from glider.analysis.behavior.summary_text import WIDTH, format_cv_summary


def _cv(**overrides):
    base = {
        "n_folds": 5,
        "n_sessions": 30,
        "fold_accuracies": [0.812, 0.788, 0.801, 0.776, 0.828],
        "fold_macro_f1": [0.761, 0.723, 0.748, 0.719, 0.789],
        "mean_accuracy": 0.801,
        "std_accuracy": 0.0189,
        "mean_macro_f1": 0.748,
        "per_class_metrics": {
            "dig": {"precision": 0.80, "recall": 0.62, "f1": 0.70, "support": 2684},
            "grooming": {"precision": 0.88, "recall": 0.75, "f1": 0.81, "support": 5996},
            "investigate": {"precision": 0.70, "recall": 0.85, "f1": 0.77, "support": 9247},
            "locomote": {"precision": 0.74, "recall": 0.82, "f1": 0.78, "support": 3388},
        },
        "confusion_matrix": {
            "labels": ["dig", "grooming", "investigate", "locomote"],
            "matrix": [
                [1664, 130, 620, 270],
                [90, 4497, 1120, 289],
                [300, 290, 7860, 797],
                [70, 40, 500, 2778],
            ],
        },
        "false_alarm_rate": None,
        "background_class_name": None,
        # Key names copied from pipeline._bout_recall, not invented — an
        # earlier fixture guessed "recall" and every assertion passed against
        # a formatter that rendered a dash for every real result.
        "bout_metrics": {
            "dig": {
                "n_bouts": 166,
                "n_pred_bouts": 150,
                "recall_any": 0.71,
                "recall_25": 0.64,
                "recall_50": 0.55,
                "precision_any": 0.80,
                "precision_25": 0.70,
                "precision_50": 0.61,
                "f1_any": 0.75,
                "f1_25": 0.67,
                "f1_50": 0.58,
            },
            "locomote": {
                "n_bouts": 145,
                "n_pred_bouts": 160,
                "recall_any": 0.88,
                "recall_25": 0.83,
                "recall_50": 0.77,
                "precision_any": 0.79,
                "precision_25": 0.75,
                "precision_50": 0.70,
                "f1_any": 0.83,
                "f1_25": 0.79,
                "f1_50": 0.73,
            },
        },
        "n_rows_kept": 43041,
    }
    base.update(overrides)
    return base


def test_it_is_not_a_dict_dump():
    text = format_cv_summary(_cv())
    assert "{" not in text
    assert "fold_macro_f1" not in text


def test_headline_carries_mean_and_spread():
    """A CV number without its spread invites the same single-split mistake."""
    text = format_cv_summary(_cv())
    assert "0.748" in text
    assert "0.019" in text or "0.0189" in text


def test_standard_deviation_of_macro_f1_is_computed_from_the_folds():
    """The dict carries std_accuracy but no std for macro F1."""
    text = format_cv_summary(_cv())
    # sd of [.761 .723 .748 .719 .789] is ~0.0264
    assert "0.026" in text or "0.027" in text


def test_every_fold_is_listed():
    text = format_cv_summary(_cv())
    for value in ("0.761", "0.723", "0.789"):
        assert value in text


def test_fold_count_and_sessions_are_reported():
    text = format_cv_summary(_cv())
    assert "5" in text and "30" in text


def test_pooled_per_class_table_is_rendered():
    text = format_cv_summary(_cv())
    for name in ("dig", "grooming", "investigate", "locomote"):
        assert name in text
    assert "precision" in text


def test_confusion_is_rendered():
    text = format_cv_summary(_cv())
    assert "CONFUSION" in text.upper()
    assert "7860" in text


def test_bout_metric_keys_match_what_the_pipeline_emits():
    """Guards the fixture. The first version of this file invented a 'recall'
    key; every assertion passed while the formatter rendered a dash for every
    real result, because only real data has the true names."""
    import inspect

    from glider.analysis.behavior import pipeline

    source = inspect.getsource(pipeline._bout_recall)
    for key in _cv()["bout_metrics"]["dig"]:
        assert f'"{key}"' in source, f"{key!r} is not a key _bout_recall emits"


def test_bout_recall_is_shown_when_present():
    """Frame accuracy hides whether a bout was caught at all."""
    text = format_cv_summary(_cv())
    assert "bout" in text.lower()
    assert "0.880" in text  # locomote recall_any


def test_bout_table_shows_partial_coverage_not_just_detection():
    """recall_any alone says a bout was touched, not how much of it was got."""
    text = format_cv_summary(_cv())
    assert "0.770" in text  # locomote recall_50
    assert "0.730" in text  # locomote f1_50


def test_bout_recall_is_skipped_when_absent():
    text = format_cv_summary(_cv(bout_metrics={}))
    assert "bout" not in text.lower()


def test_false_alarm_rate_appears_only_with_background():
    plain = format_cv_summary(_cv())
    assert "false alarm" not in plain.lower()

    withbg = format_cv_summary(_cv(false_alarm_rate=0.037, background_class_name="background"))
    assert "false alarm" in withbg.lower()
    assert "0.037" in withbg


def test_it_says_no_model_was_produced():
    """The single most confusable thing about this action."""
    text = format_cv_summary(_cv())
    assert "no model" in text.lower()


def test_lines_fit_the_pane():
    text = format_cv_summary(_cv())
    over = [line for line in text.splitlines() if len(line) > WIDTH]
    assert not over, f"lines wider than {WIDTH}: {over}"


@pytest.mark.parametrize("bad", [None, {}, [], "nope", 3])
def test_rubbish_input_does_not_raise(bad):
    assert isinstance(format_cv_summary(bad), str)


def test_a_single_fold_result_renders():
    text = format_cv_summary(
        _cv(n_folds=1, fold_accuracies=[0.8], fold_macro_f1=[0.75], mean_macro_f1=0.75)
    )
    assert isinstance(text, str)
    assert "0.75" in text


def test_missing_metrics_render_placeholders_not_none():
    text = format_cv_summary(
        _cv(mean_accuracy=None, mean_macro_f1=None, per_class_metrics={}, confusion_matrix={})
    )
    assert "None" not in text
