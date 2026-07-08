import csv
import json

import pytest

from glider.analysis.behavior.pipeline import TrainResult
from glider.analysis.behavior.report import write_training_report


def _train_result(summary):
    return TrainResult(model=None, summary=summary)


def test_writes_summary_json_for_trainresult(tmp_path):
    summary = {
        "classifier_type": "LGBMClassifier",
        "test_accuracy": 0.9,
        "classes": ["rest", "locomote"],
        "kept_label_counts": {"rest": 10, "locomote": 8},
    }
    out = write_training_report(_train_result(summary), tmp_path / "report")
    assert out == tmp_path / "report"
    p = out / "summary.json"
    assert p.exists()
    assert json.loads(p.read_text())["classifier_type"] == "LGBMClassifier"


def test_export_from_package():
    from glider.analysis.behavior import write_training_report as exported

    assert exported is write_training_report


def _full_summary():
    return {
        "classifier_type": "LGBMClassifier",
        "test_accuracy": 0.82,
        "classes": ["rest", "locomote"],
        "per_class_metrics": {
            "rest": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 40},
            "locomote": {"precision": 0.7, "recall": 0.75, "f1": 0.72, "support": 30},
        },
        "confusion_matrix": {"labels": ["rest", "locomote"], "matrix": [[32, 8], [7, 23]]},
        "top_features": [
            {"feature": "speed_a__mean", "importance": 0.4},
            {"feature": "dist_x__std", "importance": 0.2},
        ],
        "kept_label_counts": {"rest": 40, "locomote": 30},
        "per_session_label_counts": [{"rest": 20, "locomote": 15}, {"rest": 20, "locomote": 15}],
    }


def test_writes_all_csvs_when_present(tmp_path):
    out = write_training_report(_train_result(_full_summary()), tmp_path / "r")
    rows = list(csv.DictReader((out / "per_class_metrics.csv").open()))
    assert {r["class"] for r in rows} == {"rest", "locomote"}
    assert (out / "confusion_matrix.csv").exists()
    fi = list(csv.DictReader((out / "feature_importances.csv").open()))
    assert fi[0]["feature"] == "speed_a__mean"


def test_default_summary_skips_empty_metrics(tmp_path):
    # test_split=0.0 reality: keys present but EMPTY.
    summary = {
        "classifier_type": "LGBMClassifier",
        "confusion_matrix": {},
        "per_class_metrics": {},
        "top_features": [{"feature": "f0", "importance": 1.0}],
        "kept_label_counts": {"a": 5},
    }
    out = write_training_report(_train_result(summary), tmp_path / "r")
    assert not (out / "confusion_matrix.csv").exists()
    assert not (out / "per_class_metrics.csv").exists()
    assert (out / "feature_importances.csv").exists()  # top_features present


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path):
    return path.read_bytes()[:8] == _PNG_MAGIC


def test_trainresult_full_charts(tmp_path):
    out = write_training_report(_train_result(_full_summary()), tmp_path / "r")
    for name in [
        "confusion_matrix.png",
        "per_class_metrics.png",
        "feature_importances.png",
        "class_balance.png",
    ]:
        assert (out / name).exists() and _is_png(out / name)


def test_default_summary_only_two_charts(tmp_path):
    summary = {
        "classifier_type": "LGBMClassifier",
        "confusion_matrix": {},
        "per_class_metrics": {},
        "top_features": [{"feature": "f0", "importance": 1.0}],
        "kept_label_counts": {"a": 5, "b": 3},
    }
    out = write_training_report(_train_result(summary), tmp_path / "r")
    assert (out / "feature_importances.png").exists()
    assert (out / "class_balance.png").exists()
    assert not (out / "confusion_matrix.png").exists()
    assert not (out / "per_class_metrics.png").exists()


def test_single_class_does_not_crash(tmp_path):
    summary = {
        "confusion_matrix": {},
        "per_class_metrics": {},
        "top_features": [],
        "kept_label_counts": {"only": 12},
    }
    out = write_training_report(_train_result(summary), tmp_path / "r")
    assert (out / "class_balance.png").exists()  # renders; no importances (empty)


def test_hybrid_report(tmp_path, hybrid_sessions):
    pytest.importorskip("lightgbm")
    from glider.analysis.behavior.pipeline import train_hybrid_model

    sessions, tag_map = hybrid_sessions
    res = train_hybrid_model(sessions, tag_map=tag_map, fps=30.0, random_state=0)
    out = write_training_report(res, tmp_path / "h")
    assert (out / "lambda_sweep.png").exists() and _is_png(out / "lambda_sweep.png")
    assert (out / "feature_importances.png").exists() and _is_png(out / "feature_importances.png")
    s = json.loads((out / "summary.json").read_text())
    assert s["classifier_type"] == "LGBMClassifier" and "per_lambda_f1" in s
    # hybrid carries no confusion/per-class:
    assert not (out / "confusion_matrix.png").exists()
    assert not (out / "per_class_metrics.csv").exists()
