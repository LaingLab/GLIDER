import json

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
