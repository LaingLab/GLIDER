"""Tests for the convert/train/apply QThread workers.

Each worker's ``run()`` must emit ``finished`` on success and ``failed``
(never raise) on error, with the Qt-free core patched out so no real
YOLO / training / classification runs.
"""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("PyQt6")


# --- TrainWorker -------------------------------------------------------------


def test_train_worker_emits_finished(qtbot, monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    saved = {}

    class _Model:
        def save(self, p):
            saved["path"] = p

    class _Res:
        summary = {"accuracy": 0.9}
        model = _Model()

    monkeypatch.setattr(workers, "train_model", lambda *a, **k: _Res())
    w = workers.TrainWorker(
        sessions=[("a.csv", "a_ann.csv")], output=tmp_path / "m.pkl", options={}
    )
    with qtbot.waitSignal(w.finished, timeout=1000) as blocker:
        w.run()
    assert blocker.args[0] == {"accuracy": 0.9}
    assert saved["path"] == tmp_path / "m.pkl"  # worker saves explicitly


def test_train_worker_emits_failed_on_error(qtbot, monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(workers, "train_model", boom)
    w = workers.TrainWorker(
        sessions=[("a.csv", "a_ann.csv")], output=tmp_path / "m.pkl", options={}
    )
    with qtbot.waitSignal(w.failed, timeout=1000) as blocker:
        w.run()
    assert "nope" in blocker.args[0]


def test_train_worker_writes_a_report_beside_the_bundle(qtbot, monkeypatch, tmp_path):
    """The run's numbers must survive the window that produced them."""
    from glider.analysis.behavior.pipeline import TrainResult
    from glider.gui.behavior import workers

    class _Model:
        def save(self, p):
            pathlib.Path(p).write_bytes(b"")

    summary = {
        "classifier_type": "lightgbm",
        "split_strategy": "no_holdout",
        "n_sessions": 1,
        "train_accuracy": 0.9,
        "per_class_metrics": {"groom": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10}},
        "confusion_matrix": {"labels": ["groom"], "matrix": [[10]]},
        "top_features": [{"feature": "speed_mean", "importance": 0.5}],
        "kept_label_counts": {"groom": 10},
    }
    monkeypatch.setattr(
        workers, "train_model", lambda *a, **k: TrainResult(model=_Model(), summary=summary)
    )

    out = tmp_path / "mouse_v1.pkl"
    w = workers.TrainWorker(sessions=[("a.csv", "a_ann.csv")], output=out, options={})
    with qtbot.waitSignal(w.report_ready, timeout=20_000) as blocker:
        w.run()

    report = blocker.args[0]
    assert report == workers.report_dir_for(out) == tmp_path / "mouse_v1_report"
    assert (report / "summary.json").exists()
    # The CSVs are the point: the numbers have to be readable without GLIDER.
    assert (report / "per_class_metrics.csv").exists()
    assert (report / "confusion_matrix.csv").exists()
    assert (report / "feature_importances.csv").exists()


def test_a_report_that_cannot_be_written_does_not_lose_the_model(qtbot, monkeypatch, tmp_path):
    """A ten-minute fit must not be lost to a read-only share or a bad chart."""
    from glider.analysis.behavior.pipeline import TrainResult
    from glider.gui.behavior import workers

    saved = {}

    class _Model:
        def save(self, p):
            saved["path"] = p

    monkeypatch.setattr(
        workers,
        "train_model",
        lambda *a, **k: TrainResult(model=_Model(), summary={"ok": True}),
    )

    def boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr("glider.analysis.behavior.write_training_report", boom)

    w = workers.TrainWorker(
        sessions=[("a.csv", "a_ann.csv")], output=tmp_path / "m.pkl", options={}
    )
    with qtbot.waitSignal(w.finished, timeout=5000) as blocker:
        w.run()
    assert blocker.args[0] == {"ok": True}
    assert saved["path"] == tmp_path / "m.pkl"


# --- ConvertWorker -----------------------------------------------------------


def test_convert_worker_emits_finished(qtbot, monkeypatch, tmp_path):
    from glider.gui.behavior import workers
    from glider.vision.pose import dlc

    monkeypatch.setattr(workers, "infer_video", lambda **k: "raw_pose")
    monkeypatch.setattr(workers, "smooth", lambda pose: "smoothed_pose")
    written = {}
    monkeypatch.setattr(
        dlc, "to_dlc_csv", lambda pose, path: written.setdefault("pose", pose) or path
    )

    out = tmp_path / "pose.csv"
    w = workers.ConvertWorker(
        video="clip.mp4", model="yolo.pt", keypoint_names=["nose", "tail"], output=out
    )
    with qtbot.waitSignal(w.finished, timeout=1000) as blocker:
        w.run()
    assert blocker.args[0] == str(out)
    assert written["pose"] == "smoothed_pose"  # smoothed pose is what gets written


def test_convert_worker_emits_failed_on_error(qtbot, monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    def boom(**k):
        raise RuntimeError("infer blew up")

    monkeypatch.setattr(workers, "infer_video", boom)
    w = workers.ConvertWorker(
        video="clip.mp4",
        model="yolo.pt",
        keypoint_names=["nose"],
        output=tmp_path / "pose.csv",
    )
    with qtbot.waitSignal(w.failed, timeout=1000) as blocker:
        w.run()
    assert "infer blew up" in blocker.args[0]


# --- ApplyWorker -------------------------------------------------------------


def test_apply_worker_emits_finished(qtbot, monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    sentinel = object()
    monkeypatch.setattr(workers, "classify", lambda *a, **k: sentinel)
    w = workers.ApplyWorker(
        video="clip.mp4",
        model_path="m.pkl",
        yolo_path="yolo.pt",
        keypoint_names=["nose"],
        output_dir=tmp_path,
    )
    with qtbot.waitSignal(w.finished, timeout=1000) as blocker:
        w.run()
    assert blocker.args[0] is sentinel


def test_apply_worker_emits_failed_on_error(qtbot, monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    def boom(*a, **k):
        raise RuntimeError("classify failed")

    monkeypatch.setattr(workers, "classify", boom)
    w = workers.ApplyWorker(
        video="clip.mp4",
        model_path="m.pkl",
        yolo_path="yolo.pt",
        keypoint_names=["nose"],
        output_dir=tmp_path,
    )
    with qtbot.waitSignal(w.failed, timeout=1000) as blocker:
        w.run()
    assert "classify failed" in blocker.args[0]
