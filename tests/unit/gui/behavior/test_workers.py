"""Tests for the convert/train/apply QThread workers.

Each worker's ``run()`` must emit ``finished`` on success and ``failed``
(never raise) on error, with the Qt-free core patched out so no real
YOLO / training / classification runs.
"""

from __future__ import annotations

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
