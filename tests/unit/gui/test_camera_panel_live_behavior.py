"""Tests for the CameraPanel's "Live Behavior" control group + worker lifecycle.

These exercise the panel-side wiring around
:class:`glider.gui.panels.live_behavior.BehaviorInferenceWorker`: the picker
+ keypoint-name gating of the Start/Stop toggle, the load-failure path
(mismatched keypoint count), the happy path (models load → frames fan out →
preview label updates), and clean thread teardown on Stop/close.

The ultralytics stack is faked (``_load_backend`` is monkeypatched) so nothing
here touches torch or a real pose net.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from glider.vision.pose.backend import UltralyticsBackend

pytest.importorskip("PyQt6")


# ---------------------------------------------------------------------------
# ultralytics doubles (mirrors tests/unit/gui/test_live_behavior.py)
# ---------------------------------------------------------------------------


class _Torchy:
    def __init__(self, a):
        self._a = np.asarray(a)

    def __getitem__(self, i):
        return _Torchy(self._a[i])

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape


class FakeKeypoints:
    def __init__(self, xy, conf=None):
        self.xy = _Torchy(np.asarray(xy)[None]) if xy is not None else _Torchy(np.zeros((0, 0, 2)))
        self.conf = _Torchy(np.asarray(conf)[None]) if conf is not None else None


class FakeResult:
    def __init__(self, kp):
        self.keypoints = kp


class FakeYolo:
    def __init__(self, xy):
        self._xy = xy

    def predict(self, bgr, conf=0.25, verbose=False):  # noqa: ARG002
        if self._xy is None:
            return [FakeResult(FakeKeypoints(None))]
        conf_arr = np.ones(len(self._xy))
        return [FakeResult(FakeKeypoints(self._xy, conf_arr))]


def _distinct_kps(k: int) -> np.ndarray:
    return np.stack([np.linspace(0.0, 10.0, k), np.linspace(0.0, 5.0, k)], axis=1)


# ---------------------------------------------------------------------------
# panel construction + frame helper
# ---------------------------------------------------------------------------


def _make_panel(qtbot):
    from glider.gui.panels.camera_panel import CameraPanel
    from glider.vision.camera_manager import CameraManager
    from glider.vision.cv_processor import CVProcessor

    panel = CameraPanel(CameraManager(), CVProcessor())
    qtbot.addWidget(panel)
    panel.show()
    return panel


def _frame():
    from glider.gui.panels.camera_panel import FrameData

    return FrameData(frame=np.zeros((8, 8, 3), dtype=np.uint8), timestamp=0.0)


# ---------------------------------------------------------------------------
# control-group presence + enable gating
# ---------------------------------------------------------------------------


def test_live_behavior_group_gating(qtbot):
    panel = _make_panel(qtbot)

    # The toggle exists and starts disabled + labelled "Start".
    assert panel._live_behavior_btn.text() == "Start"
    assert panel._live_behavior_btn.isEnabled() is False

    # Only paths, no names → still disabled.
    panel._behavior_pkl = Path("m.pkl")
    panel._pose_model_path = Path("y.pt")
    panel._update_live_controls_enabled()
    assert panel._live_behavior_btn.isEnabled() is False

    # Names but a missing path → still disabled.
    panel._behavior_pkl = None
    panel._kp_names_edit.setText("k0, k1, k2, k3")
    assert panel._live_behavior_btn.isEnabled() is False

    # All three present → enabled.
    panel._behavior_pkl = Path("m.pkl")
    panel._update_live_controls_enabled()
    assert panel._live_behavior_btn.isEnabled() is True

    # Emptying the names field disables it again.
    panel._kp_names_edit.setText("   ")
    assert panel._live_behavior_btn.isEnabled() is False


# ---------------------------------------------------------------------------
# load-failure path (mismatched keypoint count)
# ---------------------------------------------------------------------------


def test_live_behavior_load_failed_ends_stopped(qtbot, tiny_behavior_model, tmp_path, monkeypatch):
    from glider.gui.panels import live_behavior as lb

    pkl = tmp_path / "m.pkl"
    tiny_behavior_model.save(pkl)
    monkeypatch.setattr(
        lb,
        "_load_backend",
        lambda path, names, conf=0.25: UltralyticsBackend(FakeYolo(_distinct_kps(4)), names),
    )

    warnings: list[tuple] = []
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warnings.append(a) or None)
    )

    panel = _make_panel(qtbot)
    panel._behavior_pkl = pkl
    panel._pose_model_path = tmp_path / "y.pt"
    panel._kp_names_edit.setText("a, b")  # WRONG count: model expects 4
    panel._update_live_controls_enabled()

    panel._toggle_live_behavior()  # Start → load fails on the worker thread

    qtbot.waitUntil(lambda: bool(warnings), timeout=5000)

    assert panel._behavior_running is False
    assert panel._live_behavior_btn.text() == "Start"
    assert panel._behavior_thread is None


# ---------------------------------------------------------------------------
# happy path: models load, frames fan out, preview label updates
# ---------------------------------------------------------------------------


def test_live_behavior_valid_flow_updates_preview(
    qtbot, tiny_behavior_model, tmp_path, monkeypatch
):
    from glider.gui.panels import live_behavior as lb

    pkl = tmp_path / "m.pkl"
    tiny_behavior_model.save(pkl)
    k = lb.model_keypoint_count(tiny_behavior_model)
    monkeypatch.setattr(
        lb,
        "_load_backend",
        lambda path, names, conf=0.25: UltralyticsBackend(FakeYolo(_distinct_kps(k)), names),
    )

    panel = _make_panel(qtbot)
    panel._behavior_pkl = pkl
    panel._pose_model_path = tmp_path / "y.pt"
    panel._kp_names_edit.setText(", ".join(f"k{i}" for i in range(k)))
    panel._update_live_controls_enabled()

    worker = None

    def _grab():
        nonlocal worker
        worker = panel._behavior_worker

    with qtbot.waitSignal(panel._live_behavior_btn.clicked, timeout=1000):
        panel._live_behavior_btn.click()
    worker = panel._behavior_worker
    assert worker is not None

    with qtbot.waitSignal(worker.ready, timeout=5000):
        pass  # ready is emitted from initialize kicked off by the click

    # Vocab is pushed to the preview and the button flips to Stop/running.
    assert panel._preview._vocab == list(tiny_behavior_model.classes)
    assert panel._behavior_running is True
    assert panel._live_behavior_btn.text() == "Stop"

    clf = worker._classifier
    # EVERY frame now fans out to the worker (no intake decimation), matching
    # the offline extractor which consumes consecutive frames. So the number of
    # frames needed to fill the warm-up + window is exactly that count.
    needed = clf.window + clf.warmup + 6
    for _ in range(needed):
        panel._handle_frame_input(_frame())

    qtbot.waitUntil(
        lambda: panel._preview._behavior_label in set(tiny_behavior_model.classes),
        timeout=5000,
    )
    assert panel._preview._behavior_label in set(tiny_behavior_model.classes)

    panel.stop_live_behavior()


# ---------------------------------------------------------------------------
# result_ready → preview wiring (thread-free sanity check)
# ---------------------------------------------------------------------------


def test_result_ready_updates_preview(qtbot, tiny_behavior_model, tmp_path, monkeypatch):
    from glider.gui.panels import live_behavior as lb

    pkl = tmp_path / "m.pkl"
    tiny_behavior_model.save(pkl)
    k = lb.model_keypoint_count(tiny_behavior_model)
    monkeypatch.setattr(
        lb,
        "_load_backend",
        lambda path, names, conf=0.25: UltralyticsBackend(FakeYolo(_distinct_kps(k)), names),
    )

    panel = _make_panel(qtbot)
    panel._behavior_pkl = pkl
    panel._pose_model_path = tmp_path / "y.pt"
    panel._kp_names_edit.setText(", ".join(f"k{i}" for i in range(k)))
    panel._update_live_controls_enabled()
    panel._toggle_live_behavior()

    worker = panel._behavior_worker
    with qtbot.waitSignal(worker.ready, timeout=5000):
        pass

    kps = _distinct_kps(k)
    worker.result_ready.emit("A", kps)
    qtbot.waitUntil(lambda: panel._preview._behavior_label == "A", timeout=5000)
    assert panel._preview._pose_kps is not None

    panel.stop_live_behavior()


# ---------------------------------------------------------------------------
# teardown joins the thread + clears overlays
# ---------------------------------------------------------------------------


def test_stop_live_behavior_joins_thread(qtbot, tiny_behavior_model, tmp_path, monkeypatch):
    from glider.gui.panels import live_behavior as lb

    pkl = tmp_path / "m.pkl"
    tiny_behavior_model.save(pkl)
    k = lb.model_keypoint_count(tiny_behavior_model)
    monkeypatch.setattr(
        lb,
        "_load_backend",
        lambda path, names, conf=0.25: UltralyticsBackend(FakeYolo(_distinct_kps(k)), names),
    )

    panel = _make_panel(qtbot)
    panel._behavior_pkl = pkl
    panel._pose_model_path = tmp_path / "y.pt"
    panel._kp_names_edit.setText(", ".join(f"k{i}" for i in range(k)))
    panel._update_live_controls_enabled()
    panel._toggle_live_behavior()

    worker = panel._behavior_worker
    thread = panel._behavior_thread
    with qtbot.waitSignal(worker.ready, timeout=5000):
        pass

    panel._preview.set_behavior_label("A")
    panel._preview.set_pose_overlay(_distinct_kps(k))

    panel.stop_live_behavior()

    assert thread.isRunning() is False
    assert panel._behavior_running is False
    assert panel._behavior_thread is None
    assert panel._live_behavior_btn.text() == "Start"
    assert panel._preview._pose_kps is None
    assert panel._preview._behavior_label == ""
