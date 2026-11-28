"""CameraPanel -> CVWorker hand-off is bounded, and teardown never kills a busy thread."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from glider.gui.panels.camera_panel import CameraPanel, CVWorker, FrameData  # noqa: E402
from glider.vision.camera_manager import CameraManager  # noqa: E402
from glider.vision.cv_processor import CVProcessor  # noqa: E402


def _frame(ts: float) -> FrameData:
    return FrameData(frame=np.zeros((4, 4, 3), dtype=np.uint8), timestamp=ts)


@pytest.fixture
def panel(qtbot):
    p = CameraPanel(CameraManager(), CVProcessor())
    qtbot.addWidget(p)
    sent: list[float] = []
    # Record what would cross to the worker thread instead of running CV.
    p._process_frame_requested.disconnect()
    p._process_frame_requested.connect(lambda fd: sent.append(fd.timestamp))
    p._sent = sent
    return p


def test_one_in_flight_and_only_the_newest_waits(panel):
    for ts in (1.0, 2.0, 3.0, 4.0):
        panel._submit_to_cv(_frame(ts))
    assert panel._sent == [1.0]  # the rest did not pile up in the event queue
    assert panel._cv_frames_dropped == 2  # 2.0 and 3.0 were superseded

    panel._process_cv_results_on_main_thread(_frame(1.0), [], [], None)
    assert panel._sent == [1.0, 4.0]  # the newest goes next
    panel._process_cv_results_on_main_thread(_frame(4.0), [], [], None)
    assert panel._cv_in_flight is False


def test_worker_answers_even_when_processing_fails(qtbot):
    class Broken:
        is_initialized = True
        last_result_cached = False

        def process_frame(self, frame, ts):
            raise RuntimeError("boom")

    worker = CVWorker(Broken())
    seen: list = []
    worker.results_ready.connect(lambda fd, d, t, m: seen.append((fd.timestamp, d, t)))
    worker.process_frame(_frame(7.0))
    # Without an answer the in-flight slot would stay taken forever.
    assert seen == [(7.0, [], [])]


def test_stopping_live_behavior_mid_load_does_not_delete_a_running_thread(panel, monkeypatch):
    import glider.gui.panels.camera_panel as cp

    calls = []
    monkeypatch.setattr(cp, "retire_thread", lambda *a, **k: calls.append(a) or False)
    thread, worker = MagicMock(), MagicMock()
    panel._behavior_thread, panel._behavior_worker = thread, worker
    panel.stop_live_behavior()
    # Handed to retire_thread (which parks a busy thread) instead of
    # wait(5000) + deleteLater() regardless of the result.
    assert calls == [(thread, worker)]
    thread.deleteLater.assert_not_called()


def test_every_frame_mode_queues_all_in_order(panel, monkeypatch):
    monkeypatch.setattr(type(panel._cv_processor), "needs_every_frame", True, raising=False)
    for ts in range(10):
        panel._submit_to_cv(_frame(float(ts)))
    assert panel._cv_frames_dropped == 0
    for _ in range(9):
        panel._process_cv_results_on_main_thread(_frame(0.0), [], [], None)
    assert panel._sent == [float(t) for t in range(10)]


def _recording_logger(panel):
    lg = MagicMock()
    lg.is_recording = True
    logged: list[float] = []
    lg.log_frame.side_effect = lambda ts, *a, **k: logged.append(ts)
    panel._tracking_logger = lg
    panel._dispatch_frame_tick = MagicMock()
    panel._process_frame_on_main_thread = lambda fd, d, t, m: lg.log_frame(fd.timestamp, [])
    return logged


def test_cap_drops_oldest_and_logs_in_capture_order(panel, monkeypatch):
    import glider.gui.panels.camera_panel as cp

    monkeypatch.setattr(cp, "CV_QUEUE_CAP", 2)
    monkeypatch.setattr(type(panel._cv_processor), "needs_every_frame", True, raising=False)
    monkeypatch.setattr(type(panel._cv_processor), "identity_active", True, raising=False)
    logged = _recording_logger(panel)
    for ts in (1.0, 2.0, 3.0, 4.0, 5.0):  # 1.0 in flight; 2.0, 3.0 evicted
        panel._submit_to_cv(_frame(ts))
    assert panel._cv_frames_dropped == 2
    assert [f.timestamp for f in panel._cv_queue] == [4.0, 5.0]
    for _ in range(3):
        panel._process_cv_results_on_main_thread(_frame(panel._sent[-1]), [], [], None)
    assert logged == sorted(logged) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_identity_without_classifier_logs_superseded_frames(panel, monkeypatch):
    # Cap 1 (latest wins), but every camera frame still gets a log row, so the
    # export can line rows up with video frames by timestamp.
    monkeypatch.setattr(type(panel._cv_processor), "identity_active", True, raising=False)
    monkeypatch.setattr(type(panel._cv_processor), "needs_every_frame", False, raising=False)
    logged = _recording_logger(panel)
    for ts in (1.0, 2.0, 3.0):  # 1.0 in flight; 2.0 superseded by 3.0
        panel._submit_to_cv(_frame(ts))
    panel._process_cv_results_on_main_thread(_frame(1.0), [], [], None)
    panel._process_cv_results_on_main_thread(_frame(3.0), [], [], None)
    assert logged == [1.0, 2.0, 3.0]


def test_default_mode_supersede_is_not_logged(panel):
    logged = _recording_logger(panel)
    for ts in (1.0, 2.0, 3.0):
        panel._submit_to_cv(_frame(ts))
    panel._process_cv_results_on_main_thread(_frame(1.0), [], [], None)
    panel._process_cv_results_on_main_thread(_frame(3.0), [], [], None)
    assert logged == [1.0, 3.0]


def test_lag_signal_fires_once_each_way(panel, monkeypatch):
    monkeypatch.setattr(type(panel._cv_processor), "needs_every_frame", True, raising=False)
    monkeypatch.setattr(panel, "current_fps", lambda: 2.0)
    seen: list[bool] = []
    panel.cv_lag_changed.connect(seen.append)
    for ts in range(5):  # 1 in flight, 4 queued > 2 fps
        panel._submit_to_cv(_frame(float(ts)))
    assert seen == [True]
    for _ in range(4):
        panel._process_cv_results_on_main_thread(_frame(0.0), [], [], None)
    assert seen == [True, False]
