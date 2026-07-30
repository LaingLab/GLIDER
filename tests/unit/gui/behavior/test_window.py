"""Smoke tests for the top-level Behavior Analysis window.

Just enough to prove the window builds its three tabs and can default
its project directory from config — the tab bodies (Annotate/Train/Apply)
get their own focused tests once the window exists.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def test_window_has_three_tabs(qtbot, tmp_path):
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path)
    qtbot.addWidget(win)
    titles = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert titles == ["Annotate", "Train", "Apply"]


def test_window_defaults_project_dir_from_config(qtbot):
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(parent=None)
    qtbot.addWidget(win)
    assert win.project_dir is not None


# --------------------------------------------------------------------------
# Apply tab: freeze/dart speed axis in real units
# --------------------------------------------------------------------------


def test_speed_axis_is_off_by_default(qtbot):
    """Pre-existing behaviour: no speed column unless asked for."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    assert tab._speed_group.isChecked() is False
    assert tab._speed_opts() == {}


def test_absolute_mode_carries_cm_per_second_and_the_calibration(qtbot, tmp_path):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._freeze_cm_s.setValue(1.25)
    tab._dart_cm_s.setValue(20.0)
    tab._calibration_master = tmp_path / "pose_calibration.json"

    opts = tab._speed_opts()
    assert opts["freeze_cm_s"] == pytest.approx(1.25)
    assert opts["dart_cm_s"] == pytest.approx(20.0)
    assert opts["calibration_master"] == tmp_path / "pose_calibration.json"
    assert "freeze_pct" not in opts


def test_thresholds_are_entered_in_centimetres_not_pixels(qtbot):
    # px/frame means nothing physical and shifts with camera height.
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    assert tab._freeze_cm_s.suffix().strip() == "cm/s"
    assert tab._dart_cm_s.suffix().strip() == "cm/s"


def test_percentile_mode_sends_percentiles_and_no_calibration(qtbot, tmp_path):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._calibration_master = tmp_path / "ignored.json"
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("percentile"))
    tab._freeze_pct.setValue(5.0)
    tab._dart_pct.setValue(99.0)

    opts = tab._speed_opts()
    assert opts["freeze_pct"] == pytest.approx(5.0)
    assert opts["dart_pct"] == pytest.approx(99.0)
    # Percentiles need no scale, so the calibration must not be sent.
    assert "calibration_master" not in opts
    assert "freeze_cm_s" not in opts


def test_minimum_durations_are_sent_in_seconds_in_both_modes(qtbot):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._freeze_min_s.setValue(1.5)
    tab._dart_min_s.setValue(0.2)

    assert tab._freeze_min_s.suffix().strip() == "s"
    for mode in ("absolute", "percentile"):
        tab._speed_mode.setCurrentIndex(tab._speed_mode.findData(mode))
        opts = tab._speed_opts()
        assert opts["freeze_min_s"] == pytest.approx(1.5)
        assert opts["dart_min_s"] == pytest.approx(0.2)


def test_switching_mode_hides_the_irrelevant_fields(qtbot):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab.show()

    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("absolute"))
    assert tab._freeze_cm_s.isVisible() is True
    assert tab._freeze_pct.isVisible() is False

    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("percentile"))
    assert tab._freeze_cm_s.isVisible() is False
    assert tab._freeze_pct.isVisible() is True


def test_apply_worker_forwards_speed_opts_to_classify(monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    seen = {}

    def fake_classify(video, **kwargs):
        seen.update(kwargs)
        return "result"

    monkeypatch.setattr(workers, "classify", fake_classify)
    worker = workers.ApplyWorker(
        video=tmp_path / "v.mp4",
        model_path=tmp_path / "m.pkl",
        yolo_path=tmp_path / "y.pt",
        keypoint_names=["a"],
        output_dir=tmp_path / "out",
        speed_opts={"freeze_mm_s": 10.0, "dart_mm_s": 150.0, "calibration_master": None},
    )
    worker.run()
    assert seen["freeze_mm_s"] == pytest.approx(10.0)
    assert seen["dart_mm_s"] == pytest.approx(150.0)


def test_apply_worker_without_speed_opts_passes_nothing_extra(monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    seen = {}
    monkeypatch.setattr(workers, "classify", lambda video, **kw: seen.update(kw) or "r")
    workers.ApplyWorker(
        video=tmp_path / "v.mp4",
        model_path=tmp_path / "m.pkl",
        yolo_path=tmp_path / "y.pt",
        keypoint_names=["a"],
        output_dir=tmp_path / "out",
    ).run()
    assert "freeze_mm_s" not in seen


# --------------------------------------------------------------------------
# GUI <-> core contract: every option the tab emits must be accepted downstream
# --------------------------------------------------------------------------


def _speed_opt_keys_for_every_mode(tab):
    """Union of the kwargs _speed_opts() can produce across all modes."""
    tab._speed_group.setChecked(True)
    keys = set()
    for i in range(tab._speed_mode.count()):
        tab._speed_mode.setCurrentIndex(i)
        keys |= set(tab._speed_opts())
    return keys


def test_classify_accepts_every_option_the_apply_tab_can_send(qtbot):
    """Regression: freeze_pct fell through **opts into LiveInferenceConfig."""
    import inspect

    from glider.analysis.behavior.classify import classify
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    emitted = _speed_opt_keys_for_every_mode(tab)

    accepted = set(inspect.signature(classify).parameters)
    missing = sorted(emitted - accepted)
    assert not missing, f"classify() would pass these to LiveInferenceConfig: {missing}"


def test_no_speed_option_leaks_into_the_pipeline_config(qtbot):
    """The config rejects unknown kwargs, so a leak is a hard failure at Run."""
    import inspect

    from glider.analysis.behavior.classify import classify
    from glider.analysis.behavior.classify.pipeline import LiveInferenceConfig
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    emitted = _speed_opt_keys_for_every_mode(tab)

    # Anything the tab emits must either be consumed by classify's own
    # signature or be a real config field -- never silently forwarded.
    consumed = set(inspect.signature(classify).parameters)
    config_fields = set(inspect.signature(LiveInferenceConfig).parameters)
    for key in emitted:
        assert key in consumed or key in config_fields, f"{key} would reach LiveInferenceConfig"


# --------------------------------------------------------------------------
# Annotate tab: the pose-batch handoff
#
# Batch Pose Tracking writes "<stem>DLC_<model>.csv"; this tab used to look
# only for "<stem>.csv", so pointing it at a folder that tool had just filled
# reported every single video as missing its pose data.
# --------------------------------------------------------------------------


def _pose_batch_output(tmp_path, stem, fps=30.0):
    """A video plus the CSV run_batch would have written beside it."""
    import numpy as np

    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    names = ["snout", "neck", "tail_base"]
    video = tmp_path / f"{stem}.mp4"
    video.touch()
    pose = PoseData(
        xy=np.zeros((60, len(names), 2)),
        confidence=np.ones((60, len(names))),
        keypoint_names=names,
        fps=fps,
    )
    to_dlc_csv(pose, tmp_path / f"{stem}DLC_exp-6.csv")
    return video


def _launch_capturing_sessions(qtbot, tmp_path, monkeypatch):
    """Drive _on_launch, stubbing everything past pose-CSV resolution."""
    from glider.gui.behavior import window as win_mod
    from glider.gui.behavior.annotator import main_window as annot_mod
    from glider.gui.behavior.annotator import sampler as sampler_mod

    captured: dict = {}

    def fake_propose(sessions, n_clips_total, fps=30.0, **kw):
        captured["sessions"] = list(sessions)
        captured["fps"] = fps
        return []

    class FakeAnnotator:
        def __init__(self, **kw):
            captured["annotator_kwargs"] = kw

        def show(self):
            pass

        def warn_about_load_errors(self):
            return False

    warnings_shown: list[str] = []
    monkeypatch.setattr(sampler_mod, "propose_clips_multi", fake_propose)
    monkeypatch.setattr(annot_mod, "AnnotatorWindow", FakeAnnotator)
    monkeypatch.setattr(
        win_mod.QMessageBox,
        "warning",
        lambda *a, **k: warnings_shown.append(a[2] if len(a) > 2 else ""),
    )
    monkeypatch.setattr(
        win_mod.QMessageBox,
        "critical",
        lambda *a, **k: warnings_shown.append(a[2] if len(a) > 2 else ""),
    )

    tab = win_mod.AnnotateTab(tmp_path)
    qtbot.addWidget(tab)
    tab._videos_dir = tmp_path
    tab._on_launch()
    return captured, warnings_shown


def test_annotate_accepts_batch_pose_tracking_output(qtbot, tmp_path, monkeypatch):
    """The regression: this folder used to read as 'missing pose CSV'."""
    _pose_batch_output(tmp_path, "session01")

    captured, warnings_shown = _launch_capturing_sessions(qtbot, tmp_path, monkeypatch)

    assert warnings_shown == []
    (pose_csv, video) = captured["sessions"][0]
    assert pose_csv.name == "session01DLC_exp-6.csv"
    assert video.name == "session01.mp4"


def test_annotate_uses_the_recorded_frame_rate(qtbot, tmp_path, monkeypatch):
    """A 60 fps recording must not be clipped and trimmed as though it were 30."""
    _pose_batch_output(tmp_path, "session01", fps=60.0)

    captured, warnings_shown = _launch_capturing_sessions(qtbot, tmp_path, monkeypatch)

    assert warnings_shown == []
    assert captured["fps"] == pytest.approx(60.0)
    assert captured["annotator_kwargs"]["fps"] == pytest.approx(60.0)


def test_annotate_still_reports_genuinely_missing_pose_data(qtbot, tmp_path, monkeypatch):
    (tmp_path / "session01.mp4").touch()

    captured, warnings_shown = _launch_capturing_sessions(qtbot, tmp_path, monkeypatch)

    assert "sessions" not in captured
    assert len(warnings_shown) == 1
    assert "session01.mp4" in warnings_shown[0]
    assert "Batch Pose Tracking" in warnings_shown[0]
