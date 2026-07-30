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
