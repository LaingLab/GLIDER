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
# Classifier cadence (predict_every)
# --------------------------------------------------------------------------


def test_apply_tab_cadence_default_matches_the_pipeline(qtbot):
    """The tab hardcodes the default (import-light module); pin it to the real one."""
    from glider.analysis.behavior.classify.pipeline import LiveInferenceConfig
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    assert tab._predict_every.value() == LiveInferenceConfig.predict_every


def test_cadence_cannot_be_set_below_every_frame(qtbot):
    """0 would mean 'never predict'; the pipeline clamps, but don't offer it."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._predict_every.setValue(0)
    assert tab._predict_every.value() == 1


def test_cadence_hint_reports_every_frame_at_one(qtbot):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._predict_every.setValue(1)
    assert "every frame" in tab._cadence_hint.text()
    tab._predict_every.setValue(3)
    assert "every frame" not in tab._cadence_hint.text()


def test_apply_worker_forwards_predict_every_to_classify(monkeypatch, tmp_path):
    from glider.gui.behavior import workers

    seen = {}
    monkeypatch.setattr(workers, "classify", lambda video, **kw: seen.update(kw) or "r")
    workers.ApplyWorker(
        video=tmp_path / "v.mp4",
        model_path=tmp_path / "m.pkl",
        yolo_path=tmp_path / "y.pt",
        keypoint_names=["a"],
        output_dir=tmp_path / "out",
        predict_every=1,
    ).run()
    assert seen["predict_every"] == 1


def test_apply_worker_omits_predict_every_when_unset(monkeypatch, tmp_path):
    """Unset must leave the pipeline default as the single source of truth."""
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
    assert "predict_every" not in seen


def test_predict_every_is_a_real_pipeline_config_field():
    """Regression guard: it reaches LiveInferenceConfig via classify's **opts."""
    import inspect

    from glider.analysis.behavior.classify.pipeline import LiveInferenceConfig

    assert "predict_every" in inspect.signature(LiveInferenceConfig).parameters


# --------------------------------------------------------------------------
# Advanced LightGBM knobs
# --------------------------------------------------------------------------


def test_lgbm_knobs_cover_the_dataclass():
    """The UI spec duplicates LgbmReg's defaults; pin them so they can't drift.

    A new LgbmReg field that nobody adds to _LGBM_KNOBS would be silently
    untunable from the GUI, which is exactly the bug this dialog exists to fix.
    """
    import dataclasses
    import inspect

    from glider.analysis.behavior.pipeline import LgbmReg, train_model
    from glider.gui.behavior.window import _LGBM_KNOBS

    knobs = {k.name: k.default for k in _LGBM_KNOBS}
    fields = {f.name: f.default for f in dataclasses.fields(LgbmReg)}

    # n_estimators is train_model's, not LgbmReg's -- everything else pairs up.
    assert set(knobs) - {"n_estimators"} == set(fields)
    for name, default in fields.items():
        assert knobs[name] == pytest.approx(default), name
    assert (
        knobs["n_estimators"] == inspect.signature(train_model).parameters["n_estimators"].default
    )


def test_every_knob_default_sits_inside_its_own_range():
    from glider.gui.behavior.window import _LGBM_KNOBS

    for knob in _LGBM_KNOBS:
        assert knob.minimum <= knob.default <= knob.maximum, knob.name


def test_dialog_opens_on_the_defaults(qtbot):
    from glider.gui.behavior.window import _LGBM_KNOBS, LgbmAdvancedDialog

    dialog = LgbmAdvancedDialog()
    qtbot.addWidget(dialog)
    assert dialog.values() == {k.name: k.default for k in _LGBM_KNOBS}


def test_dialog_round_trips_edited_values(qtbot):
    from glider.gui.behavior.window import LgbmAdvancedDialog

    dialog = LgbmAdvancedDialog({"num_leaves": 63, "learning_rate": 0.05})
    qtbot.addWidget(dialog)
    values = dialog.values()
    assert values["num_leaves"] == 63
    assert values["learning_rate"] == pytest.approx(0.05)
    # Unspecified knobs still come back at their defaults.
    assert values["reg_lambda"] == pytest.approx(1.0)


def test_restore_defaults_puts_every_knob_back(qtbot):
    from glider.gui.behavior.window import _LGBM_KNOBS, LgbmAdvancedDialog

    dialog = LgbmAdvancedDialog({"num_leaves": 200, "bagging_fraction": 0.5})
    qtbot.addWidget(dialog)
    dialog.restore_defaults()
    assert dialog.values() == {k.name: k.default for k in _LGBM_KNOBS}


def test_max_depth_offers_unlimited_as_its_minimum(qtbot):
    """-1 is a sentinel, not a depth; it must not read as a number in the UI."""
    from glider.gui.behavior.window import LgbmAdvancedDialog

    dialog = LgbmAdvancedDialog()
    qtbot.addWidget(dialog)
    spin = dialog._spins["max_depth"]
    assert spin.minimum() == -1
    assert spin.specialValueText() == "No limit"
    assert dialog.values()["max_depth"] == -1


def test_advanced_button_is_lightgbm_only(qtbot):
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    tab._classifier_combo.setCurrentText("rf")
    assert not tab._advanced_btn.isEnabled()
    tab._classifier_combo.setCurrentText("lightgbm")
    assert tab._advanced_btn.isEnabled()


def test_untouched_knobs_send_nothing_to_train_model(qtbot):
    """Defaults must come from the library, not be frozen in by the GUI."""
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    tab._classifier_combo.setCurrentText("lightgbm")
    assert tab._lgbm_options() == {}


def test_accepted_knobs_become_an_lgbm_reg(qtbot):
    from glider.analysis.behavior.pipeline import LgbmReg
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    tab._classifier_combo.setCurrentText("lightgbm")
    tab._lgbm_advanced = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 25,
        "min_split_gain": 0.1,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.9,
        "reg_lambda": 2.0,
    }
    opts = tab._lgbm_options()
    assert opts["n_estimators"] == 400
    assert opts["lgbm_reg"] == LgbmReg(
        num_leaves=63,
        min_child_samples=25,
        feature_fraction=0.7,
        bagging_fraction=0.9,
        reg_lambda=2.0,
        learning_rate=0.05,
        max_depth=8,
        min_split_gain=0.1,
    )


def test_random_forest_run_never_carries_lgbm_knobs(qtbot):
    """_make_classifier ignores lgbm_reg for rf; don't imply an effect."""
    from glider.gui.behavior.window import TrainTab

    tab = TrainTab()
    qtbot.addWidget(tab)
    tab._lgbm_advanced = {"n_estimators": 400, "num_leaves": 63}
    tab._classifier_combo.setCurrentText("rf")
    assert tab._lgbm_options() == {}


def test_dialog_values_are_all_accepted_by_train_model(qtbot):
    """GUI <-> core contract, matching the Apply tab's leak guard."""
    import dataclasses
    import inspect

    from glider.analysis.behavior.pipeline import LgbmReg, train_model
    from glider.gui.behavior.window import LgbmAdvancedDialog

    dialog = LgbmAdvancedDialog()
    qtbot.addWidget(dialog)
    emitted = set(dialog.values())

    accepted = set(inspect.signature(train_model).parameters)
    lgbm_fields = {f.name for f in dataclasses.fields(LgbmReg)}
    for key in emitted:
        assert key in accepted or key in lgbm_fields, f"{key} reaches neither"


def test_advanced_dialog_stores_values_on_accept(qtbot, monkeypatch):
    from glider.gui.behavior import window as win

    tab = win.TrainTab()
    qtbot.addWidget(tab)
    tab._classifier_combo.setCurrentText("lightgbm")
    monkeypatch.setattr(win.LgbmAdvancedDialog, "exec", lambda self: 1)
    tab._on_advanced()
    assert tab._lgbm_advanced is not None
    assert tab._lgbm_advanced["num_leaves"] == 31


def test_advanced_dialog_discards_values_on_cancel(qtbot, monkeypatch):
    from glider.gui.behavior import window as win

    tab = win.TrainTab()
    qtbot.addWidget(tab)
    tab._classifier_combo.setCurrentText("lightgbm")
    monkeypatch.setattr(win.LgbmAdvancedDialog, "exec", lambda self: 0)
    tab._on_advanced()
    assert tab._lgbm_advanced is None
