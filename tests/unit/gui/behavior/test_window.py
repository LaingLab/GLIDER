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


def test_percentile_mode_sends_percentiles_not_absolute_thresholds(qtbot, tmp_path):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("percentile"))
    tab._freeze_pct.setValue(5.0)
    tab._dart_pct.setValue(99.0)

    opts = tab._speed_opts()
    assert opts["freeze_pct"] == pytest.approx(5.0)
    assert opts["dart_pct"] == pytest.approx(99.0)
    assert "freeze_cm_s" not in opts


def test_the_calibration_is_sent_in_percentile_mode_too(qtbot, tmp_path):
    """Percentile thresholds need no scale, but the ethogram's cm/s does."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._calibration_master = tmp_path / "pose_calibration.json"
    for mode in ("absolute", "percentile"):
        tab._speed_mode.setCurrentIndex(tab._speed_mode.findData(mode))
        opts = tab._speed_opts()
        assert opts["calibration_master"] == tmp_path / "pose_calibration.json", mode


def test_the_annotated_video_is_off_by_default(qtbot):
    """Encoding it costs more than the inference on a long recording."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    assert tab._render_video.isChecked() is False


def test_the_video_toggle_reaches_the_worker(qtbot, tmp_path, monkeypatch):
    from glider.gui.behavior import workers

    seen = {}
    monkeypatch.setattr(workers, "classify", lambda video, **kw: seen.update(kw) or "r")
    workers.ApplyWorker(
        video=tmp_path / "v.mp4",
        model_path=tmp_path / "m.pkl",
        yolo_path=tmp_path / "y.pt",
        keypoint_names=["a"],
        output_dir=tmp_path / "out",
        write_annotated=True,
    ).run()
    assert seen["write_annotated"] is True


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
    pose_csv, video = captured["sessions"][0]
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


# --------------------------------------------------------------------------
# Annotate tab: reviewing what is already labelled, and asking for more clips
#
# The tab used to hardcode `n_clips_total=max(50, len(videos))` and had no way
# to reach existing annotations at all. Pointed at a 30-video cohort with 2,352
# saved zones it opened a queue of 50 freshly-sampled clips, so the labelling
# work already on disk was invisible.
# --------------------------------------------------------------------------


def _saved_zones(tmp_path, stem, spans, behavior="walking"):
    """Write an annotations CSV beside the pose CSV, where training reads it."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.app import annotation_path_for

    store = AnnotationStore()
    for start, end in spans:
        store.add(BehaviorZone(behavior=behavior, start_frame=start, end_frame=end))
    store.save_csv(annotation_path_for(tmp_path / f"{stem}DLC_exp-6.csv"))
    return store


def _launch_annotate(qtbot, tmp_path, monkeypatch, configure=None):
    """Drive _on_launch with the sampler and annotator stubbed out.

    ``configure(tab)`` runs after construction so a test can set the new
    controls before launching.
    """
    from glider.gui.behavior import window as win_mod
    from glider.gui.behavior.annotator import main_window as annot_mod
    from glider.gui.behavior.annotator import sampler as sampler_mod

    captured: dict = {}

    def fake_propose(sessions, n_clips_total, fps=30.0, **kw):
        captured["sessions"] = list(sessions)
        captured["n_clips_total"] = n_clips_total
        captured["exclude_zones_by_session"] = kw.get("exclude_zones_by_session")
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
    for kind in ("warning", "critical"):
        monkeypatch.setattr(
            win_mod.QMessageBox,
            kind,
            lambda *a, **k: warnings_shown.append(a[2] if len(a) > 2 else ""),
        )

    tab = win_mod.AnnotateTab(tmp_path)
    qtbot.addWidget(tab)
    tab._videos_dir = tmp_path
    if configure is not None:
        configure(tab)
    tab._on_launch()
    return captured, warnings_shown


def test_annotate_clip_count_defaults_to_fifty(qtbot, tmp_path, monkeypatch):
    """The old hardcode becomes the default, so nothing changes unasked."""
    _pose_batch_output(tmp_path, "session01")

    captured, warnings_shown = _launch_annotate(qtbot, tmp_path, monkeypatch)

    assert warnings_shown == []
    assert captured["n_clips_total"] == 50


def test_annotate_clip_count_is_configurable(qtbot, tmp_path, monkeypatch):
    """A 30-video cohort needs far more than 50 clips to label usefully."""
    _pose_batch_output(tmp_path, "session01")

    captured, _ = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._clip_count.setValue(1000)
    )

    assert captured["n_clips_total"] == 1000


def test_annotate_clip_count_never_drops_below_the_video_count(qtbot, tmp_path, monkeypatch):
    """propose_clips_multi rejects n_clips_total < len(sessions); don't hand it one."""
    for i in range(3):
        _pose_batch_output(tmp_path, f"session{i:02d}")

    captured, warnings_shown = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._clip_count.setValue(1)
    )

    assert warnings_shown == []
    assert captured["n_clips_total"] == 3


def test_review_mode_loads_every_saved_zone(qtbot, tmp_path, monkeypatch):
    """The bug: 2,352 saved zones were on disk and none of them reached the queue."""
    _pose_batch_output(tmp_path, "session01")
    _pose_batch_output(tmp_path, "session02")
    _saved_zones(tmp_path, "session01", [(0, 10), (20, 30), (40, 50)])
    _saved_zones(tmp_path, "session02", [(5, 15), (25, 35)])

    captured, warnings_shown = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._review_check.setChecked(True)
    )

    assert warnings_shown == []
    # Sampling is skipped entirely in review mode.
    assert "n_clips_total" not in captured
    clips = captured["annotator_kwargs"]["clips"]
    assert len(clips) == 5
    assert {(c.start_frame, c.end_frame) for c in clips} == {
        (0, 10),
        (20, 30),
        (40, 50),
        (5, 15),
        (25, 35),
    }


def test_review_mode_says_so_when_nothing_is_labelled_yet(qtbot, tmp_path, monkeypatch):
    """An empty review window looks identical to a broken one. Say which it is."""
    _pose_batch_output(tmp_path, "session01")

    captured, warnings_shown = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._review_check.setChecked(True)
    )

    assert "annotator_kwargs" not in captured
    assert len(warnings_shown) == 1
    assert "no annotations" in warnings_shown[0].lower()


def test_skipping_labelled_regions_passes_the_saved_zones(qtbot, tmp_path, monkeypatch):
    """Continuing a cohort must not re-propose frames already labelled."""
    _pose_batch_output(tmp_path, "session01")
    _saved_zones(tmp_path, "session01", [(0, 10), (20, 30)])

    captured, _ = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._skip_labelled_check.setChecked(True)
    )

    assert captured["exclude_zones_by_session"] == [[(0, 10), (20, 30)]]


def test_not_skipping_labelled_regions_excludes_nothing(qtbot, tmp_path, monkeypatch):
    _pose_batch_output(tmp_path, "session01")
    _saved_zones(tmp_path, "session01", [(0, 10)])

    captured, _ = _launch_annotate(qtbot, tmp_path, monkeypatch)

    assert captured["exclude_zones_by_session"] is None


def test_annotate_wires_up_the_render_more_button(qtbot, tmp_path, monkeypatch):
    """Without a sampler the window's 'render more' button never appears."""
    _pose_batch_output(tmp_path, "session01")

    captured, _ = _launch_annotate(qtbot, tmp_path, monkeypatch)

    assert callable(captured["annotator_kwargs"]["clip_sampler"])


def test_review_mode_also_wires_up_the_render_more_button(qtbot, tmp_path, monkeypatch):
    """Reviewing and then wanting fresh material is the normal workflow."""
    _pose_batch_output(tmp_path, "session01")
    _saved_zones(tmp_path, "session01", [(0, 10)])

    captured, _ = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._review_check.setChecked(True)
    )

    assert callable(captured["annotator_kwargs"]["clip_sampler"])


def test_annotate_passes_pose_csvs_to_the_annotator(qtbot, tmp_path, monkeypatch):
    """The trace needs pose data; the tab already resolved it for the sampler."""
    _pose_batch_output(tmp_path, "session01")

    captured, _ = _launch_annotate(qtbot, tmp_path, monkeypatch)

    pose_csvs = captured["annotator_kwargs"]["pose_csvs"]
    assert list(pose_csvs.values())[0].name == "session01DLC_exp-6.csv"


def test_annotate_sends_no_cohort_when_none_is_chosen(qtbot, tmp_path, monkeypatch):
    _pose_batch_output(tmp_path, "session01")

    captured, _ = _launch_annotate(qtbot, tmp_path, monkeypatch)

    assert captured["annotator_kwargs"]["cohort"] is None
    assert captured["annotator_kwargs"]["px_per_mm"] == {}


def test_annotate_loads_and_forwards_a_cohort_threshold_file(qtbot, tmp_path, monkeypatch):
    from glider.analysis.behavior.cohort_speed import CohortSpeedThresholds

    _pose_batch_output(tmp_path, "session01")
    cohort_path = tmp_path / "cohort_speed.json"
    CohortSpeedThresholds(
        freeze=0.5,
        dart=27.7,
        unit="cm/s",
        freeze_pct=10.0,
        dart_pct=99.5,
        n_sessions=30,
        n_samples=1000,
    ).save(cohort_path)

    captured, warnings_shown = _launch_annotate(
        qtbot,
        tmp_path,
        monkeypatch,
        configure=lambda t: setattr(t, "_cohort_path", cohort_path),
    )

    assert warnings_shown == []
    cohort = captured["annotator_kwargs"]["cohort"]
    assert (cohort.freeze, cohort.dart, cohort.unit) == (0.5, 27.7, "cm/s")


def test_a_broken_cohort_file_warns_and_does_not_block_labelling(qtbot, tmp_path, monkeypatch):
    """A bad threshold file costs you the lines, not the session."""
    _pose_batch_output(tmp_path, "session01")
    bad = tmp_path / "cohort_speed.json"
    bad.write_text("{not json", encoding="utf-8")

    captured, warnings_shown = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: setattr(t, "_cohort_path", bad)
    )

    assert len(warnings_shown) == 1
    assert "cohort" in warnings_shown[0].lower()
    # The annotator still opened, just without thresholds.
    assert captured["annotator_kwargs"]["cohort"] is None


def test_annotate_resolves_a_pixel_scale_per_video(qtbot, tmp_path, monkeypatch):
    """cm/s needs each video's own scale, looked up the way apply does."""
    from glider.analysis.behavior import units as units_mod

    _pose_batch_output(tmp_path, "session01")
    master = tmp_path / "pose_calibration.json"
    master.touch()
    monkeypatch.setattr(units_mod, "load_px_per_mm", lambda _m, _v: 1.35)

    captured, _ = _launch_annotate(
        qtbot,
        tmp_path,
        monkeypatch,
        configure=lambda t: setattr(t, "_calibration_master", master),
    )

    assert list(captured["annotator_kwargs"]["px_per_mm"].values()) == [1.35]


def test_an_uncalibrated_video_is_simply_absent_from_the_scales(qtbot, tmp_path, monkeypatch):
    """No scale means a px/frame trace for that video, not a crash."""
    from glider.analysis.behavior import units as units_mod

    _pose_batch_output(tmp_path, "session01")
    master = tmp_path / "pose_calibration.json"
    master.touch()
    monkeypatch.setattr(units_mod, "load_px_per_mm", lambda _m, _v: None)

    captured, _ = _launch_annotate(
        qtbot,
        tmp_path,
        monkeypatch,
        configure=lambda t: setattr(t, "_calibration_master", master),
    )

    assert captured["annotator_kwargs"]["px_per_mm"] == {}


def test_turning_the_trace_off_sends_no_pose_csvs(qtbot, tmp_path, monkeypatch):
    """Opting out must skip the pose reads entirely, not just hide a widget."""
    _pose_batch_output(tmp_path, "session01")

    captured, _ = _launch_annotate(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._speed_check.setChecked(False)
    )

    assert captured["annotator_kwargs"]["pose_csvs"] == {}


# --------------------------------------------------------------------------
# Train tab: adding sessions in bulk
#
# "Add session..." opened TWO single-file dialogs per session. A 30-video
# cohort cost 60 dialogs, and "Remove selected" then removed one row at a time.
# --------------------------------------------------------------------------


def _session_files(tmp_path, stem, with_annotations=True):
    """A pose CSV, and by default the annotations file training looks for."""
    pose = tmp_path / f"{stem}.csv"
    pose.write_text("pose\n", encoding="utf-8")
    if with_annotations:
        (tmp_path / f"{stem}_annotations.csv").write_text(
            "behavior,start_frame,end_frame,created_at,note\n", encoding="utf-8"
        )
    return pose


def _train_tab(qtbot, monkeypatch, chosen, asked=None):
    """A TrainTab whose file dialogs return ``chosen`` (and ``asked`` as fallback)."""
    from glider.gui.behavior import window as win_mod

    notes: list[str] = []
    monkeypatch.setattr(
        win_mod.QFileDialog,
        "getOpenFileNames",
        staticmethod(lambda *a, **k: ([str(p) for p in chosen], "")),
    )
    monkeypatch.setattr(
        win_mod.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *a, **k: (str(asked) if asked else "", "")),
    )
    for kind in ("information", "warning"):
        monkeypatch.setattr(
            win_mod.QMessageBox,
            kind,
            lambda *a, **k: notes.append(a[2] if len(a) > 2 else ""),
        )
    tab = win_mod.TrainTab()
    qtbot.addWidget(tab)
    return tab, notes


def test_add_sessions_pairs_every_selected_pose_csv(qtbot, tmp_path, monkeypatch):
    poses = [_session_files(tmp_path, f"s{i}") for i in range(3)]

    tab, _ = _train_tab(qtbot, monkeypatch, poses)
    tab._on_add_session()

    assert len(tab._sessions) == 3
    assert tab._sessions_list.count() == 3
    for pose, (got_pose, got_ann) in zip(poses, tab._sessions, strict=True):
        assert got_pose == pose
        assert got_ann.name == f"{pose.stem}_annotations.csv"


def test_add_sessions_skips_pose_csvs_with_no_annotations(qtbot, tmp_path, monkeypatch):
    """Never guess at a missing annotations file; say which ones were skipped."""
    good = [_session_files(tmp_path, "good1"), _session_files(tmp_path, "good2")]
    orphan = _session_files(tmp_path, "orphan", with_annotations=False)

    tab, notes = _train_tab(qtbot, monkeypatch, [*good, orphan])
    tab._on_add_session()

    assert len(tab._sessions) == 2
    assert len(notes) == 1
    assert "orphan.csv" in notes[0]


def test_add_sessions_ignores_annotation_files_chosen_by_mistake(qtbot, tmp_path, monkeypatch):
    """The folder holds both kinds; <stem>_annotations_annotations.csv is nobody's intent."""
    pose = _session_files(tmp_path, "s1")
    annotations = tmp_path / "s1_annotations.csv"

    tab, notes = _train_tab(qtbot, monkeypatch, [pose, annotations])
    tab._on_add_session()

    assert len(tab._sessions) == 1
    assert tab._sessions[0][0] == pose
    assert "s1_annotations.csv" in notes[0]


def test_add_sessions_does_not_add_the_same_session_twice(qtbot, tmp_path, monkeypatch):
    pose = _session_files(tmp_path, "s1")

    tab, notes = _train_tab(qtbot, monkeypatch, [pose])
    tab._on_add_session()
    tab._on_add_session()

    assert len(tab._sessions) == 1
    assert tab._sessions_list.count() == 1
    assert "already" in notes[-1].lower()


def test_one_pose_csv_without_annotations_falls_back_to_asking(qtbot, tmp_path, monkeypatch):
    """Annotations kept somewhere unrelated must still be reachable."""
    pose = _session_files(tmp_path, "s1", with_annotations=False)
    elsewhere = tmp_path / "somewhere_else.csv"
    elsewhere.write_text("behavior,start_frame,end_frame,created_at,note\n", encoding="utf-8")

    tab, _ = _train_tab(qtbot, monkeypatch, [pose], asked=elsewhere)
    tab._on_add_session()

    assert tab._sessions == [(pose, elsewhere)]


def test_declining_the_fallback_adds_nothing(qtbot, tmp_path, monkeypatch):
    pose = _session_files(tmp_path, "s1", with_annotations=False)

    tab, _ = _train_tab(qtbot, monkeypatch, [pose], asked=None)
    tab._on_add_session()

    assert tab._sessions == []


def test_cancelling_the_dialog_adds_nothing(qtbot, tmp_path, monkeypatch):
    tab, notes = _train_tab(qtbot, monkeypatch, [])
    tab._on_add_session()

    assert tab._sessions == []
    assert notes == []


def test_holdout_sessions_are_added_in_bulk_too(qtbot, tmp_path, monkeypatch):
    poses = [_session_files(tmp_path, f"h{i}") for i in range(3)]

    tab, _ = _train_tab(qtbot, monkeypatch, poses)
    tab._on_add_holdout()

    assert len(tab._holdout) == 3
    assert tab._holdout_list.count() == 3
    assert tab._sessions == []


def test_both_lists_allow_multi_selection(qtbot, monkeypatch):
    from PyQt6.QtWidgets import QAbstractItemView

    tab, _ = _train_tab(qtbot, monkeypatch, [])
    mode = QAbstractItemView.SelectionMode.ExtendedSelection
    assert tab._sessions_list.selectionMode() == mode
    assert tab._holdout_list.selectionMode() == mode


def test_remove_selected_removes_every_selected_row(qtbot, tmp_path, monkeypatch):
    """It removed only currentRow, so clearing 30 sessions meant 30 clicks."""
    poses = [_session_files(tmp_path, f"s{i}") for i in range(5)]

    tab, _ = _train_tab(qtbot, monkeypatch, poses)
    tab._on_add_session()
    for row in (0, 2, 4):
        tab._sessions_list.item(row).setSelected(True)
    tab._on_remove_session()

    assert [p.stem for p, _a in tab._sessions] == ["s1", "s3"]
    assert tab._sessions_list.count() == 2
    # The list and its backing store must not drift apart. Rows show names;
    # the tooltip carries the full paths.
    for i, (pose, ann) in enumerate(tab._sessions):
        item = tab._sessions_list.item(i)
        assert pose.name in item.text()
        assert item.toolTip() == f"{pose}\n{ann}"


def test_removing_nothing_selected_is_a_no_op(qtbot, tmp_path, monkeypatch):
    poses = [_session_files(tmp_path, f"s{i}") for i in range(2)]

    tab, _ = _train_tab(qtbot, monkeypatch, poses)
    tab._on_add_session()
    tab._sessions_list.clearSelection()
    tab._on_remove_session()

    assert len(tab._sessions) == 2


# --------------------------------------------------------------------------
# Train tab: the knobs that actually move the score
#
# train_model takes window, class_weight, test_split, random_state and three
# feature-family flags. None of them were reachable from the GUI, so every
# model trained here used stock defaults. Measured over 5 cross-session folds,
# the settings an earlier hand-tuned model used were worth +0.063 macro F1,
# winning 5 folds out of 5 — roughly fifteen times the effect of anything else
# tried on this cohort.
# --------------------------------------------------------------------------


def _fit_options(qtbot, tmp_path, monkeypatch, configure=None):
    """Drive _on_fit and return the options dict handed to TrainWorker."""
    from glider.gui.behavior import window as win_mod
    from glider.gui.behavior import workers as workers_mod

    captured: dict = {}

    class FakeWorker:
        progress = failed = finished = None

        def __init__(self, sessions, output, options):
            captured["sessions"] = sessions
            captured["options"] = options

        def moveToThread(self, _t):
            pass

    monkeypatch.setattr(workers_mod, "TrainWorker", FakeWorker, raising=False)

    tab = win_mod.TrainTab()
    qtbot.addWidget(tab)
    tab._sessions = [(tmp_path / "a.csv", tmp_path / "a_annotations.csv")]
    tab._output_path = tmp_path / "model.pkl"
    if configure is not None:
        configure(tab)
    try:
        tab._on_fit()
    except Exception:  # noqa: BLE001 - thread wiring past the capture point
        pass
    return tab, captured.get("options", {})


def test_window_defaults_to_thirty_and_is_sent(qtbot, tmp_path, monkeypatch):
    tab, options = _fit_options(qtbot, tmp_path, monkeypatch)
    assert tab._window_spin.value() == 30
    assert options.get("window") == 30


def test_window_is_configurable(qtbot, tmp_path, monkeypatch):
    """The 0.78 model used window 8; stock is 30, and it is worth ~0.06 F1."""
    _tab, options = _fit_options(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._window_spin.setValue(8)
    )
    assert options["window"] == 8


def test_class_weight_defaults_to_none_and_is_sent_as_none(qtbot, tmp_path, monkeypatch):
    """train_model wants None, not the string 'none'."""
    _tab, options = _fit_options(qtbot, tmp_path, monkeypatch)
    assert options.get("class_weight") is None


def test_class_weight_balanced_is_sent(qtbot, tmp_path, monkeypatch):
    def pick(tab):
        tab._class_weight_combo.setCurrentIndex(tab._class_weight_combo.findData("balanced"))

    _tab, options = _fit_options(qtbot, tmp_path, monkeypatch, configure=pick)
    assert options["class_weight"] == "balanced"


def test_random_seed_is_sent(qtbot, tmp_path, monkeypatch):
    _tab, options = _fit_options(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._seed_spin.setValue(7)
    )
    assert options["random_state"] == 7


def test_test_split_is_sent_only_when_set(qtbot, tmp_path, monkeypatch):
    _tab, options = _fit_options(qtbot, tmp_path, monkeypatch)
    assert options.get("test_split", 0.0) == 0.0

    _tab, options = _fit_options(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._test_split_spin.setValue(0.25)
    )
    assert options["test_split"] == pytest.approx(0.25)


def test_feature_family_flags_default_off_and_are_sent(qtbot, tmp_path, monkeypatch):
    """traj/motion/freq exist in train_model and defaulted off with no way to
    turn them on. Defaults stay off so nothing changes unasked."""
    _tab, options = _fit_options(qtbot, tmp_path, monkeypatch)
    for flag in ("traj_features", "motion_features", "freq_features"):
        assert options.get(flag) is False


def test_feature_family_flags_can_be_enabled(qtbot, tmp_path, monkeypatch):
    def enable(tab):
        tab._traj_check.setChecked(True)
        tab._motion_check.setChecked(True)
        tab._freq_check.setChecked(True)

    _tab, options = _fit_options(qtbot, tmp_path, monkeypatch, configure=enable)
    assert options["traj_features"] is True
    assert options["motion_features"] is True
    assert options["freq_features"] is True


def test_the_new_knobs_stay_available_for_randomforest(qtbot, tmp_path, monkeypatch):
    """These are pipeline knobs, not LightGBM ones — unlike the Advanced
    dialog, which is disabled for rf. Putting them there would have made them
    unreachable with the RandomForest backend."""

    def pick_rf(tab):
        tab._classifier_combo.setCurrentText("rf")

    tab, options = _fit_options(qtbot, tmp_path, monkeypatch, configure=pick_rf)
    assert tab._window_spin.isEnabled()
    assert tab._class_weight_combo.isEnabled()
    assert options["window"] == 30
    assert options["classifier_type"] == "rf"


def test_train_model_accepts_every_option_the_train_tab_can_send(qtbot, tmp_path, monkeypatch):
    """The GUI/core contract, mirroring the Apply tab's equivalent.

    A mistyped kwarg here would only surface as a TypeError *after* a fit
    that can take ten minutes, with the model lost.
    """
    import inspect

    from glider.analysis.behavior.pipeline import train_model

    def enable_everything(tab):
        tab._traj_check.setChecked(True)
        tab._motion_check.setChecked(True)
        tab._freq_check.setChecked(True)
        tab._background_check.setChecked(True)
        tab._mirror_check.setChecked(True)
        tab._test_split_spin.setValue(0.2)
        tab._class_weight_combo.setCurrentIndex(tab._class_weight_combo.findData("balanced"))
        tab._lgbm_advanced = {k.name: k.default for k in win_lgbm_knobs()}

    def win_lgbm_knobs():
        from glider.gui.behavior.window import _LGBM_KNOBS

        return _LGBM_KNOBS

    _tab, options = _fit_options(qtbot, tmp_path, monkeypatch, configure=enable_everything)
    assert options, "no options captured"

    accepted = set(inspect.signature(train_model).parameters)
    unknown = set(options) - accepted
    assert not unknown, f"train_model does not accept: {sorted(unknown)}"


# --------------------------------------------------------------------------
# Motion features and mirror augmentation are mutually exclusive
#
# train_model raises outright: "motion features are not supported with mirror
# augmentation yet (the source video isn't mirrored)". Both were offered as
# free-standing checkboxes, so ticking both killed the run with a ValueError
# AFTER the fit had started -- minutes in, with the model lost.
# --------------------------------------------------------------------------


def test_the_pipeline_really_does_refuse_the_combination():
    """Pins the constraint this UI rule exists to respect."""
    import inspect

    from glider.analysis.behavior import pipeline

    source = inspect.getsource(pipeline._assemble_sessions)
    assert "motion_features and mirror_augment" in source


def test_checking_mirror_disables_motion(qtbot, tmp_path, monkeypatch):
    tab, _ = _fit_options(qtbot, tmp_path, monkeypatch)
    tab._mirror_check.setChecked(True)
    assert tab._motion_check.isEnabled() is False
    assert tab._motion_check.isChecked() is False


def test_checking_motion_disables_mirror(qtbot, tmp_path, monkeypatch):
    tab, _ = _fit_options(qtbot, tmp_path, monkeypatch)
    tab._motion_check.setChecked(True)
    assert tab._mirror_check.isEnabled() is False
    assert tab._mirror_check.isChecked() is False


def test_unchecking_one_frees_the_other_again(qtbot, tmp_path, monkeypatch):
    tab, _ = _fit_options(qtbot, tmp_path, monkeypatch)
    tab._mirror_check.setChecked(True)
    tab._mirror_check.setChecked(False)
    assert tab._motion_check.isEnabled() is True
    assert tab._mirror_check.isEnabled() is True


def test_the_two_can_never_both_reach_train_model(qtbot, tmp_path, monkeypatch):
    """Whatever the user clicks, the options dict must not carry both."""

    def click_both(tab):
        tab._mirror_check.setChecked(True)
        tab._motion_check.setChecked(True)  # blocked by the rule above

    _tab, options = _fit_options(qtbot, tmp_path, monkeypatch, configure=click_both)
    assert not (options.get("motion_features") and options.get("mirror_augment"))


# --------------------------------------------------------------------------
# Train tab: cross-validation
#
# cross_validate_sessions has been in the pipeline all along and was
# unreachable from the GUI, so the only evaluation available was a single
# hand-picked holdout. On this data a 3-animal holdout has sd 0.09 and a range
# of 0.518-0.804, which makes one split close to meaningless.
# --------------------------------------------------------------------------


def _cv_options(qtbot, tmp_path, monkeypatch, configure=None):
    """Drive _on_cross_validate and return the options handed to the worker."""
    from glider.gui.behavior import window as win_mod
    from glider.gui.behavior import workers as workers_mod

    captured: dict = {}
    warned: list[str] = []

    class FakeWorker:
        progress = failed = finished = None

        def __init__(self, sessions, options):
            captured["sessions"] = sessions
            captured["options"] = options

        def moveToThread(self, _t):
            pass

    monkeypatch.setattr(workers_mod, "CrossValidateWorker", FakeWorker, raising=False)
    monkeypatch.setattr(
        win_mod.QMessageBox, "warning", lambda *a, **k: warned.append(a[2] if len(a) > 2 else "")
    )

    tab = win_mod.TrainTab()
    qtbot.addWidget(tab)
    tab._sessions = [(tmp_path / "a.csv", tmp_path / "a_annotations.csv")]
    if configure is not None:
        configure(tab)
    try:
        tab._on_cross_validate()
    except Exception:  # noqa: BLE001 - thread wiring past the capture point
        pass
    return tab, captured, warned


def test_cross_validate_sends_the_training_sessions(qtbot, tmp_path, monkeypatch):
    _tab, captured, _ = _cv_options(qtbot, tmp_path, monkeypatch)
    assert len(captured["sessions"]) == 1


def test_cross_validate_defaults_to_five_folds(qtbot, tmp_path, monkeypatch):
    tab, captured, _ = _cv_options(qtbot, tmp_path, monkeypatch)
    assert tab._folds_spin.value() == 5
    assert captured["options"]["n_folds"] == 5


def test_fold_count_is_configurable(qtbot, tmp_path, monkeypatch):
    _tab, captured, _ = _cv_options(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._folds_spin.setValue(10)
    )
    assert captured["options"]["n_folds"] == 10


def test_cross_validate_uses_the_same_settings_as_fit(qtbot, tmp_path, monkeypatch):
    """What you validate has to be what you would fit, or the number is a lie."""

    def configure(tab):
        tab._window_spin.setValue(8)
        tab._class_weight_combo.setCurrentIndex(tab._class_weight_combo.findData("balanced"))
        tab._mirror_check.setChecked(True)
        tab._traj_check.setChecked(True)
        tab._freq_check.setChecked(True)

    _tab, captured, _ = _cv_options(qtbot, tmp_path, monkeypatch, configure=configure)
    options = captured["options"]
    assert options["window"] == 8
    assert options["class_weight"] == "balanced"
    assert options["mirror_augment"] is True
    assert options["traj_features"] is True
    assert options["freq_features"] is True


def test_cross_validate_does_not_send_options_it_does_not_accept(qtbot, tmp_path, monkeypatch):
    """cross_validate_sessions takes no holdout_sessions and no test_split."""
    import inspect

    from glider.analysis.behavior.pipeline import cross_validate_sessions

    _tab, captured, _ = _cv_options(qtbot, tmp_path, monkeypatch)
    accepted = set(inspect.signature(cross_validate_sessions).parameters)
    unknown = set(captured["options"]) - accepted
    assert not unknown, f"cross_validate_sessions does not accept: {sorted(unknown)}"


def test_cross_validate_needs_sessions(qtbot, tmp_path, monkeypatch):
    _tab, captured, warned = _cv_options(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._sessions.clear()
    )
    assert "options" not in captured
    assert warned and "session" in warned[0].lower()


def test_a_failed_run_re_enables_both_buttons(qtbot, tmp_path, monkeypatch):
    """_on_train_failed is shared, so it must revive the cross-validate button
    too — otherwise one failure kills the action for the rest of the session."""
    from glider.gui.behavior import window as win_mod

    monkeypatch.setattr(win_mod.QMessageBox, "critical", lambda *a, **k: None)
    tab = win_mod.TrainTab()
    qtbot.addWidget(tab)
    tab._fit_btn.setEnabled(False)
    tab._cv_btn.setEnabled(False)

    tab._on_train_failed("boom")

    assert tab._fit_btn.isEnabled()
    assert tab._cv_btn.isEnabled()


def test_cross_validate_needs_no_output_file(qtbot, tmp_path, monkeypatch):
    """It produces no model, so demanding an output path would be nonsense."""
    tab, captured, warned = _cv_options(qtbot, tmp_path, monkeypatch)
    assert tab._output_path is None
    assert "options" in captured
    assert warned == []


def test_window_hint_reports_the_duration(qtbot, tmp_path, monkeypatch):
    """30 frames means nothing; 1.0 s at 30 fps does."""
    tab, _ = _fit_options(
        qtbot, tmp_path, monkeypatch, configure=lambda t: t._window_spin.setValue(15)
    )
    assert "0.5" in tab._window_hint.text()


def test_cohort_mode_sends_only_the_cohort_file(qtbot, tmp_path):
    """One pooled cut-off, not per-video percentiles."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._cohort_path = tmp_path / "cohort_speed.json"
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("cohort"))

    opts = tab._speed_opts()
    assert opts["cohort_thresholds"] == tmp_path / "cohort_speed.json"
    assert "freeze_pct" not in opts
    assert "freeze_cm_s" not in opts


def test_cohort_mode_still_sends_the_calibration(qtbot, tmp_path):
    """Cohort cut-offs in cm/s need this video's scale to become pixels."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._calibration_master = tmp_path / "pose_calibration.json"
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("cohort"))
    assert tab._speed_opts()["calibration_master"] == tmp_path / "pose_calibration.json"


def test_switching_to_cohort_hides_the_other_modes_fields(qtbot):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab.show()
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("cohort"))
    assert tab._freeze_cm_s.isVisible() is False
    assert tab._freeze_pct.isVisible() is False


# --------------------------------------------------------------------------
# Cohort pooling must not run on the UI thread
# --------------------------------------------------------------------------


def test_cohort_pooling_runs_on_a_worker_not_the_ui_thread():
    """A real cohort is minutes of work; a frozen UI reads as a crash."""
    import inspect

    from glider.gui.behavior.window import ApplyTab

    source = inspect.getsource(ApplyTab._on_build_cohort)
    assert "CohortSpeedWorker" in source
    assert "moveToThread" in source
    # The blocking call must not be made inline any more.
    assert "compute_cohort_thresholds(" not in source


def test_the_cohort_worker_saves_and_reports(tmp_path, monkeypatch):
    from glider.gui.behavior import workers

    class _Result:
        n_samples, n_sessions, freeze, dart = 10, 2, 0.1, 5.0
        unit, is_calibrated = "cm/s", True
        saved_to = None

        def save(self, path):
            type(self).saved_to = path

    monkeypatch.setattr(
        "glider.analysis.behavior.cohort_speed.compute_cohort_thresholds",
        lambda *a, **k: _Result(),
    )
    worker = workers.CohortSpeedWorker(
        [tmp_path / "aDLC_m.csv"], tmp_path / "cohort.json", freeze_pct=10.0, dart_pct=99.5
    )
    done = []
    worker.finished.connect(done.append)
    worker.run()
    assert done, "finished never fired"
    assert _Result.saved_to == tmp_path / "cohort.json"


def test_a_cohort_failure_is_reported_not_raised(tmp_path, monkeypatch):
    from glider.gui.behavior import workers

    def boom(*a, **k):
        raise ValueError("no usable speed samples")

    monkeypatch.setattr("glider.analysis.behavior.cohort_speed.compute_cohort_thresholds", boom)
    worker = workers.CohortSpeedWorker(
        [tmp_path / "a.csv"], tmp_path / "out.json", freeze_pct=10.0, dart_pct=99.5
    )
    failures = []
    worker.failed.connect(failures.append)
    worker.run()  # must not raise out of the thread
    assert failures == ["no usable speed samples"]


# --------------------------------------------------------------------------
# Apply tab: neither model is unconditionally required
# --------------------------------------------------------------------------


def _apply_tab(qtbot, tmp_path, *, videos=(), poses=True):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._output_dir = tmp_path / "out"
    for name in videos:
        video = tmp_path / name
        video.write_bytes(b"")
        tab._videos.append(video)
        if poses:
            (tmp_path / f"{video.stem}DLC_yolo.csv").write_text("scorer\n")
    return tab


def test_a_run_with_no_model_bundle_needs_the_speed_axis(qtbot, tmp_path):
    """Without a classifier there must be something else doing the scoring."""
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4"])
    tab._yolo_path = tmp_path / "y.pt"
    assert "speed axis" in (tab._run_blocker() or "")


def test_freezing_alone_is_enough_to_run_without_a_model(qtbot, tmp_path):
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4"])
    tab._yolo_path = tmp_path / "y.pt"
    tab._speed_group.setChecked(True)
    tab._score_darting.setChecked(False)
    assert tab._run_blocker() is None


def test_scoring_neither_half_is_not_a_run(qtbot, tmp_path):
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4"])
    tab._yolo_path = tmp_path / "y.pt"
    tab._speed_group.setChecked(True)
    tab._score_freezing.setChecked(False)
    tab._score_darting.setChecked(False)
    assert "freezing, darting, or both" in (tab._run_blocker() or "")


def test_a_speed_only_run_cannot_ask_for_an_annotated_video(qtbot, tmp_path):
    """The overlay draws predicted labels, and there is no model to predict."""
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4"])
    tab._yolo_path = tmp_path / "y.pt"
    tab._speed_group.setChecked(True)
    tab._render_video.setChecked(True)
    assert "annotated video" in (tab._run_blocker() or "")


def test_weights_are_not_needed_when_every_video_has_poses(qtbot, tmp_path):
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4", "b.mp4"])
    tab._model_path = tmp_path / "m.pkl"
    assert tab._run_blocker() is None


def test_weights_are_demanded_for_a_video_with_no_poses(qtbot, tmp_path):
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4"], poses=False)
    tab._model_path = tmp_path / "m.pkl"
    blocker = tab._run_blocker() or ""
    assert "YOLO weights" in blocker
    assert "a.mp4" in blocker


def test_turning_pose_reuse_off_makes_the_weights_required_again(qtbot, tmp_path):
    """Every video is tracked then, poses on disk or not."""
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4"])
    tab._model_path = tmp_path / "m.pkl"
    tab._reuse_poses.setChecked(False)
    assert "YOLO weights" in (tab._run_blocker() or "")


def test_clearing_the_model_returns_to_a_speed_only_run(qtbot, tmp_path):
    tab = _apply_tab(qtbot, tmp_path)
    tab._model_path = tmp_path / "m.pkl"
    tab._on_clear_model()
    assert tab._model_path is None
    assert "freezing" in tab._model_label.text()


def test_an_unscored_behaviour_sends_no_threshold(qtbot):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._score_darting.setChecked(False)
    opts = tab._speed_opts()
    assert opts["score_darting"] is False
    assert opts["dart_cm_s"] is None
    assert "dart_min_s" not in opts


def test_the_threshold_for_an_unscored_behaviour_is_hidden(qtbot):
    """A setting with no meaning invites tuning that changes nothing."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._speed_group.setChecked(True)
    tab._score_darting.setChecked(False)
    assert tab._speed_form.isRowVisible(tab._freeze_abs_row)
    assert not tab._speed_form.isRowVisible(tab._dart_abs_row)
    assert not tab._speed_form.isRowVisible(tab._dart_min_row)


def test_cohort_mode_without_a_file_is_refused_not_run_silently(qtbot, tmp_path):
    """A missing cohort file would leave the axis absent from every ethogram."""
    tab = _apply_tab(qtbot, tmp_path, videos=["a.mp4"])
    tab._model_path = tmp_path / "m.pkl"
    tab._speed_group.setChecked(True)
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("cohort"))
    assert "cut-offs from a file" in (tab._run_blocker() or "")


# --------------------------------------------------------------------------
# Apply tab: the cohort file has to say what is in it
# --------------------------------------------------------------------------


def _cohort_file(path, **kw):
    from glider.analysis.behavior.cohort_speed import CM_PER_S, CohortSpeedThresholds

    base = {
        "freeze": 0.55,
        "dart": 27.68,
        "unit": CM_PER_S,
        "freeze_pct": 10.0,
        "dart_pct": 99.5,
        "n_sessions": 30,
        "n_samples": 269_964,
        "start_s": 120.0,
        "end_s": 420.0,
    }
    CohortSpeedThresholds(**{**base, **kw}).save(path)
    return path


def test_choosing_a_cohort_file_shows_its_cut_offs(qtbot, tmp_path):
    """A path does not answer which thresholds the run will use."""
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._cohort_path = _cohort_file(tmp_path / "cohort_speed.json")
    tab._show_cohort_file()
    text = tab._cohort_label.text()
    assert "27.7 cm/s" in text
    assert "0.55 cm/s" in text
    assert "2–7 min" in text


def test_a_cohort_pooled_in_pixels_is_flagged_with_the_reason(qtbot, tmp_path):
    from glider.analysis.behavior.cohort_speed import PX_PER_FRAME
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._cohort_path = _cohort_file(
        tmp_path / "cohort_speed.json", unit=PX_PER_FRAME, n_uncalibrated=4
    )
    tab._show_cohort_file()
    text = tab._cohort_label.text()
    assert "px/frame" in text
    assert "4 session(s) had no pixel scale" in text


def test_an_unreadable_cohort_file_says_so_rather_than_looking_chosen(qtbot, tmp_path):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    tab._cohort_path = tmp_path / "not_a_cohort.json"
    tab._cohort_path.write_text("{}")
    tab._show_cohort_file()
    assert "unreadable" in tab._cohort_label.text()


def test_the_cohort_scan_takes_each_session_once(tmp_path):
    """An apply run leaves a copy of the poses in its output folder.

    Pooling both weights that animal twice, and the copy usually sits where
    its video cannot be found — which costs the whole pool its pixel scale.
    """
    from glider.gui.behavior.window import _unique_pose_csvs

    (tmp_path / "out" / "t1_d2").mkdir(parents=True)
    beside = tmp_path / "t1_d2DLC_exp-7.csv"
    beside.write_text("scorer\n")
    (tmp_path / "out" / "t1_d2" / "t1_d2DLC_exp-7.csv").write_text("scorer\n")
    (tmp_path / "t2_d2DLC_exp-7.csv").write_text("scorer\n")

    found = _unique_pose_csvs(tmp_path)
    assert len(found) == 2
    # The shallowest copy wins, because it is the one beside its video.
    assert beside in found


def test_the_cohort_scan_ignores_csvs_that_are_not_poses(tmp_path):
    from glider.gui.behavior.window import _unique_pose_csvs

    (tmp_path / "aDLC_exp-7.csv").write_text("scorer\n")
    (tmp_path / "ethogram_raw.csv").write_text("frame,behavior\n")
    assert [p.name for p in _unique_pose_csvs(tmp_path)] == ["aDLC_exp-7.csv"]
