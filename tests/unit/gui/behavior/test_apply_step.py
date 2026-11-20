"""Apply step: what scoring needs on screen, tuning in Options, handoff to Session Review."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPoint  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


def _tab(qtbot):
    from glider.gui.behavior.window import ApplyTab

    tab = ApplyTab()
    qtbot.addWidget(tab)
    return tab


def _y(tab, widget) -> int:
    return widget.mapTo(tab, QPoint(0, 0)).y()


def test_apply_required_inputs_first(qtbot, tmp_path):
    tab = _tab(qtbot)
    tab.resize(1000, 800)
    tab.show()
    QApplication.processEvents()

    required = (tab._videos_list, tab._model_path_label, tab._output_value)
    for widget in required:
        assert not tab._options.isAncestorOf(widget)
        assert tab.isAncestorOf(widget)
    # Videos, then the model, then where the results go.
    ys = [_y(tab, w) for w in required]
    assert ys == sorted(ys)

    assert tab._footer.primary is tab._run_btn
    assert tab._run_btn.text() == "Score Videos"
    assert "Add at least one video" in tab._footer.message.text()

    # Each video row says whether its poses were found.
    found = tmp_path / "a.mp4"
    found.write_bytes(b"")
    (tmp_path / "aDLC_yolo.csv").write_text("scorer\n")
    missing = tmp_path / "b.mp4"
    missing.write_bytes(b"")
    for video in (found, missing):
        tab._videos.append(video)
        tab._videos_list.addItem(video.name)
    tab._refresh_pose_match()
    assert "poses found" in tab._videos_list.item(0).text()
    assert "a.mp4" in tab._videos_list.item(0).text()
    assert "no poses" in tab._videos_list.item(1).text()
    assert tab._run_btn.text() == "Score 2 Videos"


def test_apply_options_hold_scoring_controls(qtbot):
    tab = _tab(qtbot)
    popup = tab._options
    for control in (
        tab._yolo_path_label,
        tab._keypoints_edit,
        tab._reuse_poses,
        tab._pose_dir_value,
        tab._predict_every,
        tab._range_on,
        tab._smooth_window,
        tab._offline_smooth,
        tab._min_bout_s,
        tab._speed_group,
        tab._render_video,
    ):
        assert popup.isAncestorOf(control)
    # The speed group moved intact: its rows still show and hide by mode.
    tab._speed_group.setChecked(True)
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("percentile"))
    assert tab._speed_form.isRowVisible(tab._freeze_pct_row)
    assert not tab._speed_form.isRowVisible(tab._freeze_abs_row)

    assert not popup.isVisible()
    tab._footer.options_button.click()
    assert popup.isVisible()
    tab._footer.options_button.click()
    assert not popup.isVisible()


def test_summary_names_non_default_options(qtbot):
    """A setting changed in a closed popup still shows under the inputs."""
    tab = _tab(qtbot)
    text = tab._summary.text
    assert text() == "Every 3 frames · smoothing on · freeze/dart off"

    tab._predict_every.setValue(1)
    tab._range_on.setChecked(True)
    tab._range_start.setValue(2.0)
    tab._range_end.setValue(7.0)
    tab._smooth_window.setValue(1)
    tab._offline_smooth.setValue(0)
    tab._min_bout_s.setValue(0.5)
    tab._speed_group.setChecked(True)
    tab._reuse_poses.setChecked(False)
    tab._render_video.setChecked(True)
    lowered = text().lower()
    assert lowered.startswith("every frame")
    assert "2–7 min" in lowered
    assert "smoothing off" in lowered
    assert "0.5 s minimum bout" in lowered
    assert "freeze/dart on" in lowered
    assert "tracking every video" in lowered
    assert "annotated video" in lowered


def test_summary_names_changed_smoothing_pose_folder_and_speed_mode(qtbot, tmp_path, monkeypatch):
    from glider.gui.behavior import window as win_mod

    tab = _tab(qtbot)
    text = tab._summary.text
    tab._smooth_window.setValue(15)
    assert "majority 15" in text()
    assert "centred" in text()
    assert "smoothing on" not in text()
    tab._smooth_window.setValue(win_mod._DEFAULT_SMOOTH_WINDOW)
    tab._offline_smooth.setValue(0)
    assert "centred off" in text()
    assert f"majority {win_mod._DEFAULT_SMOOTH_WINDOW}" in text()

    folder = tmp_path / "poses"
    folder.mkdir()
    monkeypatch.setattr(win_mod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(folder))
    tab._on_choose_pose_dir()
    assert "poses from" in text() and "poses" in text().split("poses from")[1]
    tab._on_clear_pose_dir()
    assert "poses from" not in text()

    tab._speed_group.setChecked(True)
    assert "freeze/dart on" in text()
    assert "percentile" not in text()
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("percentile"))
    assert "freeze/dart on (percentile)" in text()
    tab._speed_mode.setCurrentIndex(tab._speed_mode.findData("cohort"))
    assert "cohort: no file" in text()
    tab._score_darting.setChecked(False)
    assert "freeze on (cohort" in text()


def _finish_one(tab, tmp_path) -> Path:
    out = tmp_path / "out"
    (out / "a").mkdir(parents=True)
    (out / "a" / "ethogram_raw.csv").write_text("frame,behavior\n")
    tab._output_dir = out
    tab._videos = [tmp_path / "a.mp4"]
    tab._on_apply_finished(object(), tmp_path / "a.mp4", out / "a")
    return out


def test_open_in_session_review_passes_output(qtbot, tmp_path, monkeypatch):
    """The banner's button opens Session Review on the output folder, not a blank one."""
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    win = BehaviorAnalysisWindow(project_dir=tmp_path / "proj")
    qtbot.addWidget(win)
    tab = win.apply_tab
    assert tab._banner.isHidden()

    out = _finish_one(tab, tmp_path)
    assert not tab._banner.isHidden()
    assert "Scored 1 video" in tab._banner_text.text()
    assert win.step_badge(3).text() == "✓"

    with qtbot.waitSignal(win.open_in_session_review) as blocker:
        tab._review_btn.click()
    assert blocker.args == [str(out)]


def test_a_failed_run_does_not_tick_apply(qtbot, tmp_path, monkeypatch):
    from glider.gui.behavior.window import BehaviorAnalysisWindow

    monkeypatch.setattr("glider.gui.behavior.window.QMessageBox.critical", lambda *a, **k: None)
    win = BehaviorAnalysisWindow(project_dir=tmp_path / "proj")
    qtbot.addWidget(win)
    tab = win.apply_tab
    tab._output_dir = tmp_path / "out"
    tab._videos = [tmp_path / "a.mp4"]
    tab._on_apply_failed("boom", tmp_path / "a.mp4")
    assert tab._banner.isHidden()
    assert win.step_badge(3).text() == "4"


def test_main_window_opens_session_review_on_the_folder(
    main_window_factory, qtbot, tmp_path, monkeypatch
):
    from glider.gui.behavior import analysis_window
    from glider.gui.behavior import window as win_mod

    real = win_mod.BehaviorAnalysisWindow
    monkeypatch.setattr(
        win_mod,
        "BehaviorAnalysisWindow",
        lambda parent=None: real(project_dir=tmp_path / "proj", parent=parent),
    )
    monkeypatch.setattr(real, "show", lambda self: None)
    monkeypatch.setattr(analysis_window.AnalysisWindow, "show", lambda self: None)
    loaded: list[Path] = []
    monkeypatch.setattr(
        analysis_window.AnalysisWindow, "load_folder", lambda self, root: loaded.append(root)
    )

    main = main_window_factory()
    main._open_behavior_analysis()
    behavior = main._behavior_window
    qtbot.addWidget(behavior)
    behavior.open_in_session_review.emit(str(tmp_path / "out"))
    qtbot.addWidget(main._session_review_window)
    assert loaded == [tmp_path / "out"]


def test_session_review_menu_still_opens_blank(main_window_factory, qtbot, monkeypatch):
    """The menu's triggered(bool) must not reach the folder path."""
    from glider.gui.behavior import analysis_window

    monkeypatch.setattr(analysis_window.AnalysisWindow, "show", lambda self: None)
    loaded: list = []
    monkeypatch.setattr(
        analysis_window.AnalysisWindow, "load_folder", lambda self, root: loaded.append(root)
    )
    main = main_window_factory()
    main._open_session_review()
    qtbot.addWidget(main._session_review_window)
    assert loaded == []


def test_footer_names_blockers_hidden_in_options(qtbot, tmp_path):
    tab = _tab(qtbot)
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    (tmp_path / "aDLC_yolo.csv").write_text("scorer\n")
    tab._videos.append(video)
    tab._videos_list.addItem(video.name)
    tab._output_dir = tmp_path / "out"
    tab._model_path = tmp_path / "m.pkl"
    tab._refresh_pose_match()
    assert "keypoint names" in tab._footer.message.text().lower()
    tab._keypoints_edit.setText("nose,tail")
    assert tab._footer.message.text() == "Ready: 1 video, poses found for all."
    tab._reuse_poses.setChecked(False)
    assert "YOLO weights" in tab._footer.message.text()
