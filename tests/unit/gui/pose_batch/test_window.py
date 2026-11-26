"""PoseBatchWindow: drop list, video discovery, and Run-gating validation.

No model is ever loaded here, so nothing imports torch.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from glider.gui.pose_batch.window import PoseBatchWindow


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _clip(path: Path, width: int = 640, height: int = 480, frames: int = 3) -> Path:
    """A real, decodable clip — copying reads each target's true resolution."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (width, height))
    if not writer.isOpened():
        pytest.skip("no MJPG VideoWriter available in this OpenCV build")
    for i in range(frames):
        writer.write(np.full((height, width, 3), (i * 40) % 255, dtype=np.uint8))
    writer.release()
    # discover_videos resolves, so callers can compare against window._videos.
    return path.resolve()


def _cal(px_span: int, mm: float, resolution: tuple[int, int] = (640, 480)):
    """A calibration whose scale is exactly ``px_span / mm`` at *resolution*."""
    from glider.vision.calibration import CameraCalibration, LengthUnit

    cal = CameraCalibration()
    cal.add_line(
        start=(0, resolution[1] // 2),
        end=(px_span, resolution[1] // 2),
        length=mm,
        unit=LengthUnit.MILLIMETERS,
        resolution=resolution,
    )
    return cal


def _answer_overwrite(monkeypatch, button):
    """Stub the overwrite confirmation; returns the list of prompts raised."""
    prompts = []

    def question(_parent, _title, text, *args, **kwargs):
        prompts.append(text)
        return button

    monkeypatch.setattr("glider.gui.pose_batch.window.QMessageBox.question", question)
    return prompts


@pytest.fixture
def window(qtbot):
    win = PoseBatchWindow()
    qtbot.addWidget(win)
    return win


# --------------------------------------------------------------------------
# the one table: drops, sources, remove
# --------------------------------------------------------------------------


def _drop_on(widget, *paths):
    """Deliver a real file drop to *widget*."""
    from PyQt6.QtCore import QMimeData, QPointF, Qt, QUrl
    from PyQt6.QtGui import QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.dropEvent(event)
    return event


def test_there_is_no_separate_videos_list(window):
    from PyQt6.QtWidgets import QListWidget

    assert not window.findChildren(QListWidget)
    assert window._cal_table.acceptDrops() is True


def test_table_columns_are_video_folder_arena_scale(window):
    table = window._cal_table
    headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
    assert headers == ["Video", "Folder", "Arena", "Scale"]


def test_drop_folder_on_table_adds_videos(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mp4")
    _drop_on(window._cal_table, tmp_path)
    assert len(window._videos) == 2
    assert window._cal_table.rowCount() == 2


def test_drop_respects_the_recursive_setting(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mp4")
    window._recursive.setChecked(False)
    _drop_on(window._cal_table, tmp_path)
    assert len(window._videos) == 1


def test_sources_dedupe_and_emit(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    window._sources.add_paths([tmp_path, tmp_path])
    assert window._sources.paths() == [tmp_path]
    assert len(window._videos) == 1


def test_remove_drops_its_arena(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "b.mp4")
    window._sources.add_paths([tmp_path])
    _calibrate_all(window)
    keep, gone = window._videos
    window._cal_table.selectRow(1)
    window._remove_selected()
    assert window._videos == [keep]
    assert window._cal_table.rowCount() == 1
    assert window._calibrations.get_arena(gone) is None
    assert window._calibrations.get(gone) is None
    assert window._calibrations.get_arena(keep) is not None


def test_removed_video_stays_out_until_its_folder_is_added_again(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "b.mp4")
    window._sources.add_paths([tmp_path])
    window._cal_table.selectRow(0)
    window._remove_selected()
    window._recursive.setChecked(False)
    window._recursive.setChecked(True)
    assert len(window._videos) == 1
    window._add_paths([tmp_path / "a.mp4"])
    assert len(window._videos) == 2


def test_empty_table_says_what_to_do(window):
    from PyQt6.QtWidgets import QLabel

    texts = [w.text() for w in window._cal_table.viewport().findChildren(QLabel)]
    assert "Drop video folders here, or Add Folder…" in texts


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_counts_videos_from_a_dropped_directory(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mp4")
    window._sources.add_paths([tmp_path])
    assert len(window._videos) == 2
    assert "2 videos found" in window._count_label.text()


def test_recursive_toggle_changes_the_count(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "sub" / "b.mp4")
    window._sources.add_paths([tmp_path])
    window._recursive.setChecked(False)
    assert len(window._videos) == 1
    assert "1 video found" in window._count_label.text()


def test_empty_directory_reports_none_found(window, tmp_path):
    window._sources.add_paths([tmp_path])
    assert "No videos found" in window._count_label.text()


# --------------------------------------------------------------------------
# validation gating
# --------------------------------------------------------------------------


def test_run_disabled_without_a_model(window):
    assert window._run_button.isEnabled() is False
    assert "Select a model" in window._run_button.toolTip()


def _ready(window, tmp_path, names="a, b", n_keypoints=2):
    """Put the window into an otherwise-runnable state."""
    _touch(tmp_path / "v.mp4")
    window._model_path = tmp_path / "model.pt"
    window._meta = types.SimpleNamespace(
        n_keypoints=n_keypoints, keypoint_names=None, source="checkpoint"
    )
    window._sources.add_paths([tmp_path])
    window._names_field.setText(names)
    _calibrate_all(window)
    return window


def test_run_enabled_when_everything_is_valid(window, tmp_path):
    _ready(window, tmp_path)
    assert window._run_button.isEnabled() is True


def test_run_blocked_on_duplicate_names(window, tmp_path):
    _ready(window, tmp_path, names="a, a")
    assert window._run_button.isEnabled() is False
    assert "unique" in window._run_button.toolTip()


def test_run_blocked_on_keypoint_count_mismatch(window, tmp_path):
    _ready(window, tmp_path, names="a, b, c", n_keypoints=2)
    assert window._run_button.isEnabled() is False
    assert "2 keypoints but 3 names" in window._run_button.toolTip()


def test_run_blocked_when_no_videos_found(window, tmp_path):
    window._model_path = tmp_path / "model.pt"
    window._meta = None
    window._names_field.setText("a, b")
    window._refresh_videos()
    assert window._run_button.isEnabled() is False
    assert "at least one video" in window._run_button.toolTip()


def test_run_blocked_on_empty_names(window, tmp_path):
    _ready(window, tmp_path, names="")
    assert window._run_button.isEnabled() is False


def test_invalid_names_are_marked(window, tmp_path):
    _ready(window, tmp_path, names="a, a")
    assert window._names_field.styleSheet() != ""
    _ready(window, tmp_path, names="a, b")
    assert window._names_field.styleSheet() == ""


# --------------------------------------------------------------------------
# filter settings
# --------------------------------------------------------------------------


def test_filtering_off_by_default(window):
    assert window._filter_group.isChecked() is False
    assert window._filter_settings() is None


def test_filter_settings_are_read_from_the_widgets(window):
    window._filter_group.setChecked(True)
    window._filter_conf.setValue(0.4)
    window._filter_gap.setValue(7)
    settings = window._filter_settings()
    assert settings.confidence_threshold == pytest.approx(0.4)
    assert settings.max_gap == 7


def test_median_window_stays_odd(window):
    window._filter_group.setChecked(True)
    window._filter_window.setValue(6)
    # medfilt rejects even kernels, so the widget coerces upward.
    assert window._filter_window.value() % 2 == 1
    assert window._filter_settings().median_window % 2 == 1


# --------------------------------------------------------------------------
# name parsing
# --------------------------------------------------------------------------


def test_names_are_parsed_and_trimmed(window):
    window._names_field.setText(" nose , l_ear ,, r_ear ")
    assert window._current_names() == ["nose", "l_ear", "r_ear"]


# --------------------------------------------------------------------------
# calibration gating
# --------------------------------------------------------------------------


# A real, non-degenerate perimeter — same corners used across the arena test
# suite — so a confirmed homography can always be fit for it.
_TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


def _calibrate_all(window):
    """Give every discovered video a usable calibration and a confirmed arena.

    Run now requires a drawn arena (not just a line) per video, so a helper
    claiming to put the window into an "otherwise runnable" state has to
    supply both to keep meaning what its name says.
    """
    from glider.vision.arena import ArenaCalibration
    from glider.vision.calibration import CameraCalibration, LengthUnit

    for video in window._videos:
        cal = CameraCalibration()
        cal.add_line(
            start=(0, 240),
            end=(640, 240),
            length=100.0,
            unit=LengthUnit.MILLIMETERS,
            resolution=(640, 480),
        )
        window._calibrations.set(video, cal)
        window._calibrations.set_arena(
            video, ArenaCalibration(corners=_TRAPEZOID, frame_size=(640, 480))
        )
    window._cal_table.refresh()
    window._validate()


def test_run_blocked_until_every_video_is_calibrated(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "b.mp4")
    window._model_path = tmp_path / "model.pt"
    window._meta = None
    window._sources.add_paths([tmp_path])
    window._names_field.setText("a, b")
    assert window._run_button.isEnabled() is False
    # Run now gates on a drawn arena, not a line; see window.py's _validate().
    assert "arena" in window._run_button.toolTip()


def test_run_enabled_once_all_videos_are_calibrated(window, tmp_path):
    _ready(window, tmp_path)
    assert window._run_button.isEnabled() is True


def test_calibration_survives_remove_and_re_add(window, tmp_path):
    _ready(window, tmp_path)
    window._sources.clear_all()
    assert window._videos == []
    window._sources.add_paths([tmp_path])
    # Keyed by path, so it comes back without re-drawing.
    assert window._run_button.isEnabled() is True


def test_table_lists_the_discovered_videos(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    window._sources.add_paths([tmp_path])
    assert window._cal_table.rowCount() == 1


# --------------------------------------------------------------------------
# calibration actions and master file
# --------------------------------------------------------------------------


def test_clear_removes_the_selected_calibration(window, tmp_path):
    _ready(window, tmp_path)
    window._cal_table.selectAll()
    window._clear_selected_calibrations()
    assert window._run_button.isEnabled() is False


def test_master_path_defaults_beside_the_videos(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    window._sources.add_paths([tmp_path])
    assert window._master_field.text() == str(tmp_path / "pose_calibration.json")


def test_save_master_writes_the_file(window, tmp_path):
    _ready(window, tmp_path)
    window._save_master_clicked()
    master = tmp_path / "pose_calibration.json"
    assert master.exists()
    from glider.vision.calibration_set import CalibrationSet

    assert CalibrationSet.load(master).px_per_mm(window._videos[0]) is not None


def test_an_existing_master_file_is_auto_loaded(window, tmp_path, qtbot):
    """A re-run must not cost a second round of drawing."""
    from glider.vision.calibration import CameraCalibration, LengthUnit
    from glider.vision.calibration_set import CalibrationSet

    video = _touch(tmp_path / "a.mp4").resolve()
    cal = CameraCalibration()
    cal.add_line(
        start=(0, 240),
        end=(640, 240),
        length=100.0,
        unit=LengthUnit.MILLIMETERS,
        resolution=(640, 480),
    )
    seed = CalibrationSet()
    seed.set(video, cal)
    seed.save(tmp_path / "pose_calibration.json")

    window._sources.add_paths([tmp_path])
    assert window._calibrations.px_per_mm(video) is not None
    assert "pose_calibration.json" in window._log.toPlainText()


def test_a_corrupt_master_file_is_reported_not_applied(window, tmp_path, monkeypatch):
    monkeypatch.setattr("glider.gui.pose_batch.window.QMessageBox.warning", lambda *a, **k: None)
    (tmp_path / "pose_calibration.json").write_text("{not json")
    _touch(tmp_path / "a.mp4")
    window._sources.add_paths([tmp_path])
    # Discovery still works; calibration state is simply untouched.
    assert len(window._videos) == 1
    assert window._calibrations.entries == {}


def test_master_path_follows_the_videos_while_it_is_still_the_default(window, tmp_path):
    """Swapping folders must not leave the master pointing at the old one."""
    first, second = tmp_path / "A", tmp_path / "B"
    _touch(first / "x.mp4")
    _touch(second / "y.mp4")

    window._sources.add_paths([first])
    assert window._master_field.text() == str(first / "pose_calibration.json")

    window._sources.clear_all()
    window._sources.add_paths([second])
    assert window._master_field.text() == str(second / "pose_calibration.json")


def test_a_hand_set_master_path_is_never_re_defaulted(window, tmp_path):
    chosen = tmp_path / "elsewhere" / "mine.json"
    window._master_field.setText(str(chosen))
    window._master_field.editingFinished.emit()

    _touch(tmp_path / "A" / "x.mp4")
    window._sources.add_paths([tmp_path / "A"])
    assert window._master_field.text() == str(chosen)


def test_a_typed_master_path_is_read_before_it_would_be_overwritten(window, tmp_path):
    """Typing a path must load it, exactly as Browse and Load do."""
    from glider.vision.calibration_set import CalibrationSet

    ghost = _touch(tmp_path / "videos" / "ghost.mp4").resolve()
    _touch(tmp_path / "videos" / "other.mp4")
    # Kept away from the videos, so it can only arrive by being typed.
    master = tmp_path / "masters" / "rig1.json"
    seed = CalibrationSet()
    seed.set(ghost, _cal(640, 100.0))
    seed.save(master)

    window._sources.add_paths([tmp_path / "videos"])
    window._master_field.setText(str(master))
    window._master_field.editingFinished.emit()  # what a typed path fires

    assert window._calibrations.px_per_mm(ghost) is not None
    assert "rig1.json" in window._log.toPlainText()

    # And Run must not then wipe the entry it just read.
    window._write_master(master)
    data = json.loads(master.read_text(encoding="utf-8"))
    assert "ghost.mp4" in [Path(entry["video"]).name for entry in data["videos"]]


def test_master_file_describes_only_the_listed_batch(window, tmp_path):
    _ready(window, tmp_path)
    # A calibration left over from a folder the operator has since removed.
    window._calibrations.set(tmp_path / "elsewhere" / "stray.mp4", _cal(640, 100.0))

    window._save_master_clicked()

    data = json.loads((tmp_path / "pose_calibration.json").read_text(encoding="utf-8"))
    assert [Path(entry["video"]).name for entry in data["videos"]] == ["v.mp4"]


def test_an_unwritable_master_path_is_reported_not_raised(window, tmp_path, monkeypatch):
    """The field is free text; a null byte fails below the OSError layer."""
    _ready(window, tmp_path)
    monkeypatch.setattr("glider.gui.pose_batch.window.QMessageBox.critical", lambda *a, **k: None)
    assert window._write_master(Path(f"{tmp_path}/bad\0name.json")) is False
    assert "Could not write" in window._log.toPlainText()


def test_master_file_is_written_before_the_batch_starts(window, tmp_path, monkeypatch):
    _ready(window, tmp_path)
    started = {}
    monkeypatch.setattr(window, "_start_worker", lambda: started.setdefault("yes", True))
    window._start()
    assert (tmp_path / "pose_calibration.json").exists()
    assert started.get("yes") is True


def test_experiment_type_is_stored(window):
    assert window._experiment_type == ""
    window.set_experiment_type("Pup Retrieval")
    assert window._experiment_type == "Pup Retrieval"


# --------------------------------------------------------------------------
# header: step strip, Settings popup, More menu
# --------------------------------------------------------------------------


def test_step_strip_tracks_progress(window, tmp_path):
    assert window._steps.steps_text() == ["Model", "Videos", "Arenas", "Track"]
    assert window._steps.current() == 0

    for i in range(5):
        _touch(tmp_path / f"v{i}.mp4")
    window._model_path = tmp_path / "m.pt"
    window._names_field.setText("a, b")
    window._meta = types.SimpleNamespace(n_keypoints=2)
    window._sources.add_paths([tmp_path])
    for video in window._videos[:2]:
        window._calibrations.set_arena(video, _arena_cal())
    window._validate()

    assert window._steps.steps_text() == ["Model ✓", "Videos · 5 ✓", "Arenas · 2 of 5", "Track"]
    assert window._steps.current() == 2


def _arena_cal():
    from glider.vision.arena import ArenaCalibration

    return ArenaCalibration(corners=[(0, 0), (100, 0), (100, 100), (0, 100)], frame_size=(640, 480))


def test_settings_popup_holds_rare_settings(window):
    popup = window._settings_popup
    for widget in (
        window._device_combo,
        window._conf_spin,
        window._require_gpu,
        window._filter_group,
        window._recursive,
        window._overwrite,
        window._master_field,
    ):
        assert widget.window() is popup
    # ...and the everyday ones are not hiding in there.
    for widget in (window._cal_table, window._names_field, window._animals_spin):
        assert widget.window() is window


def test_settings_button_opens_the_popup(window, qtbot):
    window.show()
    window._settings_button.click()
    assert window._settings_popup.isVisible()
    window._settings_popup.hide()


def test_more_menu_actions(window):
    labels = [a.text() for a in window._more_button.menu().actions() if a.text()]
    assert labels == [
        "Re-gate tracked CSVs…",
        "Export multi-animal DLC CSV…",
        "Remove all videos",
    ]


def test_more_menu_actions_run_the_old_handlers(window, monkeypatch):
    calls = []
    monkeypatch.setattr(window, "_start_regate", lambda: calls.append("regate"))
    monkeypatch.setattr(window, "_start_export", lambda: calls.append("export"))
    monkeypatch.setattr(window, "_remove_all", lambda: calls.append("remove"))
    for action in (window._regate_action, window._export_action, window._remove_all_action):
        action.setEnabled(True)
        action.trigger()
    assert calls == ["regate", "export", "remove"]


def test_remove_all_drops_arenas(window, tmp_path, monkeypatch):
    _touch(tmp_path / "a.mp4")
    _touch(tmp_path / "b.mp4")
    window._sources.add_paths([tmp_path])
    videos = list(window._videos)
    for video in videos:
        window._calibrations.set_arena(video, _arena_cal())
    prompts = _answer_overwrite(monkeypatch, _yes())
    window._remove_all()
    assert prompts  # arenas existed, so it asked first
    assert window._videos == []
    assert window._cal_table.rowCount() == 0
    for video in videos:
        assert window._calibrations.get_arena(video) is None
        assert window._calibrations.get(video) is None


def test_remove_all_declined_keeps_everything(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    _touch(tmp_path / "a.mp4")
    window._sources.add_paths([tmp_path])
    window._calibrations.set_arena(window._videos[0], _arena_cal())
    _answer_overwrite(monkeypatch, QMessageBox.StandardButton.No)
    window._remove_all()
    assert len(window._videos) == 1
    assert window._calibrations.get_arena(window._videos[0]) is not None


def test_remove_all_without_arenas_does_not_ask(window, tmp_path, monkeypatch):
    _touch(tmp_path / "a.mp4")
    window._sources.add_paths([tmp_path])
    prompts = _answer_overwrite(monkeypatch, _yes())
    window._remove_all()
    assert not prompts
    assert window._videos == []


def _yes():
    from PyQt6.QtWidgets import QMessageBox

    return QMessageBox.StandardButton.Yes


def test_master_file_defaults_with_popup_closed(window, tmp_path):
    _touch(tmp_path / "a.mp4")
    assert not window._settings_popup.isVisible()
    window._sources.add_paths([tmp_path])
    assert window._master_field.text() == str(tmp_path / "pose_calibration.json")


def test_tour_points_at_the_table_and_settings(window):
    targets = window.tour_targets()
    assert targets["videos"] is window._calibration_card
    assert targets["calibration"] is window._calibration_card
    assert targets["filter"] is window._settings_button
    assert targets["run"] is window._run_button


# --------------------------------------------------------------------------
# footer: Track button, progress, folded log
# --------------------------------------------------------------------------


def _running(window, tmp_path, monkeypatch):
    """Start a real run whose worker never touches inference."""
    from glider.gui.pose_batch import worker as worker_mod

    class Idle(worker_mod.PoseBatchWorker):
        def run(self):
            pass

    monkeypatch.setattr(worker_mod, "PoseBatchWorker", Idle)
    _ready(window, tmp_path)
    window._start_worker()


def test_track_button_counts_videos(window, tmp_path):
    assert window._run_button.text() == "Track 0 Videos"
    _touch(tmp_path / "a.mp4")
    window._sources.add_paths([tmp_path])
    assert window._run_button.text() == "Track 1 Video"
    _touch(tmp_path / "more" / "b.mp4")
    window._sources.add_paths([tmp_path / "more"])
    assert window._run_button.text() == "Track 2 Videos"


def test_footer_passes_validation_message_through(window):
    # Fail-if-no-GPU is enforced by the worker, so the footer passes the existing
    # validation message through unchanged rather than inventing a GPU one.
    assert window._blocker_label.text() == "Select a model."
    assert not window._blocker_label.isHidden()


def test_blocker_clears_when_runnable(window, tmp_path):
    _ready(window, tmp_path)
    assert window._blocker_label.text() == ""


def test_cancel_only_while_running(window, tmp_path, monkeypatch):
    assert window._cancel_button.isHidden()
    assert window._progress_panel.isHidden()
    _running(window, tmp_path, monkeypatch)
    try:
        assert not window._cancel_button.isHidden()
        assert not window._progress_panel.isHidden()
        assert window._run_button.isHidden()
    finally:
        window._teardown_thread()
    assert window._cancel_button.isHidden()
    assert not window._run_button.isHidden()


def test_progress_label_uses_the_runs_own_video_list(window, tmp_path, monkeypatch):
    _running(window, tmp_path, monkeypatch)
    try:
        started = window._videos[0].name
        window._videos = []  # Add Folder / drop / master load mid-run
        window._on_progress(0, 1)
        assert window._video_label.text() == started
    finally:
        window._teardown_thread()


def test_new_run_starts_with_log_closed(window, tmp_path, monkeypatch):
    window._log_toggle.setChecked(True)
    _running(window, tmp_path, monkeypatch)
    try:
        assert not window._log_toggle.isChecked()
    finally:
        window._teardown_thread()


def test_more_button_has_one_arrow(window):
    assert window._more_button.text() == "More ▾"
    assert "menu-indicator" in window._more_button.styleSheet()


def test_strip_shows_tracking_k_of_n(window, tmp_path, monkeypatch):
    _running(window, tmp_path, monkeypatch)
    try:
        window._on_progress(1, 5)
        assert window._steps.steps_text()[3] == "Tracking 2 of 5"
        assert window._overall_bar.value() == 2
    finally:
        window._teardown_thread()
    assert window._steps.steps_text()[3] == "Track"


def test_remove_is_disabled_while_running(window, tmp_path, monkeypatch):
    _running(window, tmp_path, monkeypatch)
    try:
        assert not window._remove_all_action.isEnabled()
        assert not window._remove_button.isEnabled()
    finally:
        window._teardown_thread()
    assert window._remove_all_action.isEnabled()
    assert window._remove_button.isEnabled()


def test_log_is_collapsed_by_default(window):
    assert window._log.isHidden()
    assert window._log_toggle.text() == "▸ Log (0 lines)"


def test_log_toggle_counts_and_opens(window):
    window._add_log("one")
    window._add_log("two")
    assert window._log.isHidden()
    assert window._log_toggle.text() == "▸ Log (2 lines)"
    window._log_toggle.click()
    assert not window._log.isHidden()
    assert window._log_toggle.text() == "▾ Log (2 lines)"


def test_log_opens_on_error(window):
    window._add_log("[1/2] a.mp4")
    assert window._log.isHidden()
    window._add_log("FAILED a.mp4: boom")
    assert not window._log.isHidden()


def test_settings_summary_shows_active_options(window):
    assert window._settings_summary.text() == ""
    window._overwrite.setChecked(True)
    window._filter_group.setChecked(True)
    assert window._settings_summary.text() == "Smoothing on · Overwrite on"
    window._overwrite.setChecked(False)
    window._filter_group.setChecked(False)
    assert window._settings_summary.text() == ""
