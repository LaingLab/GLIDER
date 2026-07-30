"""PoseBatchWindow: drop list, video discovery, and Run-gating validation.

No model is ever loaded here, so nothing imports torch.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox

from glider.gui.pose_batch.window import PoseBatchWindow, _DropList


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
# drop list
# --------------------------------------------------------------------------


def test_drop_list_dedupes_paths(qtbot, tmp_path):
    widget = _DropList()
    qtbot.addWidget(widget)
    widget.add_paths([tmp_path, tmp_path])
    assert widget.paths() == [tmp_path]


def test_drop_list_emits_changed_on_add(qtbot, tmp_path):
    widget = _DropList()
    qtbot.addWidget(widget)
    with qtbot.waitSignal(widget.changed, timeout=1000):
        widget.add_paths([tmp_path])


def test_drop_list_accepts_drops(qtbot):
    widget = _DropList()
    qtbot.addWidget(widget)
    assert widget.acceptDrops() is True


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


def _calibrate_all(window):
    """Give every discovered video a usable calibration."""
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
    assert "need calibration" in window._run_button.toolTip()


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


def test_copy_calibration_stamps_the_other_selected_videos(window, tmp_path):
    a = _clip(tmp_path / "a.avi")
    b = _clip(tmp_path / "b.avi")
    window._sources.add_paths([tmp_path])
    assert window._videos == [a, b]
    window._calibrations.set(a, _cal(640, 100.0))
    window._cal_table.selectAll()
    window._copy_calibration_to_selected()
    assert window._calibrations.px_per_mm(b) is not None
    # A copy, not a shared object — editing one must not mutate the other.
    assert window._calibrations.get(b) is not window._calibrations.get(a)


def test_copy_does_not_silently_destroy_drawn_calibrations(window, tmp_path, monkeypatch):
    """Four individually calibrated videos must survive selectAll() + Copy."""
    for name in ("v1.avi", "v2.avi", "v3.avi", "v4.avi"):
        _clip(tmp_path / name)
    window._sources.add_paths([tmp_path])
    scales = (1.0, 2.0, 3.0, 4.0)
    for video, scale in zip(window._videos, scales, strict=True):
        window._calibrations.set(video, _cal(640, 640.0 / scale))

    prompts = _answer_overwrite(monkeypatch, QMessageBox.StandardButton.No)
    window._cal_table.selectAll()
    window._copy_calibration_to_selected()

    assert prompts, "the operator must be asked before calibrations are replaced"
    for video, scale in zip(window._videos, scales, strict=True):
        assert window._calibrations.px_per_mm(video) == pytest.approx(scale)


def test_copy_still_fills_the_uncalibrated_videos_when_overwrite_is_declined(
    window, tmp_path, monkeypatch
):
    source = _clip(tmp_path / "a.avi")
    drawn = _clip(tmp_path / "b.avi")
    blank = _clip(tmp_path / "c.avi")
    window._sources.add_paths([tmp_path])
    window._calibrations.set(source, _cal(640, 640.0))  # 1.0 px/mm
    window._calibrations.set(drawn, _cal(640, 320.0))  # 2.0 px/mm

    _answer_overwrite(monkeypatch, QMessageBox.StandardButton.No)
    window._cal_table.selectAll()
    window._copy_calibration_to_selected()

    assert window._calibrations.px_per_mm(drawn) == pytest.approx(2.0)
    assert window._calibrations.px_per_mm(blank) == pytest.approx(1.0)
    log = window._log.toPlainText()
    assert "1 uncalibrated video(s) filled" in log
    assert "0 existing calibration(s) overwritten" in log


def test_copy_overwrites_only_when_the_operator_confirms(window, tmp_path, monkeypatch):
    source = _clip(tmp_path / "a.avi")
    drawn = _clip(tmp_path / "b.avi")
    window._sources.add_paths([tmp_path])
    window._calibrations.set(source, _cal(640, 640.0))  # 1.0 px/mm
    window._calibrations.set(drawn, _cal(640, 320.0))  # 2.0 px/mm

    _answer_overwrite(monkeypatch, QMessageBox.StandardButton.Yes)
    window._cal_table.selectAll()
    window._copy_calibration_to_selected()

    assert window._calibrations.px_per_mm(drawn) == pytest.approx(1.0)
    assert "1 existing calibration(s) overwritten" in window._log.toPlainText()


def test_copy_retargets_the_calibration_to_the_destination_resolution(window, tmp_path):
    """Same rig, different recording resolution: the scale must follow the pixels."""
    a = _clip(tmp_path / "a.avi", 640, 480)
    b = _clip(tmp_path / "b.avi", 1280, 960)
    window._sources.add_paths([tmp_path])
    window._calibrations.set(a, _cal(320, 100.0, (640, 480)))  # 3.2 px/mm

    window._cal_table.selectAll()
    window._copy_calibration_to_selected()

    copied = window._calibrations.get(b)
    assert (copied.calibration_width, copied.calibration_height) == (1280, 960)
    assert window._calibrations.px_per_mm(b) == pytest.approx(2 * window._calibrations.px_per_mm(a))
    # The source itself is untouched by the retarget.
    assert window._calibrations.px_per_mm(a) == pytest.approx(3.2)


def test_copy_skips_targets_whose_resolution_cannot_be_read(window, tmp_path):
    a = _clip(tmp_path / "a.avi")
    broken = _touch(tmp_path / "b.avi").resolve()  # zero bytes: not decodable
    window._sources.add_paths([tmp_path])
    window._calibrations.set(a, _cal(640, 100.0))

    window._cal_table.selectAll()
    window._copy_calibration_to_selected()

    # Guessing a resolution is exactly the error being avoided.
    assert window._calibrations.get(broken) is None
    assert "could not be opened" in window._log.toPlainText()
    assert "b.avi" in window._log.toPlainText()


def test_calibration_dialogs_are_not_retained(window, tmp_path, monkeypatch):
    """Each dialog holds a full-res frame and pixmap; 50 of them is GBs."""
    _clip(tmp_path / "a.avi", 320, 240)
    window._sources.add_paths([tmp_path])
    from glider.gui.dialogs.calibration_dialog import CalibrationDialog

    monkeypatch.setattr(CalibrationDialog, "exec", lambda self: QDialog.DialogCode.Rejected)
    for _ in range(6):
        window._open_calibration(window._videos[0])
    QApplication.processEvents()
    # No event loop is running here, and DeferredDelete is only delivered by one;
    # in the app the deletion happens as soon as the slot returns to it.
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert window.findChildren(QDialog) == []


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
