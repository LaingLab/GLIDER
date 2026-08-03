"""Loading pre-rename bundles, and borrowing a scale from a folder-mate.

Both exist because real lab data outlives the code that made it: cohort models
predate the yolo2pose -> glider rename, and a folder of videos from one rig
should not need twenty individual calibrations.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import pytest

from glider.analysis.behavior._legacy import (
    is_legacy_bundle_error,
    legacy_module_aliases,
    load_bundle,
)
from glider.analysis.behavior.units import load_px_per_mm
from glider.vision.calibration import CameraCalibration, LengthUnit
from glider.vision.calibration_set import CalibrationSet


def _clip(path, w, h, frames=6):
    """A real decodable clip, so resolution reads come from a container."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot open an MJPG writer")
    for _ in range(frames):
        writer.write(np.zeros((h, w, 3), dtype=np.uint8))
    writer.release()
    return path


def _cal(px_per_mm, width, height):
    cal = CameraCalibration()
    cal.add_line(
        start=(0, height // 2),
        end=(width, height // 2),
        length=width / px_per_mm,
        unit=LengthUnit.MILLIMETERS,
        name="w",
        resolution=(width, height),
    )
    return cal


class TestLegacyModuleAliases:
    def test_aliases_are_scoped_not_permanent(self):
        assert "yolo2pose" not in sys.modules
        with legacy_module_aliases():
            assert "yolo2pose.train.embedding" in sys.modules
        # Nothing else may start depending on the shim.
        assert "yolo2pose" not in sys.modules
        assert "yolo2pose.train.embedding" not in sys.modules

    def test_the_alias_points_at_the_current_module(self):
        from glider.analysis.behavior import embedding as current

        with legacy_module_aliases():
            assert sys.modules["yolo2pose.train.embedding"] is current

    def test_recognises_the_pre_rename_error(self):
        assert is_legacy_bundle_error(ModuleNotFoundError(name="yolo2pose")) is True
        assert is_legacy_bundle_error(ModuleNotFoundError(name="numpy")) is False
        assert is_legacy_bundle_error(ValueError("nope")) is False

    def test_a_normal_bundle_loads_with_its_embedding_intact(self, tmp_path):
        import joblib

        path = tmp_path / "b.pkl"
        joblib.dump({"classifier": "x", "embedding": {"kept": True}}, path)
        payload, embedding_usable = load_bundle(path)
        assert embedding_usable is True
        assert payload["embedding"] == {"kept": True}


class TestFolderMateScale:
    def test_borrows_from_a_calibrated_video_in_the_same_folder(self, tmp_path):
        folder = tmp_path / "5 mgkg final mp4"
        folder.mkdir()
        calibrated = _clip(folder / "T12_5.avi", 64, 48)
        uncalibrated = _clip(folder / "T7_5.avi", 64, 48)

        cal_set = CalibrationSet()
        cal_set.set(calibrated, _cal(4.0, 64, 48))
        master = folder / "pose_calibration.json"
        cal_set.save(master)

        assert load_px_per_mm(master, uncalibrated) == pytest.approx(4.0, abs=0.05)

    def test_an_exact_entry_still_wins(self, tmp_path):
        folder = tmp_path / "rig"
        folder.mkdir()
        a = _clip(folder / "a.avi", 64, 48)
        b = _clip(folder / "b.avi", 64, 48)
        cal_set = CalibrationSet()
        cal_set.set(a, _cal(4.0, 64, 48))
        cal_set.set(b, _cal(8.0, 64, 48))
        master = folder / "m.json"
        cal_set.save(master)
        # b has its own entry; it must not borrow a's.
        assert load_px_per_mm(master, b) == pytest.approx(8.0, abs=0.05)

    def test_refuses_when_the_resolution_differs(self, tmp_path):
        """The wrong-millimetres bug: a different resolution means a different scale."""
        folder = tmp_path / "rig"
        folder.mkdir()
        calibrated = _clip(folder / "small.avi", 64, 48)
        other = _clip(folder / "big.avi", 128, 96)
        cal_set = CalibrationSet()
        cal_set.set(calibrated, _cal(4.0, 64, 48))
        master = folder / "m.json"
        cal_set.save(master)
        assert load_px_per_mm(master, other) is None

    def test_refuses_when_folder_mates_disagree(self, tmp_path):
        folder = tmp_path / "rig"
        folder.mkdir()
        a = _clip(folder / "a.avi", 64, 48)
        b = _clip(folder / "b.avi", 64, 48)
        target = _clip(folder / "c.avi", 64, 48)
        cal_set = CalibrationSet()
        cal_set.set(a, _cal(4.0, 64, 48))
        cal_set.set(b, _cal(9.0, 64, 48))
        master = folder / "m.json"
        cal_set.save(master)
        # No way to know which rig setting is right; guessing would be wrong.
        assert load_px_per_mm(master, target) is None

    def test_does_not_borrow_across_folders(self, tmp_path):
        one = tmp_path / "5 mgkg"
        two = tmp_path / "10 mgkg"
        one.mkdir()
        two.mkdir()
        calibrated = _clip(one / "a.avi", 64, 48)
        target = _clip(two / "b.avi", 64, 48)
        cal_set = CalibrationSet()
        cal_set.set(calibrated, _cal(4.0, 64, 48))
        master = tmp_path / "m.json"
        cal_set.save(master)
        assert load_px_per_mm(master, target) is None

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            (r"\\130.74.60.149\lainglab\Garrett\5 mgkg final mp4\T12_5.avi", "5 mgkg final mp4"),
            (r"Z:\Lab Members\Garrett\5 mgkg final mp4\T12_5.avi", "5 mgkg final mp4"),
            ("/mnt/lainglab/Garrett/5 mgkg final mp4/T12_5.avi", "5 mgkg final mp4"),
            (r"C:\data\RIG\a.avi", "rig"),
            ("bare.avi", ""),
        ],
    )
    def test_folder_name_is_read_whichever_separator_wrote_the_path(self, stored, expected):
        """A calibration master written on Windows must still match on Linux.

        pathlib does not treat a backslash as a separator on POSIX, so a
        stored ``\\\\host\\share\\folder\\clip.avi`` parses as ONE component and
        ``parent.name`` comes back empty -- every folder comparison then
        silently fails and no scale is ever borrowed.
        """
        from glider.analysis.behavior.units import _folder_name

        assert _folder_name(stored) == expected

    def test_matches_folders_by_name_across_unc_and_drive_letters(self, tmp_path):
        """The same share is routinely addressed as Z:\\... and \\\\host\\share\\..."""
        folder = tmp_path / "5 mgkg final mp4"
        folder.mkdir()
        target = _clip(folder / "T7_5.avi", 64, 48)

        # Stored under a path that does not exist here, but the same folder name.
        cal_set = CalibrationSet()
        cal_set.set(
            r"\\130.74.60.149\lainglab\Lab Members\Garrett\5 mgkg final mp4\T12_5.avi",
            _cal(4.0, 64, 48),
        )
        master = folder / "pose_calibration.json"
        cal_set.save(master)

        assert load_px_per_mm(master, target) == pytest.approx(4.0, abs=0.05)

    def test_the_fallback_can_be_switched_off(self, tmp_path):
        folder = tmp_path / "rig"
        folder.mkdir()
        calibrated = _clip(folder / "a.avi", 64, 48)
        target = _clip(folder / "b.avi", 64, 48)
        cal_set = CalibrationSet()
        cal_set.set(calibrated, _cal(4.0, 64, 48))
        master = folder / "m.json"
        cal_set.save(master)
        assert load_px_per_mm(master, target, allow_folder_fallback=False) is None

    def test_unreadable_video_refuses_rather_than_guessing(self, tmp_path):
        folder = tmp_path / "rig"
        folder.mkdir()
        calibrated = _clip(folder / "a.avi", 64, 48)
        broken = folder / "broken.avi"
        broken.write_bytes(b"not a video")
        cal_set = CalibrationSet()
        cal_set.set(calibrated, _cal(4.0, 64, 48))
        master = folder / "m.json"
        cal_set.save(master)
        # Resolution unverifiable, so the borrow cannot be validated.
        assert load_px_per_mm(master, broken) is None
