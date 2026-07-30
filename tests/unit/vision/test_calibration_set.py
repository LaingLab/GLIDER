"""CalibrationSet: per-video calibration map and master-file I/O."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glider.vision.calibration import CameraCalibration, LengthUnit
from glider.vision.calibration_set import SCHEMA_VERSION, CalibrationSet, CalibrationSetError


def _cal(px_per_mm: float = 6.4, *, width: int = 640, height: int = 480) -> CameraCalibration:
    """A calibration whose full-width line yields ``px_per_mm``."""
    cal = CameraCalibration()
    cal.add_line(
        start=(0, height // 2),
        end=(width, height // 2),
        length=width / px_per_mm,
        unit=LengthUnit.MILLIMETERS,
        name="width",
        resolution=(width, height),
    )
    return cal


def _degenerate() -> CameraCalibration:
    """Has a line, but the line carries no scale — so it is not a calibration."""
    cal = CameraCalibration()
    cal.add_line(
        start=(10, 10),
        end=(10, 10),
        length=100.0,
        unit=LengthUnit.MILLIMETERS,
        resolution=(640, 480),
    )
    return cal


class TestQueries:
    def test_empty_set_has_no_scale(self, tmp_path):
        assert CalibrationSet().px_per_mm(tmp_path / "a.mp4") is None

    def test_px_per_mm_delegates_to_camera_calibration(self, tmp_path):
        video = tmp_path / "a.mp4"
        cal_set = CalibrationSet()
        cal_set.set(video, _cal(6.4))
        assert cal_set.px_per_mm(video) == pytest.approx(6.4, abs=0.1)

    def test_missing_lists_uncalibrated_videos(self, tmp_path):
        a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
        cal_set = CalibrationSet()
        cal_set.set(a, _cal())
        assert cal_set.missing([a, b]) == [b]
        assert cal_set.is_complete([a, b]) is False
        assert cal_set.is_complete([a]) is True

    def test_a_scaleless_entry_counts_as_missing(self, tmp_path):
        # Present in the map but conveys no scale: the operator drew nothing usable.
        video = tmp_path / "a.mp4"
        cal_set = CalibrationSet()
        cal_set.set(video, _degenerate())
        assert cal_set.missing([video]) == [video]
        assert cal_set.px_per_mm(video) is None

    def test_entries_are_keyed_by_resolved_path(self, tmp_path):
        video = tmp_path / "a.mp4"
        video.write_bytes(b"")
        cal_set = CalibrationSet()
        cal_set.set(video, _cal())
        # A different spelling of the same file must hit the same entry.
        assert cal_set.px_per_mm(tmp_path / "sub" / ".." / "a.mp4") is not None

    def test_discard_removes_an_entry(self, tmp_path):
        video = tmp_path / "a.mp4"
        cal_set = CalibrationSet()
        cal_set.set(video, _cal())
        cal_set.discard(video)
        assert cal_set.missing([video]) == [video]


class TestMasterFile:
    def test_round_trip_preserves_lines_and_scale(self, tmp_path):
        video = tmp_path / "session01.mp4"
        video.write_bytes(b"")
        master = tmp_path / "pose_calibration.json"

        original = CalibrationSet()
        original.set(video, _cal(6.4))
        original.save(master, model=tmp_path / "exp-6.pt")

        loaded = CalibrationSet.load(master)
        assert loaded.px_per_mm(video) == pytest.approx(6.4, abs=0.1)
        assert len(loaded.get(video).lines) == 1
        assert loaded.get(video).lines[0].name == "width"

    def test_written_shape_is_the_documented_schema(self, tmp_path):
        video = tmp_path / "session01.mp4"
        master = tmp_path / "m.json"
        cal_set = CalibrationSet()
        cal_set.set(video, _cal(6.4))
        cal_set.save(master, model=tmp_path / "exp-6.pt")

        data = json.loads(master.read_text())
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["model"] == str(tmp_path / "exp-6.pt")
        assert "created" in data
        entry = data["videos"][0]
        assert entry["resolution"] == [640, 480]
        assert entry["px_per_mm"] == pytest.approx(6.4, abs=0.1)
        assert entry["mm_per_px"] == pytest.approx(1 / 6.4, abs=0.01)
        assert entry["calibration"]["lines"], "drawn lines must survive for reload"

    def test_scale_is_recomputed_not_trusted_on_load(self, tmp_path):
        """px_per_mm in the file is for analysts; the lines are the truth."""
        video = tmp_path / "session01.mp4"
        video.write_bytes(b"")
        master = tmp_path / "m.json"
        cal_set = CalibrationSet()
        cal_set.set(video, _cal(6.4))
        cal_set.save(master)

        data = json.loads(master.read_text())
        data["videos"][0]["px_per_mm"] = 999.0  # tampered / stale
        master.write_text(json.dumps(data))

        assert CalibrationSet.load(master).px_per_mm(video) == pytest.approx(6.4, abs=0.1)

    def test_mm_per_px_is_null_when_there_is_no_scale(self, tmp_path):
        master = tmp_path / "m.json"
        cal_set = CalibrationSet()
        cal_set.set(tmp_path / "a.mp4", _degenerate())
        cal_set.save(master)
        entry = json.loads(master.read_text())["videos"][0]
        assert entry["px_per_mm"] == 0
        assert entry["mm_per_px"] is None

    def test_unknown_schema_version_is_refused(self, tmp_path):
        master = tmp_path / "m.json"
        master.write_text(json.dumps({"schema_version": 99, "videos": []}))
        with pytest.raises(CalibrationSetError, match="schema_version"):
            CalibrationSet.load(master)

    def test_malformed_json_is_refused(self, tmp_path):
        master = tmp_path / "m.json"
        master.write_text("{not json")
        with pytest.raises(CalibrationSetError):
            CalibrationSet.load(master)

    def test_relocated_video_matches_on_unique_filename(self, tmp_path):
        """Data copied to another drive must not force a re-calibration."""
        master = tmp_path / "m.json"
        cal_set = CalibrationSet()
        cal_set.set(Path("Z:/gone/session01.mp4"), _cal(6.4))
        cal_set.save(master)

        moved = tmp_path / "newdrive" / "session01.mp4"
        moved.parent.mkdir()
        moved.write_bytes(b"")

        loaded = CalibrationSet.load(master, known_videos=[moved])
        assert loaded.px_per_mm(moved) == pytest.approx(6.4, abs=0.1)

    def test_ambiguous_filename_is_not_guessed(self, tmp_path):
        master = tmp_path / "m.json"
        cal_set = CalibrationSet()
        cal_set.set(Path("Z:/gone/session01.mp4"), _cal(6.4))
        cal_set.save(master)

        a = tmp_path / "one" / "session01.mp4"
        b = tmp_path / "two" / "session01.mp4"
        for p in (a, b):
            p.parent.mkdir()
            p.write_bytes(b"")

        loaded = CalibrationSet.load(master, known_videos=[a, b])
        # Two candidates, no way to know which — apply to neither.
        assert loaded.missing([a, b]) == [a, b]

    def test_load_without_known_videos_keeps_stored_paths(self, tmp_path):
        video = tmp_path / "session01.mp4"
        video.write_bytes(b"")
        master = tmp_path / "m.json"
        cal_set = CalibrationSet()
        cal_set.set(video, _cal(6.4))
        cal_set.save(master)
        assert CalibrationSet.load(master).px_per_mm(video) is not None

    def test_save_creates_parent_directories(self, tmp_path):
        master = tmp_path / "nested" / "deeper" / "m.json"
        cal_set = CalibrationSet()
        cal_set.set(tmp_path / "a.mp4", _cal())
        cal_set.save(master)
        assert master.exists()

    def test_calibration_value_that_is_a_string_is_refused(self, tmp_path):
        master = tmp_path / "m.json"
        master.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "videos": [{"video": "a.mp4", "calibration": "not-a-dict"}],
                }
            )
        )
        with pytest.raises(CalibrationSetError):
            CalibrationSet.load(master)

    def test_calibration_value_that_is_null_is_refused(self, tmp_path):
        master = tmp_path / "m.json"
        master.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "videos": [{"video": "a.mp4", "calibration": None}],
                }
            )
        )
        with pytest.raises(CalibrationSetError):
            CalibrationSet.load(master)

    def test_videos_that_is_not_a_list_is_refused(self, tmp_path):
        master = tmp_path / "m.json"
        master.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "videos": 5}))
        with pytest.raises(CalibrationSetError):
            CalibrationSet.load(master)

    def test_entry_that_is_not_a_dict_is_refused(self, tmp_path):
        master = tmp_path / "m.json"
        master.write_text(
            json.dumps({"schema_version": SCHEMA_VERSION, "videos": [["not", "a", "dict"]]})
        )
        with pytest.raises(CalibrationSetError):
            CalibrationSet.load(master)

    def test_one_malformed_entry_among_valid_ones_applies_nothing(self, tmp_path):
        # The "never half-applied" property: load() must raise, not return a
        # CalibrationSet with only the good entries filled in.
        video = tmp_path / "session01.mp4"
        video.write_bytes(b"")
        good = CalibrationSet()
        good.set(video, _cal(6.4))
        data = good.to_dict()
        data["videos"].append({"video": "bad.mp4", "calibration": "oops"})
        master = tmp_path / "m.json"
        master.write_text(json.dumps(data))

        with pytest.raises(CalibrationSetError):
            CalibrationSet.load(master)

    def test_a_failed_save_leaves_the_previous_master_file_intact(self, tmp_path, monkeypatch):
        video = tmp_path / "session01.mp4"
        video.write_bytes(b"")
        master = tmp_path / "m.json"

        original = CalibrationSet()
        original.set(video, _cal(6.4))
        original.save(master)
        original_bytes = master.read_bytes()

        def boom(*_args, **_kwargs):
            raise OSError("simulated failure")

        monkeypatch.setattr("glider.vision.calibration_set.os.replace", boom)

        updated = CalibrationSet()
        updated.set(video, _cal(3.2))
        with pytest.raises(OSError):
            updated.save(master)

        assert master.read_bytes() == original_bytes
        assert CalibrationSet.load(master).px_per_mm(video) == pytest.approx(6.4, abs=0.1)
        # No leaked temp file next to the master.
        assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())

    def test_entries_are_written_in_sorted_path_order(self, tmp_path):
        b_video = tmp_path / "b.mp4"
        a_video = tmp_path / "a.mp4"
        cal_set = CalibrationSet()
        cal_set.set(b_video, _cal(6.4))
        cal_set.set(a_video, _cal(6.4))
        master = tmp_path / "m.json"
        cal_set.save(master)

        data = json.loads(master.read_text())
        paths = [entry["video"] for entry in data["videos"]]
        assert paths == sorted(paths)

    def test_master_file_ends_with_a_newline(self, tmp_path):
        master = tmp_path / "m.json"
        CalibrationSet().save(master)
        assert master.read_text(encoding="utf-8").endswith("\n")
