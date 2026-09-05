"""CalibrationSet carrying an arena per video alongside the drawn lines."""

from __future__ import annotations

import json

import pytest

from glider.vision.arena import ArenaCalibration
from glider.vision.calibration import CameraCalibration, LengthUnit
from glider.vision.calibration_set import SCHEMA_VERSION, CalibrationSet

TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "t1_d2.mp4"
    path.touch()
    return path


@pytest.fixture
def arena():
    return ArenaCalibration(corners=TRAPEZOID, frame_size=(640, 480))


@pytest.fixture
def line():
    cal = CameraCalibration()
    cal.add_line((100, 400), (500, 400), 30.0, LengthUnit.CENTIMETERS, resolution=(640, 480))
    return cal


class TestArenaStorage:
    def test_set_and_get(self, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        assert cal_set.get_arena(video) is arena

    def test_absent_arena_is_none(self, video):
        assert CalibrationSet().get_arena(video) is None

    def test_discard_removes_it(self, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        cal_set.discard_arena(video)
        assert cal_set.get_arena(video) is None

    def test_two_spellings_of_a_path_share_an_arena(self, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        assert cal_set.get_arena(video.parent / "." / video.name) is arena

    def test_subset_carries_arenas(self, video, arena, tmp_path):
        other = tmp_path / "other.mp4"
        other.touch()
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        cal_set.set_arena(other, arena)
        picked = cal_set.subset([video])
        assert picked.get_arena(video) is arena
        assert picked.get_arena(other) is None


class TestScalePrecedence:
    def test_arena_supplies_the_scale(self, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        assert cal_set.px_per_mm(video) == pytest.approx(arena.px_per_cm_centre / 10.0)

    def test_arena_wins_over_a_drawn_line(self, video, arena, line):
        # A closed square is the better-constrained measurement, and it reports
        # the scale at the arena centre rather than wherever the line landed.
        cal_set = CalibrationSet()
        cal_set.set(video, line)
        cal_set.set_arena(video, arena)
        assert cal_set.px_per_mm(video) == pytest.approx(arena.px_per_cm_centre / 10.0)
        assert cal_set.px_per_mm(video) != pytest.approx(line.pixels_per_mm)

    def test_line_is_used_when_no_arena_is_drawn(self, video, line):
        cal_set = CalibrationSet()
        cal_set.set(video, line)
        assert cal_set.px_per_mm(video) == pytest.approx(line.pixels_per_mm)

    def test_a_degenerate_arena_falls_back_to_the_line(self, video, line):
        # A half-finished arena must not knock out a scale that already works.
        cal_set = CalibrationSet()
        cal_set.set(video, line)
        cal_set.set_arena(video, ArenaCalibration(corners=[(0.5, 0.5)] * 4))
        assert cal_set.px_per_mm(video) == pytest.approx(line.pixels_per_mm)

    def test_a_degenerate_arena_alone_gives_no_scale(self, video):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, ArenaCalibration(corners=[(0.5, 0.5)] * 4))
        assert cal_set.px_per_mm(video) is None

    def test_an_arena_alone_counts_as_calibrated(self, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        assert cal_set.missing([video]) == []
        assert cal_set.is_complete([video]) is True


class TestMasterFile:
    def test_arena_round_trips(self, tmp_path, video, arena, line):
        cal_set = CalibrationSet()
        cal_set.set(video, line)
        cal_set.set_arena(video, arena)
        path = tmp_path / "master.json"
        cal_set.save(path)

        loaded = CalibrationSet.load(path)
        assert loaded.get_arena(video).corners == arena.corners
        assert loaded.get_arena(video).frame_size == arena.frame_size
        assert loaded.get_arena(video).width_cm == arena.width_cm

    def test_an_arena_only_video_survives_a_round_trip(self, tmp_path, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        path = tmp_path / "master.json"
        cal_set.save(path)

        loaded = CalibrationSet.load(path)
        assert loaded.get_arena(video).corners == arena.corners
        assert loaded.px_per_mm(video) == pytest.approx(arena.px_per_cm_centre / 10.0)

    def test_an_arena_only_video_reports_the_frame_resolution(self, tmp_path, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        path = tmp_path / "master.json"
        cal_set.save(path)
        entry = json.loads(path.read_text())["videos"][0]
        assert entry["resolution"] == [640, 480]

    def test_derived_scale_is_written_for_analysts(self, tmp_path, video, arena):
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        path = tmp_path / "master.json"
        cal_set.save(path)
        entry = json.loads(path.read_text())["videos"][0]
        assert entry["px_per_mm"] == pytest.approx(arena.px_per_cm_centre / 10.0)

    def test_the_schema_version_is_unchanged(self, tmp_path, video, arena):
        # The arena rides in an optional per-entry key. Older builds read the
        # keys they know and ignore this one, so the file stays readable by
        # them and does not need a version bump.
        cal_set = CalibrationSet()
        cal_set.set_arena(video, arena)
        path = tmp_path / "master.json"
        cal_set.save(path)
        assert json.loads(path.read_text())["schema_version"] == SCHEMA_VERSION

    def test_a_file_with_no_arenas_still_loads(self, tmp_path, video, line):
        cal_set = CalibrationSet()
        cal_set.set(video, line)
        path = tmp_path / "master.json"
        cal_set.save(path)
        assert "arena" not in json.loads(path.read_text())["videos"][0]

        loaded = CalibrationSet.load(path)
        assert loaded.get_arena(video) is None
        assert loaded.px_per_mm(video) == pytest.approx(line.pixels_per_mm)

    def test_a_malformed_arena_aborts_the_whole_load(self, tmp_path, video, line):
        # Matches how the rest of this loader behaves: a batch run must never
        # operate on a half-read map.
        from glider.vision.calibration_set import CalibrationSetError

        cal_set = CalibrationSet()
        cal_set.set(video, line)
        path = tmp_path / "master.json"
        cal_set.save(path)
        data = json.loads(path.read_text())
        data["videos"][0]["arena"] = {"corners": [[0.1, 0.1]]}
        path.write_text(json.dumps(data))

        with pytest.raises(CalibrationSetError, match="malformed"):
            CalibrationSet.load(path)


def _set_with_arena(tmp_path, *, confirmed=True):
    video = tmp_path / "s1.mp4"
    video.write_bytes(b"")
    cal_set = CalibrationSet()
    cal_set.set_arena(video, ArenaCalibration(corners=TRAPEZOID), confirmed=confirmed)
    return cal_set, video


class TestConfirmedState:
    def test_a_drawn_arena_is_confirmed(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path)
        assert cal_set.is_arena_confirmed(video)

    def test_an_unconfirmed_arena_counts_as_missing(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        assert cal_set.get_arena(video) is not None
        assert cal_set.missing_arenas([video]) == [video]

    def test_confirming_clears_it(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        cal_set.set_arena(video, cal_set.get_arena(video), confirmed=True)
        assert cal_set.missing_arenas([video]) == []

    def test_a_degenerate_arena_counts_as_missing(self, tmp_path):
        video = tmp_path / "s1.mp4"
        cal_set = CalibrationSet()
        cal_set.set_arena(video, ArenaCalibration(corners=[(0.5, 0.5)] * 4))
        assert cal_set.missing_arenas([video]) == [video]

    def test_discarding_clears_the_unconfirmed_flag(self, tmp_path):
        """A stale flag must not outlive the arena it described.

        Written against a direct ``arenas`` write rather than ``set_arena``,
        because that is the path that actually breaks: ``_load_master`` does
        ``self._calibrations.arenas.update(loaded.arenas)``, bypassing
        ``set_arena`` and its flag-clearing. Going through ``set_arena`` here
        instead would clear the flag on its own and the test would pass even
        with the cleanup removed from ``discard_arena``.
        """
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        cal_set.discard_arena(video)
        cal_set.arenas.update({cal_set._key(video): ArenaCalibration(corners=TRAPEZOID)})
        assert cal_set.is_arena_confirmed(video)

    def test_subset_carries_confirmed_state(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        assert not cal_set.subset([video]).is_arena_confirmed(video)


def test_confirmed_arenas_write_no_extra_key(tmp_path):
    """A normal file must stay byte-identical to what earlier builds wrote."""
    cal_set, _ = _set_with_arena(tmp_path)
    assert "arena_confirmed" not in cal_set.to_dict()["videos"][0]


def test_unconfirmed_arenas_round_trip(tmp_path):
    cal_set, video = _set_with_arena(tmp_path, confirmed=False)
    master = tmp_path / "m.json"
    cal_set.save(master)
    assert not CalibrationSet.load(master, known_videos=[video]).is_arena_confirmed(video)


def test_a_file_without_the_key_loads_as_confirmed(tmp_path):
    """Every master file written before this change. Absent means drawn."""
    cal_set, video = _set_with_arena(tmp_path)
    master = tmp_path / "m.json"
    cal_set.save(master)
    assert CalibrationSet.load(master, known_videos=[video]).is_arena_confirmed(video)
