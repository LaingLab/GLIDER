"""PoseBatchWindow: arenas becoming the centre zones a run scores against."""

from __future__ import annotations

from pathlib import Path

import pytest

from glider.gui.pose_batch.window import PoseBatchWindow
from glider.vision.arena import ArenaCalibration
from glider.vision.zones import ZoneShape

TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


@pytest.fixture
def window(qtbot):
    widget = PoseBatchWindow()
    qtbot.addWidget(widget)
    return widget


def _video(tmp_path, name="t1_d2.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"")
    return path.resolve()


def _arena(corners=TRAPEZOID) -> ArenaCalibration:
    return ArenaCalibration(corners=corners, frame_size=(640, 480))


class TestZoneConfigs:
    def test_a_drawn_arena_becomes_a_zone(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        configs = window._zone_configs()
        assert list(configs) == [video]
        assert len(configs[video].zones) == 1

    def test_the_zone_is_a_polygon(self, window, tmp_path):
        # Under perspective a centred square is a quadrilateral; storing it as
        # a rectangle would put back the error the arena removes.
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        zone = window._zone_configs()[video].zones[0]
        assert zone.shape is ZoneShape.POLYGON
        assert len(zone.vertices) == 4

    def test_the_zone_records_the_frame_size(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        config = window._zone_configs()[video]
        assert (config.config_width, config.config_height) == (640, 480)

    def test_the_zone_name_carries_its_size(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._zone_cm = 12.0
        window._calibrations.set_arena(video, _arena())
        assert "12cm" in window._zone_configs()[video].zones[0].name

    def test_the_zone_follows_the_chosen_size(self, window, tmp_path):
        # Not a fixed ratio in image space: the mapping is projective, so a
        # 20 cm zone is not twice as wide on screen as a 10 cm one. What must
        # hold is that the larger zone strictly encloses the smaller.

        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        window._zone_cm = 10.0
        small = window._zone_configs()[video].zones[0].vertices
        window._zone_cm = 20.0
        large_config = window._zone_configs()[video]

        larger = large_config.zones[0]
        assert all(larger.contains_point(x, y) for x, y in small)
        span = lambda vs: max(v[0] for v in vs) - min(v[0] for v in vs)  # noqa: E731
        assert span(larger.vertices) > span(small)

    def test_videos_without_an_arena_get_no_zone(self, window, tmp_path):
        drawn, blank = _video(tmp_path, "a.mp4"), _video(tmp_path, "b.mp4")
        window._videos = [drawn, blank]
        window._calibrations.set_arena(drawn, _arena())
        assert list(window._zone_configs()) == [drawn]

    def test_no_arenas_means_no_zones(self, window, tmp_path):
        window._videos = [_video(tmp_path)]
        assert window._zone_configs() == {}


class TestBadArenas:
    def test_a_degenerate_arena_is_skipped_not_raised(self, window, tmp_path):
        # Pose inference is the expensive part of a run; a half-drawn zone must
        # not be able to stop it.
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena([(0.5, 0.5)] * 4))
        assert window._zone_configs() == {}

    def test_the_skip_is_reported(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena([(0.5, 0.5)] * 4))
        window._zone_configs()
        assert "no zone written" in window._log.toPlainText()

    def test_one_bad_arena_does_not_lose_the_others(self, window, tmp_path):
        good, bad = _video(tmp_path, "a.mp4"), _video(tmp_path, "b.mp4")
        window._videos = [good, bad]
        window._calibrations.set_arena(good, _arena())
        window._calibrations.set_arena(bad, _arena([(0.5, 0.5)] * 4))
        assert list(window._zone_configs()) == [good]

    def test_a_zone_larger_than_the_arena_is_skipped(self, window, tmp_path):
        video = _video(tmp_path)
        window._videos = [video]
        window._zone_cm = 40.0
        window._calibrations.set_arena(video, _arena())
        assert window._zone_configs() == {}


class TestRunGating:
    def test_an_arena_alone_satisfies_the_calibration_gate(self, window, tmp_path):
        # px_per_mm() answers from the arena, so a video with a perimeter and
        # no drawn line is calibrated as far as Run is concerned.
        video = _video(tmp_path)
        window._videos = [video]
        window._calibrations.set_arena(video, _arena())
        assert window._calibrations.missing(window._videos) == []


def test_load_master_applies_arenas(window, tmp_path):
    """Regression: _load_master applied entries and dropped arenas."""
    from glider.vision.calibration_set import CalibrationSet

    video = _video(tmp_path)
    master = tmp_path / "master.json"
    seed = CalibrationSet()
    seed.set_arena(video, _arena())
    seed.save(master)

    window._videos = [video.resolve()]
    window._load_master(master)

    assert window._calibrations.get_arena(video) is not None
