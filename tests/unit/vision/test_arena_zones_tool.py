"""The batch driver: zone documents that existing readers understand."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from glider.vision.arena import ArenaCalibration
from glider.vision.zones import ZoneConfiguration, ZoneShape

TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


@pytest.fixture(scope="module")
def tool():
    path = Path(__file__).resolve().parents[3] / "tools" / "arena_zones.py"
    spec = importlib.util.spec_from_file_location("arena_zones", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cal():
    return ArenaCalibration(corners=TRAPEZOID, frame_size=(640, 480))


class TestZoneDocument:
    def test_zones_module_can_load_it(self, tool, cal, tmp_path):
        path = tmp_path / "t1_zone.json"
        path.write_text(json.dumps(tool.zone_document(cal, 10.0)))
        config = ZoneConfiguration()
        assert config.load(path) is True
        assert len(config.zones) == 1

    def test_it_is_a_polygon_not_a_rectangle(self, tool, cal):
        # A perspective-correct centred square is a quadrilateral. Storing it
        # as a rectangle would reinstate exactly the error being removed.
        zone = tool.zone_document(cal, 10.0)["zones"][0]
        assert zone["shape"] == ZoneShape.POLYGON.value
        assert len(zone["vertices"]) == 4

    def test_it_records_the_frame_size(self, tool, cal):
        doc = tool.zone_document(cal, 10.0)
        assert (doc["config_width"], doc["config_height"]) == (640, 480)

    def test_the_zone_contains_the_arena_centre(self, tool, cal):
        doc = tool.zone_document(cal, 10.0)
        config = ZoneConfiguration.from_dict(doc)
        centre = cal.to_image([(15.0, 15.0)])[0]
        assert config.zones[0].contains_point(*centre) is True

    def test_the_zone_excludes_the_arena_corners(self, tool, cal):
        config = ZoneConfiguration.from_dict(tool.zone_document(cal, 10.0))
        for corner in TRAPEZOID:
            assert config.zones[0].contains_point(*corner) is False

    def test_a_point_just_outside_10cm_is_excluded(self, tool, cal):
        config = ZoneConfiguration.from_dict(tool.zone_document(cal, 10.0))
        just_in = cal.to_image([(15.0, 10.2)])[0]
        just_out = cal.to_image([(15.0, 9.8)])[0]
        assert config.zones[0].contains_point(*just_in) is True
        assert config.zones[0].contains_point(*just_out) is False

    def test_ids_are_unique_across_documents(self, tool, cal):
        a = tool.zone_document(cal, 10.0)["zones"][0]["id"]
        b = tool.zone_document(cal, 10.0)["zones"][0]["id"]
        assert a != b


class TestPersistence:
    def test_calibrations_round_trip(self, tool, cal, tmp_path):
        path = tmp_path / "arena_calibration.json"
        tool.save_calibrations(path, {"t1_d2": cal}, 10.0)
        loaded = tool.load_calibrations(path)
        assert loaded["t1_d2"].corners == cal.corners
        assert loaded["t1_d2"].frame_size == cal.frame_size

    def test_missing_file_loads_as_empty(self, tool, tmp_path):
        assert tool.load_calibrations(tmp_path / "nope.json") == {}

    def test_write_zones_names_files_after_the_video(self, tool, cal, tmp_path):
        written = tool.write_zones({"Test 1": cal, "t9_d2": cal}, tmp_path / "out", 10.0)
        assert {p.name for p in written} == {"Test 1_zone.json", "t9_d2_zone.json"}

    def test_write_zones_creates_its_directory(self, tool, cal, tmp_path):
        out = tmp_path / "deep" / "arena_zones"
        tool.write_zones({"a": cal}, out, 10.0)
        assert out.is_dir()


class TestVideoDiscovery:
    def test_finds_videos_and_ignores_everything_else(self, tool, tmp_path):
        for name in ["Test 1.mp4", "t1_d2.MP4", "notes.txt", "zone.json", "clip.avi"]:
            (tmp_path / name).touch()
        assert [p.name for p in tool.find_videos(tmp_path)] == [
            "clip.avi",
            "t1_d2.MP4",
            "Test 1.mp4",
        ]
