"""Moving zones and making them at a set size (#223): the Qt-free geometry."""

import pytest

from glider.vision.arena import ArenaCalibration
from glider.vision.zones import (
    Zone,
    ZoneConfiguration,
    ZoneShape,
    arena_sized_zone_vertices,
    clamp_translation,
    sized_zone_vertices,
    translate_zone,
)


def _zone(shape, vertices, zid="z"):
    return Zone(id=zid, name=zid, shape=shape, vertices=vertices)


class TestTranslateZone:
    @pytest.mark.parametrize(
        "shape, vertices",
        [
            (ZoneShape.RECTANGLE, [(0.2, 0.2), (0.4, 0.5)]),
            (ZoneShape.CIRCLE, [(0.5, 0.5), (0.6, 0.5)]),
            (ZoneShape.POLYGON, [(0.3, 0.3), (0.5, 0.3), (0.4, 0.6)]),
        ],
    )
    def test_every_vertex_moves_by_the_same_amount(self, shape, vertices):
        zone = _zone(shape, list(vertices))
        translate_zone(zone, 0.1, -0.05)
        for (x0, y0), (x1, y1) in zip(vertices, zone.vertices, strict=True):
            assert x1 == pytest.approx(x0 + 0.1)
            assert y1 == pytest.approx(y0 - 0.05)
        assert zone.shape == shape
        assert zone.id == "z"

    def test_a_rectangle_stops_at_the_frame_edge(self):
        zone = _zone(ZoneShape.RECTANGLE, [(0.7, 0.1), (0.9, 0.3)])
        translate_zone(zone, 0.5, -0.5)
        assert zone.vertices[1][0] == pytest.approx(1.0)
        assert zone.vertices[0][1] == pytest.approx(0.0)
        # Size is preserved: the move is clipped, the zone is not squashed.
        assert zone.vertices[1][0] - zone.vertices[0][0] == pytest.approx(0.2)

    def test_a_circle_keeps_its_centre_on_the_frame(self):
        zone = _zone(ZoneShape.CIRCLE, [(0.5, 0.5), (0.6, 0.5)])
        translate_zone(zone, 2.0, 0.0)
        assert zone.vertices[0][0] == pytest.approx(1.0)
        assert zone.vertices[1][0] == pytest.approx(1.1)

    def test_a_zone_already_over_the_edge_can_come_back_but_not_go_further(self):
        zone = _zone(ZoneShape.RECTANGLE, [(-0.1, 0.4), (0.2, 0.6)])
        assert clamp_translation(zone, -0.1, 0.0) == (0.0, 0.0)
        assert clamp_translation(zone, 0.05, 0.0) == pytest.approx((0.05, 0.0))

    def test_unclamped_moves_go_anywhere(self):
        zone = _zone(ZoneShape.RECTANGLE, [(0.7, 0.1), (0.9, 0.3)])
        translate_zone(zone, 0.5, 0.0, clamp=False)
        assert zone.vertices[0][0] == pytest.approx(1.2)

    def test_a_moved_zone_still_round_trips(self):
        zone = _zone(ZoneShape.POLYGON, [(0.3, 0.3), (0.5, 0.3), (0.4, 0.6)])
        translate_zone(zone, 0.1, 0.1)
        again = Zone.from_dict(zone.to_dict())
        assert again.vertices == pytest.approx(zone.vertices)


class TestZoneAt:
    def test_picks_the_topmost_of_overlapping_zones(self):
        config = ZoneConfiguration()
        config.add_zone(_zone(ZoneShape.RECTANGLE, [(0.0, 0.0), (0.6, 0.6)], "under"))
        config.add_zone(_zone(ZoneShape.RECTANGLE, [(0.4, 0.4), (1.0, 1.0)], "over"))
        assert config.zone_at(0.5, 0.5).id == "over"
        assert config.zone_at(0.1, 0.1).id == "under"
        assert ZoneConfiguration().zone_at(0.5, 0.5) is None


class TestSizedZone:
    def test_rectangle_is_centred_and_sized_in_pixels(self):
        verts = sized_zone_vertices(ZoneShape.RECTANGLE, (0.5, 0.5), 64, 48, (640, 480))
        (x1, y1), (x2, y2) = verts
        assert (x2 - x1) * 640 == pytest.approx(64)
        assert (y2 - y1) * 480 == pytest.approx(48)
        assert ((x1 + x2) / 2, (y1 + y2) / 2) == pytest.approx((0.5, 0.5))

    def test_circle_diameter_is_in_pixels_on_any_aspect(self):
        verts = sized_zone_vertices(ZoneShape.CIRCLE, (0.5, 0.5), 100, 0, (640, 480))
        zone = _zone(ZoneShape.CIRCLE, verts)
        pts = zone.get_pixel_vertices(640, 480)
        assert pts[1][0] - pts[0][0] == 50  # radius in pixels
        # The drawn circle and the scored circle agree: 49 px out is in, 51 is out.
        assert zone.contains_point_pixels(320, 240 + 49, 640, 480)
        assert not zone.contains_point_pixels(320, 240 + 51, 640, 480)

    def test_polygons_and_bad_sizes_are_refused(self):
        with pytest.raises(ValueError):
            sized_zone_vertices(ZoneShape.POLYGON, (0.5, 0.5), 10, 10, (640, 480))
        with pytest.raises(ValueError):
            sized_zone_vertices(ZoneShape.RECTANGLE, (0.5, 0.5), 0, 10, (640, 480))


class TestArenaSizedZone:
    @pytest.fixture
    def arena(self):
        # A 40 x 30 cm floor filling a 640 x 480 frame: 16 px/cm both ways.
        return ArenaCalibration(
            corners=[(0, 0), (1, 0), (1, 1), (0, 1)],
            width_cm=40.0,
            height_cm=30.0,
            frame_size=(640, 480),
        )

    def test_a_square_is_its_size_on_the_floor(self, arena):
        verts = arena_sized_zone_vertices(arena, ZoneShape.RECTANGLE, (0.5, 0.5), 10, 5)
        assert len(verts) == 4
        floor = arena.to_arena_cm(verts)
        xs = [p[0] for p in floor]
        ys = [p[1] for p in floor]
        assert max(xs) - min(xs) == pytest.approx(10)
        assert max(ys) - min(ys) == pytest.approx(5)
        assert (sum(xs) / 4, sum(ys) / 4) == pytest.approx((20, 15))

    def test_a_circle_is_a_polygon_of_the_right_diameter(self, arena):
        verts = arena_sized_zone_vertices(arena, ZoneShape.CIRCLE, (0.25, 0.5), 8, 0)
        assert len(verts) == 48
        floor = arena.to_arena_cm(verts)
        for x, y in floor:
            assert ((x - 10) ** 2 + (y - 15) ** 2) ** 0.5 == pytest.approx(4)

    def test_polygons_are_refused(self, arena):
        with pytest.raises(ValueError):
            arena_sized_zone_vertices(arena, ZoneShape.POLYGON, (0.5, 0.5), 10, 10)
