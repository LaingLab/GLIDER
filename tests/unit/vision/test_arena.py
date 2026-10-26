"""Arena calibration: image <-> arena-centimetre geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from glider.vision.arena import ArenaCalibration, DegenerateArenaError

# A fronto-parallel arena: a genuine 400x400 px square inside a 640x480 frame,
# which is close to what these rigs produce (30 cm at ~13.3 px/cm). Written as
# pixel spans over the frame size so it stays square rather than merely looking
# it - equal spans in *normalized* coordinates would be a 400x408 rectangle.
_L, _R, _T, _B = 120 / 640, 520 / 640, 40 / 480, 440 / 480
SQUARE = [(_L, _T), (_R, _T), (_R, _B), (_L, _B)]

# A trapezoid: the far (top) edge is narrower, which is what a camera tilted
# over the arena produces. Symmetric about x=0.5 so the centre is predictable,
# and the taper is kept near what the real videos show - t6_d2, the one session
# with all four corners in frame, runs about 6% narrower at the far wall.
TRAPEZOID = [(0.28, 0.1), (0.72, 0.1), (0.76, 0.9), (0.24, 0.9)]


def _cal(corners=SQUARE, **kw):
    return ArenaCalibration(corners=corners, **kw)


class TestConstruction:
    def test_defaults_to_a_30cm_square_arena(self):
        cal = _cal()
        assert cal.width_cm == 30.0
        assert cal.height_cm == 30.0

    def test_requires_exactly_four_corners(self):
        with pytest.raises(ValueError, match="four corners"):
            ArenaCalibration(corners=SQUARE[:3])

    def test_rejects_a_collapsed_quad(self):
        with pytest.raises(DegenerateArenaError):
            _cal(corners=[(0.5, 0.5)] * 4).homography()

    def test_rejects_a_bowtie(self):
        # Swapping two adjacent corners makes the quad self-intersect, which
        # still has a homography but not one that means anything.
        bowtie = [SQUARE[0], SQUARE[1], SQUARE[3], SQUARE[2]]
        with pytest.raises(DegenerateArenaError):
            _cal(corners=bowtie).homography()


class TestRoundTrip:
    @pytest.mark.parametrize("corners", [SQUARE, TRAPEZOID])
    def test_image_to_cm_and_back_is_identity(self, corners):
        cal = _cal(corners)
        points = [(0.1, 0.1), (0.5, 0.5), (0.83, 0.42), (0.25, 0.77)]
        back = cal.to_image(cal.to_arena_cm(points))
        for (x0, y0), (x1, y1) in zip(points, back, strict=True):
            assert x1 == pytest.approx(x0, abs=1e-9)
            assert y1 == pytest.approx(y0, abs=1e-9)

    @pytest.mark.parametrize("corners", [SQUARE, TRAPEZOID])
    def test_corners_map_to_the_arena_corners(self, corners):
        cal = _cal(corners)
        got = cal.to_arena_cm(corners)
        expected = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
        for (gx, gy), (ex, ey) in zip(got, expected, strict=True):
            assert gx == pytest.approx(ex, abs=1e-6)
            assert gy == pytest.approx(ey, abs=1e-6)


class TestCentreZone:
    def test_centre_of_a_square_arena_is_the_centre_of_the_quad(self):
        cal = _cal(SQUARE)
        vs = cal.centre_zone_vertices(10.0)
        cx = sum(v[0] for v in vs) / 4
        cy = sum(v[1] for v in vs) / 4
        assert cx == pytest.approx((_L + _R) / 2, abs=1e-9)
        assert cy == pytest.approx((_T + _B) / 2, abs=1e-9)

    def test_fronto_parallel_centre_zone_is_one_third_of_the_arena(self):
        cal = _cal(SQUARE)
        vs = cal.centre_zone_vertices(10.0)
        width = max(v[0] for v in vs) - min(v[0] for v in vs)
        height = max(v[1] for v in vs) - min(v[1] for v in vs)
        assert width == pytest.approx((_R - _L) / 3, abs=1e-9)
        assert height == pytest.approx((_B - _T) / 3, abs=1e-9)

    def test_perspective_makes_the_far_edge_shorter(self):
        # The whole point of the homography: on a tilted view a centred square
        # is not axis-aligned, and its far edge is narrower than its near edge.
        vs = _cal(TRAPEZOID).centre_zone_vertices(10.0)
        far = math.dist(vs[0], vs[1])
        near = math.dist(vs[3], vs[2])
        assert far < near

    def test_centre_zone_scales_with_requested_size(self):
        cal = _cal(SQUARE)
        small = cal.centre_zone_vertices(10.0)
        large = cal.centre_zone_vertices(20.0)
        span = lambda vs: max(v[0] for v in vs) - min(v[0] for v in vs)  # noqa: E731
        assert span(large) == pytest.approx(2 * span(small), abs=1e-9)

    def test_a_full_size_centre_zone_is_the_arena_itself(self):
        cal = _cal(TRAPEZOID)
        vs = cal.centre_zone_vertices(30.0)
        for (gx, gy), (ex, ey) in zip(vs, TRAPEZOID, strict=True):
            assert gx == pytest.approx(ex, abs=1e-9)
            assert gy == pytest.approx(ey, abs=1e-9)

    def test_rejects_a_zone_larger_than_the_arena(self):
        with pytest.raises(ValueError, match="larger than the arena"):
            _cal().centre_zone_vertices(40.0)


class TestOffFrameCorners:
    # Several videos in this cohort have the far arena corners outside the
    # frame. The projected corner is still a well-defined point, so normalized
    # coordinates outside 0-1 must survive rather than being clamped.
    CLIPPED = [(0.28, -0.14), (0.79, -0.11), (0.86, 0.92), (0.19, 0.9)]

    def test_accepts_corners_outside_the_frame(self):
        cal = _cal(self.CLIPPED)
        assert cal.to_arena_cm([self.CLIPPED[0]])[0] == pytest.approx((0.0, 0.0), abs=1e-6)

    def test_centre_zone_of_a_clipped_arena_still_lands_in_frame(self):
        vs = _cal(self.CLIPPED).centre_zone_vertices(10.0)
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in vs)

    def test_reports_when_the_arena_is_clipped(self):
        assert _cal(self.CLIPPED).is_clipped is True
        assert _cal(SQUARE).is_clipped is False


class TestLocalScale:
    def test_fronto_parallel_scale_is_uniform(self):
        cal = _cal(SQUARE, frame_size=(640, 480))
        centre = cal.px_per_cm_at(15.0, 15.0)
        for x, y in [(2.0, 2.0), (28.0, 3.0), (15.0, 28.0)]:
            assert cal.px_per_cm_at(x, y) == pytest.approx(centre, rel=1e-6)

    def test_fronto_parallel_scale_matches_the_drawn_edge(self):
        cal = _cal(SQUARE, frame_size=(640, 480))
        # The arena is 400 px across and represents 30 cm.
        assert cal.px_per_cm_at(15.0, 15.0) == pytest.approx(400 / 30, rel=1e-9)

    def test_perspective_scale_is_larger_near_the_camera(self):
        cal = _cal(TRAPEZOID, frame_size=(640, 480))
        assert cal.px_per_cm_at(15.0, 28.0) > cal.px_per_cm_at(15.0, 2.0)

    def test_centre_scale_is_the_scale_at_the_arena_centre(self):
        cal = _cal(TRAPEZOID, frame_size=(640, 480))
        assert cal.px_per_cm_centre == pytest.approx(cal.px_per_cm_at(15.0, 15.0))


class TestResiduals:
    def test_a_square_has_no_residual(self):
        r = _cal(SQUARE).residuals()
        assert r["edge_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert r["scale_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert r["suspect"] is False

    def test_mild_perspective_is_not_flagged(self):
        assert _cal(TRAPEZOID).residuals()["suspect"] is False

    def test_a_degenerate_quad_is_refused_not_reported(self):
        # residuals() and homography() must fail the same way, so a caller
        # asking either question about four points catches one error type.
        with pytest.raises(DegenerateArenaError):
            _cal(corners=[(0.5, 0.5)] * 4).residuals()

    def test_a_lopsided_quad_is_flagged(self):
        # One corner dragged far out of place: opposite edges no longer agree.
        lopsided = [(0.3, 0.1), (0.95, 0.1), (0.5, 0.9), (0.2, 0.9)]
        assert _cal(lopsided).residuals()["suspect"] is True

    def test_scale_ratio_grows_with_obliqueness(self):
        # The second measure is not redundant with edge_ratio: it reads the
        # implied camera pose rather than the outline, so it keeps rising as
        # the view tilts.
        def ratio(top_half_width):
            quad = [
                (0.5 - top_half_width, 0.1),
                (0.5 + top_half_width, 0.1),
                (0.8, 0.9),
                (0.2, 0.9),
            ]
            return _cal(quad).residuals()["scale_ratio"]

        assert ratio(0.30) < ratio(0.22) < ratio(0.14)
        assert _cal(SQUARE).residuals()["scale_ratio"] == pytest.approx(1.0, abs=1e-9)


class TestSerialization:
    def test_round_trips_through_a_dict(self):
        cal = _cal(TRAPEZOID, frame_size=(640, 480), width_cm=30.0, height_cm=30.0)
        restored = ArenaCalibration.from_dict(cal.to_dict())
        assert restored.corners == cal.corners
        assert restored.width_cm == cal.width_cm
        assert restored.frame_size == cal.frame_size

    def test_dict_is_json_safe(self):
        import json

        json.dumps(_cal().to_dict())


# ---------------------------------------------------------------- round arenas


def test_a_round_arena_seen_from_above_scales_and_contains_like_a_circle():
    """Issue #212: a 40 cm circle drawn as a 400 px-wide ellipse on 800x800."""
    arena = ArenaCalibration.from_ellipse(
        (0.5, 0.5), (0.75, 0.5), (0.5, 0.75), width_cm=40.0, frame_size=(800, 800)
    )

    assert arena.px_per_cm_centre == pytest.approx(10.0)
    # The box corner is inside the bounding square but outside the floor.
    assert arena.outside_cm(np.array([2.0]), np.array([2.0]))[0]
    assert not arena.outside_cm(np.array([20.0]), np.array([1.0]))[0]
    assert len(arena.centre_zone_vertices(10.0)) == 48


def test_a_round_arena_survives_a_round_trip_and_reopens_on_its_handles():
    arena = ArenaCalibration.from_ellipse(
        (0.4, 0.5), (0.7, 0.55), (0.42, 0.8), width_cm=35.0, frame_size=(640, 480)
    )
    again = ArenaCalibration.from_dict(arena.to_dict())

    assert again.shape == "ellipse"
    handles = np.array(again.ellipse_handles())
    assert handles == pytest.approx(np.array([(0.4, 0.5), (0.7, 0.55), (0.42, 0.8)]))


def test_an_arena_stands_in_for_the_line_calibration_the_logger_reads():
    """#212: the live rig's scale comes from its arena, through the same calls."""
    arena = ArenaCalibration.from_ellipse(
        (0.5, 0.5), (0.75, 0.5), (0.5, 0.75), width_cm=40.0, frame_size=(800, 800)
    )

    assert arena.is_calibrated
    assert arena.pixels_per_mm == pytest.approx(1.0)
    assert arena.pixels_to_mm(10.0, 800) == pytest.approx(10.0)
    assert arena.pixels_to_mm(10.0, 400) == pytest.approx(20.0)  # half-res frame
    assert arena.draw(np.zeros((800, 800, 3), np.uint8)).any()


# ---------------------------------------------------------------- plus mazes

# A 75 cm EPM seen straight down on an 800x800 frame at 10 px/cm, centred:
# arm ends at the middles of a 750 px box.
_EPM_ENDS = [(0.5, 25 / 800), (775 / 800, 0.5), (0.5, 775 / 800), (25 / 800, 0.5)]


def _epm(ends=_EPM_ENDS, **kw):
    kw = {"width_cm": 75.0, "arm_width_cm": 5.0, "frame_size": (800, 800), **kw}
    return ArenaCalibration.from_plus(ends, **kw)


class TestPlusMaze:
    """Issue #217: an elevated plus maze, outlined by the ends of its arms."""

    def test_arm_ends_fix_the_bounding_square(self):
        arena = _epm()
        assert arena.shape == "plus"
        expected = [(25 / 800, 25 / 800), (775 / 800, 25 / 800), (775 / 800, 775 / 800)]
        assert np.array(arena.corners[:3]) == pytest.approx(np.array(expected))
        assert arena.px_per_cm_centre == pytest.approx(10.0)

    def test_handles_round_trip_under_perspective(self):
        ends = [(0.5, 0.12), (0.84, 0.52), (0.5, 0.93), (0.15, 0.52)]
        arena = _epm(ends)
        assert np.array(arena.plus_handles()) == pytest.approx(np.array(ends))
        assert np.array(arena.to_arena_cm(ends)) == pytest.approx(
            np.array([(37.5, 0.0), (75.0, 37.5), (37.5, 75.0), (0.0, 37.5)]), abs=1e-9
        )

    def test_containment_is_the_cross_not_the_box(self):
        arena = _epm()
        # centre, left arm, top arm, right arm, empty corner, gap between arms, arm edge
        x = np.array([37.5, 2.0, 37.5, 73.0, 2.0, 30.0, 37.5])
        y = np.array([37.5, 37.5, 2.0, 37.5, 2.0, 30.0, 39.9])
        outside = arena.outside_cm(x, y)
        assert outside.tolist() == [False, False, False, False, True, True, False]

    def test_margin_widens_each_arm(self):
        arena = _epm()
        # 2 cm beside the top arm's edge: a head-dip over the side.
        assert arena.outside_cm(42.0, 10.0)
        assert not arena.outside_cm(42.0, 10.0, margin_cm=3.0)
        # The middle of an empty quadrant stays off the maze under that margin.
        assert arena.outside_cm(15.0, 15.0, margin_cm=3.0)

    def test_centre_zone_is_where_the_arms_cross(self):
        arena = _epm()
        centre = arena.to_arena_cm(arena.centre_zone_vertices(10.0))
        assert np.array(centre) == pytest.approx(np.array([(35, 35), (40, 35), (40, 40), (35, 40)]))

    def test_regions_tile_the_maze(self):
        regions = _epm().plus_regions_cm()
        assert set(regions) == {"top", "right", "bottom", "left", "centre"}

        def area(poly):
            (x0, y0), _, (x1, y1), _ = poly
            return (x1 - x0) * (y1 - y0)

        total = sum(area(p) for p in regions.values())
        assert total == pytest.approx(2 * 75 * 5 - 5 * 5)  # two arms, crossing once

    def test_epm_zones_follow_which_arms_are_open(self):
        arena = _epm()
        zones = arena.epm_zone_vertices("horizontal")
        assert set(zones) == {"open_1", "open_2", "closed_1", "closed_2", "centre"}
        xs = [x for x, _ in arena.to_arena_cm(zones["open_1"])]
        assert max(xs) == pytest.approx(35.0)  # the left arm
        swapped = arena.epm_zone_vertices("vertical")
        ys = [y for _, y in arena.to_arena_cm(swapped["open_1"])]
        assert max(ys) == pytest.approx(35.0)  # the top arm
        with pytest.raises(ValueError):
            arena.epm_zone_vertices("diagonal")
        with pytest.raises(ValueError):
            _cal().epm_zone_vertices()

    def test_outline_is_the_twelve_point_cross(self):
        arena = _epm()
        assert len(arena.outline(800, 800)) == 12
        drawn = arena.draw(np.zeros((800, 800, 3), np.uint8))
        assert drawn[400, 25].any()  # the left arm's end edge
        assert not drawn[100, 100].any()  # an empty quadrant

    def test_arms_must_be_narrower_than_the_maze(self):
        with pytest.raises(ValueError):
            _epm(arm_width_cm=80.0)
        with pytest.raises(ValueError):
            _epm(arm_width_cm=0.0)

    def test_arm_ends_out_of_order_are_refused(self):
        ends = [_EPM_ENDS[0], _EPM_ENDS[2], _EPM_ENDS[1], _EPM_ENDS[3]]
        with pytest.raises(DegenerateArenaError):
            _epm(ends)

    def test_round_trips_with_its_arm_width(self):
        arena = _epm(arm_width_cm=7.5)
        data = arena.to_dict()
        assert data["arm_width_cm"] == 7.5
        again = ArenaCalibration.from_dict(data)
        assert again.shape == "plus" and again.arm_width_cm == 7.5
        assert np.array(again.corners) == pytest.approx(np.array(arena.corners))

    def test_other_shapes_still_save_without_an_arm_width(self):
        # Older GLIDERs read these files, and old files lack the key.
        data = _cal().to_dict()
        assert "arm_width_cm" not in data
        assert ArenaCalibration.from_dict(data).shape == "rectangle"
