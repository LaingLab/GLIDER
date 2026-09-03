"""Arena calibration: image <-> arena-centimetre geometry."""

from __future__ import annotations

import math

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
