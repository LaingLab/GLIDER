"""Distance to the nest, and the retrieval events built on it.

Synthetic geometry and synthetic pups with answers known by construction: a
square nest, a pup carried home at a chosen second, a pup dropped on the way,
a pup that starts in the nest, and a dam who only sniffs.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.nest import nest_field
from glider.analysis.behavior.retrieval import RetrievalSettings, litter_summary, score_pup
from glider.vision.zones import Zone, ZoneShape

W, H, FPS = 200, 100, 30.0


def _square():
    """Nest covering pixels x 20..60, y 20..60."""
    return Zone("n", "nest", ZoneShape.RECTANGLE, [(20 / W, 20 / H), (60 / W, 60 / H)])


# --- nest.py -----------------------------------------------------------------


def test_signed_distance_outside_inside_and_missing():
    f = nest_field(_square(), W, H)
    d = f.at(np.array([[100.0, 40.0], [40.0, 40.0], [np.nan, 1.0], [500.0, 40.0]]))
    assert d[0] == pytest.approx(40, abs=1.5)  # 40 px right of the edge
    assert d[1] == pytest.approx(-20, abs=1.5)  # centre, 20 px deep
    assert np.isnan(d[2])  # a dropped frame is not a distance
    assert d[3] == pytest.approx(139, abs=1.5)  # clamped to the frame edge
    assert f.centre == pytest.approx((40, 40), abs=0.6)
    assert f.from_centre(np.array([100.0, 40.0])) == pytest.approx(60, abs=0.6)


def test_polygon_takes_its_nearest_pixel_not_its_vertices():
    f = nest_field(_square(), W, H)
    # A pup straddling the nest corner: every vertex is outside the nest.
    diamond = np.array([[62, 40], [70, 32], [78, 40], [70, 48]], dtype=float)
    assert f.of_polygon(diamond) == pytest.approx(2, abs=1.5)
    overlap = np.array([[50, 30], [70, 30], [70, 50], [50, 50]], dtype=float)
    assert f.of_polygon(overlap) < 0  # part of the pup is inside
    assert np.isnan(f.of_polygon(np.array([[1.0, 1.0], [2.0, 2.0]])))


def test_circle_and_empty_zone():
    circle = Zone("c", "nest", ZoneShape.CIRCLE, [(100 / W, 50 / H), (120 / W, 50 / H)])
    f = nest_field(circle, W, H)
    assert f.at(np.array([150.0, 50.0])) == pytest.approx(30, abs=1.5)
    empty = Zone("e", "nest", ZoneShape.POLYGON, [])
    with pytest.raises(ValueError, match="covers no pixels"):
        nest_field(empty, W, H)


# --- retrieval.py --------------------------------------------------------------


def _trial(seconds=20.0):
    """Frame count and a pup sitting 10 cm from a nest edge at x=0."""
    n = int(seconds * FPS)
    xy = np.tile([10.0, 0.0], (n, 1))
    nose = np.full(n, 20.0)
    return n, xy, nose


def _signals(xy):
    """The nest is everything at x <= 0, so signed distance is just x."""
    return xy[:, 0].copy()


def _carry(xy, nose, start_s, end_s, to_x):
    """The dam picks the pup up at start_s and walks it to x=to_x by end_s."""
    a, b = int(start_s * FPS), int(end_s * FPS)
    x0 = xy[a, 0]
    xy[a:b, 0] = np.linspace(x0, to_x, b - a)
    xy[b:, 0] = to_x
    nose[a:b] = 0.0


def test_carried_home_latency_and_contact():
    n, xy, nose = _trial()
    nose[int(4 * FPS) : int(5 * FPS)] = 0.5  # sniffs first, pup does not move
    _carry(xy, nose, 5.0, 7.0, -3.0)  # 13 cm in 2 s
    r = score_pup(_signals(xy), nose, xy, fps=FPS, t0=0)
    assert not r.start_in_nest
    assert r.contact_latency_s == pytest.approx(4.0, abs=0.05)
    # Crosses the edge partway through the carry, then stays.
    cross = np.flatnonzero(xy[:, 0] <= 0)[0] / FPS
    assert r.retrieval_latency_s == pytest.approx(cross, abs=0.05)
    assert (r.carries, r.drops) == (1, 0)
    assert r.carry_distance_cm == pytest.approx(13, abs=0.5)


def test_latency_is_relative_to_trial_start():
    n, xy, nose = _trial()
    _carry(xy, nose, 5.0, 7.0, -3.0)
    a = score_pup(_signals(xy), nose, xy, fps=FPS, t0=0)
    b = score_pup(_signals(xy), nose, xy, fps=FPS, t0=int(2 * FPS))
    assert b.retrieval_latency_s == pytest.approx(a.retrieval_latency_s - 2.0, abs=0.05)


def test_dropped_then_retrieved():
    n, xy, nose = _trial(30)
    _carry(xy, nose, 3.0, 4.0, 5.0)  # put down halfway
    _carry(xy, nose, 10.0, 12.0, -2.0)
    r = score_pup(_signals(xy), nose, xy, fps=FPS, t0=0)
    assert (r.carries, r.drops) == (2, 1)
    assert 10 < r.retrieval_latency_s < 12


def test_sniffing_alone_is_not_a_carry():
    n, xy, nose = _trial()
    nose[60:200] = 0.2
    xy[60:200, 0] += np.tile([0.0, 0.5, 0.0, -0.5], 35)  # jostled: fast, going nowhere
    r = score_pup(_signals(xy), nose, xy, fps=FPS, t0=0)
    assert r.carries == 0
    assert np.isnan(r.retrieval_latency_s)
    # The speed rule alone would have called it a carry; displacement stops it.
    loose = RetrievalSettings(min_carry_cm=0.0)
    assert score_pup(_signals(xy), nose, xy, fps=FPS, t0=0, settings=loose).carries > 0


def test_hidden_in_mouth_counts_as_carried_and_brief_visits_do_not_count():
    n, xy, nose = _trial()
    a = int(5 * FPS)
    nose[a - 5 : a] = 0.3
    xy[a : a + 40] = np.nan  # vanished under her head
    xy[a + 40 :, 0] = -2.0  # reappears in the nest
    r = score_pup(_signals(xy), nose, xy, fps=FPS, t0=0)
    assert r.carries == 1
    assert r.carried[a : a + 40].all()

    # Dragged across the edge for 1 s, then back out: not retrieved.
    n, xy, nose = _trial()
    xy[100:130, 0] = -1.0
    r = score_pup(_signals(xy), nose, xy, fps=FPS, t0=0)
    assert np.isnan(r.retrieval_latency_s)


def test_pup_starting_in_nest_is_excluded_and_litter_summary():
    n, xy, nose = _trial()
    home = xy.copy()
    home[:, 0] = -4.0
    fast, slow = (_trial()[1:] for _ in range(2))
    _carry(*fast, 3.0, 5.0, -3.0)
    _carry(*slow, 8.0, 10.0, -3.0)
    pups = {
        "pup_0": score_pup(_signals(home), nose, home, fps=FPS, t0=0),
        "pup_1": score_pup(_signals(slow[0]), slow[1], slow[0], fps=FPS, t0=0),
        "pup_2": score_pup(_signals(fast[0]), fast[1], fast[0], fps=FPS, t0=0),
    }
    assert pups["pup_0"].start_in_nest
    assert np.isnan(pups["pup_0"].retrieval_latency_s)
    s = litter_summary(pups)
    assert s["pups_scattered"] == 2 and s["pups_retrieved"] == 2
    assert s["retrieval_order"] == {"pup_2": 1, "pup_1": 2}
    assert s["time_all_retrieved_s"] == pytest.approx(pups["pup_1"].retrieval_latency_s)

    lost = score_pup(_signals(_trial()[1]), nose, _trial()[1], fps=FPS, t0=0)
    s = litter_summary({**pups, "pup_3": lost})
    assert np.isnan(s["time_all_retrieved_s"])


def test_rejects_mismatched_inputs_and_bad_t0():
    n, xy, nose = _trial()
    with pytest.raises(ValueError, match="same frames"):
        score_pup(_signals(xy), nose[:-1], xy, fps=FPS, t0=0)
    with pytest.raises(ValueError, match="outside"):
        score_pup(_signals(xy), nose, xy, fps=FPS, t0=n)


def test_thresholds_are_settings():
    n, xy, nose = _trial()
    _carry(xy, nose, 5.0, 7.0, -3.0)
    strict = RetrievalSettings(in_nest_s=30.0)  # longer than the trial remains
    assert np.isnan(
        score_pup(_signals(xy), nose, xy, fps=FPS, t0=0, settings=strict).retrieval_latency_s
    )


def test_pup_followed_at_the_nose_into_the_nest_when_never_seen_again():
    """The usual pattern on real videos: into her mouth, out buried in the nest, never seen."""
    n, xy, nose = _trial(30)
    a = int(5 * FPS)
    nose[a - 5 : a] = 0.3
    xy[a:] = np.nan  # never seen again
    nose_xy = np.tile([10.0, 0.0], (n, 1))
    nose_xy[a : a + 30, 0] = np.linspace(10, -1, 30)  # walks home in 2 s
    nose_xy[a + 30 :, 0] = 8.0  # then leaves the nest without the pup
    entry = a + int(np.flatnonzero(nose_xy[a : a + 30, 0] <= 0)[0])

    r = score_pup(
        _signals(xy), nose, xy, fps=FPS, t0=0, nose_xy_cm=nose_xy, nose_nest_cm=nose_xy[:, 0]
    )
    assert r.retrieval_latency_s == pytest.approx(entry / FPS, abs=0.05)
    assert (r.carries, r.drops) == (1, 0)
    assert not r.carried[entry + 1 :].any()  # her walking away is not a carry
    assert r.in_nest[-1]  # left behind, buried

    # Without her nose the carry is still seen, but capped, and never ends home.
    blind = score_pup(_signals(xy), nose, xy, fps=FPS, t0=0)
    assert np.isnan(blind.retrieval_latency_s)
    assert blind.carried.sum() <= RetrievalSettings().hidden_carry_max_s * FPS


def test_nearest_edge_point_of_the_nest():
    f = nest_field(_square(), W, H)
    assert f.edge.shape[1] == 2 and len(f.edge) > 0
    np.testing.assert_allclose(f.nearest_edge(np.array([100.0, 40.0])), (60, 40), atol=1.5)
