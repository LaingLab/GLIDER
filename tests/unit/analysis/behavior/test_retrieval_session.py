"""A whole retrieval session scored from synthetic pups, nose and setup.

The frame is 200x100 px at 10 px/cm (px_per_mm = 1.0), 10 fps. The nest is
everything left of x = 40 px. Answers are known by construction.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from glider.analysis.behavior.retrieval_session import (
    STATES,
    NotScorable,
    litter_row,
    nest_distance_frame,
    nose_to_pup_px,
    pup_rows,
    range_values,
    score_session,
)
from glider.analysis.behavior.retrieval_setup import RetrievalProtocol, RetrievalSetup
from glider.vision.pose.pups import PupTracks
from glider.vision.zones import Zone, ZoneShape

W, H, FPS, F = 200, 100, 10.0, 300


def _nest(x1=40):
    return Zone("n", "nest", ZoneShape.RECTANGLE, [(0.0, 0.0), (x1 / W, 1.0)])


def _box(x, y, r=4):
    return np.array([[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]], np.int32)


def _pups(xy):
    """PupTracks with a square outline around every seen position."""
    outlines = [
        [_box(*xy[k, f]) for k in range(len(xy)) if np.isfinite(xy[k, f]).all()]
        for f in range(xy.shape[1])
    ]
    return PupTracks(xy, outlines, np.zeros(xy.shape[:2], bool), {"fps": FPS})


def _carried_home():
    """Pup 0 sits at x=150 until the dam carries it home over 5-7 s; pup 1 is never seen."""
    xy = np.full((2, F, 2), np.nan)
    xy[0, :, :] = (150.0, 50.0)
    a, b = int(5 * FPS), int(7 * FPS)
    xy[0, a:b, 0] = np.linspace(150, 20, b - a)
    xy[0, b:, 0] = 20.0
    nose = np.tile([180.0, 50.0], (F, 1))
    nose[a - 3 : b, 0] = np.concatenate([[150.0] * 3, np.linspace(150, 20, b - a)])
    nose[b:, 0] = 100.0
    return xy, nose


def _setup(**kw):
    s = RetrievalSetup(trial_start_s=0.0, px_per_mm=1.0)
    s.set_nest(0.0, _nest())
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _score(xy, nose, setup=None, protocol=None):
    return score_session(
        _pups(xy),
        nose,
        fps=FPS,
        size=(W, H),
        setup=setup or _setup(),
        protocol=protocol or RetrievalProtocol(),
    )


def test_a_pup_carried_home_is_retrieved_and_an_unseen_pup_is_not():
    r = _score(*_carried_home())
    p0, p1 = r.pups["pup_0"], r.pups["pup_1"]
    assert 5.0 < p0.retrieval_latency_s < 7.5
    assert p0.carries == 1
    assert np.isnan(p1.retrieval_latency_s)  # never seen: not retrieved, no crash
    assert r.litter["pups_retrieved"] == 1


def test_state_codes_cover_the_scored_window():
    r = _score(*_carried_home())
    codes = r.state_codes(0)
    assert codes.shape == (F,)
    assert STATES[codes[10]] == "out"
    assert STATES[codes[-1]] == "in nest"
    assert "carried" in {STATES[c] for c in codes if c >= 0}


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        (RetrievalSetup(trial_start_s=0.0, px_per_mm=1.0), "nest"),
        (_setup(trial_start_s=None), "trial start"),
        (_setup(px_per_mm=None), "scale"),
        (_setup(trial_start_s=F / FPS + 1), "after the end"),
    ],
)
def test_not_scorable_says_why(setup, message):
    xy, nose = _carried_home()
    with pytest.raises(NotScorable, match=message):
        _score(xy, nose, setup=setup)


def test_no_nose_keypoint_is_not_scorable():
    xy, _ = _carried_home()
    with pytest.raises(NotScorable, match="'nose'.*Thresholds…"):
        score_session(
            _pups(xy), None, fps=FPS, size=(W, H), setup=_setup(), protocol=RetrievalProtocol()
        )


def test_cutoff_before_the_carry_means_not_retrieved():
    r = _score(*_carried_home(), protocol=RetrievalProtocol(cutoff_s=3.0))
    assert np.isnan(r.pups["pup_0"].retrieval_latency_s)
    assert r.end == int(3.0 * FPS)
    assert (r.state_codes(0)[r.end :] == -1).all()


def test_a_cutoff_past_the_video_is_clipped():
    r = _score(*_carried_home(), protocol=RetrievalProtocol(cutoff_s=10_000))
    assert r.end == F


def test_the_nest_period_in_force_is_used():
    xy, nose = _carried_home()
    setup = _setup()
    setup.set_nest(2.0, _nest(x1=10))  # from 2 s the nest shrinks to x < 10
    r = _score(xy, nose, setup=setup)
    assert np.isnan(r.pups["pup_0"].retrieval_latency_s)  # x=20 is outside the new nest
    assert r.period[0] == 0 and r.period[-1] == 1


def test_a_short_nose_track_is_padded_not_crashed():
    xy, nose = _carried_home()
    r = _score(xy, nose[: F // 2])
    assert r.nose_xy_px.shape == (F, 2)
    assert np.isnan(r.nose_xy_px[-1]).all()


def test_tables_and_range_values():
    r = _score(*_carried_home())
    setup, protocol = _setup(), RetrievalProtocol()
    rows = pup_rows(r, session_id="s1", group="g", setup=setup, protocol=protocol)
    assert [row["pup"] for row in rows] == ["pup_0", "pup_1"]
    assert rows[0]["retrieved"] is True and rows[1]["retrieved"] is False
    assert rows[0]["retrieval_order"] == 1 and np.isnan(rows[1]["retrieval_order"])
    assert rows[0]["cutoff_s"] == F / FPS and rows[0]["px_per_mm"] == 1.0  # the video ends first
    litter = litter_row(r, session_id="s1", group="g", setup=setup, protocol=protocol)
    assert litter["pups_retrieved"] == 1 and litter["retrieval_order"] == "pup_0"
    frame = nest_distance_frame(r)
    assert set(frame["animal"]) == {"adult_nose", "pup_0", "pup_1"}
    assert len(frame) == 3 * F
    assert list(frame.columns) == [
        "frame",
        "time_s",
        "animal",
        "x_px",
        "y_px",
        "dist_edge_cm",
        "in_nest",
        "nose_dist_cm",
        "carried",
        "hidden",
    ]
    whole = range_values(r, 0, F)
    assert whole["pups_retrieved"] == 1
    assert 0.0 <= whole["dam_in_nest_frac"] <= 1.0
    assert np.isfinite(whole["nose_nest_mean_cm"])
    assert range_values(r, 0, int(4 * FPS))["pups_retrieved"] == 0


@pytest.mark.parametrize(
    ("trial_start_s", "cutoff_s", "exported"),
    [
        (0.0, 300.0, 30.0),  # a 30 s video ends long before the cutoff
        (10.0, 300.0, 20.0),  # and was watched 20 s after a later start
        (0.0, 3.0, 3.0),  # a video that runs past the cutoff: the protocol's
    ],
)
def test_the_exported_cutoff_is_where_scoring_stopped(trial_start_s, cutoff_s, exported):
    """Censoring an unretrieved pup at 300 s when it was watched for 30 would be wrong."""
    setup, protocol = _setup(trial_start_s=trial_start_s), RetrievalProtocol(cutoff_s=cutoff_s)
    r = _score(*_carried_home(), setup=setup, protocol=protocol)
    kw = {"session_id": "s1", "group": "g", "setup": setup, "protocol": protocol}
    assert {row["cutoff_s"] for row in pup_rows(r, **kw)} == {exported}
    assert litter_row(r, **kw)["cutoff_s"] == exported


def test_a_cutoff_that_rounds_to_a_frame_still_exports_as_set():
    """20 s at 29.97 fps is 599.4 frames: scored to 599, still the 20 s cutoff, not 19.99."""
    fps, n = 30000 / 1001, 700
    setup, protocol = _setup(), RetrievalProtocol(cutoff_s=20.0)
    r = score_session(
        _pups(np.full((1, n, 2), 150.0)),
        np.full((n, 2), 180.0),
        fps=fps,
        size=(W, H),
        setup=setup,
        protocol=protocol,
    )
    assert r.end == 599
    assert (
        litter_row(r, session_id="s", group="", setup=setup, protocol=protocol)["cutoff_s"] == 20.0
    )


def _messy_pups(seed=0, n_pups=4, n_frames=40):
    """Pups that vanish, outlines that merge, frames with none, a nose that drops out."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(20, 180, (n_pups, n_frames, 2))
    xy[:, :, 1] = rng.uniform(10, 90, (n_pups, n_frames))
    xy[rng.random((n_pups, n_frames)) < 0.2] = np.nan
    outlines = []
    for f in range(n_frames):
        seen = [k for k in range(n_pups) if np.isfinite(xy[k, f]).all()]
        boxes = [_box(*np.rint(xy[k, f]), r=int(rng.integers(3, 9))) for k in seen]
        if f % 7 == 3:
            boxes = []  # the segmentation saw nothing
        elif boxes and f % 5 == 0:
            boxes.append(_box(100, 50, r=20))  # a blob tracked to no pup
        outlines.append(boxes)
    nose = rng.uniform(10, 190, (n_frames, 2))
    nose[::6] = np.nan
    nose[1::9] = xy[0, 1::9]  # on a pup: 0
    return PupTracks(xy, outlines, np.zeros((n_pups, n_frames), bool), {}), nose


def _nose_to_pup_px_per_pup(pups, nose):
    """The original pass: PupTracks.outline_for for every pup in every frame."""
    out = np.full((pups.n_pups, pups.n_frames), np.nan)
    for f in np.flatnonzero(np.isfinite(nose).all(axis=1)):
        point = (float(nose[f, 0]), float(nose[f, 1]))
        for k in range(pups.n_pups):
            outline = pups.outline_for(k, f)
            if outline is not None:
                contour = outline.reshape(-1, 1, 2).astype(np.float32)
                out[k, f] = max(-cv2.pointPolygonTest(contour, point, True), 0.0)
    return out


@pytest.mark.parametrize("seed", range(5))
def test_nose_to_pup_matches_the_per_pup_pass(seed):
    pups, nose = _messy_pups(seed)
    expected = _nose_to_pup_px_per_pup(pups, nose)
    assert np.isfinite(expected).any() and np.isnan(expected).any()
    np.testing.assert_array_equal(nose_to_pup_px(pups, nose), expected)


def test_each_outline_centre_is_found_once_per_frame(monkeypatch):
    """Not once per pup: that was 0.7 s of a 1 s pass on a 15-minute video."""
    calls = []

    class Counted(np.ndarray):
        def mean(self, *args, **kwargs):
            calls.append(1)
            return np.asarray(self).mean(*args, **kwargs)

    pups, nose = _messy_pups()
    counted = [[o.view(Counted) for o in frame] for frame in pups.outlines]
    pups = PupTracks(pups.xy, counted, pups.stitched, {})
    nose_to_pup_px(pups, nose)
    seen = np.isfinite(nose).all(axis=1)
    has_pup = np.isfinite(pups.xy).all(axis=2).any(axis=0)
    assert len(calls) <= sum(len(o) for o, f in zip(counted, seen & has_pup, strict=True) if f)
