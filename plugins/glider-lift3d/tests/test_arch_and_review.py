"""Back-arch eccentricity, session sidecars, and the Session Review panels."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from glider_lift3d import Lift3D, Lift3DConfig  # noqa: E402
from glider_lift3d.curvature import (  # noqa: E402
    back_arch,
    eccentricity,
    fit_ellipse,
    fit_ellipse_params,
    spine_profile,
)
from glider_lift3d.keypoints import KeypointRoles  # noqa: E402
from glider_lift3d.session import GRID, SidecarPaths, compute_session, load_session  # noqa: E402
from glider_lift3d.surface import CROP_CM, CROP_PX  # noqa: E402
from test_lift3d import (  # noqa: E402
    EXP16_KEYPOINTS,
    PPC,
    dark_segment,
    frame_with_mouse,
    mouse_exp16,
)

TEN = [*EXP16_KEYPOINTS, "neck", "withers", "lumbar"]


def mouse_10kp(**kw):
    xy = mouse_exp16(**kw)
    ears = (xy[0] + xy[1]) / 2
    neck = ears + 0.25 * (xy[3] - ears)
    withers = (neck + xy[3]) / 2
    lumbar = (xy[3] + xy[6]) / 2
    return np.vstack([xy, neck, withers, lumbar])


def bump_depth(crop_bgr):
    """Depth: the mouse is a dome, highest mid-body (as a real back would be)."""
    mouse = crop_bgr.mean(axis=2) < 100
    gu = np.mgrid[0:CROP_PX, 0:CROP_PX][1]
    dome = np.clip(1 - ((gu - CROP_PX / 2) / (CROP_PX / 4)) ** 2, 0, 1)
    return (1.0 - 0.05 * mouse * (0.3 + dome)).astype(np.float32)


# -- ellipse -----------------------------------------------------------------------------


@pytest.mark.parametrize(("a", "b"), [(4.0, 2.0), (4.0, 3.5), (5.0, 1.0)])
def test_fit_recovers_eccentricity_of_a_partial_arc(a, b):
    t = np.linspace(0.3, 2.8, 16)
    x, y = a * np.cos(t) + 3, b * np.sin(t) - 1
    assert eccentricity(*fit_ellipse(x, y)) == pytest.approx(eccentricity(a, b), abs=1e-6)


def test_fit_is_scale_invariant_and_returns_the_centre():
    t = np.linspace(0.3, 2.8, 16)
    x, y = 4 * np.cos(t) + 3, 2 * np.sin(t) - 1
    cx, cy, a, b, _ = fit_ellipse_params(x, y)
    assert (cx, cy, a, b) == pytest.approx((3, -1, 4, 2), abs=1e-6)
    assert eccentricity(*fit_ellipse(10 * x, 10 * y)) == pytest.approx(eccentricity(4, 2))


def test_a_flat_back_is_nearly_eccentricity_one():
    x = np.linspace(0, 7, 16)
    y = 2 + 0.01 * np.random.default_rng(0).normal(size=16)
    assert eccentricity(*fit_ellipse(x, y)) > 0.99


def test_rounder_back_scores_lower():
    """Mittal: a more hunched back gives a lower eccentricity."""
    s = np.linspace(0, 7, 16)
    flat = 0.8 * np.sin(np.pi * s / 7) + 1.0
    hunched = 3.0 * np.sin(np.pi * s / 7) + 1.0
    assert eccentricity(*fit_ellipse(s, hunched)) < eccentricity(*fit_ellipse(s, flat))


def test_spine_profile_reads_the_top_of_the_back():
    h = np.full((CROP_PX, CROP_PX), np.nan)
    h[150:170, 60:260] = np.linspace(0, 3, 200)[None, :]
    spine = np.array([[250.0, 160], [200, 160], [160, 160], [110, 160], [70, 160]])
    s, z = spine_profile(h, spine, CROP_PX / CROP_CM)
    assert len(s) == 16 and s[0] == 0 and s[-1] == pytest.approx(180 / 16, rel=1e-3)
    assert z[0] > z[-1]  # neck end is the high end here


# -- keypoint roles ----------------------------------------------------------------------


def test_spine_needs_a_neck():
    assert KeypointRoles.from_names(EXP16_KEYPOINTS).spine(mouse_exp16()) is None
    roles = KeypointRoles.from_names(TEN)
    assert roles.neck == 7 and roles.lumbar == 9
    assert roles.spine(mouse_10kp()).shape == (5, 2)


def test_back_arch_measures_both_profiles_with_a_neck_and_nothing_without():
    xy = mouse_10kp()
    frame = frame_with_mouse(xy)
    cfg = Lift3DConfig(ppc=PPC)
    r10 = Lift3D(TEN, cfg, segment=dark_segment, depth=bump_depth).process(frame, xy)
    arch = back_arch(r10, CROP_PX / CROP_CM)
    assert 0 < arch["ecc_surface"] <= 1 and 0 < arch["ecc_depth"] <= 1
    assert arch["spine_length_cm"] > 3
    r7 = Lift3D(EXP16_KEYPOINTS, cfg, segment=dark_segment, depth=bump_depth).process(frame, xy[:7])
    assert np.isnan(back_arch(r7, CROP_PX / CROP_CM)["ecc_surface"])


# -- session sidecars --------------------------------------------------------------------


def _session(tmp_path, n=5):
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    xy = np.repeat(mouse_10kp()[None], n, axis=0)
    xy[2, 0] = np.nan
    pose_csv = to_dlc_csv(
        PoseData(xy=xy, confidence=np.ones(xy.shape[:2]), keypoint_names=TEN),
        tmp_path / "s1DLC_train-17.csv",
    )
    video = tmp_path / "s1.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for _ in range(n):
        w.write(frame_with_mouse(mouse_10kp()))
    w.release()
    return video, pose_csv


def test_compute_session_writes_frame_aligned_sidecars(tmp_path):
    video, pose_csv = _session(tmp_path)
    cfg = Lift3DConfig(ppc=PPC)
    lift = Lift3D(TEN, cfg, segment=dark_segment, depth=bump_depth)
    paths = compute_session(video, pose_csv, cfg, lift=lift)
    assert paths == SidecarPaths.for_pose(pose_csv) and paths.exist()
    assert not list(tmp_path.glob("*.partial"))
    s = load_session(pose_csv)
    assert s.n_frames == 5 and s.surface.shape == (5, GRID, GRID)
    assert s.meta["has_neck"] and s.meta["used_depth"]
    assert np.isnan(s.table.loc[2, "ecc_surface"])  # no pose that frame
    assert np.isnan(np.asarray(s.surface[2], float)).all()
    assert 0 < s.table.loc[1, "ecc_surface"] <= 1
    assert np.isfinite(np.asarray(s.surface[1], float)).any()
    assert np.isfinite(s.spine[1]).all()


def test_a_stopped_run_leaves_nothing_behind(tmp_path):
    video, pose_csv = _session(tmp_path)
    cfg = Lift3DConfig(ppc=PPC)
    lift = Lift3D(TEN, cfg, segment=dark_segment, depth=bump_depth)
    calls = iter([False, False, True])
    assert compute_session(video, pose_csv, cfg, lift=lift, should_stop=lambda: next(calls)) is None
    assert not any(p.exists() for p in SidecarPaths.for_pose(pose_csv).all())
    assert not list(tmp_path.glob("*.partial"))
    assert load_session(pose_csv) is None


# -- review panels -----------------------------------------------------------------------


class _View:
    """The SessionView fields the panels read."""

    def __init__(self, pose_path, video_path):
        self.pose_path, self.video_path = pose_path, video_path
        self.px_per_mm = PPC / 10
        self.fps = 30.0
        self.resolution = (640, 480)


def test_panels_show_a_computed_session_and_seek_on_click(qtbot, tmp_path):
    from glider_lift3d.review import panels

    from glider.gui.review.extensions import PanelHost, ReviewContext, ReviewPanel

    video, pose_csv = _session(tmp_path)
    cfg = Lift3DConfig(ppc=PPC)
    compute_session(
        video, pose_csv, cfg, lift=Lift3D(TEN, cfg, segment=dark_segment, depth=bump_depth)
    )

    made = panels()
    assert [p.placement for p in made] == ["side", "timeline"]
    assert all(isinstance(p, ReviewPanel) for p in made)
    for p in made:
        qtbot.addWidget(p.widget())
    sought = []
    host = PanelHost(made)
    host.set_session(ReviewContext(_View(pose_csv, video), pose_csv, None, (0, 4), sought.append))
    host.set_frame(1)
    side, trace = made
    assert side.view.grid is not None and np.isfinite(side.view.grid).any()
    assert "arch e" in side.readout.text()
    assert not side.compute_btn.isVisible()
    assert trace.cursor.value() == 1
    assert len(trace.curves["ecc_surface"].getData()[0]) == 5
    side.view.resize(300, 240)
    side.view.grab()  # paints without error
    side.ctl.seek(3)
    assert sought == [3]


def test_panel_offers_to_compute_when_missing(qtbot, tmp_path):
    from glider_lift3d.review import panels

    from glider.gui.review.extensions import ReviewContext

    video, pose_csv = _session(tmp_path)
    side, _trace = panels()
    qtbot.addWidget(side.widget())
    side.widget().show()
    side.set_session(ReviewContext(_View(pose_csv, video), pose_csv, None, (0, 4), lambda f: None))
    assert side.compute_btn.isVisible() and side.compute_btn.isEnabled()
    assert "Not computed" in side.status.text()
    view = _View(pose_csv, video)
    view.px_per_mm = None
    side.set_session(ReviewContext(view, pose_csv, None, (0, 4), lambda f: None))
    assert not side.compute_btn.isEnabled() and "scale" in side.status.text()
