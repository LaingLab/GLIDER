"""3-D lift: keypoint roles, posture maths, pipeline with stand-in models, batch, live."""

from __future__ import annotations

import csv
import queue
import threading

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from glider.vision.lift3d import EXP16_KEYPOINTS, KeypointRoles, Lift3D, Lift3DConfig  # noqa: E402
from glider.vision.lift3d.posture import (  # noqa: E402
    REAR_NONE,
    KeypointHeight,
    body_surface,
    rear_weight,
)
from glider.vision.lift3d.surface import CROP_PX, floor_relative  # noqa: E402

EXP6 = ["nose", "l_ear", "r_ear", "l_body", "center", "r_body", "tail_base"]
PPC = 10.0


def mouse_exp16(cx=320.0, cy=240.0, length_cm=9.0):
    """A mouse facing +x in exp-16 order, nose-to-tail ``length_cm`` long."""
    span = length_cm * PPC
    nose = (cx + span / 2, cy)
    tail = (cx - span / 2, cy)
    return np.array(
        [
            (nose[0] - 0.2 * span, cy - 9),  # left ear
            (nose[0] - 0.2 * span, cy + 9),  # right ear
            nose,
            (cx, cy),
            (tail[0] + 0.15 * span, cy - 11),  # left hip
            (tail[0] + 0.15 * span, cy + 11),  # right hip
            tail,
        ],
        float,
    )


def to_exp6(xy16):
    by_name = dict(zip(EXP16_KEYPOINTS, xy16, strict=True))
    alias = {
        "nose": "nose",
        "l_ear": "left_ear",
        "r_ear": "right_ear",
        "l_body": "left_hip",
        "center": "body_center",
        "r_body": "right_hip",
        "tail_base": "tail_base",
    }
    return np.array([by_name[alias[n]] for n in EXP6])


def frame_with_mouse(xy16):
    f = np.full((480, 640, 3), 200, np.uint8)
    cv2.ellipse(
        f,
        (int(xy16[3, 0]), int(xy16[3, 1])),
        (int(np.linalg.norm(xy16[2] - xy16[6]) / 2), 18),
        0,
        0,
        360,
        (30, 30, 30),
        -1,
    )
    return f


def dark_segment(crop_bgr, _bc):
    return crop_bgr.mean(axis=2) < 100


def bump_depth(crop_bgr):
    """Depth: a tilted floor (as forward-camera models paint it), mouse 5% closer."""
    gv, gu = np.mgrid[0:CROP_PX, 0:CROP_PX]
    floor = 1.0 + 0.001 * gv
    return (floor - 0.05 * (crop_bgr.mean(axis=2) < 100)).astype(np.float32)


# -- keypoint roles ----------------------------------------------------------------------


def test_roles_map_exp6_and_exp16_to_the_same_points():
    xy16 = mouse_exp16()
    r16 = KeypointRoles.from_names(EXP16_KEYPOINTS)
    r6 = KeypointRoles.from_names(EXP6)
    np.testing.assert_allclose(r16.canonical(xy16), xy16)
    np.testing.assert_allclose(r6.canonical(to_exp6(xy16)), xy16)


def test_roles_name_what_is_missing():
    with pytest.raises(ValueError, match="tail_base"):
        KeypointRoles.from_names(["nose", "left_ear", "right_ear", "body_center", "lh", "rh", "x"])


# -- posture -----------------------------------------------------------------------------


def test_rear_weight_is_smooth_step():
    assert rear_weight(1.0) == 0.0
    assert rear_weight(REAR_NONE) == 0.0
    assert rear_weight(0.5) == 1.0


def test_sustained_short_body_raises_rear_weight_but_a_glitch_does_not():
    kh = KeypointHeight(fps=30, ref_length=9.0)
    full, short = mouse_exp16(length_cm=9.0), mouse_exp16(length_cm=4.5)
    for _ in range(30):
        kh(full, PPC)
    for _ in range(3):  # shorter than the hold
        out = kh(short, PPC)
    assert out["weight"] == 0.0
    for _ in range(30):
        out = kh(short, PPC)
    assert out["weight"] > 0.8
    assert out["rise_cm"] > 6.0


def test_body_surface_is_nan_off_the_mouse_and_rises_in_a_rear():
    mask = np.zeros((CROP_PX, CROP_PX), bool)
    cv2.ellipse(mask.view(np.uint8), (160, 160), (70, 25), 0, 0, 360, 1, -1)
    xy_c = mouse_exp16(160, 160, 14.0)  # already in crop px
    low = body_surface(mask, None, 0.0, xy_c, {"ratio": 1.0, "weight": 0.0, "rise_cm": 0.0})
    rear = body_surface(mask, None, 0.0, xy_c, {"ratio": 0.5, "weight": 1.0, "rise_cm": 7.0})
    assert np.isnan(low[0, 0]) and np.isfinite(low[160, 160])
    assert np.nanmax(low) < 4.0
    assert np.nanmax(rear) > 6.0
    # Highest toward the nose (+x), lowest toward the tail.
    assert rear[160, 220] > rear[160, 100]


def test_floor_plane_removes_the_forward_camera_tilt():
    mask = np.zeros((CROP_PX, CROP_PX), bool)
    mask[140:180, 120:200] = True
    gv, _ = np.mgrid[0:CROP_PX, 0:CROP_PX]
    dep = 1.0 + 0.002 * gv - 0.05 * mask
    rel, floor = floor_relative(dep, mask)
    assert abs(float(np.median(rel[floor]))) < 1e-6
    assert float(np.median(rel[mask])) > 0.03


# -- pipeline ----------------------------------------------------------------------------


def make_lift(names=EXP16_KEYPOINTS, depth=bump_depth, **kw):
    cfg = Lift3DConfig(ppc=PPC, fps=30.0, **kw)
    return Lift3D(names, cfg, segment=dark_segment, depth=depth)


def test_pipeline_gives_the_same_answer_for_either_keypoint_order():
    xy16 = mouse_exp16()
    frame = frame_with_mouse(xy16)
    a = make_lift().process(frame, xy16, frame_idx=0)
    b = make_lift(names=EXP6).process(frame, to_exp6(xy16), frame_idx=0)
    assert a is not None and b is not None
    np.testing.assert_allclose(a.anchors_cm, b.anchors_cm)
    assert a.used_mask_model and a.used_depth
    assert np.all(a.anchors_cm[1:4] > 1.0)  # hips, centre, ears sit on the body


def test_pipeline_skips_frames_with_missing_keypoints():
    xy = mouse_exp16()
    xy[2] = np.nan
    assert make_lift().process(frame_with_mouse(mouse_exp16()), xy) is None


def test_pipeline_runs_without_depth_or_mask_models():
    xy16 = mouse_exp16()
    cfg = Lift3DConfig(ppc=PPC, use_mask=False, use_depth=False)
    lift = Lift3D(EXP16_KEYPOINTS, cfg, segment=None, depth=None)
    r = lift.process(frame_with_mouse(xy16), xy16)
    assert r is not None and not r.used_mask_model and not r.used_depth
    assert r.depth_scale is None and np.all(np.isfinite(r.anchors_cm))


def test_async_depth_matches_sync_on_a_still_mouse():
    xy16 = mouse_exp16()
    frame = frame_with_mouse(xy16)
    sync, live = make_lift(), make_lift(async_depth=True)
    for i in range(8):
        a = sync.process(frame, xy16, frame_idx=i)
        b = live.process(frame, xy16, frame_idx=i)
    live.close()
    np.testing.assert_allclose(a.anchors_cm, b.anchors_cm, atol=1e-4)


def test_pipeline_needs_a_scale():
    with pytest.raises(ValueError, match="pixels per cm"):
        Lift3D(EXP16_KEYPOINTS, Lift3DConfig(ppc=0), segment=dark_segment, depth=bump_depth)


# -- batch -------------------------------------------------------------------------------


def test_run_video_writes_a_frame_aligned_csv(tmp_path):
    from glider.vision.lift3d.batch import CSV_COLUMNS, run_video
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    n = 6
    xy16 = mouse_exp16()
    xy = np.repeat(to_exp6(xy16)[None], n, axis=0)
    xy[3, 0] = np.nan  # an undetected frame
    pose_csv = to_dlc_csv(
        PoseData(xy=xy, confidence=np.ones(xy.shape[:2]), keypoint_names=EXP6),
        tmp_path / "poses.csv",
    )
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for _ in range(n):
        w.write(frame_with_mouse(xy16))
    w.release()

    lift = make_lift(names=EXP6)
    out = run_video(video, pose_csv, lift.config, lift=lift)
    rows = list(csv.reader(open(out)))
    assert rows[0] == CSV_COLUMNS
    assert [int(r[0]) for r in rows[1:]] == list(range(n))
    assert rows[4][2] == ""  # frame 3: no pose, empty row
    assert float(rows[2][CSV_COLUMNS.index("body_center_height")]) > 1.0


# -- live consumer -----------------------------------------------------------------------


def test_live_consumer_lifts_queued_frames_and_writes_csv(tmp_path):
    from glider.analysis.behavior.classify.lift3d_consumer import LatestLift, Lift3DConsumer

    xy16 = mouse_exp16()
    frame = frame_with_mouse(xy16)
    q: queue.Queue = queue.Queue()
    for i in range(5):
        q.put((i, frame.copy(), xy16, np.ones(7)))
    q.put(None)
    lift = make_lift()
    latest = LatestLift()
    t = Lift3DConsumer(
        q,
        threading.Event(),
        EXP16_KEYPOINTS,
        lift.config,
        csv_path=tmp_path / "lift.csv",
        latest=latest,
        lift=lift,
    )
    t.start()
    t.join(timeout=10)
    assert not t.is_alive() and t.error is None
    assert t.n_lifted == 5
    assert latest.get()[0] == 4
    assert len(list(csv.reader(open(tmp_path / "lift.csv")))) == 6


def test_fan_out_copies_the_frame_and_never_blocks():
    from glider.analysis.behavior.classify.threads import _fan_out_extra

    frame = np.zeros((4, 4, 3), np.uint8)
    q: queue.Queue = queue.Queue(maxsize=1)
    _fan_out_extra([q], (0, frame, None, None), threading.Event())
    _fan_out_extra([q], (1, frame, None, None), threading.Event())  # full: dropped, no wait
    idx, got, _, _ = q.get_nowait()
    assert idx == 0 and got is not frame
    frame[:] = 255  # the display drawing on the shared frame
    assert got.max() == 0
