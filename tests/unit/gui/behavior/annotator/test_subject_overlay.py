"""Drawing every animal, and stamping the subject onto created zones."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from glider.analysis.behavior.annotations import (  # noqa: E402
    AnnotationStore,
    BehaviorZone,
)
from glider.analysis.behavior.classify.overlay import draw_skeleton  # noqa: E402
from glider.analysis.behavior.vocabulary import Behavior  # noqa: E402
from glider.gui.behavior.annotator.main_window import AnnotatorWindow  # noqa: E402
from glider.gui.behavior.annotator.sampler import ProposedClip  # noqa: E402
from glider.vision.pose.core import PoseData  # noqa: E402


def _write_pose_csv(tmp_path, name: str, *, x0: float, n: int = 100):
    """A real per-animal DLC CSV, so the window loads it the way it will in
    production. Writing a file rather than injecting a PoseData is the point:
    it exercises the constructor parameter and the lazy loader together."""
    from glider.vision.pose.dlc import to_dlc_csv

    path = tmp_path / name
    to_dlc_csv(_pose(n, x0), path)
    return path


def _pose(n_frames: int, x0: float) -> PoseData:
    xy = np.zeros((n_frames, 2, 2), dtype=float)
    xy[:, 0, 0] = x0
    xy[:, 1, 0] = x0 + 10.0
    xy[:, :, 1] = 20.0
    return PoseData(
        xy=xy,
        confidence=np.ones((n_frames, 2)),
        keypoint_names=["a", "b"],
        fps=30.0,
    )


def _window(tmp_path, qtbot, *, individual=0, **kwargs):
    """kwargs go straight to AnnotatorWindow -- including pose_tracks, which
    must be keyed by the same video path the window builds below."""
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    video = tmp_path / "a.mp4"
    clips = [ProposedClip(0, 50, 40, 60, 0.7, str(video), individual)]
    w = AnnotatorWindow(clips=clips, videos_meta={video: ann}, **kwargs)
    qtbot.addWidget(w)
    return w, video


class _StubReader:
    def read(self, index):
        return np.zeros((8, 8, 3), dtype=np.uint8)


def test_draw_skeleton_takes_a_colour_and_defaults_to_todays():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    kp = np.array([[10.0, 10.0], [20.0, 20.0]])

    red = draw_skeleton(frame.copy(), kp, color=(0, 0, 255))
    assert red[10, 10, 2] > 0 and red[10, 10, 0] == 0

    # Default must be byte-identical to what every existing caller renders.
    default = draw_skeleton(frame.copy(), kp)
    green = draw_skeleton(frame.copy(), kp, color=(40, 200, 80), edge_color=(220, 180, 30))
    assert np.array_equal(default, green)


def test_the_overlay_hook_receives_the_index_of_the_frame_being_shown(tmp_path, qtbot):
    """A hook given the wrong index draws the previous frame's pose, which
    reads as tracking lag rather than as a bug."""
    w, _ = _window(tmp_path, qtbot)
    seen: list[int] = []

    def hook(frame, index):
        seen.append(int(index))
        return frame

    w.clip.frame_overlay = hook
    # Drive the display path directly with a known frame; set_clip needs a
    # real video file, and what is under test is the index, not decoding.
    w.clip._reader = _StubReader()
    w.clip._cap = object()
    w.clip._start_frame, w.clip._end_frame = 40, 60
    w.clip._seek_to_start()
    assert seen == [40]


def test_the_subject_is_drawn_differently_from_the_others(tmp_path, qtbot):
    """Identical rendering would leave the labeller unable to tell the
    visually identical animals apart -- which is the whole point."""
    # Drive the CONSTRUCTOR parameter, never a hand-assigned _tracks: a test
    # that seeds the cache directly passes even when nothing populates it.
    video = tmp_path / "a.mp4"
    paths = [
        _write_pose_csv(tmp_path, "animal0.csv", x0=5.0),
        _write_pose_csv(tmp_path, "animal1.csv", x0=60.0),
    ]
    w, video = _window(tmp_path, qtbot, individual=1, pose_tracks={video: paths})

    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    out = w._draw_pose(frame, 10)
    subject_px = out[20, 60].copy()  # animal 1, the subject
    other_px = out[20, 5].copy()  # animal 0
    assert subject_px.any() and other_px.any(), "both animals must be drawn"
    assert not np.array_equal(subject_px, other_px)


def test_the_overlay_never_costs_the_session(tmp_path, qtbot):
    """A malformed track loses the drawing, never the labelling."""
    video = tmp_path / "a.mp4"
    short = _write_pose_csv(tmp_path, "short.csv", x0=5.0, n=5)
    w, video = _window(tmp_path, qtbot, pose_tracks={video: [short]})
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    assert w._draw_pose(frame, 99) is frame


def test_a_zone_created_in_the_annotator_carries_the_clip_subject(tmp_path, qtbot):
    """Otherwise every zone is animal 0 and the individual column is a lie."""
    w, video = _window(tmp_path, qtbot, individual=1)
    w.vocab.add(Behavior(name="grooming", hotkey="g", color="#ff0000"))
    w._apply_label("grooming")  # main_window.py:684
    store = w.stores[video]
    assert [z.individual for z in store] == [1]


def test_re_trimming_a_zone_keeps_its_original_individual(tmp_path, qtbot):
    """A labeller who switches subject and then re-trims an earlier zone must
    not silently reassign that zone to the new animal."""
    w, video = _window(tmp_path, qtbot, individual=1)
    store = w.stores[video]
    prior = BehaviorZone("grooming", 40, 60, individual=0)
    store.add(prior)
    w._clip_zone[w.current] = prior

    w.trim_bar.set_bounds(45, 55)  # however this file's other tests set the trim
    w._persist_current_trim()  # main_window.py:734
    assert [z.individual for z in store] == [0]
