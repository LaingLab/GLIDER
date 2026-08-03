"""Wiring the speed trace into the annotator window and the clip player."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from glider.analysis.behavior.annotations import AnnotationStore  # noqa: E402
from glider.gui.behavior.annotator.main_window import AnnotatorWindow  # noqa: E402
from glider.gui.behavior.annotator.sampler import ProposedClip  # noqa: E402
from glider.gui.behavior.annotator.speed_source import SessionSpeed  # noqa: E402


def _pose_csv(tmp_path, name, n_frames=200):
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    xy = np.zeros((n_frames, 2, 2))
    xy[:, :, 0] = (np.arange(n_frames) * 3.0)[:, None]
    xy[:, :, 1] = (np.arange(n_frames) * 4.0)[:, None]
    pose = PoseData(
        xy=xy,
        confidence=np.ones((n_frames, 2)),
        keypoint_names=["a", "b"],
        fps=30.0,
    )
    path = tmp_path / name
    to_dlc_csv(pose, path)
    return path


def _window(tmp_path, qtbot, **kwargs):
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    video = tmp_path / "a.mp4"
    clips = [ProposedClip(0, 50, 40, 60, 0.7, str(video))]
    w = AnnotatorWindow(clips=clips, videos_meta={video: ann}, **kwargs)
    qtbot.addWidget(w)
    return w, video


# ---------------------------------------------------------------------------
# Backwards compatibility: the trace is entirely optional
# ---------------------------------------------------------------------------


def test_window_without_pose_data_behaves_as_before(tmp_path, qtbot):
    """Every existing caller omits the new arguments and must be unaffected."""
    w, _ = _window(tmp_path, qtbot)
    # The feature flag, not isVisible() -- an unshown widget is invisible
    # either way, so asserting on that would pass even if the trace were on.
    assert w._speed_enabled is False
    assert w.speed_cache.state(w._current_video_path()) == "absent"
    # And no work is started for a session that has no pose data.
    w._ensure_speed_loaded(w._current_video_path())
    assert w.speed_cache.state(w._current_video_path()) == "absent"


def test_pose_csvs_default_to_none_supplied(tmp_path, qtbot):
    w, _ = _window(tmp_path, qtbot)
    assert w.pose_csvs == {}


def test_pose_csvs_are_kept_by_video(tmp_path, qtbot):
    pose = _pose_csv(tmp_path, "a.csv")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
    )
    qtbot.addWidget(w)
    assert w.pose_csvs[video] == pose


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_loading_a_session_populates_the_cache_and_the_trace(tmp_path, qtbot):
    """The load is the same computation cohort thresholds are derived from."""
    from glider.analysis.behavior.classify.speed_state import causal_speed_series
    from glider.vision.pose.dlc import from_dlc_csv

    pose = _pose_csv(tmp_path, "a.csv")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
    )
    qtbot.addWidget(w)

    w._load_speed_now(video)  # synchronous path the worker calls into

    assert w.speed_cache.state(video) == "ready"
    expected = causal_speed_series(from_dlc_csv(pose).xy)
    np.testing.assert_allclose(w.speed_cache.get(video).px_per_frame, expected)


def test_an_unreadable_pose_csv_is_recorded_not_raised(tmp_path, qtbot):
    bad = tmp_path / "a.csv"
    bad.write_text("not a pose csv at all\n", encoding="utf-8")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: bad},
    )
    qtbot.addWidget(w)

    w._load_speed_now(video)

    assert w.speed_cache.state(video) == "failed"
    assert w.speed_cache.error(video)


def test_a_failed_load_is_not_retried_on_every_clip(tmp_path, qtbot):
    bad = tmp_path / "a.csv"
    bad.write_text("nope\n", encoding="utf-8")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: bad},
    )
    qtbot.addWidget(w)
    w._load_speed_now(video)
    assert w.speed_cache.begin(video) is False


def test_closing_with_a_load_in_flight_stops_the_thread(tmp_path, qtbot):
    """Qt aborts the process if a running QThread is destroyed.

    The window starts a parse on the first clip of every video, so closing
    the annotator a moment after it opens is the ordinary case, not an edge
    one -- and it took the whole test suite down before closeEvent waited.
    """
    pose = _pose_csv(tmp_path, "a.csv", n_frames=400)
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
    )
    qtbot.addWidget(w)

    w._ensure_speed_loaded(video)  # starts a real worker thread
    w.close()

    assert w._speed_threads == []


def test_a_video_with_no_pose_csv_starts_no_worker(tmp_path, qtbot):
    """Mixed sessions are normal; a missing CSV must not queue a doomed parse."""
    pose = _pose_csv(tmp_path, "a.csv")
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    ann_a, ann_b = tmp_path / "a_annotations.csv", tmp_path / "b_annotations.csv"
    AnnotationStore().save_csv(ann_a)
    AnnotationStore().save_csv(ann_b)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(b))],
        videos_meta={a: ann_a, b: ann_b},
        pose_csvs={a: pose},  # nothing for b
    )
    qtbot.addWidget(w)

    w._ensure_speed_loaded(b)

    assert w.speed_cache.state(b) == "absent"
    w.close()


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_cohort_thresholds_reach_the_trace(tmp_path, qtbot):
    from glider.analysis.behavior.cohort_speed import CohortSpeedThresholds

    cohort = CohortSpeedThresholds(
        freeze=0.5,
        dart=27.7,
        unit="cm/s",
        freeze_pct=10.0,
        dart_pct=99.5,
        n_sessions=1,
        n_samples=10,
    )
    pose = _pose_csv(tmp_path, "a.csv")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
        cohort=cohort,
        px_per_mm={video: 2.0},
    )
    qtbot.addWidget(w)
    w._load_speed_now(video)
    w._refresh_speed_trace()

    assert w.speed_trace.thresholds() == (0.5, 27.7)


def test_thresholds_are_withheld_without_a_calibration(tmp_path, qtbot):
    """cm/s lines over a px/frame trace would be convincing and wrong."""
    from glider.analysis.behavior.cohort_speed import CohortSpeedThresholds

    cohort = CohortSpeedThresholds(
        freeze=0.5,
        dart=27.7,
        unit="cm/s",
        freeze_pct=10.0,
        dart_pct=99.5,
        n_sessions=1,
        n_samples=10,
    )
    pose = _pose_csv(tmp_path, "a.csv")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
        cohort=cohort,  # no px_per_mm
    )
    qtbot.addWidget(w)
    w._load_speed_now(video)
    w._refresh_speed_trace()

    assert w.speed_trace.thresholds() == (None, None)
    assert "calibration" in w.speed_trace.status_text().lower()


# ---------------------------------------------------------------------------
# The playhead follows the clip player
# ---------------------------------------------------------------------------


def test_clip_player_reports_the_frame_it_displayed(qtbot, tmp_path, monkeypatch):
    """The playhead is only honest if it is the frame actually on screen."""
    from glider.gui.behavior.annotator import capture_cache as cc_mod
    from glider.gui.behavior.annotator.capture_cache import VideoCaptureCache
    from glider.gui.behavior.annotator.clip_player import ClipPlayer

    class FakeCap:
        def __init__(self, path):
            self.path = path

        def isOpened(self):
            return True

        def set(self, *a, **k):
            return True

        def read(self):
            return True, np.zeros((4, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cc_mod, "_open_capture", lambda p: FakeCap(p))
    video = tmp_path / "v.mp4"
    video.write_bytes(b"")

    player = ClipPlayer(capture_cache=VideoCaptureCache(max_open=2))
    qtbot.addWidget(player)

    seen: list[int] = []
    player.frame_changed.connect(seen.append)

    player.set_clip(video, 40, 60, fps=30.0)
    assert seen == [40]  # the first frame shown is the clip's start

    player._on_tick()
    assert seen == [40, 41]


def test_window_moves_the_playhead_when_the_player_reports_a_frame(tmp_path, qtbot):
    pose = _pose_csv(tmp_path, "a.csv")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
    )
    qtbot.addWidget(w)
    w._load_speed_now(video)
    w._refresh_speed_trace()

    w.clip.frame_changed.emit(47)
    assert w.speed_trace._playhead == 47


def test_trace_is_given_the_same_window_as_the_trim_bar(tmp_path, qtbot):
    pose = _pose_csv(tmp_path, "a.csv")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
    )
    qtbot.addWidget(w)
    w._load_speed_now(video)
    w._refresh_speed_trace()

    assert (w.speed_trace._win_start, w.speed_trace._win_end) == (
        w.trim_bar._win_start,
        w.trim_bar._win_end,
    )


def test_session_speed_is_built_with_the_videos_scale(tmp_path, qtbot):
    pose = _pose_csv(tmp_path, "a.csv")
    video = tmp_path / "a.mp4"
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    w = AnnotatorWindow(
        clips=[ProposedClip(0, 50, 40, 60, 0.7, str(video))],
        videos_meta={video: ann},
        pose_csvs={video: pose},
        px_per_mm={video: 2.0},
    )
    qtbot.addWidget(w)
    w._load_speed_now(video)

    session: SessionSpeed = w.speed_cache.get(video)
    assert session.is_calibrated is True
    assert session.unit == "cm/s"
