"""The annotator's speed data: per-frame trace, units, and the load cache.

Pure -- no Qt, no cv2 -- so it runs without an event loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.gui.behavior.annotator.speed_source import (
    CM_PER_S,
    PX_PER_FRAME,
    SessionSpeed,
    SpeedCache,
    load_session_speed,
)


def _session(values, fps=30.0, px_per_mm=None):
    return SessionSpeed(px_per_frame=np.asarray(values, dtype=float), fps=fps, px_per_mm=px_per_mm)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def test_uncalibrated_session_reports_pixels():
    s = _session([0.0, 1.0, 2.0])
    assert s.unit == PX_PER_FRAME
    assert s.is_calibrated is False


def test_calibrated_session_reports_cm_per_second():
    s = _session([0.0, 1.0], fps=30.0, px_per_mm=2.0)
    assert s.unit == CM_PER_S
    assert s.is_calibrated is True


def test_calibrated_values_convert_to_cm_per_second():
    """px/frame -> px/s -> mm/s -> cm/s, the same arithmetic cohort_speed uses."""
    s = _session([1.0], fps=30.0, px_per_mm=2.0)
    # 1 px/frame * 30 fps / 2 px/mm = 15 mm/s = 1.5 cm/s
    assert s.at(0) == pytest.approx(1.5)


def test_uncalibrated_values_pass_through_unconverted():
    s = _session([4.0])
    assert s.at(0) == pytest.approx(4.0)


@pytest.mark.parametrize("px_per_mm", [0.0, -1.0, None])
def test_a_useless_scale_is_not_a_calibration(px_per_mm):
    """A zero or negative scale would divide the trace into nonsense."""
    assert _session([1.0], px_per_mm=px_per_mm).is_calibrated is False


def test_a_useless_frame_rate_is_not_a_calibration():
    assert _session([1.0], fps=0.0, px_per_mm=2.0).is_calibrated is False


# ---------------------------------------------------------------------------
# Indexing by frame
# ---------------------------------------------------------------------------


def test_at_returns_nan_outside_the_recording():
    """The trim window is padded past the clip and may run off either end."""
    s = _session([1.0, 2.0, 3.0])
    assert np.isnan(s.at(-1))
    assert np.isnan(s.at(3))
    assert s.at(2) == pytest.approx(3.0)


def test_window_returns_one_value_per_requested_frame():
    s = _session([0.0, 1.0, 2.0, 3.0, 4.0])
    assert s.window(1, 4).tolist() == [1.0, 2.0, 3.0]


def test_window_pads_with_nan_past_the_ends():
    """A window overhanging the recording keeps its length, so x-mapping holds."""
    s = _session([5.0, 6.0])
    win = s.window(-2, 4)
    assert win.size == 6
    assert np.isnan(win[[0, 1, 4, 5]]).all()
    np.testing.assert_allclose(win[2:4], [5.0, 6.0])


def test_window_entirely_outside_is_all_nan():
    s = _session([1.0, 2.0])
    assert np.isnan(s.window(50, 55)).all()


def test_empty_window_is_empty_not_an_error():
    s = _session([1.0, 2.0])
    assert s.window(1, 1).size == 0


def test_window_is_converted_like_at():
    s = _session([1.0, 2.0], fps=30.0, px_per_mm=2.0)
    np.testing.assert_allclose(s.window(0, 2), [1.5, 3.0])


# ---------------------------------------------------------------------------
# Loading from a pose CSV
# ---------------------------------------------------------------------------


def _write_pose(tmp_path, name="s.csv", n_frames=30):
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    xy = np.zeros((n_frames, 2, 2))
    # Constant 3-4-5 motion so the expected speed is easy to reason about.
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


def test_load_session_speed_reads_a_pose_csv(tmp_path):
    path = _write_pose(tmp_path)
    s = load_session_speed(path)
    assert s.px_per_frame.size == 30
    assert s.px_per_frame[0] == 0.0


def test_load_session_speed_matches_the_shared_signal(tmp_path):
    """The trace must be the same signal cohort thresholds are derived from."""
    from glider.analysis.behavior.classify.speed_state import causal_speed_series
    from glider.vision.pose.dlc import from_dlc_csv

    path = _write_pose(tmp_path)
    expected = causal_speed_series(from_dlc_csv(path).xy)

    np.testing.assert_allclose(load_session_speed(path).px_per_frame, expected)


def test_load_session_speed_takes_the_scale_it_is_given(tmp_path):
    path = _write_pose(tmp_path)
    assert load_session_speed(path, px_per_mm=2.0).is_calibrated is True


# ---------------------------------------------------------------------------
# SpeedCache: one parse per video, and failures are remembered
# ---------------------------------------------------------------------------


def test_cache_starts_empty():
    cache = SpeedCache()
    assert cache.state("v.mp4") == "absent"
    assert cache.get("v.mp4") is None


def test_cache_returns_what_was_stored():
    cache = SpeedCache()
    session = _session([1.0])
    cache.store("v.mp4", session)
    assert cache.state("v.mp4") == "ready"
    assert cache.get("v.mp4") is session


def test_cache_marks_work_in_flight_so_it_is_not_started_twice():
    """Landing on two clips from one video must not spawn two parses."""
    cache = SpeedCache()
    assert cache.begin("v.mp4") is True
    assert cache.begin("v.mp4") is False
    assert cache.state("v.mp4") == "loading"


def test_cache_does_not_restart_work_already_done():
    cache = SpeedCache()
    cache.store("v.mp4", _session([1.0]))
    assert cache.begin("v.mp4") is False


def test_cache_remembers_a_failure_and_does_not_retry_forever():
    """A pose CSV that won't parse must not be re-attempted on every clip."""
    cache = SpeedCache()
    cache.begin("v.mp4")
    cache.fail("v.mp4", "unreadable")
    assert cache.state("v.mp4") == "failed"
    assert cache.error("v.mp4") == "unreadable"
    assert cache.begin("v.mp4") is False
    assert cache.get("v.mp4") is None


def test_cache_keys_are_paths_however_they_are_spelled(tmp_path):
    """The same video reached as str and as Path is one entry, not two."""
    cache = SpeedCache()
    video = tmp_path / "v.mp4"
    cache.store(str(video), _session([1.0]))
    assert cache.state(video) == "ready"
