"""Loading an analysed session and asking questions about a window of it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glider.analysis.behavior.session_view import (
    SessionView,
    SessionViewError,
    find_session_poses,
)

NAMES = ["nose", "l_ear", "r_ear", "tail_base"]


def _write_session(folder, *, n=300, fps=30.0, with_poses=True, with_meta=True):
    """An apply-run output folder: ethogram, poses, sidecar."""
    folder.mkdir(parents=True, exist_ok=True)
    labels = ["groom"] * 100 + ["locomote"] * 100 + ["groom"] * 100
    pd.DataFrame({"frame": range(n), "behavior": labels[:n]}).to_csv(
        folder / "ethogram_raw.csv", index=False
    )
    if with_poses:
        from glider.vision.pose.core import PoseData
        from glider.vision.pose.dlc import to_dlc_csv

        # Travels 1 px per frame along x, so distance is exactly known.
        xy = np.zeros((n, len(NAMES), 2))
        xy[:, :, 0] = np.arange(n)[:, None]
        to_dlc_csv(
            PoseData(
                xy=xy,
                confidence=np.ones((n, len(NAMES))),
                keypoint_names=NAMES,
                fps=fps,
                metadata={"resolution": (640, 480)} if with_meta else {},
            ),
            folder / "vDLC_exp-7.csv",
        )
    return folder / "ethogram_raw.csv"


class TestLoading:
    def test_loads_labels_and_frames(self, tmp_path):
        etho = _write_session(tmp_path / "v")
        view = SessionView.load(etho)
        assert view.n_rows == 300
        assert view.labels[0] == "groom"
        assert view.labels[150] == "locomote"

    def test_finds_the_poses_beside_the_ethogram(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v"))
        assert view.xy is not None
        assert view.keypoint_names == NAMES

    def test_reads_the_resolution_from_the_sidecar(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v"))
        assert view.resolution == (640, 480)

    def test_it_remembers_which_pose_csv_it_read(self, tmp_path):
        """Repairing a sidecar later needs to know which one to write to."""
        view = SessionView.load(_write_session(tmp_path / "v"))
        assert view.pose_path == tmp_path / "v" / "vDLC_exp-7.csv"

    def test_a_session_without_poses_still_loads(self, tmp_path):
        """Behaviour statistics must not depend on the poses surviving."""
        view = SessionView.load(_write_session(tmp_path / "v", with_poses=False))
        assert view.n_rows == 300
        assert view.xy is None
        assert view.resolution is None
        assert view.pose_path is None

    def test_a_missing_resolution_is_none_not_guessed(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v", with_meta=False))
        # Inferring it from the coordinate range would shrink the arena to
        # wherever the animal happened to go.
        assert view.resolution is None

    def test_a_non_ethogram_csv_is_refused(self, tmp_path):
        path = tmp_path / "not_an_ethogram.csv"
        pd.DataFrame({"a": [1]}).to_csv(path, index=False)
        with pytest.raises(SessionViewError, match="behavior"):
            SessionView.load(path)

    def test_an_unreadable_file_is_refused(self, tmp_path):
        with pytest.raises(SessionViewError):
            SessionView.load(tmp_path / "absent.csv")


class TestScrubbing:
    def test_the_label_at_a_frame(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v"))
        assert view.label_at(0) == "groom"
        assert view.label_at(150) == "locomote"
        assert view.label_at(250) == "groom"

    def test_a_frame_past_the_end_has_no_label(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v"))
        assert view.label_at(99999) == "groom"  # last known row
        assert view.label_at(-5) == ""

    def test_the_centroid_averages_the_keypoints(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v"))
        centre = view.centroid()
        assert centre.shape == (300, 2)
        assert centre[10][0] == pytest.approx(10.0)

    def test_the_trail_covers_the_requested_seconds(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v"))
        trail = view.trail(200, seconds=2.0)  # 2 s at 30 fps = 60 frames
        assert len(trail) == 60
        assert trail[-1][0] == pytest.approx(200.0)

    def test_the_trail_is_clipped_at_the_start(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v"))
        assert len(view.trail(10, seconds=5.0)) == 11

    def test_no_poses_means_no_trail(self, tmp_path):
        view = SessionView.load(_write_session(tmp_path / "v", with_poses=False))
        assert view.trail(50) is None


class TestSegmentStats:
    def _view(self, tmp_path, px_per_mm=None):
        view = SessionView.load(_write_session(tmp_path / "v"))
        view.px_per_mm = px_per_mm
        return view

    def test_bouts_cover_only_the_selected_window(self, tmp_path):
        view = self._view(tmp_path)
        stats = view.segment_stats(100, 199)  # the locomote block
        assert set(stats.bouts["state"]) == {"locomote"}
        assert stats.duration_s == pytest.approx(100 / 30.0)

    def test_a_window_spanning_two_behaviours_reports_both(self, tmp_path):
        view = self._view(tmp_path)
        stats = view.segment_stats(50, 249)
        assert set(stats.bouts["state"]) == {"groom", "locomote"}

    def test_distance_uses_the_calibration(self, tmp_path):
        # 1 px/frame for 100 frames at 4 px/mm = 100/4/10 = 2.5 cm.
        view = self._view(tmp_path, px_per_mm=4.0)
        stats = view.segment_stats(0, 100)
        assert stats.distance_cm == pytest.approx(2.5, rel=1e-6)

    def test_speed_is_reported_in_centimetres_per_second(self, tmp_path):
        # 1 px/frame at 4 px/mm and 30 fps = 0.75 cm/s.
        view = self._view(tmp_path, px_per_mm=4.0)
        stats = view.segment_stats(0, 100)
        assert stats.mean_speed_cm_s == pytest.approx(0.75, rel=1e-6)
        assert stats.peak_speed_cm_s == pytest.approx(0.75, rel=1e-6)

    def test_no_calibration_means_no_distance_rather_than_pixels(self, tmp_path):
        """Reporting a pixel count as if it were a distance would be worse."""
        view = self._view(tmp_path)
        stats = view.segment_stats(0, 100)
        assert stats.distance_cm is None
        assert stats.mean_speed_cm_s is None

    def test_window_thresholds_are_reported_with_their_unit(self, tmp_path):
        view = self._view(tmp_path, px_per_mm=4.0)
        stats = view.segment_stats(0, 200)
        assert stats.threshold_unit == "cm/s"
        assert stats.freeze_threshold <= stats.dart_threshold

    def test_window_thresholds_fall_back_to_pixels_uncalibrated(self, tmp_path):
        view = self._view(tmp_path)
        stats = view.segment_stats(0, 200)
        assert stats.threshold_unit == "px/frame"

    def test_a_reversed_selection_is_accepted(self, tmp_path):
        view = self._view(tmp_path)
        assert view.segment_stats(200, 100).start_frame == 100

    def test_the_loaded_labels_are_never_recomputed(self, tmp_path):
        """Segment thresholds are informational; the ethogram is the record."""
        view = self._view(tmp_path, px_per_mm=4.0)
        before = list(view.labels)
        view.segment_stats(0, 150)
        assert view.labels == before


class TestBoutTableUnits:
    """The columns are named `_s`; compute_bouts speaks milliseconds."""

    def _view(self, tmp_path, labels):
        folder = tmp_path / "v"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"frame": range(len(labels)), "behavior": labels}).to_csv(
            folder / "ethogram_raw.csv", index=False
        )
        return SessionView.load(folder / "ethogram_raw.csv")

    def test_durations_are_seconds_not_milliseconds(self, tmp_path):
        # 90 frames at 30 fps is 3 s, not 3000.
        view = self._view(tmp_path, ["groom"] * 90 + ["locomote"] * 90)
        bouts = view.segment_stats(0, 179).bouts.set_index("state")
        assert bouts.loc["groom", "total_s"] == pytest.approx(3.0)
        assert bouts.loc["groom", "mean_s"] == pytest.approx(3.0)
        assert bouts.loc["groom", "median_s"] == pytest.approx(3.0)

    def test_the_totals_add_up_to_the_window(self, tmp_path):
        view = self._view(tmp_path, ["groom"] * 90 + ["locomote"] * 90)
        stats = view.segment_stats(0, 179)
        assert stats.bouts["total_s"].sum() == pytest.approx(stats.duration_s)
        assert stats.bouts["fraction"].sum() == pytest.approx(1.0)

    def test_unscored_frames_get_no_bout_row(self, tmp_path):
        view = self._view(tmp_path, ["groom"] * 30 + [""] * 30 + ["groom"] * 30)
        bouts = view.segment_stats(0, 89).bouts
        assert list(bouts["state"]) == ["groom"]
        # Not bridged: two separate grooming bouts either side of the gap.
        assert int(bouts.iloc[0]["n_bouts"]) == 2
        # The gap still counts against the fraction of the window.
        assert bouts.iloc[0]["fraction"] == pytest.approx(2 / 3)

    def test_an_entirely_unscored_window_yields_an_empty_table(self, tmp_path):
        view = self._view(tmp_path, [""] * 60)
        bouts = view.segment_stats(0, 59).bouts
        assert bouts.empty
        assert list(bouts.columns) == [
            "state",
            "n_bouts",
            "total_s",
            "fraction",
            "mean_s",
            "median_s",
        ]


class TestFindingThePoses:
    """Reusing already-tracked poses copies nothing into the output folder,
    so looking only beside the ethogram finds nothing at all."""

    def _layout(self, tmp_path, *, poses_where="videos", manifest=None):
        """videos/<stem>DLC_exp-7.csv + videos/outputs/<stem>/ethogram_raw.csv"""
        import json

        videos = tmp_path / "videos"
        out = videos / "outputs" / "t4_d2"
        out.mkdir(parents=True)
        pd.DataFrame({"frame": range(30), "behavior": ["groom"] * 30}).to_csv(
            out / "ethogram_raw.csv", index=False
        )
        target = {"videos": videos, "output": out, "elsewhere": tmp_path / "far"}[poses_where]
        target.mkdir(parents=True, exist_ok=True)
        name = "t4_d2DLC_exp-7.csv" if poses_where != "output" else "t4_d2DLC_exp-7.csv"
        pose_csv = _write_pose_csv(target / name)
        if manifest is not None:
            (out / "run.json").write_text(json.dumps(manifest(pose_csv)))
        return out / "ethogram_raw.csv", pose_csv

    def test_poses_beside_the_ethogram_are_still_found(self, tmp_path):
        etho, pose_csv = self._layout(tmp_path, poses_where="output")
        assert find_session_poses(etho) == pose_csv

    def test_poses_left_with_the_videos_are_found_by_searching_up(self, tmp_path):
        etho, pose_csv = self._layout(tmp_path, poses_where="videos")
        assert find_session_poses(etho) == pose_csv

    def test_the_manifest_path_wins(self, tmp_path):
        etho, pose_csv = self._layout(
            tmp_path, poses_where="elsewhere", manifest=lambda p: {"pose_csv": str(p)}
        )
        assert find_session_poses(etho) == pose_csv

    def test_a_stale_manifest_path_falls_back_to_searching(self, tmp_path):
        etho, pose_csv = self._layout(
            tmp_path,
            poses_where="videos",
            manifest=lambda _p: {"pose_csv": str(tmp_path / "gone" / "x.csv")},
        )
        assert find_session_poses(etho) == pose_csv

    def test_another_animals_poses_are_never_picked_up(self, tmp_path):
        """The search is anchored on the output folder's name."""
        etho, _ = self._layout(tmp_path, poses_where="elsewhere")
        _write_pose_csv(tmp_path / "videos" / "t9_d2DLC_exp-7.csv")
        assert find_session_poses(etho) is None

    def test_the_view_loads_poses_found_upward(self, tmp_path):
        etho, pose_csv = self._layout(tmp_path, poses_where="videos")
        view = SessionView.load(etho)
        assert view.xy is not None
        assert view.pose_path == pose_csv

    def test_an_explicit_pose_csv_overrides_discovery(self, tmp_path):
        etho, _ = self._layout(tmp_path, poses_where="videos")
        chosen = _write_pose_csv(tmp_path / "hand_picked.csv")
        view = SessionView.load(etho, pose_csv=chosen)
        assert view.pose_path == chosen


def _write_pose_csv(path, n=30):
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    path.parent.mkdir(parents=True, exist_ok=True)
    to_dlc_csv(
        PoseData(
            xy=np.zeros((n, len(NAMES), 2)),
            confidence=np.ones((n, len(NAMES))),
            keypoint_names=NAMES,
            fps=30.0,
        ),
        path,
    )
    return path
