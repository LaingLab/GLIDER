"""Matching one folder of pose CSVs to a list of videos in the Apply tab.

Batch Pose Tracking writes its CSVs wherever it was pointed, which is usually
not beside the videos. Without this the operator copies a CSV per video by
hand, or — worse — does not, and every video is silently tracked again.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("sklearn")

from glider.gui.behavior.window import ApplyTab  # noqa: E402

NAMES = ["nose", "left_ear", "right_ear", "tail_base"]


def _video(folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}.mp4"
    path.write_bytes(b"")
    return path


def _pose_csv(folder, stem, names=NAMES, suffix="DLC_exp-7"):
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stem}{suffix}.csv"
    to_dlc_csv(
        PoseData(
            xy=np.zeros((3, len(names), 2)),
            confidence=np.ones((3, len(names))),
            keypoint_names=list(names),
            fps=30.0,
        ),
        path,
    )
    return path


@pytest.fixture
def tab(qtbot):
    widget = ApplyTab()
    qtbot.addWidget(widget)
    return widget


class TestMatching:
    def test_poses_in_a_separate_folder_are_matched(self, tab, tmp_path):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, f"t{i}") for i in range(3)]
        for i in range(3):
            _pose_csv(poses, f"t{i}")
        tab._pose_dir = poses
        matched, unmatched = tab._match_poses()
        assert len(matched) == 3
        assert unmatched == []

    def test_without_a_folder_it_still_looks_beside_each_video(self, tab, tmp_path):
        videos = tmp_path / "videos"
        tab._videos = [_video(videos, "t0")]
        _pose_csv(videos, "t0")
        assert tab._match_poses() == (tab._videos, [])

    def test_videos_with_no_csv_are_reported(self, tab, tmp_path):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0"), _video(videos, "t1")]
        _pose_csv(poses, "t0")
        tab._pose_dir = poses
        matched, unmatched = tab._match_poses()
        assert [p.stem for p in matched] == ["t0"]
        assert [p.stem for p in unmatched] == ["t1"]

    def test_a_missing_folder_matches_nothing_rather_than_raising(self, tab, tmp_path):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        tab._pose_dir = tmp_path / "not_there"
        assert tab._match_poses()[1] == tab._videos


class TestTheSummaryLabel:
    def test_it_confirms_full_coverage(self, tab, tmp_path):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, f"t{i}") for i in range(2)]
        for i in range(2):
            _pose_csv(poses, f"t{i}")
        tab._pose_dir = poses
        tab._refresh_pose_match()
        assert "all 2" in tab._pose_match_label.text()

    def test_it_names_what_is_missing(self, tab, tmp_path):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0"), _video(videos, "t1")]
        _pose_csv(poses, "t0")
        tab._pose_dir = poses
        tab._refresh_pose_match()
        text = tab._pose_match_label.text()
        assert "1 of 2" in text and "t1.mp4" in text
        assert "tracked from scratch" in text

    def test_with_reuse_off_it_says_tracking_will_run(self, tab, tmp_path):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        tab._reuse_poses.setChecked(False)
        assert "Tracking will run" in tab._pose_match_label.text()

    def test_choosing_a_folder_updates_it(self, tab, tmp_path, monkeypatch):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0")]
        _pose_csv(poses, "t0")
        monkeypatch.setattr(
            "glider.gui.behavior.window.QFileDialog.getExistingDirectory",
            lambda *a, **k: str(poses),
        )
        tab._on_choose_pose_dir()
        assert tab._pose_dir == poses
        assert "all 1" in tab._pose_match_label.text()

    def test_clearing_returns_to_the_video_folder(self, tab, tmp_path):
        tab._pose_dir = tmp_path / "poses"
        tab._on_clear_pose_dir()
        assert tab._pose_dir is None
        assert "beside each video" in tab._pose_dir_label.text()


class TestTheRunGate:
    def test_full_coverage_asks_nothing(self, tab, tmp_path, monkeypatch):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0")]
        _pose_csv(poses, "t0")
        tab._pose_dir = poses
        monkeypatch.setattr(
            "glider.gui.behavior.window.QMessageBox.question",
            lambda *a, **k: pytest.fail("nothing to ask about"),
        )
        assert tab._confirm_unmatched_poses() is True

    def test_it_asks_before_tracking_from_scratch(self, tab, tmp_path, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0"), _video(videos, "t1")]
        _pose_csv(poses, "t0")
        tab._pose_dir = poses
        asked = {}

        def question(_parent, title, text, buttons, default):
            asked.update(title=title, text=text, default=default)
            return QMessageBox.StandardButton.No

        monkeypatch.setattr("glider.gui.behavior.window.QMessageBox.question", question)
        assert tab._confirm_unmatched_poses() is False
        assert "t1.mp4" in asked["text"]
        # Defaulting to Yes would make a mistyped folder cost hours.
        assert asked["default"] == QMessageBox.StandardButton.No

    def test_reuse_off_skips_the_question_entirely(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        tab._reuse_poses.setChecked(False)
        monkeypatch.setattr(
            "glider.gui.behavior.window.QMessageBox.question",
            lambda *a, **k: pytest.fail("tracking was chosen deliberately"),
        )
        assert tab._confirm_unmatched_poses() is True


class TestTheRunActuallyUsesTheseSettings:
    """These checkboxes existed but were never read: `_run_next` built the
    worker without them, so every run re-tracked and the annotated-video box
    did nothing."""

    def _capture_worker_kwargs(self, tab, tmp_path, monkeypatch):
        seen = {}

        class _Sig:
            def connect(self, *a, **k):
                pass

        class _FakeWorker:
            # _run_next connects these and hands `run` to the thread.
            progress = _Sig()
            failed = _Sig()
            finished = _Sig()

            def __init__(self, **kwargs):
                seen.update(kwargs)

            def moveToThread(self, _thread):
                pass

            def run(self):
                pass

        class _FakeThread:
            def __init__(self, *a, **k):
                self.started = _Sig()

            def start(self):
                pass

        monkeypatch.setattr("glider.gui.behavior.workers.ApplyWorker", _FakeWorker)
        monkeypatch.setattr("glider.gui.behavior.window.QThread", _FakeThread)
        tab._output_dir = tmp_path / "out"
        tab._model_path = tmp_path / "m.pkl"
        tab._yolo_path = tmp_path / "y.pt"
        tab._keypoint_names = NAMES
        tab._queue = list(tab._videos)
        tab._run_next()
        return seen

    def test_the_chosen_pose_folder_reaches_the_worker(self, tab, tmp_path, monkeypatch):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0")]
        _pose_csv(poses, "t0")
        tab._pose_dir = poses
        seen = self._capture_worker_kwargs(tab, tmp_path, monkeypatch)
        assert seen["reuse_existing_poses"] is True
        assert seen["pose_dir"] == poses

    def test_unchecking_reuse_forces_tracking(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        tab._pose_dir = tmp_path / "poses"
        tab._reuse_poses.setChecked(False)
        seen = self._capture_worker_kwargs(tab, tmp_path, monkeypatch)
        assert seen["reuse_existing_poses"] is False
        # The folder must not leak through and quietly re-enable reuse.
        assert seen["pose_dir"] is None

    def test_the_annotated_video_box_reaches_the_worker(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        tab._render_video.setChecked(True)
        seen = self._capture_worker_kwargs(tab, tmp_path, monkeypatch)
        assert seen["write_annotated"] is True

    def test_annotated_video_is_off_by_default(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        seen = self._capture_worker_kwargs(tab, tmp_path, monkeypatch)
        assert seen["write_annotated"] is False


class TestTheKeypointCrossCheck:
    def test_it_reads_the_csv_from_the_chosen_folder(self, tab, tmp_path):
        """A foreign folder is the likeliest source of foreign names, so the
        check must follow it rather than silently retire."""
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0")]
        _pose_csv(poses, "t0", names=["snout", "ear_l", "ear_r", "tail"])
        tab._pose_dir = poses
        message = tab._pose_csv_disagreement(NAMES)
        assert message is not None and "snout" in message

    def test_agreeing_names_produce_no_message(self, tab, tmp_path):
        videos, poses = tmp_path / "videos", tmp_path / "poses"
        tab._videos = [_video(videos, "t0")]
        _pose_csv(poses, "t0", names=NAMES)
        tab._pose_dir = poses
        assert tab._pose_csv_disagreement(NAMES) is None


class TestLabelStabilityControls:
    """Both knobs existed in the pipeline; neither was reachable from here.

    A per-frame classifier switches label ~10x more often than a mouse
    changes behavior, so every bout-level number was at the mercy of a
    default nobody could see or change.
    """

    def test_smoothing_defaults_to_absorbing_flicker(self, tab):
        assert tab._smooth_window.value() > 1

    def test_the_minimum_bout_filter_is_off_by_default(self, tab):
        """What counts as a bout is a scoring decision, not a default."""
        assert tab._min_bout_s.value() == 0.0

    def test_both_reach_the_worker(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        tab._smooth_window.setValue(7)
        tab._min_bout_s.setValue(0.5)
        seen = TestTheRunActuallyUsesTheseSettings()._capture_worker_kwargs(
            tab, tmp_path, monkeypatch
        )
        assert seen["smooth_window"] == 7
        assert seen["min_bout_s"] == pytest.approx(0.5)

    def test_a_zero_minimum_is_passed_as_no_filter(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        seen = TestTheRunActuallyUsesTheseSettings()._capture_worker_kwargs(
            tab, tmp_path, monkeypatch
        )
        assert seen["min_bout_s"] is None


class TestScaleAdvisories:
    def test_nothing_to_say_asks_nothing(self, tab, tmp_path, monkeypatch):
        monkeypatch.setattr(tab, "_scale_advisories", list)
        monkeypatch.setattr(
            "glider.gui.behavior.window.QMessageBox.question",
            lambda *a, **k: pytest.fail("no advisory, no dialog"),
        )
        assert tab._confirm_scale_advisories() is True

    def test_an_advisory_is_shown_and_can_be_declined(self, tab, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox

        monkeypatch.setattr(tab, "_scale_advisories", lambda: ["the animal is tiny"])
        seen = {}

        def question(_parent, _title, text, _buttons, default):
            seen.update(text=text, default=default)
            return QMessageBox.StandardButton.No

        monkeypatch.setattr("glider.gui.behavior.window.QMessageBox.question", question)
        assert tab._confirm_scale_advisories() is False
        assert "the animal is tiny" in seen["text"]
        # Advisory, not a hazard: proceeding is the default here.
        assert seen["default"] == QMessageBox.StandardButton.Yes

    def test_a_broken_check_never_blocks_a_run(self, tab, tmp_path, monkeypatch):
        """A diagnostic that breaks a run is worse than one that stays quiet."""
        tab._videos = [_video(tmp_path / "videos", "t0")]
        _pose_csv(tmp_path / "videos", "t0")
        tab._model_path = tmp_path / "not-a-bundle.pkl"
        tab._model_path.write_text("garbage")
        assert tab._scale_advisories() == []


class TestTheTimeRange:
    """Scoring minutes 2-7 of each video instead of all of it."""

    def test_it_is_off_by_default(self, tab):
        assert tab._range_on.isChecked() is False
        assert tab._time_range() == (None, None)

    def test_minutes_become_seconds(self, tab):
        tab._range_on.setChecked(True)
        tab._range_start.setValue(2.0)
        tab._range_end.setValue(7.0)
        assert tab._time_range() == (120.0, 420.0)

    def test_an_open_end_means_to_the_end_of_the_video(self, tab):
        tab._range_on.setChecked(True)
        tab._range_start.setValue(2.0)
        tab._range_end.setValue(0.0)  # the spin's special "to the end" value
        start, end = tab._time_range()
        assert start == 120.0
        assert end is None

    def test_the_spins_follow_the_checkbox(self, tab):
        assert tab._range_start.isEnabled() is False
        tab._range_on.setChecked(True)
        assert tab._range_start.isEnabled() is True
        assert tab._range_end.isEnabled() is True

    def test_the_hint_reports_the_span(self, tab):
        tab._range_on.setChecked(True)
        tab._range_start.setValue(2.0)
        tab._range_end.setValue(7.0)
        assert "300 s" in tab._range_hint.text()

    def test_a_backwards_window_is_called_out(self, tab):
        tab._range_on.setChecked(True)
        tab._range_start.setValue(7.0)
        tab._range_end.setValue(2.0)
        assert "before it starts" in tab._range_hint.text()

    def test_the_window_reaches_the_worker(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        tab._range_on.setChecked(True)
        tab._range_start.setValue(2.0)
        tab._range_end.setValue(7.0)
        seen = TestTheRunActuallyUsesTheseSettings()._capture_worker_kwargs(
            tab, tmp_path, monkeypatch
        )
        assert seen["start_s"] == pytest.approx(120.0)
        assert seen["end_s"] == pytest.approx(420.0)

    def test_no_window_passes_nothing(self, tab, tmp_path, monkeypatch):
        tab._videos = [_video(tmp_path / "videos", "t0")]
        seen = TestTheRunActuallyUsesTheseSettings()._capture_worker_kwargs(
            tab, tmp_path, monkeypatch
        )
        assert seen["start_s"] is None
        assert seen["end_s"] is None
