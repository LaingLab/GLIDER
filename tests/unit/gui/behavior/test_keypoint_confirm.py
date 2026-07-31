"""The labelled-frame confirmation shown before an Apply run.

A wrong keypoint order produces an empty ethogram with no error anywhere, so
this gate exists to make the mapping visible before an inference pass is spent.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.classify.features_stream import (
    expected_keypoint_order,
    keypoint_order_problem,
)
from glider.gui.behavior.keypoint_confirm import KeypointConfirmDialog, annotate_keypoints

NAMES = ["left_ear", "right_ear", "nose", "body_center", "left_hip", "right_hip", "tail_base"]


class _FakeModel:
    """Only the attribute the order recovery reads."""

    def __init__(self, names=NAMES, stats=("mean", "std", "max")):
        self.feature_names = [f"speed_{n}__{s}" for s in stats for n in names]


class TestExpectedKeypointOrder:
    def test_recovers_the_training_order(self):
        assert expected_keypoint_order(_FakeModel()) == NAMES

    def test_order_is_first_appearance_not_alphabetical(self):
        # The bug this guards: sorting silently reorders body parts.
        assert expected_keypoint_order(_FakeModel()) != sorted(NAMES)

    def test_unrecoverable_model_yields_empty(self):
        class NoSpeed:
            feature_names = ["dist_a_b__mean", "angle_x__mean"]

        assert expected_keypoint_order(NoSpeed()) == []


class TestKeypointOrderProblem:
    def test_matching_names_have_no_problem(self):
        assert keypoint_order_problem(_FakeModel(), NAMES) is None

    def test_a_permutation_is_caught(self):
        swapped = ["right_ear", "left_ear", *NAMES[2:]]
        problem = keypoint_order_problem(_FakeModel(), swapped)
        assert problem is not None
        assert "position 0" in problem
        assert "left_ear" in problem

    def test_wrong_count_is_caught(self):
        problem = keypoint_order_problem(_FakeModel(), NAMES[:5])
        assert "7 keypoints" in problem and "5 names" in problem

    def test_an_unreadable_model_does_not_invent_a_failure(self):
        class NoSpeed:
            feature_names = ["dist_a_b__mean"]

        # Better to let the run proceed than to block on a model we can't read.
        assert keypoint_order_problem(NoSpeed(), NAMES) is None


class TestAnnotateKeypoints:
    def _frame(self, w=256, h=256):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_draws_something_for_every_keypoint(self):
        pts = np.array([[20.0 + 25 * i, 40.0 + 10 * i] for i in range(7)])
        out = annotate_keypoints(self._frame(), pts, NAMES)
        assert out.any(), "nothing was drawn"

    def test_does_not_mutate_the_input_frame(self):
        frame = self._frame()
        pts = np.array([[30.0, 30.0]] * 7)
        annotate_keypoints(frame, pts, NAMES)
        assert not frame.any(), "the caller's frame was modified"

    def test_small_frames_are_upscaled_for_legibility(self):
        # 256x256 arena video is unreadable at native size.
        out = annotate_keypoints(self._frame(256, 256), np.array([[10.0, 10.0]]), ["nose"])
        assert max(out.shape[:2]) >= 720

    def test_undetected_keypoints_are_skipped_not_drawn_at_zero(self):
        pts = np.array([[np.nan, np.nan], [100.0, 100.0]])
        out = annotate_keypoints(self._frame(), pts, ["missing", "present"])
        # A NaN drawn as (0,0) would put ink in the top-left corner.
        assert not out[:40, :40].any()

    def test_surplus_points_do_not_raise(self):
        pts = np.array([[50.0, 50.0]] * 9)
        annotate_keypoints(self._frame(), pts, NAMES)  # 9 points, 7 names


class TestConfirmDialog:
    def _dialog(self, qtbot, tmp_path, monkeypatch, warning=None):
        # Never load a real pose model in tests.
        monkeypatch.setattr(
            "glider.gui.behavior.keypoint_confirm.first_detected_frame",
            lambda *a, **k: None,
        )
        dialog = KeypointConfirmDialog(
            tmp_path / "v.mp4", tmp_path / "y.pt", NAMES, warning=warning
        )
        qtbot.addWidget(dialog)
        return dialog

    def test_confirm_is_disabled_until_a_frame_is_shown(self, qtbot, tmp_path, monkeypatch):
        dialog = self._dialog(qtbot, tmp_path, monkeypatch)
        assert dialog._ok.isEnabled() is False

    def test_showing_a_frame_enables_confirmation(self, qtbot, tmp_path, monkeypatch):
        dialog = self._dialog(qtbot, tmp_path, monkeypatch)
        dialog.show_frame(np.zeros((80, 80, 3), dtype=np.uint8))
        assert dialog._ok.isEnabled() is True

    def test_the_legend_names_every_position(self, qtbot, tmp_path, monkeypatch):
        dialog = self._dialog(qtbot, tmp_path, monkeypatch)
        text = dialog._legend.text()
        for i, name in enumerate(NAMES):
            assert f"{i}:{name}" in text

    def test_a_bundle_warning_is_surfaced(self, qtbot, tmp_path, monkeypatch):
        dialog = self._dialog(
            qtbot, tmp_path, monkeypatch, warning="position 0 should be 'left_ear'"
        )
        assert "left_ear" in dialog._warning.text()

    def test_a_failed_preview_still_allows_a_deliberate_run(self, qtbot, tmp_path, monkeypatch):
        dialog = self._dialog(qtbot, tmp_path, monkeypatch)
        dialog._on_failed("no animal detected")
        # Blocking entirely would strand anyone whose first video starts empty.
        assert dialog._ok.isEnabled() is True
        assert "without checking" in dialog._ok.text()


class TestRunIsGated:
    def _ready_tab(self, qtbot, tmp_path):
        from glider.gui.behavior.window import ApplyTab

        tab = ApplyTab()
        qtbot.addWidget(tab)
        tab._model_path = tmp_path / "m.pkl"
        tab._yolo_path = tmp_path / "y.pt"
        tab._output_dir = tmp_path / "out"
        video = tmp_path / "v.mp4"
        video.write_bytes(b"")
        tab._videos = [video]
        tab._keypoints_edit.setText(",".join(NAMES))
        return tab

    def test_declining_the_confirmation_does_not_start_a_run(self, qtbot, tmp_path, monkeypatch):
        tab = self._ready_tab(qtbot, tmp_path)
        started = []
        monkeypatch.setattr(tab, "_confirm_keypoints", lambda names: False)
        monkeypatch.setattr(tab, "_run_next", lambda: started.append(True))
        tab._on_run()
        assert started == []
        assert tab._run_btn.isEnabled() is True

    def test_confirming_starts_the_run(self, qtbot, tmp_path, monkeypatch):
        tab = self._ready_tab(qtbot, tmp_path)
        started = []
        monkeypatch.setattr(tab, "_confirm_keypoints", lambda names: True)
        monkeypatch.setattr(tab, "_run_next", lambda: started.append(True))
        tab._on_run()
        assert started == [True]
        assert tab._keypoint_names == NAMES

    def test_the_confirmation_sees_the_names_that_were_typed(self, qtbot, tmp_path, monkeypatch):
        tab = self._ready_tab(qtbot, tmp_path)
        tab._keypoints_edit.setText(" nose , left_ear ,, right_ear ")
        seen = {}

        def capture(names):
            seen["names"] = names
            return False  # must decline: a truthy return starts a real run

        monkeypatch.setattr(tab, "_confirm_keypoints", capture)
        monkeypatch.setattr(tab, "_run_next", lambda: pytest.fail("run should not start"))
        tab._on_run()
        assert seen["names"] == ["nose", "left_ear", "right_ear"]

    def test_an_unreadable_bundle_does_not_block_the_run(self, qtbot, tmp_path):
        tab = self._ready_tab(qtbot, tmp_path)
        # _model_path points at a file that isn't a bundle at all.
        assert tab._keypoint_warning(NAMES) is None


@pytest.mark.parametrize("count", [1, 7, 12])
def test_annotate_handles_any_keypoint_count(count):
    pts = np.column_stack([np.linspace(10, 200, count), np.linspace(10, 200, count)])
    out = annotate_keypoints(
        np.zeros((256, 256, 3), dtype=np.uint8), pts, [f"k{i}" for i in range(count)]
    )
    assert out.any()
