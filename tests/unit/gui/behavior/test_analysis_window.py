"""The session review window: ethogram scrubber, keypoint canvas, segments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtGui import QColor, QKeyEvent  # noqa: E402

from glider.analysis.behavior.session_view import SessionView  # noqa: E402
from glider.gui.behavior.analysis_window import (  # noqa: E402
    AnalysisWindow,
    KeypointCanvas,
    behavior_qcolor,
)
from glider.gui.review.timeline import HEADER_W, TimelineView  # noqa: E402

NAMES = ["nose", "l_ear", "r_ear", "tail_base"]


def _session(folder, *, n=300, with_poses=True, with_resolution=True):
    folder.mkdir(parents=True, exist_ok=True)
    labels = ["groom"] * 100 + ["locomote"] * 100 + ["groom"] * 100
    pd.DataFrame({"frame": range(n), "behavior": labels[:n]}).to_csv(
        folder / "ethogram_raw.csv", index=False
    )
    if with_poses:
        from glider.vision.pose.core import PoseData
        from glider.vision.pose.dlc import to_dlc_csv

        xy = np.zeros((n, len(NAMES), 2))
        xy[:, :, 0] = np.arange(n)[:, None]
        xy[:, :, 1] = 100.0
        to_dlc_csv(
            PoseData(
                xy=xy,
                confidence=np.ones((n, len(NAMES))),
                keypoint_names=NAMES,
                fps=30.0,
                metadata={"resolution": (640, 480)} if with_resolution else {},
            ),
            folder / "vDLC_exp-7.csv",
        )
    return folder / "ethogram_raw.csv"


class TestBehaviorColours:
    def test_a_behaviour_gets_a_stable_colour(self):
        assert behavior_qcolor("groom") == behavior_qcolor("groom")

    def test_different_behaviours_differ(self):
        assert behavior_qcolor("groom") != behavior_qcolor("locomote")

    def test_unscored_frames_read_as_background(self):
        from glider.gui.styles import colors

        assert behavior_qcolor("") == QColor(colors.BORDER)


class TestKeypointCanvas:
    def _canvas(self, qtbot, tmp_path, **kw):
        canvas = KeypointCanvas()
        qtbot.addWidget(canvas)
        canvas.resize(400, 300)
        canvas.set_view(SessionView.load(_session(tmp_path / "v", **kw)))
        return canvas

    def test_it_draws_a_session_with_poses(self, qtbot, tmp_path):
        canvas = self._canvas(qtbot, tmp_path)
        canvas.set_frame(150)
        assert canvas.grab().toImage().width() == 400

    def test_without_poses_it_says_why(self, qtbot, tmp_path):
        canvas = self._canvas(qtbot, tmp_path, with_poses=False)
        assert "No pose CSV" in canvas._why_blank()

    def test_without_a_resolution_it_says_why(self, qtbot, tmp_path):
        """Stretching points to their own range would redraw the arena."""
        canvas = self._canvas(qtbot, tmp_path, with_resolution=False)
        assert canvas._transform() is None
        assert "resolution" in canvas._why_blank()

    def test_the_transform_preserves_aspect_ratio(self, qtbot, tmp_path):
        canvas = self._canvas(qtbot, tmp_path)
        scale, dx, dy = canvas._transform()
        # 640x480 into 400x300 -> limited by width
        assert scale == pytest.approx(400 / 640)
        assert dy == pytest.approx((300 - 480 * scale) / 2)

    def test_the_trail_is_drawn_in_the_accent_colour(self, qtbot, tmp_path):
        """A QSS rgba() string is an invalid QColor, i.e. black on a black canvas."""
        canvas = self._canvas(qtbot, tmp_path)
        canvas.set_frame(150)
        # The fixture walks 1 px/frame along x at y=100; 640x480 into 400x300
        # scales by 0.625, so the trail is a horizontal line at y~62.
        image = canvas.grab().toImage()
        row = [image.pixelColor(x, y) for x in range(20, 80) for y in (61, 62, 63)]
        assert any(c.blue() > c.red() + 10 for c in row)

    def test_trail_settings_are_applied(self, qtbot, tmp_path):
        canvas = self._canvas(qtbot, tmp_path)
        canvas.set_trail(2.5, True)
        assert canvas._trail_s == pytest.approx(2.5)
        canvas.set_trail(2.5, False)
        assert canvas._show_trail is False


class TestAnalysisWindow:
    def _win(self, qtbot, tmp_path, **kw):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v", **kw))
        return win

    def test_loading_populates_the_widgets(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        assert win._view is not None
        assert "300" in win._session_text.text()

    def test_selecting_a_window_fills_the_bout_table(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(100, 199)
        assert win._bouts.rowCount() == 1
        assert win._bouts.item(0, 0).text() == "locomote"

    def test_selection_still_arrives_in_frames(self, qtbot, tmp_path):
        """The swap from EthogramBar to TimelineView must not change the
        unit the tables receive. If it does, every window statistic is
        computed over the wrong range and nothing raises."""
        win = self._win(qtbot, tmp_path)
        received: list[tuple[int, int]] = []
        win._bar.selection_changed.connect(lambda a, b: received.append((a, b)))
        win._bar.set_selection(100, 199)
        assert received == [(100, 199)]
        assert win._bouts.item(0, 0).text() == "locomote"

    def test_a_span_across_behaviours_lists_both(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(50, 250)
        states = {win._bouts.item(r, 0).text() for r in range(win._bouts.rowCount())}
        assert states == {"groom", "locomote"}

    def test_the_summary_marks_segment_thresholds_as_comparison_only(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(0, 200)
        text = win._summary.text()
        assert "comparison" in text and "unchanged" in text

    def test_an_uncalibrated_session_says_so_rather_than_reporting_pixels(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(0, 200)
        assert "no calibration" in win._summary.text().lower()

    def test_select_whole_session(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._select_all()
        assert win._bar.selection() == (0, 299)

    def test_scrubbing_moves_the_clock(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._set_frame(90)  # 3 s at 30 fps
        assert win._clock.text() == "00:00:03:00"

    def test_playback_stops_at_the_end(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._set_frame(298)
        win._toggle_play()
        win._advance()  # 299
        win._advance()  # would pass the end
        assert win._timer.isActive() is False
        assert "Play" in win._play.text()

    def test_the_repair_button_is_hidden_when_nothing_needs_repairing(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        assert win._fix_resolution.isHidden() is True

    def test_the_repair_button_appears_for_a_sidecar_without_a_resolution(self, qtbot, tmp_path):
        """Runs from before the field existed must still be viewable."""
        win = self._win(qtbot, tmp_path, with_resolution=False)
        assert win._fix_resolution.isHidden() is False

    def test_repairing_writes_the_resolution_and_reloads(self, qtbot, tmp_path, monkeypatch):
        win = self._win(qtbot, tmp_path, with_resolution=False)
        video = tmp_path / "v" / "v.mp4"
        video.write_bytes(b"")  # never opened; the reader is stubbed
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getOpenFileName",
            lambda *a, **k: (str(video), ""),
        )
        monkeypatch.setattr("glider.vision.video_source.video_resolution", lambda _p: (800, 600))
        win._resolution_from_video()
        assert win._view.resolution == (800, 600)
        assert win._fix_resolution.isHidden() is True

    def test_an_unreadable_video_is_reported_and_changes_nothing(
        self, qtbot, tmp_path, monkeypatch
    ):
        win = self._win(qtbot, tmp_path, with_resolution=False)
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getOpenFileName",
            lambda *a, **k: (str(tmp_path / "broken.mp4"), ""),
        )
        monkeypatch.setattr("glider.vision.video_source.video_resolution", lambda _p: None)
        warned = []
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QMessageBox.warning",
            lambda *a, **k: warned.append(a[-1]),
        )
        win._resolution_from_video()
        assert warned and win._view.resolution is None

    def test_cancelling_the_video_picker_changes_nothing(self, qtbot, tmp_path, monkeypatch):
        win = self._win(qtbot, tmp_path, with_resolution=False)
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getOpenFileName",
            lambda *a, **k: ("", ""),
        )
        win._resolution_from_video()
        assert win._view.resolution is None

    def test_a_bad_file_is_reported_not_raised(self, qtbot, tmp_path, monkeypatch):
        shown = []
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QMessageBox.critical",
            lambda *a, **k: shown.append(a[-1]),
        )
        win = AnalysisWindow()
        qtbot.addWidget(win)
        bad = tmp_path / "nope.csv"
        bad.write_text("a,b\n1,2\n")
        win.load(bad)
        assert shown and win._view is None


class TestWhenPosesAreElsewhere:
    """Reusing tracked poses writes none into the output folder, so the
    window has to say what to do rather than just draw nothing."""

    def _session_without_poses(self, tmp_path):
        folder = tmp_path / "outputs" / "t4"
        folder.mkdir(parents=True)
        pd.DataFrame({"frame": range(30), "behavior": ["groom"] * 30}).to_csv(
            folder / "ethogram_raw.csv", index=False
        )
        return folder / "ethogram_raw.csv"

    def test_the_picker_appears_only_when_poses_are_missing(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(self._session_without_poses(tmp_path))
        assert win._pick_poses.isHidden() is False

        win.load(_session(tmp_path / "with_poses"))
        assert win._pick_poses.isHidden() is True

    def test_the_blank_canvas_points_at_the_button(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(self._session_without_poses(tmp_path))
        why = win._canvas._why_blank()
        assert "Choose pose CSV" in why
        assert "run.json" in why  # says where it looked, not just that it failed

    def test_choosing_a_csv_loads_it(self, qtbot, tmp_path, monkeypatch):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        etho = self._session_without_poses(tmp_path)
        win.load(etho)
        assert win._view.xy is None

        # A pose CSV somewhere the search would never look.
        elsewhere = tmp_path / "somewhere" / "else"
        elsewhere.mkdir(parents=True)
        from glider.vision.pose.core import PoseData
        from glider.vision.pose.dlc import to_dlc_csv

        chosen = elsewhere / "whatever.csv"
        to_dlc_csv(
            PoseData(
                xy=np.zeros((30, len(NAMES), 2)),
                confidence=np.ones((30, len(NAMES))),
                keypoint_names=NAMES,
                fps=30.0,
            ),
            chosen,
        )
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getOpenFileName",
            lambda *a, **k: (str(chosen), ""),
        )
        win._choose_pose_csv()
        assert win._view.xy is not None
        assert win._view.pose_path == chosen
        assert win._pick_poses.isHidden() is True

    def test_the_summary_names_the_pose_file_that_was_used(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        assert "vDLC_exp-7.csv" in win._session_text.text()

    def test_cancelling_the_picker_changes_nothing(self, qtbot, tmp_path, monkeypatch):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(self._session_without_poses(tmp_path))
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getOpenFileName",
            lambda *a, **k: ("", ""),
        )
        win._choose_pose_csv()
        assert win._view.xy is None


def _key(win, key, modifier=Qt.KeyboardModifier.NoModifier):
    from PyQt6.QtGui import QKeyEvent

    win.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, key, modifier))


class TestKeyboardScrubbing:
    """One pixel of a 45,000-frame bar is tens of frames, so a bout boundary
    cannot be found with the mouse at all."""

    def _win(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        win._set_frame(150)
        return win

    def test_right_advances_exactly_one_frame(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_Right)
        assert win._frame == 151

    def test_left_steps_back_one_frame(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_Left)
        assert win._frame == 149

    def test_shift_steps_ten(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
        assert win._frame == 160

    def test_ctrl_steps_one_second(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
        assert win._frame == 180  # 30 fps

    def test_stepping_past_the_end_holds_there(self, qtbot, tmp_path):
        """Wrapping to the start would read as a glitch."""
        win = self._win(qtbot, tmp_path)
        win._set_frame(299)
        _key(win, Qt.Key.Key_Right)
        assert win._frame == 299

    def test_stepping_before_the_start_holds_at_zero(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._set_frame(0)
        _key(win, Qt.Key.Key_Left)
        assert win._frame == 0

    def test_home_and_end_jump_to_the_edges(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_End)
        assert win._frame == 299
        _key(win, Qt.Key.Key_Home)
        assert win._frame == 0

    def test_space_toggles_playback(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_Space)
        assert win._timer.isActive() is True
        _key(win, Qt.Key.Key_Space)
        assert win._timer.isActive() is False

    def test_stepping_stops_playback(self, qtbot, tmp_path):
        """Nudging a frame while playing should leave you on that frame."""
        win = self._win(qtbot, tmp_path)
        win._toggle_play()
        _key(win, Qt.Key.Key_Right)
        assert win._timer.isActive() is False
        assert "Play" in win._play.text()

    def test_keys_are_inert_before_a_session_is_loaded(self, qtbot):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        _key(win, Qt.Key.Key_Right)  # must not raise
        assert win._frame == 0


class TestTheTimelineHasOneLane:
    """Freezing and darting are values of the behaviour, not a parallel
    track, so a second lane would draw the same frames twice."""

    def _session_with_freezing(self, tmp_path, n=300):
        folder = tmp_path / "sp"
        folder.mkdir(parents=True, exist_ok=True)
        labels = ["groom"] * n
        for i in range(100, 160):
            labels[i] = "freezing"
        pd.DataFrame(
            {
                "frame": range(n),
                "behavior": labels,
                "speed_px_frame": [0.4] * n,
                "speed_cm_s": [1.2] * n,
            }
        ).to_csv(folder / "ethogram_raw.csv", index=False)
        return folder / "ethogram_raw.csv"

    def _timeline_view(self, qtbot, tmp_path):
        view = TimelineView()
        qtbot.addWidget(view)
        view.resize(HEADER_W + 300, 120)
        view.set_view(SessionView.load(self._session_with_freezing(tmp_path)))
        return view

    def test_the_ethogram_is_one_lane(self, qtbot, tmp_path):
        view = self._timeline_view(qtbot, tmp_path)
        assert [row.kind for row in view._rows()] == ["group", "behavior"]

    def test_freezing_is_drawn_in_its_own_colour(self, qtbot, tmp_path):
        view = self._timeline_view(qtbot, tmp_path)
        y = int(view.lane_rect("ethogram").top() + 5)
        image = view.grab().toImage()
        assert image.pixelColor(int(view.x_of_frame(130)), y) != image.pixelColor(
            int(view.x_of_frame(20)), y
        )

    def test_freeze_bouts_reach_the_one_table(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(self._session_with_freezing(tmp_path))
        win._bar.set_selection(0, 299)
        states = {win._bouts.item(r, 0).text() for r in range(win._bouts.rowCount())}
        assert states == {"groom", "freezing"}


class TestVideoPlayback:
    def test_the_video_toggle_is_disabled_without_one(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        assert win._video_on.isEnabled() is False

    def test_a_video_enables_the_toggle_and_is_named(self, qtbot, tmp_path):
        import cv2

        folder = tmp_path / "v"
        etho = _session(folder)
        clip = folder / "v.mp4"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
        for i in range(300):
            writer.write(np.full((48, 64, 3), i % 255, dtype=np.uint8))
        writer.release()

        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(etho)
        assert win._view.video_path == clip
        assert win._video_on.isEnabled() is True
        assert "v.mp4" in win._session_text.text()
        # And it actually decodes.
        assert win._canvas._frame_image(10) is not None
        win._canvas._close_reader()


class TestACohortOfSessions:
    """A cohort is the unit of analysis, not a session — the question is
    almost always what thirty animals did over the same stretch."""

    def _cohort(self, tmp_path, n=4, rows=300):
        ethograms = []
        for i in range(n):
            folder = tmp_path / "outputs" / f"t{i}"
            folder.mkdir(parents=True)
            # Different behaviour per animal, same length.
            labels = (["groom"] * (50 + 20 * i) + ["locomote"] * rows)[:rows]
            for f in range(100, 100 + 30 * (i + 1)):
                labels[f] = "freezing"
            pd.DataFrame(
                {
                    "frame": range(rows),
                    "behavior": labels,
                    "speed_px_frame": [0.4] * rows,
                    "speed_cm_s": [1.2] * rows,
                }
            ).to_csv(folder / "ethogram_raw.csv", index=False)
            ethograms.append(folder / "ethogram_raw.csv")
        return ethograms

    def test_many_sessions_load_at_once(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        assert len(win._cohort) == 4
        assert win._pool.count() == 4

    def test_a_single_session_is_one_row(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        assert win._pool.count() == 1

    def test_switching_session_keeps_the_window(self, qtbot, tmp_path):
        """The window is the question; changing which animal answers it must
        not silently reset it."""
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._bar.set_selection(100, 199)
        win._pool.select(2)
        assert win._bar.selection() == (100, 199)
        assert win._view is win._cohort[2][1]

    def test_the_same_window_is_applied_to_every_session(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        rows = win.cohort_rows(100, 199)
        assert len(rows) == 4
        assert {r["session"] for r in rows} == {"t0", "t1", "t2", "t3"}
        assert all(r["duration_s"] == pytest.approx(100 / 30.0) for r in rows)

    def test_freezing_is_reported_per_session(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        rows = {r["session"]: r for r in win.cohort_rows(0, 299)}
        # Animal i freezes for 30*(i+1) frames.
        assert rows["t0"]["freezing_s"] == pytest.approx(30 / 30.0)
        assert rows["t3"]["freezing_s"] == pytest.approx(120 / 30.0)

    def test_the_cohort_table_fills_on_selection(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._tables.setCurrentWidget(win._cohort_table)  # filled only while shown
        win._bar.set_selection(0, 299)
        assert win._cohort_table.rowCount() == 4
        assert "Cohort (4)" == win._tables.tabText(1)

    def test_one_unreadable_session_does_not_lose_the_rest(self, qtbot, tmp_path, monkeypatch):
        ethograms = self._cohort(tmp_path)
        bad = tmp_path / "outputs" / "broken" / "ethogram_raw.csv"
        bad.parent.mkdir(parents=True)
        bad.write_text("a,b\n1,2\n")
        warned = []
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QMessageBox.warning",
            lambda *a, **k: warned.append(a[-1]),
        )
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many([*ethograms, bad])
        assert len(win._cohort) == 4
        assert warned and "broken" in warned[0]

    def test_export_writes_a_row_per_session(self, qtbot, tmp_path, monkeypatch):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._bar.set_selection(0, 299)
        out = tmp_path / "window_summary.csv"
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), ""),
        )
        win._export_window()
        written = pd.read_csv(out)
        assert len(written) == 4
        assert {"session", "start_frame", "end_frame", "freezing_s"} <= set(written.columns)
        assert (written["start_frame"] == 0).all()

    def test_export_is_disabled_until_a_window_is_chosen(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        assert win._export_btn.isEnabled() is False
        win._bar.set_selection(0, 99)
        assert win._export_btn.isEnabled() is True


class TestClearingTheRangeClearsItsNumbers:
    def test_clearing_empties_every_range_table(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        win._bar.set_selection(100, 199)
        assert win._bouts.rowCount() == 1
        win._bar.clear_selection()
        assert win._bouts.rowCount() == 0
        assert win._zone_table.rowCount() == 0
        assert win._cohort_table.rowCount() == 0
        assert win._tables.tabText(win._tables.indexOf(win._cohort_table)) == "Cohort"

    def test_a_new_file_does_not_show_the_old_files_bouts(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "a"))
        win._bar.set_selection(100, 199)
        win.load(_session(tmp_path / "b"))
        assert win._bouts.rowCount() == 0


class TestZonesInTheWindow:
    """The spatial suite existed but could not be reached from a video-derived
    session at all — no time in zone, no entries, no heatmap."""

    def _zones(self):
        from glider.vision.zones import Zone, ZoneConfiguration, ZoneShape

        config = ZoneConfiguration()
        config.add_zone(
            Zone(
                id="centre",
                name="centre",
                shape=ZoneShape.RECTANGLE,
                vertices=[(0.25, 0.25), (0.75, 0.75)],
            )
        )
        return config

    def _win(self, qtbot, tmp_path, with_zones=True):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        if with_zones:
            win._zones = self._zones()
            win._canvas.set_zones(win._zones)
        return win

    def test_the_zone_table_fills_on_selection(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(0, 299)
        assert win._zone_table.rowCount() >= 1
        assert "ZONES (" in win._inspector.zones_title.text()

    def test_without_zones_the_table_stays_empty(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path, with_zones=False)
        win._bar.set_selection(0, 299)
        assert win._zone_table.rowCount() == 0
        assert win._inspector.zones_title.text() == "ZONES"

    def test_zone_columns_reach_the_cohort_export(self, qtbot, tmp_path, monkeypatch):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(0, 299)
        out = tmp_path / "window.csv"
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), ""),
        )
        win._export_window()
        written = pd.read_csv(out)
        zone_cols = [c for c in written.columns if c.startswith("zone_")]
        assert any(c.endswith("_s") for c in zone_cols)
        assert any(c.endswith("_entries") for c in zone_cols)
        assert any(c.endswith("_latency_s") for c in zone_cols)

    def test_window_summary_carries_the_panel_numbers(self, qtbot, tmp_path):
        """Everything the Selected-window panel shows should reach the CSV.

        The window-derived thresholds are not the applied ones: the panel calls
        them what this window alone would give, and they are what a reader needs
        to judge whether the loaded thresholds suited this stretch.
        """
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(0, 299)
        rows = win.cohort_rows(0, 299)

        assert rows, "no cohort rows to check"
        row = rows[0]
        for column in (
            "duration_s",
            "duration_min",
            "distance_cm",
            "mean_cm_s",
            "peak_cm_s",
            "window_freeze_threshold",
            "window_dart_threshold",
            "window_threshold_unit",
        ):
            assert column in row, f"missing column: {column}"

        assert row["duration_min"] == pytest.approx(row["duration_s"] / 60.0)
        # unit-neutral on purpose: an uncalibrated session reports px/frame
        assert row["window_threshold_unit"] in ("cm/s", "px/frame", "")

    def test_the_panel_numbers_reach_the_written_csv(self, qtbot, tmp_path, monkeypatch):
        """The dict is not the deliverable — the file on disk is."""
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(0, 299)
        out = tmp_path / "window.csv"
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), ""),
        )
        win._export_window()
        written = pd.read_csv(out)
        for column in (
            "duration_min",
            "window_freeze_threshold",
            "window_dart_threshold",
            "window_threshold_unit",
        ):
            assert column in written.columns, f"missing column: {column}"

    def test_the_heatmap_is_off_until_asked_for(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(0, 299)
        assert win._canvas._heatmap is None

    def test_the_heatmap_appears_for_the_selected_window(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        assert win._canvas._heatmap is not None

    def test_turning_it_off_clears_the_overlay(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        win._heatmap_on.setChecked(False)
        assert win._canvas._heatmap is None

    def test_has_heatmap_follows_the_drawn_overlay(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        assert win._canvas.has_heatmap() is False
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        assert win._canvas.has_heatmap() is True
        win._heatmap_on.setChecked(False)
        assert win._canvas.has_heatmap() is False

    def test_export_button_is_disabled_until_a_heatmap_is_drawn(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        assert win._export_heatmap_btn.isEnabled() is False
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        assert win._export_heatmap_btn.isEnabled() is True

    def test_turning_the_heatmap_off_disables_the_export(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        win._heatmap_on.setChecked(False)
        assert win._export_heatmap_btn.isEnabled() is False
        assert win._heatmap_grid is None

    def test_a_spatial_error_leaves_the_export_disabled(self, qtbot, tmp_path, monkeypatch):
        """The checkbox stays checked on this path, so the checkbox alone is
        not a safe enable rule."""
        from glider.analysis.behavior import spatial

        win = self._win(qtbot, tmp_path)

        def _boom(*_a, **_k):
            raise spatial.SpatialError("no poses")

        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        assert win._export_heatmap_btn.isEnabled() is True  # a real grid first

        monkeypatch.setattr(spatial, "occupancy_grid", _boom)
        win._bar.set_selection(0, 199)  # re-fire; the checkbox stays checked

        assert win._heatmap_on.isChecked() is True
        assert win._heatmap_grid is None
        assert win._export_heatmap_btn.isEnabled() is False

    def test_loading_another_session_clears_the_heatmap(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        assert win._export_heatmap_btn.isEnabled() is True

        second = _session(tmp_path / "second")
        win.load(second)

        assert win._canvas.has_heatmap() is False
        assert win._heatmap_grid is None
        assert win._export_heatmap_btn.isEnabled() is False

    def test_exporting_writes_a_png_and_a_csv(self, qtbot, tmp_path, monkeypatch):
        from glider.gui.behavior import analysis_window as mod

        win = self._win(qtbot, tmp_path)
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)

        target = tmp_path / "out" / "picked.png"
        target.parent.mkdir()
        monkeypatch.setattr(mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))

        win._export_heatmap()

        assert target.exists()
        csv_path = target.with_suffix(".csv")
        assert csv_path.exists()

        # The file must be the grid the overlay was drawn from, not a fresh
        # computation: same shape, same values, same orientation (rows y,
        # columns x). Recomputing at a different bin count would pass a
        # mere existence check.
        grid = win._heatmap_grid[0]
        table = pd.read_csv(csv_path, index_col=0)
        assert table.shape == (grid.shape[1], grid.shape[0])
        assert table.to_numpy().tolist() == grid.T.astype(int).tolist()

    def test_exporting_is_a_no_op_when_cancelled(self, qtbot, tmp_path, monkeypatch):
        from glider.gui.behavior import analysis_window as mod

        win = self._win(qtbot, tmp_path)
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        monkeypatch.setattr(mod.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))

        win._export_heatmap()  # must not raise

    def test_the_canvas_paints_with_zones_and_heatmap(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 299)
        # The canvas is laid out by the window, so pin that it paints rather
        # than what size the layout gave it.
        image = win._canvas.grab().toImage()
        assert image.width() > 0 and image.height() > 0

    def test_a_session_without_poses_does_not_break_zones(self, qtbot, tmp_path):
        folder = tmp_path / "nop"
        folder.mkdir()
        pd.DataFrame({"frame": range(60), "behavior": ["groom"] * 60}).to_csv(
            folder / "ethogram_raw.csv", index=False
        )
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(folder / "ethogram_raw.csv")
        win._zones = self._zones()
        win._heatmap_on.setChecked(True)
        win._bar.set_selection(0, 59)  # must not raise
        assert win._zone_table.rowCount() == 0


def _windowed_session(tmp_path, first=3600, n=9000):
    """An ethogram that starts partway into its video, as a windowed run does."""
    folder = tmp_path / "outputs" / "t9"
    folder.mkdir(parents=True)
    pd.DataFrame({"frame": range(first, first + n), "behavior": ["groom"] * n}).to_csv(
        folder / "ethogram_raw.csv", index=False
    )
    return folder / "ethogram_raw.csv"


class TestTheTimelineCoversTheEthogram:
    """A run that scored minutes 2-7 has no frames before 3600, and a
    timeline drawn from zero spends its first eighth showing nothing."""

    def test_the_bar_starts_where_the_ethogram_starts(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_windowed_session(tmp_path))
        assert win._bar.frame_bounds() == (3600, 12599)

    def test_the_playhead_opens_on_the_first_scored_frame(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_windowed_session(tmp_path))
        assert win._frame == 3600

    def _timeline_view(self, qtbot, tmp_path):
        view = TimelineView()
        qtbot.addWidget(view)
        view.resize(HEADER_W + 300, 120)
        view.set_view(SessionView.load(_windowed_session(tmp_path)))
        return view

    def test_clicking_the_far_left_lands_on_the_first_frame(self, qtbot, tmp_path):
        view = self._timeline_view(qtbot, tmp_path)
        assert view.frame_at_x(HEADER_W) == 3600
        # 9000 frames across 300 px is 30 frames per pixel.
        assert 12570 <= view.frame_at_x(HEADER_W + 299.9) <= 12599

    def test_the_scored_range_fills_the_width(self, qtbot, tmp_path):
        view = self._timeline_view(qtbot, tmp_path)
        assert view.x_of_frame(3600) == pytest.approx(HEADER_W)
        assert view.x_of_frame(12599) == pytest.approx(HEADER_W + 300.0, abs=0.5)

    def test_stepping_left_holds_at_the_first_scored_frame(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_windowed_session(tmp_path))
        _key(win, Qt.Key.Key_Left)
        assert win._frame == 3600

    def test_playback_stops_at_the_last_scored_frame(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_windowed_session(tmp_path))
        win._set_frame(12599)
        win._toggle_play()
        win._advance()
        assert win._timer.isActive() is False

    def test_a_whole_session_is_unaffected(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        assert win._bar.frame_bounds() == (0, 299)
        assert win._frame == 0


class TestSwitchingSessionsIsCheap:
    """Recomputing thirty sessions to redraw a table that did not change made
    flicking between animals feel like the app had hung."""

    def _cohort(self, tmp_path, n=4, rows=300):
        out = []
        for i in range(n):
            folder = tmp_path / "outputs" / f"t{i}"
            folder.mkdir(parents=True)
            pd.DataFrame({"frame": range(rows), "behavior": ["groom"] * rows}).to_csv(
                folder / "ethogram_raw.csv", index=False
            )
            out.append(folder / "ethogram_raw.csv")
        return out

    def test_the_cohort_table_is_computed_once_per_window(self, qtbot, tmp_path, monkeypatch):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))

        calls = []
        original = SessionView.segment_stats
        monkeypatch.setattr(
            SessionView,
            "segment_stats",
            lambda self, *a, **k: (calls.append(1), original(self, *a, **k))[1],
        )
        win._bar.set_selection(0, 299)
        after_first = len(calls)
        win._pool.select(2)
        win._pool.select(3)
        # Switching costs the shown session only, not the cohort again.
        assert len(calls) - after_first <= 2

    def test_changing_the_window_does_recompute(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._bar.set_selection(0, 299)
        first = win.cohort_rows(0, 299)
        assert win.cohort_rows(0, 299) is first  # cached
        assert win.cohort_rows(0, 199) is not first  # different window

    def test_loading_a_new_cohort_drops_the_cache(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path, n=4))
        win._bar.set_selection(0, 299)
        win.cohort_rows(0, 299)
        win.load_many(self._cohort(tmp_path / "second", n=2))
        assert len(win.cohort_rows(0, 299)) == 2


class TestColoursAreActuallyDistinct:
    """Two behaviours in one session must never be drawn the same.

    The palette slot used to come from a hash of the name, which has no reason
    to avoid collisions and routinely put neighbouring behaviours on adjacent
    hues. Given the session's own label set, the first N palette entries are
    handed out in order instead.
    """

    def test_every_behaviour_in_a_session_gets_its_own_colour(self):
        from glider.gui.behavior.analysis_window import behavior_order, behavior_qcolor

        names = ["darting", "dig", "freezing", "grooming", "rearing", "sniffing"]
        order = behavior_order(names)
        colours = [behavior_qcolor(n, order).name() for n in order]
        assert len(set(colours)) == len(names)

    def test_the_hash_fallback_could_collide_and_the_order_fixes_it(self):
        """Not hypothetical: anagrams hash identically."""
        from glider.gui.behavior.analysis_window import behavior_qcolor

        assert behavior_qcolor("stop").name() == behavior_qcolor("post").name()
        order = ["post", "stop"]
        assert behavior_qcolor("stop", order).name() != behavior_qcolor("post", order).name()

    def test_the_order_does_not_depend_on_which_animal_came_first(self):
        from glider.gui.behavior.analysis_window import behavior_order

        assert behavior_order(["b", "a", "b"]) == behavior_order(["a", "b", "a"])

    def test_unscored_frames_are_not_given_a_behaviour_colour(self):
        from glider.gui.behavior.analysis_window import behavior_qcolor
        from glider.gui.styles import colors

        assert behavior_qcolor("", ["a"]).name() == QColor(colors.BORDER).name()

    def test_the_palette_has_no_duplicates(self):
        from glider.analysis.behavior.vocabulary import DEFAULT_PALETTE

        assert len(set(DEFAULT_PALETTE)) == len(DEFAULT_PALETTE)

    def test_the_overlay_and_the_vocabulary_cannot_drift(self):
        """One palette, converted — not two lists kept in step by hand."""
        from glider.analysis.behavior.classify.overlay import _PALETTE_BGR
        from glider.analysis.behavior.vocabulary import DEFAULT_PALETTE

        assert len(_PALETTE_BGR) == len(DEFAULT_PALETTE)
        b, g, r = _PALETTE_BGR[0]
        assert f"#{r:02x}{g:02x}{b:02x}" == DEFAULT_PALETTE[0]


class TestSteppingBetweenBoutsInTheWindow:
    """`[` and `]` move to bout boundaries, which is what review consists of."""

    def _window(self, qtbot, tmp_path):
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_session(tmp_path / "t1"))
        return window

    def test_the_picker_lists_the_session_s_behaviours(self, qtbot, tmp_path):
        window = self._window(qtbot, tmp_path)
        items = [window._bout_filter.itemText(i) for i in range(window._bout_filter.count())]
        assert items == ["Any change", "groom", "locomote"]

    def test_stepping_forward_lands_on_the_next_boundary(self, qtbot, tmp_path):
        window = self._window(qtbot, tmp_path)
        window._set_frame(40)
        window._step_bout(+1)
        assert window._frame == 100  # groom -> locomote

    def test_stepping_back_lands_on_the_boundary_behind(self, qtbot, tmp_path):
        window = self._window(qtbot, tmp_path)
        window._set_frame(250)
        window._step_bout(-1)
        assert window._frame == 200

    def test_a_chosen_behaviour_skips_the_others(self, qtbot, tmp_path):
        window = self._window(qtbot, tmp_path)
        window._bout_filter.setCurrentIndex(window._bout_filter.findData("groom"))
        window._set_frame(50)
        window._step_bout(+1)
        assert window._frame == 200  # the second groom bout, not the locomote start

    def test_the_last_bout_holds_rather_than_wrapping(self, qtbot, tmp_path):
        """Wrapping to the top of the session while reviewing the end is a trap."""
        window = self._window(qtbot, tmp_path)
        window._set_frame(280)
        window._step_bout(+1)
        assert window._frame == 200

    def test_the_readout_names_the_bout_and_how_far_into_it(self, qtbot, tmp_path):
        window = self._window(qtbot, tmp_path)
        window._set_frame(115)
        text = window._bout_label.text()
        assert "locomote" in text
        assert "3.33 s" in text  # 100 frames at 30 fps
        assert "0.50 s in" in text  # 15 frames past the start

    def test_the_brackets_are_wired_to_the_stepper(self, qtbot, tmp_path):
        window = self._window(qtbot, tmp_path)
        window._set_frame(40)
        window.keyPressEvent(
            QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_BracketRight, Qt.KeyboardModifier(0))
        )
        assert window._frame == 100


class TestTheCohortTabShowsTheAppliedThresholds:
    """The cut-offs that produced the freezing and darting columns beside them.

    A cohort file is pooled in px/frame and only becomes cm/s through each
    video's own scale, so this is per session — and it is the only place the
    number a methods section has to quote actually appears.
    """

    def _session_with_run(self, tmp_path, name, manifest):
        import json

        path = _session(tmp_path / name)
        (path.parent / "run.json").write_text(json.dumps({"schema_version": 1, **manifest}))
        return path

    def test_the_cut_offs_appear_in_the_rows(self, qtbot, tmp_path):
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(
            self._session_with_run(
                tmp_path,
                "t1",
                {"freeze_threshold": 0.25, "dart_threshold": 12.0, "cm_s_per_px_frame": 2.0},
            )
        )
        row = window.cohort_rows(0, 299)[0]
        assert row["freeze_threshold_cm_s"] == pytest.approx(0.5)
        assert row["dart_threshold_cm_s"] == pytest.approx(24.0)

    def test_the_table_shows_them_in_cm_per_second(self, qtbot, tmp_path):
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(
            self._session_with_run(
                tmp_path,
                "t1",
                {"freeze_threshold": 0.25, "dart_threshold": 12.0, "cm_s_per_px_frame": 2.0},
            )
        )
        window._tables.setCurrentWidget(window._cohort_table)  # filled only while shown
        window._select_all()
        headers = [
            window._cohort_table.horizontalHeaderItem(c).text()
            for c in range(window._cohort_table.columnCount())
        ]
        assert "Freeze < (cm/s)" in headers
        assert window._cohort_table.item(0, headers.index("Freeze < (cm/s)")).text() == "0.50"

    def test_an_uncalibrated_run_shows_pixels_rather_than_nothing(self, qtbot, tmp_path):
        """It had real thresholds; it just never had a scale."""
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(
            self._session_with_run(
                tmp_path, "t1", {"freeze_threshold": 0.25, "dart_threshold": 12.0}
            )
        )
        assert window._threshold_text(window.cohort_rows(0, 299)[0], "freeze") == "0.250 px/f"

    def test_a_run_with_no_manifest_shows_a_dash(self, qtbot, tmp_path):
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_session(tmp_path / "t1"))
        assert window._threshold_text(window.cohort_rows(0, 299)[0], "dart") == "—"


def _recording(folder, *, n=300, fps=30.0, states=("resting", "active"), flow_start=0):
    """A GLIDER recording directory: tracking + events, one LED on a pin.

    Written here rather than imported from ``tests/unit/analysis/conftest.py``
    — pytest runs this repository in importlib mode, which deliberately does
    not let one test module import another. Only the columns the timeline
    reads are present.
    """
    from datetime import datetime, timedelta

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 5, 25, 14, 0, 30)

    def stamp(frame):
        return (base + timedelta(seconds=frame / fps)).isoformat(timespec="milliseconds")

    with open(folder / "rec_tracking.csv", "w", encoding="utf-8") as f:
        f.write("# GLIDER Tracking Data\n\n")
        f.write("frame,timestamp,elapsed_ms,object_id,behavioral_state\n")
        for i in range(n):
            state = states[0] if i < n // 2 else states[1]
            f.write(f"{i},{stamp(i)},{i / fps * 1000:.1f},0,{state}\n")

    with open(folder / "rec_events.csv", "w", encoding="utf-8") as f:
        f.write("# GLIDER Device Event Log\n\n")
        f.write(
            "frame,timestamp,elapsed_ms,source,board_id,device_id,device_type,"
            "pin,pin_type,value\n"
        )
        rows = [
            (flow_start, "flow_marker", "", "", "", "", "", "start"),
            (30, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "1"),
            (150, "output_write", "board0", "led1", "LED", "5", "DIGITAL", "0"),
            (n - 1, "flow_marker", "", "", "", "", "", "end"),
        ]
        for frame, source, board, device, kind, pin, pin_type, value in rows:
            f.write(
                f"{frame},{stamp(frame)},{frame / fps * 1000:.1f},{source},"
                f"{board},{device},{kind},{pin},{pin_type},{value}\n"
            )
    return folder


def _ethogram(folder, labels):
    """An ethogram CSV alone, the way an apply run writes one."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": range(len(labels)), "behavior": labels}).to_csv(
        folder / "ethogram_raw.csv", index=False
    )
    return folder / "ethogram_raw.csv"


class TestFindingTheRecordingTheEthogramCameFrom:
    """The raster is only ever seen if the recording folder is found.

    An apply run writes ``<output>/<video stem>/ethogram_raw.csv``, so the
    ethogram's own folder is one level BELOW the recording, and the canonical
    layout buries it one level below that. Loading ``ethogram_csv.parent`` as
    the recording therefore found nothing in either real layout: discovery saw
    no ``# GLIDER …`` markers, ``build_timeline`` got an empty session, and no
    hardware lane was ever drawn by any GLIDER path at all.
    """

    def _lane_keys(self, window):
        timeline = window._bar._timeline
        return [] if timeline is None else [lane.key for lane in timeline.lanes]

    def test_an_apply_run_layout_finds_its_recording(self, qtbot, tmp_path):
        """Ethogram in a per-video SUBFOLDER of the recording directory.

        This is the shape that shipped broken; an ethogram dropped beside the
        CSVs proves nothing, because no GLIDER path produces that.
        """
        recording = _recording(tmp_path / "rec")
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_ethogram(recording / "v", ["groom"] * 300))
        assert self._lane_keys(window) == ["led1"]

    def test_the_canonical_layout_finds_its_recording(self, qtbot, tmp_path):
        """Ethogram under ``sessions/<id>/analysis/``, CSVs at ``sessions/<id>/``."""
        session = _recording(tmp_path / "sessions" / "s1")
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_ethogram(session / "analysis", ["groom"] * 300))
        assert self._lane_keys(window) == ["led1"]

    def test_the_found_lanes_are_actually_given_room_to_draw(self, qtbot, tmp_path):
        recording = _recording(tmp_path / "rec")
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_ethogram(recording / "v", ["groom"] * 300))
        bar = window._bar
        bar.resize(HEADER_W + 300, bar.sizeHint().height())
        heights = [row.height for row in bar._rows() if row.kind == "hardware"]
        assert heights and min(heights) > 0

    def test_an_ethogram_with_no_recording_anywhere_still_loads(self, qtbot, tmp_path):
        """Behaviour lanes, no raster — the honest answer, not an exception."""
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_ethogram(tmp_path / "loose" / "v", ["groom"] * 300))
        assert self._lane_keys(window) == []
        assert window._bar.frame_bounds() == (0, 299)

    def test_a_broken_events_csv_costs_the_raster_and_nothing_else(self, qtbot, tmp_path):
        """A KeyError out of ``build_timeline`` must not reach the GUI.

        The event log is the one artifact a plugin can write, so a missing
        column is a real failure mode — and the whole review window went down
        with it, over a lane nobody had asked for.
        """
        recording = _recording(tmp_path / "rec")
        events = recording / "rec_events.csv"
        text = events.read_text(encoding="utf-8")
        events.write_text(text.replace("board_id,", ""), encoding="utf-8")
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_ethogram(recording / "v", ["groom"] * 300))
        assert self._lane_keys(window) == []

    def test_the_recording_is_parsed_once_per_cohort_not_once_per_click(
        self, qtbot, tmp_path, monkeypatch
    ):
        """Stepping through a cohort re-adopts a session on every click.

        The CSVs are megabytes and they are parsed on the GUI thread, so a
        cohort sharing one recording must read it once, not once per click.
        """
        import glider.gui.behavior.analysis_window as module

        recording = _recording(tmp_path / "rec")
        parsed = []
        real = module._recording_or_none

        def counting(folder):
            parsed.append(folder)
            return real(folder)

        monkeypatch.setattr(module, "_recording_or_none", counting)

        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load_many(
            [
                _ethogram(recording / "v1", ["groom"] * 300),
                _ethogram(recording / "v2", ["locomote"] * 300),
            ]
        )
        window._show_session(1)
        window._show_session(0)
        assert self._lane_keys(window) == ["led1"]
        assert parsed.count(recording.resolve()) == 1


class TestTheTableAgreesWithTheBar:
    """One behaviour, one colour — in the bar and in the bout table.

    The bar pools labels across every behaviour lane to build its order, so a
    session with both an ethogram and a tracking lane gives a behaviour a
    different palette slot than the ethogram's labels alone would. The table
    recomputed the ethogram-only order, and a bout then wore one colour in the
    legend and another in the stripe directly above it.
    """

    def _chip_colour(self, window, row=0):
        icon = window._bouts.item(row, 0).icon()
        return icon.pixmap(10, 10).toImage().pixelColor(5, 5)

    def test_a_bouts_chip_is_the_colour_the_bar_paints(self, qtbot, tmp_path):
        recording = _recording(tmp_path / "rec")
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_ethogram(recording / "v", ["dart"] * 300))

        bar = window._bar
        bar.resize(600, bar.sizeHint().height())
        window._select_all()

        assert window._bouts.item(0, 0).text() == "dart"
        chip = self._chip_colour(window)
        # The ethogram lane is the top row; frame 150 is well inside it.
        y = int(bar.lane_rect("ethogram").top() + 5)
        painted = bar.grab().toImage().pixelColor(int(bar.x_of_frame(150)), y)
        assert painted == chip

        # And the tracking lane really is shifting the order, so the equality
        # above is not two identical computations agreeing by accident.
        assert behavior_qcolor("dart", ["dart"]) != chip

    def test_the_bout_picker_offers_the_bar_s_behaviours(self, qtbot, tmp_path):
        recording = _recording(tmp_path / "rec")
        window = AnalysisWindow()
        qtbot.addWidget(window)
        window.load(_ethogram(recording / "v", ["dart"] * 300))
        offered = [window._bout_filter.itemData(i) for i in range(window._bout_filter.count())]
        assert offered[0] is None  # "Any change"
        assert offered[1:] == window._bar.behavior_order()


class TestOpeningARecording:
    """A rig run opens without anyone having scored it."""

    def test_a_recording_folder_opens_with_its_lanes(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(_recording(tmp_path / "rec"))
        assert win._pool.count() == 1
        assert [lane.key for lane in win._bar.timeline().lanes] == ["led1"]
        assert win._view.labels[0] == "resting"

    def test_selecting_a_range_reports_the_hardware(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(_recording(tmp_path / "rec"))
        win._select_all()
        assert win._inspector.hardware.rowCount() == 1
        assert win._inspector.hardware.item(0, 1).text() == "4.0 s on"

    def test_a_folder_with_one_ethogram_opens_it(self, qtbot, tmp_path):
        ethogram = _ethogram(tmp_path / "out" / "v", ["groom"] * 300)
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(tmp_path / "out")
        assert win._ethogram_csv == ethogram

    def test_a_folder_with_nothing_says_what_it_looked_for(self, qtbot, tmp_path, monkeypatch):
        said = []
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QMessageBox.critical",
            lambda *a, **k: said.append(a[-1]),
        )
        (tmp_path / "empty").mkdir()
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(tmp_path / "empty")
        assert said and "no GLIDER recording" in said[0]


class TestOpeningACohortFolder:
    def _cohort(self, tmp_path):
        for sid in ("s1", "s2"):
            _ethogram(tmp_path / "sessions" / sid / "analysis", ["groom"] * 300)
        (tmp_path / "glider_project.json").write_text(
            '{"sessions": {"s1": {"group": "ChR2"}, "s2": {"group": "eYFP"}}}',
            encoding="utf-8",
        )

    def test_sessions_are_grouped_by_the_manifest(self, qtbot, tmp_path):
        self._cohort(tmp_path)
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_folder(tmp_path)
        assert win._ids == ["s1", "s2"]
        assert win._groups == ["ChR2", "eYFP"]
        assert win._pool.tree.topLevelItemCount() == 2

    def test_a_broken_manifest_warns_and_still_loads(self, qtbot, tmp_path, monkeypatch):
        self._cohort(tmp_path)
        (tmp_path / "glider_project.json").write_text("{not json", encoding="utf-8")
        warned = []
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QMessageBox.warning",
            lambda *a, **k: warned.append(a[-1]),
        )
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_folder(tmp_path)
        assert win._pool.count() == 2
        assert warned and "ungrouped" in warned[0]


class TestEditingKeys:
    def _win(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        return win

    def test_i_and_o_set_the_range(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._set_frame(50)
        _key(win, Qt.Key.Key_I)
        win._set_frame(120)
        _key(win, Qt.Key.Key_O)
        assert win._bar.selection() == (50, 120)

    def test_x_selects_the_bout_under_the_playhead(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)  # groom 0-99, locomote 100-199, groom 200-299
        win._set_frame(150)
        _key(win, Qt.Key.Key_X)
        assert win._bar.selection() == (100, 199)

    def test_z_zooms_to_the_range_and_shift_z_fits(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(100, 199)
        _key(win, Qt.Key.Key_Z)
        assert win._bar.viewport.span == pytest.approx(100.0)
        _key(win, Qt.Key.Key_Z, Qt.KeyboardModifier.ShiftModifier)
        assert win._bar.viewport.span == pytest.approx(300.0)

    def test_escape_clears_the_range(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(100, 199)
        _key(win, Qt.Key.Key_Escape)
        assert win._bar.selection() is None
        assert win._export_btn.isEnabled() is False

    def test_l_plays_and_doubles(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_L)
        assert win._timer.isActive() and win._rate == 1
        _key(win, Qt.Key.Key_L)
        assert win._rate == 2
        _key(win, Qt.Key.Key_K)
        assert not win._timer.isActive()

    def test_j_plays_backward(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._set_frame(150)
        _key(win, Qt.Key.Key_J)
        win._advance()
        assert win._frame == 149

    def test_cmd_a_selects_the_whole_session(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        assert win._bar.selection() == (0, 299)

    def test_the_sessions_list_does_not_take_keyboard_focus(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        assert win._pool.tree.focusPolicy() == Qt.FocusPolicy.NoFocus


class TestLoopAndFollow:
    def _win(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        return win

    def test_loop_wraps_inside_the_range(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(100, 120)
        win._timeline_panel.loop.setChecked(True)
        win._set_frame(120)
        win._toggle_play()
        win._advance()
        assert win._frame == 100
        assert win._timer.isActive()

    def test_playback_pages_the_timeline(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.viewport.show(0, 50)
        win._set_frame(50)
        win._toggle_play()
        win._advance()
        assert win._bar.viewport.start == pytest.approx(51.0)


class TestTheRangeMenu:
    def test_set_in_here_uses_the_cursor_frame(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        menu = win._timeline_menu(120)
        texts = [a.text() for a in menu.actions() if a.text()]
        assert "Set In here\tI" in texts and "Zoom to range\tZ" in texts
        next(a for a in menu.actions() if a.text() == "Set In here\tI").trigger()
        assert win._bar.selection() == (120, 299)

    def test_range_actions_are_off_without_a_range(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        menu = win._timeline_menu(120)
        zoom = next(a for a in menu.actions() if a.text() == "Zoom to range\tZ")
        assert zoom.isEnabled() is False

    def test_the_select_whole_session_item_is_offered(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        from PyQt6.QtGui import QKeySequence

        # ⌘A on macOS, Ctrl+A elsewhere: the platform's own text for Select All.
        native = QKeySequence(QKeySequence.StandardKey.SelectAll).toString(
            QKeySequence.SequenceFormat.NativeText
        )
        menu = win._timeline_menu(120)
        texts = [a.text() for a in menu.actions() if a.text()]
        assert f"Select whole session\t{native}" in texts
        select_all = next(
            a for a in menu.actions() if a.text() == f"Select whole session\t{native}"
        )
        assert select_all.isEnabled() is True
        select_all.trigger()
        assert win._bar.selection() == (0, 299)


class TestTheHud:
    def test_the_hud_names_the_behaviour(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        win._set_frame(150)
        behavior, _chips = win._canvas._hud
        assert behavior[0].startswith("locomote")

    def test_the_hud_lists_active_outputs(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(_recording(tmp_path / "rec"))
        win._set_frame(60)  # the LED is on from frame 30 to 150
        _behavior, chips = win._canvas._hud
        assert [text for text, _colour in chips] == ["led1"]


class TestHiddenLanesAreRemembered:
    def test_a_hidden_lane_stays_hidden_on_reopen(self, qtbot, tmp_path):
        folder = _recording(tmp_path / "rec")
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(folder)
        win._bar.set_hidden({"led1"})
        again = AnalysisWindow()
        qtbot.addWidget(again)
        again.open_path(folder)
        assert again._bar.hidden() == {"led1"}

    def test_same_named_sessions_in_two_cohorts_do_not_share_hidden_lanes(self, qtbot, tmp_path):
        a = _recording(tmp_path / "cohortA" / "sessions" / "m01")
        b = _recording(tmp_path / "cohortB" / "sessions" / "m01")
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(a)
        win._bar.set_hidden({"led1"})
        other = AnalysisWindow()
        qtbot.addWidget(other)
        other.open_path(b)
        assert other._bar.hidden() == set()


def _centroid_recording(folder, frames, *, with_state=True):
    """A tracking CSV with a centroid, for the paths ``_recording`` does not reach."""
    from datetime import datetime, timedelta

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 5, 25, 14, 0, 30)
    state = ",behavioral_state" if with_state else ""
    with open(folder / "rec_tracking.csv", "w", encoding="utf-8") as f:
        f.write("# GLIDER Tracking Data\n\n")
        f.write(f"frame,timestamp,elapsed_ms,object_id,center_x,center_y{state}\n")
        for i, frame in enumerate(frames):
            stamp = (base + timedelta(seconds=i / 30.0)).isoformat(timespec="milliseconds")
            tail = ",rest" if with_state else ""
            f.write(f"{frame},{stamp},{i / 30.0 * 1000:.1f},0,{100 + i},200{tail}\n")
    return folder


def _said(monkeypatch, kind):
    said = []
    monkeypatch.setattr(
        f"glider.gui.behavior.analysis_window.QMessageBox.{kind}",
        lambda *a, **k: said.append(a[-1]),
    )
    return said


class TestNothingOnDiskAbortsTheWindow:
    """GLIDER installs no excepthook: an exception out of a menu action kills
    the process, and a running experiment with it."""

    def test_negative_frames_are_reported_when_opened(self, qtbot, tmp_path, monkeypatch):
        said = _said(monkeypatch, "critical")
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(_centroid_recording(tmp_path / "rec", range(-10, 0)))
        assert said and "could not be opened" in said[0]
        assert win._view is None

    def test_negative_frames_cost_one_session_of_a_cohort(self, qtbot, tmp_path, monkeypatch):
        warned = _said(monkeypatch, "warning")
        _recording(tmp_path / "good")
        _centroid_recording(tmp_path / "bad", range(-10, 0))
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_folder(tmp_path)
        assert win._ids == ["good"]
        assert warned and "bad" in warned[0]

    def test_an_unreadable_folder_is_reported(self, qtbot, tmp_path, monkeypatch):
        said = _said(monkeypatch, "critical")

        def boom(_root):
            raise PermissionError("denied")

        monkeypatch.setattr("glider.gui.behavior.analysis_window.discover_sessions", boom)
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_folder(tmp_path)
        assert said and "denied" in said[0]


class TestAFolderWithoutTracking:
    """CV off writes no tracking CSV: the folder is an offline apply run's source."""

    def test_its_one_ethogram_opens(self, qtbot, tmp_path, monkeypatch):
        said = _said(monkeypatch, "critical")
        folder = _recording(tmp_path / "rec")
        (folder / "rec_tracking.csv").unlink()
        ethogram = _ethogram(folder / "v", ["groom"] * 300)
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(folder)
        assert said == []
        assert win._ethogram_csv == ethogram

    def test_without_one_it_names_both_problems(self, qtbot, tmp_path, monkeypatch):
        said = _said(monkeypatch, "critical")
        folder = _recording(tmp_path / "rec")
        (folder / "rec_tracking.csv").unlink()
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(folder)
        assert said and "no tracking CSV" in said[0] and "ethogram_raw.csv" in said[0]


class TestReopeningReadsTheDiskAgain:
    def test_a_growing_recording_shows_its_new_rows(self, qtbot, tmp_path):
        folder = _recording(tmp_path / "rec")
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(folder)
        assert win._view.n_rows == 300
        with open(folder / "rec_tracking.csv", "a", encoding="utf-8") as f:
            for i in range(300, 400):
                f.write(f"{i},2026-05-25T14:00:40.000,{i / 30.0 * 1000:.1f},0,active\n")
        win.open_path(folder)
        assert win._view.n_rows == 400


class TestTheCohortTableFillsWhenShown:
    """Filling it costs a pass per animal; a drag must not pay that per mouse move."""

    def _cohort(self, tmp_path):
        return [_ethogram(tmp_path / f"t{i}", ["groom"] * 300) for i in range(3)]

    def test_a_hidden_tab_is_not_filled(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._bar.set_selection(0, 99)
        assert win._cohort_table.rowCount() == 0

    def test_showing_the_tab_fills_it(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._bar.set_selection(0, 99)
        win._tables.setCurrentWidget(win._cohort_table)
        assert win._cohort_table.rowCount() == 3
        assert win._tables.tabText(1) == "Cohort (3)"

    def test_a_new_range_refills_the_shown_tab(self, qtbot, tmp_path, monkeypatch):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._tables.setCurrentWidget(win._cohort_table)
        asked = []
        real = win.cohort_rows
        monkeypatch.setattr(win, "cohort_rows", lambda s, e: asked.append((s, e)) or real(s, e))
        win._bar.set_selection(10, 20)
        assert asked == [(10, 20)]


class TestPlaybackAtSpeed:
    def test_eight_x_lands_on_the_last_frame(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        win._set_frame(295)
        win._toggle_play()
        win._rate = 8
        win._advance()
        assert win._frame == 299
        assert win._timer.isActive() is False


class TestEditKeysLeaveShortcutsAlone:
    def _win(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        return win

    def test_cmd_z_does_not_zoom(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(100, 199)
        _key(win, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        assert win._bar.viewport.span == pytest.approx(300.0)

    def test_an_auto_repeated_l_does_not_double_the_rate(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_L)
        win.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress, Qt.Key.Key_L, Qt.KeyboardModifier.NoModifier, "", True
            )
        )
        assert win._rate == 1


class TestSwitchingCohortsIsAllOrNothing:
    def test_a_failed_build_keeps_the_old_cohort(self, qtbot, tmp_path, monkeypatch):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "a"))
        before = (list(win._cohort), list(win._ids), list(win._groups))

        def boom(*_a):
            raise RuntimeError("boom")

        monkeypatch.setattr(win, "_timeline_for", boom)
        view = SessionView.load(_session(tmp_path / "b"))
        with pytest.raises(RuntimeError):
            win._set_cohort([(tmp_path / "b" / "ethogram_raw.csv", view)])
        assert (win._cohort, win._ids, win._groups) == before


class TestARecordingIsHonestAboutItsPoses:
    def test_a_centroid_is_not_called_poses(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(_centroid_recording(tmp_path / "rec", range(1, 301)))
        assert win._session_state.text() == "Centroid only"

    def test_the_blank_canvas_does_not_offer_a_pose_csv(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(_centroid_recording(tmp_path / "rec", range(1, 301)))
        why = win._canvas._why_blank()  # no calibration, no video: no arena size
        assert "pose CSV" not in why and "Set arena size" not in why

    def test_no_behavioral_state_draws_no_behaviour_lane(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.open_path(_centroid_recording(tmp_path / "rec", range(1, 301), with_state=False))
        assert [row.key for row in win._bar._rows() if row.kind == "group"] == []


class TestTheSelectAllShortcutIsNative:
    def test_the_menu_and_the_shortcuts_list_use_the_platform_text(
        self, qtbot, tmp_path, monkeypatch
    ):
        from PyQt6.QtGui import QKeySequence

        native = QKeySequence(QKeySequence.StandardKey.SelectAll).toString(
            QKeySequence.SequenceFormat.NativeText
        )
        assert native in ("⌘A", "Ctrl+A")
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        texts = [a.text() for a in win._timeline_menu(0).actions()]
        assert f"Select whole session\t{native}" in texts
        shown = _said(monkeypatch, "information")
        win._show_shortcuts()
        assert f"{native}  select the whole session" in shown[0]


class TestTheVideoStaysPut:
    """Text that changes every frame must not resize the panels around the video."""

    def test_the_canvas_does_not_move_as_the_bout_text_changes(self, qtbot, tmp_path):
        labels = ["rest"] * 100 + ["grooming_bilateral_face_wash"] * 100 + ["rear"] * 9800
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.resize(1440, 900)
        win.show()
        qtbot.waitExposed(win)
        win.load(_ethogram(tmp_path / "v", labels))
        qtbot.wait(50)  # a size change reaches the splitter a few event passes later
        geometry = []
        for frame in (5, 150, 9990):
            win._set_frame(frame)
            qtbot.wait(50)
            geometry.append(win._canvas.geometry())
        assert geometry[0] == geometry[1] == geometry[2]


class TestTheTopBarMenus:
    def test_the_menu_buttons_are_one_size(self, qtbot):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.show()
        qtbot.waitExposed(win)
        buttons = (win._open_btn, win._zones_btn, win._export_menu_btn)
        assert len({(b.width(), b.height()) for b in buttons}) == 1
        assert win._tour_btn.height() == win._open_btn.height()


class TestTheRangeIsMeasuredOnEachSessionsOwnZero:
    """Current range is seconds from each animal's own flow start: the time rule.

    Animal b's rig ran a second before flow started, so "1 s to 3 s into the
    protocol" is frames 60-119 for it and 30-89 for animal a.
    """

    def _two(self, qtbot, tmp_path):
        root = tmp_path / "cohort"
        _recording(root / "a", flow_start=0)
        _recording(root / "b", flow_start=30)
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_folder(root)
        win._pool.select(win._ids.index("a"))
        return win

    def test_the_same_seconds_are_different_frames_in_each_animal(self, qtbot, tmp_path):
        win = self._two(qtbot, tmp_path)
        rows = {r["session"]: r for r in win.cohort_rows(30, 89)}
        assert (rows["a"]["start_frame"], rows["a"]["end_frame"]) == (30, 89)
        assert (rows["b"]["start_frame"], rows["b"]["end_frame"]) == (60, 119)
        assert rows["b"]["start_s"] == pytest.approx(1.0)
        assert rows["b"]["t0"] == "flow_start"

    def test_switching_animals_keeps_the_seconds_not_the_frames(self, qtbot, tmp_path):
        win = self._two(qtbot, tmp_path)
        win._bar.set_selection(30, 89)
        win._pool.select(win._ids.index("b"))
        assert win._bar.selection() == (60, 119)

    def test_the_range_stays_put_when_the_shown_animal_changes(self, qtbot, tmp_path):
        """A 15 fps animal cannot show 1.033 s; the range must not become what it can show."""
        root = tmp_path / "rates"
        _recording(root / "a", fps=30.0)
        _recording(root / "c", fps=15.0)
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_folder(root)
        win._pool.select(win._ids.index("a"))
        win._bar.set_selection(31, 89)
        before = win._current_span
        win._pool.select(win._ids.index("c"))
        assert win._current_span == before
        win._pool.select(win._ids.index("a"))
        assert win._bar.selection() == (31, 89)

    def test_a_range_survives_a_hop_through_an_animal_it_does_not_reach(self, qtbot, tmp_path):
        """A no-overlap hop clears the bar's own selection, not just the shown
        animal's -- the next switch must still restore it from `_current_span`,
        not from what the bar happened to show right before the switch."""
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(
            [
                _ethogram(tmp_path / "long", ["groom"] * 300),
                _ethogram(tmp_path / "short", ["groom"] * 150),
            ]
        )
        win._bar.set_selection(240, 299)
        win._pool.select(win._ids.index("short"))  # the range doesn't reach it
        assert win._bar.selection() is None
        win._pool.select(win._ids.index("long"))
        assert win._bar.selection() == (240, 299)

    def test_a_range_given_in_seconds_is_the_selection_given_in_frames(self, qtbot, tmp_path):
        win = self._two(qtbot, tmp_path)
        assert win.range_rows(1.0, 3.0) == win.cohort_rows(30, 89)

    def test_time_on_per_device_joins_the_row(self, qtbot, tmp_path):
        """The LED is on from frame 30 to 150: 1 s to 5 s after a's flow start."""
        win = self._two(qtbot, tmp_path)
        row = next(r for r in win.range_rows(0.0, 3.0) if r["session"] == "a")
        (key,) = [k for k in row if k.startswith("hw_")]
        assert row[key] == pytest.approx(2.0, abs=0.05)

    def test_a_range_past_a_short_session_is_clipped_or_outside(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(
            [
                _ethogram(tmp_path / "long", ["groom"] * 300),
                _ethogram(tmp_path / "short", ["groom"] * 150),
            ]
        )
        rows = {r["session"]: r for r in win.range_rows(8.0, 12.0)}
        assert (rows["long"]["start_frame"], rows["long"]["end_frame"]) == (240, 299)
        assert rows["short"].get("outside") is True

    def test_a_behaviour_one_animal_never_showed_is_zero_seconds_for_it(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(
            [
                _ethogram(tmp_path / "a", ["groom"] * 300),
                _ethogram(tmp_path / "b", ["rear"] * 300),
            ]
        )
        rows = {r["session"]: r for r in win.range_rows(0.0, 5.0)}
        assert rows["a"]["rear_s"] == 0.0 and rows["b"]["groom_s"] == 0.0

    def test_the_export_carries_seconds_and_no_private_columns(self, qtbot, tmp_path, monkeypatch):
        win = self._two(qtbot, tmp_path)
        win._bar.set_selection(30, 89)
        out = tmp_path / "window_summary.csv"
        monkeypatch.setattr(
            "glider.gui.behavior.analysis_window.QFileDialog.getSaveFileName",
            lambda *a, **k: (str(out), ""),
        )
        win._export_window()
        written = pd.read_csv(out)
        assert {"start_s", "end_s", "start_frame", "end_frame", "t0"} <= set(written.columns)
        assert not [c for c in written.columns if c.startswith("_")]


class TestTheCohortTabToleratesOutsideRows:
    """Controller ruling: `_fill_cohort` still renders from `cohort_rows` until
    Task 11 removes it, and an outside row lacks the Phase 1 keys entirely."""

    def test_the_cohort_tab_fills_without_error_past_a_short_session(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(
            [
                _ethogram(tmp_path / "long", ["groom"] * 300),
                _ethogram(tmp_path / "short", ["groom"] * 150),
            ]
        )
        win._tables.setCurrentWidget(win._cohort_table)
        win._bar.set_selection(240, 299)  # the shown (first, "long") session's own frames
        rows = {r["session"]: r for r in win.cohort_rows(240, 299)}
        assert rows["short"].get("outside") is True
        short_row = next(
            r
            for r in range(win._cohort_table.rowCount())
            if win._cohort_table.item(r, 0).text() == "short"
        )
        values = [win._cohort_table.item(short_row, c).text() for c in range(1, 9)]
        assert all(v == "—" for v in values)


class TestMarkersInTheWindow:
    """M, shift-M, up and down, the editor, and the files beside the data."""

    def _win(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        csv = _session(tmp_path / "v")
        win.load(csv)
        return win, csv.parent

    def _saved(self, folder, name="review_markers.json"):
        import json

        return json.loads((folder / name).read_text())

    def test_m_adds_a_point_marker_at_the_playhead_and_saves_it(self, qtbot, tmp_path):
        win, folder = self._win(qtbot, tmp_path)
        win._set_frame(90)
        _key(win, Qt.Key.Key_M)
        win._editor.close()
        saved = self._saved(folder)
        (marker,) = saved["markers"]
        assert marker["kind"] == "point" and marker["start_s"] == pytest.approx(3.0)
        assert saved["t0"] == "video_start"
        assert [m.start_s for m in win._bar.markers()] == [pytest.approx(3.0)]

    def test_reopening_restores_the_markers(self, qtbot, tmp_path):
        win, folder = self._win(qtbot, tmp_path)
        win._set_frame(90)
        _key(win, Qt.Key.Key_M)
        win._editor.close()
        win.load(folder / "ethogram_raw.csv")
        assert [m.start_s for m in win._bar.markers()] == [pytest.approx(3.0)]

    def test_shift_m_keeps_the_range(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        win._bar.set_selection(30, 89)
        _key(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        win._editor.close()
        (marker,) = win._bar.markers()
        assert marker.kind == "range"
        assert (marker.start_s, marker.end_s) == (pytest.approx(1.0), pytest.approx(3.0))

    def test_shift_m_without_a_range_says_what_to_do(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        assert win._bar.markers() == []
        assert "In and Out" in win.statusBar().currentMessage()

    def test_the_editor_names_the_marker(self, qtbot, tmp_path):
        win, folder = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_M)
        win._editor.name.setText("door stuck")
        win._editor.done_btn.click()
        assert self._saved(folder)["markers"][0]["name"] == "door stuck"
        assert win._inspector.tabText(2) == "Markers (1)"

    def test_the_editor_deletes_the_marker(self, qtbot, tmp_path):
        win, folder = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_M)
        win._editor.delete_btn.click()
        assert self._saved(folder)["markers"] == []
        assert win._bar.markers() == []

    def test_dragging_a_marker_saves_where_it_landed(self, qtbot, tmp_path):
        win, folder = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_M)
        win._editor.close()
        (marker,) = win._bar.markers()
        win._bar.marker_changed.emit(marker.id, 5.0, None)
        assert self._saved(folder)["markers"][0]["start_s"] == pytest.approx(5.0)

    def test_up_and_down_step_between_markers(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        for frame in (30, 150):
            win._set_frame(frame)
            _key(win, Qt.Key.Key_M)
            win._editor.close()
        win._set_frame(0)
        _key(win, Qt.Key.Key_Down)
        assert win._frame == 30
        _key(win, Qt.Key.Key_Down)
        assert win._frame == 150
        _key(win, Qt.Key.Key_Up)
        assert win._frame == 30

    def test_going_to_a_range_marker_selects_it(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        win._bar.set_selection(30, 89)
        _key(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        win._editor.close()
        win._bar.clear_selection()
        (marker,) = win._bar.markers()
        win._go_to_marker(marker.id)
        assert win._bar.selection() == (30, 89) and win._frame == 30

    def test_the_range_card_names_the_markers_the_range_is_inside(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        win._bar.set_selection(30, 89)
        _key(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        win._editor.name.setText("Stim")
        win._editor.done_btn.click()
        win._bar.set_selection(40, 50)
        assert win._inspector.inside.text() == "inside Stim"

    def test_the_menu_offers_markers(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        texts = [a.text() for a in win._timeline_menu(120).actions() if a.text()]
        assert "Add marker here\tM" in texts and "Save range as marker…\t⇧M" in texts

    def test_an_unreadable_file_is_left_alone_and_turns_the_tools_off(
        self, qtbot, tmp_path, monkeypatch
    ):
        folder = tmp_path / "v"
        csv = _session(folder)
        (folder / "review_markers.json").write_text("{not json")
        warned = _said(monkeypatch, "warning")
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(csv)
        assert warned and "review_markers.json" in warned[0]
        assert not win._timeline_panel.marker_btn.isEnabled()
        _key(win, Qt.Key.Key_M)
        assert (folder / "review_markers.json").read_text() == "{not json"
        assert win._bar.markers() == []

    def test_a_failed_write_keeps_the_marker_and_says_where(self, qtbot, tmp_path, monkeypatch):
        win, _folder = self._win(qtbot, tmp_path)
        critical = _said(monkeypatch, "critical")

        def boom(*_a, **_k):
            raise OSError("read-only share")

        monkeypatch.setattr("glider.analysis.markers.save_markers", boom)
        _key(win, Qt.Key.Key_M)
        win._editor.close()
        assert critical and "review_markers.json" in critical[0]
        assert len(win._bar.markers()) == 1

    def test_a_single_session_cannot_share_a_range(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_M)
        assert not win._editor.whole_cohort.isEnabled()
        win._editor.close()

    def test_a_cohort_range_is_kept_beside_the_cohort_and_shown_in_every_session(
        self, qtbot, tmp_path
    ):
        root = tmp_path / "cohort"
        _session(root / "a")
        _session(root / "b")
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_folder(root)
        win._bar.set_selection(30, 89)
        _key(win, Qt.Key.Key_M, Qt.KeyboardModifier.ShiftModifier)
        win._editor.name.setText("Stim")
        win._editor.whole_cohort.setChecked(True)
        win._editor.done_btn.click()
        assert [m["scope"] for m in self._saved(root, "cohort_markers.json")["markers"]] == [
            "cohort"
        ]
        assert self._saved(win._session_folder(win._shown))["markers"] == []
        win._pool.select(1 - win._shown)
        assert [m.name for m in win._bar.markers()] == ["Stim"]

    def test_the_status_bar_counts_markers(self, qtbot, tmp_path):
        win, _folder = self._win(qtbot, tmp_path)
        _key(win, Qt.Key.Key_M)
        win._editor.close()
        assert "1 marker" in win._status.text()
