"""The session review window: ethogram scrubber, keypoint canvas, segments."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QColor, QMouseEvent  # noqa: E402

from glider.analysis.behavior.session_view import SessionView  # noqa: E402
from glider.gui.behavior.analysis_window import (  # noqa: E402
    AnalysisWindow,
    EthogramBar,
    KeypointCanvas,
    behavior_qcolor,
)

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


class TestEthogramBar:
    def _bar(self, qtbot, tmp_path, **kw):
        bar = EthogramBar()
        qtbot.addWidget(bar)
        bar.resize(300, 46)
        bar.set_view(SessionView.load(_session(tmp_path / "v", **kw)))
        return bar

    def test_an_empty_bar_does_not_crash(self, qtbot):
        bar = EthogramBar()
        qtbot.addWidget(bar)
        bar.set_view(None)
        bar.resize(200, 46)
        bar.grab()  # forces a paint

    def test_it_paints_a_loaded_session(self, qtbot, tmp_path):
        bar = self._bar(qtbot, tmp_path)
        image = bar.grab().toImage()
        assert image.width() == 300

    def test_clicking_scrubs_to_that_frame(self, qtbot, tmp_path):
        bar = self._bar(qtbot, tmp_path)
        seen = []
        bar.scrubbed.connect(seen.append)
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(150, 20),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        bar.mousePressEvent(event)
        # Half way along a 300-frame session.
        assert seen and 140 <= seen[0] <= 160

    def test_a_selection_is_reported_and_kept(self, qtbot, tmp_path):
        bar = self._bar(qtbot, tmp_path)
        seen = []
        bar.selection_changed.connect(lambda a, b: seen.append((a, b)))
        bar.set_selection(200, 50)
        # Reversed input is normalised rather than refused.
        assert bar.selection() == (50, 200)
        assert seen == [(50, 200)]

    def test_the_selection_tints_the_bands_rather_than_hiding_them(self, qtbot, tmp_path):
        """`colors.with_alpha` is a QSS string; QPainter reads it as opaque black."""
        bar = self._bar(qtbot, tmp_path)
        before = bar.grab().toImage().pixelColor(150, 23)
        bar.set_selection(100, 199)  # 1 px per frame, so x=150 is inside
        after = bar.grab().toImage().pixelColor(150, 23)
        assert after != QColor(0, 0, 0)
        assert after != before  # the tint is actually visible

    def test_scrubbing_does_not_clear_the_selection(self, qtbot, tmp_path):
        """A window survives looking around inside it."""
        bar = self._bar(qtbot, tmp_path)
        bar.set_selection(50, 200)
        bar.set_frame(120)
        assert bar.selection() == (50, 200)


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
        assert "300" in win._summary.text()

    def test_selecting_a_window_fills_the_bout_table(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._bar.set_selection(100, 199)
        assert win._bouts.rowCount() == 1
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
        assert "3.00" in win._clock.text()

    def test_playback_stops_at_the_end(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        win._set_frame(298)
        win._toggle_play()
        win._advance()  # 299
        win._advance()  # would pass the end
        assert win._timer.isActive() is False
        assert win._play.text() == "Play"

    def test_the_repair_button_is_hidden_when_nothing_needs_repairing(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path)
        assert win._fix_resolution.isVisibleTo(win) is False

    def test_the_repair_button_appears_for_a_sidecar_without_a_resolution(self, qtbot, tmp_path):
        """Runs from before the field existed must still be viewable."""
        win = self._win(qtbot, tmp_path, with_resolution=False)
        assert win._fix_resolution.isVisibleTo(win) is True

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
        assert win._fix_resolution.isVisibleTo(win) is False

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
        assert win._pick_poses.isVisibleTo(win) is True

        win.load(_session(tmp_path / "with_poses"))
        assert win._pick_poses.isVisibleTo(win) is False

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
        assert win._pick_poses.isVisibleTo(win) is False

    def test_the_summary_names_the_pose_file_that_was_used(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        assert "vDLC_exp-7.csv" in win._summary.text()

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
        assert win._play.text() == "Play"

    def test_keys_are_inert_before_a_session_is_loaded(self, qtbot):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        _key(win, Qt.Key.Key_Right)  # must not raise
        assert win._frame == 0


class TestTheSpeedAxisIsShown:
    def _session_with_speed(self, tmp_path, n=300):
        folder = tmp_path / "sp"
        folder.mkdir(parents=True, exist_ok=True)
        speed = [""] * n
        for i in range(100, 160):
            speed[i] = "freezing"
        pd.DataFrame(
            {
                "frame": range(n),
                "behavior": ["groom"] * n,
                "speed": speed,
                "speed_px_frame": [0.4] * n,
                "speed_cm_s": [1.2] * n,
            }
        ).to_csv(folder / "ethogram_raw.csv", index=False)
        return folder / "ethogram_raw.csv"

    def test_the_bar_paints_a_second_lane(self, qtbot, tmp_path):
        bar = EthogramBar()
        qtbot.addWidget(bar)
        bar.resize(300, 46)
        bar.set_view(SessionView.load(self._session_with_speed(tmp_path)))
        image = bar.grab().toImage()
        # Same column, two lanes: posture on top, freezing beneath it.
        assert image.pixelColor(130, 10) != image.pixelColor(130, 42)

    def test_a_session_without_a_speed_axis_uses_the_full_height(self, qtbot, tmp_path):
        bar = EthogramBar()
        qtbot.addWidget(bar)
        bar.resize(300, 46)
        bar.set_view(SessionView.load(_session(tmp_path / "plain")))
        image = bar.grab().toImage()
        assert image.pixelColor(130, 10) == image.pixelColor(130, 42)

    def test_freeze_bouts_reach_the_table(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(self._session_with_speed(tmp_path))
        win._bar.set_selection(0, 299)
        states = {win._bouts.item(r, 0).text() for r in range(win._bouts.rowCount())}
        assert "freezing" in states
        assert "groom" in states
        assert "— speed axis —" in states  # the two are stacked, never summed


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
        assert "v.mp4" in win._summary.text()
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
            speed = [""] * rows
            for f in range(100, 100 + 30 * (i + 1)):
                speed[f] = "freezing"
            pd.DataFrame(
                {
                    "frame": range(rows),
                    "behavior": labels,
                    "speed": speed,
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
        assert win._sessions.count() == 4

    def test_the_picker_hides_for_a_single_session(self, qtbot, tmp_path):
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load(_session(tmp_path / "v"))
        assert win._sessions.isVisibleTo(win) is False

    def test_switching_session_keeps_the_window(self, qtbot, tmp_path):
        """The window is the question; changing which animal answers it must
        not silently reset it."""
        win = AnalysisWindow()
        qtbot.addWidget(win)
        win.load_many(self._cohort(tmp_path))
        win._bar.set_selection(100, 199)
        win._sessions.setCurrentIndex(2)
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
        assert "Zones (" in win._tables.tabText(2)

    def test_without_zones_the_table_stays_empty(self, qtbot, tmp_path):
        win = self._win(qtbot, tmp_path, with_zones=False)
        win._bar.set_selection(0, 299)
        assert win._zone_table.rowCount() == 0
        assert win._tables.tabText(2) == "Zones"

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

    def test_clicking_the_far_left_lands_on_the_first_frame(self, qtbot, tmp_path):
        bar = EthogramBar()
        qtbot.addWidget(bar)
        bar.resize(300, 46)
        bar.set_view(SessionView.load(_windowed_session(tmp_path)))
        assert bar._frame_at(0.0) == 3600
        # 9000 frames across 300 px is 30 frames per pixel, so the rightmost
        # pixel lands inside the final 30 rather than exactly on the last.
        assert 12570 <= bar._frame_at(299.9) <= 12599

    def test_the_scored_range_fills_the_width(self, qtbot, tmp_path):
        bar = EthogramBar()
        qtbot.addWidget(bar)
        bar.resize(300, 46)
        bar.set_view(SessionView.load(_windowed_session(tmp_path)))
        assert bar._x_of(3600) == pytest.approx(0.0)
        assert bar._x_of(12599) == pytest.approx(300.0, abs=0.5)

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
        win._sessions.setCurrentIndex(2)
        win._sessions.setCurrentIndex(3)
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
