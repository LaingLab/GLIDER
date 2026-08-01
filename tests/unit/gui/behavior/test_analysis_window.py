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
