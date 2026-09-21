"""Session Review against a real recording, when one is supplied.

The repository is public and cohort data is unpublished, so no real recording
is committed. Point GLIDER_REVIEW_RECORDING at a recording folder to run this:

    GLIDER_REVIEW_RECORDING=/path/to/session QT_QPA_PLATFORM=offscreen \\
        uv run --no-sync pytest tests/integration/test_review_real_recording.py -q

It checks the two things synthetic fixtures cannot: that real tracking frames
line up with the real video, and that a real rig's events become lanes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

RECORDING = os.environ.get("GLIDER_REVIEW_RECORDING")

pytestmark = [
    pytest.mark.real_recording,
    pytest.mark.skipif(not RECORDING, reason="set GLIDER_REVIEW_RECORDING to a recording folder"),
]


def test_a_real_recording_opens_and_measures(qtbot, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings

    import glider.gui.behavior.analysis_window as review
    from glider.analysis import Session
    from glider.analysis.behavior.session_view import SessionView
    from glider.gui.behavior.analysis_window import AnalysisWindow

    # The window saves its layout on close; never into the developer's settings.
    path = str(tmp_path / "session_review.ini")
    monkeypatch.setattr(review, "_settings", lambda: QSettings(path, QSettings.Format.IniFormat))

    folder = Path(RECORDING)
    session = Session.load(folder)
    view = SessionView.from_recording(session)
    if view.video_path is not None:
        assert view.video_is_aligned, (
            f"{view.video_frames:,} video frames against tracking frames "
            f"{int(view.frames[0])}..{int(view.frames[-1])} "
            f"(first_video_frame={view.first_video_frame})"
        )

    window = AnalysisWindow()
    qtbot.addWidget(window)
    window.open_path(folder)
    first, last = window._bar.frame_bounds()
    window._bar.set_selection(first, last)
    if not session.events.empty:
        assert window._bar.timeline().lanes, "the event log produced no hardware lanes"
    if view.px_per_mm:
        assert window._inspector.distance.text() != "—"
    else:
        assert "no calibration" in window._summary.text().lower()
