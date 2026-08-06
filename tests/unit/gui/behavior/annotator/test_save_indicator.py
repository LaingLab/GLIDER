"""The save indicator must describe the session on screen.

It sits in the header beside the video title, so a filename left over from a
previous session reads as "this clip belongs to that session". Reviewing saved
annotations crosses videos constantly, and jumping by clip number crosses them
in one keystroke, which is how this surfaced.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from glider.analysis.behavior.annotations import (  # noqa: E402
    AnnotationStore,
    BehaviorZone,
)
from glider.gui.behavior.annotator.main_window import AnnotatorWindow  # noqa: E402
from glider.gui.behavior.annotator.sampler import ProposedClip  # noqa: E402


def _two_sessions(tmp_path, qtbot):
    """Two videos, one clip each, so navigation crosses a session boundary."""
    meta, clips = {}, []
    for i, name in enumerate(("alpha", "beta")):
        video = tmp_path / f"{name}.mp4"
        ann = tmp_path / f"{name}_annotations.csv"
        store = AnnotationStore()
        store.add(BehaviorZone(behavior="dig", start_frame=10, end_frame=40))
        store.save_csv(ann)
        meta[video] = ann
        clips.append(ProposedClip(i, 25, 10, 40, 1.0, str(video)))
    w = AnnotatorWindow(clips=clips, videos_meta=meta)
    qtbot.addWidget(w)
    return w


class TestSaveIndicator:
    def test_it_names_the_session_being_shown(self, tmp_path, qtbot):
        w = _two_sessions(tmp_path, qtbot)
        assert "alpha" in w.save_indicator.text()
        assert "beta" not in w.save_indicator.text()

    def test_moving_to_another_session_updates_it(self, tmp_path, qtbot):
        w = _two_sessions(tmp_path, qtbot)
        w._go(+1)
        assert "beta" in w.save_indicator.text()
        assert "alpha" not in w.save_indicator.text()

    def test_a_save_in_one_session_does_not_follow_you_to_the_next(self, tmp_path, qtbot):
        """The regression: the indicator kept the file it last wrote.

        Persisting a trim stamps the indicator with the session being left,
        immediately before the title updates to the session being entered --
        so the header showed one video's name beside another's filename.
        """
        w = _two_sessions(tmp_path, qtbot)
        w._save_annotations_for_video(tmp_path / "alpha.mp4")
        assert "alpha" in w.save_indicator.text()

        w._go_to(2)  # jump to beta
        assert (
            "alpha" not in w.save_indicator.text()
        ), "a filename from the session we left must not sit beside beta's title"
        assert "beta" in w.save_indicator.text()

    def test_a_save_in_the_current_session_is_confirmed(self, tmp_path, qtbot):
        """Context is not worth losing the 'it saved' feedback for."""
        w = _two_sessions(tmp_path, qtbot)
        w._save_annotations_for_video(tmp_path / "alpha.mp4")
        assert w.save_indicator.text().startswith("saved")
        # Still confirmed after an unrelated refresh, because we are still here.
        w._refresh_all()
        assert w.save_indicator.text().startswith("saved")
        assert "alpha" in w.save_indicator.text()

    def test_an_unreadable_file_is_reported_for_the_session_shown(self, tmp_path, qtbot):
        w = _two_sessions(tmp_path, qtbot)
        w.load_errors[tmp_path / "beta.mp4"] = "malformed"
        w._go(+1)
        text = w.save_indicator.text()
        assert "beta" in text and "unreadable" in text
        # And the healthy session must not inherit the warning.
        w._go(-1)
        assert "unreadable" not in w.save_indicator.text()

    def test_it_survives_having_no_clips(self, tmp_path, qtbot):
        video = tmp_path / "alpha.mp4"
        ann = tmp_path / "alpha_annotations.csv"
        AnnotationStore().save_csv(ann)
        w = AnnotatorWindow(clips=[], videos_meta={video: ann})
        qtbot.addWidget(w)
        w._refresh_all()  # must not raise
