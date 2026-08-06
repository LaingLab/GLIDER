"""Jumping straight to a clip by number.

Reviewing a flagged subset means arriving with a list of clip numbers and
wanting to see those clips. Stepping there with prev/next is impractical once
the queue runs to four figures.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from glider.analysis.behavior.annotations import AnnotationStore  # noqa: E402
from glider.gui.behavior.annotator.main_window import (  # noqa: E402
    AnnotatorWindow,
    clamp_clip_index,
)
from glider.gui.behavior.annotator.sampler import ProposedClip  # noqa: E402

# ---------------------------------------------------------------------------
# The index arithmetic, without a widget
# ---------------------------------------------------------------------------


class TestClampClipIndex:
    """Takes a 1-based clip number, returns a 0-based index into `clips`."""

    def test_it_converts_from_the_number_shown_to_the_index_stored(self):
        assert clamp_clip_index(1, 10) == 0
        assert clamp_clip_index(7, 10) == 6
        assert clamp_clip_index(10, 10) == 9

    def test_above_the_end_lands_on_the_last_clip(self):
        """Clamping beats erroring: a stale number should still show a clip."""
        assert clamp_clip_index(999, 10) == 9

    def test_below_the_start_lands_on_the_first_clip(self):
        assert clamp_clip_index(0, 10) == 0
        assert clamp_clip_index(-5, 10) == 0

    def test_an_empty_clip_list_has_no_valid_index(self):
        assert clamp_clip_index(1, 0) is None
        assert clamp_clip_index(0, 0) is None


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def _window(tmp_path, qtbot, n_clips=5):
    ann = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(ann)
    video = tmp_path / "a.mp4"
    clips = [
        ProposedClip(i, 50 + i * 100, 40 + i * 100, 60 + i * 100, 0.7, str(video))
        for i in range(n_clips)
    ]
    w = AnnotatorWindow(clips=clips, videos_meta={video: ann})
    qtbot.addWidget(w)
    return w


class TestGoTo:
    def test_jumping_moves_to_that_clip(self, tmp_path, qtbot):
        w = _window(tmp_path, qtbot)
        w._go_to(4)
        assert w.current == 3

    def test_a_number_past_the_end_stops_at_the_last_clip(self, tmp_path, qtbot):
        w = _window(tmp_path, qtbot)
        w._go_to(500)
        assert w.current == 4

    def test_the_box_shows_the_current_clip(self, tmp_path, qtbot):
        w = _window(tmp_path, qtbot)
        w._go_to(3)
        assert w.goto_spin.value() == 3

    def test_the_box_follows_the_other_nav_controls(self, tmp_path, qtbot):
        """It is a position readout as much as an input."""
        w = _window(tmp_path, qtbot)
        w._go(+1)
        assert w.goto_spin.value() == w.current + 1
        w._go(+1)
        assert w.goto_spin.value() == w.current + 1

    def test_its_range_matches_the_number_of_clips(self, tmp_path, qtbot):
        w = _window(tmp_path, qtbot, n_clips=7)
        assert w.goto_spin.minimum() == 1
        assert w.goto_spin.maximum() == 7
        assert w.goto_total.text().endswith("7")

    def test_the_trim_is_saved_before_jumping_away(self, tmp_path, qtbot):
        """Same guarantee prev/next give -- a jump must not drop an edit."""
        w = _window(tmp_path, qtbot)
        calls = []
        w._persist_current_trim = lambda: calls.append(w.current)
        w._go_to(5)
        assert calls == [0], "the clip being left should be persisted first"

    def test_an_empty_clip_list_does_not_crash(self, tmp_path, qtbot):
        ann = tmp_path / "a_annotations.csv"
        AnnotationStore().save_csv(ann)
        video = tmp_path / "a.mp4"
        w = AnnotatorWindow(clips=[], videos_meta={video: ann})
        qtbot.addWidget(w)
        w._go_to(3)
        assert w.current == 0


def test_numbering_matches_the_sidebar(tmp_path, qtbot):
    """The sidebar already counts from 1; the box must agree with it."""
    w = _window(tmp_path, qtbot)
    w._go_to(2)
    assert w.progress_label.text().startswith("clip 2 /")
    assert w.goto_spin.value() == 2
