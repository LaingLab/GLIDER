"""Cross-validation scores only frames whose feature window is inside one bout.

The rolling window is causal, so the first ``window-1`` frames of every bout
are described by features that mostly summarise the *previous* behavior. They
cannot be classified correctly however good the model is, and scoring them
measures where the annotator drew the boundary instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.pipeline import _clean_window_rows


def _rows(labels, session=0):
    y = np.array(labels, dtype=object)
    return y, np.full(len(y), session), np.arange(len(y))


class TestCleanWindowRows:
    def test_the_first_frames_of_a_bout_are_excluded(self):
        y, sess, frame = _rows(["a"] * 10)
        ok = _clean_window_rows(y, sess, frame, window=4)
        # Frames 0-2 have a window reaching before the bout started.
        assert list(ok[:3]) == [False, False, False]
        assert ok[3:].all()

    def test_the_end_of_a_bout_is_kept(self):
        """Only the leading edge is contaminated; trailing frames are fine."""
        y, sess, frame = _rows(["a"] * 6 + ["b"] * 6)
        ok = _clean_window_rows(y, sess, frame, window=3)
        assert ok[5], "last frame of bout 'a' has a full clean window"
        assert not ok[6], "first frame of bout 'b' still looks at 'a'"
        assert ok[8], "three frames into 'b' the window is clean"

    def test_a_bout_shorter_than_the_window_is_never_scorable(self):
        y, sess, frame = _rows(["a"] * 3 + ["b"] * 2 + ["c"] * 3)
        ok = _clean_window_rows(y, sess, frame, window=4)
        assert not ok[3:5].any()

    def test_sessions_do_not_run_into_each_other(self):
        """Session 1's opening frames must not borrow session 0's window."""
        y = np.array(["a"] * 8, dtype=object)
        sess = np.array([0] * 4 + [1] * 4)
        frame = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        ok = _clean_window_rows(y, sess, frame, window=3)
        assert ok[2] and ok[3]
        assert not ok[4] and not ok[5]
        assert ok[6]

    def test_a_gap_in_frame_numbers_restarts_the_window(self):
        """Dropped frames break continuity even inside one label."""
        y = np.array(["a"] * 6, dtype=object)
        sess = np.zeros(6, dtype=int)
        frame = np.array([0, 1, 2, 50, 51, 52])
        ok = _clean_window_rows(y, sess, frame, window=3)
        assert ok[2]
        assert not ok[3] and not ok[4]
        assert ok[5]

    def test_window_of_one_keeps_everything(self):
        y, sess, frame = _rows(["a", "b", "a"])
        assert _clean_window_rows(y, sess, frame, window=1).all()

    def test_rows_arriving_out_of_order_are_handled(self):
        """The matrix is grouped by session, not guaranteed frame-sorted."""
        y = np.array(["a"] * 5, dtype=object)
        sess = np.zeros(5, dtype=int)
        frame = np.array([4, 0, 2, 1, 3])
        ok = _clean_window_rows(y, sess, frame, window=3)
        # Frames 2, 3 and 4 qualify wherever they sit in the array.
        got = dict(zip(frame, ok, strict=True))
        assert got == {0: False, 1: False, 2: True, 3: True, 4: True}


@pytest.mark.parametrize("window", [2, 5, 8])
def test_the_excluded_count_matches_the_window(window):
    """One bout loses exactly window-1 frames, whatever the window."""
    y, sess, frame = _rows(["a"] * 40)
    ok = _clean_window_rows(y, sess, frame, window=window)
    assert (~ok).sum() == window - 1
