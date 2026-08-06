"""Mirror augmentation must not empty the scoring set.

Regression: ``_assemble_for_cv`` appends the mirrored variant of a session with
the SAME session id and the SAME frame numbers as the original. Sorting rows by
``(session, frame)`` therefore interleaves them -- frame 0 real, frame 0
mirror, frame 1 real, frame 1 mirror -- so a continuity test of
``frame[i] == frame[i-1] + 1`` never holds, no run ever reaches ``window-1``,
and every fold's evaluation set comes back empty.

The visible symptom was cross-validation reporting "0 folds" with no macro F1
and no accuracy, for every run with mirror augmentation on.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.analysis.behavior.pipeline import _clean_window_rows


def _mirrored(n=20, window=8):
    """One session assembled the way mirror augmentation assembles it."""
    y = np.array(["dig"] * n * 2, dtype=object)
    sess = np.zeros(n * 2, dtype=int)
    frame = np.concatenate([np.arange(n), np.arange(n)])
    mirror = np.array([False] * n + [True] * n)
    return y, sess, frame, mirror, window


class TestMirroredRows:
    def test_mirroring_does_not_empty_the_scoring_set(self):
        y, sess, frame, mirror, window = _mirrored()
        ok = _clean_window_rows(y, sess, frame, window, mirror=mirror)
        assert ok.any(), "every fold would score nothing"

    def test_the_real_rows_are_unaffected_by_their_mirrored_twins(self):
        """A mirrored copy must not interrupt the original's run."""
        y, sess, frame, mirror, window = _mirrored(n=20, window=8)
        with_mirror = _clean_window_rows(y, sess, frame, window, mirror=mirror)
        alone = _clean_window_rows(y[:20], sess[:20], frame[:20], window)
        assert list(with_mirror[:20]) == list(alone)

    def test_mirrored_copies_are_judged_on_their_own_run(self):
        y, sess, frame, mirror, window = _mirrored(n=20, window=8)
        ok = _clean_window_rows(y, sess, frame, window, mirror=mirror)
        # Same shape as the originals: the first window-1 excluded, rest kept.
        assert list(ok[20:]) == list(ok[:20])

    def test_it_still_works_when_no_mirror_column_is_given(self):
        """evaluate_model never mirrors and passes no mirror array."""
        y = np.array(["dig"] * 20, dtype=object)
        ok = _clean_window_rows(y, np.zeros(20, dtype=int), np.arange(20), 8)
        assert ok.sum() == 13

    def test_two_mirrored_sessions_stay_separate(self):
        y = np.array(["dig"] * 40, dtype=object)
        sess = np.array([0] * 10 + [0] * 10 + [1] * 10 + [1] * 10)
        frame = np.tile(np.arange(10), 4)
        mirror = np.array([False] * 10 + [True] * 10 + [False] * 10 + [True] * 10)
        ok = _clean_window_rows(y, sess, frame, 4, mirror=mirror)
        # Each of the four blocks loses its own leading window-1 rows.
        for block in range(4):
            chunk = ok[block * 10 : (block + 1) * 10]
            assert list(chunk[:3]) == [False, False, False]
            assert chunk[3:].all()


@pytest.mark.parametrize("window", [2, 8, 15])
def test_scored_rows_survive_end_to_end_with_mirroring(tmp_path, window):
    """The failure users saw: cross-validation reporting zero folds."""
    pytest.importorskip("sklearn")
    import numpy as np

    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.pipeline import cross_validate_sessions
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    names = ["snout", "neck", "tail"]
    rng = np.random.default_rng(0)
    sessions = []
    for s in range(3):
        n = 400
        xy = rng.normal(300, 20, size=(n, len(names), 2))
        xy[n // 2 :, 0, :] += 100
        pose = PoseData(
            xy=xy,
            confidence=np.ones((n, len(names))),
            keypoint_names=names,
            fps=30.0,
        )
        pose_csv = tmp_path / f"s{s}.csv"
        to_dlc_csv(pose, pose_csv)
        store = AnnotationStore()
        store.add(BehaviorZone(behavior="walk", start_frame=20, end_frame=n // 2 - 1))
        store.add(BehaviorZone(behavior="rear", start_frame=n // 2, end_frame=n - 1))
        ann = tmp_path / f"s{s}_annotations.csv"
        store.save_csv(ann)
        sessions.append((pose_csv, ann))

    res = cross_validate_sessions(
        sessions,
        window=window,
        fps=30.0,
        n_estimators=5,
        classifier_type="rf",
        mirror_augment=True,
        n_folds=3,
    )
    assert res["n_folds"] == 3, "mirror augmentation must not empty the folds"
    assert res["mean_macro_f1"] is not None
    assert res["per_class_metrics"]
    assert res["n_rows_scored"] > 0
