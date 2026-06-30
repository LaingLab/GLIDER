"""Tests for per-frame label series construction.

Ported from yolo2pose/tests/test_train.py (the label-series tests).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Label series
# ---------------------------------------------------------------------------


def test_build_label_series_basic():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="locomote", start_frame=0, end_frame=10))
    store.add(BehaviorZone(behavior="rest", start_frame=20, end_frame=30))
    s = build_label_series(store, n_frames=40)
    assert (s[:10] == "locomote").all()
    assert (s[10:20] == "").all()
    assert (s[20:30] == "rest").all()
    assert (s[30:40] == "").all()


def test_build_label_series_marks_multi_behavior_overlap_as_ambiguous():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import AMBIGUOUS, build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="locomote", start_frame=10, end_frame=30))
    store.add(BehaviorZone(behavior="rear", start_frame=20, end_frame=40))
    s = build_label_series(store, n_frames=50)
    # 0-9 empty; 10-19 locomote; 20-29 ambiguous (both); 30-39 rear; 40-49 empty
    assert (s[:10] == "").all()
    assert (s[10:20] == "locomote").all()
    assert (s[20:30] == AMBIGUOUS).all()
    assert (s[30:40] == "rear").all()


def test_build_label_series_zero_length():
    from glider.analysis.behavior.annotations import AnnotationStore
    from glider.analysis.behavior.labels import build_label_series

    s = build_label_series(AnnotationStore(), n_frames=0)
    assert len(s) == 0


def test_build_label_series_treats_multi_behavior_as_ambiguous():
    """Annotator's exclusion markers must drop frames from training."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import AMBIGUOUS, build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="rearing", start_frame=0, end_frame=10))
    store.add(BehaviorZone(behavior="multi-behavior", start_frame=20, end_frame=30))
    store.add(BehaviorZone(behavior="unclear", start_frame=40, end_frame=50))
    s = build_label_series(store, n_frames=60)
    assert (s[0:10] == "rearing").all()
    # multi-behavior + unclear zones collapse to AMBIGUOUS so the
    # training pipeline drops those frames.
    assert (s[20:30] == AMBIGUOUS).all()
    assert (s[40:50] == AMBIGUOUS).all()


def test_build_label_and_group_series_assigns_zone_ids():
    """Group IDs should match the zone insertion order; unannotated → -1."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import build_label_and_group_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="a", start_frame=0, end_frame=10))
    store.add(BehaviorZone(behavior="b", start_frame=20, end_frame=30))
    labels, groups = build_label_and_group_series(store, n_frames=40)
    # First zone → group 0, second → group 1, unannotated → -1.
    assert (groups[0:10] == 0).all()
    assert (groups[10:20] == -1).all()
    assert (groups[20:30] == 1).all()
    assert (groups[30:40] == -1).all()


def test_build_label_series_merge_map_remaps_behavior():
    """A merge_map renames behaviors before the label series is built."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="grooming", start_frame=0, end_frame=10))
    store.add(BehaviorZone(behavior="flank groom", start_frame=20, end_frame=30))
    s = build_label_series(
        store,
        n_frames=40,
        merge_map={"grooming": "groom", "flank groom": "groom"},
    )
    assert (s[0:10] == "groom").all()
    assert (s[10:20] == "").all()
    assert (s[20:30] == "groom").all()


def test_build_label_series_merge_collapses_overlap_to_merged_class():
    """Overlapping zones of two merged behaviors collapse to the merged
    class instead of being dropped as ambiguous — the remap happens
    BEFORE ambiguity is computed."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import AMBIGUOUS, build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="grooming", start_frame=10, end_frame=30))
    store.add(BehaviorZone(behavior="flank groom", start_frame=20, end_frame=40))
    s = build_label_series(
        store,
        n_frames=50,
        merge_map={"grooming": "groom", "flank groom": "groom"},
    )
    # The 20-29 overlap is now a single class, not ambiguous.
    assert (s[10:40] == "groom").all()
    assert (s == AMBIGUOUS).sum() == 0


def test_build_label_series_merge_leaves_reserved_and_unannotated():
    """Reserved exclusion labels and unannotated frames are untouched by
    a merge_map (reserved labels can't be merge members)."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import AMBIGUOUS, build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="sniff", start_frame=0, end_frame=10))
    store.add(BehaviorZone(behavior="unclear", start_frame=20, end_frame=30))
    s = build_label_series(
        store,
        n_frames=40,
        merge_map={"sniff": "explore", "rearing": "explore"},
    )
    assert (s[0:10] == "explore").all()
    assert (s[10:20] == "").all()
    assert (s[20:30] == AMBIGUOUS).all()
    assert (s[30:40] == "").all()


def test_exclude_drops_behavior_frames():
    """An excluded behavior's frames become unannotated; an overlap with a
    kept behavior resolves to the kept one (not AMBIGUOUS)."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="immobile", start_frame=0, end_frame=10))
    store.add(BehaviorZone(behavior="grooming", start_frame=5, end_frame=15))
    s = build_label_series(store, n_frames=15, exclude={"immobile"})
    # 0-4: immobile-only → excluded → unannotated. 5-14: grooming (the
    # overlap is no longer ambiguous because immobile is gone).
    assert (s[0:5] == "").all()
    assert (s[5:15] == "grooming").all()


def test_exclude_overrides_merge():
    """Excluding a behavior takes precedence over merging it."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="immobile", start_frame=0, end_frame=10))
    s = build_label_series(
        store, n_frames=10, merge_map={"immobile": "other"}, exclude={"immobile"}
    )
    assert (s == "").all()  # excluded, not merged into "other"


def test_build_label_series_multi_behavior_wins_over_named_overlap():
    """A multi-behavior zone overlapping a named zone marks the overlap
    region as AMBIGUOUS, never as the named behavior."""
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.labels import AMBIGUOUS, build_label_series

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="rearing", start_frame=0, end_frame=20))
    store.add(BehaviorZone(behavior="multi-behavior", start_frame=10, end_frame=15))
    s = build_label_series(store, n_frames=20)
    # 0-9 rearing; 10-14 AMBIGUOUS (multi-behavior wins); 15-19 rearing.
    assert (s[0:10] == "rearing").all()
    assert (s[10:15] == AMBIGUOUS).all()
    assert (s[15:20] == "rearing").all()
