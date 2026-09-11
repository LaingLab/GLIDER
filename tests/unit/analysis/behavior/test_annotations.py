"""Round-trip smoke tests for annotations.py and vocabulary.py."""

import pytest


def test_annotation_store_roundtrip(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    store = AnnotationStore()
    store.add(BehaviorZone(behavior="rear", start_frame=0, end_frame=10))
    out = tmp_path / "ann.csv"
    store.save_csv(out)
    restored = AnnotationStore.load_csv(out)
    assert len(restored) == len(store)
    assert [z.behavior for z in restored] == [z.behavior for z in store]
    # Frame bounds must survive the int round-trip, not just the labels.
    z_orig, z_rest = next(iter(store)), next(iter(restored))
    assert z_rest.start_frame == z_orig.start_frame
    assert z_rest.end_frame == z_orig.end_frame


def test_vocabulary_roundtrip(tmp_path):
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary

    vocab = Vocabulary([Behavior(name="rear", hotkey="r"), Behavior(name="groom", hotkey="g")])
    out = tmp_path / "vocab.json"
    vocab.save(out)
    restored = Vocabulary.load(out)
    assert restored.names() == vocab.names()
    assert restored.hotkeys() == vocab.hotkeys()


def test_two_animals_share_a_behavior_at_overlapping_frames(tmp_path):
    """The case that is impossible today: both animals grooming at once.

    Not a contrived overlap -- this is what a social assay looks like for
    most of its duration, and the same-behavior overlap rule rejected it.
    """
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    store = AnnotationStore()
    store.add(BehaviorZone("grooming", 120, 180, individual=0))
    store.add(BehaviorZone("grooming", 140, 195, individual=1))
    assert len(store) == 2

    path = store.save_csv(tmp_path / "pair_annotations.csv")
    reloaded = AnnotationStore.load_csv(path)
    assert len(reloaded) == 2
    assert {z.individual for z in reloaded} == {0, 1}


def test_same_animal_same_behavior_still_rejected():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone, OverlapError

    store = AnnotationStore()
    store.add(BehaviorZone("grooming", 120, 180, individual=1))
    with pytest.raises(OverlapError) as e:
        store.add(BehaviorZone("grooming", 140, 195, individual=1))
    # The message must name the animal, or a real double-label on animal 1
    # reads identically to the legitimate two-animal case. Assert on the
    # full phrase, not a bare "1" -- the frame numbers below (140, 195,
    # 120, 180) all contain the digit "1" too, so a bare "1" in str(e.value)
    # would stay green even if "for animal {individual}" were deleted from
    # the message entirely.
    assert "animal 1" in str(e.value)


def test_a_csv_with_no_individual_column_loads_as_animal_zero(tmp_path):
    """Every annotations CSV that exists today has no such column."""
    from glider.analysis.behavior.annotations import AnnotationStore

    path = tmp_path / "legacy_annotations.csv"
    path.write_text(
        "behavior,start_frame,end_frame,created_at,note\n"
        "rearing,10,20,2026-01-01T00:00:00+00:00,\n"
    )
    store = AnnotationStore.load_csv(path)
    assert [z.individual for z in store] == [0]


def test_a_blank_individual_cell_loads_as_animal_zero(tmp_path):
    """What a spreadsheet round-trip produces; int('') would raise."""
    from glider.analysis.behavior.annotations import AnnotationStore

    path = tmp_path / "blank_annotations.csv"
    path.write_text(
        "behavior,start_frame,end_frame,created_at,note,individual\n"
        "rearing,10,20,2026-01-01T00:00:00+00:00,,\n"
    )
    store = AnnotationStore.load_csv(path)
    assert [z.individual for z in store] == [0]


def test_load_csv_filters_to_one_individual(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    store = AnnotationStore()
    store.add(BehaviorZone("grooming", 120, 180, individual=0))
    store.add(BehaviorZone("grooming", 140, 195, individual=1))
    store.add(BehaviorZone("rearing", 300, 340, individual=0))
    path = store.save_csv(tmp_path / "pair_annotations.csv")

    assert len(AnnotationStore.load_csv(path)) == 3
    only_0 = AnnotationStore.load_csv(path, individual=0)
    assert len(only_0) == 2
    assert all(z.individual == 0 for z in only_0)
    only_1 = AnnotationStore.load_csv(path, individual=1)
    assert [z.behavior for z in only_1] == ["grooming"]
