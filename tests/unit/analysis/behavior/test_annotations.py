"""Round-trip smoke tests for annotations.py and vocabulary.py."""


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
