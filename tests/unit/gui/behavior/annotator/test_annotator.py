"""Tests for the annotator's pure-Python data model.

Covers :mod:`glider.analysis.behavior.annotations` and
:mod:`glider.analysis.behavior.vocabulary`. The Qt widgets aren't covered
here — that needs a display + manual testing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Annotation path resolution (must match `yolo2pose train`)
# ---------------------------------------------------------------------------


def test_annotation_path_is_pose_csv_sibling_not_video_sibling():
    """Regression: the annotator must write annotations next to the POSE
    CSV — exactly where `yolo2pose train` reads them (cli resolves them as
    ``pose_csv.parent / f'{stem}_annotations.csv'``). In a split
    ``--videos-dir`` / ``--pose-csv-dir`` layout, writing next to the video
    instead meant review edits silently never reached training."""
    from glider.gui.behavior.annotator.app import annotation_path_for

    pose_csv = Path("/data/poses/t10_d2.csv")
    ann = annotation_path_for(pose_csv)
    assert ann == Path("/data/poses/t10_d2_annotations.csv")
    assert ann.parent.name == "poses"  # next to the pose CSV, not the video


# ---------------------------------------------------------------------------
# BehaviorZone
# ---------------------------------------------------------------------------


def test_zone_basic_construction():
    from glider.analysis.behavior.annotations import BehaviorZone

    z = BehaviorZone(behavior="rearing", start_frame=100, end_frame=120)
    assert z.behavior == "rearing"
    assert z.start_frame == 100
    assert z.end_frame == 120
    assert z.duration_frames == 20
    assert z.created_at  # auto-stamped


def test_zone_rejects_invalid_frames():
    from glider.analysis.behavior.annotations import BehaviorZone

    with pytest.raises(ValueError):
        BehaviorZone(behavior="x", start_frame=10, end_frame=10)  # zero length
    with pytest.raises(ValueError):
        BehaviorZone(behavior="x", start_frame=10, end_frame=5)  # negative length
    with pytest.raises(ValueError):
        BehaviorZone(behavior="x", start_frame=-1, end_frame=5)
    with pytest.raises(ValueError):
        BehaviorZone(behavior="", start_frame=0, end_frame=5)
    with pytest.raises(ValueError):
        BehaviorZone(behavior="   ", start_frame=0, end_frame=5)


def test_zone_covers_and_overlaps():
    from glider.analysis.behavior.annotations import BehaviorZone

    z = BehaviorZone(behavior="a", start_frame=10, end_frame=20)
    assert z.covers(10) and z.covers(15) and z.covers(19)
    assert not z.covers(9) and not z.covers(20)  # half-open end

    z2 = BehaviorZone(behavior="a", start_frame=15, end_frame=25)
    assert z.overlaps(z2)
    z3 = BehaviorZone(behavior="a", start_frame=20, end_frame=30)
    # z ends at 20 (exclusive), z3 starts at 20: touch but don't overlap.
    assert not z.overlaps(z3)
    z4 = BehaviorZone(behavior="b", start_frame=15, end_frame=18)
    # overlaps() doesn't consider behavior name — that's the store's job.
    assert z.overlaps(z4)


# ---------------------------------------------------------------------------
# AnnotationStore
# ---------------------------------------------------------------------------


def test_store_add_remove_and_iter():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    s = AnnotationStore()
    assert len(s) == 0
    assert not s

    z1 = BehaviorZone(behavior="a", start_frame=0, end_frame=5)
    s.add(z1)
    z2 = BehaviorZone(behavior="b", start_frame=2, end_frame=7)
    s.add(z2)
    assert len(s) == 2
    assert list(s) == [z1, z2]

    assert s.remove(z1) is True
    assert s.remove(z1) is False
    assert len(s) == 1


def test_store_rejects_same_behavior_overlap():
    from glider.analysis.behavior.annotations import (
        AnnotationStore,
        BehaviorZone,
        OverlapError,
    )

    s = AnnotationStore()
    s.add(BehaviorZone(behavior="rearing", start_frame=10, end_frame=30))
    # Overlapping rearing zone — rejected.
    with pytest.raises(OverlapError):
        s.add(BehaviorZone(behavior="rearing", start_frame=20, end_frame=40))
    # Adjacent (touching but not overlapping) is allowed.
    s.add(BehaviorZone(behavior="rearing", start_frame=30, end_frame=40))
    assert len(s) == 2


def test_store_allows_cross_behavior_overlap():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    s = AnnotationStore()
    s.add(BehaviorZone(behavior="rearing", start_frame=10, end_frame=30))
    s.add(BehaviorZone(behavior="grooming", start_frame=20, end_frame=40))
    assert len(s) == 2


def test_store_zones_at_frame():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    s = AnnotationStore()
    s.add(BehaviorZone(behavior="rearing", start_frame=10, end_frame=20))
    s.add(BehaviorZone(behavior="grooming", start_frame=15, end_frame=25))
    s.add(BehaviorZone(behavior="locomote", start_frame=30, end_frame=40))

    at_18 = s.zones_at_frame(18)
    assert {z.behavior for z in at_18} == {"rearing", "grooming"}
    assert s.zones_at_frame(5) == []
    assert {z.behavior for z in s.zones_at_frame(35)} == {"locomote"}


def test_store_counts_and_totals():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    s = AnnotationStore()
    s.add(BehaviorZone(behavior="a", start_frame=0, end_frame=10))
    s.add(BehaviorZone(behavior="a", start_frame=20, end_frame=25))
    s.add(BehaviorZone(behavior="b", start_frame=0, end_frame=100))
    assert s.counts_by_behavior() == {"a": 2, "b": 1}
    assert s.total_frames_by_behavior() == {"a": 15, "b": 100}


def test_store_csv_round_trip(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone

    s = AnnotationStore()
    s.add(BehaviorZone(behavior="rearing", start_frame=10, end_frame=20, note="quick"))
    s.add(BehaviorZone(behavior="grooming", start_frame=15, end_frame=25))
    s.add(BehaviorZone(behavior="locomote", start_frame=30, end_frame=40))

    csv_path = tmp_path / "annotations.csv"
    s.save_csv(csv_path)
    assert csv_path.exists()
    text = csv_path.read_text()
    # Header + 3 data rows.
    assert text.count("\n") >= 4
    assert "rearing" in text and "quick" in text

    s2 = AnnotationStore.load_csv(csv_path)
    assert len(s2) == 3
    # CSV save sorts by start_frame; check that's reflected.
    starts = [z.start_frame for z in s2]
    assert starts == sorted(starts)
    rearing = [z for z in s2 if z.behavior == "rearing"][0]
    assert rearing.note == "quick"


def test_store_load_missing_file_returns_empty(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore

    s = AnnotationStore.load_csv(tmp_path / "does_not_exist.csv")
    assert len(s) == 0


def test_store_load_malformed_csv_raises(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore

    bad = tmp_path / "bad.csv"
    bad.write_text("not,the,right,header\n1,2,3,4\n")
    with pytest.raises(ValueError):
        AnnotationStore.load_csv(bad)


def test_store_replace_rejects_overlaps():
    from glider.analysis.behavior.annotations import (
        AnnotationStore,
        BehaviorZone,
        OverlapError,
    )

    s = AnnotationStore()
    bad = [
        BehaviorZone(behavior="a", start_frame=0, end_frame=10),
        BehaviorZone(behavior="a", start_frame=5, end_frame=15),  # overlaps
    ]
    with pytest.raises(OverlapError):
        s.replace(bad)
    # State unchanged on rejection.
    assert len(s) == 0


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_vocab_add_and_lookup():
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary

    v = Vocabulary()
    v.add(Behavior(name="rearing", hotkey="1"))
    v.add(Behavior(name="grooming", hotkey="2"))
    assert "rearing" in v
    assert len(v) == 2
    assert v.behavior_for_hotkey("1").name == "rearing"
    assert v.behavior_for_hotkey("9") is None
    assert v.behavior_for_name("grooming").hotkey == "2"
    # Default palette is applied when no color was given.
    assert v.behavior_for_name("rearing").color.startswith("#")


def test_vocab_rejects_duplicate_hotkey_or_name():
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary, VocabularyError

    v = Vocabulary()
    v.add(Behavior(name="a", hotkey="1"))
    with pytest.raises(VocabularyError):
        v.add(Behavior(name="b", hotkey="1"))  # hotkey collision
    with pytest.raises(VocabularyError):
        v.add(Behavior(name="a", hotkey="2"))  # name collision


def test_vocab_validates_hotkey_length():
    from glider.analysis.behavior.vocabulary import Behavior, VocabularyError

    with pytest.raises(VocabularyError):
        Behavior(name="x", hotkey="")
    with pytest.raises(VocabularyError):
        Behavior(name="x", hotkey="ab")  # must be single char


def test_vocab_validates_color():
    from glider.analysis.behavior.vocabulary import Behavior, VocabularyError

    Behavior(name="x", hotkey="1", color="#1d4ed8")  # ok
    Behavior(name="x", hotkey="2", color="#abc")  # 3-char shorthand ok
    with pytest.raises(VocabularyError):
        Behavior(name="x", hotkey="3", color="blue")  # not hex
    with pytest.raises(VocabularyError):
        Behavior(name="x", hotkey="4", color="#zzz")


def test_vocab_remove():
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary

    v = Vocabulary()
    v.add(Behavior(name="a", hotkey="1"))
    v.add(Behavior(name="b", hotkey="2"))
    assert v.remove("a") is True
    assert v.remove("a") is False  # already gone
    assert "a" not in v
    # The hotkey "1" is freed up.
    v.add(Behavior(name="c", hotkey="1"))
    assert v.behavior_for_hotkey("1").name == "c"


def test_vocab_json_round_trip(tmp_path):
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary

    v = Vocabulary()
    v.add(Behavior(name="rearing", hotkey="1", color="#1d4ed8"))
    v.add(Behavior(name="grooming", hotkey="2"))
    path = tmp_path / "vocab.json"
    v.save(path)
    payload = json.loads(path.read_text())
    assert "behaviors" in payload and len(payload["behaviors"]) == 2

    v2 = Vocabulary.load(path)
    assert v2.names() == ["rearing", "grooming"]
    assert v2.color_for("rearing") == "#1d4ed8"


def test_vocab_yaml_round_trip(tmp_path):
    pytest.importorskip("yaml")
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary

    v = Vocabulary()
    v.add(Behavior(name="rearing", hotkey="1"))
    v.add(Behavior(name="grooming", hotkey="2"))
    path = tmp_path / "vocab.yaml"
    v.save(path)
    v2 = Vocabulary.load(path)
    assert v2.names() == v.names()
    assert v2.hotkeys() == v.hotkeys()


def test_vocab_from_dict_validates_shape():
    from glider.analysis.behavior.vocabulary import Vocabulary, VocabularyError

    with pytest.raises(VocabularyError):
        Vocabulary.from_dict({})  # missing 'behaviors'
    with pytest.raises(VocabularyError):
        Vocabulary.from_dict({"behaviors": "nope"})  # wrong type
    with pytest.raises(VocabularyError):
        Vocabulary.from_dict({"behaviors": ["just a string"]})  # wrong entry type


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn")  # k-means++ needs scikit-learn


def _make_three_regime_pose(seed: int = 42, n_frames: int = 600):
    """Same fixture as test_train — three movement regimes with distinct stats."""
    import numpy as np

    from glider.vision.pose.core import PoseData

    rng = np.random.default_rng(seed)
    n_kpts = 5
    names = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
    xy = np.empty((n_frames, n_kpts, 2))
    per = n_frames // 3
    for regime, slc in enumerate([slice(0, per), slice(per, 2 * per), slice(2 * per, n_frames)]):
        n = slc.stop - slc.start
        t = np.arange(n)
        if regime == 0:
            cx = 50 + 0.5 * t
            cy = 200 + 0.0 * t
            jitter = 0.4
        elif regime == 1:
            cx = 350 + 3 * np.sin(0.5 * t)
            cy = 200 + 3 * np.cos(0.5 * t)
            jitter = 1.0
        else:
            cx = np.full(n, 360.0)
            cy = np.full(n, 200.0)
            jitter = 0.15
        offsets = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)
        for k in range(n_kpts):
            xy[slc, k, 0] = cx + offsets[k, 0] + rng.normal(0, jitter, n)
            xy[slc, k, 1] = cy + offsets[k, 1] + rng.normal(0, jitter, n)
    confidence = __import__("numpy").full((n_frames, n_kpts), 0.95)
    return PoseData(xy=xy, confidence=confidence, keypoint_names=names, fps=30.0)


def test_propose_clips_shapes_and_clip_lengths(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.gui.behavior.annotator.sampler import propose_clips
    from glider.vision.pose.dlc import to_dlc_csv

    pose = _make_three_regime_pose()
    pose_csv = tmp_path / "session.csv"
    to_dlc_csv(pose, pose_csv)

    clips = propose_clips(
        pose_csv=pose_csv,
        video_path="/synthetic/session.mp4",
        spec=FeatureSpec(body_axis=(0, pose.n_keypoints - 1)),
        window=15,
        n_clips=20,
        fps=30.0,
        clip_lengths_seconds=(0.4, 0.6, 0.8, 1.0),
    )
    assert len(clips) == 20

    # All clip seconds must come from the configured set.
    allowed = {0.4, 0.6, 0.8, 1.0}
    for c in clips:
        assert c.clip_seconds in allowed
        # Frame bounds are valid + non-empty.
        assert 0 <= c.start_frame < c.end_frame <= pose.n_frames
        # Duration in frames roughly matches clip_seconds * fps.
        expected = int(round(c.clip_seconds * 30.0))
        assert abs(c.duration_frames - expected) <= 1


def test_propose_clips_window_indices_are_diverse(tmp_path):
    """k-means++ should pick window indices that span the temporal axis,
    not cluster at one end."""
    from glider.analysis.behavior.features import FeatureSpec
    from glider.gui.behavior.annotator.sampler import propose_clips
    from glider.vision.pose.dlc import to_dlc_csv

    pose = _make_three_regime_pose()
    pose_csv = tmp_path / "session.csv"
    to_dlc_csv(pose, pose_csv)

    clips = propose_clips(
        pose_csv=pose_csv,
        video_path="/x.mp4",
        spec=FeatureSpec(body_axis=(0, pose.n_keypoints - 1)),
        window=10,
        n_clips=12,
        fps=30.0,
    )
    centers = [c.center_frame for c in clips]
    # At least one center from each of the three regimes.
    regions = [
        any(0 <= c < 200 for c in centers),
        any(200 <= c < 400 for c in centers),
        any(400 <= c < 600 for c in centers),
    ]
    assert all(regions), f"k-means++ missed a regime: {centers}"


def test_propose_clips_deterministic_with_seed(tmp_path):
    from glider.analysis.behavior.features import FeatureSpec
    from glider.gui.behavior.annotator.sampler import propose_clips
    from glider.vision.pose.dlc import to_dlc_csv

    pose = _make_three_regime_pose()
    pose_csv = tmp_path / "session.csv"
    to_dlc_csv(pose, pose_csv)

    spec = FeatureSpec(body_axis=(0, pose.n_keypoints - 1))
    a = propose_clips(
        pose_csv=pose_csv,
        video_path="/x.mp4",
        spec=spec,
        window=15,
        n_clips=10,
        fps=30.0,
        random_state=7,
    )
    b = propose_clips(
        pose_csv=pose_csv,
        video_path="/x.mp4",
        spec=spec,
        window=15,
        n_clips=10,
        fps=30.0,
        random_state=7,
    )
    assert [c.center_frame for c in a] == [c.center_frame for c in b]
    assert [c.clip_seconds for c in a] == [c.clip_seconds for c in b]


def test_propose_clips_caps_at_available_windows(tmp_path):
    """Asking for more clips than there are valid windows should yield
    only the available ones rather than crashing."""
    from glider.analysis.behavior.features import FeatureSpec
    from glider.gui.behavior.annotator.sampler import propose_clips
    from glider.vision.pose.dlc import to_dlc_csv

    pose = _make_three_regime_pose(n_frames=60)
    pose_csv = tmp_path / "session.csv"
    to_dlc_csv(pose, pose_csv)

    clips = propose_clips(
        pose_csv=pose_csv,
        video_path="/x.mp4",
        spec=FeatureSpec(body_axis=(0, pose.n_keypoints - 1)),
        window=30,
        n_clips=1000,  # absurd
        fps=30.0,
        min_frame_gap=0,  # disable the temporal-gap constraint for this test
    )
    # Definitely fewer than 1000.
    assert 0 < len(clips) <= 60


def test_propose_clips_respects_min_frame_gap(tmp_path):
    """No two picked clips should be within `min_frame_gap` frames of
    each other when the constraint is active."""
    import numpy as np

    from glider.analysis.behavior.features import FeatureSpec
    from glider.gui.behavior.annotator.sampler import propose_clips
    from glider.vision.pose.dlc import to_dlc_csv

    pose = _make_three_regime_pose(n_frames=900)
    pose_csv = tmp_path / "session.csv"
    to_dlc_csv(pose, pose_csv)

    clips = propose_clips(
        pose_csv=pose_csv,
        video_path="/x.mp4",
        spec=FeatureSpec(body_axis=(0, pose.n_keypoints - 1)),
        window=15,
        n_clips=10,
        fps=30.0,
        min_frame_gap=60,
    )
    indices = sorted(c.window_index for c in clips)
    # Pairwise gaps must all be >= the constraint.
    diffs = np.diff(indices)
    assert (
        diffs >= 60
    ).all(), f"some clips were closer than 60 frames apart: gaps = {diffs.tolist()}"


def test_multi_video_window_loads_per_video_stores(tmp_path):
    """Smoke test: AnnotatorWindow takes a `videos_meta` dict and loads one
    AnnotationStore per video. Skips if PyQt6 isn't installed.
    """
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    # Two videos, each with a pre-existing annotations CSV.
    a_csv = tmp_path / "a_annotations.csv"
    b_csv = tmp_path / "b_annotations.csv"
    store_a = AnnotationStore([BehaviorZone("rear", 0, 10)])
    store_b = AnnotationStore([BehaviorZone("groom", 100, 120)])
    store_a.save_csv(a_csv)
    store_b.save_csv(b_csv)

    videos_meta = {tmp_path / "a.mp4": a_csv, tmp_path / "b.mp4": b_csv}
    clips = [
        ProposedClip(0, 5, 0, 10, 0.5, str(tmp_path / "a.mp4")),
        ProposedClip(0, 110, 100, 120, 0.7, str(tmp_path / "b.mp4")),
    ]
    _app = QApplication.instance() or QApplication([])
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta)
    assert len(w.stores) == 2
    assert any(z.behavior == "rear" for z in w.stores[tmp_path / "a.mp4"])
    assert any(z.behavior == "groom" for z in w.stores[tmp_path / "b.mp4"])


def test_apply_label_routes_to_correct_video_store(tmp_path):
    """A label applied to a clip from video A lands in video A's store only."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    # Two empty annotation CSVs.
    a_csv = tmp_path / "a_annotations.csv"
    b_csv = tmp_path / "b_annotations.csv"
    AnnotationStore().save_csv(a_csv)
    AnnotationStore().save_csv(b_csv)

    vocab = Vocabulary()
    vocab.add(Behavior(name="rear", hotkey="r", color="#000"))

    videos_meta = {tmp_path / "a.mp4": a_csv, tmp_path / "b.mp4": b_csv}
    clip_from_a = ProposedClip(0, 5, 0, 10, 0.5, str(tmp_path / "a.mp4"))
    clip_from_b = ProposedClip(0, 110, 100, 120, 0.7, str(tmp_path / "b.mp4"))
    clips = [clip_from_a, clip_from_b]
    _app = QApplication.instance() or QApplication([])
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta, vocab=vocab)

    # Navigate to clip 0 (video A) and label it.
    w.current = 0
    w._apply_label("rear")

    assert any(z.behavior == "rear" for z in w.stores[tmp_path / "a.mp4"])
    assert not list(w.stores[tmp_path / "b.mp4"])


def test_save_writes_to_correct_per_video_csv(tmp_path):
    """Saving routes through `_save_annotations_for_video` and writes only the
    affected video's annotations CSV."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    a_csv = tmp_path / "a_annotations.csv"
    b_csv = tmp_path / "b_annotations.csv"
    AnnotationStore().save_csv(a_csv)
    AnnotationStore().save_csv(b_csv)

    vocab = Vocabulary()
    vocab.add(Behavior(name="rear", hotkey="r", color="#000"))

    videos_meta = {tmp_path / "a.mp4": a_csv, tmp_path / "b.mp4": b_csv}
    clips = [ProposedClip(0, 5, 0, 10, 0.5, str(tmp_path / "a.mp4"))]
    _app = QApplication.instance() or QApplication([])
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta, vocab=vocab)
    w.current = 0
    w._apply_label("rear")

    # A's CSV should now contain the rear zone.
    loaded_a = AnnotationStore.load_csv(a_csv)
    assert any(z.behavior == "rear" for z in loaded_a)
    # B's CSV should be untouched (mtime equality is fragile across filesystems;
    # check that loading it yields zero zones, which is the meaningful invariant).
    loaded_b = AnnotationStore.load_csv(b_csv)
    assert list(loaded_b) == []


def test_load_any_returns_the_record_whatever_the_inputs(tmp_path):
    """Resuming must not depend on today's settings matching yesterday's.

    ResumeCache.load() is hash-gated, which is right for "reuse this
    automatically". Deliberately clicking Resume is a different question: the
    operator is asking for whatever queue is saved, not for a queue matching
    the checkboxes currently on screen.
    """
    from glider.gui.behavior.annotator.resume_cache import ResumeCache

    cache = ResumeCache(tmp_path)
    cache.save(inputs={"n_clips": 100, "exclude_labeled": False}, clip_payload=[{"a": 1}])

    assert cache.load(inputs={"n_clips": 250}) is None  # hash-gated, as designed
    record = cache.load_any()
    assert record is not None
    assert record["clips"] == [{"a": 1}]
    assert record["inputs"]["n_clips"] == 100


def test_load_any_is_none_when_there_is_no_queue(tmp_path):
    from glider.gui.behavior.annotator.resume_cache import ResumeCache

    assert ResumeCache(tmp_path).load_any() is None


def test_load_any_survives_a_corrupt_file(tmp_path):
    from glider.gui.behavior.annotator.resume_cache import CACHE_FILENAME, ResumeCache

    (tmp_path / CACHE_FILENAME).write_text("{not json", encoding="utf-8")
    assert ResumeCache(tmp_path).load_any() is None


def test_build_review_clips_loads_zones(tmp_path):
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.app import build_review_clips

    a = tmp_path / "a.mp4"
    a_csv = tmp_path / "a_annotations.csv"
    AnnotationStore([BehaviorZone("groom", 100, 120), BehaviorZone("sniff", 200, 230)]).save_csv(
        a_csv
    )
    clips = build_review_clips({a: a_csv}, fps=30.0)
    assert len(clips) == 2
    assert {(c.start_frame, c.end_frame) for c in clips} == {(100, 120), (200, 230)}


def testmake_more_sampler_advances_seed(tmp_path):
    """The render-more sampler returns clips and advances its seed each call
    so repeated presses surface different picks."""
    from glider.analysis.behavior.features import FeatureSpec
    from glider.gui.behavior.annotator.app import make_more_sampler
    from glider.vision.pose.dlc import to_dlc_csv

    pose = _make_three_regime_pose()
    pose_csv = tmp_path / "s.csv"
    to_dlc_csv(pose, pose_csv)
    video = tmp_path / "s.mp4"
    sampler = make_more_sampler(
        [(video, pose_csv)],
        spec=FeatureSpec(body_axis=(0, pose.n_keypoints - 1)),
        window=15,
        fps=30.0,
        spatial_weight=1.0,
        min_frame_gap=None,
        base_seed=100,
    )
    a = sampler(5)
    b = sampler(5)
    assert 0 < len(a) <= 5
    assert all(hasattr(c, "center_frame") for c in a)
    assert [c.center_frame for c in a] != [c.center_frame for c in b]


def test_merge_behavior_zones_renames_and_unions_overlap():
    from glider.analysis.behavior.annotations import (
        BehaviorZone,
        merge_behavior_zones,
    )

    zones = [
        BehaviorZone("grooming", 100, 120),
        BehaviorZone("flank groom", 115, 140),  # overlaps grooming after rename
        BehaviorZone("locomote", 200, 230),
    ]
    out = merge_behavior_zones(zones, ["flank groom"], "grooming")
    groom = sorted((z for z in out if z.behavior == "grooming"), key=lambda z: z.start_frame)
    assert [(z.start_frame, z.end_frame) for z in groom] == [(100, 140)]
    assert any(z.behavior == "locomote" and (z.start_frame, z.end_frame) == (200, 230) for z in out)
    assert not any(z.behavior == "flank groom" for z in out)


def test_merge_behavior_zones_keeps_nonoverlapping_separate():
    from glider.analysis.behavior.annotations import (
        BehaviorZone,
        merge_behavior_zones,
    )

    zones = [
        BehaviorZone("grooming", 100, 120),
        BehaviorZone("flank groom", 200, 230),  # disjoint
    ]
    out = merge_behavior_zones(zones, ["flank groom"], "grooming")
    groom = sorted((z for z in out if z.behavior == "grooming"), key=lambda z: z.start_frame)
    assert [(z.start_frame, z.end_frame) for z in groom] == [(100, 120), (200, 230)]


def test_merge_behavior_zones_multi_source():
    from glider.analysis.behavior.annotations import (
        BehaviorZone,
        merge_behavior_zones,
    )

    zones = [
        BehaviorZone("sniff", 10, 20),
        BehaviorZone("rearing", 18, 30),  # overlaps sniff after rename
        BehaviorZone("dig", 50, 60),
    ]
    out = merge_behavior_zones(zones, ["rearing", "dig"], "sniff")
    sniff = sorted((z for z in out if z.behavior == "sniff"), key=lambda z: z.start_frame)
    assert [(z.start_frame, z.end_frame) for z in sniff] == [(10, 30), (50, 60)]
    assert not any(z.behavior in ("rearing", "dig") for z in out)


def test_zones_to_clips_maps_bounds_center_seconds():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.sampler import zones_to_clips

    store = AnnotationStore([BehaviorZone("groom", 100, 130)])
    clips = zones_to_clips(store, "/x.mp4", fps=30.0)
    assert len(clips) == 1
    c = clips[0]
    assert (c.start_frame, c.end_frame) == (100, 130)
    assert c.center_frame == 115
    assert abs(c.clip_seconds - 1.0) < 1e-9  # 30 frames @ 30 fps
    assert c.video_path == "/x.mp4"


def test_zones_to_clips_sorted_by_start():
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.sampler import zones_to_clips

    store = AnnotationStore(
        [
            BehaviorZone("a", 300, 320),
            BehaviorZone("b", 50, 70),
            BehaviorZone("c", 150, 160),
        ]
    )
    clips = zones_to_clips(store, "/x.mp4", fps=30.0)
    assert [c.start_frame for c in clips] == [50, 150, 300]


def test_apply_label_saves_trimmed_bounds(tmp_path):
    """A label saves the trim bar's current [in,out), not the proposed clip."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    a_csv = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(a_csv)
    vocab = Vocabulary()
    vocab.add(Behavior(name="rear", hotkey="r", color="#000"))

    videos_meta = {tmp_path / "a.mp4": a_csv}
    clips = [ProposedClip(0, 5, 0, 10, 0.5, str(tmp_path / "a.mp4"))]
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta, vocab=vocab)

    w.current = 0
    w.trim_bar.set_bounds(3, 7)
    w._apply_label("rear")

    zones = [z for z in w.stores[tmp_path / "a.mp4"] if z.behavior == "rear"]
    assert len(zones) == 1
    assert (zones[0].start_frame, zones[0].end_frame) == (3, 7)


def test_relabel_replaces_prior_trimmed_zone(tmp_path):
    """Re-labeling the same clip removes its prior zone (via the clip→zone
    map) and saves the new trim — no same-behavior OverlapError, no dup."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    a_csv = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(a_csv)
    vocab = Vocabulary()
    vocab.add(Behavior(name="rear", hotkey="r", color="#000"))

    videos_meta = {tmp_path / "a.mp4": a_csv}
    clips = [ProposedClip(0, 5, 0, 10, 0.5, str(tmp_path / "a.mp4"))]
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta, vocab=vocab)

    w.current = 0
    w.trim_bar.set_bounds(3, 7)
    w._apply_label("rear")
    w.current = 0
    w.trim_bar.set_bounds(2, 9)
    w._apply_label("rear")

    zones = [z for z in w.stores[tmp_path / "a.mp4"] if z.behavior == "rear"]
    assert len(zones) == 1
    assert (zones[0].start_frame, zones[0].end_frame) == (2, 9)


def test_startup_seeds_clip_zone_map_from_existing_overlap(tmp_path):
    """A pre-labeled (trimmed) zone is re-associated with its proposed clip
    on load, so the clip counts as labeled and is skipped."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    a_csv = tmp_path / "a_annotations.csv"
    # Trimmed zone (4,8) overlaps the proposed clip0 (0,10) but not clip1.
    AnnotationStore([BehaviorZone("rear", 4, 8)]).save_csv(a_csv)

    videos_meta = {tmp_path / "a.mp4": a_csv}
    clips = [
        ProposedClip(0, 5, 0, 10, 0.5, str(tmp_path / "a.mp4")),
        ProposedClip(1, 105, 100, 110, 0.5, str(tmp_path / "a.mp4")),
    ]
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta)

    assert 0 in w._clip_zone
    assert 1 not in w._clip_zone
    # First unlabeled clip is clip 1, since clip 0 is already labeled.
    assert w._first_unlabeled_index() == 1


def test_render_more_appends_new_and_filters_labelled_regions(tmp_path):
    """Render-more appends sampled clips, dropping any centered inside an
    existing zone, and jumps to the first new clip."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip, zones_to_clips

    a = tmp_path / "a.mp4"
    a_csv = tmp_path / "a_annotations.csv"
    store = AnnotationStore([BehaviorZone("groom", 100, 120)])
    store.save_csv(a_csv)
    videos_meta = {a: a_csv}
    clips = zones_to_clips(store, a, fps=30.0)  # one labelled clip [100,120]

    def fake_sampler(n):
        return [
            ProposedClip(0, 110, 105, 115, 0.3, str(a)),  # inside groom → drop
            ProposedClip(0, 300, 295, 305, 0.3, str(a)),  # fresh → keep
        ]

    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta, clip_sampler=fake_sampler)
    before = len(w.clips)
    w._render_more_clips(2)
    assert len(w.clips) == before + 1
    assert w.clips[-1].center_frame == 300
    assert w.current == before  # jumped to the first new clip


def test_render_more_disabled_without_sampler(tmp_path):
    """No sampler injected → no render button (review-only)."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import zones_to_clips

    a = tmp_path / "a.mp4"
    a_csv = tmp_path / "a_annotations.csv"
    store = AnnotationStore([BehaviorZone("groom", 100, 120)])
    store.save_csv(a_csv)
    clips = zones_to_clips(store, a, fps=30.0)

    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = AnnotatorWindow(clips=clips, videos_meta={a: a_csv}, clip_sampler=None)
    assert w.render_button is None


def test_merge_behavior_relabels_across_stores_and_persists(tmp_path):
    """GUI merge renames zones in every video's store, unions overlaps,
    removes the source behavior, and writes the CSVs."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore, BehaviorZone
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a_csv = tmp_path / "a_annotations.csv"
    b_csv = tmp_path / "b_annotations.csv"
    AnnotationStore([BehaviorZone("flank groom", 100, 120)]).save_csv(a_csv)
    AnnotationStore(
        [BehaviorZone("grooming", 50, 70), BehaviorZone("flank groom", 60, 90)]
    ).save_csv(b_csv)
    vocab = Vocabulary()
    vocab.add(Behavior(name="grooming", hotkey="1", color="#000"))
    vocab.add(Behavior(name="flank groom", hotkey="2", color="#111"))

    videos_meta = {a: a_csv, b: b_csv}
    clips = [ProposedClip(0, 110, 100, 120, 0.5, str(a))]
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta, vocab=vocab)

    w._merge_behavior(["flank groom"], "grooming")

    store_a = sorted((z.start_frame, z.end_frame, z.behavior) for z in w.stores[a])
    store_b = sorted((z.start_frame, z.end_frame, z.behavior) for z in w.stores[b])
    assert store_a == [(100, 120, "grooming")]
    assert store_b == [(50, 90, "grooming")]  # 50-70 + 60-90 unioned
    assert "flank groom" not in w.vocab
    # Persisted to disk.
    reloaded = AnnotationStore.load_csv(b_csv)
    assert not any(z.behavior == "flank groom" for z in reloaded)
    assert any(z.behavior == "grooming" for z in reloaded)


def test_big_label_shows_current_clip_label(tmp_path):
    """The current clip's label is mirrored in the big top-right display."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.analysis.behavior.annotations import AnnotationStore
    from glider.analysis.behavior.vocabulary import Behavior, Vocabulary
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    a_csv = tmp_path / "a_annotations.csv"
    AnnotationStore().save_csv(a_csv)
    vocab = Vocabulary()
    vocab.add(Behavior(name="rear", hotkey="r", color="#000"))

    videos_meta = {tmp_path / "a.mp4": a_csv}
    clips = [ProposedClip(0, 5, 0, 10, 0.5, str(tmp_path / "a.mp4"))]
    app = QApplication.instance() or QApplication([])  # noqa: F841
    w = AnnotatorWindow(clips=clips, videos_meta=videos_meta, vocab=vocab)

    # Unlabeled clip → muted placeholder.
    assert w.big_label.text() == "—"

    w.current = 0
    w._apply_label("rear")
    assert w.big_label.text() == "rear"


def test_clip_player_set_loop_bounds_updates_range():
    """The player can change its loop [in,out) live without a capture."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        pytest.skip("PyQt6 required")
    from glider.gui.behavior.annotator.clip_player import ClipPlayer

    app = QApplication.instance() or QApplication([])  # noqa: F841 (keep ref)
    player = ClipPlayer()
    player.set_loop_bounds(100, 150)
    assert player.loop_bounds() == (100, 150)
    # Out must stay above in.
    player.set_loop_bounds(200, 100)
    in_f, out_f = player.loop_bounds()
    assert in_f < out_f


def test_propose_clips_spatial_weight_pushes_picks_apart_spatially(tmp_path):
    """Crank up spatial_weight on a fixture where behavior is uniform
    but the mouse occupies different arena positions; the sampler
    should cover the spatial extent."""
    import numpy as np

    from glider.analysis.behavior.features import FeatureSpec
    from glider.gui.behavior.annotator.sampler import propose_clips
    from glider.vision.pose.core import PoseData
    from glider.vision.pose.dlc import to_dlc_csv

    # Build a pose where the mouse just walks left to right at constant
    # geometry — behavior is uniform, only location changes.
    rng = np.random.default_rng(0)
    n_frames = 600
    n_kpts = 5
    names = ["snout", "left_ear", "right_ear", "neck", "tail_base"]
    xy = np.empty((n_frames, n_kpts, 2))
    cx = np.linspace(50, 950, n_frames)  # mouse traverses the arena
    offsets = np.array([[0, -20], [-10, -10], [10, -10], [0, 0], [0, 30]], dtype=float)
    for k in range(n_kpts):
        xy[:, k, 0] = cx + offsets[k, 0] + rng.normal(0, 0.3, n_frames)
        xy[:, k, 1] = 200 + offsets[k, 1] + rng.normal(0, 0.3, n_frames)
    conf = np.full((n_frames, n_kpts), 0.95)
    pose = PoseData(xy=xy, confidence=conf, keypoint_names=names, fps=30.0)
    pose_csv = tmp_path / "walking.csv"
    to_dlc_csv(pose, pose_csv)

    spec = FeatureSpec(body_axis=(0, n_kpts - 1))
    clips = propose_clips(
        pose_csv=pose_csv,
        video_path="/x.mp4",
        spec=spec,
        window=15,
        n_clips=8,
        fps=30.0,
        spatial_weight=2.0,  # crank it
        min_frame_gap=10,
    )
    # All temporal centers should be reasonably spread (since the mouse
    # moves left-to-right, temporal == spatial here).
    centers = sorted(c.center_frame for c in clips)
    assert (
        centers[-1] - centers[0] > 0.5 * n_frames
    ), "spatially-weighted sampler clustered picks in one half of the video"


# ---------------------------------------------------------------------------
# Unreadable annotation files
#
# load_csv returns an empty store for a MISSING file (the normal first-run
# case). A malformed file is different: it holds labelling work we can't
# read, and save_csv truncates, so treating the two the same meant the first
# keypress destroyed the file with nothing on screen saying so.
# ---------------------------------------------------------------------------


def _window_over_a_corrupt_csv(qtbot, tmp_path):
    """One video whose annotations CSV has the wrong columns entirely."""
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    corrupt = tmp_path / "a_annotations.csv"
    corrupt.write_text("something,else\n1,2\n")
    video = tmp_path / "a.mp4"
    clips = [ProposedClip(0, 5, 0, 10, 0.5, str(video))]

    window = AnnotatorWindow(clips=clips, videos_meta={video: corrupt})
    qtbot.addWidget(window)
    return window, video, corrupt


def test_unreadable_annotations_are_recorded_not_swallowed(qtbot, tmp_path):
    window, video, _corrupt = _window_over_a_corrupt_csv(qtbot, tmp_path)

    assert video in window.load_errors
    assert "missing required columns" in window.load_errors[video]


def test_a_missing_file_is_not_an_error(qtbot, tmp_path):
    """Opening a video for the first time must stay perfectly saveable."""
    from glider.gui.behavior.annotator.main_window import AnnotatorWindow
    from glider.gui.behavior.annotator.sampler import ProposedClip

    video = tmp_path / "fresh.mp4"
    ann = tmp_path / "fresh_annotations.csv"  # deliberately never created
    clips = [ProposedClip(0, 5, 0, 10, 0.5, str(video))]

    window = AnnotatorWindow(clips=clips, videos_meta={video: ann})
    qtbot.addWidget(window)

    assert window.load_errors == {}
    assert window.load_error_message() == ""


def test_saving_never_truncates_a_file_it_could_not_read(qtbot, tmp_path):
    """The regression: this used to overwrite the file with an empty store."""
    window, video, corrupt = _window_over_a_corrupt_csv(qtbot, tmp_path)
    before = corrupt.read_text()

    window._save_annotations_for_video(video)

    assert corrupt.read_text() == before
    assert "unreadable" in window.save_indicator.text()


def test_labelling_is_refused_rather_than_silently_dropped(qtbot, tmp_path, monkeypatch):
    """Accepting labels that can never be written would cost a whole session."""
    from glider.gui.behavior.annotator import main_window as mw

    window, video, corrupt = _window_over_a_corrupt_csv(qtbot, tmp_path)
    before = corrupt.read_text()

    shown: list[str] = []
    monkeypatch.setattr(
        mw.QMessageBox, "warning", lambda *a, **k: shown.append(a[2] if len(a) > 2 else "")
    )
    window._apply_label("rearing")

    assert shown and "reopen the annotator" in shown[0]
    assert corrupt.read_text() == before
    assert not window.stores[video]


def test_the_warning_names_every_unreadable_file(qtbot, tmp_path):
    window, _video, corrupt = _window_over_a_corrupt_csv(qtbot, tmp_path)

    message = window.load_error_message()
    assert corrupt.name in message
    assert "left untouched" in message
