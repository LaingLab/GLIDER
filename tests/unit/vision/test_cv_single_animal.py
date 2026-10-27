"""Single-animal mode (issue #218): only the most confident detection is tracked."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from glider.vision.cv_processor import (
    SINGLE_ANIMAL_TRACK_ID,
    CVProcessor,
    CVSettings,
    Detection,
    DetectionBackend,
    keep_most_confident,
)


def _det(x: int, conf: float, track_id: int | None = None) -> Detection:
    return Detection(
        class_id=0, class_name="mouse", confidence=conf, bbox=(x, 10, 10, 10), track_id=track_id
    )


def _frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


# --- keep_most_confident -----------------------------------------------------


def test_keeps_only_the_most_confident_detection():
    low, high, mid = _det(0, 0.4), _det(20, 0.9), _det(40, 0.6)
    assert keep_most_confident([low, high, mid]) == [high]


def test_empty_and_single_frames_pass_through():
    only = _det(0, 0.3)
    assert keep_most_confident([]) == []
    assert keep_most_confident([only]) == [only]


def test_ties_keep_the_earliest_detection():
    first, second = _det(0, 0.8), _det(20, 0.8)
    assert keep_most_confident([first, second])[0] is first


def test_keypoints_travel_with_the_kept_detection():
    low, high = _det(0, 0.4), _det(20, 0.9)
    low._keypoints = np.zeros((3, 3))
    high._keypoints = np.ones((3, 3))
    (kept,) = keep_most_confident([low, high])
    assert np.array_equal(kept._keypoints, np.ones((3, 3)))


def test_does_not_mutate_the_input_list():
    dets = [_det(0, 0.4), _det(20, 0.9)]
    keep_most_confident(dets)
    assert len(dets) == 2


# --- CVSettings persistence --------------------------------------------------


def test_single_animal_defaults_off_and_round_trips():
    assert CVSettings().single_animal is False
    on = CVSettings(single_animal=True)
    assert on.to_dict()["single_animal"] is True
    assert CVSettings.from_dict(on.to_dict()).single_animal is True
    assert on.copy().single_animal is True


def test_files_saved_before_the_setting_stay_multi_animal():
    data = CVSettings().to_dict()
    del data["single_animal"]
    assert CVSettings.from_dict(data).single_animal is False


# --- CVProcessor integration -------------------------------------------------


def _init(proc: CVProcessor) -> CVProcessor:
    """Bring the processor up without loading a model.

    Background subtraction initializes for real (it builds the centroid
    tracker); ByteTrack would try to load weights, so it is only marked ready
    and ``_detect`` is stubbed to stand in for the model.
    """
    if proc.settings.backend == DetectionBackend.YOLO_BYTETRACK:
        proc._initialized = True
    else:
        assert proc.initialize()
    return proc


def _processor(single_animal: bool, backend: DetectionBackend, dets: list[Detection]):
    proc = _init(CVProcessor(CVSettings(backend=backend, single_animal=single_animal)))
    proc._detect = MagicMock(spec=proc._detect, side_effect=lambda *_a, **_k: list(dets))
    return proc


def test_processor_tracks_every_detection_when_off():
    dets = [_det(0, 0.4), _det(40, 0.9)]
    proc = _processor(False, DetectionBackend.BACKGROUND_SUBTRACTION, dets)
    detections, tracked, _ = proc.process_frame(_frame(), 0.0)
    assert len(detections) == 2
    assert len(tracked) == 2


def test_processor_tracks_only_the_most_confident_when_on():
    dets = [_det(0, 0.4), _det(40, 0.9)]
    proc = _processor(True, DetectionBackend.BACKGROUND_SUBTRACTION, dets)
    detections, tracked, _ = proc.process_frame(_frame(), 0.0)
    assert [d.confidence for d in detections] == [0.9]
    assert [t.confidence for t in tracked] == [0.9]


def test_centroid_track_id_is_stable_when_the_best_detection_hops():
    """The winner jumps across the arena; it is still one animal, one ID."""
    proc = _init(
        CVProcessor(CVSettings(backend=DetectionBackend.BACKGROUND_SUBTRACTION, single_animal=True))
    )
    frames = [
        [_det(0, 0.9), _det(40, 0.5)],
        [_det(0, 0.5), _det(40, 0.9)],
        [_det(2, 0.9), _det(42, 0.5)],
    ]
    proc._detect = MagicMock(spec=proc._detect, side_effect=frames)
    ids = [[t.track_id for t in proc.process_frame(_frame(), float(i))[1]] for i in range(3)]
    assert ids == [[0], [0], [0]]


def test_bytetrack_ids_are_pinned_when_the_best_track_changes():
    proc = _init(
        CVProcessor(CVSettings(backend=DetectionBackend.YOLO_BYTETRACK, single_animal=True))
    )
    frames = [
        [_det(0, 0.9, track_id=3), _det(40, 0.5, track_id=7)],
        [_det(0, 0.5, track_id=3), _det(40, 0.9, track_id=7)],
        [_det(40, 0.8, track_id=12)],  # ByteTrack re-acquired under a new ID
    ]
    proc._detect = MagicMock(spec=proc._detect, side_effect=frames)
    results = [proc.process_frame(_frame(), float(i))[1] for i in range(3)]
    assert [[t.track_id for t in r] for r in results] == [[SINGLE_ANIMAL_TRACK_ID]] * 3
    # One identity, so its age keeps counting across the hop and the re-acquire.
    assert [r[0].age for r in results] == [1, 2, 3]


def test_bytetrack_boxes_without_an_id_are_still_dropped():
    proc = _processor(True, DetectionBackend.YOLO_BYTETRACK, [_det(0, 0.9), _det(40, 0.5)])
    detections, tracked, _ = proc.process_frame(_frame(), 0.0)
    assert detections[0].track_id is None
    assert tracked == []


def test_bytetrack_ids_are_untouched_when_off():
    dets = [_det(0, 0.9, track_id=3), _det(40, 0.5, track_id=7)]
    proc = _processor(False, DetectionBackend.YOLO_BYTETRACK, dets)
    _, tracked, _ = proc.process_frame(_frame(), 0.0)
    assert sorted(t.track_id for t in tracked) == [3, 7]
