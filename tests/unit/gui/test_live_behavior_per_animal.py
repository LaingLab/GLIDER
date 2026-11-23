"""Per-animal live behaviour: one classifier stream per named animal.

With identity on, the CV processor's tracker already hands back each named
animal's keypoints every frame, so ``AnimalBehaviorClassifiers`` runs the same
extractor/buffer/model chain as ``LiveBehaviorClassifier`` once per name.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.gui.panels.live_behavior import (
    AnimalBehaviorClassifiers,
    KeypointMismatchError,
    LiveBehaviorClassifier,
    model_keypoint_count,
)
from glider.vision.cv_processor import TrackedObject
from glider.vision.pose.backend import UltralyticsBackend


def _names(model) -> list[str]:
    return [f"k{i}" for i in range(model_keypoint_count(model))]


def _kps(k: int, t: int, speed: float) -> np.ndarray:
    base = np.stack([np.linspace(0.0, 10.0, k), np.linspace(0.0, 5.0, k)], axis=1)
    return base + speed * t + np.sin(t * speed) * np.arange(k)[:, None]


def _obj(name: str, kps, visible: bool = True) -> TrackedObject:
    return TrackedObject(
        track_id=0,
        class_name="mouse",
        bbox=(0, 0, 1, 1),
        confidence=1.0,
        centroid=(0, 0),
        keypoints=kps,
        animal=name,
        visible=visible,
    )


def _reference(model, frames):
    """A single-animal classifier fed the same keypoint stream."""

    class _NoYolo:
        def predict(self, *a, **k):  # pragma: no cover - never called
            raise AssertionError

    ref = LiveBehaviorClassifier(model, UltralyticsBackend(_NoYolo(), _names(model)))
    return [ref.classify_keypoints(kps) for kps in frames]


def test_two_animals_get_independent_streams(tiny_behavior_model):
    model = tiny_behavior_model
    k = model_keypoint_count(model)
    clf = AnimalBehaviorClassifiers(model, _names(model), ["Dam", "Pup"])
    n = 20
    dam = [_kps(k, t, 0.5) for t in range(n)]
    pup = [_kps(k, t, 3.0) for t in range(n)]

    got = {"Dam": [], "Pup": []}
    for t in range(n):
        # Order in the list must not matter; extras (animal None) are ignored.
        events = clf.push(
            [_obj("Pup", pup[t]), _obj(None, dam[t]), _obj("Dam", dam[t])], float(t), t
        )
        assert [e.animal for e in events] == ["Dam", "Pup"]
        assert all(e.visible for e in events)
        assert all(e.frame_index == t and e.timestamp == float(t) for e in events)
        for e in events:
            got[e.animal].append(e.behavior)

    assert got["Dam"] == _reference(model, dam)
    assert got["Pup"] == _reference(model, pup)
    assert got["Dam"][-1] in set(model.classes)


def test_hidden_animal_holds_label_and_keeps_its_window(tiny_behavior_model):
    model = tiny_behavior_model
    k = model_keypoint_count(model)
    clf = AnimalBehaviorClassifiers(model, _names(model), ["Dam"])
    n = 15
    for t in range(n):
        (last,) = clf.push([_obj("Dam", _kps(k, t, 1.0))], float(t), t)
    assert last.behavior in set(model.classes)

    (hidden,) = clf.push([_obj("Dam", _kps(k, n, 1.0), visible=False)], float(n), n)
    assert hidden.visible is False
    assert hidden.behavior == last.behavior
    assert hidden.animal == "Dam"

    # A missing named animal counts as hidden too.
    (gone,) = clf.push([], float(n + 1), n + 1)
    assert gone.visible is False and gone.behavior == last.behavior

    # Back in view: the extractor saw NaN rows (not a reset), so the window is
    # still full and a label comes straight back without a fresh warm-up.
    (back,) = clf.push([_obj("Dam", _kps(k, n + 2, 1.0))], float(n + 2), n + 2)
    assert back.visible is True
    assert clf._streams["Dam"][1].is_full()
    assert back.behavior != ""


def test_keypoints_with_confidence_column_are_gated(tiny_behavior_model):
    model = tiny_behavior_model
    k = model_keypoint_count(model)
    clf = AnimalBehaviorClassifiers(model, _names(model), ["Dam"])
    kps3 = np.concatenate([_kps(k, 0, 1.0), np.zeros((k, 1))], axis=1)
    (event,) = clf.push([_obj("Dam", kps3)], 0.0, 0)
    assert np.isnan(event.keypoints).all()
    assert event.keypoints.shape == (k, 2)


def test_reset_rearms_warmup(tiny_behavior_model):
    model = tiny_behavior_model
    k = model_keypoint_count(model)
    clf = AnimalBehaviorClassifiers(model, _names(model), ["Dam"])
    for t in range(15):
        clf.push([_obj("Dam", _kps(k, t, 1.0))], float(t), t)
    clf.reset()
    (event,) = clf.push([_obj("Dam", _kps(k, 0, 1.0))], 0.0, 0)
    assert event.behavior == ""


def test_wrong_keypoint_names_rejected(tiny_behavior_model):
    with pytest.raises(KeypointMismatchError):
        AnimalBehaviorClassifiers(tiny_behavior_model, ["a", "b"], ["Dam"])


def test_cv_processor_forwards_events_and_needs_every_frame(tiny_behavior_model, monkeypatch):
    from glider.vision.cv_processor import CVProcessor, CVSettings, DetectionBackend

    model = tiny_behavior_model
    k = model_keypoint_count(model)
    settings = CVSettings(
        backend=DetectionBackend.YOLO_V8,
        identity_mode="appearance",
        animals=[{"name": "Dam", "class_name": None}],
    )
    cv = CVProcessor(settings)
    cv._initialized = True
    monkeypatch.setattr(cv, "_detect_yolo", lambda frame: [])
    monkeypatch.setattr(
        cv,
        "_get_identity",
        lambda: type("I", (), {"update": lambda s, d: [_obj("Dam", _kps(k, 0, 1.0))]})(),
    )

    got = []
    cv.on_animal_behavior(got.append)
    cv.process_frame(np.zeros((8, 8, 3), np.uint8), 1.0)
    assert got == [] and not cv.needs_every_frame

    cv.set_animal_classifiers(AnimalBehaviorClassifiers(model, _names(model), ["Dam"]))
    assert cv.needs_every_frame
    cv.process_frame(np.zeros((8, 8, 3), np.uint8), 2.0)
    assert [e.animal for e in got] == ["Dam"]

    cv.set_animal_classifiers(None)
    assert not cv.needs_every_frame
