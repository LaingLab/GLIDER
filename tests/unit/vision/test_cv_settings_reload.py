"""CVSettings reload + serialization contracts.

``update_settings`` previously reinitialized only when the *backend* enum
changed. Picking different weights in the settings dialog therefore left the
old model loaded while the UI displayed the new path — one model's detections
silently attributed to another. It must reload on a model-path change too.
"""

from __future__ import annotations

import pytest

from glider.vision.cv_processor import (
    CVProcessor,
    CVSettings,
    DetectionBackend,
    parse_keypoint_names,
)


@pytest.fixture
def spy_init(monkeypatch):
    """Build a processor whose initialize() is recorded rather than run."""

    def _make(settings: CVSettings) -> tuple[CVProcessor, list]:
        proc = CVProcessor(settings)
        calls: list[bool] = []
        monkeypatch.setattr(proc, "initialize", lambda: calls.append(True))
        return proc, calls

    return _make


def test_model_path_change_triggers_reload(spy_init):
    proc, calls = spy_init(CVSettings(backend=DetectionBackend.YOLO_V8, model_path="old.pt"))
    proc.update_settings(CVSettings(backend=DetectionBackend.YOLO_V8, model_path="new.pt"))
    assert calls, "changing model_path must reload the model"


def test_bytetrack_model_path_change_triggers_reload(spy_init):
    proc, calls = spy_init(CVSettings(backend=DetectionBackend.YOLO_BYTETRACK, model_path="old.pt"))
    proc.update_settings(CVSettings(backend=DetectionBackend.YOLO_BYTETRACK, model_path="new.pt"))
    assert calls


def test_backend_change_still_triggers_reload(spy_init):
    proc, calls = spy_init(CVSettings(backend=DetectionBackend.BACKGROUND_SUBTRACTION))
    proc.update_settings(CVSettings(backend=DetectionBackend.MOTION_ONLY))
    assert calls


def test_unchanged_model_path_does_not_reload(spy_init):
    """Reloading a model is expensive; don't do it on unrelated edits."""
    proc, calls = spy_init(CVSettings(backend=DetectionBackend.YOLO_V8, model_path="same.pt"))
    proc.update_settings(
        CVSettings(
            backend=DetectionBackend.YOLO_V8,
            model_path="same.pt",
            confidence_threshold=0.75,
        )
    )
    assert not calls


def test_model_path_ignored_for_backend_that_loads_no_model(spy_init):
    """Background subtraction never reads model_path — don't reinit for it."""
    proc, calls = spy_init(
        CVSettings(backend=DetectionBackend.BACKGROUND_SUBTRACTION, model_path="a.pt")
    )
    proc.update_settings(
        CVSettings(backend=DetectionBackend.BACKGROUND_SUBTRACTION, model_path="b.pt")
    )
    assert not calls


# --- keypoint_names ---------------------------------------------------------


def test_keypoint_names_round_trip():
    settings = CVSettings(keypoint_names=["nose", "left_ear", "right_ear"])
    restored = CVSettings.from_dict(settings.to_dict())
    assert restored.keypoint_names == ["nose", "left_ear", "right_ear"]


def test_keypoint_names_default_empty():
    assert CVSettings().keypoint_names == []
    assert CVSettings.from_dict({}).keypoint_names == []


def test_keypoint_names_not_shared_between_instances():
    """A mutable default would alias every CVSettings to one list."""
    a = CVSettings()
    b = CVSettings()
    a.keypoint_names.append("nose")
    assert b.keypoint_names == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("nose, left_ear, right_ear", ["nose", "left_ear", "right_ear"]),
        ("nose,left_ear", ["nose", "left_ear"]),
        ("  nose ,  tail  ", ["nose", "tail"]),
        ("nose,, tail", ["nose", "tail"]),
        ("", []),
        ("   ", []),
        (",,,", []),
        ("nose", ["nose"]),
    ],
)
def test_parse_keypoint_names(text, expected):
    assert parse_keypoint_names(text) == expected
