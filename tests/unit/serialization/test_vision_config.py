"""The ``vision`` domain of the .glider file (schema 1.1.0).

CV settings — backend, model path, keypoint names — used to be dropped on
every save: ``CVSettings.to_dict``/``from_dict`` existed but nothing in the
serializer called them, so the tracking model choice did not survive a
save/load and a .glider file was not portable in the way the docs claim.
"""

from __future__ import annotations

import json

import pytest

from glider.core.experiment_session import ExperimentSession
from glider.serialization.schema import (
    SCHEMA_VERSION,
    ExperimentSchema,
    MetadataSchema,
    SchemaValidationError,
    VisionConfigSchema,
)
from glider.serialization.serializer import ExperimentSerializer
from glider.vision.cv_processor import CVSettings, DetectionBackend


@pytest.fixture
def serializer() -> ExperimentSerializer:
    return ExperimentSerializer()


@pytest.fixture
def cv_settings() -> CVSettings:
    return CVSettings(
        backend=DetectionBackend.YOLO_BYTETRACK,
        model_path="/models/mouse_pose.pt",
        keypoint_names=["nose", "left_ear", "right_ear"],
        confidence_threshold=0.65,
    )


# --- VisionConfigSchema -----------------------------------------------------


def test_vision_schema_round_trip():
    schema = VisionConfigSchema(settings={"backend": "YOLO_V8", "model_path": "m.pt"})
    assert VisionConfigSchema.from_dict(schema.to_dict()).settings == schema.settings


def test_vision_schema_defaults_empty():
    assert VisionConfigSchema().settings == {}
    assert VisionConfigSchema.from_dict({}).settings == {}


def test_vision_schema_rejects_non_dict():
    with pytest.raises(SchemaValidationError):
        VisionConfigSchema.from_dict(["not", "a", "dict"])


def test_vision_schema_copies_payload():
    """The schema must not alias the caller's dict."""
    payload = {"backend": "YOLO_V8"}
    schema = VisionConfigSchema.from_dict(payload)
    payload["backend"] = "MOTION_ONLY"
    assert schema.settings["backend"] == "YOLO_V8"


# --- ExperimentSchema integration -------------------------------------------


def test_experiment_schema_includes_vision():
    schema = ExperimentSchema(metadata=MetadataSchema(name="t"))
    assert "vision" in schema.to_dict()


def test_experiment_schema_vision_round_trip():
    schema = ExperimentSchema(
        metadata=MetadataSchema(name="t"),
        vision=VisionConfigSchema(settings={"backend": "YOLO_V8"}),
    )
    restored = ExperimentSchema.from_dict(json.loads(schema.to_json()))
    assert restored.vision.settings == {"backend": "YOLO_V8"}


def test_pre_1_1_file_without_vision_block_still_loads():
    """Files written before the vision domain existed must load unchanged."""
    legacy = {
        "schema_version": "1.0.0",
        "metadata": {"name": "legacy"},
        "hardware": {},
        "flow": {},
        "dashboard": {},
    }
    schema = ExperimentSchema.from_dict(legacy)
    assert schema.vision.settings == {}
    # An empty block round-trips to plain CVSettings defaults.
    assert CVSettings.from_dict(schema.vision.settings).backend == CVSettings().backend


def test_schema_version_is_a_minor_bump():
    """Older installs gate on major only, so 1.x files stay loadable there."""
    major, minor, _patch = SCHEMA_VERSION.split(".")
    assert major == "1"
    assert int(minor) >= 1


def test_legacy_file_is_not_rejected_by_validation(serializer, tmp_path):
    path = tmp_path / "legacy.glider"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "metadata": {"name": "legacy"},
                "hardware": {},
                "flow": {},
                "dashboard": {},
            }
        )
    )
    schema = serializer.load(path)
    assert schema.metadata.name == "legacy"


# --- Serializer round trip --------------------------------------------------


def test_save_persists_vision_settings(serializer, tmp_path, cv_settings):
    path = tmp_path / "exp.glider"
    session = ExperimentSession()
    session.name = "vision_test"

    serializer.save(path, session, vision_settings=cv_settings.to_dict())

    written = json.loads(path.read_text())
    assert written["vision"]["backend"] == "YOLO_BYTETRACK"
    assert written["vision"]["model_path"] == "/models/mouse_pose.pt"
    assert written["vision"]["keypoint_names"] == ["nose", "left_ear", "right_ear"]


def test_saved_vision_settings_restore_to_equal_cvsettings(serializer, tmp_path, cv_settings):
    """The full round trip an operator actually cares about."""
    path = tmp_path / "exp.glider"
    session = ExperimentSession()
    session.name = "vision_test"

    serializer.save(path, session, vision_settings=cv_settings.to_dict())
    restored = CVSettings.from_dict(serializer.load(path).vision.settings)

    assert restored.backend == DetectionBackend.YOLO_BYTETRACK
    assert restored.model_path == "/models/mouse_pose.pt"
    assert restored.keypoint_names == ["nose", "left_ear", "right_ear"]
    assert restored.confidence_threshold == 0.65


def test_save_without_vision_settings_writes_empty_block(serializer, tmp_path):
    path = tmp_path / "exp.glider"
    session = ExperimentSession()
    session.name = "no_vision"

    serializer.save(path, session)

    assert json.loads(path.read_text())["vision"] == {}
