"""GliderCore contract: CV settings survive a .glider save/load round trip.

The serializer carries the vision block opaquely (it must not import cv2), so
GliderCore owns the CVSettings <-> dict conversion on both ends. These tests
cover that wiring end-to-end — the part an operator actually experiences as
"my tracking model is still selected after reopening the experiment".
"""

from __future__ import annotations

import pytest

from glider.core.experiment_session import ExperimentSession
from glider.core.glider_core import GliderCore
from glider.vision.cv_processor import CVSettings, DetectionBackend


@pytest.fixture
def core() -> GliderCore:
    c = GliderCore()
    c._session = ExperimentSession()
    c._session.name = "cv_persistence"
    return c


@pytest.mark.asyncio
async def test_cv_settings_survive_save_load(core, tmp_path):
    path = tmp_path / "exp.glider"
    core.cv_processor.update_settings(
        CVSettings(
            backend=DetectionBackend.YOLO_BYTETRACK,
            model_path="/models/mouse_pose.pt",
            keypoint_names=["nose", "left_ear", "right_ear"],
            confidence_threshold=0.8,
        )
    )

    await core.save_experiment(path)

    # A fresh core, as if the app had been restarted.
    reopened = GliderCore()
    assert reopened.cv_processor.settings.model_path is None  # default
    await reopened.load_experiment(path)

    restored = reopened.cv_processor.configured_settings
    assert restored.backend == DetectionBackend.YOLO_BYTETRACK
    assert restored.model_path == "/models/mouse_pose.pt"
    assert restored.keypoint_names == ["nose", "left_ear", "right_ear"]
    assert restored.confidence_threshold == 0.8


@pytest.mark.asyncio
async def test_missing_weights_do_not_destroy_the_operators_backend_choice(core, tmp_path):
    """The scenario this whole distinction exists for.

    Machine A configures YOLO. Machine B opens the file without the weights
    (or without ultralytics — a Pi opening a desktop-authored experiment), so
    the processor degrades to background subtraction at runtime. If machine B
    then saves, the operator's YOLO choice must still be in the file: a
    runtime degradation is not a configuration change.
    """
    authored = tmp_path / "authored.glider"
    core.cv_processor.update_settings(
        CVSettings(
            backend=DetectionBackend.YOLO_BYTETRACK,
            model_path=str(tmp_path / "absent.pt"),
        )
    )
    await core.save_experiment(authored)

    # Machine B: weights absent, so loading degrades the *running* backend.
    machine_b = GliderCore()
    machine_b._session = ExperimentSession()
    await machine_b.load_experiment(authored)
    assert machine_b.cv_processor.settings.backend == DetectionBackend.BACKGROUND_SUBTRACTION
    assert machine_b.cv_processor.configured_settings.backend == DetectionBackend.YOLO_BYTETRACK

    # ...and re-saving there must not write the degradation back.
    resaved = tmp_path / "resaved.glider"
    await machine_b.save_experiment(resaved)

    import json

    assert json.loads(resaved.read_text())["vision"]["backend"] == "YOLO_BYTETRACK"


@pytest.mark.asyncio
async def test_loading_legacy_file_keeps_current_cv_settings(core, tmp_path):
    """A pre-1.1.0 file has no vision block; don't stomp live settings."""
    path = tmp_path / "legacy.glider"
    await core.save_experiment(path)

    # Strip the vision block, simulating a file written before schema 1.1.0.
    import json

    data = json.loads(path.read_text())
    del data["vision"]
    data["schema_version"] = "1.0.0"
    path.write_text(json.dumps(data))

    reopened = GliderCore()
    reopened.cv_processor.update_settings(
        CVSettings(backend=DetectionBackend.MOTION_ONLY, model_path="/keep/me.pt")
    )
    await reopened.load_experiment(path)

    assert reopened.cv_processor.settings.backend == DetectionBackend.MOTION_ONLY
    assert reopened.cv_processor.settings.model_path == "/keep/me.pt"


@pytest.mark.asyncio
async def test_corrupt_vision_block_does_not_break_load(core, tmp_path):
    """A hand-edited/garbage vision block must not take the whole file down."""
    path = tmp_path / "corrupt.glider"
    await core.save_experiment(path)

    import json

    data = json.loads(path.read_text())
    data["vision"] = {"backend": "NOT_A_REAL_BACKEND"}
    path.write_text(json.dumps(data))

    reopened = GliderCore()
    await reopened.load_experiment(path)  # must not raise

    assert reopened.session is not None
    assert reopened.cv_processor.settings.backend == CVSettings().backend
