"""The settings dialog must not be handed live manager state.

``CameraSettingsDialog`` edits its settings objects **in place**
(``_save_settings`` assigns straight onto ``self._cv_settings`` /
``self._camera_settings``) and ``get_*_settings()`` returns those same objects.
``main_window`` used to pass the live objects from the manager, so the dialog's
edit landed on the manager's own state *before* ``update_settings`` /
``apply_settings`` ran. Their change-detection then read the already-mutated
value and compared it to itself:

    old_backend = self._settings.backend   # already the NEW value
    self._settings = settings              # same object; no-op
    if old_backend != settings.backend:    # always False -> never reloads

So choosing a new CV backend, a new model, or a new camera index in the dialog
silently never took effect, while the UI and the saved .glider file both
reported the new choice.

These tests model the ALIASING path deliberately. An earlier test claimed to
cover this but rebuilt the settings via ``from_dict(to_dict())`` — a copy —
so it passed while the real path was broken. Anything here that constructs a
fresh CVSettings instead of reusing the object handed out is not testing the
bug.
"""

from __future__ import annotations

import pytest

from glider.vision.camera_manager import CameraManager, CameraSettings
from glider.vision.cv_processor import CVProcessor, CVSettings, DetectionBackend


class _FakeDialog:
    """Stands in for CameraSettingsDialog's aliasing contract.

    Deliberately mirrors the real class's two load-bearing behaviours rather
    than importing it: it keeps the exact object it was handed
    (camera_settings_dialog.py:66), mutates that object's fields in place
    (~:1240), and hands the same object back (:1270). Instantiating the real
    dialog needs a QApplication and builds ~20 widget groups; the contract
    under test is just these three lines.
    """

    def __init__(self, cv_settings: CVSettings):
        self._cv_settings = cv_settings

    def edit_backend(self, backend: DetectionBackend) -> None:
        self._cv_settings.backend = backend

    def edit_model_path(self, path: str) -> None:
        self._cv_settings.model_path = path

    def get_cv_settings(self) -> CVSettings:
        return self._cv_settings


def test_copy_isolates_the_processor_from_dialog_edits():
    """The fix: main_window hands out .copy(), so edits can't reach _settings."""
    proc = CVProcessor(CVSettings(backend=DetectionBackend.BACKGROUND_SUBTRACTION))
    proc.initialize()

    dialog = _FakeDialog(proc.settings.copy())  # what main_window now does
    dialog.edit_backend(DetectionBackend.MOTION_ONLY)

    # The edit must not have reached the processor yet.
    assert proc.settings.backend == DetectionBackend.BACKGROUND_SUBTRACTION

    proc.update_settings(dialog.get_cv_settings())
    assert proc.settings.backend == DetectionBackend.MOTION_ONLY
    assert proc.active_backend == DetectionBackend.MOTION_ONLY


def test_backend_change_through_the_dialog_actually_reloads():
    """The regression test for the bug itself.

    Without the copy, update_settings' change detection compares the mutated
    value to itself, no reload fires, and active_backend never moves.
    """
    proc = CVProcessor(CVSettings(backend=DetectionBackend.BACKGROUND_SUBTRACTION))
    proc.initialize()
    assert proc.active_backend == DetectionBackend.BACKGROUND_SUBTRACTION

    dialog = _FakeDialog(proc.settings.copy())
    dialog.edit_backend(DetectionBackend.MOTION_ONLY)
    proc.update_settings(dialog.get_cv_settings())

    assert proc.active_backend == DetectionBackend.MOTION_ONLY


def test_model_path_change_through_the_dialog_actually_reloads(tmp_path, monkeypatch):
    """PR #41's reload-on-model-path fix, exercised through the real path.

    It was previously only covered with independently-constructed settings,
    which is not how the dialog behaves.
    """
    proc = CVProcessor(
        CVSettings(backend=DetectionBackend.YOLO_V8, model_path=str(tmp_path / "a.pt"))
    )
    calls: list[bool] = []
    monkeypatch.setattr(proc, "initialize", lambda: calls.append(True))

    dialog = _FakeDialog(proc.settings.copy())
    dialog.edit_model_path(str(tmp_path / "b.pt"))
    proc.update_settings(dialog.get_cv_settings())

    assert calls, "a model-path change made in the dialog must reload the model"


def test_copy_does_not_share_the_keypoint_names_list():
    """replace() alone would share the list; .copy() must not."""
    settings = CVSettings(keypoint_names=["nose", "tail"])
    duplicate = settings.copy()
    duplicate.keypoint_names.append("left_ear")

    assert settings.keypoint_names == ["nose", "tail"]
    assert duplicate.keypoint_names == ["nose", "tail", "left_ear"]


def test_copy_preserves_every_field():
    settings = CVSettings(
        backend=DetectionBackend.YOLO_BYTETRACK,
        model_path="/m.pt",
        keypoint_names=["nose"],
        confidence_threshold=0.75,
        overlay_color=(1, 2, 3),
    )
    assert settings.copy().to_dict() == settings.to_dict()


def test_camera_settings_copy_isolates_the_manager():
    """CameraManager.apply_settings has the identical change-detection shape.

    Aliased, ``old_index != settings.camera_index`` is always False and the
    camera never reconnects to the newly-chosen index.
    """
    from dataclasses import replace

    manager = CameraManager()
    manager.apply_settings(CameraSettings(camera_index=0))

    handed_to_dialog = replace(manager.settings)  # what main_window now does
    handed_to_dialog.camera_index = 2

    # The manager must not see the edit until apply_settings runs.
    assert manager.settings.camera_index == 0

    manager.apply_settings(handed_to_dialog)
    assert manager.settings.camera_index == 2


@pytest.mark.parametrize(
    "field, value",
    [
        ("enabled", False),
        ("draw_overlays", False),
        ("vision_cone_enabled", True),
    ],
)
def test_camera_panel_quick_toggles_still_mutate_live_settings(field, value):
    """Guard-rail for the fix we deliberately did NOT make.

    camera_panel's quick toggles (`:874`, `:881`, `:885`) mutate
    ``cv_processor.settings`` in place and never call update_settings —
    CVProcessor reads these fields straight off _settings each frame. Copying
    inside the `settings` getter would have made those checkboxes silently
    do nothing. The copy belongs at the dialog call site, not here.
    """
    proc = CVProcessor(CVSettings())
    setattr(proc.settings, field, value)
    assert getattr(proc.settings, field) == value
