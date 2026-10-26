"""The camera settings dialog loads, saves and gates the single-animal option."""

from __future__ import annotations

from glider.gui.dialogs.camera_settings_dialog import CameraSettingsDialog
from glider.vision.cv_processor import CVSettings, DetectionBackend


def _select_backend(dialog: CameraSettingsDialog, backend: DetectionBackend) -> None:
    combo = dialog._backend_combo
    combo.setCurrentIndex(combo.findData(backend))


def test_checkbox_loads_and_saves_the_setting(qtbot):
    dialog = CameraSettingsDialog(cv_settings=CVSettings(single_animal=True))
    qtbot.addWidget(dialog)
    assert dialog._single_animal_cb.isChecked()

    dialog._single_animal_cb.setChecked(False)
    dialog._save_settings()
    assert dialog.get_cv_settings().single_animal is False


def test_checkbox_defaults_off(qtbot):
    dialog = CameraSettingsDialog(cv_settings=CVSettings())
    qtbot.addWidget(dialog)
    assert not dialog._single_animal_cb.isChecked()


def test_checkbox_hidden_where_it_would_do_nothing(qtbot):
    dialog = CameraSettingsDialog(cv_settings=CVSettings())
    qtbot.addWidget(dialog)
    assert not dialog._single_animal_cb.isHidden()  # Background Sub
    _select_backend(dialog, DetectionBackend.MOTION_ONLY)
    assert dialog._single_animal_cb.isHidden()
    _select_backend(dialog, DetectionBackend.POSE_MODEL)
    assert dialog._single_animal_cb.isHidden()


def test_checkbox_follows_the_cv_enabled_toggle(qtbot):
    dialog = CameraSettingsDialog(cv_settings=CVSettings())
    qtbot.addWidget(dialog)
    dialog._cv_enabled_cb.setChecked(False)
    assert not dialog._single_animal_cb.isEnabled()
    dialog._cv_enabled_cb.setChecked(True)
    assert dialog._single_animal_cb.isEnabled()
