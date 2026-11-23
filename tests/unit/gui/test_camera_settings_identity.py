"""The camera settings dialog's Identity section: mode, animals, tuning."""

from __future__ import annotations

import pytest

from glider.gui.dialogs.camera_settings_dialog import CameraSettingsDialog
from glider.vision.cv_processor import CVSettings, DetectionBackend


@pytest.fixture(autouse=True)
def _ultralytics_present(monkeypatch):
    import glider.vision.yolo_install as yolo_install

    monkeypatch.setattr(yolo_install, "ensure_ultralytics_installed", lambda parent: True)


def _dialog(qtbot, class_names=(), **kw):
    kw.setdefault("backend", DetectionBackend.YOLO_V8)
    d = CameraSettingsDialog(cv_settings=CVSettings(**kw), class_names=list(class_names))
    qtbot.addWidget(d)
    return d


def _set_mode(d, mode):
    d._identity_mode_combo.setCurrentIndex(d._identity_mode_combo.findData(mode))


def test_modes_are_offered_in_plain_words(qtbot):
    d = _dialog(qtbot)
    combo = d._identity_mode_combo
    assert [combo.itemText(i) for i in range(combo.count())] == [
        "Off",
        "By appearance",
        "Name at start",
    ]
    assert combo.currentData() == "off"
    assert d._animals_table.isHidden()


def test_round_trips_into_cv_settings(qtbot):
    d = _dialog(
        qtbot,
        class_names=["dam", "pup"],
        identity_mode="appearance",
        animals=[{"name": "Dam", "class_name": "dam"}, {"name": "Pup 1", "class_name": "pup"}],
        identity_relabel_s=1.5,
        identity_separation_px=70.0,
        identity_max_travel_px=30.0,
    )
    assert d._animals_table.rowCount() == 2
    assert d._animals_table.item(1, 0).text() == "Pup 1"
    assert d._animals_table.cellWidget(1, 1).currentText() == "pup"

    d._animals_table.item(0, 0).setText("Mother")
    d._save_settings()
    s = d.get_cv_settings()
    assert s.identity_mode == "appearance"
    assert s.animals == [
        {"name": "Mother", "class_name": "dam"},
        {"name": "Pup 1", "class_name": "pup"},
    ]
    assert (s.identity_relabel_s, s.identity_separation_px, s.identity_max_travel_px) == (
        1.5,
        70.0,
        30.0,
    )


def test_class_choices_come_from_the_loaded_model(qtbot):
    d = _dialog(qtbot, class_names=["dam", "pup"], identity_mode="appearance")
    d._add_animal_btn.click()
    combo = d._animals_table.cellWidget(0, 1)
    assert [combo.itemText(i) for i in range(combo.count())] == ["dam", "pup"]


def test_blank_rows_are_dropped(qtbot):
    d = _dialog(qtbot, identity_mode="continuity")
    d._add_animal_btn.click()
    d._add_animal_btn.click()
    d._animals_table.item(0, 0).setText("Mouse A")
    d._save_settings()
    assert [a["name"] for a in d.get_cv_settings().animals] == ["Mouse A"]


def test_name_at_start_hides_class_and_relabel(qtbot):
    d = _dialog(qtbot, identity_mode="appearance")
    assert not d._animals_table.isColumnHidden(1)
    assert not d._identity_relabel_spin.isHidden()
    _set_mode(d, "continuity")
    assert d._animals_table.isColumnHidden(1)
    assert d._identity_relabel_spin.isHidden()
    assert not d._animals_table.isHidden()


def test_mode_disabled_with_the_reason_off_yolo(qtbot):
    d = _dialog(qtbot, backend=DetectionBackend.POSE_MODEL)
    assert not d._identity_mode_combo.isEnabled()
    assert "YOLO" in d._identity_mode_combo.toolTip()
    d._backend_combo.setCurrentIndex(d._backend_combo.findData(DetectionBackend.YOLO_V8))
    assert d._identity_mode_combo.isEnabled()
    assert d._identity_mode_combo.toolTip() == ""
