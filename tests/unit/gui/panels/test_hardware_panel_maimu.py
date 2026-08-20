"""Adding a Maimu through the Add Device dialog.

Drives the real dialog by standing in for QDialog.exec: fill the widgets, accept,
then read what the panel handed to the hardware manager. The point is that a
researcher who picks "Maimu" and hits Scan gets a device with the stimulator's
UUIDs already in place -- nothing to paste.
"""

import pytest
from PyQt6.QtWidgets import QComboBox, QDialog, QLineEdit, QMessageBox

from glider.gui.panels.hardware_panel import HardwarePanel
from glider.hal.devices.maimu import DEFAULT_SERVICE_UUID, DEFAULT_WRITE_CHAR_UUID

pytestmark = pytest.mark.usefixtures("qtbot")

MAIMU_ITEM = "Maimu (BLE stimulator)"


def _panel(qtbot, mock_hardware_manager):
    panel = HardwarePanel(
        mock_hardware_manager,
        session_fn=lambda: None,
        # The dialog fires an auto-initialize coroutine; close it rather than
        # dropping it, so the test doesn't trail a never-awaited warning.
        run_async_fn=lambda coro: coro.close(),
        show_add_buttons=True,
    )
    qtbot.addWidget(panel)
    return panel


def _find(dialog, cls, predicate):
    return next(w for w in dialog.findChildren(cls) if predicate(w))


def _add_maimu(qtbot, monkeypatch, mock_hardware_manager, *, fill):
    """Open Add Device, choose Maimu, let ``fill`` complete the form, accept."""
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    def fake_exec(dialog):
        type_combo = _find(dialog, QComboBox, lambda c: c.findText(MAIMU_ITEM) >= 0)
        type_combo.setCurrentText(MAIMU_ITEM)  # rebuilds the settings form
        _find(dialog, QLineEdit, lambda e: e.placeholderText() == "e.g., led_1").setText("stim_1")
        fill(dialog)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)

    _panel(qtbot, mock_hardware_manager)._on_add_device()
    return mock_hardware_manager.add_device_multi_pin.call_args


def test_maimu_is_offered_in_the_dialog(qtbot, monkeypatch, mock_hardware_manager):
    call = _add_maimu(
        qtbot,
        monkeypatch,
        mock_hardware_manager,
        fill=lambda d: _find(d, QComboBox, lambda c: c.isEditable()).setCurrentText(
            "AA:BB:CC:DD:EE:FF"
        ),
    )
    assert call is not None, "picking Maimu did not reach add_device_multi_pin"
    device_id, device_type, _board_id, pins = call.args
    assert (device_id, device_type, pins) == ("stim_1", "Maimu", {})


def test_uuids_are_prefilled_so_nothing_must_be_pasted(qtbot, monkeypatch, mock_hardware_manager):
    call = _add_maimu(
        qtbot,
        monkeypatch,
        mock_hardware_manager,
        fill=lambda d: _find(d, QComboBox, lambda c: c.isEditable()).setCurrentText(
            "AA:BB:CC:DD:EE:FF"
        ),
    )
    assert call.kwargs["settings"]["address"] == "AA:BB:CC:DD:EE:FF"
    assert call.kwargs["settings"]["write_char_uuid"] == DEFAULT_WRITE_CHAR_UUID
    assert call.kwargs["settings"]["service_uuid"] == DEFAULT_SERVICE_UUID


def test_scanned_entry_saves_the_address_not_its_label(qtbot, monkeypatch, mock_hardware_manager):
    """A scan lists peripherals by advertised name; the address is the item data."""

    def fill(dialog):
        addr = _find(dialog, QComboBox, lambda c: c.isEditable())
        addr.addItem("Maimu-01", "11:22:33:44:55:66")
        addr.setCurrentIndex(addr.count() - 1)

    call = _add_maimu(qtbot, monkeypatch, mock_hardware_manager, fill=fill)
    assert call.kwargs["settings"]["address"] == "11:22:33:44:55:66"


def test_advertised_name_is_saved_for_a_portable_file(qtbot, monkeypatch, mock_hardware_manager):
    def fill(dialog):
        _find(dialog, QLineEdit, lambda e: "advertised name" in e.placeholderText()).setText(
            "Maimu-01"
        )

    call = _add_maimu(qtbot, monkeypatch, mock_hardware_manager, fill=fill)
    # The device's display name and its advertised BLE name are different
    # things; the advertised one travels in settings.
    assert call.kwargs["name"] == "stim_1"
    assert call.kwargs["settings"]["name"] == "Maimu-01"
    assert call.kwargs["settings"]["address"] == ""
