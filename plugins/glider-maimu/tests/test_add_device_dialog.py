"""Adding a Maimu through GLIDER's Add Device dialog, as a plugin.

The device used to be a built-in with its own hand-written branch in the
hardware panel. As a plugin it goes through the generic path instead: the
dialog lists anything in DEVICE_REGISTRY it does not recognise and renders that
class's SETTINGS_SCHEMA. This is the test that the generic path is actually
good enough to replace the special case -- in particular that the address
field still offers **Scan**, which before the ``ble_address`` schema type only
the built-ins could have.
"""

from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QComboBox, QDialog, QLineEdit, QMessageBox, QPushButton

from glider.gui.panels.hardware_panel import HardwarePanel
from glider_maimu.device import DEFAULT_SERVICE_UUID, DEFAULT_WRITE_CHAR_UUID

pytestmark = pytest.mark.usefixtures("qtbot")

# The dialog labels plugin-supplied types this way; see _on_add_device.
MAIMU_ITEM = "Maimu (plugin)"


@pytest.fixture
def hardware_manager():
    manager = MagicMock()
    manager.boards = {"ble_board": MagicMock()}
    manager.devices = {}
    return manager


def _find(dialog, cls, predicate):
    return next(w for w in dialog.findChildren(cls) if predicate(w))


def _add_maimu(qtbot, monkeypatch, hardware_manager, *, fill=lambda d: None):
    """Open Add Device, choose the Maimu, let ``fill`` complete the form, accept."""
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    captured = {}

    def fake_exec(dialog):
        captured["dialog"] = dialog
        type_combo = _find(dialog, QComboBox, lambda c: c.findText(MAIMU_ITEM) >= 0)
        type_combo.setCurrentText(MAIMU_ITEM)  # rebuilds the settings form
        _find(dialog, QLineEdit, lambda e: e.placeholderText() == "e.g., led_1").setText("stim_1")
        fill(dialog)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", fake_exec)

    panel = HardwarePanel(
        hardware_manager,
        session_fn=lambda: None,
        run_async_fn=lambda coro: coro.close(),
        show_add_buttons=True,
    )
    qtbot.addWidget(panel)
    panel._on_add_device()
    return hardware_manager.add_device_multi_pin.call_args, captured.get("dialog")


def _address_combo(dialog):
    return _find(dialog, QComboBox, lambda c: c.isEditable())


def test_the_plugin_device_is_offered(qtbot, monkeypatch, hardware_manager):
    """Registered by the plugin, listed by the dialog, with no core change."""
    call, _dialog = _add_maimu(qtbot, monkeypatch, hardware_manager)

    assert call is not None, "picking the Maimu did not reach add_device_multi_pin"
    device_id, device_type, _board_id, pins = call.args
    assert (device_id, device_type, pins) == ("stim_1", "Maimu", {})


def test_the_address_field_still_offers_scan(qtbot, monkeypatch, hardware_manager):
    """The whole point of the ble_address schema type. Without it a plugin BLE
    device gets a bare text box and the researcher has to paste a MAC."""
    _call, dialog = _add_maimu(qtbot, monkeypatch, hardware_manager)

    assert any(b.text() == "Scan" for b in dialog.findChildren(QPushButton))


def test_the_uuids_arrive_prefilled(qtbot, monkeypatch, hardware_manager):
    call, _dialog = _add_maimu(
        qtbot,
        monkeypatch,
        hardware_manager,
        fill=lambda d: _address_combo(d).setCurrentText("AA:BB:CC:DD:EE:FF"),
    )

    settings = call.kwargs["settings"]
    assert settings["address"] == "AA:BB:CC:DD:EE:FF"
    assert settings["write_char_uuid"] == DEFAULT_WRITE_CHAR_UUID
    assert settings["service_uuid"] == DEFAULT_SERVICE_UUID


def test_a_scanned_entry_saves_the_address_not_its_label(qtbot, monkeypatch, hardware_manager):
    """A scan lists peripherals by advertised name; the address is item data."""

    def fill(dialog):
        combo = _address_combo(dialog)
        combo.addItem("Maimu-01", "11:22:33:44:55:66")
        combo.setCurrentIndex(combo.count() - 1)

    call, _dialog = _add_maimu(qtbot, monkeypatch, hardware_manager, fill=fill)

    assert call.kwargs["settings"]["address"] == "11:22:33:44:55:66"


def test_the_advertised_name_is_saved_separately(qtbot, monkeypatch, hardware_manager):
    """The device's display name and its advertised BLE name are different
    things; the advertised one travels in settings."""

    def fill(dialog):
        _find(
            dialog,
            QLineEdit,
            lambda e: "Resolve the address" in (e.toolTip() or ""),
        ).setText("Maimu-01")

    call, _dialog = _add_maimu(qtbot, monkeypatch, hardware_manager, fill=fill)

    assert call.kwargs["name"] == "stim_1"
    assert call.kwargs["settings"]["name"] == "Maimu-01"
