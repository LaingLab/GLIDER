"""DeviceControlPanel must let a GenericI2C device be read.

Regression: GenericI2C was missing from the panel's readable-type allowlist, so
selecting it left the entire Input Reading group (incl. the Read button) greyed
out — the GUI literally could not read an I2C device.
"""

from glider.gui.panels.device_control_panel import DeviceControlPanel


class _Board:
    is_connected = True


class _Device:
    def __init__(self, device_type):
        self.id = "dev_1"
        self.name = "sensor"
        self.device_type = device_type
        self.board = _Board()
        self._initialized = True
        self.pins = {}


class _HW:
    def __init__(self, device):
        self.devices = {device.id: device}
        self._device = device

    def get_device(self, device_id):
        return self._device if device_id == self._device.id else None


def _select_only_device(panel):
    panel.refresh_devices()
    panel._device_combo.setCurrentIndex(1)  # index 0 is the "-- Select Device --" placeholder


def test_read_controls_enabled_for_generic_i2c(qtbot):
    panel = DeviceControlPanel(_HW(_Device("GenericI2C")), run_async_fn=lambda coro: None)
    qtbot.addWidget(panel)
    _select_only_device(panel)
    assert panel._input_group.isEnabled() is True


def test_read_controls_disabled_for_output_device(qtbot):
    # Guard against "enable for everything" — outputs are still not readable.
    panel = DeviceControlPanel(_HW(_Device("DigitalOutput")), run_async_fn=lambda coro: None)
    qtbot.addWidget(panel)
    _select_only_device(panel)
    assert panel._input_group.isEnabled() is False
