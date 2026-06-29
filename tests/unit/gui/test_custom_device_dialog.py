"""
Smoke tests for the Custom Device Builder dialog: it builds a correct
declarative definition from the form for both transports.
"""

from __future__ import annotations

from glider.gui.dialogs.custom_device_dialog import CustomDeviceDialog


def _set_row(dialog, row, name, op, register=None, primary=False):
    entry = dialog._action_rows[row]
    entry["name"].setText(name)
    entry["op"].setCurrentIndex(entry["op"].findData(op))
    if register is not None:
        entry["reg"].setValue(register)
    entry["primary"].setChecked(primary)


def test_builds_i2c_definition(qtbot):
    dlg = CustomDeviceDialog()
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("MyTempSensor")
    _set_row(dlg, 0, "temp", "read_word", register=0x0E, primary=True)

    defn = dlg._build_definition()
    assert defn["name"] == "MyTempSensor"
    assert defn["transport"] == "i2c"
    assert {f["key"] for f in defn["settings"]} == {"i2c_bus", "i2c_address"}
    assert defn["actions"][0] == {
        "name": "temp",
        "op": "read_word",
        "params": {"register": 0x0E},
        "primary": True,
    }


def test_write_op_gets_runtime_value(qtbot):
    dlg = CustomDeviceDialog()
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("Cfg")
    _set_row(dlg, 0, "set_cfg", "write_byte", register=1)
    action = dlg._build_definition()["actions"][0]
    assert action["runtime_args"] == ["value"]


def test_transport_switch_to_gpio(qtbot):
    dlg = CustomDeviceDialog()
    qtbot.addWidget(dlg)
    dlg._name_edit.setText("Relay")
    # Switch to GPIO -> rows reset, ops become GPIO ops.
    dlg._transport_combo.setCurrentIndex(dlg._transport_combo.findData("gpio"))
    _set_row(dlg, 0, "on", "set_high", primary=True)

    defn = dlg._build_definition()
    assert defn["transport"] == "gpio"
    assert {f["key"] for f in defn["settings"]} == {"pin"}
    # GPIO actions carry no register params.
    assert defn["actions"][0] == {"name": "on", "op": "set_high", "primary": True}
