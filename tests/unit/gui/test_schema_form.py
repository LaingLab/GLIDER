# tests/unit/gui/test_schema_form.py
from PyQt6.QtWidgets import QComboBox, QFormLayout, QWidget

from glider.gui.widgets.schema_form import build_schema_widgets, read_schema_widget


def _form():
    w = QWidget()
    layout = QFormLayout(w)
    return w, layout


def test_builds_int_and_bool_and_reads_back(qtbot):
    schema = [
        {"key": "n", "label": "N", "type": "int", "default": 7, "min": 0, "max": 10},
        {"key": "flag", "label": "Flag", "type": "bool", "default": True},
    ]
    _w, layout = _form()
    out = {}
    build_schema_widgets(layout, schema, out, values={"n": 3})
    assert read_schema_widget(*out["n"]) == 3  # saved value wins over default
    assert read_schema_widget(*out["flag"]) is True


def test_enum_field_builds_combo_and_reads_value(qtbot):
    schema = [
        {
            "key": "mode",
            "label": "Mode",
            "type": "enum",
            "choices": [["full", "Full"], ["half", "Half"]],
            "default": "half",
        }
    ]
    _w, layout = _form()
    out = {}
    build_schema_widgets(layout, schema, out)
    widget, ftype = out["mode"]
    assert isinstance(widget, QComboBox)
    assert read_schema_widget(widget, ftype) == "half"


def test_device_ref_field_lists_filtered_devices(qtbot):
    class _Dev:
        def __init__(self, dt, name):
            self.device_type, self.name = dt, name

    devices = {"a": _Dev("PWMOutput", "Motor"), "b": _Dev("DigitalInput", "Btn")}
    schema = [{"key": "ramp", "label": "Ramp", "type": "device_ref", "device_filter": "PWMOutput"}]
    _w, layout = _form()
    out = {}
    build_schema_widgets(layout, schema, out, devices=devices, values={"ramp": "a"})
    widget, ftype = out["ramp"]
    assert read_schema_widget(widget, ftype) == "a"  # only the PWM device is selectable
