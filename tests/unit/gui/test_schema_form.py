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


def test_device_ref_unknown_saved_id_falls_back_to_none(qtbot):
    class _Dev:
        def __init__(self, dt, name):
            self.device_type, self.name = dt, name

    devices = {"a": _Dev("PWMOutput", "Motor")}
    schema = [{"key": "ramp", "label": "Ramp", "type": "device_ref", "device_filter": "PWMOutput"}]
    _w, layout = _form()
    out = {}
    # "gone" was saved but no longer exists -> graceful fallback to -- None --.
    build_schema_widgets(layout, schema, out, devices=devices, values={"ramp": "gone"})
    widget, ftype = out["ramp"]
    assert read_schema_widget(widget, ftype) is None


def test_enum_default_not_in_choices_falls_back_to_first(qtbot):
    schema = [
        {
            "key": "mode",
            "label": "Mode",
            "type": "enum",
            "choices": [["full", "Full"], ["half", "Half"]],
            "default": "quarter",
        }
    ]
    _w, layout = _form()
    out = {}
    build_schema_widgets(layout, schema, out)
    widget, ftype = out["mode"]
    assert read_schema_widget(widget, ftype) == "full"


# --- the ble_address field type -----------------------------------------------


def _ble_form(qtbot, run_async=None, default=""):
    """Render a one-field schema containing a BLE address."""
    from PyQt6.QtWidgets import QFormLayout, QWidget

    from glider.gui.widgets.schema_form import build_schema_widgets

    host = QWidget()
    qtbot.addWidget(host)
    layout = QFormLayout(host)
    out: dict = {}
    build_schema_widgets(
        layout,
        [{"key": "address", "label": "Address", "type": "ble_address", "default": default}],
        out,
        run_async=run_async,
    )
    return host, out


def test_a_ble_address_field_offers_scan_when_it_can_scan(qtbot):
    """A plugin BLE device gets the same Scan button the built-ins have,
    without the hardware panel special-casing it by name."""
    from PyQt6.QtWidgets import QPushButton

    host, out = _ble_form(qtbot, run_async=lambda coro: coro.close())

    assert "address" in out
    assert [b.text() for b in host.findChildren(QPushButton)] == ["Scan"]


def test_it_degrades_to_a_typeable_field_without_a_runner(qtbot):
    """Scanning is async. A form with no loop to hand still has to render --
    an address can always be typed."""
    from PyQt6.QtWidgets import QPushButton

    host, out = _ble_form(qtbot, run_async=None)

    assert host.findChildren(QPushButton) == []
    widget, ftype = out["address"]
    assert widget.isEditable()
    assert ftype == "ble_address"


def test_clicking_scan_runs_the_scan_coroutine(qtbot):
    from PyQt6.QtWidgets import QPushButton

    started = []

    def _runner(coro):
        started.append(coro)
        coro.close()  # don't actually touch the BLE stack in a test

    host, _out = _ble_form(qtbot, run_async=_runner)
    host.findChildren(QPushButton)[0].click()

    assert started, "the Scan button did not run anything"


def test_a_typed_address_reads_back(qtbot):
    from glider.gui.widgets.schema_form import read_schema_widget

    _host, out = _ble_form(qtbot)
    widget, ftype = out["address"]
    widget.setCurrentText("AA:BB:CC:DD:EE:FF")

    assert read_schema_widget(widget, ftype) == "AA:BB:CC:DD:EE:FF"


def test_a_scanned_entry_reads_back_its_address_not_its_label(qtbot):
    """A scan lists peripherals by advertised name; the address is item data."""
    from glider.gui.widgets.schema_form import read_schema_widget

    _host, out = _ble_form(qtbot)
    widget, ftype = out["address"]
    widget.addItem("Maimu-01", "11:22:33:44:55:66")
    widget.setCurrentIndex(widget.count() - 1)

    assert read_schema_widget(widget, ftype) == "11:22:33:44:55:66"


def test_a_pasted_scan_label_is_stripped(qtbot):
    from glider.gui.widgets.schema_form import read_schema_widget

    _host, out = _ble_form(qtbot)
    widget, ftype = out["address"]
    widget.setCurrentText("AA:BB:CC:DD:EE:FF (Maimu-01)")

    assert read_schema_widget(widget, ftype) == "AA:BB:CC:DD:EE:FF"


def test_a_saved_address_is_shown(qtbot):
    _host, out = _ble_form(qtbot, default="AA:BB:CC:DD:EE:FF")

    assert out["address"][0].currentText() == "AA:BB:CC:DD:EE:FF"


def test_an_empty_field_reads_back_empty(qtbot):
    from glider.gui.widgets.schema_form import read_schema_widget

    _host, out = _ble_form(qtbot)
    widget, ftype = out["address"]

    assert read_schema_widget(widget, ftype) == ""
