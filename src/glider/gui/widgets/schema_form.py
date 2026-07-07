"""Render a SETTINGS_SCHEMA field list into a QFormLayout and read it back.

Shared by the Add-Device form (HardwarePanel) and the WaitForInput behavior
settings (NodeEditorController). Supported types: int, hex, float, bool, str,
enum, device_ref.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox


def build_schema_widgets(layout, schema, out: dict, *, values=None, devices=None) -> None:
    values = values or {}
    devices = devices or {}
    for field in schema:
        key = field.get("key")
        if not key:
            continue
        ftype = field.get("type", "str")
        default = values.get(key, field.get("default"))
        if ftype in ("int", "hex"):
            widget = QSpinBox()
            widget.setRange(int(field.get("min", 0)), int(field.get("max", 2_000_000_000)))
            if ftype == "hex":
                widget.setDisplayIntegerBase(16)
                widget.setPrefix("0x")
            widget.setValue(int(default or 0))
        elif ftype == "float":
            widget = QDoubleSpinBox()
            widget.setDecimals(int(field.get("decimals", 2)))
            widget.setRange(float(field.get("min", 0.0)), float(field.get("max", 1e9)))
            widget.setValue(float(default if default is not None else 0.0))
        elif ftype == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(default))
        elif ftype == "enum":
            widget = QComboBox()
            for value, label in field.get("choices", []):
                widget.addItem(str(label), value)
            idx = widget.findData(default)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
        elif ftype == "device_ref":
            widget = QComboBox()
            widget.addItem("-- None --", None)
            wanted = field.get("device_filter")
            for dev_id, dev in devices.items():
                if wanted and getattr(dev, "device_type", None) != wanted:
                    continue
                widget.addItem(
                    f"{getattr(dev, 'name', dev_id)} ({getattr(dev, 'device_type', '')})", dev_id
                )
            idx = widget.findData(default)
            widget.setCurrentIndex(idx if idx >= 0 else 0)
        else:
            widget = QLineEdit()
            if default is not None:
                widget.setText(str(default))
        if field.get("help"):
            widget.setToolTip(str(field["help"]))
        out[key] = (widget, ftype)
        layout.addRow(f"{field.get('label', key)}:", widget)


def read_schema_widget(widget, ftype: str):
    if ftype in ("int", "hex", "float"):
        return widget.value()
    if ftype == "bool":
        return widget.isChecked()
    if ftype in ("enum", "device_ref"):
        return widget.currentData()
    return widget.text().strip()
