"""Render a SETTINGS_SCHEMA field list into a QFormLayout and read it back.

Shared by the Add-Device form (HardwarePanel) and the WaitForInput behavior
settings (NodeEditorController). Supported types: int, hex, float, bool, str,
enum, device_ref.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox


def build_schema_widgets(
    layout,
    schema: list[dict],
    out: dict,
    *,
    values: dict | None = None,
    devices: dict | None = None,
) -> None:
    """Render a SETTINGS_SCHEMA field list into ``layout`` and record widgets.

    Each ``field`` is a dict with at minimum a ``key`` (skipped if missing) and
    an optional ``type`` (default ``"str"``) and ``label`` (default = ``key``).
    Per-field-type key contract:

    - ``int`` / ``hex`` -> ``QSpinBox``; honors ``min``/``max`` (defaults
      ``0`` / ``2_000_000_000``). ``hex`` displays base-16 with a ``0x`` prefix.
    - ``float`` -> ``QDoubleSpinBox``; honors ``decimals`` (default ``2``) and
      ``min``/``max`` (defaults ``0.0`` / ``1e9``).
    - ``bool`` -> ``QCheckBox``.
    - ``str`` (fallback) -> ``QLineEdit``.
    - ``enum`` -> ``QComboBox``; ``choices`` is a list of ``[value, label]``
      pairs. The selection falls back to the first choice if the resolved
      default is not among the choice values.
    - ``device_ref`` -> ``QComboBox`` populated from ``devices`` plus a leading
      ``-- None --`` entry. ``device_filter``, if set, is matched against each
      device's ``device_type`` attribute.

    Any field may carry a ``help`` string, applied as the widget's tooltip.

    Params:
        layout: A ``QFormLayout`` (or compatible) that receives one row per
            field via ``layout.addRow("<label>:", widget)``.
        schema: The field list described above.
        out: Populated in place with ``out[key] = (widget, ftype)`` for later
            read-back via :func:`read_schema_widget`.
        values: Optional saved values; ``values[key]`` overrides a field's
            ``default`` when present.
        devices: Optional mapping of ``dev_id -> device`` used to populate
            ``device_ref`` combos (device objects expose ``name`` and
            ``device_type`` attributes).
    """
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
            # An unknown/removed saved id isn't found -> fall back to "-- None --" (None).
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
    """Read the current value from a widget produced by :func:`build_schema_widgets`.

    ``widget`` and ``ftype`` are the pair stored in ``out[key]``. Returns:

    - ``int`` / ``hex`` / ``float`` -> the numeric ``widget.value()``.
    - ``bool`` -> ``widget.isChecked()``.
    - ``enum`` / ``device_ref`` -> ``widget.currentData()`` (the field's stored
      value / device id, or ``None`` for the ``-- None --`` device_ref entry).
    - anything else (``str``) -> the stripped ``widget.text()``.
    """
    if ftype in ("int", "hex", "float"):
        return widget.value()
    if ftype == "bool":
        return widget.isChecked()
    if ftype in ("enum", "device_ref"):
        return widget.currentData()
    return widget.text().strip()
