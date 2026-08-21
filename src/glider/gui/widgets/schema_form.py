"""Render a SETTINGS_SCHEMA field list into a QFormLayout and read it back.

Shared by the Add-Device form (HardwarePanel) and the WaitForInput behavior
settings (NodeEditorController). Supported types: int, hex, float, bool, str,
enum, device_ref.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QWidget,
)


def build_schema_widgets(
    layout,
    schema: list[dict],
    out: dict,
    *,
    values: dict | None = None,
    devices: dict | None = None,
    run_async=None,
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
        elif ftype == "ble_address":
            # The widget added to the layout is a container (combo + Scan), but
            # the value lives on the combo, so that is what gets stored.
            container, widget = build_ble_address_widget(run_async, layout.parentWidget())
            if default:
                widget.setCurrentText(str(default))
            if field.get("help"):
                container.setToolTip(str(field["help"]))
            out[key] = (widget, ftype)
            layout.addRow(f"{field.get('label', key)}:", container)
            continue
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


def build_ble_address_widget(run_async=None, parent=None):
    """An editable BLE address combo, with a Scan button when it can scan.

    Returns ``(container, combo)``: add the container to a layout, read the
    value back from the combo with ``read_schema_widget(combo, "ble_address")``.

    Scanning is asynchronous, so the caller supplies ``run_async``. Without one
    -- a form with no event loop to hand, such as a headless test -- the combo
    is still returned and still editable; only the button is omitted. That
    degradation is deliberate: an address can always be typed.
    """
    from PyQt6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton

    combo = QComboBox()
    combo.setEditable(True)
    combo.setMinimumWidth(240)
    combo.lineEdit().setPlaceholderText("BLE address (or Scan)")

    container = QWidget(parent)
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(combo)

    if run_async is None:
        return container, combo

    scan_btn = QPushButton("Scan")
    scan_btn.setToolTip("Discover nearby BLE peripherals (~5s)")

    def do_scan(_=False):
        scan_btn.setEnabled(False)
        scan_btn.setText("Scanning\u2026")

        async def _scan():
            try:
                # Scanning discovers peripherals via the host BLE adapter -- it
                # does not depend on which board is selected, so scan directly
                # via the BLE board's (static) scanner.
                from glider.hal.boards.ble_board import BLEBoard

                results = await BLEBoard.scan(timeout=8.0)
                combo.clear()
                if not results:
                    combo.addItem("(no devices found)", None)
                for peripheral in results:
                    # An unnamed peripheral shows as its address and signal
                    # strength, with its advertised services in the tooltip --
                    # which is how you tell which bare MAC is the stimulator
                    # when its name did not survive the scan response.
                    combo.addItem(peripheral.label, peripheral.address)
                    detail = peripheral.address
                    if peripheral.service_uuids:
                        detail += chr(10) + "services: " + ", ".join(peripheral.service_uuids)
                    combo.setItemData(combo.count() - 1, detail, Qt.ItemDataRole.ToolTipRole)
            except ImportError:
                QMessageBox.critical(parent, "Scan failed", "bleak is not installed.")
            except Exception as e:  # noqa: BLE001 - surfaced to the user
                QMessageBox.critical(parent, "Scan failed", str(e))
            finally:
                scan_btn.setEnabled(True)
                scan_btn.setText("Scan")

        run_async(_scan())

    scan_btn.clicked.connect(do_scan)
    row.addWidget(scan_btn)
    return container, combo


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
    if ftype == "ble_address":
        # An item picked from a scan carries the address as its data; anything
        # typed by hand is the text, minus a trailing " (name)" if a scan label
        # was pasted in.
        address = widget.currentData()
        if not address:
            raw = widget.currentText().strip()
            address = raw.split(" (")[0].strip() if raw else ""
        return address
    if ftype in ("enum", "device_ref"):
        return widget.currentData()
    return widget.text().strip()
