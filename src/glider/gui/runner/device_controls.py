"""Runner Device Controls — touch controls generated from each device's actions.

Instead of a fixed list of supported device families, this widget builds one
control per controllable action of every configured device, choosing the widget
from that action's declared value semantics (``device.value_spec(action)``):

* switch-kind value -> a stateful ON/OFF switch,
* whole-number value -> a slider (commit-on-release) paired with a bounded spin
  box for precise entry (nudge by the declared step, or type an exact value);
  a range too large for a usable slider falls back to the spin box alone,
* an action that takes no value -> a single command button,
* a readable action -> a Read button + value label (button-driven).

Custom/declarative devices therefore gain correct, range-aware controls
automatically. This widget is PURE UI: it emits action-generic signals; an
external async handler dispatches them through ``device.execute_action`` (the one
chokepoint that clamps to the declared range and serializes per device).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QScroller,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glider.hal.value_spec import KIND_SWITCH, KIND_WHOLE

if TYPE_CHECKING:
    from glider.core.hardware_manager import HardwareManager

_CONTROL_MIN_HEIGHT = 72  # touch-target floor (matches the runner tab height)
_SPACING = 10

# A range with more discrete steps than this is not usefully draggable on a
# ~400px touch track, so it renders as precise entry alone (no slider).
_SLIDER_MAX_STEPS = 4096

# Actions whose value is a readable measurement rather than a command.
_READ_ACTIONS = ("read", "read_voltage", "read_analog", "read_digital")

# Standard digital primitives that are subsumed by the single stateful switch
# (the "set" action), so they are not rendered as separate buttons.
_SWITCH_REDUNDANT = ("on", "off", "toggle")

# Secondary helper actions covered by a primary control of the same device.
_SECONDARY = ("set_percent",)


class RunnerDeviceControls(QWidget):
    """Touch controls generated from every device's declared actions."""

    # (device_id, action_name, value)
    action_write_requested = pyqtSignal(str, str, object)
    # (device_id, action_name) — an action that takes no value
    action_fire_requested = pyqtSignal(str, str)
    # (device_id, action_name) — a readable action
    read_requested = pyqtSignal(str, str)

    def __init__(self, hardware_manager: HardwareManager, parent: QWidget | None = None):
        super().__init__(parent)
        self._hardware_manager = hardware_manager
        # value-label widgets keyed by (dev_id, action) for read display updates
        self._value_labels: dict[tuple[str, str], QLabel] = {}
        # interactive widgets keyed by (dev_id, action) -> {role: widget}, for
        # driving and for tests
        self._widgets: dict[tuple[str, str], dict[str, QWidget]] = {}
        self.setObjectName("runnerDeviceControls")
        self._setup_ui()

    # --- UI scaffolding ---

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(_SPACING)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setStyleSheet("background-color: transparent;")
        QScroller.grabGesture(
            self._scroll.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(_SPACING)
        self._scroll.setWidget(self._content)
        layout.addWidget(self._scroll, 1)

    # --- Control planning ---

    def _controls_for(self, device) -> list[tuple[str, str, object]]:
        """Return ``(kind, action, spec)`` descriptors for a device's controls.

        Order follows the device's own action order (stable across refreshes).
        Redundant digital primitives and secondary helpers are skipped when a
        primary control already covers them, so a digital output shows one
        switch (not On/Off/Toggle) and a PWM output shows one slider (not also
        ``set_percent``).
        """
        actions = list(device.actions.keys())
        has_switch = any(
            (s := device.value_spec(a)) is not None and s.kind == KIND_SWITCH for a in actions
        )
        has_primary_value = any(device.value_spec(a) is not None for a in actions)
        controls: list[tuple[str, str, object]] = []
        read_seen = False
        for action in actions:
            spec = device.value_spec(action)
            if spec is not None and spec.kind == KIND_SWITCH:
                controls.append(("switch", action, spec))
            elif spec is not None and spec.kind == KIND_WHOLE:
                if action in _SECONDARY and has_primary_value:
                    continue
                controls.append(("slider", action, spec))
            elif action in _READ_ACTIONS or action.startswith("read"):
                if not read_seen:  # one read display per device
                    controls.append(("read", action, None))
                    read_seen = True
            elif action in _SWITCH_REDUNDANT and has_switch:
                continue
            else:
                controls.append(("button", action, None))
        return controls

    # --- Public API ---

    def refresh(self) -> None:
        """Rebuild all device sections from ``hardware_manager.devices``."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._value_labels.clear()
        self._widgets.clear()

        for dev_id, device in self._hardware_manager.devices.items():
            section = self._make_device_section(dev_id, device)
            if section is not None:
                self._content_layout.addWidget(section)
        self._content_layout.addStretch(1)

    def _make_device_section(self, dev_id: str, device) -> QWidget | None:
        controls = self._controls_for(device)
        if not controls:
            return None
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        heading = QLabel(dev_id)
        heading.setProperty("class", "device-section-header")
        layout.addWidget(heading)

        for kind, action, spec in controls:
            builder = {
                "switch": self._make_switch,
                "slider": self._make_slider,
                "read": self._make_read,
                "button": self._make_button,
            }[kind]
            layout.addWidget(builder(dev_id, action, spec))
        return section

    # --- Control builders ---

    def _labeled(self, action: str, unit: str = "") -> QLabel:
        text = _title(action) + (f" ({unit})" if unit else "")
        return QLabel(text)

    def _make_switch(self, dev_id: str, action: str, spec) -> QWidget:
        block = QWidget()
        row = QHBoxLayout(block)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_SPACING)
        row.addWidget(self._labeled(action))
        switch = QPushButton("OFF")
        switch.setCheckable(True)
        switch.setMinimumHeight(_CONTROL_MIN_HEIGHT)

        def _on_toggle(checked: bool) -> None:
            switch.setText("ON" if checked else "OFF")
            self.action_write_requested.emit(dev_id, action, bool(checked))

        switch.toggled.connect(_on_toggle)
        row.addWidget(switch, 1)
        self._widgets[(dev_id, action)] = {"switch": switch}
        return block

    def _make_slider(self, dev_id: str, action: str, spec) -> QWidget:
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._labeled(action, spec.unit))

        row = QHBoxLayout()
        row.setSpacing(_SPACING)

        spin = QSpinBox()
        spin.setRange(spec.min, spec.max)
        spin.setSingleStep(max(1, spec.step))
        spin.setMinimumHeight(_CONTROL_MIN_HEIGHT)

        steps = (spec.max - spec.min) // max(1, spec.step)
        slider = None
        if steps <= _SLIDER_MAX_STEPS:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(spec.min, spec.max)
            slider.setSingleStep(max(1, spec.step))
            slider.setMinimumHeight(_CONTROL_MIN_HEIGHT)

            # Live readout during drag, but commit (write) only on release, so
            # dragging never ramps the device through intermediate values.
            slider.valueChanged.connect(lambda v: _set_quiet(spin, v))
            slider.sliderReleased.connect(
                lambda: self.action_write_requested.emit(dev_id, action, slider.value())
            )
            row.addWidget(slider, 1)

        # The spin box is the precise-entry path: type a value or nudge by step;
        # committing it writes and syncs the slider.
        def _commit_spin() -> None:
            if slider is not None:
                _set_quiet(slider, spin.value())
            self.action_write_requested.emit(dev_id, action, spin.value())

        spin.editingFinished.connect(_commit_spin)
        row.addWidget(spin)
        layout.addLayout(row)
        self._widgets[(dev_id, action)] = {"spin": spin, "slider": slider}
        return block

    def _make_button(self, dev_id: str, action: str, spec) -> QWidget:
        btn = QPushButton(_title(action))
        btn.setMinimumHeight(_CONTROL_MIN_HEIGHT)
        btn.clicked.connect(lambda _=False: self.action_fire_requested.emit(dev_id, action))
        self._widgets[(dev_id, action)] = {"button": btn}
        return btn

    def _make_read(self, dev_id: str, action: str, spec) -> QWidget:
        block = QWidget()
        row = QHBoxLayout(block)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_SPACING)
        read_btn = QPushButton("Read")
        read_btn.setMinimumHeight(_CONTROL_MIN_HEIGHT)
        read_btn.clicked.connect(lambda _=False: self.read_requested.emit(dev_id, action))
        value_label = QLabel("—")
        self._value_labels[(dev_id, action)] = value_label
        self._widgets[(dev_id, action)] = {"read": read_btn}
        row.addWidget(read_btn)
        row.addWidget(value_label, 1)
        return block

    def set_read_value(self, dev_id: str, action: str, text: str) -> None:
        lbl = self._value_labels.get((dev_id, action))
        if lbl is not None:
            lbl.setText(text)


def _title(action: str) -> str:
    """Humanize an action identifier for display (``set_angle`` -> ``Set Angle``)."""
    return action.replace("_", " ").title()


def _set_quiet(widget, value) -> None:
    """Set a widget's value without emitting its change signals (avoids loops)."""
    widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(False)
