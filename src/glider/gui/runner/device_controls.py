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
automatically. When a session function (a StartFunction→EndFunction chain) is
present, a Functions section of large run buttons is shown above the device
controls. This widget is PURE UI: it emits action-generic signals (and a
``function_run_requested`` signal); an external async handler dispatches device
actions through ``device.execute_action`` (the one chokepoint that clamps to the
declared range and serializes per device) and runs functions through the flow
engine's shared runner.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
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

from glider.core.graph_functions import build_picker_labels, list_graph_functions
from glider.gui.widgets.schema_form import build_schema_widgets, read_schema_widget
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
    # (start_node_id) — run a graph function from its tap button
    function_run_requested = pyqtSignal(str)
    # (device_id, action_name, [positional args]) -- an action driven with the
    # values from its declared argument fields.
    action_call_requested = pyqtSignal(str, str, object)

    def __init__(
        self,
        hardware_manager: HardwareManager,
        session_fn: Callable[[], object] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._hardware_manager = hardware_manager
        # Returns the current ExperimentSession (or None). When absent, the
        # Functions section is never built (used by device-only tests).
        self._session_fn = session_fn
        # value-label widgets keyed by (dev_id, action) for read display updates
        self._value_labels: dict[tuple[str, str], QLabel] = {}
        # interactive widgets keyed by (dev_id, action) -> {role: widget}, for
        # driving and for tests
        self._widgets: dict[tuple[str, str], dict[str, QWidget]] = {}
        # function run buttons keyed by start_node_id -> (button, base_label)
        self._function_buttons: dict[str, tuple[QPushButton, str]] = {}
        # last device-confirmed value per (dev_id, action) slider, so a failed
        # write can revert the control to what the device actually holds
        self._committed: dict[tuple[str, str], int] = {}
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

        # Persistent status strip: failures and clamp/load warnings land here as
        # icon + text and stay until superseded, since a touch-screen operator
        # never sees a transient status-bar toast. Hidden until there's news.
        self._status = QLabel()
        self._status.setObjectName("runnerStatusStrip")
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        layout.addWidget(self._status)

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
            # A read action is a measurement, not a command — classify it first,
            # even though its value_spec carries the reading's range (which nodes
            # use). Otherwise a whole-number read spec (e.g. AnalogInput "read")
            # would render as a write slider that calls read() with an argument.
            if action in _READ_ACTIONS or action.startswith("read"):
                if not read_seen:  # one read display per device
                    controls.append(("read", action, None))
                    read_seen = True
            elif spec is not None and spec.kind == KIND_SWITCH:
                controls.append(("switch", action, spec))
            elif spec is not None and spec.kind == KIND_WHOLE:
                if action in _SECONDARY and has_primary_value:
                    continue
                controls.append(("slider", action, spec))
            elif action in _SWITCH_REDUNDANT and has_switch:
                continue
            elif _needs_args(device, action):
                # An action with required arguments must never become a plain
                # fire button: that button called execute_action(action) with
                # none of them, which is a TypeError every time it is pressed.
                schema = _args_schema(device, action)
                if schema:
                    controls.append(("action_args", action, schema))
                # No declared schema means there is nothing to send. Omitted
                # rather than shown disabled: the runner is a touchscreen with
                # no tooltips, so a dead control explains nothing.
            else:
                controls.append(("button", action, None))
        return controls

    # --- Public API ---

    def refresh(self) -> None:
        """Rebuild the Functions section and all device sections.

        Functions (from the session graph) render above the device controls in
        the single shared scroll area, so a touchscreen operator scrolls one
        list. Both are rebuilt together since either can change on a load.
        """
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._value_labels.clear()
        self._widgets.clear()
        self._function_buttons.clear()
        self._committed.clear()

        functions_section = self._make_functions_section()
        if functions_section is not None:
            self._content_layout.addWidget(functions_section)

        for dev_id, device in self._hardware_manager.devices.items():
            section = self._make_device_section(dev_id, device)
            if section is not None:
                self._content_layout.addWidget(section)
        self._content_layout.addStretch(1)

    def _make_functions_section(self) -> QWidget | None:
        """Build the Functions run-button section, or None when there are none.

        Only complete functions (a reachable EndFunction) are offered — an
        incomplete chain would only ever time out. Duplicate display names are
        disambiguated; each button still binds by ``start_node_id``.
        """
        if self._session_fn is None:
            return None
        session = self._session_fn()
        infos = [f for f in list_graph_functions(session) if f.has_end]
        if not infos:
            return None

        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        heading = QLabel("Functions")
        heading.setProperty("class", "device-section-header")
        layout.addWidget(heading)

        for label, start_node_id in build_picker_labels(infos):
            btn = QPushButton(label)
            btn.setMinimumHeight(_CONTROL_MIN_HEIGHT)
            btn.clicked.connect(
                lambda _=False, sid=start_node_id: self.function_run_requested.emit(sid)
            )
            self._function_buttons[start_node_id] = (btn, label)
            layout.addWidget(btn)
        return section

    def set_function_running(self, start_node_id: str, running: bool) -> None:
        """Show a per-button running affordance (disabled + 'Running…').

        Re-enabled only when the caller reports the run truly ended, so the
        button never invites a second tap while the chain may still be driving
        hardware.
        """
        entry = self._function_buttons.get(start_node_id)
        if entry is None:
            return
        btn, base_label = entry
        btn.setEnabled(not running)
        btn.setText(f"{base_label} — Running…" if running else base_label)

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
                "action_args": self._make_action_args,
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
        # Seed the confirmed value to what the control starts at.
        self._committed[(dev_id, action)] = spin.value()
        return block

    def _make_button(self, dev_id: str, action: str, spec) -> QWidget:
        btn = QPushButton(_title(action))
        btn.setMinimumHeight(_CONTROL_MIN_HEIGHT)
        btn.clicked.connect(lambda _=False: self.action_fire_requested.emit(dev_id, action))
        self._widgets[(dev_id, action)] = {"button": btn}
        return btn

    def _make_action_args(self, dev_id: str, action: str, schema) -> QWidget:
        """Labelled fields plus a fire button, for an action with arguments.

        Stored under an ``args`` key rather than ``spin``/``slider`` on
        purpose: those keys drive the optimistic-revert path in
        on_action_failed, and there is no single committed value to snap an
        argument form back to.
        """
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._labeled(action))

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        fields: dict[str, tuple] = {}
        build_schema_widgets(form, schema, fields)
        for widget, _ftype in fields.values():
            widget.setMinimumHeight(_CONTROL_MIN_HEIGHT)
        layout.addLayout(form)

        btn = QPushButton(_title(action))
        btn.setMinimumHeight(_CONTROL_MIN_HEIGHT)
        btn.clicked.connect(
            lambda _=False: self.action_call_requested.emit(
                dev_id,
                action,
                [read_schema_widget(w, t) for w, t in fields.values()],
            )
        )
        layout.addWidget(btn)

        self._widgets[(dev_id, action)] = {"args": fields, "button": btn}
        return block

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

    # --- Status + optimistic feedback ---

    _LEVEL_ICON = {"error": "✗", "warn": "⚠", "info": "·"}

    def show_status(self, text: str, level: str = "error") -> None:
        """Show a persistent icon+text notice (text, not color alone)."""
        self._status.setText(f"{self._LEVEL_ICON.get(level, '·')}  {text}")
        self._status.setProperty("level", level)
        self._status.setVisible(True)
        # Re-polish so the dynamic 'level' property restyles the strip.
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)

    def clear_status(self) -> None:
        self._status.clear()
        self._status.setVisible(False)

    def on_action_succeeded(self, dev_id: str, action: str, *value) -> None:
        """A write/command succeeded: record the confirmed slider value, clear status.

        Recording the confirmed value lets a *later* failed write revert the
        slider to what the device actually holds rather than the rejected value.
        """
        key = (dev_id, action)
        widgets = self._widgets.get(key, {})
        if value and ("spin" in widgets or "slider" in widgets):
            try:
                self._committed[key] = int(value[0])
            except (TypeError, ValueError):
                pass
        self.clear_status()

    def on_action_failed(self, dev_id: str, action: str, message: str) -> None:
        """A write failed: revert an optimistic control and surface the failure."""
        key = (dev_id, action)
        widgets = self._widgets.get(key, {})
        switch = widgets.get("switch")
        if switch is not None:
            # The switch optimistically flipped on tap; put it back.
            switch.blockSignals(True)
            switch.setChecked(not switch.isChecked())
            switch.setText("ON" if switch.isChecked() else "OFF")
            switch.blockSignals(False)
        elif "spin" in widgets or "slider" in widgets:
            # The slider/spin shows the rejected value; snap it back to the last
            # value the device confirmed so the UI can't imply a set that failed.
            revert_to = self._committed.get(key)
            if revert_to is not None:
                if widgets.get("spin") is not None:
                    _set_quiet(widgets["spin"], revert_to)
                if widgets.get("slider") is not None:
                    _set_quiet(widgets["slider"], revert_to)
        self.show_status(message, level="error")


def _needs_args(device, action: str) -> bool:
    """Whether ``action`` cannot be called with no arguments.

    Asks the device first; falls back to signature introspection for a plugin
    device predating ``action_needs_args``, so an unknown device never gets a
    button that raises TypeError on the first press.
    """
    asker = getattr(device, "action_needs_args", None)
    if callable(asker):
        try:
            return bool(asker(action))
        except Exception:
            pass
    import inspect

    try:
        parameters = inspect.signature(device.actions[action]).parameters.values()
    except (KeyError, TypeError, ValueError):
        return False
    return any(
        param.default is inspect.Parameter.empty
        and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        for param in parameters
    )


def _args_schema(device, action: str) -> list[dict]:
    """The device's declared argument fields for ``action`` (empty if none)."""
    asker = getattr(device, "action_args_schema", None)
    if not callable(asker):
        return []
    try:
        return list(asker(action) or [])
    except Exception:
        return []


def _title(action: str) -> str:
    """Humanize an action identifier for display (``set_angle`` -> ``Set Angle``)."""
    return action.replace("_", " ").title()


def _set_quiet(widget, value) -> None:
    """Set a widget's value without emitting its change signals (avoids loops)."""
    widget.blockSignals(True)
    widget.setValue(value)
    widget.blockSignals(False)
