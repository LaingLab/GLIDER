"""The one piece of Builder chrome that cannot be summoned.

Everything else in the shell is on demand: the panels collapse to rails, the
palette appears on a keystroke, the menus are four. This strip is the exception,
and the exception is the point of it. Live hardware and recording state cannot
wait to be asked for -- a board dropping 40 minutes into an unattended overnight
run has to be visible to whoever walks past the rig, not to whoever thinks to
open the Hardware panel. That is also why the device dots live here rather than
being left to that panel: the panel can be collapsed, and this cannot.

Left to right, in 40 fixed pixels: the two panel toggles, the experiment name
with its dirty marker, the run-state pill, a spacer, one dot per device, and the
``Ctrl K`` hint.

Four decisions carry meaning rather than convenience:

* **A device carries its name, not just a dot.** The dot is 8 px. Anyone who has
  to act on a red one at 3 a.m. needs to know *which* board, and a colour alone
  cannot say that. The name is what makes the alarm actionable.

* **An unrecognised device state is ``unknown``, never ``ok``.** States arrive
  from drivers, so the vocabulary can widen without this file changing. Painting
  a word nobody recognises the same green as a working board is the one
  behaviour that would make this widget actively harmful, so unfamiliar states
  go neutral and keep their raw text in the tooltip.

* **An unrecognised *run* state raises.** The asymmetry is deliberate. Device
  states come from hardware and the outside world gets the benefit of the doubt;
  run state comes from our own code, and a pill quietly stuck on "Idle" through
  a recording is precisely the failure this widget exists to prevent.

* **The toggles are told, not asked.** They reflect the panels; they do not own
  them. :meth:`StatusStrip.set_left_expanded` therefore sets the button without
  re-emitting, or echoing a panel's state back would ping-pong with the click
  that caused it.

**No colour is set from Python here.** Every part carries an ``objectName`` --
and, where it varies, a dynamic ``state`` property -- and ``desktop.qss`` owns
the rest. A widget-level ``setStyleSheet`` out-prioritises the application
stylesheet, so one stray call would make that widget the single thing in the
shell the theme cannot reach. Qt resolves property selectors at polish time
rather than when the property is set, so :func:`_restyle` re-polishes anything
whose ``state`` changed; without it the pill keeps the colour it opened with
while reporting the right property. This is the pattern ``plugin_card.py``
established and ``side_panel.py`` follows.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QWidget,
)

__all__ = [
    "DEVICE_STATES",
    "RUN_STATE_TEXT",
    "STRIP_HEIGHT",
    "StatusStrip",
]

#: Height of the strip, in pixels. The budget the whole redesign is measured
#: against: one strip in place of a menu bar and two toolbars.
STRIP_HEIGHT = 40

#: Side of a device dot, in pixels. Even so ``desktop.qss`` can halve it for the
#: border radius and get a circle rather than a lozenge.
_DOT = 8

#: The run states, and the word the pill shows for each. Four distinct labels on
#: purpose -- a pill that reads the same in two states is decoration.
RUN_STATE_TEXT = {
    "idle": "Idle",
    "running": "Running",
    "recording": "Recording",
    "error": "Error",
}

#: The device states this strip knows how to paint. Anything else becomes
#: ``"unknown"``; see the module docstring for why that is never ``"ok"``.
DEVICE_STATES = ("ok", "warn", "error", "unknown")

#: Shown when a session has no file yet, which is the state the Builder opens
#: in. A blank strip would read as broken rather than as new.
_UNNAMED = "Untitled"

#: The dirty marker. Its own label rather than part of the name, so the
#: stylesheet can grey it without greying the name and so a caller reading the
#: name back gets the name.
_DIRTY_MARK = "— edited"


def _restyle(widget: QWidget) -> None:
    """Make Qt re-evaluate *widget* after a dynamic property changed.

    Qt resolves ``[state="..."]`` selectors at polish time, not when the
    property is set, so a rule applied later never takes effect without this.

    (The third copy of this helper in the GUI, after ``plugin_card.py`` and
    ``side_panel.py``. Kept local rather than imported across modules so no
    widget depends on another widget's private name; worth consolidating into
    the styles package once something needs a fourth.)
    """
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)


class StatusStrip(QFrame):
    """The always-present strip above the Builder's content surface.

    Args:
        parent: Standard Qt parent.

    Signals:
        left_toggled: The left panel's toggle was pressed. The strip does not
            act on it -- the owner collapses the panel and echoes the result
            back through :meth:`set_left_expanded`.
        right_toggled: As above, for the right panel.
        palette_requested: The ``Ctrl K`` hint was pressed. It is a button and
            not a decoration because a hint that teaches a shortcut and then
            refuses to perform it is worse than no hint.
    """

    left_toggled = pyqtSignal()
    right_toggled = pyqtSignal()
    palette_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusStrip")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedHeight(STRIP_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._run_state = "idle"
        self._device_names: list[str] = []
        self._chips: list[QFrame] = []
        self._dots: list[QLabel] = []
        self._names: list[QLabel] = []

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 0, 12, 0)
        outer.setSpacing(12)

        toggles = QFrame(self)
        toggles.setObjectName("statusStripToggles")
        toggles.setFrameShape(QFrame.Shape.NoFrame)
        toggle_layout = QHBoxLayout(toggles)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        toggle_layout.setSpacing(4)
        self._left_toggle = self._build_toggle("left", "▌", self.left_toggled)
        self._right_toggle = self._build_toggle("right", "▐", self.right_toggled)
        toggle_layout.addWidget(self._left_toggle)
        toggle_layout.addWidget(self._right_toggle)
        outer.addWidget(toggles)

        self._name = QLabel(_UNNAMED, self)
        self._name.setObjectName("statusStripExperiment")
        outer.addWidget(self._name)

        self._dirty = QLabel(_DIRTY_MARK, self)
        self._dirty.setObjectName("statusStripDirty")
        self._dirty.setVisible(False)
        outer.addWidget(self._dirty)

        self._pill = QLabel(RUN_STATE_TEXT["idle"], self)
        self._pill.setObjectName("statusStripPill")
        self._pill.setProperty("state", "idle")
        outer.addWidget(self._pill)

        outer.addStretch(1)

        self._devices = QFrame(self)
        self._devices.setObjectName("statusStripDevices")
        self._devices.setFrameShape(QFrame.Shape.NoFrame)
        self._device_layout = QHBoxLayout(self._devices)
        self._device_layout.setContentsMargins(0, 0, 0, 0)
        self._device_layout.setSpacing(10)
        outer.addWidget(self._devices)

        self._palette_hint = QToolButton(self)
        self._palette_hint.setObjectName("statusStripPaletteHint")
        self._palette_hint.setText("Ctrl K")
        self._palette_hint.setToolTip("Command palette (Ctrl+K)")
        self._palette_hint.setCursor(Qt.CursorShape.PointingHandCursor)
        self._palette_hint.clicked.connect(lambda _checked=False: self.palette_requested.emit())
        outer.addWidget(self._palette_hint)

    # ----------------------------------------------------------------- build

    def _build_toggle(self, side: str, glyph: str, signal) -> QToolButton:
        button = QToolButton(self)
        button.setObjectName("statusStripToggle")
        button.setProperty("side", side)
        button.setText(glyph)
        button.setCheckable(True)
        button.setChecked(True)
        button.setToolTip(f"Show or hide the {side} panel")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda _checked=False: signal.emit())
        return button

    # ------------------------------------------------------------ experiment

    def set_experiment(self, name: str, dirty: bool = False) -> None:
        """Show *name*, with the dirty marker beside it when *dirty*.

        An empty name falls back to :data:`_UNNAMED` rather than blanking the
        strip: the Builder opens before any file exists, and that is a new
        session rather than a broken one.
        """
        self._name.setText(name.strip() or _UNNAMED)
        self._dirty.setVisible(bool(dirty))

    def name_label(self) -> QLabel:
        """The label carrying the experiment name."""
        return self._name

    def dirty_label(self) -> QLabel:
        """The label carrying the dirty marker. Hidden while saved."""
        return self._dirty

    # -------------------------------------------------------------- run state

    def run_state(self) -> str:
        """The current run state."""
        return self._run_state

    def set_run_state(self, state: str) -> None:
        """Move the pill to *state*.

        Args:
            state: One of the keys of :data:`RUN_STATE_TEXT`.

        Raises:
            ValueError: *state* is not one of the four. Deliberately loud: run
                state comes from our own code, and a pill silently stuck on the
                previous state through a recording is the exact failure this
                widget exists to prevent.
        """
        if state not in RUN_STATE_TEXT:
            raise ValueError(
                f"unknown run state {state!r}; expected one of {sorted(RUN_STATE_TEXT)}"
            )
        self._run_state = state
        self._pill.setText(RUN_STATE_TEXT[state])
        self._pill.setProperty("state", state)
        _restyle(self._pill)

    def pill(self) -> QLabel:
        """The run-state pill."""
        return self._pill

    # ---------------------------------------------------------------- devices

    def set_devices(self, devices: Iterable[Sequence[str]]) -> None:
        """Show one dot, named, per device.

        Args:
            devices: ``(name, state)`` pairs, in the order they should appear.
                A state outside :data:`DEVICE_STATES` renders neutral and keeps
                its raw text in the tooltip -- never the healthy green.

        When the names are unchanged the existing widgets are updated in place
        and only the dots whose state actually moved are re-polished. That keeps
        a strip refreshed on a timer from churning every widget on it, and it is
        what makes "only that device changed" a claim anyone can check.
        """
        items = [(str(name), str(state)) for name, state in devices]
        names = [name for name, _ in items]

        if names != self._device_names:
            self._rebuild(items)
            return

        for chip, dot, (name, raw) in zip(self._chips, self._dots, items, strict=True):
            self._apply_device(chip, dot, name, raw)

    def _rebuild(self, items: list[tuple[str, str]]) -> None:
        for chip in self._chips:
            self._device_layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._chips, self._dots, self._names = [], [], []

        for name, raw in items:
            chip = QFrame(self._devices)
            chip.setObjectName("statusStripDevice")
            chip.setFrameShape(QFrame.Shape.NoFrame)
            layout = QHBoxLayout(chip)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            dot = QLabel(chip)
            dot.setObjectName("statusStripDot")
            dot.setFixedSize(_DOT, _DOT)
            layout.addWidget(dot)

            label = QLabel(name, chip)
            label.setObjectName("statusStripDeviceName")
            layout.addWidget(label)

            self._apply_device(chip, dot, name, raw)
            self._device_layout.addWidget(chip)
            self._chips.append(chip)
            self._dots.append(dot)
            self._names.append(label)

        self._device_names = [name for name, _ in items]
        # An empty rig is the normal first thing this widget is told, and an
        # empty container would still claim its spacing beside the hint.
        self._devices.setVisible(bool(items))

    def _apply_device(self, chip: QFrame, dot: QLabel, name: str, raw: str) -> None:
        state = raw if raw in DEVICE_STATES else "unknown"
        chip.setToolTip(f"{name} — {raw}")
        if dot.property("state") == state:
            return
        for widget in (chip, dot):
            widget.setProperty("state", state)
            _restyle(widget)

    def device_names(self) -> list[str]:
        """The device names as rendered, in order."""
        return [label.text() for label in self._names]

    def device_dots(self) -> list[QLabel]:
        """The device dots, in order."""
        return list(self._dots)

    def device_chips(self) -> list[QFrame]:
        """The dot-and-name pairs, in order."""
        return list(self._chips)

    # ---------------------------------------------------------------- toggles

    def set_left_expanded(self, expanded: bool) -> None:
        """Show the left panel as open or collapsed, without emitting."""
        self._set_toggle(self._left_toggle, expanded)

    def set_right_expanded(self, expanded: bool) -> None:
        """Show the right panel as open or collapsed, without emitting."""
        self._set_toggle(self._right_toggle, expanded)

    @staticmethod
    def _set_toggle(button: QToolButton, expanded: bool) -> None:
        blocked = button.blockSignals(True)
        button.setChecked(bool(expanded))
        button.blockSignals(blocked)

    def left_toggle(self) -> QToolButton:
        """The left panel's toggle button."""
        return self._left_toggle

    def right_toggle(self) -> QToolButton:
        """The right panel's toggle button."""
        return self._right_toggle

    def palette_hint(self) -> QToolButton:
        """The ``Ctrl K`` hint, which is a button."""
        return self._palette_hint
