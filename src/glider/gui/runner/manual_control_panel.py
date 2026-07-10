"""
Manual Control Panel - Touch-optimized button grid for the Runner.

A 2-column scrollable grid of large buttons. Each button is bound to a graph
StartFunction chain (by ``start_node_id``) and emits ``function_run_requested``
on tap so an external component can run the chain. Assignment, reassignment,
and clearing happen inline on the touchscreen.

This widget is PURE UI: it does not run anything itself.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QScrollArea,
    QScroller,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from glider.core.graph_functions import find_run_param, list_graph_functions
from glider.gui.runner.device_controls import RunnerDeviceControls
from glider.gui.styles import colors

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore

logger = logging.getLogger(__name__)

# States in which manual-control buttons must be blocked.
_BLOCKED_STATES = {"ERROR", "STOPPING"}

# How long (ms) to hold a button before the press-and-hold menu opens.
_LONG_PRESS_MS = 600


def build_picker_labels(infos):
    """Build (display_label, start_node_id) pairs, disambiguating dup names.

    Functions sharing a name get a suffix of the last 4 chars of their node id
    so the operator can tell them apart. Buttons still bind by id.
    """
    name_counts = Counter(info.name for info in infos)
    labels = []
    for info in infos:
        if name_counts[info.name] > 1:
            display = f"{info.name} {info.start_node_id[-4:]}"
        else:
            display = info.name
        labels.append((display, info.start_node_id))
    return labels


class _SlotTile(QWidget):
    """A run button with an always-enabled ✕ delete badge in the corner.

    The badge is a sibling of the run button (not a child), so disabling the run
    button (e.g. no hardware connected) does not disable the badge -- delete
    stays available in any state.
    """

    def __init__(self, run_button: QPushButton, delete_button: QPushButton, parent=None):
        super().__init__(parent)
        self._run = run_button
        self._delete = delete_button
        self._run.setParent(self)
        self._delete.setParent(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._run)
        self._delete.raise_()  # float over the run button's top-right

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 4
        self._delete.move(self.width() - self._delete.width() - margin, margin)


class ManualControlPanel(QWidget):
    """Touch button-grid bound to graph functions by ``start_node_id``."""

    function_run_requested = pyqtSignal(str)  # emits a start_node_id
    # emits (start_node_id, param dict) for a parameterized run (e.g. N revolutions)
    function_run_requested_param = pyqtSignal(str, object)

    # Re-emitted from the live-run device controls page so an external
    # component (main_window) can wire hardware handlers in one place.
    set_digital_requested = pyqtSignal(str, bool)
    toggle_digital_requested = pyqtSignal(str)
    set_pwm_requested = pyqtSignal(str, int)

    def __init__(self, core: GliderCore, parent: QWidget | None = None):
        super().__init__(parent)
        self._core = core
        self._state = "IDLE"
        self._running_node_id: str | None = None

        # slot -> QPushButton for the real (assigned) tiles.
        self._slot_buttons: dict[int, QPushButton] = {}
        # slot -> always-enabled ✕ delete badge.
        self._delete_buttons: dict[int, QPushButton] = {}
        # slot -> press-and-hold timer.
        self._press_timers: dict[int, QTimer] = {}

        # Run stopwatch: monotonic start of the active run, and the last
        # completed duration per function (keyed by start_node_id, so it
        # survives grid rebuilds and re-packing).
        self._run_start: float | None = None
        self._last_durations: dict[str, float] = {}
        self._run_tick = QTimer(self)
        self._run_tick.setInterval(200)
        self._run_tick.timeout.connect(self._update_tile_labels)

        self.setObjectName("manualControlPanel")
        self._setup_ui()
        self.refresh()

    # --- UI scaffolding ---

    def _setup_ui(self) -> None:
        """Build the scroll area + content grid once."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

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
        self._grid = QGridLayout(self._content)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(8)
        self._scroll.setWidget(self._content)

        # Panel-level right-click menu: fires even over a disabled tile (the
        # event propagates from the disabled child to this enabled container).
        self._content.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._content.customContextMenuRequested.connect(self._on_content_context_menu)

        # Mode-aware stack: page 0 is the function grid (idle / manual runs),
        # page 1 is the live-run device controls (recorded experiment running).
        self._stack = QStackedWidget()
        self._stack.addWidget(self._scroll)  # index 0

        self._device_controls = RunnerDeviceControls(self._core.hardware_manager)
        self._stack.addWidget(self._device_controls)  # index 1

        self._device_controls.set_digital_requested.connect(self.set_digital_requested)
        self._device_controls.toggle_digital_requested.connect(self.toggle_digital_requested)
        self._device_controls.set_pwm_requested.connect(self.set_pwm_requested)

        layout.addWidget(self._stack, 1)

    # --- Public API ---

    def refresh(self) -> None:
        """Rebuild the grid from session.manual_controls (sorted by slot).

        Mode-aware: while the live-run device controls page is showing,
        refresh that page instead of rebuilding the (hidden) function grid.
        """
        if getattr(self, "_stack", None) is not None and self._stack.currentIndex() == 1:
            self._device_controls.refresh()
            return

        # Tear down old buttons/timers.
        for timer in self._press_timers.values():
            timer.stop()
        self._press_timers.clear()
        self._slot_buttons.clear()
        self._delete_buttons.clear()

        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        entries = sorted(self._core.session.manual_controls, key=lambda e: e["slot"])
        known_ids = {info.start_node_id for info in list_graph_functions(self._core.session)}

        columns = 2
        index = 0
        for entry in entries:
            slot = entry["slot"]
            is_missing = entry["start_node_id"] not in known_ids
            tile = self._make_slot_button(slot, entry, is_missing)
            self._grid.addWidget(tile, index // columns, index % columns)
            index += 1

        # Trailing "+ Assign" tile.
        assign_btn = QPushButton("＋ Assign")
        assign_btn.setMinimumHeight(96)
        assign_btn.setProperty("manualTile", "assign")
        assign_btn.clicked.connect(lambda: self._open_picker(None))
        self._grid.addWidget(assign_btn, index // columns, index % columns)

        self._apply_enable_state()
        self._update_tile_labels()  # restore running/last-run text after rebuild

    def assign_function(self, slot: int, start_node_id: str, label: str) -> None:
        """Upsert (by slot) an entry into manual_controls and refresh."""
        entries = [dict(e) for e in self._core.session.manual_controls]
        for entry in entries:
            if entry["slot"] == slot:
                entry["start_node_id"] = start_node_id
                entry["label"] = label
                break
        else:
            entries.append({"slot": slot, "start_node_id": start_node_id, "label": label})
        self._core.session.set_manual_controls(entries)
        self.refresh()

    def clear_slot(self, slot: int) -> None:
        """Remove the entry at slot and re-pack remaining slots to 0..n-1."""
        remaining = [
            dict(e)
            for e in sorted(self._core.session.manual_controls, key=lambda e: e["slot"])
            if e["slot"] != slot
        ]
        for new_slot, entry in enumerate(remaining):
            entry["slot"] = new_slot
        self._core.session.set_manual_controls(remaining)
        self.refresh()

    def activate_slot(self, slot: int) -> None:
        """Run the slot's function, prompting for a value if it's parameterized.

        If the function contains a revolution- or counts-mode WaitForInput, a
        touch number pad asks for the target (revolutions or counts), which is
        injected into that node before the function runs.
        """
        entry = self._entry_for_slot(slot)
        if entry is None:
            return
        start_node_id = entry["start_node_id"]
        run_param = self._run_param(start_node_id)
        if run_param is None:
            self.function_run_requested.emit(start_node_id)
            return

        from glider.gui.dialogs.number_pad_dialog import NumberPadDialog

        value = NumberPadDialog.get_int(
            run_param.label, value=run_param.value, minimum=1, maximum=10_000_000, parent=self
        )
        if value is None:
            return  # cancelled
        param = {"node_id": run_param.node_id, "state_key": run_param.state_key, "value": value}
        self.function_run_requested_param.emit(start_node_id, param)

    def _run_param(self, start_node_id: str):
        """The parameterizable (revolution/counts) WaitForInput here, if any."""
        try:
            session = self._core.session
            if session is None:
                return None
            return find_run_param(start_node_id, session.flow)
        except Exception:  # never block a run on detection
            return None

    def is_slot_enabled(self, slot: int) -> bool:
        """Effective enabled state for a slot, per the spec rules."""
        return self._compute_enabled(slot)

    def set_running(self, node_id_or_none: str | None) -> None:
        """Mark a node as the active run (or clear), driving the run stopwatch.

        On start, the running tile counts up live; on finish it freezes showing
        that run's duration until the function is run again.
        """
        if node_id_or_none is not None:
            self._running_node_id = node_id_or_none
            self._run_start = time.monotonic()
            self._run_tick.start()
        else:
            if self._running_node_id is not None and self._run_start is not None:
                self._last_durations[self._running_node_id] = time.monotonic() - self._run_start
            self._running_node_id = None
            self._run_start = None
            self._run_tick.stop()
        self._apply_enable_state()
        self._update_tile_labels()

    def update_state(self, state_name: str) -> None:
        """Update the core state name and re-apply enable-state.

        A recorded experiment ("RUNNING", delivered here) swaps the function
        grid for the live-run device controls. A manual function run (see
        set_running) never reaches this method with "RUNNING" and must not
        switch modes.
        """
        self._state = state_name
        self._apply_enable_state()

        running = state_name == "RUNNING"
        self._stack.setCurrentIndex(1 if running else 0)
        if running:
            self._device_controls.refresh()

    # --- Enable-state (single source of truth) ---

    def _compute_enabled(self, slot: int) -> bool:
        entry = self._entry_for_slot(slot)
        slot_is_assigned = entry is not None
        hardware_connected = self._core.hardware_manager.is_any_board_connected()
        return (
            hardware_connected
            and self._state not in _BLOCKED_STATES
            and self._running_node_id is None
            and slot_is_assigned
        )

    def _apply_enable_state(self) -> None:
        """Push computed enable-state + running visuals onto every button."""
        for slot, btn in self._slot_buttons.items():
            entry = self._entry_for_slot(slot)
            is_running = entry is not None and entry["start_node_id"] == self._running_node_id
            if is_running:
                btn.setEnabled(False)
                btn.setProperty("manualTile", "running")
            else:
                btn.setEnabled(self._compute_enabled(slot))
                # Preserve "missing" styling; otherwise mark as a normal slot.
                if btn.property("manualTile") != "missing":
                    btn.setProperty("manualTile", "slot")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # --- Run stopwatch ---

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        total = int(seconds)
        return f"{total // 60}:{total % 60:02d}"

    def _update_tile_labels(self) -> None:
        """Show the live run time on the running tile and the last duration on
        finished ones; plain label otherwise."""
        for slot, btn in self._slot_buttons.items():
            entry = self._entry_for_slot(slot)
            if entry is None:
                continue
            label = entry.get("label", "")
            nid = entry["start_node_id"]
            if nid == self._running_node_id and self._run_start is not None:
                btn.setText(f"{label}\n{self._fmt_elapsed(time.monotonic() - self._run_start)}")
            elif nid in self._last_durations:
                btn.setText(f"{label}\nLast: {self._fmt_elapsed(self._last_durations[nid])}")
            else:
                btn.setText(label)

    # --- Helpers ---

    def _entry_for_slot(self, slot: int) -> dict | None:
        for entry in self._core.session.manual_controls:
            if entry["slot"] == slot:
                return entry
        return None

    def _make_slot_button(self, slot: int, entry: dict, is_missing: bool) -> QWidget:
        """Create a slot tile: a run button + an always-enabled ✕ delete badge."""
        btn = QPushButton(entry.get("label", ""))
        btn.setMinimumHeight(96)
        btn.setProperty("manualTile", "missing" if is_missing else "slot")
        if is_missing:
            btn.setStyleSheet(f"color: {colors.TEXT_MUTED};")

        btn.clicked.connect(lambda _checked=False, s=slot: self._on_clicked(s))

        # Right-click (desktop) opens Reassign/Clear. The menu is handled at the
        # panel level (see _on_content_context_menu) rather than per-button,
        # because a disabled tile (e.g. no hardware connected) doesn't receive
        # context-menu events. Touch devices use the long-press below.

        # Press-and-hold timer opens the per-slot context menu (touchscreens).
        timer = QTimer(btn)
        timer.setSingleShot(True)
        timer.setInterval(_LONG_PRESS_MS)
        timer.timeout.connect(lambda s=slot: self._show_slot_menu(s))
        self._press_timers[slot] = timer

        # Guard timer access: refresh() clears _press_timers and reparenting a
        # still-pressed button makes Qt emit released() synchronously, which
        # would otherwise KeyError on a slot that's mid-rebuild.
        btn.pressed.connect(lambda s=slot: self._start_press_timer(s))
        btn.released.connect(lambda s=slot: self._stop_press_timer(s))

        self._slot_buttons[slot] = btn

        # Always-enabled ✕ delete badge (sibling of btn in the tile), so it
        # stays clickable on desktop and touch even when the tile is disabled.
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(28, 28)
        del_btn.setToolTip("Remove this button")
        del_btn.setProperty("manualTile", "delete")
        del_btn.clicked.connect(lambda _checked=False, s=slot: self.clear_slot(s))
        self._delete_buttons[slot] = del_btn

        return _SlotTile(btn, del_btn)

    def _start_press_timer(self, slot: int) -> None:
        timer = self._press_timers.get(slot)
        if timer is not None:
            timer.start()

    def _stop_press_timer(self, slot: int) -> None:
        timer = self._press_timers.get(slot)
        if timer is not None:
            timer.stop()

    def _on_content_context_menu(self, pos) -> None:
        """Right-click anywhere on a tile -> that slot's Reassign/Clear menu.

        Resolves the tile under the cursor via childAt, which returns disabled
        widgets too, so Clear works regardless of a tile's run-enabled state.
        """
        child = self._content.childAt(pos)
        if child is None:
            return
        for slot, btn in self._slot_buttons.items():
            if btn is child or btn.isAncestorOf(child):
                self._show_slot_menu(slot)
                return

    def _on_clicked(self, slot: int) -> None:
        """A real tap: cancel any pending long-press and activate."""
        timer = self._press_timers.get(slot)
        if timer is not None:
            timer.stop()
        self.activate_slot(slot)

    # --- Long-press menu ---

    def _show_slot_menu(self, slot: int) -> None:
        """Press-and-hold menu: Reassign / Clear."""
        btn = self._slot_buttons.get(slot)
        if btn is None:
            return
        menu = QMenu(self)
        reassign_action = menu.addAction("Reassign")
        reassign_action.triggered.connect(lambda: self._open_picker(slot))
        clear_action = menu.addAction("Clear")
        clear_action.triggered.connect(lambda: self.clear_slot(slot))
        menu.exec(btn.mapToGlobal(btn.rect().center()))

    # --- Assignment picker ---

    def _next_free_slot(self) -> int:
        used = {e["slot"] for e in self._core.session.manual_controls}
        n = 0
        while n in used:
            n += 1
        return n

    def _open_picker(self, slot: int | None) -> None:
        """Open a touch-friendly picker of assignable functions.

        Only functions with has_end is True are offered (a chain with no
        EndFunction would only ever time out). On selection, assign by id.
        """
        infos = [info for info in list_graph_functions(self._core.session) if info.has_end]
        labels = build_picker_labels(infos)
        # Map display label back to (start_node_id, original name) for assignment.
        meta = {disp: (nid, info.name) for (disp, nid), info in zip(labels, infos, strict=True)}

        target_slot = slot if slot is not None else self._next_free_slot()

        dialog = QDialog(self)
        dialog.setWindowTitle("Assign Function")
        dialog.setMinimumSize(360, 480)
        dlg_layout = QVBoxLayout(dialog)
        dlg_layout.setContentsMargins(8, 8, 8, 8)
        dlg_layout.setSpacing(8)

        list_widget = QListWidget()
        list_widget.setProperty("manualPicker", True)
        for disp, _nid in labels:
            item = QListWidgetItem(disp)
            item.setSizeHint(item.sizeHint().expandedTo(QSize(0, 56)))
            list_widget.addItem(item)
        QScroller.grabGesture(
            list_widget.viewport(), QScroller.ScrollerGestureType.LeftMouseButtonGesture
        )
        dlg_layout.addWidget(list_widget, 1)

        def _accept(item: QListWidgetItem) -> None:
            disp = item.text()
            nid, name = meta[disp]
            self.assign_function(slot=target_slot, start_node_id=nid, label=name)
            dialog.accept()

        list_widget.itemClicked.connect(_accept)
        dialog.exec()
