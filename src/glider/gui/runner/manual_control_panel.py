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
from collections import Counter
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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
    QVBoxLayout,
    QWidget,
)

from glider.core.graph_functions import list_graph_functions
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


class ManualControlPanel(QWidget):
    """Touch button-grid bound to graph functions by ``start_node_id``."""

    function_run_requested = pyqtSignal(str)  # emits a start_node_id

    def __init__(self, core: GliderCore, parent: QWidget | None = None):
        super().__init__(parent)
        self._core = core
        self._state = "IDLE"
        self._running_node_id: str | None = None

        # slot -> QPushButton for the real (assigned) tiles.
        self._slot_buttons: dict[int, QPushButton] = {}
        # slot -> press-and-hold timer.
        self._press_timers: dict[int, QTimer] = {}

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

        layout.addWidget(self._scroll, 1)

    # --- Public API ---

    def refresh(self) -> None:
        """Rebuild the grid from session.manual_controls (sorted by slot)."""
        # Tear down old buttons/timers.
        for timer in self._press_timers.values():
            timer.stop()
        self._press_timers.clear()
        self._slot_buttons.clear()

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
            btn = self._make_slot_button(slot, entry, is_missing)
            self._slot_buttons[slot] = btn
            self._grid.addWidget(btn, index // columns, index % columns)
            index += 1

        # Trailing "+ Assign" tile.
        assign_btn = QPushButton("＋ Assign")
        assign_btn.setMinimumHeight(96)
        assign_btn.setProperty("manualTile", "assign")
        assign_btn.clicked.connect(lambda: self._open_picker(None))
        self._grid.addWidget(assign_btn, index // columns, index % columns)

        self._apply_enable_state()

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
        """Emit function_run_requested for the slot's bound function."""
        entry = self._entry_for_slot(slot)
        if entry is None:
            return
        self.function_run_requested.emit(entry["start_node_id"])

    def is_slot_enabled(self, slot: int) -> bool:
        """Effective enabled state for a slot, per the spec rules."""
        return self._compute_enabled(slot)

    def set_running(self, node_id_or_none: str | None) -> None:
        """Mark a node as the active run (or clear) and re-apply visuals."""
        self._running_node_id = node_id_or_none
        self._apply_enable_state()

    def update_state(self, state_name: str) -> None:
        """Update the core state name and re-apply enable-state."""
        self._state = state_name
        self._apply_enable_state()

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

    # --- Helpers ---

    def _entry_for_slot(self, slot: int) -> dict | None:
        for entry in self._core.session.manual_controls:
            if entry["slot"] == slot:
                return entry
        return None

    def _make_slot_button(self, slot: int, entry: dict, is_missing: bool) -> QPushButton:
        """Create a bound slot button with click + press-and-hold wiring."""
        btn = QPushButton(entry.get("label", ""))
        btn.setMinimumHeight(96)
        btn.setProperty("manualTile", "missing" if is_missing else "slot")
        if is_missing:
            btn.setStyleSheet(f"color: {colors.TEXT_MUTED};")

        btn.clicked.connect(lambda _checked=False, s=slot: self._on_clicked(s))

        # Press-and-hold timer opens the per-slot context menu.
        timer = QTimer(btn)
        timer.setSingleShot(True)
        timer.setInterval(_LONG_PRESS_MS)
        timer.timeout.connect(lambda s=slot: self._show_slot_menu(s))
        self._press_timers[slot] = timer

        btn.pressed.connect(lambda s=slot: self._press_timers[s].start())
        btn.released.connect(lambda s=slot: self._press_timers[s].stop())

        return btn

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
            item.setSizeHint(item.sizeHint().expandedTo(item.sizeHint().__class__(0, 56)))
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
