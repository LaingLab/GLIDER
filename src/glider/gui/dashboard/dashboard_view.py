"""The 2x2 quadrant dashboard: hosts four panels + one benched, applies
pick/drag swaps via the pure layout model, and persists the arrangement.

Panels are constructed by the caller (they need `core`/main-window slots) and
handed in as a {panel_key: QWidget} map. DashboardView owns only placement.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from glider.gui.dashboard.layout import (
    QUADRANTS,
    DashboardLayout,
    apply_drag_swap,
    apply_pick,
    default_layout,
)
from glider.gui.dashboard.layout_store import load_layout, save_layout
from glider.gui.dashboard.panel_registry import PANEL_NAMES
from glider.gui.dashboard.quadrant_host import QuadrantHost


class DashboardView(QWidget):
    """2x2 grid of QuadrantHosts with picker/drag swap and persistence."""

    layout_changed = pyqtSignal()

    def __init__(
        self,
        panels: dict[str, QWidget],
        save_path: Path | None = None,
        banner: QWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._panels = dict(panels)
        self._save_path = save_path
        self._banner = banner
        self._hosts: dict[str, QuadrantHost] = {}
        self._bench_holder = QWidget(self)
        self._bench_holder.hide()
        self._layout = load_layout(save_path) if save_path is not None else default_layout()

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._persist)

        self._setup_ui()
        self._apply_layout(self._layout, persist=False)

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if self._banner is not None:
            outer.addWidget(self._banner)
            # Hidden until a run is live with Run Control benched (see
            # update_banner) — mirrors RunnerShell, which hides its banner at
            # construction. Without this the banner shows on first entry in
            # IDLE, duplicating the Run Control timer and STOP button.
            self._banner.hide()

        for q in QUADRANTS:
            host = QuadrantHost(q)
            host.panel_selected.connect(self._on_panel_selected)
            host.swap_requested.connect(self._on_swap_requested)
            self._hosts[q] = host

        self._left_split = QSplitter(Qt.Orientation.Vertical)
        self._left_split.addWidget(self._hosts["top_left"])
        self._left_split.addWidget(self._hosts["bottom_left"])

        self._right_split = QSplitter(Qt.Orientation.Vertical)
        self._right_split.addWidget(self._hosts["top_right"])
        self._right_split.addWidget(self._hosts["bottom_right"])

        self._outer_split = QSplitter(Qt.Orientation.Horizontal)
        self._outer_split.addWidget(self._left_split)
        self._outer_split.addWidget(self._right_split)
        outer.addWidget(self._outer_split, 1)

        for split in (self._outer_split, self._left_split, self._right_split):
            split.splitterMoved.connect(lambda *_: self._save_timer.start())

    # --- public API ---

    def host(self, quadrant_id: str) -> QuadrantHost:
        return self._hosts[quadrant_id]

    def panel(self, panel_key: str) -> QWidget:
        return self._panels[panel_key]

    def current_layout(self) -> DashboardLayout:
        return self._layout

    def update_state(self, state_name: str) -> None:
        for widget in self._panels.values():
            if hasattr(widget, "update_state"):
                widget.update_state(state_name)

    def set_banner_time(self, text: str) -> None:
        if self._banner is not None and hasattr(self._banner, "set_time"):
            self._banner.set_time(text)

    def update_banner(self, state_name: str, recording: bool) -> None:
        if self._banner is None:
            return
        shown = {self._hosts[q].current_panel_key for q in QUADRANTS}
        live = state_name in ("RUNNING", "PAUSED")
        visible = live and "run_control" not in shown
        self._banner.setVisible(visible)
        if visible and hasattr(self._banner, "set_state"):
            self._banner.set_state(state_name, recording=recording)

    # --- layout application ---

    def _apply_layout(self, layout: DashboardLayout, persist: bool = True) -> None:
        # Park every panel in the hidden holder first so reassignments never
        # collide (a panel briefly living in two hosts). QuadrantHost.set_panel
        # only evicts a previous panel still parented under its own body, so
        # parked panels are not clobbered back to a null parent.
        # Park all panels first (cheap here; revisit if a panel does
        # show/hide-triggered work).
        for widget in self._panels.values():
            widget.setParent(self._bench_holder)
        for q in QUADRANTS:
            key = layout.assignment[q]
            self._hosts[q].set_panel(self._panels[key], key, PANEL_NAMES[key])
        self._layout = layout
        self._restore_splitter_sizes(layout)
        if persist:
            self._persist()
        self.layout_changed.emit()

    def _on_panel_selected(self, quadrant_id: str, panel_key: str) -> None:
        self._apply_layout(apply_pick(self._layout, quadrant_id, panel_key))

    def _on_swap_requested(self, source_id: str, target_id: str) -> None:
        self._apply_layout(apply_drag_swap(self._layout, source_id, target_id))

    # --- persistence ---

    def _current_sizes(self) -> DashboardLayout:
        return replace(
            self._layout,
            outer_sizes=tuple(self._outer_split.sizes()),
            left_sizes=tuple(self._left_split.sizes()),
            right_sizes=tuple(self._right_split.sizes()),
        )

    def _restore_splitter_sizes(self, layout: DashboardLayout) -> None:
        if layout.outer_sizes:
            self._outer_split.setSizes(list(layout.outer_sizes))
        if layout.left_sizes:
            self._left_split.setSizes(list(layout.left_sizes))
        if layout.right_sizes:
            self._right_split.setSizes(list(layout.right_sizes))

    def _persist(self) -> None:
        if self._save_path is None:
            return
        self._layout = self._current_sizes()
        save_layout(self._layout, self._save_path)
