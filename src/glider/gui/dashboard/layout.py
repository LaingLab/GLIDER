"""Pure, UI-free layout model and swap logic for the 2x2 dashboard.

`DashboardLayout` holds an immutable-by-convention assignment of quadrant id ->
panel key plus optional splitter sizes. `apply_pick` and `apply_drag_swap`
return NEW layouts (never mutate) enforcing the invariant that each panel
occupies at most one quadrant. The benched panel is derived, never stored.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from glider.gui.dashboard.panel_registry import PANEL_KEYS

QUADRANTS: tuple[str, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")

_DEFAULT_ASSIGNMENT = {
    "top_left": "run_control",
    "top_right": "device_states",
    "bottom_left": "camera",
    "bottom_right": "experiment_info",
}


@dataclass(frozen=True)
class DashboardLayout:
    """Which panel is shown in each quadrant, plus splitter geometry.

    `assignment` maps every quadrant id to a panel key (each key unique).
    `outer_sizes` is the horizontal splitter's [left, right] pixel sizes;
    `left_sizes` / `right_sizes` are the two vertical splitters' [top, bottom].
    Sizes are advisory; empty means "let Qt distribute evenly".
    """

    assignment: dict[str, str]
    outer_sizes: tuple[int, ...] = ()
    left_sizes: tuple[int, ...] = ()
    right_sizes: tuple[int, ...] = ()

    def benched_panel(self) -> str:
        shown = set(self.assignment.values())
        for key in PANEL_KEYS:
            if key not in shown:
                return key
        raise AssertionError("all panels assigned — impossible with 4 slots / 5 panels")

    def with_assignment(self, assignment: dict[str, str]) -> DashboardLayout:
        return replace(self, assignment=dict(assignment))


def default_layout() -> DashboardLayout:
    return DashboardLayout(assignment=dict(_DEFAULT_ASSIGNMENT))


def _quadrant_showing(layout: DashboardLayout, panel_key: str) -> str | None:
    for quad, key in layout.assignment.items():
        if key == panel_key:
            return quad
    return None


def apply_pick(layout: DashboardLayout, quadrant: str, panel_key: str) -> DashboardLayout:
    """Show `panel_key` in `quadrant`.

    If `panel_key` is already shown in another quadrant, swap the two quadrants.
    Otherwise (it was benched) move it in and bench whatever `quadrant` held.
    """
    if panel_key not in PANEL_KEYS:
        raise ValueError(f"unknown panel key: {panel_key!r}")
    if quadrant not in QUADRANTS:
        raise ValueError(f"unknown quadrant: {quadrant!r}")

    new_assignment = dict(layout.assignment)
    current_here = new_assignment[quadrant]
    if current_here == panel_key:
        return layout  # no-op

    other = _quadrant_showing(layout, panel_key)
    if other is not None:
        new_assignment[quadrant], new_assignment[other] = panel_key, current_here
    else:
        new_assignment[quadrant] = panel_key
    return layout.with_assignment(new_assignment)


def apply_drag_swap(layout: DashboardLayout, source: str, target: str) -> DashboardLayout:
    """Swap the panels shown in `source` and `target` quadrants."""
    for q in (source, target):
        if q not in QUADRANTS:
            raise ValueError(f"unknown quadrant: {q!r}")
    if source == target:
        return layout
    new_assignment = dict(layout.assignment)
    new_assignment[source], new_assignment[target] = (
        new_assignment[target],
        new_assignment[source],
    )
    return layout.with_assignment(new_assignment)
