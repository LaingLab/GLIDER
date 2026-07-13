import pytest

from glider.gui.dashboard.layout import (
    QUADRANTS,
    apply_drag_swap,
    apply_pick,
    default_layout,
)


def test_default_layout_assigns_four_distinct_panels():
    layout = default_layout()
    assigned = [layout.assignment[q] for q in QUADRANTS]
    assert assigned == ["run_control", "device_states", "camera", "experiment_info"]
    assert len(set(assigned)) == 4


def test_benched_panel_is_the_unassigned_one():
    layout = default_layout()
    assert layout.benched_panel() == "manual_controls"


def test_pick_benched_panel_into_quadrant_benches_previous_occupant():
    layout = default_layout()
    new = apply_pick(layout, "top_left", "manual_controls")
    assert new.assignment["top_left"] == "manual_controls"
    assert new.benched_panel() == "run_control"
    assert len(set(new.assignment.values())) == 4


def test_pick_panel_already_shown_elsewhere_swaps_the_two_quadrants():
    layout = default_layout()
    new = apply_pick(layout, "top_left", "device_states")
    assert new.assignment["top_left"] == "device_states"
    assert new.assignment["top_right"] == "run_control"


def test_pick_same_panel_into_its_own_quadrant_is_noop():
    layout = default_layout()
    new = apply_pick(layout, "top_left", "run_control")
    assert new.assignment == layout.assignment


def test_drag_swap_exchanges_two_quadrants():
    layout = default_layout()
    new = apply_drag_swap(layout, "top_left", "bottom_right")
    assert new.assignment["top_left"] == "experiment_info"
    assert new.assignment["bottom_right"] == "run_control"


def test_drag_swap_same_quadrant_is_noop():
    layout = default_layout()
    new = apply_drag_swap(layout, "top_left", "top_left")
    assert new.assignment == layout.assignment


def test_apply_functions_do_not_mutate_input():
    layout = default_layout()
    before = dict(layout.assignment)
    apply_pick(layout, "top_left", "manual_controls")
    apply_drag_swap(layout, "top_left", "bottom_right")
    assert layout.assignment == before


def test_pick_unknown_panel_key_raises():
    layout = default_layout()
    with pytest.raises(ValueError):
        apply_pick(layout, "top_left", "nonsense")


def test_pick_unknown_quadrant_raises():
    layout = default_layout()
    with pytest.raises(ValueError):
        apply_pick(layout, "nonsense", "run_control")


def test_drag_swap_unknown_quadrant_raises():
    layout = default_layout()
    with pytest.raises(ValueError):
        apply_drag_swap(layout, "top_left", "nonsense")
