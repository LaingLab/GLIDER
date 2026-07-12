"""Unit tests for per-action value semantics (Slice 1 foundation).

Covers ActionValueSpec.validate / .contains and clamp_to_spec, including the two
reproducible-today defects this feature fixes: NaN crashing the write path and
over-range values wrapping to an unrelated number instead of clamping.
"""

from __future__ import annotations

import math

import pytest

from glider.hal.value_spec import (
    KIND_SWITCH,
    KIND_WHOLE,
    ActionValueSpec,
    clamp_to_spec,
)

# --- validate() --------------------------------------------------------------


def test_valid_whole_spec_has_no_errors():
    assert ActionValueSpec(KIND_WHOLE, 0, 4095, step=1, unit="mL/min").validate() == []


def test_valid_switch_spec_has_no_errors():
    assert ActionValueSpec(KIND_SWITCH, 0, 1).validate() == []


@pytest.mark.parametrize(
    "spec, needle",
    [
        (ActionValueSpec("bogus", 0, 10), "unknown value kind"),
        (ActionValueSpec(KIND_WHOLE, 10, 10), "must be less than"),  # min == max
        (ActionValueSpec(KIND_WHOLE, 10, 5), "must be less than"),  # min > max
        (ActionValueSpec(KIND_WHOLE, 0, 10, step=0), "must be positive"),
        (ActionValueSpec(KIND_WHOLE, 0, 10, step=-2), "must be positive"),
        (ActionValueSpec(KIND_WHOLE, 0, 10, step=20), "cannot exceed the range"),
        (ActionValueSpec(KIND_SWITCH, 0, 255), "switch kind must have range 0..1"),
    ],
)
def test_invalid_specs_report_the_problem(spec, needle):
    errors = spec.validate()
    assert errors, f"expected an error for {spec}"
    assert any(needle in e for e in errors), errors


def test_bool_bounds_are_rejected_not_coerced():
    # True/False are ints in Python; they must not pose as 0/1 bounds silently.
    errors = ActionValueSpec(KIND_WHOLE, False, True).validate()  # type: ignore[arg-type]
    assert any("whole number" in e for e in errors)


# --- contains() --------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected", [(0, True), (2048, True), (4095, True), (-1, False), (4096, False)]
)
def test_contains_range(value, expected):
    assert ActionValueSpec(KIND_WHOLE, 0, 4095).contains(value) is expected


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "nope", None])
def test_contains_rejects_non_finite_and_non_numeric(value):
    assert ActionValueSpec(KIND_WHOLE, 0, 4095).contains(value) is False


# --- clamp_to_spec() ---------------------------------------------------------


def test_in_range_value_passes_through_unclamped():
    spec = ActionValueSpec(KIND_WHOLE, 0, 255)
    assert clamp_to_spec(200, spec) == (200, False)


def test_over_and_under_range_values_clamp():
    spec = ActionValueSpec(KIND_WHOLE, 0, 255)
    assert clamp_to_spec(300, spec) == (255, True)
    assert clamp_to_spec(-5, spec) == (0, True)


def test_no_wrap_on_massive_overflow():
    # The bug this replaces: `70000 & 0xFFFF == 4464` silently commanded an
    # unrelated value. Clamping must land on the ceiling instead.
    spec = ActionValueSpec(KIND_WHOLE, 0, 0xFFFF)
    clamped, did = clamp_to_spec(70000, spec)
    assert (clamped, did) == (0xFFFF, True)
    assert clamped != (70000 & 0xFFFF)


def test_fractional_input_truncates_without_flagging_a_clamp():
    spec = ActionValueSpec(KIND_WHOLE, 0, 255)
    # 3.9 -> 3 is truncation, not range clamping.
    assert clamp_to_spec(3.9, spec) == (3, False)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected_before_range_handling(value):
    spec = ActionValueSpec(KIND_WHOLE, 0, 255)
    with pytest.raises(ValueError, match="non-finite"):
        clamp_to_spec(value, spec)


@pytest.mark.parametrize("value", ["nope", None, object()])
def test_non_numeric_values_are_rejected(value):
    spec = ActionValueSpec(KIND_WHOLE, 0, 255)
    with pytest.raises(ValueError, match="non-numeric"):
        clamp_to_spec(value, spec)
