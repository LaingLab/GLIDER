"""Per-action value semantics for hardware devices.

A device action that carries a value — a PWM level, a pump rate, a servo angle —
declares what that value *means* through an :class:`ActionValueSpec`: its kind,
range, step, and unit. This is the single authority every layer reads (node
property editors, the generated runner controls, and write-time clamping), so no
layer re-hardcodes an 8-bit/10-bit range.

See ``docs/features/device-value-semantics/`` for the full design. Notably the
``decimal`` (fractional) kind is intentionally *not* supported yet — no device
commands fractional values today; add it when the first one appears.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Value kinds.
KIND_SWITCH = "switch"  # on/off; range is always 0..1
KIND_WHOLE = "whole"  # a bounded whole number
VALUE_KINDS = frozenset({KIND_SWITCH, KIND_WHOLE})


@dataclass(frozen=True)
class ActionValueSpec:
    """What an action's value means: kind, bounds, step, and display strings.

    Immutable so it can be shared freely between the device, its nodes, and the
    runner without any of them mutating a "single authority".
    """

    kind: str
    min: int
    max: int
    step: int = 1
    unit: str = ""
    label: str = ""

    def validate(self) -> list[str]:
        """Return human-readable problems with this spec; empty means valid.

        This is the save-time gate: a definition whose declared value block
        produces a non-empty list is rejected before the device registers.
        """
        errors: list[str] = []
        if self.kind not in VALUE_KINDS:
            errors.append(
                f"unknown value kind {self.kind!r} (expected one of {sorted(VALUE_KINDS)})"
            )
        # bool is an int subclass; reject it so True/False can't pose as bounds.
        for name in ("min", "max", "step"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool):
                errors.append(f"{name} must be a whole number (got {v!r})")
        if errors:
            # Range comparisons below assume usable integers.
            return errors

        if self.min >= self.max:
            errors.append(f"min ({self.min}) must be less than max ({self.max})")
        if self.step <= 0:
            errors.append(f"step ({self.step}) must be positive")
        elif self.step > (self.max - self.min):
            errors.append(f"step ({self.step}) cannot exceed the range ({self.max - self.min})")
        if self.kind == KIND_SWITCH and (self.min, self.max) != (0, 1):
            errors.append(f"switch kind must have range 0..1 (got {self.min}..{self.max})")
        return errors

    def contains(self, value: object) -> bool:
        """True if ``value`` is a finite number within [min, max].

        Used to mark out-of-range *reads* (which pass through unclamped) — a
        non-finite or non-numeric reading is treated as out of range.
        """
        try:
            v = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        if not math.isfinite(v):
            return False
        return self.min <= v <= self.max


def clamp_to_spec(value: object, spec: ActionValueSpec) -> tuple[int, bool]:
    """Clamp ``value`` into ``spec``'s range; return ``(clamped_int, did_clamp)``.

    Non-finite / non-numeric values (NaN, ±inf, strings, None) are rejected with
    :class:`ValueError` **before** any range handling — so they can never fall
    through to ``int()`` (which crashes on NaN) or to a bitmask (which would
    alias an over-range value to an unrelated number, e.g. ``70000 & 0xFFFF``).

    ``did_clamp`` reports only range clamping, not the whole-number truncation of
    a fractional input.
    """
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ValueError(f"non-numeric value for {spec.label or spec.kind}: {value!r}") from e
    if not math.isfinite(v):
        raise ValueError(f"non-finite value for {spec.label or spec.kind}: {value!r}")
    raw = int(v)
    clamped = max(spec.min, min(spec.max, raw))
    return clamped, clamped != raw
