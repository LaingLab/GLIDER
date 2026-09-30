"""How a device's link state is worded and coloured.

One module because three surfaces render the same five states -- the hardware
tree, the Device Control panel, and the status strip -- and each of them
inventing its own word is how the status bar came to read "Connected" beside a
red dot. The strip's own board-level mapping (DEVICE_STATE_BY_BOARD_STATE in
main_window) is the same idea for boards; this is its device sibling.
"""

from __future__ import annotations

from glider.gui.styles import colors
from glider.hal.base_board import ConnectionState

#: What each state is called in a status line.
_TEXT = {
    ConnectionState.CONNECTED: "Ready",
    ConnectionState.CONNECTING: "Connecting…",
    ConnectionState.RECONNECTING: "Reconnecting…",
    ConnectionState.DISCONNECTED: "Disconnected",
    ConnectionState.ERROR: "Error",
}

#: What each state is on the status strip's four-colour scale.
_STRIP = {
    ConnectionState.CONNECTED: "ok",
    ConnectionState.CONNECTING: "warn",
    ConnectionState.RECONNECTING: "warn",
    ConnectionState.DISCONNECTED: "error",
    ConnectionState.ERROR: "error",
}


def link_status_text(state: object) -> str:
    """The word for ``state`` in a status line.

    "Ready" rather than "Connected" because that is the word the hardware tree
    already used for a device that was good to go, and the tree is where most
    people read it.
    """
    return _TEXT.get(state, "Unknown")


def link_strip_state(state: object) -> str:
    """``state`` as one of the status strip's DEVICE_STATES.

    An unrecognised state renders neutral rather than green: a state nobody
    mapped is not evidence that anything is healthy.
    """
    return _STRIP.get(state, "unknown")


#: The strip's four-colour scale, in CSS.
_STRIP_COLOR = {
    "ok": colors.SUCCESS,
    "warn": colors.WARNING,
    "error": colors.ERROR,
    "unknown": colors.TEXT_DISABLED,
}


def link_status_color(state: object) -> str:
    """The colour ``link_status_text(state)`` should be painted in.

    Routed through :func:`link_strip_state` rather than given its own table, so
    a card and the status strip two inches away cannot come to disagree about
    whether a device is healthy. That is the whole failure this module exists
    to prevent, and a green pill reading "Disconnected" would be a fresh
    instance of it.
    """
    return _STRIP_COLOR[link_strip_state(state)]


def link_is_usable(state: object) -> bool:
    """Whether a command sent right now has a link to travel over.

    Only CONNECTED. RECONNECTING is honest about trying, but a button pressed
    during one fails, and offering a press that is certain to fail is worse
    than grey.
    """
    return state is ConnectionState.CONNECTED


def _resolution_bits(device: object, attr: str, default: int) -> int:
    bits = getattr(getattr(getattr(device, "_board", None), "capabilities", None), attr, None)
    return bits if isinstance(bits, int) and bits > 0 else default


def device_state_display(device: object) -> tuple[str, str, str]:
    """``(text, background colour, font size)`` for a device's live-state pill.

    One function because the Run tab and the dashboard's Device States panel
    each carried their own copy, and both only read ``_state`` -- which PWM
    outputs (``_value``) and servos (``_angle``) do not have, so those cards
    never moved during a run.
    """
    device_type = getattr(device, "device_type", "")
    if device_type == "AnalogInput":
        value = getattr(device, "_last_value", None)
        if value is None:
            return "---", colors.BORDER, "11px"
        full_scale = (1 << _resolution_bits(device, "analog_resolution", 10)) - 1
        return f"{value}\n{value / full_scale * 5.0:.2f}V", colors.ACCENT, "11px"
    if device_type == "PWMOutput":
        value = getattr(device, "_value", None)
        if value is None:
            return "---", colors.BORDER, "14px"
        full_scale = (1 << _resolution_bits(device, "pwm_resolution", 8)) - 1
        color = colors.ACCENT if value else colors.TEXT_MUTED
        return f"{round(value / full_scale * 100)}%", color, "14px"
    if device_type == "Servo":
        angle = getattr(device, "_angle", None)
        if angle is None:
            return "---", colors.BORDER, "14px"
        return f"{angle}\u00b0", colors.ACCENT, "14px"
    state = getattr(device, "_state", None)
    if state is None:
        return "---", colors.BORDER, "14px"
    if isinstance(state, bool):
        return ("HIGH", colors.SUCCESS, "14px") if state else ("LOW", colors.TEXT_MUTED, "14px")
    return str(state)[:6], colors.ACCENT, "14px"
