"""How a device's link state is worded and coloured.

One module because three surfaces render the same five states -- the hardware
tree, the Device Control panel, and the status strip -- and each of them
inventing its own word is how the status bar came to read "Connected" beside a
red dot. The strip's own board-level mapping (DEVICE_STATE_BY_BOARD_STATE in
main_window) is the same idea for boards; this is its device sibling.
"""

from __future__ import annotations

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


def link_is_usable(state: object) -> bool:
    """Whether a command sent right now has a link to travel over.

    Only CONNECTED. RECONNECTING is honest about trying, but a button pressed
    during one fails, and offering a press that is certain to fail is worse
    than grey.
    """
    return state is ConnectionState.CONNECTED
