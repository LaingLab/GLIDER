"""Corner radii for the GLIDER theme.

Three steps, and a fourth case that is not a step at all.

The app had fifteen distinct radii across 145 declarations - 2, 3, 5, 7, 9, 13
and so on - not because anything needed them but because each was chosen next
to whatever was on screen at the time. Corners that nearly match read as
sloppier than corners that plainly differ, so the fix is a scale small enough
that picking from it is easier than inventing a number.

    SMALL   4px   controls: buttons, inputs, combo boxes, menu items,
                  scrollbars, indicators, chips, tabs
    MEDIUM  8px   containers: cards, panels, dialogs, menus, popovers
    LARGE  12px   large surfaces: hero controls, dashboards, touch panels

**Circles and pills are geometric, not stylistic.** A radio indicator is 18px
across with a 9px radius because that is what makes it round; a status dot is
8px with a 4px radius for the same reason. Snapping those to the scale would
turn circles into squircles, so they keep their own value and are marked
``/* geometric */`` in the stylesheets, which is also what lets the scale test
tell a deliberate exception from a fresh invention.

Nothing here is a Qt dependency, so it is importable from a headless test.
"""

from __future__ import annotations

__all__ = ["LARGE", "MEDIUM", "NONE", "SCALE", "SMALL", "pill"]

#: Deliberately square. A flush edge where a control meets its container -
#: distinct from "nobody chose", which is what an absent radius means.
NONE = 0

#: Controls. Anything the pointer acts on directly.
SMALL = 4

#: Containers. Anything that holds controls.
MEDIUM = 8

#: Large surfaces: hero controls, dashboard cards, touch-mode panels, where a
#: container radius would look tight against the size of the thing.
LARGE = 12

#: Every value a stylistic radius may take. The scale test reads this.
SCALE = (NONE, SMALL, MEDIUM, LARGE)


def pill(height: int) -> int:
    """Radius that makes something *height* pixels tall fully round.

    For circles and pills, where the radius follows from the size rather than
    from the scale. Use it instead of writing the halved number, so the two
    cannot drift apart when the size changes - which is how a status dot ends
    up an oval.
    """
    if height < 0:
        raise ValueError(f"height must not be negative, got {height}")
    return height // 2
