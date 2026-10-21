"""Drag-to-scroll for the Pi's scrollable panels, and the undo for it.

On the touchscreen a finger drag on the content is how a list is scrolled; the
thin bars are only indicators. ``FingerScroll`` grabs a ``QScroller`` on every
scroll area under the roots it is given and remembers each one, so Switch to
Desktop can release exactly what was grabbed: a scroller left on a re-hosted
panel swallows the mouse moves that desktop drag-and-drop needs.
"""

from __future__ import annotations

from PyQt6 import sip
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QComboBox,
    QHeaderView,
    QScroller,
    QWidget,
)

_GESTURE = QScroller.ScrollerGestureType.LeftMouseButtonGesture


def _skipped(widget: QWidget) -> bool:
    """Not a panel to drag-scroll.

    A table's or tree's header is a scroll area too, but dragging on it
    resizes columns; and setting a scroll mode on a header that has not laid
    out its sections crashes on macOS. A combo box's popup list scrolls itself.
    """
    if isinstance(widget, QHeaderView):
        return True
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QComboBox):
            return True
        parent = parent.parentWidget()
    return False


class FingerScroll:
    """The scroll areas given drag-scrolling, and their item views' old modes."""

    def __init__(self) -> None:
        self._grabbed: list[tuple[QAbstractScrollArea, QAbstractItemView.ScrollMode | None]] = []

    def add(self, *roots: QWidget) -> None:
        """Grab every scroll area in ``roots`` (a root that is one included).

        Item views also scroll per pixel, so a flick glides rather than jumping
        a row at a time.
        """
        self._grabbed = [(a, m) for a, m in self._grabbed if not sip.isdeleted(a)]
        done = {id(area) for area, _ in self._grabbed}
        for root in roots:
            areas = root.findChildren(QAbstractScrollArea)
            if isinstance(root, QAbstractScrollArea):
                areas.insert(0, root)
            for area in areas:
                if id(area) in done or _skipped(area):
                    continue
                done.add(id(area))
                QScroller.grabGesture(area.viewport(), _GESTURE)
                mode = None
                if isinstance(area, QAbstractItemView):
                    mode = area.verticalScrollMode()
                    area.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
                self._grabbed.append((area, mode))

    def release(self) -> None:
        """Ungrab everything and put the item views' scroll modes back."""
        for area, mode in self._grabbed:
            if sip.isdeleted(area):
                continue
            QScroller.ungrabGesture(area.viewport())
            if mode is not None:
                area.setVerticalScrollMode(mode)
        self._grabbed.clear()
