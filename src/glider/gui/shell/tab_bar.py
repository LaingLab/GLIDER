"""The top-level navigation: Dashboard | Experiment | Run, centred.

GLIDER used to have two surfaces and a menu item that toggled between them
(``View ▸ Dashboard``), which worked while there were two. There are now three,
and a hidden toggle cannot express three -- nor tell you which one you are on.
This is that toggle promoted to something you can see.

**It is deliberately ignorant.** The bar knows three string keys and nothing
else: not the stack, not the pages, not ``MainWindow``. It emits
:attr:`ShellTabBar.tab_selected` when a button is pressed, and
:meth:`ShellTabBar.set_current` moves the highlight without emitting. Those two
halves are what let the owner wire it in both directions -- every existing
programmatic switch (``switch_to_builder``, the Pi layout action, the menu
toggle) moves the highlight through ``set_current`` without the bar ever
calling back into the code that just moved it.

**Centred means centred on the window, not on the buttons.** A stretch of equal
weight either side, and nothing else in the row. The temptation is to hang a
status item off one end, and the moment anything does, the group stops being
centred and starts drifting by the width of whatever was added -- which reads
as a rendering bug, because it is not obvious there is anything over there.
Status belongs on :class:`~glider.gui.shell.status_strip.StatusStrip`, one row
below, which is built to carry it.

**No colour is set from Python here**, following the rest of ``shell/``: every
part carries an ``objectName`` and ``desktop.qss`` owns the appearance.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QToolButton,
    QWidget,
)

__all__ = ["TAB_BAR_HEIGHT", "TAB_KEYS", "TAB_LABELS", "ShellTabBar"]

#: Row height in pixels -- a *minimum*, not a fixed size. Deliberately a little
#: under :data:`~glider.gui.shell.status_strip.STRIP_HEIGHT` so that on the
#: Dashboard, where both rows are visible, the pair reads as one chrome block
#: with the navigation on top rather than as two competing bars.
#:
#: A minimum rather than ``setFixedHeight`` because the touch sheet gives these
#: buttons finger-sized padding, and a fixed 36px row clips them on the Pi
#: instead of growing. The stylesheet owns the tab size; this owns the floor.
TAB_BAR_HEIGHT = 36

#: The tabs, in order. These strings are the API: the owner maps them to pages.
TAB_KEYS: tuple[str, ...] = ("dashboard", "experiment", "run")

#: What each key is called on screen.
TAB_LABELS: dict[str, str] = {
    "dashboard": "Dashboard",
    "experiment": "Experiment",
    "run": "Run",
}


class ShellTabBar(QFrame):
    """A centred, exclusive row of three tabs.

    Args:
        parent: Standard Qt parent.

    Signals:
        tab_selected: Emitted with a key from :data:`TAB_KEYS` when the *user*
            presses a tab. Never emitted by :meth:`set_current`.
    """

    tab_selected = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("shellTabBar")
        self.setMinimumHeight(TAB_BAR_HEIGHT)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(4)
        row.addStretch(1)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QToolButton] = {}

        for key in TAB_KEYS:
            button = QToolButton(self)
            button.setObjectName("shellTab")
            button.setText(TAB_LABELS[key])
            button.setCheckable(True)
            button.setAutoRaise(True)
            # Qt gives a QToolButton the arrow cursor even when it behaves as a
            # navigation control; the pointing hand is what says "this moves
            # you" before you have learned the bar.
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked, k=key: self._on_clicked(k))
            self._group.addButton(button)
            row.addWidget(button)
            self._buttons[key] = button

        row.addStretch(1)
        self._buttons[TAB_KEYS[0]].setChecked(True)
        self._current = TAB_KEYS[0]

    # ------------------------------------------------------------- selection

    def _on_clicked(self, key: str) -> None:
        """A user press. Re-pressing the current tab is swallowed.

        Qt would otherwise deliver ``clicked`` for a tab that is already
        checked, and the owner would run a page switch to the page it is on --
        which, on the Dashboard, means the camera panel is torn out and put
        back for nothing.
        """
        if key == self._current:
            return
        self._current = key
        self.tab_selected.emit(key)

    def current(self) -> str:
        """The highlighted tab's key."""
        return self._current

    def set_current(self, key: str) -> None:
        """Move the highlight without emitting :attr:`tab_selected`.

        This is the return path for switches that did not start here -- the
        View menu, the Pi touchscreen layout action, a run starting. Unknown
        keys are ignored rather than raising: a caller that has drifted out of
        step should leave the bar alone, not take the window down.
        """
        button = self._buttons.get(key)
        if button is None:
            return
        self._current = key
        was_blocked = button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(was_blocked)

    def buttons(self) -> dict[str, QToolButton]:
        """The tab buttons by key, for tests and for the command palette."""
        return dict(self._buttons)
