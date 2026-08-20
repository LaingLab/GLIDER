"""One collapsible tabbed side panel.

The panel is a tab strip over a content area that collapses to a 34 px icon
rail. It knows nothing about node libraries, hardware trees or cameras -- tabs
are supplied by the caller, which is what lets both sides of the Builder be the
same class and lets this file be tested without a ``MainWindow``.

Three decisions here carry meaning rather than convenience:

* **Collapsed is a rail, not hidden.** Pure VS Code or Obsidian would hide the
  side entirely and expect the user to know a shortcut. GLIDER has no existing
  users -- for the foreseeable future *every* user is a first-time user -- so
  34 px of icons keeps the areas visible and gives back a click target that
  does not require knowing a shortcut exists. This is the deliberate deviation
  from the reference applications.

* **Expanding restores the width the user chose**, not a default. Someone who
  drags the panel to 320 px and collapses it expects 320 px back; handing them
  the default instead silently discards a layout decision.

* **Collapsing parks a tab's widget, it does not destroy it.** The body is
  hidden, never emptied, so the hosted widgets keep their parent and their
  state. The right panel will eventually host a live camera feed, and a
  reparented-then-orphaned widget is the classic Qt crash in this codebase.

**No colour is set from Python here.** Every part carries an ``objectName`` --
and, where it varies, a dynamic ``state`` property -- and ``desktop.qss`` owns
the rest. A widget-level ``setStyleSheet`` out-prioritises the application
stylesheet, so one stray call would make that widget the single thing in the
shell the theme cannot reach. Qt resolves property selectors at polish time
rather than when the property is set, so
:func:`glider.gui.styles.restyle` re-polishes anything whose ``state``
changed; without it the ``[state="collapsed"]`` rules would never fire.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWIDGETSIZE_MAX,
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import restyle

__all__ = ["DEFAULT_WIDTH", "MIN_EXPANDED_WIDTH", "RAIL_WIDTH", "SidePanel"]

#: Width of the collapsed rail, in pixels. Sized to a 30 px hit target plus its
#: 2 px margins -- comfortably clickable without becoming a second sidebar.
RAIL_WIDTH = 34

#: Width a panel opens at before anyone has dragged it.
DEFAULT_WIDTH = 260

#: The narrowest useful expanded width. Below this the tab labels collide and
#: the panel is worse than the rail it collapsed from.
MIN_EXPANDED_WIDTH = 160

#: Side of a rail button, in pixels.
_RAIL_BUTTON = 30


class SidePanel(QFrame):
    """A tab strip over a content area, collapsible to an icon rail.

    Args:
        side: ``"left"`` or ``"right"``. Decides which edge the rail sits on so
            it stays on the *outside* of the window on both sides, and is
            published as a ``side`` property for the stylesheet.
        parent: Standard Qt parent.

    Signals:
        expanded_changed: The panel expanded (``True``) or collapsed
            (``False``). Emitted only on an actual change.
        current_changed: The current tab's key. Emitted only on an actual
            change, and *not* for the implicit selection of the first tab
            added -- nothing can be connected to it at that point, and a caller
            populating tabs is not making a user-visible selection.
    """

    expanded_changed = pyqtSignal(bool)
    current_changed = pyqtSignal(str)

    def __init__(self, *, side: str = "left", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self._side = "right" if side == "right" else "left"
        self.setProperty("side", self._side)
        self.setProperty("state", "expanded")

        self._keys: list[str] = []
        self._widgets: dict[str, QWidget] = {}
        self._tab_buttons: dict[str, QToolButton] = {}
        self._rail_buttons: dict[str, QToolButton] = {}
        self._current: str = ""
        self._expanded: bool = True
        self._expanded_width: int = DEFAULT_WIDTH

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._rail = self._build_rail()
        self._body = self._build_body()

        # The rail belongs on the window's outer edge on both sides, so the
        # content area always sits against the centre surface.
        if self._side == "right":
            outer.addWidget(self._body)
            outer.addWidget(self._rail)
        else:
            outer.addWidget(self._rail)
            outer.addWidget(self._body)

        self._rail.setVisible(False)
        self.setMinimumWidth(MIN_EXPANDED_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    # ----------------------------------------------------------------- build

    def _build_rail(self) -> QFrame:
        rail = QFrame(self)
        rail.setObjectName("sidePanelRail")
        rail.setFrameShape(QFrame.Shape.NoFrame)
        rail.setFixedWidth(RAIL_WIDTH)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(4)
        layout.addStretch(1)
        self._rail_layout = layout
        return rail

    def _build_body(self) -> QFrame:
        body = QFrame(self)
        body.setObjectName("sidePanelBody")
        body.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        tabs = QFrame(body)
        tabs.setObjectName("sidePanelTabs")
        tabs.setFrameShape(QFrame.Shape.NoFrame)
        self._tab_layout = QHBoxLayout(tabs)
        self._tab_layout.setContentsMargins(6, 4, 6, 4)
        self._tab_layout.setSpacing(2)
        self._tab_layout.addStretch(1)
        layout.addWidget(tabs)

        self._stack = QStackedWidget(body)
        self._stack.setObjectName("sidePanelStack")
        layout.addWidget(self._stack, 1)
        return body

    # ------------------------------------------------------------------ tabs

    def add_tab(self, key: str, label: str, widget: QWidget, icon_text: str = "") -> None:
        """Add a tab hosting *widget*.

        Args:
            key: Stable identifier, used by :meth:`set_current` and by layout
                persistence. Must be unique within the panel.
            label: Human-readable text on the expanded tab.
            widget: The content. The panel takes it into its stack and keeps it
                alive across collapses; it is never deleted here.
            icon_text: Glyph or letter for the collapsed rail. Defaults to the
                first character of *label*, so a caller that has no icon still
                gets a distinguishable button rather than a blank one.

        Raises:
            ValueError: *key* is already in use.
        """
        if key in self._widgets:
            raise ValueError(f"duplicate tab key: {key!r}")

        tab = QToolButton(self)
        tab.setObjectName("sidePanelTab")
        tab.setText(label)
        tab.setCheckable(True)
        tab.setCursor(Qt.CursorShape.PointingHandCursor)
        tab.clicked.connect(lambda _checked=False, k=key: self.set_current(k))
        self._tab_layout.insertWidget(self._tab_layout.count() - 1, tab)

        rail_button = QToolButton(self._rail)
        rail_button.setObjectName("sidePanelRailButton")
        rail_button.setText(icon_text or label[:1].upper())
        rail_button.setToolTip(label)
        rail_button.setCheckable(True)
        rail_button.setFixedSize(_RAIL_BUTTON, _RAIL_BUTTON)
        rail_button.setCursor(Qt.CursorShape.PointingHandCursor)
        rail_button.clicked.connect(lambda _checked=False, k=key: self._open_at(k))
        self._rail_layout.insertWidget(self._rail_layout.count() - 1, rail_button)

        self._stack.addWidget(widget)
        self._keys.append(key)
        self._widgets[key] = widget
        self._tab_buttons[key] = tab
        self._rail_buttons[key] = rail_button

        if len(self._keys) == 1:
            # First tab wins by default, silently: no user made this choice.
            self._current = key
            self._stack.setCurrentWidget(widget)
        self._sync_selection()

    def keys(self) -> list[str]:
        """Tab keys, in the order they were added."""
        return list(self._keys)

    def widget_for(self, key: str) -> QWidget:
        """The widget hosted by tab *key*.

        Raises:
            KeyError: No such tab.
        """
        return self._widgets[key]

    def current_key(self) -> str:
        """The current tab's key, or ``""`` while the panel has no tabs."""
        return self._current

    def set_current(self, key: str) -> None:
        """Make tab *key* current, emitting :attr:`current_changed` on a change.

        Raises:
            KeyError: No such tab. Deliberately loud rather than silently
                ignored: the caller restoring a saved tab is the one that knows
                what to fall back to, and a panel that quietly kept showing the
                wrong tab would look like a persistence bug instead.
        """
        if key not in self._widgets:
            raise KeyError(key)
        if key == self._current:
            return
        self._current = key
        self._stack.setCurrentWidget(self._widgets[key])
        self._sync_selection()
        self.current_changed.emit(key)

    def _sync_selection(self) -> None:
        for key, button in self._tab_buttons.items():
            button.setChecked(key == self._current)
        for key, button in self._rail_buttons.items():
            button.setChecked(key == self._current)

    def _open_at(self, key: str) -> None:
        """A rail button was pressed: select that tab, then expand to it."""
        self.set_current(key)
        self.set_expanded(True)

    # ------------------------------------------------------------- collapsing

    @property
    def expanded(self) -> bool:
        """Whether the content area is showing. ``False`` means the rail."""
        return self._expanded

    def expanded_width(self) -> int:
        """The width the panel will return to when expanded.

        While expanded this is the live width, so a caller saving the layout
        gets the width the user is currently looking at rather than the one
        they had before the last collapse.
        """
        return self.width() if self._expanded else self._expanded_width

    def set_expanded(self, expanded: bool) -> None:
        """Expand or collapse, emitting :attr:`expanded_changed` on a change.

        Collapsing records the current width first, so expanding returns to it
        rather than to :data:`DEFAULT_WIDTH`.
        """
        expanded = bool(expanded)
        if expanded == self._expanded:
            return

        if not expanded:
            self._expanded_width = max(self.width(), MIN_EXPANDED_WIDTH)

        self._expanded = expanded
        # Hidden, never emptied: the hosted widgets keep their parent and their
        # state, and are shown again as they were.
        self._body.setVisible(expanded)
        self._rail.setVisible(not expanded)
        self.setProperty("state", "expanded" if expanded else "collapsed")
        restyle(self)

        if expanded:
            self.setMaximumWidth(QWIDGETSIZE_MAX)
            self.setMinimumWidth(MIN_EXPANDED_WIDTH)
            self.resize(self._expanded_width, self.height())
        else:
            self.setFixedWidth(RAIL_WIDTH)
            self.resize(RAIL_WIDTH, self.height())

        self.expanded_changed.emit(expanded)

    def toggle(self) -> None:
        """Flip between the content area and the rail."""
        self.set_expanded(not self._expanded)

    # -------------------------------------------------------------- accessors

    def rail_buttons(self) -> list[QToolButton]:
        """The collapsed rail's buttons, in tab order."""
        return [self._rail_buttons[key] for key in self._keys]

    def tab_buttons(self) -> list[QToolButton]:
        """The expanded tab strip's buttons, in tab order."""
        return [self._tab_buttons[key] for key in self._keys]
