"""The Experiment tab: everything about *this* experiment, on one surface.

Experiment setup used to live in four modal dialogs reached from a menu --
Experiment Settings, Zone Configuration, Lab Setup -- which is why the common
complaint about GLIDER was that nobody could find these fields. A modal you have
to know the name of is not discoverable, and three of them cannot be open at
once, so "check the subject list against the zones" meant closing one to look at
the other.

This page is those editors, embedded. The dialogs themselves are unchanged and
still work as dialogs for every other caller; each grew a method that drops its
window frame and its close buttons -- ``embed()`` on the zone and vocabulary
editors, ``detach_sections()`` on the experiment dialog, which hands back its
two group boxes as separate pages. Reusing them rather than rewriting them is
deliberate: the zone editor alone is 800 lines of drawing interaction debugged
against real camera frames, and a second copy would be a second set of those
bugs.

**Sections, not one long scroll.** The editors are 500-700px tall each and the
zone editor wants a live camera preview beside its table; stacked, the page
would be several screens deep and the preview would routinely be off-screen
while you drew into it. A rail on the left switches between them, each entry
carrying a glyph painted by :class:`_SectionGlyphEngine` so the four are
distinguishable at a glance rather than by reading four labels.

**Sections are built on first visit, and dropped when the session changes.**
Two reasons, and the second is the load-bearing one:

* Constructing the zone editor grabs a camera frame. Doing that at window
  startup, for a tab the user may never open, costs a camera round-trip on
  every launch.
* The editors bind to session-owned objects at construction --
  :class:`~glider.gui.dialogs.experiment_dialog.ExperimentDialog` to the
  session, the zone editor to the window's
  :class:`~glider.vision.zones.ZoneConfiguration`. *New Experiment* replaces
  both. A section built against the previous experiment would keep editing an
  object nothing reads any more: edits that appear to work and are silently
  discarded. :meth:`ExperimentPage.reset` is what the window calls to prevent
  that, and it must be called on every session change.

**No colour is set from Python here**: every part carries an ``objectName``
and ``desktop.qss`` owns the appearance. The rail glyphs are the exception that
proves it, and they live in
:mod:`~glider.gui.widgets.pastel_glyphs` rather than here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.widgets.pastel_glyphs import glyph_icon

logger = logging.getLogger(__name__)

__all__ = ["SECTION_KEYS", "SECTION_LABELS", "ExperimentPage"]

#: The sections, in rail order. These strings are the API: the window supplies
#: one builder per key.
#:
#: Metadata and Mice were one section, and splitting them is what the rail is
#: for: they are edited at different times by different people -- the protocol
#: and experimenter once when the experiment is designed, the animal list again
#: before every cohort -- and stacking them meant scrolling past seven fields
#: you were not there to change to reach the table you were.
SECTION_KEYS: tuple[str, ...] = ("metadata", "mice", "zones", "vocabulary", "plugins")

#: What each section is called on the rail.
SECTION_LABELS: dict[str, str] = {
    "metadata": "Metadata",
    "mice": "Mice",
    "zones": "Zones",
    "vocabulary": "Lab Vocabulary",
    "plugins": "Plugins",
}

#: Rail icon edge length in pixels. 20 rather than a more conventional 16
#: because these are illustrations, not symbols: the mouse has an ear, an eye
#: and whiskers, and below about 18px those stop resolving and it degrades into
#: a pink blob. The rail buttons are 36px tall, so the extra 4px costs nothing.
ICON_PX = 20

#: Width of the section rail in pixels. Wide enough for the longest label above
#: at the shipped font, and fixed so the editors beside it do not reflow when
#: the selection moves.
RAIL_WIDTH = 168


class ExperimentPage(QWidget):
    """A section rail beside a stack of lazily-built editors.

    Args:
        builders: One zero-argument callable per key in :data:`SECTION_KEYS`,
            each returning the widget for that section. Called at most once per
            section per session; a builder that raises costs its own section a
            placeholder and leaves the rest of the page working.
        parent: Standard Qt parent.
    """

    def __init__(
        self,
        builders: dict[str, Callable[[], QWidget]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("experimentPage")
        self._builders = dict(builders)
        self._built: dict[str, QWidget] = {}
        self._buttons: dict[str, QToolButton] = {}

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        row.addWidget(self._build_rail())

        self._stack = QStackedWidget(self)
        self._stack.setObjectName("experimentSections")
        row.addWidget(self._stack, 1)

        # Index 0 is a permanent placeholder the stack falls back to while a
        # section is being built (and if one fails). Without it, the first
        # show()  before any section exists would leave the stack empty and
        # the page would paint as a hole beside the rail.
        self._placeholder = QLabel("")
        self._placeholder.setObjectName("experimentPlaceholder")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._stack.addWidget(self._placeholder)

        self._current = SECTION_KEYS[0]
        self._buttons[self._current].setChecked(True)

    # ------------------------------------------------------------------ build

    def _build_rail(self) -> QWidget:
        rail = QFrame(self)
        rail.setObjectName("experimentRail")
        rail.setFixedWidth(RAIL_WIDTH)

        column = QVBoxLayout(rail)
        column.setContentsMargins(8, 12, 8, 12)
        column.setSpacing(2)

        heading = QLabel("Experiment")
        heading.setObjectName("experimentRailHeading")
        column.addWidget(heading)
        column.addSpacing(6)

        for key in SECTION_KEYS:
            button = QToolButton(rail)
            button.setObjectName("experimentRailItem")
            button.setText(SECTION_LABELS[key])
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            # Full rail width, so the hit target is the row rather than the
            # words. A QToolButton sizes itself to its label by default, which
            # made every entry a different width and left most of the rail
            # looking clickable but inert -- you aim at "Mice" and hit nothing,
            # because the button stops where the text does.
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setIcon(glyph_icon(button, key, ICON_PX))
            button.setIconSize(QSize(ICON_PX, ICON_PX))
            button.clicked.connect(lambda _checked, k=key: self.show_section(k))
            column.addWidget(button)
            self._buttons[key] = button

        column.addStretch(1)
        return rail

    # --------------------------------------------------------------- sections

    def show_section(self, key: str) -> None:
        """Bring ``key`` to the front, building it if this is its first visit.

        Unknown keys are ignored rather than raising: this is reached from a
        button press, and a mis-wired rail should not take the window down.
        """
        if key not in self._builders:
            return
        self._current = key
        button = self._buttons.get(key)
        if button is not None and not button.isChecked():
            was_blocked = button.blockSignals(True)
            button.setChecked(True)
            button.blockSignals(was_blocked)

        widget = self._built.get(key)
        if widget is None:
            widget = self._build_section(key)
        if widget is None:
            self._stack.setCurrentWidget(self._placeholder)
            return
        self._stack.setCurrentWidget(widget)

    def _build_section(self, key: str) -> QWidget | None:
        """Run the builder for ``key`` once, and remember the result.

        A builder that raises is caught and reported into the placeholder
        rather than propagated. These builders construct camera-backed editors
        against whatever hardware is plugged in at the time; a rig with a
        camera that has gone away should lose the Zones section, not the
        Experiment tab and everything on it.
        """
        try:
            widget = self._builders[key]()
        except Exception:
            logger.exception("Experiment section %r could not be built", key)
            self._placeholder.setText(
                f"The {SECTION_LABELS[key].replace('&&', '&')} section could not be opened.\n"
                "See the log for details."
            )
            return None
        self._stack.addWidget(widget)
        self._built[key] = widget
        return widget

    def reset(self) -> None:
        """Drop every built section so the next visit rebuilds against the
        current session.

        Called by the window on New and Open. The widgets are removed from the
        stack and ``deleteLater``-ed rather than merely forgotten, because they
        hold camera frames and Qt signal connections to session objects that
        are about to be replaced.
        """
        for key, widget in list(self._built.items()):
            self._stack.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
            del self._built[key]
        self._stack.setCurrentWidget(self._placeholder)
        self._placeholder.setText("")

    def refresh(self) -> None:
        """Re-show the current section, building it if it is not up yet.

        This is what the window calls when the Experiment tab is entered, so a
        tab that has never been visited builds its first section on arrival
        rather than showing the empty placeholder until something is clicked.
        """
        self.show_section(self._current)

    # ------------------------------------------------------------------ probes

    def current_section(self) -> str:
        """The selected section's key."""
        return self._current

    def built_sections(self) -> dict[str, QWidget]:
        """The sections built so far, by key. For tests."""
        return dict(self._built)

    def rail_buttons(self) -> dict[str, QToolButton]:
        """The rail buttons by key. For tests."""
        return dict(self._buttons)
