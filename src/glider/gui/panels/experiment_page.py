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

**No colour is set from Python here except the rail glyphs**, which are
illustrations rather than controls -- see :class:`_SectionGlyphEngine`. Every
other part carries an ``objectName`` and ``desktop.qss`` owns the appearance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from PyQt6.QtCore import QPointF, QRect, QRectF, QSize, Qt
from PyQt6.QtGui import (
    QColor,
    QIcon,
    QIconEngine,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors

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
SECTION_KEYS: tuple[str, ...] = ("metadata", "mice", "zones", "vocabulary")

#: What each section is called on the rail.
SECTION_LABELS: dict[str, str] = {
    "metadata": "Metadata",
    "mice": "Mice",
    "zones": "Zones",
    "vocabulary": "Lab Vocabulary",
}

#: Rail icon edge length in pixels. Small enough to sit beside 13px text
#: without crowding it, large enough that the four glyphs stay distinguishable.
ICON_PX = 16

#: Width of the section rail in pixels. Wide enough for the longest label above
#: at the shipped font, and fixed so the editors beside it do not reflow when
#: the selection moves.
RAIL_WIDTH = 168


class _SectionGlyphEngine(QIconEngine):
    """Paints one rail icon: a small pastel illustration of what the section is.

    Vector, and painted fresh per request, for the same reason
    :class:`~glider.gui.shell.status_strip._SidebarGlyphEngine` is: these are
    drawn at 16px, and a fixed-resolution pixmap stretched to a fractional
    device pixel ratio smears. One class rather than four SVG files for the
    same reason.

    **These are the one place in the GUI where Python names colours**, against
    the rule the rest of this package follows -- and the exception is the
    point. Everything else on screen is a control, and a control's colour is a
    *state*, which is why ``desktop.qss`` owns it. These are pictures, and a
    picture's colour is its identity: you find Mice by its pink mouse rather
    than by reading four labels, which is the whole reason a rail of icons
    beats a rail of words. An icon that went accent on selection, the way the
    label beside it does, would throw that away for the one entry you had
    already found. The pairs live in :mod:`~glider.gui.styles.colors`, so a
    re-theme still has one place to go.
    """

    #: Fill and detail colour per section key.
    _PALETTE: dict[str, tuple[str, str]] = {
        "metadata": (colors.PASTEL_LILAC, colors.PASTEL_LILAC_DEEP),
        "mice": (colors.PASTEL_ROSE, colors.PASTEL_ROSE_DEEP),
        "zones": (colors.PASTEL_MINT, colors.PASTEL_MINT_DEEP),
        "vocabulary": (colors.PASTEL_PEACH, colors.PASTEL_PEACH_DEEP),
    }

    def __init__(self, widget: QWidget, key: str) -> None:
        super().__init__()
        self._widget = widget
        self._key = key

    def clone(self) -> _SectionGlyphEngine:
        return _SectionGlyphEngine(self._widget, self._key)

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        # Same override, for the same reason, as the sidebar glyph: this Qt
        # build's default QIconEngine.pixmap() hands back an opaque block
        # rather than painting onto a transparent buffer, so every icon comes
        # out as one solid swatch.
        pixmap = QPixmap(size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(0, 0, size.width(), size.height()), mode, state)
        painter.end()
        return pixmap

    def paint(
        self, painter: QPainter | None, rect: QRect, mode: QIcon.Mode, state: QIcon.State
    ) -> None:
        if painter is None:
            return
        pair = self._PALETTE.get(self._key)
        if pair is None:
            return
        fill, detail = QColor(pair[0]), QColor(pair[1])

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        box = QRectF(rect).adjusted(1.0, 1.0, -1.0, -1.0)
        # Line weight is a fraction of the glyph, not a constant: a 1px rule
        # looks spindly on a Retina panel and heavy on a 1x one.
        stroke = max(box.height() * 0.11, 1.0)

        {
            "metadata": self._draw_metadata,
            "mice": self._draw_mouse,
            "zones": self._draw_zones,
            "vocabulary": self._draw_vocabulary,
        }[self._key](painter, box, fill, detail, stroke)
        painter.restore()

    @staticmethod
    def _line_pen(color: QColor, width: float) -> QPen:
        pen = QPen(color)
        pen.setWidthF(width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        return pen

    # -- the four glyphs. Each fills a soft shape, then draws its detail. --

    @classmethod
    def _draw_metadata(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A filled card with three lines written on it."""
        w, h = box.width(), box.height()
        card = QRectF(box.left() + w * 0.10, box.top(), w * 0.80, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(card, w * 0.14, w * 0.14)

        p.setPen(cls._line_pen(detail, stroke))
        for i, frac in enumerate((0.30, 0.52, 0.74)):
            y = card.top() + h * frac
            # A short last line, the way a paragraph ends -- three equal
            # strokes read as a barcode rather than as writing.
            right = card.right() - w * (0.40 if i == 2 else 0.18)
            p.drawLine(QPointF(card.left() + w * 0.18, y), QPointF(right, y))

    @classmethod
    def _draw_mouse(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A mouse in profile: filled body and ear, a curling tail."""
        w, h = box.width(), box.height()

        # Tail first, so the body covers where it joins.
        tail = QPainterPath(QPointF(box.left() + w * 0.62, box.top() + h * 0.80))
        tail.cubicTo(
            QPointF(box.right() + w * 0.04, box.bottom()),
            QPointF(box.right(), box.top() + h * 0.34),
            QPointF(box.right() - w * 0.22, box.top() + h * 0.26),
        )
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(detail, stroke))
        p.drawPath(tail)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(detail)
        p.drawEllipse(QRectF(box.left() + w * 0.08, box.top() + h * 0.04, w * 0.34, h * 0.34))

        p.setBrush(fill)
        p.drawEllipse(QRectF(box.left(), box.top() + h * 0.26, w * 0.70, h * 0.60))

        # An eye, in the deeper tone. Without it the body is just a circle.
        eye = w * 0.10
        p.setBrush(detail)
        p.drawEllipse(QRectF(box.left() + w * 0.14, box.top() + h * 0.46, eye, eye))

    @classmethod
    def _draw_zones(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """An arena, with one zone marked inside it."""
        w, h = box.width(), box.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(box, w * 0.20, w * 0.20)

        p.setBrush(detail)
        p.drawEllipse(QRectF(box.left() + w * 0.18, box.top() + h * 0.18, w * 0.38, h * 0.38))

    @classmethod
    def _draw_vocabulary(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A list: three terms, each with its bullet."""
        w, h = box.width(), box.height()
        dot = w * 0.22
        for frac in (0.14, 0.5, 0.86):
            y = box.top() + h * frac
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(detail)
            p.drawEllipse(QRectF(box.left(), y - dot / 2, dot, dot))
            p.setPen(cls._line_pen(fill, stroke * 1.5))
            p.drawLine(QPointF(box.left() + w * 0.40, y), QPointF(box.right(), y))


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
            # The engine reads its colour off this button, so the icon has to
            # be built per button rather than shared across the rail.
            button.setIcon(QIcon(_SectionGlyphEngine(button, key)))
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
