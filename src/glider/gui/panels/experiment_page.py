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

#: Rail icon edge length in pixels. 20 rather than a more conventional 16
#: because these are illustrations, not symbols: the mouse has an ear, an eye
#: and whiskers, and below about 18px those stop resolving and it degrades into
#: a pink blob. The rail buttons are 36px tall, so the extra 4px costs nothing.
ICON_PX = 20

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

    # -- the four glyphs --
    #
    # Each is built from three tones: the pastel `fill`, the deeper `detail`,
    # and a `light` highlight derived from the fill. Two tones give you a
    # silhouette with a mark on it; the third is what makes a shape read as
    # lit from somewhere and turns a diagram into an illustration.

    @staticmethod
    def _light(fill: QColor) -> QColor:
        """The highlight tone: the fill, lifted.

        Derived rather than named in :mod:`~glider.gui.styles.colors` because
        it is not a decision -- it is the same colour with more light on it,
        and four more constants would be four more things to keep in step with
        the fills they belong to.
        """
        return fill.lighter(112)

    @classmethod
    def _draw_metadata(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A record card: header band, ruled lines, and a clip at the top."""
        w, h = box.width(), box.height()
        card = QRectF(box.left() + w * 0.08, box.top() + h * 0.06, w * 0.84, h * 0.94)
        radius = w * 0.13

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(card, radius, radius)

        # Header band, clipped to the card so it keeps the top two corners.
        clip = QPainterPath()
        clip.addRoundedRect(card, radius, radius)
        p.save()
        p.setClipPath(clip)
        p.setBrush(detail)
        p.drawRect(QRectF(card.left(), card.top(), card.width(), h * 0.22))
        p.restore()

        # The clip: a tab straddling the header, in the light tone so it reads
        # as sitting on top rather than as a hole in the band.
        tab = QRectF(card.center().x() - w * 0.13, box.top(), w * 0.26, h * 0.13)
        p.setBrush(cls._light(fill))
        p.drawRoundedRect(tab, w * 0.05, w * 0.05)

        # Ruled lines. Widths vary and the last is short, the way a paragraph
        # ends -- equal strokes read as a barcode rather than as writing.
        p.setPen(cls._line_pen(detail, stroke * 0.85))
        for frac, right_inset in ((0.42, 0.16), (0.60, 0.16), (0.78, 0.44)):
            y = card.top() + card.height() * frac
            p.drawLine(
                QPointF(card.left() + w * 0.15, y),
                QPointF(card.right() - w * right_inset, y),
            )

    @classmethod
    def _draw_mouse(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A mouse in three-quarter profile: ear, snout, eye, whiskers, tail."""
        w, h = box.width(), box.height()
        light = cls._light(fill)

        # Tail first: the body has to cover the joint, or it reads as a wire
        # threaded through the animal.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(detail, stroke * 0.9))
        tail = QPainterPath(QPointF(box.left() + w * 0.58, box.top() + h * 0.82))
        tail.cubicTo(
            QPointF(box.right() + w * 0.02, box.bottom() + h * 0.02),
            QPointF(box.right() + w * 0.02, box.top() + h * 0.30),
            QPointF(box.right() - w * 0.20, box.top() + h * 0.20),
        )
        p.drawPath(tail)

        p.setPen(Qt.PenStyle.NoPen)

        # Ear: outer in the deep tone, inner in the light one.
        ear = QRectF(box.left() + w * 0.20, box.top() + h * 0.02, w * 0.32, h * 0.32)
        p.setBrush(detail)
        p.drawEllipse(ear)
        p.setBrush(light)
        p.drawEllipse(ear.adjusted(w * 0.07, h * 0.07, -w * 0.07, -h * 0.07))

        # Haunch, then head, so the head sits in front.
        p.setBrush(fill)
        p.drawEllipse(QRectF(box.left() + w * 0.22, box.top() + h * 0.30, w * 0.52, h * 0.56))
        p.drawEllipse(QRectF(box.left() + w * 0.02, box.top() + h * 0.34, w * 0.44, h * 0.48))

        # Snout, tapering forward off the head.
        snout = QPainterPath(QPointF(box.left() + w * 0.22, box.top() + h * 0.52))
        snout.quadTo(
            QPointF(box.left() - w * 0.04, box.top() + h * 0.62),
            QPointF(box.left() + w * 0.20, box.top() + h * 0.76),
        )
        snout.closeSubpath()
        p.setBrush(light)
        p.drawPath(snout)

        # Nose, then eye. The nose is what fixes which way it is facing.
        p.setBrush(detail)
        nose = w * 0.09
        p.drawEllipse(QRectF(box.left() - w * 0.01, box.top() + h * 0.60, nose, nose))
        eye = w * 0.11
        p.drawEllipse(QRectF(box.left() + w * 0.16, box.top() + h * 0.48, eye, eye))

        # Whiskers: two strokes off the snout, thinner than everything else so
        # they read as hair rather than as limbs.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(detail, stroke * 0.5))
        for dy in (0.60, 0.72):
            p.drawLine(
                QPointF(box.left() + w * 0.04, box.top() + h * dy),
                QPointF(box.left() - w * 0.06, box.top() + h * (dy - 0.10)),
            )

        # A hind foot, so the body has something to stand on.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(detail)
        p.drawEllipse(QRectF(box.left() + w * 0.30, box.top() + h * 0.78, w * 0.20, h * 0.14))

    @classmethod
    def _draw_zones(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """An arena from above: floor, a centre zone, and a corner zone."""
        w, h = box.width(), box.height()
        radius = w * 0.20

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(box, radius, radius)

        clip = QPainterPath()
        clip.addRoundedRect(box, radius, radius)
        p.save()
        p.setClipPath(clip)

        # Corner zone: a quadrant of the floor, in the light tone. Clipped to
        # the arena, so it keeps the rounded corner instead of squaring it off
        # -- which is the whole reason there is a clip here.
        corner = QPainterPath()
        corner.moveTo(box.topRight())
        corner.lineTo(QPointF(box.right() - w * 0.42, box.top()))
        corner.quadTo(
            QPointF(box.right() - w * 0.20, box.top() + h * 0.20),
            QPointF(box.right(), box.top() + h * 0.42),
        )
        corner.closeSubpath()
        p.setBrush(cls._light(fill))
        p.drawPath(corner)
        # Outlined as well as filled. A tint alone is nearly invisible against
        # the floor it is drawn on -- and an outline is what a zone *is*, so
        # the edge is also the more honest picture.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(detail, stroke * 0.7))
        p.drawPath(corner)
        p.restore()

        # Wall, inset, so the floor has an edge rather than just ending.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(detail, stroke * 0.7))
        inset = w * 0.10
        p.drawRoundedRect(box.adjusted(inset, inset, -inset, -inset), radius * 0.7, radius * 0.7)

        # Centre zone: filled ring, the shape an open-field centre is drawn as.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(detail)
        centre = w * 0.34
        p.drawEllipse(
            QRectF(box.center().x() - centre / 2, box.center().y() - centre / 2, centre, centre)
        )

    @classmethod
    def _draw_vocabulary(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """An open book: two pages, a spine, ruled terms, and a bookmark.

        A book rather than a bulleted list, which is what this was: a list of
        three lines is what half the icons in any toolbar already look like,
        and the section is a reference the lab writes once and consults after.
        """
        w, h = box.width(), box.height()
        top, bottom = box.top() + h * 0.16, box.bottom() - h * 0.06
        mid_x = box.center().x()
        light = cls._light(fill)

        # Each page is a leaf: square on the outside, curving up to the spine.
        for sign, tone in ((-1, fill), (1, light)):
            outer = mid_x + sign * w * 0.50
            page = QPainterPath(QPointF(mid_x, top))
            page.quadTo(
                QPointF(mid_x + sign * w * 0.28, top - h * 0.10),
                QPointF(outer, top + h * 0.06),
            )
            page.lineTo(QPointF(outer, bottom - h * 0.06))
            page.quadTo(
                QPointF(mid_x + sign * w * 0.28, bottom - h * 0.16),
                QPointF(mid_x, bottom),
            )
            page.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(tone)
            p.drawPath(page)

        # Ruled terms, two to a page, shorter towards the spine so they follow
        # the curve rather than cutting across it.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(detail, stroke * 0.55))
        for frac in (0.42, 0.62):
            y = box.top() + h * frac
            for sign in (-1, 1):
                p.drawLine(
                    QPointF(mid_x + sign * w * 0.10, y),
                    QPointF(mid_x + sign * w * 0.40, y),
                )

        # Spine, and a bookmark hanging out of it.
        p.setPen(cls._line_pen(detail, stroke * 0.8))
        p.drawLine(QPointF(mid_x, top + h * 0.02), QPointF(mid_x, bottom))

        # Bookmark, hanging down the right-hand page rather than out of the
        # spine: at the spine it sits on the darkest part of the drawing and
        # disappears into it.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(detail)
        left = mid_x + w * 0.20
        width = w * 0.13
        mark = QPainterPath(QPointF(left, top - h * 0.02))
        mark.lineTo(QPointF(left + width, top - h * 0.04))
        mark.lineTo(QPointF(left + width, top + h * 0.30))
        mark.lineTo(QPointF(left + width / 2, top + h * 0.22))
        mark.lineTo(QPointF(left, top + h * 0.32))
        mark.closeSubpath()
        p.drawPath(mark)


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
