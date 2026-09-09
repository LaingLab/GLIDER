"""The section-rail and tool-card icons: Lucide line art, in pastel.

Used by the Experiment tab's section rail and the Analyze tab's tool cards.
Lives here rather than in either of them because both need the same set.

**The shapes are Lucide** (``gui/styles/icons/lucide/``, ISC -- the licence
travels with them in that directory and is shipped by ``package-data``). ISC was
the deciding factor over the alternatives: it permits commercial use and
redistribution with no attribution obligation, so vendoring the files into a
public repository carries nothing that has to be tracked or re-checked later.
The set that preceded these was hand-drawn here for exactly that reason, and the
drawings are gone now -- Lucide's are better, and a house icon set is a
maintenance burden nobody asked for.

**The colour is ours.** Lucide ships monochrome ``stroke="currentColor"`` paths;
each icon is recoloured to a pastel from :mod:`~glider.gui.styles.colors` on the
way to the painter. That is not decoration:

* Colour is what makes the rail readable at a glance. You find Mice by the pink
  rat rather than by reading five labels, which is the whole reason an icon
  beats a word there.
* It is deliberately *not* a state. The icon keeps its hue when its row is
  selected, while the label beside it goes accent. An icon that changed colour
  on selection would throw away the identity for the one entry you had already
  found.

The **light** tone of each pair is used rather than the deep one: these are thin
strokes on a near-black ground, and the deeper halves of the pairs -- which
exist for filled shapes -- go muddy at 2px.

Rendered through :class:`QSvgRenderer` at whatever size is asked for, rather
than from a cached pixmap, so they stay crisp on a Retina panel and at the two
quite different sizes in use (20px on the rail, 40px on a card).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from PyQt6.QtCore import QRect, QRectF, QSize, Qt
from PyQt6.QtGui import QIcon, QIconEngine, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QWidget

from glider.gui.styles import colors

logger = logging.getLogger(__name__)

__all__ = ["GLYPH_COLOURS", "GLYPH_KEYS", "ICON_DIR", "PastelGlyphEngine", "glyph_icon"]

#: Where the vendored Lucide SVGs live, beside the ISC licence that covers them.
ICON_DIR = Path(__file__).resolve().parent.parent / "styles" / "icons" / "lucide"

#: Stroke colour per glyph key.
#:
#: The Analyze tab's five reuse the Experiment rail's hues rather than
#: multiplying them: the two pages are never on screen together, so a shared hue
#: costs nothing, and ten distinct pastels would be a colour wheel rather than a
#: palette.
GLYPH_COLOURS: dict[str, str] = {
    # Experiment rail
    "metadata": colors.PASTEL_LILAC,
    "mice": colors.PASTEL_ROSE,
    "zones": colors.PASTEL_MINT,
    "vocabulary": colors.PASTEL_PEACH,
    "plugins": colors.PASTEL_SKY,
    # Analyze tab
    "behavior": colors.PASTEL_LILAC,
    "pose": colors.PASTEL_ROSE,
    "review": colors.PASTEL_PEACH,
    "multicam": colors.PASTEL_SKY,
    "devices": colors.PASTEL_MINT,
}

#: Every glyph that can be drawn.
GLYPH_KEYS: tuple[str, ...] = tuple(GLYPH_COLOURS)


@lru_cache(maxsize=32)
def _recoloured_svg(key: str) -> bytes | None:
    """The SVG for ``key`` with its stroke colour substituted in.

    Lucide draws with ``stroke="currentColor"``, which is a CSS notion Qt's SVG
    renderer does not resolve -- left alone it renders black, which on this
    theme is invisible. Substituting the literal is the whole recolouring step.

    Returns ``None`` if the file is missing or unreadable, which the caller
    turns into a blank icon rather than a crash: a missing asset should cost a
    button its picture and nothing else.
    """
    colour = GLYPH_COLOURS.get(key)
    if colour is None:
        logger.debug("No colour registered for glyph %r", key)
        return None
    try:
        source = (ICON_DIR / f"{key}.svg").read_text(encoding="utf-8")
    except OSError:
        logger.warning("Glyph %r could not be read from %s", key, ICON_DIR)
        return None
    return source.replace("currentColor", colour).encode("utf-8")


class PastelGlyphEngine(QIconEngine):
    """Paints one recoloured Lucide glyph, keyed by name.

    Args:
        widget: Kept for call compatibility and future use; the colour comes
            from :data:`GLYPH_COLOURS`, not from the widget's palette. That is
            deliberate -- see the module docstring on why these do not follow
            the selection state.
        key: One of :data:`GLYPH_KEYS`.
    """

    def __init__(self, widget: QWidget | None, key: str) -> None:
        super().__init__()
        self._widget = widget
        self._key = key

    def clone(self) -> PastelGlyphEngine:
        return PastelGlyphEngine(self._widget, self._key)

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        # QIconEngine's default pixmap() is documented to rasterise onto a
        # transparent buffer by calling paint(), but on this Qt build it hands
        # back a fully opaque one instead -- every icon comes out as a solid
        # block, indistinguishable from every other. Overriding it with exactly
        # what the documentation promises is what QToolButton's icon drawing
        # actually needs to show the shape.
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
        data = _recoloured_svg(self._key)
        if data is None:
            return
        renderer = QSvgRenderer(data)
        if not renderer.isValid():  # pragma: no cover - corrupt vendored asset
            logger.warning("Glyph %r is not a renderable SVG", self._key)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        renderer.render(painter, QRectF(rect))
        painter.restore()


def glyph_icon(widget: QWidget | None, key: str, size: int) -> QIcon:
    """A :class:`QIcon` drawing ``key``, sized for ``widget``.

    Touching the pixmap here forces the engine through its own paint path once,
    at construction rather than at first show -- which is where a missing or
    corrupt asset would otherwise surface, as a blank button nobody can explain.
    """
    icon = QIcon(PastelGlyphEngine(widget, key))
    icon.pixmap(QSize(size, size))
    return icon
