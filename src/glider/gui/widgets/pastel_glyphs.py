"""Pastel glyphs: small illustrations drawn rather than shipped.

Used by the Experiment tab's section rail and the Analyze tab's tool cards.
Lives here rather than in either of them because both need the same set, and a
second copy would be a second set of drawings to keep in step.

Vector, and painted fresh per request, for the same reason
:class:`~glider.gui.shell.status_strip._SidebarGlyphEngine` is: these are drawn
small, and a fixed-resolution pixmap stretched to a fractional device pixel
ratio smears. One engine rather than a folder of SVGs for the same reason --
two really, since a monochrome icon set would need a selected variant of each.

**This is the one place in the GUI where Python names colours**, against the
rule the rest of the package follows, and the exception is the point.
Everything else on screen is a control, and a control's colour is a *state*,
which is why ``desktop.qss`` owns it. These are pictures, and a picture's
colour is its identity: you find Mice by its pink mouse rather than by reading
four labels, which is the whole reason an icon beats a word here. An icon that
went accent on selection, the way the label beside it does, would throw that
away for the one entry you had already found. The pairs come from
:mod:`~glider.gui.styles.colors`, so a re-theme still has one place to go.
"""

from __future__ import annotations

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
from PyQt6.QtWidgets import QWidget

from glider.gui.styles import colors

__all__ = ["GLYPH_KEYS", "PastelGlyphEngine", "glyph_icon"]


class PastelGlyphEngine(QIconEngine):
    """Paints one pastel illustration, keyed by name.

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
        "plugins": (colors.PASTEL_SKY, colors.PASTEL_SKY_DEEP),
        # Analyze tab. The hues repeat from the set above rather than
        # multiplying it: the two pages are never on screen together, so a
        # shared hue costs nothing, and five more pastels would start to look
        # like a colour wheel rather than a palette.
        "behavior": (colors.PASTEL_LILAC, colors.PASTEL_LILAC_DEEP),
        "pose": (colors.PASTEL_ROSE, colors.PASTEL_ROSE_DEEP),
        "review": (colors.PASTEL_PEACH, colors.PASTEL_PEACH_DEEP),
        "multicam": (colors.PASTEL_SKY, colors.PASTEL_SKY_DEEP),
        "devices": (colors.PASTEL_MINT, colors.PASTEL_MINT_DEEP),
    }

    def __init__(self, widget: QWidget, key: str) -> None:
        super().__init__()
        self._widget = widget
        self._key = key

    def clone(self) -> PastelGlyphEngine:
        return PastelGlyphEngine(self._widget, self._key)

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
            "plugins": self._draw_plugin,
            "behavior": self._draw_behavior,
            "pose": self._draw_pose,
            "review": self._draw_review,
            "multicam": self._draw_multicam,
            "devices": self._draw_devices,
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

    @classmethod
    def _draw_plugin(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A jigsaw piece: a tab on one side, a socket on the other.

        Both, not one. A piece with only a tab is a shape with a bump; the pair
        is what says "this fits into something" -- which is the whole idea of a
        plugin, and the reason the jigsaw piece became the convention.
        """
        w, h = box.width(), box.height()
        body = QRectF(box.left() + w * 0.06, box.top() + h * 0.10, w * 0.80, h * 0.80)
        knob = w * 0.17

        piece = QPainterPath()
        piece.addRoundedRect(body, w * 0.12, w * 0.12)

        # Tab on the right, socket on the left, both at the same height so the
        # piece reads as one row of a puzzle rather than as a random blob.
        y = body.center().y()
        tab = QPainterPath()
        tab.addEllipse(QPointF(body.right(), y), knob, knob)
        socket = QPainterPath()
        socket.addEllipse(QPointF(body.left(), y), knob, knob)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawPath(piece.united(tab).subtracted(socket))

        # A notch of the deeper tone along the top, so the piece has a face
        # rather than being a flat silhouette.
        p.save()
        clip = QPainterPath()
        clip.addRoundedRect(body, w * 0.12, w * 0.12)
        p.setClipPath(clip)
        p.setBrush(detail)
        p.drawRect(QRectF(body.left(), body.top(), body.width(), h * 0.20))
        p.restore()

        # And the highlight on the tab, which is the part that overhangs.
        p.setBrush(cls._light(fill))
        p.drawEllipse(QPointF(body.right(), y), knob * 0.52, knob * 0.52)

    # -- Analyze tab --

    @classmethod
    def _draw_behavior(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """An ethogram: three lanes of behaviour blocks over time.

        The picture the tool actually produces, rather than a generic bar
        chart -- what makes it recognisable is that the lanes are ragged and
        aligned to different moments, which is what an ethogram looks like and
        what a bar chart never does.
        """
        w, h = box.width(), box.height()
        lane_h = h * 0.20
        # (lane, [(start, width)]) as fractions of the width.
        lanes = (
            (0.06, ((0.00, 0.34), (0.44, 0.26), (0.80, 0.20))),
            (0.40, ((0.10, 0.22), (0.40, 0.50))),
            (0.74, ((0.00, 0.18), (0.26, 0.20), (0.56, 0.44))),
        )
        for i, (top_frac, blocks) in enumerate(lanes):
            y = box.top() + h * top_frac
            tone = detail if i == 1 else fill
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(tone)
            for x_frac, width_frac in blocks:
                p.drawRoundedRect(
                    QRectF(box.left() + w * x_frac, y, w * width_frac, lane_h),
                    lane_h * 0.35,
                    lane_h * 0.35,
                )

    @classmethod
    def _draw_pose(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A keypoint skeleton: joints, and the links between them."""
        w, h = box.width(), box.height()
        # Nose, ears, shoulders, hips, tail-base -- roughly the rodent skeleton
        # the pose models emit, read down the body.
        nose = QPointF(box.left() + w * 0.16, box.top() + h * 0.14)
        ear_l = QPointF(box.left() + w * 0.02, box.top() + h * 0.40)
        ear_r = QPointF(box.left() + w * 0.40, box.top() + h * 0.34)
        spine = QPointF(box.left() + w * 0.42, box.top() + h * 0.62)
        hip_l = QPointF(box.left() + w * 0.24, box.top() + h * 0.86)
        tail = QPointF(box.left() + w * 0.86, box.top() + h * 0.90)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(fill, stroke * 1.1))
        for a, b in ((nose, ear_l), (nose, ear_r), (ear_r, spine), (spine, hip_l), (spine, tail)):
            p.drawLine(a, b)

        p.setPen(Qt.PenStyle.NoPen)
        radius = w * 0.11
        for i, point in enumerate((nose, ear_l, ear_r, spine, hip_l, tail)):
            # The nose in the deeper tone: a skeleton with every joint the same
            # weight has no front, and which end is the head is the one thing
            # you check a pose overlay for.
            p.setBrush(detail if i == 0 else cls._light(fill))
            p.drawEllipse(point, radius, radius)

    @classmethod
    def _draw_review(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A recording being played back: a frame, a play mark, a scrub bar.

        This was a scrubber alone -- a track with a playhead across it -- and a
        thin horizontal bar crossed by a tall vertical one just reads as a plus
        sign. The frame is what makes the same two marks say "video".
        """
        w, h = box.width(), box.height()
        frame = QRectF(box.left(), box.top(), w, h * 0.72)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(frame, w * 0.14, w * 0.14)

        # Play mark, centred on the frame rather than on the whole glyph: the
        # scrub bar below is not part of the picture it sits in.
        play = QPainterPath()
        cx, cy = frame.center().x() + w * 0.03, frame.center().y()
        size = min(frame.width(), frame.height()) * 0.42
        play.moveTo(QPointF(cx - size * 0.45, cy - size * 0.6))
        play.lineTo(QPointF(cx + size * 0.6, cy))
        play.lineTo(QPointF(cx - size * 0.45, cy + size * 0.6))
        play.closeSubpath()
        p.setBrush(detail)
        p.drawPath(play)

        # Scrub bar under the frame, part-filled: a full one would say the
        # clip had finished, which is not the state anyone reviews in.
        bar_h = h * 0.16
        bar_y = box.bottom() - bar_h
        p.setBrush(cls._light(fill))
        p.drawRoundedRect(QRectF(box.left(), bar_y, w, bar_h), bar_h / 2, bar_h / 2)
        p.setBrush(detail)
        p.drawRoundedRect(QRectF(box.left(), bar_y, w * 0.55, bar_h), bar_h / 2, bar_h / 2)

    @classmethod
    def _draw_multicam(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """Four camera tiles, one of them recording."""
        w, h = box.width(), box.height()
        gap = w * 0.10
        tile_w = (w - gap) / 2
        tile_h = (h - gap) / 2
        radius = tile_w * 0.22

        p.setPen(Qt.PenStyle.NoPen)
        for row in (0, 1):
            for col in (0, 1):
                tile = QRectF(
                    box.left() + col * (tile_w + gap),
                    box.top() + row * (tile_h + gap),
                    tile_w,
                    tile_h,
                )
                p.setBrush(fill)
                p.drawRoundedRect(tile, radius, radius)

        # One tile live: the deeper tone plus a dot. A grid of four identical
        # squares is a grid; the odd one out is what says these are cameras and
        # one of them is running.
        live = QRectF(box.left(), box.top(), tile_w, tile_h)
        p.setBrush(detail)
        p.drawRoundedRect(live, radius, radius)
        dot = tile_w * 0.30
        p.setBrush(cls._light(fill))
        p.drawEllipse(live.center(), dot / 2, dot / 2)

    @classmethod
    def _draw_devices(cls, p: QPainter, box: QRectF, fill, detail, stroke) -> None:
        """A processor: a die with pins down both sides."""
        w, h = box.width(), box.height()
        die = QRectF(box.left() + w * 0.18, box.top() + h * 0.18, w * 0.64, h * 0.64)

        # Pins first, so the die covers where they meet it.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(cls._line_pen(detail, stroke * 0.9))
        for frac in (0.34, 0.5, 0.66):
            y = box.top() + h * frac
            p.drawLine(QPointF(box.left(), y), QPointF(die.left() + w * 0.04, y))
            p.drawLine(QPointF(die.right() - w * 0.04, y), QPointF(box.right(), y))
        for frac in (0.34, 0.5, 0.66):
            x = box.left() + w * frac
            p.drawLine(QPointF(x, box.top()), QPointF(x, die.top() + h * 0.04))
            p.drawLine(QPointF(x, die.bottom() - h * 0.04), QPointF(x, box.bottom()))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(die, w * 0.14, w * 0.14)
        p.setBrush(cls._light(fill))
        inner = die.adjusted(w * 0.14, h * 0.14, -w * 0.14, -h * 0.14)
        p.drawRoundedRect(inner, w * 0.08, w * 0.08)


#: Every glyph this engine can draw.
GLYPH_KEYS: tuple[str, ...] = tuple(PastelGlyphEngine._PALETTE)


def glyph_icon(widget: QWidget, key: str, size: int) -> QIcon:
    """A :class:`QIcon` drawing ``key``, sized for ``widget``.

    The engine is built per widget rather than shared: it reads its box from
    whatever it is asked to paint into, and keeping the association explicit is
    what stops one icon being handed to two buttons that then disagree about
    how big it should be.
    """
    icon = QIcon(PastelGlyphEngine(widget, key))
    # Touching the pixmap here forces the engine through its own paint path
    # once, at construction, rather than at first show -- which is where a
    # broken drawer would otherwise surface as a blank button.
    icon.pixmap(QSize(size, size))
    return icon
