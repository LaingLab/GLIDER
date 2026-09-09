"""The Analyze tab: the tool windows, with a visible front door.

Behavior Analysis, Batch Pose Tracking, Session Review, Multi-Camera Recording
and the device check all lived on the Tools menu, which is the worst place for
them. Nobody opens a menu looking for something they have not been told exists,
and these are exactly the features you cannot know to look for -- which is why
the complaint about GLIDER was never "that tool is bad", it was that nobody
knew it was there.

**They stay windows.** Each is a ``QMainWindow`` that wants a whole screen, and
the multi-camera grid says so in its own docstring: sixteen tiles need a
monitor, often the second one. So this page is a set of front doors rather than
an attempt to inline them -- the thing that was missing was never the window,
it was any sign that the window existed.

**A card says why a tool is unavailable, where the menu could only grey out.**
Three of the five need an optional dependency stack, and a greyed menu item
with a tooltip is a dead end you have to hover to read. On a card there is room
to print the install line, so "Behavior Analysis is greyed out" stops being a
support question.

**No colour is set from Python here**: every part carries an ``objectName`` and
``desktop.qss`` owns the appearance. The card glyphs are the exception, and
they live in :mod:`~glider.gui.widgets.pastel_glyphs`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from glider.gui.widgets.pastel_glyphs import glyph_icon

logger = logging.getLogger(__name__)

__all__ = ["CARD_ICON_PX", "ToolCard", "ToolsPage"]

#: Card glyph edge length in pixels. Much larger than the rail's 20px: a card
#: is a target you pick out of a page, and the illustration is what you pick it
#: out by.
CARD_ICON_PX = 40


@dataclass(frozen=True)
class ToolCard:
    """One tool, as the page needs to describe it.

    Args:
        key: Identifies the tool to the owner; emitted by
            :attr:`ToolsPage.tool_chosen`.
        title: The tool's name.
        description: What it does, in one line.
        glyph: A key from :mod:`~glider.gui.widgets.pastel_glyphs`.
        available: Whether it can be opened at all.
        unavailable_reason: Shown in place of the description when
            ``available`` is False. Say what to install, not that something is
            missing -- the card exists to end the question, not to restate it.
    """

    key: str
    title: str
    description: str
    glyph: str
    available: bool = True
    unavailable_reason: str = ""


class ToolsPage(QWidget):
    """A column of tool cards.

    Args:
        cards: The tools, in the order they should be offered.
        parent: Standard Qt parent.

    Signals:
        tool_chosen: Emitted with a card's ``key`` when an *available* card is
            clicked. An unavailable one never emits: it has nothing to open.
    """

    tool_chosen = pyqtSignal(str)

    def __init__(self, cards: list[ToolCard], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("toolsPage")
        self._cards = list(cards)
        self._frames: dict[str, QFrame] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setObjectName("toolsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        host = QWidget()
        host.setObjectName("toolsHost")
        column = QVBoxLayout(host)
        column.setContentsMargins(28, 24, 28, 24)
        column.setSpacing(10)

        heading = QLabel("Tools")
        heading.setObjectName("toolsHeading")
        column.addWidget(heading)

        subheading = QLabel(
            "Analysis and recording tools. Each opens in its own window, so you "
            "can keep working here while it runs."
        )
        subheading.setObjectName("toolsSubheading")
        subheading.setWordWrap(True)
        column.addWidget(subheading)
        column.addSpacing(8)

        for card in self._cards:
            frame = self._build_card(card)
            self._frames[card.key] = frame
            column.addWidget(frame)

        column.addStretch(1)
        scroll.setWidget(host)

    # ------------------------------------------------------------------ build

    def _build_card(self, card: ToolCard) -> QFrame:
        frame = QFrame()
        frame.setObjectName("toolCard")
        # Read by desktop.qss, which greys the whole card rather than only the
        # text: a card that looks live and does nothing is worse than one that
        # plainly says it is not ready.
        frame.setProperty("available", "true" if card.available else "false")
        if card.available:
            frame.setCursor(Qt.CursorShape.PointingHandCursor)
        frame.mouseReleaseEvent = lambda _event, key=card.key: self._on_clicked(key)

        row = QHBoxLayout(frame)
        row.setContentsMargins(16, 14, 16, 14)
        row.setSpacing(14)

        icon = QLabel()
        icon.setObjectName("toolCardIcon")
        icon.setPixmap(
            glyph_icon(frame, card.glyph, CARD_ICON_PX).pixmap(QSize(CARD_ICON_PX, CARD_ICON_PX))
        )
        icon.setFixedSize(CARD_ICON_PX, CARD_ICON_PX)
        row.addWidget(icon, 0, Qt.AlignmentFlag.AlignTop)

        text = QVBoxLayout()
        text.setSpacing(2)

        title = QLabel(card.title)
        title.setObjectName("toolCardTitle")
        text.addWidget(title)

        body = QLabel(card.description if card.available else card.unavailable_reason)
        body.setObjectName("toolCardBody" if card.available else "toolCardUnavailable")
        body.setWordWrap(True)
        text.addWidget(body)

        row.addLayout(text, 1)
        return frame

    # ------------------------------------------------------------- selection

    def _on_clicked(self, key: str) -> None:
        """A card press. Unavailable cards are inert.

        Checked here rather than by not connecting, so the reason lives beside
        the card that carries it -- and so a card whose availability is
        recomputed later needs nothing rewiring.
        """
        card = next((c for c in self._cards if c.key == key), None)
        if card is None or not card.available:
            return
        self.tool_chosen.emit(key)

    # ---------------------------------------------------------------- probes

    def cards(self) -> list[ToolCard]:
        """The cards, in order. For tests."""
        return list(self._cards)

    def card_frames(self) -> dict[str, QFrame]:
        """The card widgets by key. For tests."""
        return dict(self._frames)
