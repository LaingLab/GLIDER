"""The Tools tab: the tool windows, with a visible front door.

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
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from glider.gui.widgets.pastel_glyphs import glyph_icon

logger = logging.getLogger(__name__)

__all__ = ["CARD_ICON_PX", "DEFAULT_SUBHEADING", "ToolCard", "ToolsPage"]

#: Card glyph edge length in pixels. Much larger than the rail's 20px: a card
#: is a target you pick out of a page, and the illustration is what you pick it
#: out by.
CARD_ICON_PX = 40

#: The Tools tab's line under its heading.
DEFAULT_SUBHEADING = (
    "Analysis and recording tools. Each opens in its own window, so you "
    "can keep working here while it runs."
)


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
        install_extra: The GLIDER extra that makes an unavailable tool
            available. When set, the card offers an Install button for it.
    """

    key: str
    title: str
    description: str
    glyph: str
    available: bool = True
    unavailable_reason: str = ""
    install_extra: str = ""


class ToolsPage(QWidget):
    """A column of tool cards.

    Args:
        cards: The tools, in the order they should be offered.
        heading: The title above the cards.
        subheading: The line under the title.
        plain: Name and icon only. The description becomes the card's
            tooltip, and the line under each card is shown only while an
            install is reporting progress.
        embedded: Build for a host that already scrolls -- the landing page's
            right-hand column -- rather than as a page of its own: no scroll
            area of its own (a scroll area inside a scroll area traps the
            wheel) and no page margins.
        parent: Standard Qt parent.

    Signals:
        tool_chosen: Emitted with a card's ``key`` when an *available* card is
            clicked. An unavailable one never emits: it has nothing to open.
        install_requested: Emitted with an extra's name when a card's Install
            button is pressed.
    """

    tool_chosen = pyqtSignal(str)
    install_requested = pyqtSignal(str)

    def __init__(
        self,
        cards: list[ToolCard],
        parent: QWidget | None = None,
        *,
        heading: str = "Tools",
        subheading: str = DEFAULT_SUBHEADING,
        embedded: bool = False,
        plain: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("toolsEmbedded" if embedded else "toolsPage")
        self._plain = plain
        self._cards = list(cards)
        self._frames: dict[str, QFrame] = {}
        self._install_rows: dict[str, tuple[QLabel, QPushButton]] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        host = QWidget()
        host.setObjectName("toolsHost")
        column = QVBoxLayout(host)
        if embedded:
            column.setContentsMargins(0, 0, 0, 0)
        else:
            column.setContentsMargins(28, 24, 28, 24)
        column.setSpacing(10)

        if heading:
            heading_label = QLabel(heading)
            heading_label.setObjectName("toolsHeading")
            column.addWidget(heading_label)

        if subheading:
            subheading_label = QLabel(subheading)
            subheading_label.setObjectName("toolsSubheading")
            subheading_label.setWordWrap(True)
            column.addWidget(subheading_label)
        if heading or subheading:
            column.addSpacing(8)

        for card in self._cards:
            frame = self._build_card(card)
            self._frames[card.key] = frame
            column.addWidget(frame)

        if embedded:
            column.addStretch(1)
            outer.addWidget(host)
            return

        column.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setObjectName("toolsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(host)
        outer.addWidget(scroll)

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
        if self._plain:
            frame.setToolTip(body.text())
            body.setVisible(False)

        row.addLayout(text, 1)
        if not card.available and card.install_extra:
            button = QPushButton("Install")
            button.setObjectName("toolCardInstall")
            button.clicked.connect(
                lambda _=False, extra=card.install_extra: self.install_requested.emit(extra)
            )
            row.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
            self._install_rows[card.key] = (body, button)
        return frame

    def set_install_status(self, extra: str, message: str, *, busy: bool) -> None:
        """Show an install's progress on every card waiting on *extra*."""
        for card in self._cards:
            if card.install_extra == extra and card.key in self._install_rows:
                body, button = self._install_rows[card.key]
                body.setText(message)
                body.setVisible(True)
                button.setEnabled(not busy)

    def install_button(self, key: str) -> QPushButton | None:
        """The card's Install button, if it has one. For tests."""
        row = self._install_rows.get(key)
        return row[1] if row else None

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
