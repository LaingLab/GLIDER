"""Shared building blocks for GLIDER's standalone tool windows.

The Behavior Analysis, Batch Pose Tracking and Session Review windows are all
the same shape of thing: a dense configuration surface on one side, a run
control and its output on the other. They were each built as one flat
:class:`QVBoxLayout`, which is why they read as a wall of controls with no
sense of what to do first.

This module supplies the pieces that give them a common structure -- a window
header, cards, labelled rows, a run rail -- plus the theme they were missing
entirely.

**The theme bug.** ``MainWindow`` applies ``desktop.qss`` to itself, and Qt
propagates a stylesheet to a widget's *children*. All three tool windows are
constructed with ``parent=None``, so they are not children of anything and
received no stylesheet at all: they rendered in the platform's default light
palette while their painted internals (the ethogram bar, the keypoint canvas)
drew themselves in ``colors.CANVAS`` near-black. :func:`apply_tool_theme` is
what closes that gap, and every tool window calls it.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import STYLES_DIR, load_stylesheet

__all__ = [
    "Card",
    "CardGrid",
    "RunRail",
    "StatusPill",
    "ToolHeader",
    "apply_tool_theme",
    "attach_empty_state",
    "caption",
    "data_font",
    "hint",
    "labelled_row",
    "path_label",
    "readable_text_on",
    "scroll_column",
    "set_button_role",
    "set_path_text",
    "set_text_role",
]


# Spacing scale. One set of numbers across all three windows, so a card in the
# Apply tab and a card in Batch Pose Tracking are visibly the same object.
GUTTER = 16  # between the two columns, and around the page
CARD_GAP = 12  # between stacked cards
ROW_GAP = 8  # between rows inside a card

# Three type roles, not one.
#
# The tools had a single face at two sizes, with weight as the only signal --
# which is fine for prose and wrong for an instrument. Two of the three roles
# below exist because of what this app measures:
#
# * UI      the system stack, inherited from desktop.qss. Prose and controls.
# * LABEL   uppercase, tracked. Card titles and column heads, where the letters
#           are read as a tag rather than as a word.
# * DATA    monospace. Every *measured* quantity -- durations, thresholds,
#           frame counts, distances. A cohort table is read down its columns to
#           compare animals, and proportional digits make 1.00 narrower than
#           8.88, so the decimal points do not line up and the column cannot be
#           scanned. Monospace is load-bearing here, not styling.
_DATA_FAMILIES = ("SF Mono", "JetBrains Mono", "Menlo", "Consolas", "DejaVu Sans Mono")


def data_font(size: int = 0, *, bold: bool = False) -> QFont:
    """The monospace face for measured values. See the type-roles note above."""
    font = QFont()
    font.setFamilies(list(_DATA_FAMILIES))
    font.setStyleHint(QFont.StyleHint.Monospace)
    if size:
        font.setPixelSize(size)
    font.setBold(bold)
    return font


def _tracked(font: QFont, spacing: float) -> QFont:
    """Letter-spacing, which Qt style sheets have no property for."""
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    return font


TEXT_ON_LIGHT = "#0a0e13"  # colors.CANVAS
TEXT_ON_DARK = "#ffffff"


def readable_text_on(background: str) -> str:
    """Black or white for *background*, whichever is easier to read on it.

    Needed wherever a colour comes from data rather than the theme -- a
    behaviour's vocabulary colour is chosen by whoever set up the project, so
    no single text colour works: dark text is right on amber and green and
    close to illegible on crimson or violet.

    Relative luminance per WCAG 2.x, then whichever of the two extremes has
    the better contrast ratio against it.
    """
    hex_colour = background.lstrip("#")
    if len(hex_colour) == 3:
        hex_colour = "".join(c * 2 for c in hex_colour)
    try:
        channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    except (ValueError, IndexError):
        return TEXT_ON_DARK  # unparseable: assume a dark ground and stay legible
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    # Contrast against white vs against black, per the WCAG formula.
    return (
        TEXT_ON_LIGHT if (1.05 / (luminance + 0.05)) < ((luminance + 0.05) / 0.05) else TEXT_ON_DARK
    )


def _tool_stylesheet() -> str:
    """``desktop.qss`` with ``tools.qss`` layered over it, icon paths resolved.

    Later rules win in Qt style sheets, so the tool layer can quiet down
    desktop's accent-filled default button without either file knowing about
    the other.

    ``@ICONS@`` in the QSS becomes the absolute path of ``styles/icons``. Qt
    resolves ``url()`` relative to the *application's* working directory, not
    the stylesheet, and it has no data-URI support -- so a token substituted at
    load time is the only way to reference the arrow assets from a package that
    may be installed anywhere. Posix separators because QSS ``url()`` rejects
    Windows backslashes.
    """
    sheet = load_stylesheet("desktop") + "\n" + load_stylesheet("tools")
    return sheet.replace("@ICONS@", (STYLES_DIR / "icons").as_posix())


def apply_tool_theme(widget: QWidget) -> None:
    """Give a top-level tool window the Deep Navy theme."""
    widget.setStyleSheet(_tool_stylesheet())


# ---------------------------------------------------------------------------
# roles
# ---------------------------------------------------------------------------


def _restyle(widget: QWidget) -> None:
    """Make Qt re-evaluate a widget's style after a dynamic property changes.

    Qt resolves property selectors when the style is polished, not when the
    property is set, so a ``[role="primary"]`` rule applied later never takes
    effect without this.
    """
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


def set_button_role(button: QPushButton, role: str) -> QPushButton:
    """Tag a button as ``primary``, ``danger``, ``ghost`` or ``icon``.

    Returns the button so it can be used inline in a layout call.
    """
    button.setProperty("role", role)
    if role == "primary":
        button.setCursor(Qt.CursorShape.PointingHandCursor)
    _restyle(button)
    return button


def set_text_role(label: QLabel, role: str) -> QLabel:
    """Tag a label as ``caption``/``muted``/``hint``/``value``/``path``/status."""
    label.setProperty("textRole", role)
    _restyle(label)
    return label


def caption(text: str) -> QLabel:
    """The left-hand name of a setting."""
    return set_text_role(QLabel(text), "caption")


def hint(text: str = "") -> QLabel:
    """Small explanatory text under a control. Wraps."""
    label = set_text_role(QLabel(text), "hint")
    label.setWordWrap(True)
    return label


def path_label(placeholder: str) -> QLabel:
    """A slot showing the chosen file or folder.

    Starts empty-looking (italic, muted) and fills in when
    :func:`set_path_text` is given something. Elides rather than wraps: lab
    paths are UNC shares several folders deep, and letting one wrap to four
    lines shoves every button in the row around.
    """
    label = set_text_role(QLabel(placeholder), "path")
    label.setProperty("filled", "false")
    label.setWordWrap(False)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    label.setMinimumWidth(120)
    return label


def set_path_text(label: QLabel, text: str, *, filled: bool) -> None:
    """Update a :func:`path_label`, keeping its filled/empty styling in step."""
    label.setText(text)
    label.setProperty("filled", "true" if filled else "false")
    _restyle(label)


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------


CAPTION_WIDTH = 132
NUMBER_WIDTH = 150  # a spin box wide enough for its value and suffix, no wider


def labelled_row(text: str, *widgets: QWidget | QLayout, stretch: int = 0) -> QHBoxLayout:
    """``caption  [ widget ] [ button ]`` on one line, captions column-aligned.

    The caption gets a fixed width so consecutive rows line up without a
    :class:`QFormLayout` -- which cannot hold a mixed row of a stretching value
    and two trailing buttons, the shape almost every row in these windows has.
    """
    row = QHBoxLayout()
    row.setSpacing(ROW_GAP)
    label = caption(text)
    label.setFixedWidth(CAPTION_WIDTH)
    label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    row.addWidget(label)

    expands = False
    for i, item in enumerate(widgets):
        weight = (1 if i == 0 else 0) if stretch == 0 else (stretch if i == 0 else 0)
        if isinstance(item, QLayout):
            row.addLayout(item, weight)
            expands = True
            continue
        # A number field stretched across the row reads as a text box and puts
        # the value nowhere in particular. Cap it in Python rather than QSS:
        # a QSS max-width clips the paint but leaves the layout still handing
        # out the full width, which knocks the whole caption column sideways.
        if isinstance(item, (QSpinBox, QDoubleSpinBox)):
            item.setMaximumWidth(NUMBER_WIDTH)
        row.addWidget(item, weight)
        policy = item.sizePolicy().horizontalPolicy()
        if weight and policy in (QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored):
            expands = True

    # Nothing in the row can absorb slack, so park it on the right instead of
    # letting Qt spread it between the caption and the field.
    if not expands:
        row.addStretch(1)
    return row


class _EmptyState(QLabel):
    """What to do while a list is empty, shown in place of a black well.

    Every one of these lists starts empty, and an empty ``QListWidget`` is a
    sizeable void that reads as something failing to load rather than as
    somewhere to put things.

    A child of the viewport rather than an item in the model, so it cannot be
    selected, removed, or mistaken for real content -- and it tracks the model
    rather than being toggled by each call site, so a list filled by
    drag-and-drop behaves like one filled from a dialog.

    ``refresh`` has to be a bound method of a QObject, not a closure. PyQt
    drops a connection when its *receiver* QObject dies, and a plain function
    has no receiver: the model outlives the view briefly during teardown, so a
    closure kept firing against an already-deleted C++ widget and raised
    ``RuntimeError`` inside the Qt event loop.
    """

    def __init__(self, view, message: str):
        super().__init__(message, view.viewport())
        self._view = view
        set_text_role(self, "hint")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        model = view.model()
        for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset):
            signal.connect(self.refresh)
        # The viewport has no resize signal, so watch its events instead of
        # reassigning the view's resizeEvent -- monkeypatching that leaves the
        # original bound method alive on a widget Qt is trying to destroy.
        view.viewport().installEventFilter(self)
        self.refresh()

    def eventFilter(self, watched, event):  # Qt camelCase override
        if event.type() == QEvent.Type.Resize:
            self.refresh()
        return super().eventFilter(watched, event)

    def refresh(self, *_args) -> None:
        try:
            empty = self._view.model().rowCount() == 0
            rect = self._view.viewport().rect()
        except RuntimeError:  # the view is mid-teardown; nothing left to show
            return
        self.setVisible(empty)
        self.setGeometry(rect.adjusted(16, 0, -16, 0))


def attach_empty_state(view, message: str) -> QLabel:
    """Show *message* over *view* whenever it holds no rows."""
    return _EmptyState(view, message)


def separator() -> QFrame:
    """A hairline rule inside a card."""
    line = QFrame()
    line.setObjectName("CardSeparator")
    line.setFrameShape(QFrame.Shape.HLine)
    return line


def scroll_column(width: int = 0) -> tuple[QScrollArea, QVBoxLayout]:
    """A vertically scrolling column of cards.

    Returns ``(area, layout)``. The configuration side of every tool window is
    taller than a laptop screen at some setting or other, and scrolling one
    column beats scrolling the whole window and taking the run button with it.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    holder = QWidget()
    holder.setObjectName("ToolPage")
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, GUTTER // 2, 0)  # room for the scrollbar
    layout.setSpacing(CARD_GAP)
    area.setWidget(holder)
    if width:
        area.setMinimumWidth(width)
    return area, layout


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------


class ToolHeader(QFrame):
    """The title strip at the top of a tool window.

    Says what the window is and what it is for, and holds the actions that
    apply to the whole window (open, load, switch session) rather than to one
    section of it.
    """

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("ToolHeader")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(GUTTER, 10, GUTTER, 10)
        layout.setSpacing(ROW_GAP)

        text = QVBoxLayout()
        text.setSpacing(1)
        self._title = QLabel(title)
        self._title.setObjectName("ToolHeaderTitle")
        text.addWidget(self._title)
        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("ToolHeaderSubtitle")
        self._subtitle.setVisible(bool(subtitle))
        text.addWidget(self._subtitle)
        layout.addLayout(text)
        layout.addStretch(1)

        self._actions = QHBoxLayout()
        self._actions.setSpacing(6)
        layout.addLayout(self._actions)

    def set_subtitle(self, text: str) -> None:
        self._subtitle.setText(text)
        self._subtitle.setVisible(bool(text))

    def add_action(self, widget: QWidget) -> QWidget:
        """Put a widget in the header's right-hand action group."""
        self._actions.addWidget(widget)
        return widget

    def add_stretch(self) -> None:
        self._actions.addStretch(1)


class StatusPill(QLabel):
    """A small coloured state chip: idle / running / ok / warn / error."""

    def __init__(self, text: str = "Idle", parent=None):
        super().__init__(text, parent)
        self.setObjectName("StatusPill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state("idle", text)

    def set_state(self, state: str, text: str | None = None) -> None:
        self.setProperty("state", state)
        if text is not None:
            self.setText(text)
        _restyle(self)


class Card(QFrame):
    """A titled surface holding one group of related controls.

    The unit the redesigned windows are built from. A card owns its own
    padding and a body layout, so callers add rows rather than managing
    margins -- which is what kept the old layouts from having any consistent
    rhythm.
    """

    def __init__(self, title: str = "", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(GUTTER, 12, GUTTER, 14)
        outer.setSpacing(10)

        self._header = QHBoxLayout()
        self._header.setSpacing(ROW_GAP)
        self._title = QLabel(title.upper())  # QSS has no text-transform
        self._title.setObjectName("CardTitle")
        # Uppercase at 11px sets too tight to read as a label; tracking is what
        # makes a run of capitals scan as a tag rather than a shouted word.
        self._title.setFont(_tracked(self._title.font(), 0.9))
        self._header.addWidget(self._title)

        self._subtitle = QLabel(subtitle)
        self._subtitle.setObjectName("CardSubtitle")
        self._subtitle.setVisible(bool(subtitle))
        self._header.addWidget(self._subtitle)
        self._header.addStretch(1)

        self._badge = QLabel("")
        self._badge.setObjectName("CardBadge")
        self._badge.setVisible(False)
        self._header.addWidget(self._badge)

        self._has_header = bool(title)
        if self._has_header:
            outer.addLayout(self._header)

        self.body = QVBoxLayout()
        self.body.setSpacing(ROW_GAP)
        outer.addLayout(self.body)
        self._outer = outer

    # -- content ----------------------------------------------------------

    def add(self, item: QWidget | QLayout, stretch: int = 0) -> None:
        if isinstance(item, QLayout):
            self.body.addLayout(item, stretch)
        else:
            self.body.addWidget(item, stretch)

    def add_row(self, text: str, *widgets: QWidget | QLayout) -> QHBoxLayout:
        row = labelled_row(text, *widgets)
        self.body.addLayout(row)
        return row

    def add_separator(self) -> None:
        self.body.addWidget(separator())

    # -- header -----------------------------------------------------------

    def set_badge(self, text: str) -> None:
        """A short right-aligned summary: "3 videos", "off", "not set"."""
        self._badge.setText(text)
        self._badge.setVisible(bool(text))

    def add_header_widget(self, widget: QWidget) -> QWidget:
        """Put a control in the card's title row (a toggle, a small action)."""
        self._header.addWidget(widget)
        return widget


class CardGrid(QWidget):
    """Two side-by-side columns of cards that collapse to one when narrow.

    Used for rows of small cards that would each waste half a screen on their
    own.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(CARD_GAP)
        self._grid.setVerticalSpacing(CARD_GAP)
        self._n = 0

    def add_card(self, card: Card) -> Card:
        self._grid.addWidget(card, self._n // 2, self._n % 2)
        self._n += 1
        return card


class RunRail(QWidget):
    """The right-hand column: the primary action, its progress, and its output.

    Pinned beside the configuration rather than under it. In the old layouts
    the Run button sat at the bottom of a long stack, so on a laptop screen the
    thing the window exists to do was off-screen while you set it up -- and the
    results pane below it grew during a run and pushed it further away.
    """

    def __init__(self, action_text: str = "Run", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(CARD_GAP)

        self.card = Card("Run")
        self.status = StatusPill("Idle")
        self.card.add_header_widget(self.status)

        self.button = QPushButton(action_text)
        set_button_role(self.button, "primary")
        self.button.setMinimumHeight(38)

        self.secondary = QPushButton("Cancel")
        set_button_role(self.secondary, "danger")
        self.secondary.setVisible(False)

        buttons = QHBoxLayout()
        buttons.setSpacing(ROW_GAP)
        buttons.addWidget(self.button, 1)
        buttons.addWidget(self.secondary)
        self.card.add(buttons)

        self.blocker = hint("")
        self.blocker.setVisible(False)
        self.card.add(self.blocker)

        layout.addWidget(self.card)
        # Trailing filler so a rail holding only the run card keeps that card
        # hugging its contents at the top, instead of the card stretching to
        # the window height with its rows drifting apart. Anything added with
        # a stretch of its own retires the filler.
        layout.addStretch(1)
        self._layout = layout

    def add(self, item: QWidget | QLayout, stretch: int = 0) -> None:
        index = self._layout.count() - 1  # before the filler
        if isinstance(item, QLayout):
            self._layout.insertLayout(index, item, stretch)
        else:
            self._layout.insertWidget(index, item, stretch)
        if stretch:
            self._layout.setStretch(self._layout.count() - 1, 0)

    def set_blocker(self, message: str) -> None:
        """Say why Run is unavailable, in place, instead of only on click."""
        self.blocker.setText(message)
        self.blocker.setVisible(bool(message))
        set_text_role(self.blocker, "warning" if message else "hint")
