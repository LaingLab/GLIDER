"""The one piece of Builder chrome that cannot be summoned.

Everything else in the shell is on demand: the panels collapse to rails, the
palette appears on a keystroke, the menus are four. This strip is the exception,
and the exception is the point of it. Live hardware and recording state cannot
wait to be asked for -- a board dropping 40 minutes into an unattended overnight
run has to be visible to whoever walks past the rig, not to whoever thinks to
open the Hardware panel. That is also why the device dots live here rather than
being left to that panel: the panel can be collapsed, and this cannot.

Left to right, in 40 fixed pixels: the left panel's toggle, the experiment name
with its dirty marker, the run-state pill, a spacer, one dot per device, the
``Ctrl K`` hint, and the right panel's toggle.

Five decisions carry meaning rather than convenience:

* **A device carries its name, not just a dot.** The dot is 8 px. Anyone who has
  to act on a red one at 3 a.m. needs to know *which* board, and a colour alone
  cannot say that. The name is what makes the alarm actionable.

* **An unrecognised device state is ``unknown``, never ``ok``.** States arrive
  from drivers, so the vocabulary can widen without this file changing. Painting
  a word nobody recognises the same green as a working board is the one
  behaviour that would make this widget actively harmful, so unfamiliar states
  go neutral and keep their raw text in the tooltip. A caller that *has* mapped
  its own vocabulary onto these four passes the raw text alongside the state,
  so the same promise holds for it: ``warn`` cannot tell a handshake in flight
  from a board that has already dropped mid-recording, and the tooltip must.

* **An unrecognised *run* state raises.** The asymmetry is deliberate. Device
  states come from hardware and the outside world gets the benefit of the doubt;
  run state comes from our own code, and a pill quietly stuck on "Idle" through
  a recording is precisely the failure this widget exists to prevent.

* **The toggles are told, not asked.** They reflect the panels; they do not own
  them. :meth:`StatusStrip.set_left_expanded` therefore sets the button without
  re-emitting, or echoing a panel's state back would ping-pong with the click
  that caused it.

* **Each toggle sits at the edge of the panel it controls**, and draws a small
  icon rather than a character. All three used to be bunched at the far left,
  which put the right panel's button the width of a monitor away from the
  right panel. The glyph was ``▌``/``▐``, half-block characters that are
  absent from many fonts and read as a rendering artefact where they are
  present; an arrow replaced those, and this icon replaces the arrow -- a
  small rounded window with a bar on the side the panel actually lives on,
  filled while the panel is open and hollow (an outline and a divider, no
  fill) while it is collapsed. It is a map of the layout rather than a
  direction to press, so the state is legible from the shape and not only
  from the fill, and it still depends on no font: :class:`_PanelToggle`
  paints it itself, reading its colour from the widget's own palette rather
  than from anything named in this file.

**No colour is set from Python here.** Every part carries an ``objectName`` --
and, where it varies, a dynamic ``state`` property -- and ``desktop.qss`` owns
the rest. A widget-level ``setStyleSheet`` out-prioritises the application
stylesheet, so one stray call would make that widget the single thing in the
shell the theme cannot reach. Qt resolves property selectors at polish time
rather than when the property is set, so
:func:`glider.gui.styles.restyle` re-polishes anything whose ``state``
changed; without it the pill keeps the colour it opened with while reporting
the right property. The toggle's icon keeps the same promise a different way:
it is painted, not stencilled from a file, and every paint reads
``self.palette()`` fresh rather than a colour captured once -- the same
reason the arrow it replaced never needed a colour of its own.

**Touch mode is a reversible flag, off by default.** On the Pi (480 px wide, no
menu bar, no title bar) the strip is the first row of the header, beside a gear
button. :meth:`StatusStrip.set_touch_mode` drops what a finger cannot use and a
480 px row cannot afford -- the panel toggles and the ``Ctrl K`` hint -- shows
the dirty marker (a frameless window has no title bar to carry the modified
mark), and lets long names elide instead of widening the window. The strip also
grows from :data:`STRIP_HEIGHT` to 44 px, because the pill is a tap target and a
finger needs the room. It only flips visibility, a height, a minimum width and the
gaps between parts; ``set_touch_mode(False)`` puts every one of them back, so the
desktop strip is exactly what it was.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from PyQt6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QIcon,
    QIconEngine,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPalette,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from glider.gui.styles import restyle

__all__ = [
    "DEVICE_STATES",
    "RUN_STATE_TEXT",
    "STRIP_HEIGHT",
    "StatusStrip",
]

#: Height of the strip, in pixels. The budget the whole redesign is measured
#: against: one strip in place of a menu bar and two toolbars.
STRIP_HEIGHT = 40

#: Side of a device dot, in pixels. Even so ``desktop.qss`` can halve it for the
#: border radius and get a circle rather than a lozenge.
_DOT = 8

#: Logical size of a panel toggle's icon. ``desktop.qss`` boxes the button to
#: 24x22; this leaves a few pixels of margin on every side rather than
#: claiming the whole button, and is a request to Qt's layout, not a raster --
#: :class:`_SidebarGlyphEngine` paints fresh at whatever device pixel ratio
#: actually applies, so nothing here caps how crisp the icon can render.
_TOGGLE_ICON_SIZE = QSize(16, 14)

#: The run states, and the word the pill shows for each. Four distinct labels on
#: purpose -- a pill that reads the same in two states is decoration.
RUN_STATE_TEXT = {
    "idle": "Idle",
    "running": "Running",
    "recording": "Recording",
    "error": "Error",
}

#: The same words in touch mode, where the pill shares 480 px with five boards
#: and whatever font the platform falls back to. Only recording is shortened:
#: the mockup's "● REC" is the Pi's word for it, and the clock after it is what
#: the operator reads.
_TOUCH_RUN_STATE_TEXT = {**RUN_STATE_TEXT, "recording": "\u25cf REC"}

#: The device states this strip knows how to paint. Anything else becomes
#: ``"unknown"``; see the module docstring for why that is never ``"ok"``.
DEVICE_STATES = ("ok", "warn", "error", "unknown")

#: Shown when a session has no file yet, which is the state the Builder opens
#: in. A blank strip would read as broken rather than as new.
_UNNAMED = "Untitled"

#: The dirty marker. Its own label rather than part of the name, so the
#: stylesheet can grey it without greying the name and so a caller reading the
#: name back gets the name.
_DIRTY_MARK = "— edited"

#: The dirty marker in touch mode: a dot, because "— edited" costs a board's
#: worth of width on the Pi header and a frameless kiosk shows nothing else.
_TOUCH_DIRTY_MARK = "\u25cf"

#: Narrowest an eliding name may get in touch mode, in pixels. Deliberately tiny:
#: it is only the floor the layout can squeeze to, reached when the dirty marker,
#: a running pill and five boards all compete for 480 px, and the pill and marker
#: do not shrink. With room to spare the names take it. A device name may go all
#: the way to (almost) nothing, leaving its dot: how wide names are depends on the
#: platform's fallback font, and the dots are what have to stay.
_TOUCH_NAME_MIN = 32
_TOUCH_DEVICE_MIN = 1  # not 0: a zero minimum falls back to the text width

#: The strip's fixed height in touch mode: a 44 px tap target, to match the gear
#: button beside it.
_TOUCH_HEIGHT = 44

#: ``(side margin, gap between the strip's parts, gap between device chips)`` in
#: pixels, on desktop and in touch mode. Touch packs tighter because the same
#: worst case has to share 480 px with whatever styling the Pi's stylesheet adds.
_GAPS = (12, 12, 10)
_TOUCH_GAPS = (8, 6, 6)


def _fit_aspect(box: QRectF, ratio: float) -> QRectF:
    """*box*, inset on one axis so its width:height ratio is exactly *ratio*.

    The glyph keeps its own proportions regardless of the box the button
    hands it -- a slightly wider icon size or a slightly taller one both
    still draw the same shape, just centred differently within what they
    were given.
    """
    if box.width() / box.height() > ratio:
        target_width = box.height() * ratio
        inset = (box.width() - target_width) / 2
        return box.adjusted(inset, 0, -inset, 0)
    target_height = box.width() / ratio
    inset = (box.height() - target_height) / 2
    return box.adjusted(0, inset, 0, -inset)


class _SidebarGlyphEngine(QIconEngine):
    """Paints the sidebar-toggle glyph: a window with a bar on one side.

    A vector glyph rather than a shipped raster, so it is painted fresh at
    whatever size and device pixel ratio Qt actually requests -- the strip is
    only :data:`STRIP_HEIGHT` pixels tall, so this is drawn small, and a
    fixed-resolution pixmap stretched to a fractional device pixel ratio is
    exactly the kind of icon that smears.

    Its colour is read from *widget*'s palette at paint time rather than
    stored on the engine. ``desktop.qss`` sets ``color`` on
    ``QToolButton#statusStripToggle`` and Qt resolves that into the widget's
    palette at polish time; reading it live, on every paint, is what lets this
    icon follow a re-theme the same way the arrow it replaces always did,
    without this file naming a colour of its own.
    """

    #: The glyph's own width:height ratio. Independent of the button's actual
    #: box (24x22 in ``desktop.qss``) -- see :func:`_fit_aspect`.
    _ASPECT = 8 / 7

    #: Share of the glyph's width given to the bar. A third, not a half, so
    #: the shape reads as "a panel beside a window" and not a divided
    #: rectangle, matching the proportions in the reference images.
    _BAR_FRACTION = 0.34

    #: Corner radius and stroke width, both as a fraction of the glyph's own
    #: height so they scale with it instead of looking chunkier the smaller
    #: it is drawn.
    _RADIUS_FRACTION = 0.22
    _STROKE_FRACTION = 0.085

    def __init__(self, widget: QWidget, on_the_left: bool, filled: bool) -> None:
        super().__init__()
        self._widget = widget
        self._on_the_left = on_the_left
        self._filled = filled

    def clone(self) -> _SidebarGlyphEngine:
        return _SidebarGlyphEngine(self._widget, self._on_the_left, self._filled)

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        # QIconEngine's own default pixmap() is documented to rasterise onto a
        # transparent buffer by calling paint(), but on this Qt build it hands
        # back a fully opaque pixmap instead: two engines that genuinely paint
        # different shapes both come back as one solid block, indistinguishable
        # from each other despite being built from distinct icons. Overriding
        # it with exactly what the documentation promises -- transparent
        # buffer, then paint() -- is what QToolButton's own icon drawing
        # actually needs to show the shape rather than a blank swatch.
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
        color = self._widget.palette().color(QPalette.ColorRole.WindowText)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        box = _fit_aspect(QRectF(rect).adjusted(0.5, 0.5, -0.5, -0.5), self._ASPECT)
        radius = box.height() * self._RADIUS_FRACTION
        stroke = max(box.height() * self._STROKE_FRACTION, 1.0)

        pen = QPen(color)
        pen.setWidthF(stroke)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(box, radius, radius)

        bar_width = box.width() * self._BAR_FRACTION
        if self._on_the_left:
            bar = QRectF(box.left(), box.top(), bar_width, box.height())
            divider_x = bar.right()
        else:
            bar = QRectF(box.right() - bar_width, box.top(), bar_width, box.height())
            divider_x = bar.left()

        # Clip fill and divider to the outer shape, so the bar's own corners
        # follow the window's curve instead of squaring it off.
        clip = QPainterPath()
        clip.addRoundedRect(box, radius, radius)
        painter.setClipPath(clip)

        if self._filled:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRect(bar)
        else:
            painter.drawLine(QPointF(divider_x, box.top()), QPointF(divider_x, box.bottom()))

        painter.restore()


class _PanelToggle(QToolButton):
    """One panel's toggle: an icon that reflects the panel beside it.

    Args:
        side: ``"left"`` or ``"right"``. Published as a ``side`` property for
            the stylesheet, said in :meth:`accessibleName` for anyone not
            reading the strip's geometry, and what decides which edge of the
            icon the bar sits on.
        parent: Standard Qt parent.

    Checked means the panel is open, as before. What is new is that the icon
    is a small map of the window rather than a direction to press: a rounded
    rectangle with a bar on the side the panel actually lives on, filled
    while the panel is open and hollow -- an outline and a divider, no fill
    -- while it is collapsed.

    The refresh hangs off two Qt virtuals rather than off the ``toggled``
    signal, because neither route alone sees both ways the state moves:

    * ``toggled`` is out, because :meth:`StatusStrip.set_left_expanded` sets the
      button with its signals blocked -- it would miss the path the owner uses
      to echo a panel back, which is most of them.
    * :meth:`checkStateSet` catches every programmatic ``setChecked``, blocked
      signals included -- but *not* a click: ``QAbstractButtonPrivate::click``
      raises ``blockRefresh`` around the state change, and ``setChecked`` skips
      ``checkStateSet`` while it is up.
    * :meth:`nextCheckState` is what a click (and the space bar) goes through
      instead.

    Both, therefore. A button that refreshed only when something echoed back
    would show the wrong icon on any strip whose signal nobody has connected
    yet.
    """

    def __init__(self, side: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._side = "right" if side == "right" else "left"
        self.setObjectName("statusStripToggle")
        self.setProperty("side", self._side)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setIconSize(_TOGGLE_ICON_SIZE)
        on_the_left = self._side == "left"
        self._icon_open = QIcon(_SidebarGlyphEngine(self, on_the_left, filled=True))
        self._icon_collapsed = QIcon(_SidebarGlyphEngine(self, on_the_left, filled=False))
        self.setChecked(True)
        self._turn_arrow()

    def checkStateSet(self) -> None:  # noqa: N802 - Qt override
        super().checkStateSet()
        self._turn_arrow()

    def nextCheckState(self) -> None:  # noqa: N802 - Qt override
        super().nextCheckState()
        self._turn_arrow()

    def _turn_arrow(self) -> None:
        """Show the icon for the current checked state, and name it to match.

        Kept as one explicit push, called from both :meth:`checkStateSet` and
        :meth:`nextCheckState`, for the reason the class docstring gives: Qt
        already repaints this button on every path that changes its checked
        state, but only this method also knows which of the two pre-built
        icons (open, collapsed) that repaint should show, and updates
        :meth:`accessibleName` to match -- a screen reader needs the same
        open-or-collapsed fact a sighted user reads off the icon's shape.
        """
        expanded = self.isChecked()
        self.setIcon(self._icon_open if expanded else self._icon_collapsed)
        state_word = "expanded" if expanded else "collapsed"
        self.setAccessibleName(f"{self._side.capitalize()} panel, {state_word}")


class _ElidedLabel(QLabel):
    """A label that can paint "abc…" in place of text wider than it is.

    The text itself is never shortened, so :meth:`text` stays the full string for
    anyone reading it back. What changes is the layout's view of it: a QLabel
    refuses to be narrower than its text, so :meth:`set_elide_min` sets an
    explicit minimum width (which outranks the size hint) and starts painting the
    elided form; ``None`` undoes both.
    """

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._elide = False

    def set_elide_min(self, width: int | None) -> None:
        self._elide = width is not None
        self.setMinimumWidth(width or 0)
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 - Qt override
        if not self._elide:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        self.drawFrame(painter)
        area = self.contentsRect()
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, area.width())
        painter.drawText(area, int(self.alignment()), text)


def _as_device(device: Sequence[str]) -> tuple[str, str, str]:
    """``(name, state)`` or ``(name, state, detail)`` as a full triple.

    A caller that gives no detail is saying the state word *is* the raw text,
    which is true of anything that has not mapped a wider vocabulary onto these
    four states.
    """
    name, state, *rest = (str(part) for part in device)
    return name, state, rest[0] if rest else state


class StatusStrip(QFrame):
    """The always-present strip above the Builder's content surface.

    Args:
        parent: Standard Qt parent.

    Signals:
        left_toggled: The left panel's toggle was pressed. The strip does not
            act on it -- the owner collapses the panel and echoes the result
            back through :meth:`set_left_expanded`.
        right_toggled: As above, for the right panel.
        palette_requested: The ``Ctrl K`` hint was pressed. It is a button and
            not a decoration because a hint that teaches a shortcut and then
            refuses to perform it is worse than no hint.
    """

    left_toggled = pyqtSignal()
    right_toggled = pyqtSignal()
    palette_requested = pyqtSignal()
    #: The pill was tapped. Emitted in any mode; only the Pi header listens.
    pill_clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusStrip")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFixedHeight(STRIP_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._run_state = "idle"
        self._run_detail = ""
        self._touch = False
        self._is_dirty = False
        self._desktop_visible: dict[QWidget, bool] = {}
        self._device_names: list[str] = []
        self._chips: list[QFrame] = []
        self._dots: list[QLabel] = []
        self._names: list[QLabel] = []

        outer = self._outer = QHBoxLayout(self)

        self._left_toggle = self._build_toggle("left", self.left_toggled)
        outer.addWidget(self._left_toggle)

        self._name = _ElidedLabel(_UNNAMED, self)
        self._name.setObjectName("statusStripExperiment")
        outer.addWidget(self._name)

        # Built, wired and never shown. Both used to sit here permanently and
        # both were noise for the state the app is in almost all the time: the
        # marker said "edited" beside a name you had just edited, and the pill
        # said "Idle" beside an experiment that was, self-evidently, idle.
        # Kept as widgets rather than deleted because set_experiment and
        # set_run_state are called from a dozen places and still carry real
        # information -- see set_run_state, which shows the pill again the
        # moment the state stops being idle.
        self._dirty = QLabel(_DIRTY_MARK, self)
        self._dirty.setObjectName("statusStripDirty")
        self._dirty.setVisible(False)
        outer.addWidget(self._dirty)

        self._pill = QLabel(RUN_STATE_TEXT["idle"], self)
        self._pill.setObjectName("statusStripPill")
        self._pill.setProperty("state", "idle")
        self._pill.setVisible(False)
        self._pill.installEventFilter(self)
        outer.addWidget(self._pill)

        outer.addStretch(1)

        self._devices = QFrame(self)
        self._devices.setObjectName("statusStripDevices")
        self._devices.setFrameShape(QFrame.Shape.NoFrame)
        self._device_layout = QHBoxLayout(self._devices)
        self._device_layout.setContentsMargins(0, 0, 0, 0)
        self._set_gaps(_GAPS)
        outer.addWidget(self._devices)

        self._palette_hint = QToolButton(self)
        self._palette_hint.setObjectName("statusStripPaletteHint")
        native = QKeySequence("Ctrl+K").toString(QKeySequence.SequenceFormat.NativeText)
        self._palette_hint.setText(native.replace("+", " "))
        self._palette_hint.setToolTip(f"Command palette ({native})")
        self._palette_hint.setCursor(Qt.CursorShape.PointingHandCursor)
        self._palette_hint.clicked.connect(lambda _checked=False: self.palette_requested.emit())
        outer.addWidget(self._palette_hint)

        # Last, so it lands on the right edge: a control belongs beside the
        # thing it controls, and the right panel is the far side of the window.
        self._right_toggle = self._build_toggle("right", self.right_toggled)
        outer.addWidget(self._right_toggle)

    # ----------------------------------------------------------------- build

    def _build_toggle(self, side: str, signal) -> _PanelToggle:
        button = _PanelToggle(side, self)
        button.setToolTip(f"Show or hide the {side} panel")
        button.clicked.connect(lambda _checked=False: signal.emit())
        return button

    # ------------------------------------------------------------ touch mode

    def _set_gaps(self, gaps: tuple[int, int, int]) -> None:
        margin, gap, device_gap = gaps
        self._outer.setContentsMargins(margin, 0, margin, 0)
        self._outer.setSpacing(gap)
        self._device_layout.setSpacing(device_gap)

    def set_touch_mode(self, enabled: bool) -> None:
        """Fit the strip to the Pi header, or back to the desktop.

        Touch mode hides the panel toggles and the ``Ctrl K`` hint, shows the
        dirty marker, makes the strip 44 px tall, and lets the experiment name and
        the device names elide so a long name and five boards still fit in
        480 px. Reversible: ``False`` restores every one of them, the hidden
        controls to whatever visibility they had when touch mode was switched on.
        """
        enabled = bool(enabled)
        if enabled != self._touch:
            self._touch = enabled
            # Put back what the desktop had, not "shown": the window hides the
            # panel toggles on every page but the Dashboard, and forcing them
            # visible here would undo that on the way out.
            for widget in (self._left_toggle, self._right_toggle, self._palette_hint):
                if enabled:
                    self._desktop_visible[widget] = not widget.isHidden()
                    widget.setVisible(False)
                else:
                    widget.setVisible(self._desktop_visible.pop(widget, True))
        self._set_gaps(_TOUCH_GAPS if enabled else _GAPS)
        self.setFixedHeight(_TOUCH_HEIGHT if enabled else STRIP_HEIGHT)
        self._dirty.setText(_TOUCH_DIRTY_MARK if self._touch else _DIRTY_MARK)
        self._dirty.setVisible(self._touch and self._is_dirty)
        self.set_run_state(self._run_state, self._run_detail or None)
        self._name.set_elide_min(_TOUCH_NAME_MIN if self._touch else None)
        for label in self._names:
            label.set_elide_min(_TOUCH_DEVICE_MIN if self._touch else None)

    # ------------------------------------------------------------ experiment

    def set_experiment(self, name: str, dirty: bool = False) -> None:
        """Show *name*, with the dirty marker beside it when *dirty*.

        An empty name falls back to :data:`_UNNAMED` rather than blanking the
        strip: the Builder opens before any file exists, and that is a new
        session rather than a broken one.
        """
        self._name.setText(name.strip() or _UNNAMED)
        # Not shown on desktop. The marker is kept and kept up to date so
        # dirty_label() still reports honestly to anything that asks, but a
        # strip that appends "- edited" to a name the moment you type in it is
        # telling you something you already know. Nothing is lost: _check_save
        # still stops a close, a New or an Open over unsaved work. Touch mode
        # is the exception: a frameless kiosk window has no title-bar modified
        # mark, so this is the only place the Pi can say "unsaved".
        self._is_dirty = bool(dirty)
        self._dirty.setVisible(self._touch and self._is_dirty)

    def name_label(self) -> QLabel:
        """The label carrying the experiment name."""
        return self._name

    def dirty_label(self) -> QLabel:
        """The label carrying the dirty marker. Hidden while saved."""
        return self._dirty

    # -------------------------------------------------------------- run state

    def run_state(self) -> str:
        """The current run state."""
        return self._run_state

    def run_detail(self) -> str:
        """The text shown after the state word, or ``""``."""
        return self._run_detail

    def set_run_state(self, state: str, detail: str | None = None) -> None:
        """Move the pill to *state*, optionally followed by *detail*.

        Args:
            state: One of the keys of :data:`RUN_STATE_TEXT`.
            detail: Extra text rendered after the state word -- the mockup's
                elapsed clock, ``set_run_state("recording", "04:12")``, reads
                "Recording 04:12". **Nothing here runs a timer**: a widget that
                owned one would keep ticking after the run it describes had
                stopped, which is the same lie as a stale pill. Omitting it
                clears any previous detail, so last run's clock cannot survive
                into the next state.

        Raises:
            ValueError: *state* is not one of the four. Deliberately loud: run
                state comes from our own code, and a pill silently stuck on the
                previous state through a recording is the exact failure this
                widget exists to prevent.
        """
        if state not in RUN_STATE_TEXT:
            raise ValueError(
                f"unknown run state {state!r}; expected one of {sorted(RUN_STATE_TEXT)}"
            )
        self._run_state = state
        self._run_detail = str(detail).strip() if detail else ""
        text = (_TOUCH_RUN_STATE_TEXT if self._touch else RUN_STATE_TEXT)[state]
        self._pill.setText(f"{text} {self._run_detail}" if self._run_detail else text)
        self._pill.setProperty("state", state)
        # Shown only when it has something to say. Idle is the state the app is
        # in almost all of the time, and a pill that is always there stops
        # being read -- which is the failure mode for the one indicator that
        # has to be noticed when a run starts, or a board drops out of one at
        # 3am. Everything but idle appears.
        self._pill.setVisible(state != "idle")
        restyle(self._pill)

    def pill(self) -> QLabel:
        """The run-state pill."""
        return self._pill

    def eventFilter(self, watched: QObject | None, event: QEvent | None) -> bool:  # noqa: N802
        """Turn a left-button release over the pill into :attr:`pill_clicked`."""
        if (
            watched is self._pill
            and isinstance(event, QMouseEvent)
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
            and self._pill.rect().contains(event.position().toPoint())
        ):
            self.pill_clicked.emit()
        return super().eventFilter(watched, event)

    # ---------------------------------------------------------------- devices

    def set_devices(self, devices: Iterable[Sequence[str]]) -> None:
        """Show one dot, named, per device.

        Args:
            devices: ``(name, state)`` pairs -- or ``(name, state, detail)``
                triples -- in the order they should appear. A state outside
                :data:`DEVICE_STATES` renders neutral and keeps its raw text in
                the tooltip, never the healthy green. *detail* is what the
                tooltip says instead of the state word, and exists because a
                caller that maps its own vocabulary onto these four states would
                otherwise destroy the only text there was: ``warn`` cannot tell
                a handshake in flight from a board that has already dropped, and
                that difference is the whole reason anyone hovers a dot.

        When the names are unchanged the existing widgets are updated in place
        and only the dots whose state actually moved are re-polished. That keeps
        a strip refreshed on a timer from churning every widget on it, and it is
        what makes "only that device changed" a claim anyone can check.
        """
        items = [_as_device(device) for device in devices]
        names = [name for name, _, _ in items]

        if names != self._device_names:
            self._rebuild(items)
            return

        for chip, dot, item in zip(self._chips, self._dots, items, strict=True):
            self._apply_device(chip, dot, *item)

    def _rebuild(self, items: list[tuple[str, str, str]]) -> None:
        for chip in self._chips:
            self._device_layout.removeWidget(chip)
            chip.setParent(None)
            chip.deleteLater()
        self._chips, self._dots, self._names = [], [], []

        for name, raw, detail in items:
            chip = QFrame(self._devices)
            chip.setObjectName("statusStripDevice")
            chip.setFrameShape(QFrame.Shape.NoFrame)
            layout = QHBoxLayout(chip)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)

            dot = QLabel(chip)
            dot.setObjectName("statusStripDot")
            dot.setFixedSize(_DOT, _DOT)
            layout.addWidget(dot)

            label = _ElidedLabel(name, chip)
            label.setObjectName("statusStripDeviceName")
            if self._touch:
                label.set_elide_min(_TOUCH_DEVICE_MIN)
            layout.addWidget(label)

            self._apply_device(chip, dot, name, raw, detail)
            self._device_layout.addWidget(chip)
            self._chips.append(chip)
            self._dots.append(dot)
            self._names.append(label)

        self._device_names = [name for name, _, _ in items]
        # An empty rig is the normal first thing this widget is told, and an
        # empty container would still claim its spacing beside the hint.
        self._devices.setVisible(bool(items))

    def _apply_device(self, chip: QFrame, dot: QLabel, name: str, raw: str, detail: str) -> None:
        state = raw if raw in DEVICE_STATES else "unknown"
        # Set before the early return: a device can stay amber while what made
        # it amber changes, and that change is exactly what the tooltip is for.
        chip.setToolTip(f"{name} — {detail}")
        if dot.property("state") == state:
            return
        for widget in (chip, dot):
            widget.setProperty("state", state)
            restyle(widget)

    def device_names(self) -> list[str]:
        """The device names as rendered, in order."""
        return [label.text() for label in self._names]

    def device_dots(self) -> list[QLabel]:
        """The device dots, in order."""
        return list(self._dots)

    def device_chips(self) -> list[QFrame]:
        """The dot-and-name pairs, in order."""
        return list(self._chips)

    # ---------------------------------------------------------------- toggles

    def set_left_expanded(self, expanded: bool) -> None:
        """Show the left panel as open or collapsed, without emitting."""
        self._set_toggle(self._left_toggle, expanded)

    def set_right_expanded(self, expanded: bool) -> None:
        """Show the right panel as open or collapsed, without emitting."""
        self._set_toggle(self._right_toggle, expanded)

    @staticmethod
    def _set_toggle(button: QToolButton, expanded: bool) -> None:
        blocked = button.blockSignals(True)
        button.setChecked(bool(expanded))
        button.blockSignals(blocked)

    def left_toggle(self) -> QToolButton:
        """The left panel's toggle button."""
        return self._left_toggle

    def right_toggle(self) -> QToolButton:
        """The right panel's toggle button."""
        return self._right_toggle

    def palette_hint(self) -> QToolButton:
        """The ``Ctrl K`` hint, which is a button."""
        return self._palette_hint
