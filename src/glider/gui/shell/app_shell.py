"""The Builder frame: one strip, one panel each side, one content surface.

This module composes :class:`~glider.gui.shell.status_strip.StatusStrip` and
two :class:`~glider.gui.shell.side_panel.SidePanel` instances around a centre
widget it is handed. **The centre is a constructor argument and the shell never
inspects it** -- that is what keeps the frame testable without a node graph, a
``GliderCore`` or a ``MainWindow``, and what will let the same frame host
something else later.

Four things here are less obvious than they look:

* **A ``SidePanel`` cannot keep its own width inside a ``QSplitter``.** The
  panel's ``setFixedWidth`` wins on collapse, because a splitter honours a
  min == max child. On *expand* it does not: the splitter re-imposes the sizes
  it last handed out, so a panel the user dragged to 320 px silently comes back
  at whatever the splitter felt like. :meth:`AppShell._apply_splitter_sizes`
  exists for exactly this, and reads
  :meth:`~glider.gui.shell.side_panel.SidePanel.expanded_width` -- which
  reports the live width while expanded and the remembered one while
  collapsed -- rather than trusting the splitter.

* **The strip's toggles are wired in both directions.** Pressing one collapses
  a panel; a panel changing state moves the button back. Without the return
  path the buttons desynchronise the first time a panel collapses by any other
  route -- a rail click, or a restored layout -- and a button describing a
  state that is not on screen is worse than no button.

* **A long experiment name is a layout hazard, and the fix belongs here.** A
  ``QLabel`` reports its whole text as its minimum width, and that minimum
  propagates out through the strip to the window: an experiment named after its
  protocol can demand ~450 px under the shipped stylesheet and nearly 1 000 px
  under a default font, i.e. a window that cannot be made narrow enough for a
  laptop. Eliding in the strip was rejected because it would corrupt what
  ``name_label().text()`` reads back, so the cap is structural --
  :data:`NAME_CAP` on the label's *contribution*, with the full text intact
  underneath. The Builder's minimum width is the frame's business, not the
  strip's.

* **Geometry is stored as plain numbers, not ``saveGeometry()``.** Qt's
  ``restoreGeometry`` quietly relocates an off-screen window itself, in a way
  this module cannot inspect or predict, which makes "geometry ignored, window
  centred" (spec §10 -- a rig whose second monitor is not always plugged in)
  neither implementable nor testable. Four integers are validated before they
  are applied.

**Failure posture matches the vocabulary store**: every value read back from
:class:`QSettings` is checked, a malformed or partial one falls back to the
default and logs, and nothing here raises. A settings file must never be able
to stop the app starting.

**No colour is set from Python here**, following ``side_panel.py`` and
``status_strip.py``: every part carries an ``objectName`` and ``desktop.qss``
owns the rest.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QFrame,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from glider.gui.shell.side_panel import (
    DEFAULT_WIDTH,
    MIN_EXPANDED_WIDTH,
    RAIL_WIDTH,
    SidePanel,
)
from glider.gui.shell.status_strip import StatusStrip

__all__ = ["MIN_CENTRE_WIDTH", "NAME_CAP", "SETTINGS_PREFIX", "AppShell"]

logger = logging.getLogger(__name__)

#: Every key this module reads or writes lives under this ``QSettings`` group.
SETTINGS_PREFIX = "shell"

#: The narrowest the content surface may be squeezed to by two expanded panels.
#: Below this the centre stops being a work surface and the panels should give
#: way instead.
MIN_CENTRE_WIDTH = 240

#: The widest an experiment name may *demand* of the window, in pixels. The
#: label still holds the whole name; this caps what it forces on everything
#: else. Roughly 40 characters at the shipped font -- long enough to read a
#: real protocol name, short enough that a laptop can still show the Builder.
NAME_CAP = 280

#: Splitter indices. Named because the centre is replaced by index.
_LEFT, _CENTRE, _RIGHT = 0, 1, 2


class AppShell(QWidget):
    """A status strip above a ``SidePanel | centre | SidePanel`` splitter.

    Args:
        centre: The content surface. Any widget: the shell hosts it and never
            inspects it. ``None`` gives an empty placeholder, replaceable later
            with :meth:`set_centre`.
        parent: Standard Qt parent.

    The shell does not populate its panels -- the owner calls
    ``shell.left.add_tab(...)``. It also does not save its own layout on close:
    it is a page of the Builder's stack, not a window, so the window that owns
    it calls :meth:`save_layout` when it closes.
    """

    def __init__(self, centre: QWidget | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("appShell")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._strip = StatusStrip(self)
        outer.addWidget(self._strip)

        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._splitter.setObjectName("appShellSplitter")
        self._splitter.setChildrenCollapsible(False)
        outer.addWidget(self._splitter, 1)

        self._left = SidePanel(side="left", parent=self._splitter)
        self._centre_host = self._build_centre_host()
        self._right = SidePanel(side="right", parent=self._splitter)

        self._splitter.addWidget(self._left)
        self._splitter.addWidget(self._centre_host)
        self._splitter.addWidget(self._right)
        # Panels keep the width they were given when the window resizes; the
        # centre absorbs the difference. Without this the splitter scales all
        # three proportionally and a dragged panel width drifts on every
        # resize.
        self._splitter.setStretchFactor(_LEFT, 0)
        self._splitter.setStretchFactor(_CENTRE, 1)
        self._splitter.setStretchFactor(_RIGHT, 0)

        self._centre: QWidget | None = None
        self.set_centre(centre if centre is not None else QWidget())

        self._cap_experiment_name()
        self._wire_toggles()

    # ----------------------------------------------------------------- build

    def _build_centre_host(self) -> QFrame:
        """A fixed splitter child holding whatever the centre currently is.

        The host keeps the splitter's three indices stable, so replacing the
        centre never touches the splitter and never disturbs the panel sizes
        either side of it.
        """
        host = QFrame(self._splitter)
        host.setObjectName("appShellCentre")
        host.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._centre_layout = layout
        host.setMinimumWidth(MIN_CENTRE_WIDTH)
        return host

    def _cap_experiment_name(self) -> None:
        """Stop a long experiment name widening the whole window.

        See the module docstring: the label keeps its full text, but only
        :data:`NAME_CAP` pixels of it are allowed to become a demand on the
        Builder's minimum width.
        """
        self._strip.name_label().setMaximumWidth(NAME_CAP)

    def _wire_toggles(self) -> None:
        """Connect each toggle to its panel, **and each panel back**.

        The return path is what keeps a button honest when its panel is
        collapsed by some other route.
        """
        self._strip.left_toggled.connect(self._left.toggle)
        self._strip.right_toggled.connect(self._right.toggle)
        self._left.expanded_changed.connect(self._on_left_expanded)
        self._right.expanded_changed.connect(self._on_right_expanded)
        self._strip.set_left_expanded(self._left.expanded)
        self._strip.set_right_expanded(self._right.expanded)

    # ------------------------------------------------------------- the pieces

    @property
    def strip(self) -> StatusStrip:
        """The always-present status strip."""
        return self._strip

    @property
    def left(self) -> SidePanel:
        """The left panel. Populated by the owner, not by the shell."""
        return self._left

    @property
    def right(self) -> SidePanel:
        """The right panel."""
        return self._right

    @property
    def splitter(self) -> QSplitter:
        """The ``panel | centre | panel`` splitter."""
        return self._splitter

    @property
    def centre(self) -> QWidget:
        """The content surface currently hosted."""
        return self._centre  # type: ignore[return-value]

    def set_centre(self, widget: QWidget) -> None:
        """Host *widget* as the content surface, destroying the previous one.

        The shell owns what it hosts, so the outgoing widget is scheduled for
        deletion rather than left parentless -- an orphaned top-level widget
        that nothing holds is the leak this replaces. A caller that wants to
        keep the old surface must take a reference and reparent it *before*
        calling this.
        """
        previous = self._centre
        if previous is widget:
            return
        if previous is not None:
            self._centre_layout.removeWidget(previous)
            previous.setParent(None)
            previous.deleteLater()
        self._centre = widget
        self._centre_layout.addWidget(widget)

    # ----------------------------------------------------------- panel widths

    def _on_left_expanded(self, expanded: bool) -> None:
        self._strip.set_left_expanded(expanded)
        self._apply_splitter_sizes()

    def _on_right_expanded(self, expanded: bool) -> None:
        self._strip.set_right_expanded(expanded)
        self._apply_splitter_sizes()

    def _apply_splitter_sizes(self) -> None:
        """Re-impose the panels' own widths on the splitter.

        The splitter would otherwise hand an expanding panel back whatever it
        last allocated, discarding the width the user dragged. This is the one
        place that knows the panels' widths are theirs and not the splitter's.
        """
        total = self._splitter.width()
        if total <= 0:
            return  # not laid out yet; showEvent will come back to this
        available = max(total - self._splitter.handleWidth() * 2, 0)
        left = self._left.expanded_width() if self._left.expanded else RAIL_WIDTH
        right = self._right.expanded_width() if self._right.expanded else RAIL_WIDTH
        centre = max(available - left - right, MIN_CENTRE_WIDTH)
        self._splitter.setSizes([left, centre, right])

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Apply the panel widths once the splitter finally has a size.

        A layout restored before the window is shown has no splitter to act
        on; this is where it lands.
        """
        super().showEvent(event)
        self._apply_splitter_sizes()

    # ----------------------------------------------------------- persistence

    def save_layout(self, settings: QSettings) -> None:
        """Write the current layout under the ``shell/`` prefix.

        Never raises: this runs while the window is closing, and a settings
        file that cannot be written is not a reason to fail a shutdown.
        """
        try:
            window = self.window()
            rect = window.geometry()
            geometry = f"{rect.x()},{rect.y()},{rect.width()},{rect.height()}"
            values = {
                "left_expanded": self._left.expanded,
                "right_expanded": self._right.expanded,
                "left_tab": self._left.current_key(),
                "right_tab": self._right.current_key(),
                "left_width": self._left.expanded_width(),
                "right_width": self._right.expanded_width(),
                "geometry": geometry,
            }
            for key, value in values.items():
                settings.setValue(f"{SETTINGS_PREFIX}/{key}", value)
            settings.sync()
        except Exception:  # pragma: no cover - defensive, see docstring
            logger.warning("Could not save the shell layout", exc_info=True)

    def restore_layout(self, settings: QSettings) -> None:
        """Apply a previously saved layout, falling back value by value.

        Every value is checked independently, so a partial or corrupt settings
        file restores the parts it got right and defaults the rest. Nothing
        here raises.
        """
        try:
            self._restore_tabs(settings)
            self._restore_sides(settings)
            self._restore_geometry(settings)
        except Exception:  # pragma: no cover - defensive, see docstring
            logger.warning("Could not restore the shell layout; using defaults", exc_info=True)

    def _restore_tabs(self, settings: QSettings) -> None:
        for side, panel in (("left", self._left), ("right", self._right)):
            keys = panel.keys()
            if not keys:
                continue  # a panel whose tabs failed to build still has to start
            raw = settings.value(f"{SETTINGS_PREFIX}/{side}_tab")
            if raw in (None, ""):
                continue
            key = str(raw)
            # SidePanel.set_current refuses an unknown key on purpose, so the
            # fallback happens here, where the default is known.
            if key not in keys:
                logger.warning(
                    "Saved %s panel tab %r no longer exists; showing %r instead",
                    side,
                    key,
                    keys[0],
                )
                key = keys[0]
            panel.set_current(key)

    def _restore_sides(self, settings: QSettings) -> None:
        for side, panel in (("left", self._left), ("right", self._right)):
            width = _as_int(
                settings.value(f"{SETTINGS_PREFIX}/{side}_width"),
                DEFAULT_WIDTH,
                f"{side} panel width",
            )
            width = max(width, MIN_EXPANDED_WIDTH)
            expanded = _as_bool(
                settings.value(f"{SETTINGS_PREFIX}/{side}_expanded"),
                True,
                f"{side} panel state",
            )
            # Resize while expanded and *then* collapse: SidePanel records the
            # width it is at when it collapses, which is what it will come back
            # to. There is no public setter for that width, and there should
            # not be one -- it is the user's, not ours.
            if panel.expanded:
                panel.resize(width, panel.height())
            panel.set_expanded(expanded)
        self._apply_splitter_sizes()

    def _restore_geometry(self, settings: QSettings) -> None:
        raw = settings.value(f"{SETTINGS_PREFIX}/geometry")
        if raw in (None, ""):
            return
        window = self.window()
        rect = _as_rect(raw)
        if rect is None:
            logger.warning("Ignoring malformed saved window geometry %r", raw)
            self._centre_window(window)
            return
        if not _on_a_screen(rect):
            # Real on a rig whose second monitor is not always plugged in.
            logger.warning(
                "Saved window geometry %s is off every available screen; centring instead",
                rect,
            )
            self._centre_window(window)
            return
        window.setGeometry(*rect)

    @staticmethod
    def _centre_window(window: QWidget) -> None:
        """Put *window* in the middle of the primary screen, at its own size."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover - a headless run with no screen
            return
        available = screen.availableGeometry()
        size = window.size()
        window.move(
            available.x() + (available.width() - size.width()) // 2,
            available.y() + (available.height() - size.height()) // 2,
        )


# --------------------------------------------------------------- value parsing


def _as_bool(raw: object, default: bool, what: str) -> bool:
    """A saved flag, or *default* with a warning if it is not one.

    A missing key is not a warning -- that is simply a layout nobody has saved
    yet, which is every first launch.
    """
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw)
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("true", "1", "yes", "on"):
            return True
        if text in ("false", "0", "no", "off"):
            return False
    logger.warning("Ignoring malformed saved %s %r; using %r", what, raw, default)
    return default


def _as_int(raw: object, default: int, what: str) -> int:
    """A saved number, or *default* with a warning if it is not one."""
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Ignoring malformed saved %s %r; using %r", what, raw, default)
        return default


def _as_rect(raw: object) -> tuple[int, int, int, int] | None:
    """``"x,y,w,h"`` as four integers, or ``None`` if it is not that.

    A rectangle with no area is treated as malformed rather than obeyed: a
    zero-width window is indistinguishable from a missing one to the user.
    """
    parts = str(raw).split(",")
    if len(parts) != 4:
        return None
    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _on_a_screen(rect: tuple[int, int, int, int]) -> bool:
    """Whether *rect* overlaps any screen currently attached."""
    x, y, width, height = rect
    for screen in QGuiApplication.screens():
        available = screen.availableGeometry()
        if (
            x < available.x() + available.width()
            and x + width > available.x()
            and y < available.y() + available.height()
            and y + height > available.y()
        ):
            return True
    return False
