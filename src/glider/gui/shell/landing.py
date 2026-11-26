"""The landing page: what GLIDER shows before there is an experiment to show.

Until now the app opened straight onto an empty node graph belonging to an
unnamed, unsaved session. That is a fine state for the person who built the
app and a confusing one for everybody else: nothing on screen says whether you
are looking at a new experiment, the one you had open yesterday, or a blank
canvas that will be thrown away. This page is the answer to "what am I looking
at" -- you are looking at a chooser, and nothing is open yet.

It is a page of ``MainWindow``'s stack, not a window. That is what lets the
navigation tabs stay hidden while it is up (there is nothing for them to
navigate between yet) and what lets *Close Experiment* come back here without
tearing down and rebuilding the window.

**The recent list is stored, pruned and capped here** rather than in
``MainWindow``, because this is the only thing that reads it.
:func:`remember_experiment` is called by the window on every successful open and
save-as. Two properties matter and are easy to lose:

* **A path that cannot be reached is hidden on read, never dropped on write.**
  Files move, external drives get unplugged, shares go offline, and a recent
  list whose entries fail when clicked is worse than a short one. Filtering on
  read means the list is correct whenever it is looked at, without a
  filesystem watcher -- and an entry on a share that is down today comes back
  when the share does.
* **Reading the list can neither raise nor hang.** It runs while the main
  window is being built, so an ``OSError`` from a revoked share used to stop
  GLIDER launching, and a ``stat`` on a dead server can block for tens of
  seconds. Each entry is checked on its own thread under one shared deadline.
* **Order is most-recent-first with no duplicates**, so re-opening the same
  experiment moves it to the top rather than adding a second row.

**No colour is set from Python here**, following the rest of ``shell/``: every
part carries an ``objectName`` and ``desktop.qss`` owns the appearance.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from glider.gui.panels.tools_page import ToolCard, ToolsPage

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_RECENT",
    "PROBE_TIMEOUT_S",
    "RECENT_KEY",
    "STACK_BELOW_PX",
    "LandingPage",
    "forget_experiment",
    "recent_experiments",
    "remember_experiment",
]

#: ``QSettings`` key holding the recent-experiment paths, newest first.
RECENT_KEY = "landing/recent"

#: How many entries the list keeps. Long enough to cover the handful of
#: protocols a rig actually cycles through, short enough that the page stays a
#: chooser rather than becoming a file manager.
MAX_RECENT = 8

#: How long the landing page will wait, in total, to learn which recent entries
#: still exist. They are checked in parallel, so this bounds the whole list, not
#: each entry: one dead network share must not hold the window back.
PROBE_TIMEOUT_S = 1.5

#: Logo edge length in logical pixels on the landing page.
_LOGO_PX = 96

#: Below this width the tools column drops beneath the start card instead of
#: sitting beside it, so a narrow window scrolls down rather than sideways.
STACK_BELOW_PX = 980


# --------------------------------------------------------------- recent list


def _stored_paths(settings: QSettings) -> list[str]:
    """The raw stored list, de-duplicated, with no filesystem access at all.

    Never raises: a malformed or missing value reads as empty, exactly as a
    first launch does.
    """
    try:
        raw = settings.value(RECENT_KEY, [], type=list) or []
    except Exception:  # pragma: no cover - depends on the settings backend
        logger.debug("Recent experiment list could not be read", exc_info=True)
        return []
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        text = str(entry).strip()
        if not text:
            continue
        key = _identity(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _identity(path: str | Path) -> str:
    """A comparison key for a path that touches nothing on disk.

    ``abspath`` + ``normcase`` rather than ``Path.resolve``: resolving a path
    on a share that has gone away is itself a network round trip, and the
    whole point here is to never block on one.
    """
    return os.path.normcase(os.path.abspath(str(path)))


def _exists(path: Path) -> bool:
    """``path.exists()``, but ``False`` instead of raising.

    ``Path.exists`` only swallows the "not found" family of errors. A share
    you have lost access to raises ``PermissionError``; a server that has gone
    raises other ``OSError`` codes. Unhandled, that took down the landing page
    and with it GLIDER's start-up.
    """
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _existing(paths: list[Path], timeout: float) -> list[Path]:
    """The members of ``paths`` that exist, checked in parallel under a deadline.

    Each check runs on a daemon thread because a ``stat`` on an unreachable
    network path can block for tens of seconds, and does so on the GUI thread
    during start-up. A check that has not answered by the deadline counts as
    missing for this read only -- nothing is written back, so the entry
    reappears once its drive does. Daemon threads, not an executor: an
    executor's workers are joined at interpreter exit, so one hung ``stat``
    would then stop GLIDER from closing.
    """
    found: dict[int, bool] = {}
    lock = threading.Lock()

    def _check(index: int, path: Path) -> None:
        result = _exists(path)
        with lock:
            found[index] = result

    threads = [
        threading.Thread(target=_check, args=(i, p), daemon=True, name="glider-recent-probe")
        for i, p in enumerate(paths)
    ]
    for thread in threads:
        thread.start()
    deadline = time.perf_counter() + timeout
    for thread in threads:
        thread.join(max(0.0, deadline - time.perf_counter()))
    with lock:
        return [p for i, p in enumerate(paths) if found.get(i, False)]


def recent_experiments(
    settings: QSettings | None = None, *, timeout: float = PROBE_TIMEOUT_S
) -> list[Path]:
    """The recent experiments that can be reached right now, newest first.

    Never raises and never blocks for longer than ``timeout``: a recent entry
    on a removed file, an unplugged drive or a share you no longer have access
    to is simply left off the list, and must not be able to stop the landing
    page -- and so GLIDER -- from starting.
    """
    s = settings if settings is not None else QSettings()
    candidates: list[Path] = []
    for text in _stored_paths(s)[:MAX_RECENT]:
        try:
            candidates.append(Path(text))
        except Exception:  # pragma: no cover - Path() of a str barely raises
            continue
    return _existing(candidates, timeout)


def remember_experiment(path: str | Path, settings: QSettings | None = None) -> None:
    """Move ``path`` to the top of the recent list, de-duplicated and capped.

    Works on the stored list, not the reachable one: an entry whose share is
    offline today keeps its place instead of being forgotten for good the next
    time anything is saved.
    """
    s = settings if settings is not None else QSettings()
    target = os.path.abspath(str(path))
    key = _identity(target)
    kept = [p for p in _stored_paths(s) if _identity(p) != key]
    s.setValue(RECENT_KEY, [target, *kept][:MAX_RECENT])


def forget_experiment(path: str | Path, settings: QSettings | None = None) -> None:
    """Drop ``path`` from the recent list. Missing entries are not an error."""
    s = settings if settings is not None else QSettings()
    key = _identity(path)
    s.setValue(RECENT_KEY, [p for p in _stored_paths(s) if _identity(p) != key])


# ---------------------------------------------------------------- the widget


class LandingPage(QWidget):
    """Start card on the left -- logo, two primary actions, the recent list --
    and, when given tool cards, the tool launchers on the right.

    The tools are here because none of them needs an experiment: Session
    Review, Batch Pose Tracking and the rest read and write files on disk. With
    the Tools tab hidden on this page and the Tools menu off the bar, they used
    to be reachable only by first creating an experiment nobody wanted.

    Args:
        settings: Injectable so a test never reads the developer's real recent
            list -- and, more sharply, so a test's writes never land in it.
        parent: Standard Qt parent.
        tool_cards: The tools to offer beside the start card. ``None`` or empty
            gives the start card alone.

    Signals:
        new_requested: *New Experiment* pressed.
        open_requested: *Open Experiment…* pressed.
        recent_requested: A recent row pressed, with its path as a string.
        tool_chosen: An available tool card pressed, with its key.
        install_requested: A tool card's Install pressed, with the extra.
    """

    new_requested = pyqtSignal()
    open_requested = pyqtSignal()
    recent_requested = pyqtSignal(str)
    tool_chosen = pyqtSignal(str)
    install_requested = pyqtSignal(str)

    def __init__(
        self,
        settings: QSettings | None = None,
        parent: QWidget | None = None,
        *,
        tool_cards: list[ToolCard] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("landingPage")
        self._settings = settings if settings is not None else QSettings()
        self._recent_buttons: list[QToolButton] = []
        self._tools: ToolsPage | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The whole card is inside a scroll area so a short window (the Pi's
        # 480px, a laptop with the dock up) scrolls rather than clipping the
        # buttons off the bottom, where they cannot be reached at all.
        scroll = QScrollArea(self)
        scroll.setObjectName("landingScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        host = QWidget()
        host.setObjectName("landingHost")
        column = QVBoxLayout(host)
        column.setContentsMargins(40, 48, 40, 48)
        column.setSpacing(0)
        column.addStretch(1)

        card = self._build_card()
        # One box whose direction flips with the window width: side by side
        # when there is room, stacked when there is not (see resizeEvent).
        self._columns = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self._columns.setSpacing(28)
        self._columns.addStretch(1)
        self._columns.addWidget(card, 0, Qt.AlignmentFlag.AlignTop)
        if tool_cards:
            tools_column = self._build_tools_column(list(tool_cards))
            self._columns.addWidget(tools_column, 0, Qt.AlignmentFlag.AlignTop)
        self._columns.addStretch(1)
        column.addLayout(self._columns)

        column.addStretch(2)
        scroll.setWidget(host)

        self.refresh_recent()

    # ------------------------------------------------------------------ build

    def _build_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("landingCard")
        # A floor as well as a ceiling. Without the floor the card shrinks to
        # its widest child -- the buttons -- and the subtitle wraps to three
        # lines inside a column narrower than the words in it.
        card.setMinimumWidth(420)
        card.setMaximumWidth(560)
        body = QVBoxLayout(card)
        body.setContentsMargins(36, 32, 36, 32)
        body.setSpacing(0)

        logo = QLabel()
        logo.setObjectName("landingLogo")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = self._logo_pixmap()
        if pixmap is not None:
            logo.setPixmap(pixmap)
        body.addWidget(logo)
        body.addSpacing(16)

        title = QLabel("GLIDER")
        title.setObjectName("landingTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(title)
        body.addSpacing(28)

        self._new_button = QPushButton("New Experiment")
        self._new_button.setObjectName("landingPrimary")
        self._new_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._new_button.clicked.connect(self.new_requested.emit)
        body.addWidget(self._new_button)
        body.addSpacing(8)

        self._open_button = QPushButton("Open Experiment…")
        self._open_button.setObjectName("landingSecondary")
        self._open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._open_button.clicked.connect(self.open_requested.emit)
        body.addWidget(self._open_button)

        body.addSpacing(24)

        self._recent_heading = QLabel("Recent")
        self._recent_heading.setObjectName("landingSectionLabel")
        body.addWidget(self._recent_heading)
        body.addSpacing(4)

        self._recent_host = QWidget()
        self._recent_layout = QVBoxLayout(self._recent_host)
        self._recent_layout.setContentsMargins(0, 0, 0, 0)
        self._recent_layout.setSpacing(2)
        body.addWidget(self._recent_host)

        body.addStretch(1)

        return card

    def _build_tools_column(self, cards: list[ToolCard]) -> QWidget:
        """The right-hand column: the tool cards."""
        host = QFrame()
        host.setObjectName("landingTools")
        host.setMinimumWidth(420)
        host.setMaximumWidth(620)
        body = QVBoxLayout(host)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Imported here, not at the top: ``glider.gui.panels`` pulls the camera
        # panel and the vision stack with it, which this module must not.
        from glider.gui.panels.tools_page import ToolsPage

        self._tools = ToolsPage(
            cards,
            heading="",
            subheading="",
            embedded=True,
            plain=True,
        )
        self._tools.tool_chosen.connect(self.tool_chosen.emit)
        self._tools.install_requested.connect(self.install_requested.emit)
        body.addWidget(self._tools)
        return host

    # ------------------------------------------------------------- layout

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_direction(event.size().width())

    def _apply_direction(self, width: int) -> None:
        """Side by side when the window is wide enough, stacked when not."""
        stacked = width < STACK_BELOW_PX
        direction = (
            QBoxLayout.Direction.TopToBottom if stacked else QBoxLayout.Direction.LeftToRight
        )
        if self._columns.direction() != direction:
            self._columns.setDirection(direction)
        # Side by side, no alignment: both columns then take the row's full
        # height, so the start card and the tool list end on the same line.
        align = Qt.AlignmentFlag.AlignHCenter if stacked else Qt.AlignmentFlag(0)
        for i in range(self._columns.count()):
            item = self._columns.itemAt(i)
            if item.widget() is not None:
                item.setAlignment(align)
        self._columns.invalidate()

    def is_stacked(self) -> bool:
        """Whether the tools column sits below the start card. For tests."""
        return self._columns.direction() == QBoxLayout.Direction.TopToBottom

    def tools(self) -> ToolsPage | None:
        """The embedded tool cards, if this page was given any."""
        return self._tools

    @staticmethod
    def _logo_pixmap() -> QPixmap | None:
        """The app icon at landing size, or ``None`` if it will not load.

        Best-effort in the same way :mod:`glider.gui.splash` is: a missing
        asset costs the page its logo and nothing else.
        """
        try:
            from glider.assets import get_icon_path

            source = QPixmap(str(get_icon_path(256)))
            if source.isNull():
                return None
            return source.scaled(
                _LOGO_PX,
                _LOGO_PX,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        except Exception:  # pragma: no cover - cosmetic
            logger.debug("Landing logo could not be loaded", exc_info=True)
            return None

    # ----------------------------------------------------------------- recent

    def refresh_recent(self) -> None:
        """Rebuild the recent rows from settings.

        Called on construction and every time the window opens or saves an
        experiment. The whole list is rebuilt rather than patched: it is at most
        :data:`MAX_RECENT` rows, and a patch would have to reproduce the
        de-duplication and pruning that :func:`recent_experiments` already does.
        """
        while self._recent_layout.count():
            item = self._recent_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._recent_buttons.clear()

        entries = recent_experiments(self._settings)
        if not entries:
            empty = QLabel("No experiments yet — start a new one above.")
            empty.setObjectName("landingEmpty")
            empty.setWordWrap(True)
            self._recent_layout.addWidget(empty)
            return

        for path in entries:
            button = QToolButton()
            button.setObjectName("landingRecent")
            button.setText(path.stem)
            # The full path is the tooltip, not the label: two protocols with
            # the same stem in different directories are common, and a row that
            # cannot tell you which one it is would be a trap.
            button.setToolTip(str(path))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.clicked.connect(lambda _checked, p=str(path): self.recent_requested.emit(p))
            self._recent_layout.addWidget(button)
            self._recent_buttons.append(button)

    def recent_buttons(self) -> list[QToolButton]:
        """The recent rows, for tests."""
        return list(self._recent_buttons)
