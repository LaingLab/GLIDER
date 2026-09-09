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

* **A path that no longer resolves is dropped on read, not on write.** Files
  move, external drives get unplugged, and a recent list whose entries fail
  when clicked is worse than a short one. Pruning on read means the list is
  correct whenever it is looked at, without a filesystem watcher.
* **Order is most-recent-first with no duplicates**, so re-opening the same
  experiment moves it to the top rather than adding a second row.

**No colour is set from Python here**, following the rest of ``shell/``: every
part carries an ``objectName`` and ``desktop.qss`` owns the appearance.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_RECENT",
    "RECENT_KEY",
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

#: Logo edge length in logical pixels on the landing page.
_LOGO_PX = 96


# --------------------------------------------------------------- recent list


def recent_experiments(settings: QSettings | None = None) -> list[Path]:
    """The recent experiments that still exist on disk, newest first.

    Never raises: a malformed or missing settings value reads as an empty list,
    exactly as a first launch does. A settings file must not be able to stop
    the landing page from drawing.
    """
    s = settings if settings is not None else QSettings()
    try:
        raw = s.value(RECENT_KEY, [], type=list) or []
    except Exception:  # pragma: no cover - depends on the settings backend
        logger.debug("Recent experiment list could not be read", exc_info=True)
        return []

    out: list[Path] = []
    for entry in raw:
        try:
            path = Path(str(entry))
        except Exception:
            continue
        if path in out:
            continue
        if path.exists():
            out.append(path)
    return out[:MAX_RECENT]


def remember_experiment(path: str | Path, settings: QSettings | None = None) -> None:
    """Move ``path`` to the top of the recent list, de-duplicated and capped."""
    s = settings if settings is not None else QSettings()
    try:
        target = Path(path).resolve()
    except Exception:  # pragma: no cover - unresolvable path
        return
    kept = [p for p in recent_experiments(s) if p.resolve() != target]
    s.setValue(RECENT_KEY, [str(target), *(str(p) for p in kept)][:MAX_RECENT])


def forget_experiment(path: str | Path, settings: QSettings | None = None) -> None:
    """Drop ``path`` from the recent list. Missing entries are not an error."""
    s = settings if settings is not None else QSettings()
    try:
        target = Path(path).resolve()
    except Exception:  # pragma: no cover - unresolvable path
        return
    kept = [str(p) for p in recent_experiments(s) if p.resolve() != target]
    s.setValue(RECENT_KEY, kept)


# ---------------------------------------------------------------- the widget


class LandingPage(QWidget):
    """Logo, two primary actions, and the recent list.

    Args:
        settings: Injectable so a test never reads the developer's real recent
            list -- and, more sharply, so a test's writes never land in it.
        parent: Standard Qt parent.

    Signals:
        new_requested: *New Experiment* pressed.
        open_requested: *Open Experiment…* pressed.
        recent_requested: A recent row pressed, with its path as a string.
        lab_setup_requested: *Set up your lab* pressed.
        guide_requested: *User Guide* pressed.
    """

    new_requested = pyqtSignal()
    open_requested = pyqtSignal()
    recent_requested = pyqtSignal(str)
    lab_setup_requested = pyqtSignal()
    guide_requested = pyqtSignal()

    def __init__(self, settings: QSettings | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("landingPage")
        self._settings = settings if settings is not None else QSettings()
        self._recent_buttons: list[QToolButton] = []

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
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        column.addLayout(row)

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

        subtitle = QLabel("General Laboratory Interface for Design, Experimentation, and Recording")
        subtitle.setObjectName("landingSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        body.addWidget(subtitle)
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

        body.addSpacing(20)

        footer = QHBoxLayout()
        footer.setSpacing(16)
        lab_link = QToolButton()
        lab_link.setObjectName("landingLink")
        lab_link.setText("Set up your lab")
        lab_link.setCursor(Qt.CursorShape.PointingHandCursor)
        lab_link.clicked.connect(self.lab_setup_requested.emit)
        footer.addWidget(lab_link)

        guide_link = QToolButton()
        guide_link.setObjectName("landingLink")
        guide_link.setText("User Guide")
        guide_link.setCursor(Qt.CursorShape.PointingHandCursor)
        guide_link.clicked.connect(self.guide_requested.emit)
        footer.addWidget(guide_link)
        footer.addStretch(1)
        body.addLayout(footer)

        return card

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
