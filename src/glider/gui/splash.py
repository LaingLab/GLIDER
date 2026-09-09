"""The launch splash: the logo on nothing, while the app builds itself behind it.

GLIDER's start-up is slow for reasons that are not going away -- plugin
discovery, hardware enumeration, and (on the vision path) importing torch --
and until now all of it happened against a bare desktop with no window and no
sign the app had been launched at all. Double-launching from the Dock is the
predictable result. This module covers that gap.

Three decisions here are worth stating, because each is a way the splash could
have gone wrong:

* **The floor is a floor, not a duration.** :data:`MIN_VISIBLE_MS` is the
  *shortest* the splash may be up, not the longest. Initialisation that takes
  longer keeps it up until it finishes -- a splash that vanished on a timer
  while the app was still building would hand the user an empty desktop again,
  which is the exact problem it exists to solve. Conversely a machine that
  initialises in 300 ms still gets the full floor, so the splash never appears
  as a flash the eye reads as a glitch.

* **Translucency is best-effort and never load-bearing.** ``WA_TranslucentBackground``
  is honoured on macOS and on Windows with composition on, and is honoured on
  Linux only where a compositor is running -- under a bare X session the region
  outside the logo paints black instead. Nothing downstream depends on which
  happened, so an uncomposited desktop gets an opaque splash and no error.

* **Nothing here may prevent launch.** Every entry point is wrapped by its
  caller and :func:`show_splash` returns ``None`` rather than raising if the
  icon is missing or the platform will not give us a window. A branded launch
  is a nicety; starting is not.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from time import monotonic

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen, QWidget

logger = logging.getLogger(__name__)

__all__ = ["MIN_VISIBLE_MS", "SPLASH_LOGO_PX", "GliderSplash", "show_splash"]

#: How long the splash stays up at minimum, in milliseconds. Initialisation
#: that outlasts this keeps it up; initialisation that beats it waits.
MIN_VISIBLE_MS = 5000

#: Logo edge length in *logical* pixels. The 512px asset is downscaled to this
#: with a smooth transform, so the splash stays crisp on a Retina display where
#: Qt asks for twice this many device pixels.
SPLASH_LOGO_PX = 220


class GliderSplash(QSplashScreen):
    """A frameless, translucent splash that closes no earlier than the floor.

    Construct via :func:`show_splash` rather than directly -- it is the piece
    that knows how to fail quietly.
    """

    def __init__(self, pixmap: QPixmap) -> None:
        super().__init__(pixmap)
        self.setWindowFlags(
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # The mask makes the transparent margin click-through as well as
        # invisible, so a splash sitting over a Finder window does not eat the
        # click that was meant for it.
        mask = pixmap.mask()
        if not mask.isNull():
            self.setMask(mask)
        self._shown_at = monotonic()

    def remaining_ms(self) -> int:
        """Milliseconds left on the floor; 0 once it has elapsed."""
        elapsed_ms = (monotonic() - self._shown_at) * 1000
        return max(0, int(MIN_VISIBLE_MS - elapsed_ms))

    def finish_when_due(
        self,
        window: QWidget | None = None,
        on_shown: Callable[[], None] | None = None,
    ) -> None:
        """Reveal ``window`` as soon as the floor allows, then close.

        **The hand-off is what shows the window**, which is the whole point:
        the caller must not ``show()`` it first. A window shown before the
        splash's timer fires sits on screen beside the splash for the rest of
        the floor -- the splash stops being a splash and becomes a sticker on
        an app that has already started.

        If the floor has elapsed the hand-off is immediate; otherwise it is
        scheduled for the remainder, so a fast machine still sees the whole
        splash and a slow one waits no longer than the work already took.

        Args:
            window: The window to reveal. ``None`` just closes the splash.
            on_shown: Run once the window is up -- for anything that puts a
                dialog over it (the first-run welcome), which must not be
                allowed to appear while the splash is still the visible app.
        """
        remaining = self.remaining_ms()
        if remaining == 0:
            self._hand_off(window, on_shown)
            return
        QTimer.singleShot(remaining, lambda: self._hand_off(window, on_shown))

    def _hand_off(
        self,
        window: QWidget | None,
        on_shown: Callable[[], None] | None = None,
    ) -> None:
        try:
            if window is not None:
                window.show()
                window.raise_()
                window.activateWindow()
            self.close()
        except RuntimeError:  # pragma: no cover - window deleted mid-launch
            logger.debug("Splash hand-off target was gone", exc_info=True)
        if on_shown is not None:
            try:
                on_shown()
            except Exception:  # pragma: no cover - caller's own failure
                logger.warning("Post-splash callback failed", exc_info=True)


def show_splash() -> GliderSplash | None:
    """Build, show and return the splash, or ``None`` if it cannot be built.

    Best-effort by contract: a missing asset, an unreadable PNG, or a platform
    that will not give us a splash window all return ``None`` and log at debug.
    Callers show a window either way.
    """
    try:
        from glider.assets import get_icon_path

        source = QPixmap(str(get_icon_path(512)))
        if source.isNull():
            logger.debug("Splash icon could not be loaded")
            return None

        pixmap = source.scaled(
            SPLASH_LOGO_PX,
            SPLASH_LOGO_PX,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        splash = GliderSplash(pixmap)
        splash.show()
        splash.raise_()
        # Paint it NOW. show() only queues the expose; the caller's next move is
        # init_glider, which blocks this thread for seconds at a time, so
        # without a forced round of event processing the splash's first paint
        # lands *after* everything it was supposed to cover -- which is how the
        # main window ends up on screen before the logo it was hiding behind.
        QApplication.processEvents()
        return splash
    except Exception:  # pragma: no cover - cosmetic path, never blocks launch
        logger.debug("Could not show splash screen", exc_info=True)
        return None
