"""The launch splash: a mouse paraglides across and the wordmark appears behind it.

GLIDER's start-up is slow for reasons that are not going away -- plugin
discovery, hardware enumeration, and (on the vision path) importing torch --
and without a splash all of it happens against a bare desktop with no sign the
app had been launched at all. Double-launching from the Dock is the predictable
result. This module covers that gap.

Decisions worth stating, because each is a way the splash could go wrong:

* **The animation runs in its own process.** Initialisation blocks the GUI
  thread for seconds at a time (imports, enumeration), and an animation drawn
  on that thread freezes with it. The child is a few-hundred-millisecond
  PyQt-only process that plays ``assets/splash.gif`` and exits when told to --
  or when the parent dies, which it learns from its stdin closing.

* **The floor is a floor, not a duration.** :data:`MIN_VISIBLE_MS` is the
  *shortest* the splash may be up. Initialisation that takes longer keeps it
  up; the animation plays once and holds on the wordmark. A splash that
  vanished on a timer while the app was still building would hand the user the
  empty desktop again.

* **Nothing here may prevent launch.** :func:`show_splash` returns ``None``
  rather than raising if the animation is missing or the child will not start,
  and the child itself is best-effort. A branded launch is a nicety; starting
  is not.

The animation is made in ``tools/splash-animation`` (Remotion).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from time import monotonic

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

__all__ = ["CHILD_ARG", "MIN_VISIBLE_MS", "GliderSplash", "run_child", "show_splash"]

#: How long the splash stays up at minimum, in milliseconds. Initialisation
#: that outlasts this keeps it up; initialisation that beats it waits. Just
#: longer than the animation, so a fast launch still sees the whole flight.
MIN_VISIBLE_MS = 5000

#: ``argv[1]`` that makes ``python -m glider`` (or the frozen app) the splash.
CHILD_ARG = "_splash"

#: The GIF is rendered at twice its on-screen size, for HiDPI displays.
FRAME_SCALE = 2

#: A child that outlives this has lost its parent in a way stdin did not catch.
CHILD_MAX_LIFETIME_MS = 120_000

SPLASH_GIF = Path(__file__).resolve().parent.parent / "assets" / "splash.gif"


class GliderSplash:
    """The parent's handle on the splash process.

    Construct via :func:`show_splash` rather than directly -- it is the piece
    that knows how to fail quietly.
    """

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._shown_at = monotonic()

    def isVisible(self) -> bool:  # noqa: N802 - mirrors the QWidget it replaced
        return self._process.poll() is None

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

        **The hand-off is what shows the window**: the caller must not
        ``show()`` it first, or it sits on screen beside the splash for the
        rest of the floor.

        Args:
            window: The window to reveal. ``None`` just closes the splash.
            on_shown: Run once the window is up -- for anything that puts a
                dialog over it (the first-run welcome), which must not appear
                while the splash is still the visible app.
        """
        remaining = self.remaining_ms()
        if remaining == 0:
            self._hand_off(window, on_shown)
            return
        QTimer.singleShot(remaining, lambda: self._hand_off(window, on_shown))

    def close(self) -> None:
        """Tell the child to go. Safe to call more than once."""
        try:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()  # the child exits on EOF
            if self._process.poll() is None:
                self._process.terminate()
        except OSError:  # pragma: no cover - already gone
            pass

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
        except RuntimeError:  # pragma: no cover - window deleted mid-launch
            logger.debug("Splash hand-off target was gone", exc_info=True)
        self.close()
        if on_shown is not None:
            try:
                on_shown()
            except Exception:  # pragma: no cover - caller's own failure
                logger.warning("Post-splash callback failed", exc_info=True)


def _child_command() -> list[str]:
    if getattr(sys, "frozen", False):  # PyInstaller: the app binary is the entry
        return [sys.executable, CHILD_ARG]
    return [sys.executable, "-m", "glider", CHILD_ARG]


def show_splash() -> GliderSplash | None:
    """Start the splash process and return its handle, or ``None`` if it cannot.

    Best-effort by contract: a missing animation or a child that will not
    spawn returns ``None`` and logs at debug. Callers show a window either way.
    """
    if not SPLASH_GIF.is_file():
        logger.debug("Splash animation missing: %s", SPLASH_GIF)
        return None
    try:
        process = subprocess.Popen(
            _child_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except Exception:  # pragma: no cover - cosmetic path, never blocks launch
        logger.debug("Could not start the splash process", exc_info=True)
        return None
    return GliderSplash(process)


# --- the child process -------------------------------------------------------


def _hide_from_dock() -> None:
    """Keep the splash process out of the macOS Dock and app switcher.

    It is a second process, so without this macOS gives it a Dock icon of its
    own (a second GLIDER, or Python's rocket from source) beside the real one.
    ``NSApplicationActivationPolicyAccessory`` is the "windows, no Dock icon"
    policy; done through the ObjC runtime to avoid a pyobjc dependency.
    """
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        send = objc.objc_msgSend
        send.restype = ctypes.c_void_p
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        ns_app = send(
            objc.objc_getClass(b"NSApplication"), objc.sel_registerName(b"sharedApplication")
        )
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        send(ns_app, objc.sel_registerName(b"setActivationPolicy:"), 1)
    except Exception:  # pragma: no cover - cosmetic
        logger.debug("Could not hide the splash from the Dock", exc_info=True)


def _exit_when_parent_lets_go() -> None:
    """Block on stdin; EOF means the parent closed it or died. Then go.

    Reads the raw fd: a frozen windowed app has ``sys.stdin = None`` even when
    the parent handed it a pipe.
    """
    try:
        while os.read(0, 1024):
            pass
    except OSError:
        # No usable stdin: the parent's terminate() and the lifetime cap cover it.
        return
    os._exit(0)


def run_child() -> int:
    """Entry point of the splash process: play the animation until told to stop."""
    from PyQt6.QtGui import QMovie, QPainter
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    _hide_from_dock()

    movie = QMovie(str(SPLASH_GIF))
    if not movie.isValid():
        return 1
    movie.jumpToFrame(0)
    size = movie.currentPixmap().size() / FRAME_SCALE

    class Player(QWidget):
        def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
            frame = movie.currentPixmap()
            frame.setDevicePixelRatio(FRAME_SCALE)
            QPainter(self).drawPixmap(0, 0, frame)

    player = Player(
        None,
        Qt.WindowType.SplashScreen
        | Qt.WindowType.FramelessWindowHint
        | Qt.WindowType.WindowStaysOnTopHint,
    )
    # Best-effort: honoured on macOS and composited desktops; elsewhere the
    # rounded corners paint opaque and nothing else changes.
    player.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    player.setFixedSize(size)
    screen = app.primaryScreen()
    if screen is not None:
        player.move(screen.availableGeometry().center() - player.rect().center())
    movie.frameChanged.connect(lambda _n: player.update())

    threading.Thread(target=_exit_when_parent_lets_go, daemon=True).start()
    QTimer.singleShot(CHILD_MAX_LIFETIME_MS, app.quit)

    player.show()
    movie.start()
    return app.exec()
