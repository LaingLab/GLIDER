"""In-app update checker.

On application startup we run a single, cheap, best-effort check against the
GitHub Releases API. If a newer tagged release exists, a modeless dialog tells
the user about it and offers three actions:

* **Download** — open the release page in the system browser.
* **Later** — dismiss; check again next launch.
* **Don't ask again** — suppress future prompts *for this specific version*.
  Next release resets the prompt.

The check is strictly non-blocking: it runs on a :class:`QThread`, has a 5-second
timeout, and silently swallows all network errors. The user must never see a
traceback or a "could not connect" toast at launch because of us — we'd rather
miss a prompt than scare non-technical lab staff.

Design invariants
-----------------
* Pure logic (:func:`is_newer`, :func:`fetch_latest_release`) is importable and
  unit-testable without a :class:`QApplication`. The Qt plumbing lives in
  :class:`UpdateChecker` and is exercised via ``pytest-qt``.
* Persistent "skip this version" state is stored in :class:`QSettings` under the
  organisation set in ``__main__.py`` (``LaingLab / GLIDER``) — matching where
  every other user preference lives. Key: ``updater/skipped_version``.
* The GitHub owner/repo is defined in one place (:data:`OWNER` / :data:`REPO`)
  so a future fork or move is a one-line edit.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import httpx
from PyQt6.QtCore import QObject, QSettings, QThread, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox, QWidget

from glider._version import __version__
from packaging import version

logger = logging.getLogger(__name__)

# GitHub coordinates. Change these in a single place if the project ever
# moves; the Pi Imager manifest and release workflow should be kept in sync.
OWNER = "LaingLab"
REPO = "glider"
RELEASES_API = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{OWNER}/{REPO}/releases/latest"

# How long we're willing to block a worker thread on the GitHub API before
# giving up. Five seconds is long enough for a slow home/lab connection and
# short enough that a dead network doesn't leave threads lingering.
REQUEST_TIMEOUT_SECONDS = 5.0

# QSettings key for the "don't show this version again" flag. Scoped under
# ``updater/`` so it doesn't collide with other preferences.
SKIPPED_VERSION_KEY = "updater/skipped_version"


class UpdateInfo(NamedTuple):
    """Immutable description of a release fetched from GitHub."""

    latest_version: str
    release_url: str
    release_notes: str


# --- Pure helpers (testable without Qt) --------------------------------------


def parse_version(v: str) -> version.Version:
    """Parse a version string, tolerant of a leading ``v`` or ``V``.

    GitHub release tags conventionally look like ``v1.2.3``; our in-code
    version is ``1.2.3``. Normalising both sides lets comparisons Just Work.
    """
    return version.parse(v.lstrip("vV").strip())


def is_newer(latest: str, current: str) -> bool:
    """Return True iff ``latest`` represents a strictly newer release.

    Returns False on any parsing failure — we never want a malformed tag on
    GitHub to cause us to nag the user about a "newer version" that isn't.
    """
    try:
        return parse_version(latest) > parse_version(current)
    except version.InvalidVersion:
        logger.debug("Could not parse version strings: latest=%r current=%r", latest, current)
        return False


def fetch_latest_release(
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    *,
    api_url: str = RELEASES_API,
) -> UpdateInfo | None:
    """Query GitHub's latest-release endpoint.

    Returns None on any failure (network, HTTP error, missing fields, draft
    release, or prerelease tag). The ``api_url`` argument exists for tests —
    nothing in the app overrides it.
    """
    try:
        response = httpx.get(
            api_url,
            timeout=timeout,
            headers={"Accept": "application/vnd.github+json"},
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        # We want truly any failure mode — DNS, TLS, 404, 403 rate-limit,
        # JSON parse errors — to degrade silently. Log at debug so noise
        # doesn't end up in user-visible logs.
        logger.debug("Update check failed", exc_info=True)
        return None

    # Ignore drafts and prereleases — a stable lab tool shouldn't ever get
    # nagged about a 1.3.0-rc1 during a real experiment.
    if data.get("draft") or data.get("prerelease"):
        return None

    tag_name = data.get("tag_name")
    if not tag_name:
        return None

    return UpdateInfo(
        latest_version=tag_name,
        release_url=data.get("html_url") or RELEASES_PAGE,
        release_notes=(data.get("body") or "").strip(),
    )


# --- Qt plumbing -------------------------------------------------------------


class _UpdateFetcher(QThread):
    """Worker thread that hits the GitHub API exactly once.

    We use a dedicated :class:`QThread` rather than :class:`QThreadPool` because
    the work is a single, short, one-shot I/O call and the lifetime management
    is trivial. The parent :class:`UpdateChecker` owns this thread.
    """

    result = pyqtSignal(object)  # emits UpdateInfo | None

    def run(self) -> None:  # pragma: no cover — thin wrapper; logic in fetch_latest_release
        self.result.emit(fetch_latest_release())


class UpdateChecker(QObject):
    """Controller that runs a release check and prompts the user about results.

    The same instance handles both modes:

    * ``check(silent=True)`` — run at startup, only shows UI if an update is
      available and the user hasn't already said "don't ask again" for it.
    * ``check(silent=False)`` — run from Help → Check for Updates, always shows
      some kind of UI (up-to-date, error, or update-available).

    The two modes share the worker thread and result-handling path.
    """

    # Emitted after every check, regardless of outcome. Mostly useful for
    # tests, but also consumed by e.g. a status-bar "last checked" indicator
    # if anyone wants to wire one up later.
    check_complete = pyqtSignal(object)  # UpdateInfo | None

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        current_version: str | None = None,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._parent_widget = parent
        self._current_version = current_version or __version__
        # Injecting QSettings keeps tests free of the global organisation/app
        # name state set up by QApplication.
        self._settings = settings if settings is not None else QSettings()
        self._thread: _UpdateFetcher | None = None
        self._silent = True

    # --- Public API ---

    def check(self, *, silent: bool) -> None:
        """Start a check. No-op if one is already in flight."""
        if self._thread is not None and self._thread.isRunning():
            logger.debug("Update check already in flight; skipping")
            return
        self._silent = silent
        self._thread = _UpdateFetcher(self)
        self._thread.result.connect(self._on_result)
        # Qt cleans up the thread after it finishes emitting; deleteLater
        # avoids leaking the underlying QObject.
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    # --- Internals ---

    @pyqtSlot(object)
    def _on_result(self, info: UpdateInfo | None) -> None:
        """Handle the worker's result on the GUI thread."""
        self.check_complete.emit(info)

        if info is None:
            if not self._silent:
                QMessageBox.information(
                    self._parent_widget,
                    "Check for Updates",
                    "Could not reach the update server.\nPlease try again later.",
                )
            return

        if not is_newer(info.latest_version, self._current_version):
            if not self._silent:
                QMessageBox.information(
                    self._parent_widget,
                    "Check for Updates",
                    f"You're up to date — GLIDER {self._current_version} "
                    "is the latest release.",
                )
            return

        # Newer version exists. In silent (startup) mode, honour the
        # "don't ask again for this specific tag" preference. The manual
        # Help → Check for Updates path always shows the prompt so the user
        # can un-snooze by choosing "Later" or "Download".
        skipped = self._settings.value(SKIPPED_VERSION_KEY, "", type=str)
        if self._silent and skipped == info.latest_version:
            logger.debug("Suppressing update prompt for skipped version %s", skipped)
            return

        self._show_prompt(info)

    def _show_prompt(self, info: UpdateInfo) -> None:
        """Show the three-button update dialog and act on the user's choice."""
        box = QMessageBox(self._parent_widget)
        box.setWindowTitle("GLIDER Update Available")
        box.setIcon(QMessageBox.Icon.Information)
        box.setText(
            f"GLIDER {info.latest_version} is available.\n"
            f"You have version {self._current_version}."
        )
        if info.release_notes:
            # Trim to a handful of lines — full release notes can be very
            # long and we don't want the dialog to grow without bound.
            snippet_lines = info.release_notes.splitlines()[:8]
            box.setInformativeText("\n".join(snippet_lines))

        download_btn = box.addButton("Download", QMessageBox.ButtonRole.AcceptRole)
        later_btn = box.addButton("Later", QMessageBox.ButtonRole.RejectRole)
        skip_btn = box.addButton("Don't ask again", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(download_btn)

        box.exec()
        clicked = box.clickedButton()

        if clicked is download_btn:
            QDesktopServices.openUrl(QUrl(info.release_url))
        elif clicked is skip_btn:
            self._settings.setValue(SKIPPED_VERSION_KEY, info.latest_version)
            logger.info("User opted to skip version %s", info.latest_version)
        # "Later" is the default fallback — no state change.
        del later_btn  # silence unused-local in strict mode


def schedule_startup_check(main_window: QWidget, *, delay_ms: int = 3000) -> UpdateChecker:
    """Kick off a silent update check a few seconds after the window shows.

    Returning the :class:`UpdateChecker` keeps it alive (otherwise the QObject
    would be garbage-collected mid-flight). Callers should store the returned
    reference on the main window.
    """
    from PyQt6.QtCore import QTimer

    checker = UpdateChecker(main_window)
    # Delay so the first-paint isn't competing with a cold-start network
    # syscall; this also makes it obvious that the window came up first.
    QTimer.singleShot(delay_ms, lambda: checker.check(silent=True))
    return checker
