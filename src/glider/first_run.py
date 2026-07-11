"""First-launch setup: create the data directory and show a welcome dialog.

The goal is to make the first launch feel like a product, not a dev script, for
a non-technical user:

* Create the user-visible experiment-output directory if it doesn't exist.
* Pop a small welcome dialog. In Builder mode it offers **Take the Tour**
  (launches the interactive walkthrough from :mod:`glider.gui.onboarding`),
  **Skip**, and **Open User Guide**; in Runner mode — where the tour's dock
  targets don't exist — it keeps the original **Start** / **Open User Guide**
  pair.
* Persist a ``first_run/complete`` flag in :class:`QSettings` so subsequent
  launches skip the welcome. (The tour records its own completion separately;
  it stays replayable from Help → Replay Tutorial.)
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QSettings, QStandardPaths, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QMessageBox, QWidget

from glider.updater import OWNER, REPO

logger = logging.getLogger(__name__)

# QSettings key. Scoped so we can add more first-run state (e.g. picked-data-dir)
# later without colliding with other preferences.
FIRST_RUN_COMPLETE_KEY = "first_run/complete"

# Where the "Open User Guide" button takes the user. Points at the docs entry on
# the public repo — swap for a GitHub Pages URL if/when we host docs separately.
USER_GUIDE_URL = f"https://github.com/{OWNER}/{REPO}#readme"


def default_data_dir() -> Path:
    """The user-visible directory we want to create for experiment outputs.

    :func:`QStandardPaths.writableLocation` gives us the platform-correct
    Documents directory (``%USERPROFILE%\\Documents`` on Windows,
    ``~/Documents`` on macOS, ``$XDG_DOCUMENTS_DIR`` or ``~`` on Linux).
    """
    docs = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
    # Fall back to the user's home directory if Qt returns an empty string
    # (rare; happens on misconfigured Linux environments without XDG dirs).
    base = Path(docs) if docs else Path.home()
    return base / "GLIDER"


def ensure_data_dir(path: Path | None = None) -> Path:
    """Create ``path`` (and parents) if missing. Idempotent."""
    target = path or default_data_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Permission issues on a locked-down machine shouldn't prevent launch —
        # log and move on. The in-app file dialogs will surface the real error
        # later if the user actually tries to save there.
        logger.warning("Could not create data directory %s", target, exc_info=True)
    return target


def is_first_run(settings: QSettings | None = None) -> bool:
    """Return True if the welcome flow hasn't completed yet."""
    s = settings if settings is not None else QSettings()
    return not s.value(FIRST_RUN_COMPLETE_KEY, False, type=bool)


def _mark_complete(settings: QSettings) -> None:
    settings.setValue(FIRST_RUN_COMPLETE_KEY, True)


def show_welcome_dialog(
    parent: QWidget | None,
    data_dir: Path,
    *,
    offer_tour: bool = False,
) -> str:
    """Display the welcome message; return ``"tour"`` or ``"start"``.

    With ``offer_tour`` the primary action becomes **Take the Tour** (with a
    **Skip** escape hatch); otherwise the original single **Start** button.
    Dismissing the dialog any other way (including via Open User Guide or the
    window close button) counts as ``"start"``.
    """
    box = QMessageBox(parent)
    box.setWindowTitle("Welcome to GLIDER")
    box.setIcon(QMessageBox.Icon.Information)
    box.setText("Welcome to GLIDER!")
    box.setInformativeText(
        "Experiment files and recordings will be saved to:\n"
        f"{data_dir}\n\n"
        "You can change this later from File → Preferences."
    )
    guide_btn = box.addButton("Open User Guide", QMessageBox.ButtonRole.HelpRole)
    tour_btn = None
    if offer_tour:
        box.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
        tour_btn = box.addButton("Take the Tour", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(tour_btn)
    else:
        start_btn = box.addButton("Start", QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(start_btn)

    box.exec()
    clicked = box.clickedButton()
    if clicked is guide_btn:
        QDesktopServices.openUrl(QUrl(USER_GUIDE_URL))
        return "start"
    if tour_btn is not None and clicked is tour_btn:
        return "tour"
    return "start"


def run_first_run_if_needed(
    parent: QWidget | None,
    *,
    settings: QSettings | None = None,
) -> Path:
    """Full first-run orchestration.

    Always ensures the data directory exists (cheap and idempotent). Only shows
    the welcome dialog on the very first launch, then records completion in
    :class:`QSettings`. Returns the resolved data directory so callers can wire
    it into defaults (e.g. the save-file dialog's initial path).
    """
    s = settings if settings is not None else QSettings()
    data_dir = ensure_data_dir()
    if is_first_run(s):
        # Only offer the tour where its spotlight targets exist: a Builder-mode
        # main window with _start_tour. In Runner mode (or under a bare parent
        # in tests) fall back to the plain welcome.
        offer_tour = (
            getattr(parent, "is_runner_mode", True) is False
            and getattr(parent, "_start_tour", None) is not None
        )
        choice = "start"
        try:
            choice = show_welcome_dialog(parent, data_dir, offer_tour=offer_tour)
        finally:
            # Even if the dialog was dismissed via window-close rather than a
            # button, we want to flip the flag — never prompt the same person
            # twice.
            _mark_complete(s)
        if choice == "tour":
            try:
                parent._start_tour()
            except Exception:
                # The tour is a nice-to-have; a failure here must never break
                # the first launch.
                logger.warning("Could not start the onboarding tour", exc_info=True)
    return data_dir
