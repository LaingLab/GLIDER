"""Lazy installer for ultralytics (YOLO).

We deliberately ship our installers *without* ultralytics because it's licensed
AGPL-3.0 — bundling it would impose source-availability obligations on anyone
who redistributes GLIDER. Instead, the first time a user selects a YOLO
detection backend, this module:

1. Detects whether ``ultralytics`` is already importable.
2. If not, asks the user for consent (with a plain-English AGPL disclosure).
3. Runs ``pip install ultralytics`` into the active Python environment on a
   background :class:`QThread` so the GUI stays responsive.

PyInstaller-frozen builds can't pip-install (``sys.executable`` is the
frozen app binary, not an interpreter). In that case we skip the auto-install
and show a short dialog explaining how to get YOLO support — the user's
options are to use a source install or to drop an add-on pack into the
plugin directory.

Scope notes
-----------
* We only handle ``ultralytics``; its transitive deps (torch, torchvision)
  come along via pip. Torch is heavy — ~700 MB — so we warn about size
  before starting.
* Cancellation is not supported mid-install. pip isn't safe to SIGKILL
  (partial extraction leaves broken wheels in the site-packages). If the
  user really wants out, they can close the app; the next launch re-prompts.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from typing import NamedTuple

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# Cache the availability check — importlib.util.find_spec is cheap but not
# free, and we call it every time the user changes the backend combo.
_cached_available: bool | None = None


class InstallResult(NamedTuple):
    """Outcome of a pip-install run."""

    success: bool
    message: str  # human-readable; surfaced in the final dialog on failure


# --- Detection (pure, testable) ---------------------------------------------


def is_ultralytics_available(*, use_cache: bool = True) -> bool:
    """Return True if ``ultralytics`` can be imported.

    Uses :func:`importlib.util.find_spec` rather than a bare ``import`` so
    we don't pay the cost of actually loading torch just to answer a yes/no
    question. The result is memoised across the process; pass
    ``use_cache=False`` to force a re-check after a fresh install.
    """
    global _cached_available
    if use_cache and _cached_available is not None:
        return _cached_available
    _cached_available = importlib.util.find_spec("ultralytics") is not None
    return _cached_available


def _invalidate_cache() -> None:
    """Reset the availability cache — call right after a successful install."""
    global _cached_available
    _cached_available = None


def can_auto_install() -> bool:
    """Return True if this process can run ``pip install`` into itself.

    PyInstaller sets ``sys.frozen`` and ``sys._MEIPASS`` on frozen builds; in
    that mode ``sys.executable`` points at the frozen launcher binary and has
    no ``pip`` module, so we must not pretend we can install. Source
    installs (development checkouts, the Pi image, any ``pip install -e .``
    deployment) *can* install because ``sys.executable`` is a real Python.
    """
    return not getattr(sys, "frozen", False)


# --- Subprocess install (testable via monkeypatching subprocess.run) --------


def install_ultralytics_blocking(timeout: float = 600.0) -> InstallResult:
    """Run ``pip install ultralytics`` and block until it finishes.

    Separate from the Qt flow so tests can exercise the subprocess layer
    without dragging in Qt. Called from the QThread worker below.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"]
    logger.info("Running: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return InstallResult(False, "Install timed out after 10 minutes.")
    except FileNotFoundError as e:
        # Can only happen in weird packaging scenarios (sys.executable missing).
        return InstallResult(False, f"Could not launch pip: {e}")

    if completed.returncode == 0:
        _invalidate_cache()
        return InstallResult(True, "Installed.")

    # pip's stderr carries the useful error lines; keep it short for the
    # dialog but log the full output for post-mortem.
    logger.error("pip install failed:\n%s\n%s", completed.stdout, completed.stderr)
    tail = (completed.stderr or completed.stdout or "").splitlines()[-5:]
    return InstallResult(False, "pip install failed:\n" + "\n".join(tail))


# --- Qt plumbing ------------------------------------------------------------


class _InstallWorker(QThread):
    """Runs :func:`install_ultralytics_blocking` off the GUI thread."""

    completed = pyqtSignal(object)  # emits InstallResult

    def run(self) -> None:  # pragma: no cover — thin thread wrapper
        self.completed.emit(install_ultralytics_blocking())


class _InstallProgressDialog(QDialog):
    """Modal progress dialog shown while pip is running.

    Busy indicator rather than a progress bar — pip's line-by-line output is
    noisy and not worth parsing for a percentage. We just show "Installing…"
    and rely on the subprocess timeout to bound the worst case.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Installing YOLO")
        self.setModal(True)
        # Remove the "?" help button — we have no help to offer here.
        self.setWindowFlags(self.windowFlags() & ~0x00000040)  # Qt.WindowContextHelpButtonHint

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Downloading and installing the YOLO object detection\n"
                "library (ultralytics) and its dependencies.\n\n"
                "This may take several minutes — ~700 MB will be downloaded."
            )
        )
        # No percentage is available here — pip reports nothing we can parse —
        # so this is a pure "still working" state.
        from glider.gui.widgets import BusyIndicator

        layout.addWidget(BusyIndicator("Installing…", size=110))
        # Intentionally no Cancel button — killing pip mid-install leaves the
        # environment in an inconsistent state. Users who really want out can
        # close the app.
        self._result: InstallResult | None = None

    def set_result(self, result: InstallResult) -> None:
        self._result = result
        self.accept() if result.success else self.reject()

    def result_info(self) -> InstallResult | None:
        return self._result


def _show_frozen_build_message(parent: QWidget | None) -> None:
    """Tell the user that auto-install isn't available in a frozen build."""
    QMessageBox.information(
        parent,
        "YOLO not available in this build",
        "YOLO object detection requires the ultralytics library, which is\n"
        "not included in this installer for licensing reasons (AGPL-3.0).\n\n"
        "To enable YOLO, use a source install of GLIDER "
        "(`pip install -e .`) and re-select the YOLO backend, or contact\n"
        "your lab administrator for an add-on pack.",
    )


def _confirm_agpl(parent: QWidget | None) -> bool:
    """Show the AGPL disclosure + consent prompt. Return True on accept."""
    box = QMessageBox(parent)
    box.setWindowTitle("Install YOLO (ultralytics)?")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText("YOLO object detection is not installed.")
    box.setInformativeText(
        "GLIDER can download and install the ultralytics library now.\n\n"
        "Heads-up:\n"
        "• About 700 MB will be downloaded (PyTorch + model weights).\n"
        "• ultralytics is licensed under AGPL-3.0. That means if you later\n"
        "  distribute this installation of GLIDER to others, you may owe\n"
        "  them the modified source. For personal/lab use, this isn't a\n"
        "  concern.\n\n"
        "Proceed with install?"
    )
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.No)
    return box.exec() == QMessageBox.StandardButton.Yes


# --- Orchestrator -----------------------------------------------------------


def ensure_ultralytics_installed(parent: QWidget | None = None) -> bool:
    """Return True if ultralytics is available after this call.

    The caller is expected to check the return value and revert whatever UI
    change prompted the request (e.g. switching the detection-backend combo
    back to "Background Sub") if this returns False.
    """
    if is_ultralytics_available():
        return True

    if not can_auto_install():
        _show_frozen_build_message(parent)
        return False

    if not _confirm_agpl(parent):
        return False

    # Run the install on a worker thread so the GUI stays responsive.
    dialog = _InstallProgressDialog(parent)
    worker = _InstallWorker(dialog)
    worker.completed.connect(dialog.set_result)
    worker.completed.connect(worker.deleteLater)
    worker.start()
    dialog.exec()

    result = dialog.result_info()
    if result is None or not result.success:
        QMessageBox.warning(
            parent,
            "Install failed",
            (result.message if result else "Install was interrupted."),
        )
        return False

    # Paranoia: re-check via find_spec rather than trusting the subprocess
    # exit code alone. Cache was invalidated inside install_ultralytics_blocking.
    return is_ultralytics_available(use_cache=False)
