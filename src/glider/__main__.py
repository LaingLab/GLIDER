"""
GLIDER Main Entry Point

Launches the GLIDER application with proper Qt/asyncio integration
using qasync for non-blocking hardware operations.

Usage:
    python -m glider              # Auto-detect mode based on screen size
    python -m glider --builder    # Force Builder (desktop) mode
    python -m glider --runner     # Force Runner (touch) mode
    python -m glider --file path  # Open an experiment file
"""

import sys

# --- Windows torch / PyQt6 DLL load-order workaround ------------------------
# On Windows, importing PyQt6 before torch poisons torch's native library load:
# the first ``import torch`` afterwards fails with
# "[WinError 1114] A dynamic link library (DLL) initialization routine failed.
#  Error loading ...\\torch\\lib\\c10.dll". GLIDER only imports torch lazily
# (via ultralytics, deep in the vision pipeline), which always runs *after* the
# GUI has loaded Qt — so on Windows the YOLO/keypoint model silently fails to
# load and tracking falls back to background subtraction ("contour tracking").
# Importing torch here, before any PyQt6 import below, loads its DLLs while the
# order is still clean. Best-effort: torch is an optional dependency (absent on
# minimal / Raspberry Pi installs), so any failure is swallowed and the vision
# layer degrades exactly as it does today.
if sys.platform == "win32":
    try:
        import torch  # noqa: F401  (imported for its DLL-load side effect)
    except Exception:
        pass

import argparse
import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt

if TYPE_CHECKING:
    from glider.core.glider_core import GliderCore
    from glider.gui.main_window import MainWindow
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

# Configure logging before importing GLIDER modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("glider")


def get_system_font_family() -> str:
    """Get the appropriate system font family for the current platform."""
    if sys.platform == "darwin":
        return ".AppleSystemUIFont"
    elif sys.platform == "win32":
        return "Segoe UI"
    else:
        return "DejaVu Sans"


# Reverse-domain application ID. Must match the installer's publisher/app pair
# and is used by Windows to group taskbar icons, drive jump-list identity, and
# keep pinned shortcuts stable across updates. Keep in line with
# ``QApplication.setOrganizationDomain`` below.
_APP_USER_MODEL_ID = "com.lainglab.glider"


def _install_windows_app_id() -> None:
    """Tell Windows this process is 'GLIDER', not 'python.exe'.

    Without this call, the taskbar groups all PyQt windows under whatever exe
    launched Python (the interpreter during dev, or the frozen launcher after
    packaging), and pinning the running app gives you a shortcut to the
    interpreter instead of to GLIDER. Must run *before* any window is shown.
    No-op on non-Windows platforms or if shell32 is unavailable.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
    except Exception:  # pragma: no cover — best-effort cosmetic fix
        logger.debug("Could not set AppUserModelID", exc_info=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    from glider._version import __version__

    parser = argparse.ArgumentParser(
        prog="glider",
        description="GLIDER - General Laboratory Interface for Design, Experimentation, and Recording",
    )
    parser.add_argument(
        "--version",
        "--v",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the version and exit",
    )
    parser.add_argument(
        "--builder",
        action="store_true",
        help="Force Builder (desktop IDE) mode",
    )
    parser.add_argument(
        "--runner",
        action="store_true",
        help="Force Runner (touch dashboard) mode",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Open an experiment file on startup",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-plugins",
        action="store_true",
        help="Disable plugin loading",
    )
    parser.add_argument(
        "--gpu-check",
        action="store_true",
        help="Print GPU/accelerator diagnostics (CUDA/MPS/CPU) and exit",
    )
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Configure logging level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("glider").setLevel(level)

    if debug:
        # Enable detailed logging for all GLIDER modules
        for name in ["glider.core", "glider.hal", "glider.gui", "glider.plugins"]:
            logging.getLogger(name).setLevel(logging.DEBUG)


def _print_gpu_check() -> int:
    """Print accelerator diagnostics and exit — the handler for ``--gpu-check``.

    Reuses the pose subsystem's device utilities so the report matches exactly
    what inference resolves at runtime (CUDA > MPS > CPU). Runs before any Qt /
    core init so it works as a quick headless diagnostic.
    """
    try:
        from glider.vision.pose.device import diagnose, format_gpu_info, resolve_device
    except Exception as e:  # pragma: no cover - only on a broken vision install
        print(f"GPU check unavailable: could not import device utilities ({e})")
        return 1

    print("GLIDER GPU / device check")
    print("-" * 40)
    print(format_gpu_info())
    print()
    marks = {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "·"}
    for check, status, detail in diagnose():
        print(f"  {marks.get(status, '?')} {check}: {detail}")
    print()
    try:
        selected = resolve_device(None)
    except Exception as e:
        selected = f"unavailable ({e.__class__.__name__})"
    print(f"Inference will use: {selected}")
    return 0


async def init_glider(
    app: QApplication,
    args: argparse.Namespace,
) -> "GliderCore":
    """
    Initialize the GLIDER core system.

    Args:
        app: The Qt application instance
        args: Parsed command-line arguments

    Returns:
        Initialized GliderCore instance
    """
    from glider.core.glider_core import GliderCore
    from glider.plugins.plugin_manager import PluginManager

    # Create and initialize core instance
    core = GliderCore()
    await core.initialize()

    # Load plugins unless disabled (plugins already loaded in initialize, but allow extra)
    if not args.no_plugins and core._plugin_manager is None:
        plugin_manager = PluginManager()
        await plugin_manager.discover_plugins()
        await plugin_manager.load_plugins()
        # Plugins register their nodes directly with FlowEngine during load

    # Load experiment file if specified
    if args.file and args.file.exists():
        await core.load_experiment(args.file)
        logger.info(f"Loaded experiment: {args.file}")

    return core


def create_main_window(
    app: QApplication,
    core: "GliderCore",
    force_mode: str | None = None,
) -> "MainWindow":
    """
    Create the main application window.

    Args:
        app: The Qt application instance
        core: The initialized GliderCore
        force_mode: Force "builder" or "runner" mode, or None for auto-detect

    Returns:
        The main window instance
    """
    from glider.gui.main_window import MainWindow
    from glider.gui.styles import get_desktop_stylesheet, get_touch_stylesheet
    from glider.gui.view_manager import ViewManager, ViewMode

    # Create view manager to detect display mode
    view_manager = ViewManager(app)

    # Determine mode
    if force_mode == "builder":
        view_manager.mode = ViewMode.DESKTOP
        is_runner = False
    elif force_mode == "runner":
        view_manager.mode = ViewMode.RUNNER
        is_runner = True
    else:
        is_runner = view_manager.is_runner_mode

    # Create main window with view_manager to avoid duplicate detection
    window = MainWindow(core, view_manager=view_manager)

    # Apply appropriate stylesheet
    if is_runner:
        stylesheet = get_touch_stylesheet()
        window.switch_to_runner()
        logger.info("Starting in Runner mode")
    else:
        stylesheet = get_desktop_stylesheet()
        window.switch_to_builder()
        logger.info("Starting in Builder mode")

    window.setStyleSheet(stylesheet)

    return window


async def main_async(app: QApplication, args: argparse.Namespace) -> int:
    """
    Async main function.

    Args:
        app: The Qt application
        args: Parsed arguments

    Returns:
        Exit code
    """
    try:
        # Initialize GLIDER
        core = await init_glider(app, args)

        # Determine forced mode
        force_mode = None
        if args.builder:
            force_mode = "builder"
        elif args.runner:
            force_mode = "runner"

        # Create and show main window
        window = create_main_window(app, core, force_mode)

        # Create an event to signal when app should close
        close_event = asyncio.Event()

        # Connect app aboutToQuit to set the event
        app.aboutToQuit.connect(close_event.set)

        window.show()

        # Wait for the close event
        await close_event.wait()

        # Cleanup
        await core.shutdown()

        return 0

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


def main() -> int:
    """
    Main entry point.

    Sets up Qt application with qasync event loop integration.
    """
    # Parse arguments
    args = parse_args()

    # Headless diagnostic: report accelerators and exit before any Qt/core init.
    if args.gpu_check:
        return _print_gpu_check()

    # Setup logging
    setup_logging(args.debug)

    logger.info("Starting GLIDER...")

    # Tell Windows we're GLIDER, not python.exe. Must come before QApplication
    # is constructed so the taskbar entry is tagged correctly on first paint.
    _install_windows_app_id()

    # Create Qt application
    # Enable high DPI support.
    #
    # Round fractional device-pixel ratios down to the nearest integer rather
    # than passing them through. On macOS at a scaled ("More Space"/"Larger
    # Text") resolution or on an external non-Retina display, PassThrough yields
    # a fractional DPR (e.g. 1.6, 2.4) that lands 1px QSS borders on half-pixel
    # boundaries — they get anti-aliased across two rows and read as faint,
    # doubled hairlines around panels. RoundPreferFloor keeps hairlines crisp.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.RoundPreferFloor
    )

    app = QApplication(sys.argv)
    app.setApplicationName("GLIDER")
    app.setOrganizationName("LaingLab")
    app.setOrganizationDomain("lainglab.com")

    # Application icon. Shown in the taskbar, dock, Alt-Tab, window title bar,
    # and any dialog that doesn't set its own icon. Swallow failures so a
    # missing asset never blocks launch — users can still run the app without
    # a branded icon.
    try:
        from glider.assets import get_app_icon

        app.setWindowIcon(get_app_icon())
    except Exception:
        logger.warning("Could not load application icon", exc_info=True)

    # Set default application font to prevent "Point size <= 0" warnings
    default_font = QFont(get_system_font_family(), 10)
    app.setFont(default_font)

    try:
        # Import qasync and create event loop
        import qasync

        loop = qasync.QEventLoop(app)
        asyncio.set_event_loop(loop)

        # Schedule the async initialization
        async def run_app():
            try:
                core = await init_glider(app, args)

                force_mode = None
                if args.builder:
                    force_mode = "builder"
                elif args.runner:
                    force_mode = "runner"

                window = create_main_window(app, core, force_mode)
                window.show()

                # Packaging phase-1 wiring: first-run welcome + post-launch
                # silent update check. Both are best-effort — any failure
                # here must never prevent the app from coming up.
                try:
                    from glider.first_run import run_first_run_if_needed

                    run_first_run_if_needed(window)
                except Exception:
                    logger.warning("First-run setup failed", exc_info=True)

                try:
                    from PyQt6.QtCore import QTimer

                    # Delay so the update check doesn't race first-paint or
                    # compete with hardware enumeration on slow machines.
                    QTimer.singleShot(3000, lambda: window.check_for_updates(silent=True))
                except Exception:
                    logger.debug("Could not schedule startup update check", exc_info=True)

                # Store core reference for cleanup
                app._glider_core = core

            except Exception as e:
                logger.exception(f"Initialization error: {e}")
                app.quit()

        # Run initialization
        with loop:
            loop.run_until_complete(run_app())
            # Now run the Qt event loop via qasync
            loop.run_forever()

        # Cleanup — shutdown may already have been called during closeEvent,
        # and the event loop may already be closed, so guard both conditions.
        if hasattr(app, "_glider_core") and not loop.is_closed():
            try:
                loop.run_until_complete(app._glider_core.shutdown())
            except RuntimeError:
                pass  # Loop closed or shutdown already completed

        return 0

    except ImportError:
        logger.warning("qasync not available, running without async support")
        # Fallback without async - limited functionality
        return run_sync_fallback(app, args)


def run_sync_fallback(app: QApplication, args: argparse.Namespace) -> int:
    """
    Synchronous fallback when qasync is not available.

    This mode has limited functionality (no async hardware operations).
    """
    from glider.core.glider_core import GliderCore
    from glider.gui.main_window import MainWindow
    from glider.gui.styles import get_desktop_stylesheet, get_touch_stylesheet
    from glider.gui.view_manager import ViewManager, ViewMode

    logger.warning("Running in synchronous mode - hardware operations may block")

    # Set default application font to prevent "Point size <= 0" warnings
    default_font = QFont(get_system_font_family(), 10)
    app.setFont(default_font)

    # Create and initialize core (sync version - limited)
    core = GliderCore()
    # Run initialize synchronously
    loop = asyncio.new_event_loop()
    loop.run_until_complete(core.initialize())
    loop.close()

    # Determine mode
    view_manager = ViewManager(app)
    if args.builder:
        view_manager.mode = ViewMode.DESKTOP
        is_runner = False
    elif args.runner:
        view_manager.mode = ViewMode.RUNNER
        is_runner = True
    else:
        is_runner = view_manager.is_runner_mode

    # Create window with view_manager to avoid duplicate detection
    window = MainWindow(core, view_manager=view_manager)

    if is_runner:
        window.setStyleSheet(get_touch_stylesheet())
        window.switch_to_runner()
    else:
        window.setStyleSheet(get_desktop_stylesheet())
        window.switch_to_builder()

    window.show()

    # Run Qt event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
