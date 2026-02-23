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

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="glider",
        description="GLIDER - General Laboratory Interface for Design, Experimentation, and Recording",
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
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Configure logging level."""
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("glider").setLevel(level)

    if debug:
        # Enable detailed logging for all GLIDER modules
        for name in ["glider.core", "glider.hal", "glider.gui", "glider.plugins"]:
            logging.getLogger(name).setLevel(logging.DEBUG)


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
    force_mode: Optional[str] = None,
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

    # Setup logging
    setup_logging(args.debug)

    logger.info("Starting GLIDER...")

    # Create Qt application
    # Enable high DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("GLIDER")
    app.setOrganizationName("LaingLab")
    app.setOrganizationDomain("lainglab.com")

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
