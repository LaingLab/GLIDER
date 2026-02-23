"""
View Manager - Handles responsive interface design and mode switching.

Detects the runtime environment and creates appropriate view hierarchies
for either Desktop (Builder) or Runner (Raspberry Pi) mode.
"""

import logging
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtCore import QSize
    from PyQt6.QtWidgets import QApplication, QWidget

logger = logging.getLogger(__name__)


class ViewMode(Enum):
    """Application view modes."""

    DESKTOP = auto()  # Full IDE mode
    RUNNER = auto()  # Touch-optimized dashboard mode
    AUTO = auto()  # Auto-detect based on screen size


class ViewManager:
    """
    Manages view mode detection and switching.

    Detects display properties at startup and creates distinct
    view hierarchies based on the runtime environment.
    """

    # Screen dimensions for Runner mode detection
    RUNNER_WIDTH_THRESHOLD = 480
    RUNNER_HEIGHT_THRESHOLD = 800
    PI_SCREEN_WIDTH = 480
    PI_SCREEN_HEIGHT = 800

    def __init__(self, app: "QApplication"):
        """
        Initialize the view manager.

        Args:
            app: The Qt application instance
        """
        self._app = app
        self._mode = ViewMode.AUTO
        self._detected_mode: ViewMode | None = None
        self._screen_size: QSize | None = None
        self._style_path = Path(__file__).parent / "styles"

    @property
    def mode(self) -> ViewMode:
        """Current view mode."""
        if self._mode == ViewMode.AUTO:
            return self._detect_mode()
        return self._mode

    @mode.setter
    def mode(self, value: ViewMode) -> None:
        """Set the view mode."""
        self._mode = value

    @property
    def is_runner_mode(self) -> bool:
        """Whether currently in runner mode."""
        return self.mode == ViewMode.RUNNER

    @property
    def is_desktop_mode(self) -> bool:
        """Whether currently in desktop mode."""
        return self.mode == ViewMode.DESKTOP

    @property
    def screen_size(self) -> "QSize":
        """Get the primary screen size."""
        if self._screen_size is None:
            from PyQt6.QtWidgets import QApplication

            screen = QApplication.primaryScreen()
            if screen:
                self._screen_size = screen.size()
            else:
                from PyQt6.QtCore import QSize

                self._screen_size = QSize(1920, 1080)
        return self._screen_size

    def _detect_mode(self) -> ViewMode:
        """Auto-detect the appropriate mode based on screen size."""
        if self._detected_mode is not None:
            return self._detected_mode

        width = self.screen_size.width()
        height = self.screen_size.height()

        logger.info(f"Detected screen size: {width}x{height}")

        # Check for typical Raspberry Pi touchscreen dimensions
        # Portrait mode: 480x800
        if width == self.PI_SCREEN_WIDTH and height == self.PI_SCREEN_HEIGHT:
            self._detected_mode = ViewMode.RUNNER
            logger.info("Auto-detected Runner mode (480x800 portrait)")
        # Landscape mode: 800x480
        elif width == self.PI_SCREEN_HEIGHT and height == self.PI_SCREEN_WIDTH:
            self._detected_mode = ViewMode.RUNNER
            logger.info("Auto-detected Runner mode (800x480 landscape)")
        # Generic small screen detection
        elif width <= self.RUNNER_WIDTH_THRESHOLD:
            self._detected_mode = ViewMode.RUNNER
            logger.info("Auto-detected Runner mode (narrow screen)")
        elif height <= self.RUNNER_WIDTH_THRESHOLD and width > height:
            self._detected_mode = ViewMode.RUNNER
            logger.info("Auto-detected Runner mode (rotated screen)")
        else:
            self._detected_mode = ViewMode.DESKTOP
            logger.info("Auto-detected Desktop mode")

        return self._detected_mode

    def is_portrait(self) -> bool:
        """Check if screen is in portrait orientation."""
        return self.screen_size.height() > self.screen_size.width()

    def get_stylesheet_path(self) -> Path:
        """Get the appropriate stylesheet path for current mode."""
        if self.is_runner_mode:
            return self._style_path / "touch.qss"
        else:
            return self._style_path / "desktop.qss"

    def apply_stylesheet(self, widget: "QWidget") -> None:
        """Apply the appropriate stylesheet to a widget."""
        style_path = self.get_stylesheet_path()

        if style_path.exists():
            with open(style_path) as f:
                stylesheet = f.read()
            widget.setStyleSheet(stylesheet)
            logger.info(f"Applied stylesheet: {style_path}")
        else:
            logger.warning(f"Stylesheet not found: {style_path}")

    def get_minimum_button_size(self) -> tuple:
        """Get minimum button size for current mode."""
        if self.is_runner_mode:
            return (80, 60)  # Large touch targets
        else:
            return (30, 24)  # Standard size

    def get_font_size(self, element: str = "default") -> int:
        """Get appropriate font size for current mode."""
        sizes = {
            "desktop": {
                "default": 10,
                "heading": 14,
                "label": 9,
                "button": 10,
            },
            "runner": {
                "default": 16,
                "heading": 24,
                "label": 14,
                "button": 18,
            },
        }

        mode_key = "runner" if self.is_runner_mode else "desktop"
        return sizes[mode_key].get(element, sizes[mode_key]["default"])

    def get_padding(self) -> int:
        """Get appropriate padding for current mode."""
        if self.is_runner_mode:
            return 20
        else:
            return 8

    def get_scrollbar_width(self) -> int:
        """Get appropriate scrollbar width for current mode."""
        if self.is_runner_mode:
            return 40  # Wide touch-friendly scrollbars
        else:
            return 12  # Standard width

    def should_show_node_graph(self) -> bool:
        """Whether to show the node graph editor."""
        return self.is_desktop_mode

    def should_enable_kinetic_scrolling(self) -> bool:
        """Whether to enable kinetic scrolling."""
        return self.is_runner_mode

    def get_window_flags(self):
        """Get appropriate window flags for current mode."""
        from PyQt6.QtCore import Qt

        if self.is_runner_mode:
            # Frameless fullscreen for kiosk mode
            return Qt.WindowType.FramelessWindowHint
        else:
            return Qt.WindowType.Window

    def configure_widget_for_touch(self, widget: "QWidget") -> None:
        """Configure a widget for touch interaction."""
        if not self.is_runner_mode:
            return

        from PyQt6.QtWidgets import QScroller

        # Enable kinetic scrolling
        QScroller.grabGesture(widget, QScroller.ScrollerGestureType.LeftMouseButtonGesture)

    def get_layout_config(self) -> dict:
        """Get layout configuration for current mode."""
        if self.is_runner_mode:
            return {
                "margins": (10, 10, 10, 10),
                "spacing": 10,
                "stretch_factors": [1],
            }
        else:
            return {
                "margins": (4, 4, 4, 4),
                "spacing": 4,
                "stretch_factors": [3, 1],  # Main area, properties panel
            }
