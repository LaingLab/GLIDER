"""
GLIDER Styles - Qt Style Sheets for desktop and touch modes.
"""

from pathlib import Path

from glider.gui.styles import colors  # noqa: F401

STYLES_DIR = Path(__file__).parent


def load_stylesheet(name: str) -> str:
    """
    Load a stylesheet by name.

    Args:
        name: Stylesheet name without extension (e.g., 'desktop', 'touch')

    Returns:
        The stylesheet content as a string
    """
    style_path = STYLES_DIR / f"{name}.qss"
    if style_path.exists():
        return style_path.read_text(encoding="utf-8")
    return ""


def get_desktop_stylesheet() -> str:
    """Get the desktop mode stylesheet."""
    return load_stylesheet("desktop")


def get_touch_stylesheet() -> str:
    """Get the touch/runner mode stylesheet."""
    return load_stylesheet("touch")
