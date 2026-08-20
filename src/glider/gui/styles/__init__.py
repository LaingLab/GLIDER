"""
GLIDER Styles - Qt Style Sheets for desktop and touch modes.
"""

from pathlib import Path

from PyQt6.QtWidgets import QWidget

from glider.gui.styles import colors  # noqa: F401

STYLES_DIR = Path(__file__).parent


def restyle(widget: QWidget) -> None:
    """Make Qt re-evaluate *widget* after a dynamic property changed.

    Qt resolves ``[state="..."]`` and ``[role="..."]`` selectors at *polish*
    time, not when the property is set, so a rule that ought to fire once the
    property moves never takes effect without this. The symptom is a widget
    whose property is provably correct and whose colour never changed --
    plausible enough to survive a code review and invisible to a test that only
    reads the property back.

    This lives in the styles package because four widgets had grown their own
    private copy of it (``plugin_card``, ``tool_ui``, ``side_panel``,
    ``status_strip``) and a fifth was about to. Three of the four were
    identical; ``tool_ui``'s also called ``update()``, which is the superset and
    is what is kept here -- a repaint after a restyle is what the caller wanted
    in every case.
    """
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
    widget.update()


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
