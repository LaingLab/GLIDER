"""Animal Annotator's own styling, on top of the desktop stylesheet.

Only what the desktop sheet has no rule for: the two-part mode switch, the
dropdown-style template pickers, a checked "Skip" button, compact lists and
the panel frames. Everything else uses the app's existing roles --
``buttonRole="secondary"`` for buttons that are not the main action and
``textRole`` for muted and section text -- so the window looks like GLIDER.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QPushButton, QWidget

from glider.gui.styles import colors, radius

__all__ = ["muted", "secondary", "section", "studio_stylesheet"]

_AMBER = "#fbbf24"
#: The desktop sheet's disabled text colour (TEXT_TERTIARY-ish).
_DISABLED = "#475569"


def studio_stylesheet() -> str:
    return f"""
QPushButton[segment="left"], QPushButton[segment="right"] {{
    background-color: {colors.SURFACE_2};
    color: {colors.TEXT_SECONDARY};
    border: 1px solid {colors.BORDER};
    padding: 5px 16px;
}}
QPushButton[segment="left"] {{
    border-top-left-radius: {radius.SMALL}px;
    border-bottom-left-radius: {radius.SMALL}px;
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
}}
QPushButton[segment="right"] {{
    border-top-right-radius: {radius.SMALL}px;
    border-bottom-right-radius: {radius.SMALL}px;
    border-top-left-radius: 0px;
    border-bottom-left-radius: 0px;
    border-left: none;
}}
QPushButton[segment="left"]:hover, QPushButton[segment="right"]:hover {{
    color: {colors.TEXT_PRIMARY};
}}
QPushButton[segment="left"]:checked, QPushButton[segment="right"]:checked {{
    background-color: {colors.ACCENT};
    border-color: {colors.ACCENT};
    color: {colors.CANVAS};
}}
QPushButton[buttonRole="secondary"]:checked {{
    background-color: rgba(251, 191, 36, 0.16);
    color: {_AMBER};
}}
QPushButton[buttonRole="secondary"]:disabled {{
    background-color: {colors.SURFACE_1};
    color: {_DISABLED};
}}
QToolButton[stepper="true"] {{
    background-color: {colors.SURFACE_2};
    border: none;
    border-radius: {radius.SMALL}px;
    padding: 6px;
}}
QToolButton[stepper="true"]:hover {{
    background-color: {colors.BORDER};
}}
QFrame[studioPanel="true"] QListWidget {{
    border: none;
    background-color: transparent;
}}
QToolButton[studioPicker="true"] {{
    background-color: {colors.SURFACE_2};
    border: 1px solid {colors.BORDER};
    border-radius: {radius.SMALL}px;
    padding: 6px 10px;
    color: {colors.TEXT_PRIMARY};
}}
QToolButton[studioPicker="true"]:hover {{
    border-color: {colors.ACCENT};
}}
QToolButton[studioPicker="true"]::menu-indicator {{
    image: none;
    width: 0px;
}}
QLabel[studioTitle="true"] {{
    font-size: 15px;
    font-weight: 600;
    color: {colors.TEXT_PRIMARY};
}}
QListWidget[compact="true"]::item {{
    padding: 4px 6px;
}}
QFrame[studioPanel="true"] {{
    background-color: {colors.SURFACE_1};
    border: 1px solid {colors.BORDER};
    border-radius: {radius.SMALL}px;
}}
"""


def secondary(button: QPushButton) -> QPushButton:
    """Mark a button as a secondary action (the app's quieter button style)."""
    button.setProperty("buttonRole", "secondary")
    return button


def muted(label: QLabel, small: bool = False) -> QLabel:
    """Grey, secondary text; ``small`` for help lines."""
    label.setProperty("textRole", "muted" if small else "secondary")
    return label


def section(text: str, parent: QWidget | None = None) -> QLabel:
    """An upper-case section heading, as the app's side panels use."""
    label = QLabel(text.upper(), parent)
    label.setProperty("textRole", "section")
    return label
