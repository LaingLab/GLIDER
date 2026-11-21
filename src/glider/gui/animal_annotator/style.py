"""Animal Annotator's own styling, on top of the desktop stylesheet.

Only what the desktop sheet has no rule for: the editor's tool rail and hint
pill, the dropdown-style template pickers, a checked "Skip" button, compact lists and
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
QPushButton[chip="true"] {{
    background-color: {colors.SURFACE_1};
    color: {colors.TEXT_SECONDARY};
    border: 1px solid {colors.BORDER};
    border-radius: {radius.LARGE}px;
    padding: 3px 11px;
}}
QPushButton[chip="true"]:hover, QToolButton[chip="true"]:hover {{
    color: {colors.TEXT_PRIMARY};
}}
QPushButton[chip="true"]:checked {{
    background-color: rgba(56, 189, 248, 0.10);
    border-color: {colors.ACCENT};
    color: {colors.ACCENT};
}}
QToolButton[chip="true"] {{
    background-color: {colors.SURFACE_1};
    color: {colors.TEXT_SECONDARY};
    border: 1px solid {colors.BORDER};
    border-radius: {radius.LARGE}px;
    padding: 3px 11px;
}}
QToolButton[chip="true"]::menu-indicator {{
    image: none;
    width: 0px;
}}
QLabel[notice="true"] {{
    color: {_AMBER};
}}
QLabel[stat="true"] {{
    font-size: 17px;
    font-weight: 600;
    color: {colors.TEXT_PRIMARY};
}}
QFrame[studioStrip="true"] {{
    background-color: {colors.SURFACE_1};
    border: none;
    border-top: 1px solid {colors.BORDER};
    border-bottom: 1px solid {colors.BORDER};
}}
QFrame[studioStrip="true"] QWidget, QFrame[studioStrip="true"] QLabel {{
    background-color: transparent;
    border: none;
}}
QLineEdit[studioSearch="true"] {{
    background-color: {colors.SURFACE_1};
    border: 1px solid {colors.BORDER};
    border-radius: {radius.LARGE}px;
    padding: 3px 10px;
    color: {colors.TEXT_PRIMARY};
}}
QLineEdit[studioSearch="true"]:focus {{
    border-color: {colors.ACCENT};
}}
QPushButton[danger="true"] {{
    color: #f87171;
}}
QPushButton[danger="true"]:hover {{
    background-color: rgba(248, 113, 113, 0.14);
}}
QFrame[studioSelection="true"] {{
    background-color: rgba(56, 189, 248, 0.08);
    border: none;
    border-top: 1px solid {colors.BORDER};
}}
QToolButton[railTool="true"] {{
    background-color: transparent;
    color: {colors.TEXT_SECONDARY};
    border: none;
    border-radius: {radius.SMALL}px;
    min-width: 58px;
    padding: 4px 2px;
    font-size: 11px;
}}
QToolButton[railTool="true"][withMenu="true"] {{
    padding-right: 12px;
}}
QToolButton[railTool="true"]::menu-button {{
    border: none;
    width: 12px;
}}
QToolButton[railTool="true"]:hover {{
    background-color: {colors.SURFACE_2};
    color: {colors.TEXT_PRIMARY};
}}
QToolButton[railTool="true"]:checked {{
    background-color: rgba(56, 189, 248, 0.16);
    color: {colors.ACCENT};
}}
QLabel[studioHint="true"] {{
    background-color: rgba(0, 0, 0, 0.72);
    color: {colors.TEXT_PRIMARY};
    border: 1px solid {colors.BORDER};
    border-radius: {radius.MEDIUM}px;
    padding: 3px 14px;
    margin-top: 10px;
}}
QLabel[splitBadge="true"] {{
    background-color: {colors.SURFACE_2};
    border-radius: {radius.SMALL}px;
    padding: 1px 7px;
    font-weight: 600;
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
