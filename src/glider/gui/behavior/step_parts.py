"""Pieces every Behavior Analysis step is built from.

Each step shows its required inputs, then one footer: a readiness message,
an optional Options button and one primary button. Tuning lives in an
:class:`OptionsPopup` built once and kept, so every widget in it stays a
member of its tab -- values set while it is closed still reach the run, and
the run reads them the way it always did.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QAbstractButton,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from glider.gui.widgets.tool_ui import (
    GUTTER,
    apply_tool_theme,
    caption,
    set_button_role,
    set_text_role,
)


class StepFooter(QFrame):
    """``[ readiness message ........ Options  Primary ▾ ]``.

    The message is where anything blocking the primary action is named --
    a setting hidden in Options must never block silently.
    """

    def __init__(self, primary_text: str, *, options: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("StepFooter")
        row = QHBoxLayout(self)
        row.setContentsMargins(GUTTER, 10, GUTTER, 10)
        row.setSpacing(10)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        set_text_role(self.message, "muted")
        row.addWidget(self.message, 1)

        self.options_button: QPushButton | None = None
        if options:
            self.options_button = QPushButton("Options")
            row.addWidget(self.options_button)

        # A QToolButton so the step's other actions can hang off it as a menu
        # (Resume / Label Clips / Review, Fit / Cross-validate) instead of
        # competing for space as separate buttons.
        self.primary = QToolButton()
        self.primary.setText(primary_text)
        self.primary.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.primary.setMinimumHeight(38)
        set_button_role(self.primary, "primary")
        row.addWidget(self.primary)
        self.menu: QMenu | None = None

    def set_message(self, text: str, *, warning: bool = False) -> None:
        self.message.setText(text)
        set_text_role(self.message, "warning" if warning else "muted")

    def set_primary_text(self, text: str) -> None:
        self.primary.setText(text)

    def add_menu_action(self, text: str, slot: Callable[[], None]) -> QAction:
        """Hang another action off the primary button's menu arrow."""
        if self.menu is None:
            self.menu = QMenu(self.primary)
            self.primary.setMenu(self.menu)
            self.primary.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        action = self.menu.addAction(text)
        action.triggered.connect(lambda _checked=False: slot())
        return action


class OptionsPopup(QDialog):
    """A step's rarely-touched settings, in a non-modal panel by its button.

    A dialog rather than a ``Qt.Popup``: a popup closes the moment a file
    dialog opens from inside it, and several options are file pickers.
    """

    def __init__(self, anchor: QWidget, title: str = "Options"):
        super().__init__(anchor)
        self._anchor = anchor
        self.setWindowTitle(title)
        self.setModal(False)
        self.setMinimumWidth(460)
        self._column = QVBoxLayout(self)
        self._column.setContentsMargins(GUTTER, GUTTER, GUTTER, GUTTER)
        self._column.setSpacing(6)
        # Top-level windows get no theme from their parent (see apply_tool_theme).
        apply_tool_theme(self)
        if isinstance(anchor, QAbstractButton):
            anchor.clicked.connect(self.toggle)

    def add_section(self, title: str) -> QFormLayout:
        if self._column.count():
            self._column.addSpacing(8)
        self._column.addWidget(caption(title.upper()))
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)
        self._column.addLayout(form)
        return form

    def toggle(self) -> None:
        if self.isVisible():
            self.hide()
            return
        self.adjustSize()
        # Right-aligned above the anchor: the button sits in the step's footer.
        corner = self._anchor.mapToGlobal(QPoint(self._anchor.width(), 0))
        x, y = corner.x() - self.width(), corner.y() - self.height() - 8
        # Kept on the anchor's screen: a window near an edge would otherwise
        # push the panel (and its title bar) out of reach.
        screen = self._anchor.screen()
        if screen is not None:
            area = screen.availableGeometry()
            x = max(area.left(), min(x, area.right() - self.width()))
            y = max(area.top(), min(y, area.bottom() - self.height()))
        self.move(x, y)
        self.show()
        self.raise_()
        self.activateWindow()


def summary_label() -> QLabel:
    """The one line under a step's inputs naming the non-default options in force."""
    label = QLabel("")
    label.setWordWrap(True)
    set_text_role(label, "muted")
    return label
