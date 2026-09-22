"""The marker editor: a small popover anchored under a marker.

Name, colour and note, as Resolve's marker dialog has them, plus the two
things a review marker needs that an edit marker does not: whether it is
shared with the whole cohort, and turning a point into the In/Out range.
"""

from __future__ import annotations

from dataclasses import replace

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
)

from glider.analysis import markers as mk
from glider.gui.review.timeline import swatch
from glider.gui.review.viewport import format_seconds
from glider.gui.widgets.tool_ui import apply_tool_theme, data_font, set_button_role, set_text_role

__all__ = ["COHORT_NEEDS_A_FOLDER", "MarkerEditor"]

COHORT_NEEDS_A_FOLDER = "Open a folder as a cohort to share ranges"


def _swatch_icon(name: str, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(swatch(name))
        painter.drawRoundedRect(0, 0, size, size, 3, 3)
    finally:
        painter.end()
    return QIcon(pixmap)


class MarkerEditor(QFrame):
    """Edit one marker. ``done`` hands back an edited copy; ``deleted`` its id."""

    done = pyqtSignal(object)  # mk.Marker
    deleted = pyqtSignal(str)  # marker id

    def __init__(
        self,
        marker,
        *,
        cohort_problem: str | None,
        in_out: tuple[float, float] | None,
        parent=None,
    ):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("MarkerEditor")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._marker = replace(marker)
        self._in_out = in_out
        column = QVBoxLayout(self)
        column.setContentsMargins(14, 12, 14, 12)
        column.setSpacing(9)

        head = QHBoxLayout()
        self.kind_label = QLabel()
        self.kind_label.setObjectName("MarkerEditorTitle")
        self.time_label = QLabel()
        self.time_label.setFont(data_font(9))
        set_text_role(self.time_label, "muted")
        head.addWidget(self.kind_label)
        head.addStretch(1)
        head.addWidget(self.time_label)
        column.addLayout(head)

        self.name = QLineEdit(marker.name)
        self.name.setPlaceholderText("Name")
        # No returnPressed hook: QLineEdit passes Return on to its parent after
        # emitting it, so keyPressEvent below already sees it -- both would
        # finish twice.
        column.addWidget(self.name)

        row = QHBoxLayout()
        row.setSpacing(4)
        self.swatches = QButtonGroup(self)
        self.swatches.setExclusive(True)
        for name in mk.SWATCHES:
            button = QToolButton()
            button.setObjectName("MarkerSwatch")
            button.setCheckable(True)
            button.setIcon(_swatch_icon(name))
            button.setToolTip(name.capitalize())
            button.setProperty("swatch", name)
            button.setChecked(name == marker.color)
            self.swatches.addButton(button)
            row.addWidget(button)
        row.addStretch(1)
        column.addLayout(row)

        self.note = QPlainTextEdit(marker.note)
        self.note.setPlaceholderText("Note")
        self.note.setFixedHeight(56)
        column.addWidget(self.note)

        scope = QHBoxLayout()
        scope.setSpacing(0)
        self.this_session = QToolButton()
        self.this_session.setText("This session")
        self.whole_cohort = QToolButton()
        self.whole_cohort.setText("Whole cohort")
        group = QButtonGroup(self)
        for button in (self.this_session, self.whole_cohort):
            button.setObjectName("MarkerScope")
            button.setCheckable(True)
            group.addButton(button)
            scope.addWidget(button)
        scope.addStretch(1)
        cohort = marker.scope == "cohort"
        self.whole_cohort.setChecked(cohort)
        self.this_session.setChecked(not cohort)
        if cohort_problem is not None:
            self.whole_cohort.setEnabled(False)
            self.whole_cohort.setToolTip(cohort_problem)
        column.addLayout(scope)

        self.extend = QPushButton("Extend to In/Out")
        self.extend.setVisible(not marker.is_range)
        self.extend.setEnabled(in_out is not None)
        self.extend.setToolTip(
            "Turn this point into a range marker over the selected In/Out"
            if in_out is not None
            else "Select a range first (I and O set In and Out)"
        )
        self.extend.clicked.connect(self._extend)
        column.addWidget(self.extend)

        buttons = QHBoxLayout()
        self.delete_btn = QPushButton("Delete")
        set_button_role(self.delete_btn, "danger")
        self.cancel_btn = QPushButton("Cancel")
        self.done_btn = QPushButton("Done")
        set_button_role(self.done_btn, "primary")
        buttons.addWidget(self.delete_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.done_btn)
        column.addLayout(buttons)
        self.delete_btn.clicked.connect(self._delete)
        self.cancel_btn.clicked.connect(self.close)
        self.done_btn.clicked.connect(self._finish)

        self._show_kind()
        self.setMinimumWidth(300)
        apply_tool_theme(self)

    def _show_kind(self) -> None:
        m = self._marker
        self.kind_label.setText("Range marker" if m.is_range else "Point marker")
        if m.is_range:
            self.time_label.setText(f"{format_seconds(m.start_s)} – {format_seconds(m.end_s)}")
        else:
            self.time_label.setText(format_seconds(m.start_s))

    def _extend(self) -> None:
        if self._in_out is None:
            return
        self._marker.kind = "range"
        self._marker.start_s, self._marker.end_s = self._in_out
        self.extend.setVisible(False)
        self._show_kind()

    def edited(self):
        """The marker as the fields now describe it."""
        chosen = self.swatches.checkedButton()
        return replace(
            self._marker,
            name=self.name.text().strip(),
            note=self.note.toPlainText().strip(),
            color=chosen.property("swatch") if chosen is not None else self._marker.color,
            scope="cohort" if self.whole_cohort.isChecked() else "session",
        )

    def _finish(self) -> None:
        self.done.emit(self.edited())
        self.close()

    def _delete(self) -> None:
        self.deleted.emit(self._marker.id)
        self.close()

    def keyPressEvent(self, event):  # noqa: N802 - Qt override
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not self.note.hasFocus():
            self._finish()
            return
        super().keyPressEvent(event)

    def popup(self, at: QPoint) -> None:
        """Show under the global point ``at``, kept on its screen."""
        self.adjustSize()
        screen = QApplication.screenAt(at) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        x = max(area.left(), min(at.x() - 22, area.right() - self.width()))
        y = max(area.top(), min(at.y() + 6, area.bottom() - self.height()))
        self.move(x, y)
        self.show()
        self.name.setFocus()
        self.name.selectAll()
