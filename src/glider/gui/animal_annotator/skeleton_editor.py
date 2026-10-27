"""Skeleton template editor and picker, after the Ultralytics Platform annotator.

- :class:`SkeletonCanvas` -- the grid where keypoints are placed. Click empty
  space to place a keypoint; click one to select it; drag to move;
  Delete removes the selected keypoint; Esc deselects. Keypoints are not
  joined by lines.
- :class:`SkeletonEditorDialog` -- name, description, *Start from…*, the canvas,
  and one row per keypoint (index, colour, name, delete).
- :class:`TemplatePicker` -- the toolbar button listing default and custom
  templates, with *New Template…*.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QColorDialog,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from glider.gui.animal_annotator.style import muted, studio_stylesheet
from glider.gui.styles import colors, radius
from glider.gui.widgets.pastel_glyphs import lucide_icon
from glider.vision.training.templates import (
    SkeletonTemplate,
    TemplateError,
    builtin_templates,
    delete_custom_template,
    list_custom_templates,
    save_custom_template,
)

logger = logging.getLogger(__name__)

__all__ = ["SkeletonCanvas", "SkeletonEditorDialog", "TemplatePicker", "edit_template"]

_POINT_R = 5.0
_HIT = 9.0  # pixels: how close a click must be to a keypoint
_DRAG_START = 3.0  # pixels moved before a press becomes a drag
_GRID = 20  # grid spacing in pixels
_MARGIN = 14


class SkeletonCanvas(QWidget):
    """Place and move a template's keypoints on a square grid.

    Signals:
        changed: Keypoints changed. ``structure`` is True when
            keypoints were added or removed, so the row list must be rebuilt.
        selection_changed: The selected keypoint index, or -1.
    """

    changed = pyqtSignal(bool)
    selection_changed = pyqtSignal(int)

    def __init__(self, template: SkeletonTemplate, parent=None, *, read_only: bool = False):
        super().__init__(parent)
        self._template = template
        self._read_only = read_only
        #: Keypoints can be moved and renamed but not added or removed.
        self.locked = False
        self.selected = -1
        self._press: tuple[int, QPointF] | None = None  # keypoint pressed, where
        self._dragging = False
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus if read_only else Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(not read_only)

    def set_template(self, template: SkeletonTemplate) -> None:
        self._template = template
        self.select(-1)
        self.update()

    # -- geometry -------------------------------------------------------------

    def _area(self) -> QRectF:
        """The square the 0-1 template coordinates map onto."""
        # A thumbnail can't spare the editor's margin: it would be all margin.
        margin = 3 if min(self.width(), self.height()) < 100 else _MARGIN
        side = max(1.0, min(self.width(), self.height()) - 2 * margin)
        return QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)

    def to_widget(self, x: float, y: float) -> QPointF:
        area = self._area()
        return QPointF(area.left() + x * area.width(), area.top() + y * area.height())

    def to_template(self, p: QPointF) -> tuple[float, float]:
        area = self._area()
        x = (p.x() - area.left()) / area.width()
        y = (p.y() - area.top()) / area.height()
        return (min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))

    def keypoint_at(self, p: QPointF) -> int:
        best, best_d = -1, _HIT
        for i, k in enumerate(self._template.keypoints):
            q = self.to_widget(k.x, k.y)
            d = math.hypot(q.x() - p.x(), q.y() - p.y())
            if d < best_d:
                best, best_d = i, d
        return best

    # -- editing --------------------------------------------------------------

    def select(self, index: int) -> None:
        if index != self.selected:
            self.selected = index
            self.selection_changed.emit(index)
            self.update()

    def add_at(self, p: QPointF) -> int:
        """Place a keypoint at widget point ``p`` and select it."""
        x, y = self.to_template(p)
        index = self._template.add_keypoint(x, y)
        self.select(index)
        self.changed.emit(True)
        return index

    def delete_selected(self) -> None:
        if self.selected < 0 or self.locked:
            return
        self._template.remove_keypoint(self.selected)
        self.selected = -1
        self.selection_changed.emit(-1)
        self.changed.emit(True)
        self.update()

    # -- events ---------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._read_only:
            return
        p = event.position()
        hit = self.keypoint_at(p)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if hit >= 0:
            self._press = (hit, p)
            self._dragging = False
            return
        if self.locked:
            self.select(-1)
            return
        self.add_at(p)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        p = event.position()
        if self._press is not None:
            index, start = self._press
            if (
                not self._dragging
                and math.hypot(p.x() - start.x(), p.y() - start.y()) > _DRAG_START
            ):
                self._dragging = True
                self.select(index)
            if self._dragging:
                k = self._template.keypoints[index]
                k.x, k.y = self.to_template(p)
                self.update()
            return
        if not self._read_only:
            over = self.keypoint_at(p) >= 0
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if over else Qt.CursorShape.CrossCursor
            )

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press is None or event.button() != Qt.MouseButton.LeftButton:
            return
        index, _start = self._press
        self._press = None
        if self._dragging:
            self._dragging = False
            self.changed.emit(False)
            return
        # A click on a keypoint toggles its selection.
        self.select(-1 if self.selected == index else index)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
        elif event.key() == Qt.Key.Key_Escape and self.selected >= 0:
            self.select(-1)
        else:
            super().keyPressEvent(event)

    # -- drawing --------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(colors.CANVAS))
        grid = QPen(QColor(colors.BORDER))
        grid.setWidthF(1.0)
        painter.setPen(grid)
        for x in range(0, self.width(), _GRID):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), _GRID):
            painter.drawLine(0, y, self.width(), y)

        kps = self._template.keypoints
        small = self._read_only and min(self.width(), self.height()) < 200
        tiny = min(self.width(), self.height()) < 60
        radius = _POINT_R * (0.35 if tiny else 0.7 if small else 1.0)

        label_font = QFont(painter.font())
        label_font.setPointSizeF(max(6.5, label_font.pointSizeF() * 0.75))
        painter.setFont(label_font)
        for i, k in enumerate(kps):
            c = self.to_widget(k.x, k.y)
            if i == self.selected:
                ring = QPen(QColor(colors.TEXT_PRIMARY))
                ring.setWidthF(2.0)
                painter.setPen(ring)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(c, radius + 4, radius + 4)
            painter.setPen(QPen(QColor(0, 0, 0, 160), 1.0))
            painter.setBrush(QColor(*k.colour))
            painter.drawEllipse(c, radius, radius)
            if not small:
                painter.setPen(QColor(colors.TEXT_SECONDARY))
                painter.drawText(QPointF(c.x() + radius + 3, c.y() + radius + 2), str(i))
        painter.end()


class _KeypointRow(QWidget):
    """Index, colour swatch, name and delete button for one keypoint."""

    renamed = pyqtSignal(int, str)
    recoloured = pyqtSignal(int)
    delete_requested = pyqtSignal(int)
    focused = pyqtSignal(int)

    def __init__(self, index: int, name: str, colour, deletable: bool, parent=None):
        super().__init__(parent)
        self.index = index
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(6)
        number = QLabel(str(index))
        number.setFixedWidth(22)
        number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        muted(number, small=True)
        layout.addWidget(number)
        self.swatch = QToolButton()
        self.swatch.setFixedSize(18, 18)
        self.swatch.setToolTip("Change colour")
        self.swatch.clicked.connect(lambda: self.recoloured.emit(self.index))
        self.set_colour(colour)
        layout.addWidget(self.swatch)
        self.name = QLineEdit(name)
        self.name.textEdited.connect(lambda text: self.renamed.emit(self.index, text))
        self.name.installEventFilter(self)
        layout.addWidget(self.name, 1)
        delete = QToolButton()
        delete.setIcon(lucide_icon("trash-2", colors.TEXT_SECONDARY, 14))
        delete.setToolTip("Delete keypoint" if deletable else "Can't delete: frames are labelled")
        delete.setEnabled(deletable)
        delete.setAutoRaise(True)
        delete.clicked.connect(lambda: self.delete_requested.emit(self.index))
        layout.addWidget(delete)

    def set_colour(self, colour) -> None:
        r, g, b = colour
        self.swatch.setStyleSheet(
            f"QToolButton {{ background: rgb({r},{g},{b}); border: 1px solid {colors.BORDER};"
            f" border-radius: {radius.SMALL}px; }}"
        )

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(
            f"_KeypointRow {{ background: {colors.SURFACE_2}; border-radius: {radius.SMALL}px; }}"
            if selected
            else ""
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, selected)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.name and event.type() == event.Type.FocusIn:
            self.focused.emit(self.index)
        return False


class SkeletonEditorDialog(QDialog):
    """Create or edit a skeleton template.

    Args:
        template: Edited in place on a copy; read the result with
            :meth:`template` after the dialog is accepted.
        on_save: Called with the template when Save is pressed; returns an error
            message to keep the dialog open, or None to close it. Where the
            template goes (custom templates folder, a project) is the caller's.
        locked: Keypoints can't be added or removed -- the project already has
            labels, which store keypoints by position.
        title: Window title, e.g. "New Skeleton Template".
    """

    def __init__(
        self,
        template: SkeletonTemplate | None = None,
        *,
        on_save: Callable[[SkeletonTemplate], str | None] | None = None,
        locked: bool = False,
        title: str = "New Skeleton Template",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(studio_stylesheet())
        self._template = (template or SkeletonTemplate("")).copy()
        self._on_save = on_save
        self._locked = locked
        self._rows: list[_KeypointRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setProperty("studioTitle", True)
        layout.addWidget(heading)
        sub = QLabel(
            "Click the canvas to place keypoints. Drag to reposition."
            if not locked
            else "This project has labelled frames, so keypoints can be renamed, recoloured "
            "and moved, but not added or removed."
        )
        muted(sub)
        sub.setWordWrap(True)
        layout.addWidget(sub)

        top = QHBoxLayout()
        name_col = QVBoxLayout()
        name_col.addWidget(muted(QLabel("Name")))
        self._name = QLineEdit(self._template.name)
        self._name.setPlaceholderText("e.g. Rat, side view")
        name_col.addWidget(self._name)
        top.addLayout(name_col, 2)
        desc_col = QVBoxLayout()
        desc_col.addWidget(muted(QLabel("Description")))
        self._description = QLineEdit(self._template.description)
        self._description.setPlaceholderText("What it's for")
        desc_col.addWidget(self._description)
        top.addLayout(desc_col, 4)
        start_col = QVBoxLayout()
        start_col.addWidget(QLabel(""))
        self._start_from = QToolButton()
        self._start_from.setProperty("studioPicker", True)
        self._start_from.setText("Start from…  ▾")
        self._start_from.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._start_from.setMenu(self._start_menu())
        self._start_from.setEnabled(not locked)
        start_col.addWidget(self._start_from)
        top.addLayout(start_col)
        layout.addLayout(top)

        body = QHBoxLayout()
        canvas_col = QVBoxLayout()
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas = SkeletonCanvas(self._template)
        self._canvas.locked = locked
        self._canvas.setMinimumSize(380, 380)
        self._canvas.changed.connect(self._on_canvas_changed)
        self._canvas.selection_changed.connect(self._on_selection_changed)
        frame_layout.addWidget(self._canvas)
        canvas_col.addWidget(frame, 1)
        self._summary = muted(QLabel(), small=True)
        canvas_col.addWidget(self._summary)
        body.addLayout(canvas_col, 3)

        list_col = QVBoxLayout()
        hint = QLabel(
            "Click to place. Drag to move. Delete removes the selected keypoint."
            if not locked
            else "Drag to move."
        )
        muted(hint, small=True)
        hint.setWordWrap(True)
        list_col.addWidget(hint)
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._rows_host)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_col.addWidget(scroll, 1)
        body.addLayout(list_col, 2)
        layout.addLayout(body, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.setFlat(True)  # secondary, as in the Platform dialog
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self._save = QPushButton("Save")
        self._save.setDefault(True)
        self._save.clicked.connect(self._on_save_clicked)
        buttons.addWidget(self._save)
        layout.addLayout(buttons)

        self._rebuild_rows()
        self.resize(900, 600)

    # -- results ----------------------------------------------------------------

    def template(self) -> SkeletonTemplate:
        self._template.name = self._name.text().strip()
        self._template.description = self._description.text().strip()
        return self._template

    @property
    def canvas(self) -> SkeletonCanvas:
        return self._canvas

    # -- start from ---------------------------------------------------------------

    def _start_menu(self) -> QMenu:
        menu = QMenu(self)
        menu.addSection("Default Templates")
        for template in builtin_templates():
            action = menu.addAction(f"{template.name}  ({len(template.keypoints)})")
            action.triggered.connect(lambda _c=False, t=template: self.start_from(t))
        customs = list_custom_templates()
        if customs:
            menu.addSection("Custom Templates")
            for template in customs:
                action = menu.addAction(f"{template.name}  ({len(template.keypoints)})")
                action.triggered.connect(lambda _c=False, t=template: self.start_from(t))
        menu.addSeparator()
        menu.addAction("Keypoint schema file…").triggered.connect(self._start_from_schema)
        return menu

    def start_from(self, source: SkeletonTemplate) -> None:
        """Replace the keypoints with ``source``'s."""
        if self._template.keypoints:
            answer = QMessageBox.question(
                self,
                "Start from template",
                f"Replace the {len(self._template.keypoints)} keypoints here with "
                f"{source.name}'s {len(source.keypoints)}?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        copy = source.copy()
        self._template.keypoints = copy.keypoints
        if not self._name.text().strip():
            self._name.setText(f"{source.name} (copy)")
        if not self._description.text().strip():
            self._description.setText(source.description)
        self._canvas.select(-1)
        self._on_canvas_changed(True)

    def _start_from_schema(self) -> None:
        from glider.analysis.behavior.keypoint_schema import KeypointSchema, KeypointSchemaError
        from glider.vision.training.templates import TemplateKeypoint, palette_colour

        path, _ = QFileDialog.getOpenFileName(self, "Keypoint schema", "", "JSON (*.json)")
        if not path:
            return
        try:
            schema = KeypointSchema.load(path)
        except KeypointSchemaError as e:
            QMessageBox.warning(self, "Keypoint schema", str(e))
            return
        source = SkeletonTemplate(
            Path(path).stem,
            [
                TemplateKeypoint(k.name, k.x, k.y, palette_colour(i))
                for i, k in enumerate(schema.keypoints)
            ],
        )
        self.start_from(source)

    # -- rows -----------------------------------------------------------------------

    def _rebuild_rows(self) -> None:
        for row in self._rows:
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._rows = []
        for i, k in enumerate(self._template.keypoints):
            row = _KeypointRow(i, k.name, k.colour, deletable=not self._locked)
            row.renamed.connect(self._on_renamed)
            row.recoloured.connect(self._on_recolour)
            row.delete_requested.connect(self._on_delete)
            row.focused.connect(self._canvas.select)
            self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
            self._rows.append(row)
        self._on_selection_changed(self._canvas.selected)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n = len(self._template.keypoints)
        self._summary.setText(f"{n} keypoint{'s' * (n != 1)}")

    def _on_canvas_changed(self, structure: bool) -> None:
        if structure:
            self._rebuild_rows()
        else:
            self._refresh_summary()
        self._canvas.update()

    def _on_selection_changed(self, index: int) -> None:
        for row in self._rows:
            row.set_selected(row.index == index)
        if 0 <= index < len(self._rows):
            parent = self._rows_host.parentWidget()
            while parent is not None and not isinstance(parent, QScrollArea):
                parent = parent.parentWidget()
            if parent is not None:
                parent.ensureWidgetVisible(self._rows[index])

    def _on_renamed(self, index: int, name: str) -> None:
        self._template.keypoints[index].name = name.strip()

    def _on_recolour(self, index: int) -> None:
        k = self._template.keypoints[index]
        colour = QColorDialog.getColor(QColor(*k.colour), self, f"Colour for {k.name}")
        if colour.isValid():
            k.colour = (colour.red(), colour.green(), colour.blue())
            self._rows[index].set_colour(k.colour)
            self._canvas.update()

    def _on_delete(self, index: int) -> None:
        if self._locked:
            return
        self._canvas.select(index)
        self._canvas.delete_selected()

    def _on_save_clicked(self) -> None:
        template = self.template()
        problem = template.problem()
        if problem is None and self._on_save is not None:
            problem = self._on_save(template)
        if problem:
            QMessageBox.warning(self, "Skeleton template", problem[0].upper() + problem[1:] + ".")
            return
        self.accept()


def edit_template(
    parent: QWidget | None, template: SkeletonTemplate | None = None
) -> SkeletonTemplate | None:
    """Open the editor on a custom template (or a new one) and save it.

    Returns the saved template, or None if cancelled.
    """
    original = template.name if template is not None and not template.builtin else None

    def save(t: SkeletonTemplate) -> str | None:
        try:
            save_custom_template(t, replacing=original)
        except (TemplateError, OSError) as e:
            return str(e)
        return None

    dialog = SkeletonEditorDialog(
        template,
        on_save=save,
        title="Edit Skeleton Template" if original else "New Skeleton Template",
        parent=parent,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    return dialog.template()


class _TemplateItem(QWidget):
    """One row of the picker: preview, name, count, description, edit button."""

    def __init__(self, template: SkeletonTemplate, checked: bool, editable: bool, picker):
        super().__init__()
        self._template = template
        self._picker = picker
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        preview = SkeletonCanvas(template, read_only=True)
        preview.setFixedSize(QSize(34, 34))
        preview.setMinimumSize(34, 34)
        layout.addWidget(preview)
        text = QVBoxLayout()
        text.setSpacing(0)
        title = QLabel(
            f"<b>{template.name}</b> <span style='color:{colors.TEXT_SECONDARY}'>"
            f"<sup>{len(template.keypoints)}</sup></span>" + ("  ✓" if checked else "")
        )
        text.addWidget(title)
        if template.description:
            desc = QLabel(template.description)
            muted(desc, small=True)
            text.addWidget(desc)
        layout.addLayout(text, 1)
        if editable:
            edit = QToolButton()
            edit.setIcon(lucide_icon("square-pen", colors.TEXT_SECONDARY, 14))
            edit.setToolTip("Edit template")
            edit.setAutoRaise(True)
            edit.clicked.connect(lambda: picker.edit(template))
            layout.addWidget(edit)
            delete = QToolButton()
            delete.setIcon(lucide_icon("trash-2", colors.TEXT_SECONDARY, 14))
            delete.setToolTip("Delete template")
            delete.setAutoRaise(True)
            delete.clicked.connect(lambda: picker.delete(template))
            layout.addWidget(delete)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._picker.choose(self._template)

    def enterEvent(self, event) -> None:
        self.setStyleSheet(f"_TemplateItem {{ background: {colors.SURFACE_2}; }}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def leaveEvent(self, event) -> None:
        self.setStyleSheet("")


class TemplatePicker(QToolButton):
    """Toolbar button: pick a default or custom skeleton template, or make one.

    Signals:
        template_chosen: The template picked (a fresh copy).
    """

    template_chosen = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current: SkeletonTemplate | None = None
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setProperty("studioPicker", True)
        self._menu = QMenu(self)
        self._menu.aboutToShow.connect(self._populate)
        self.setMenu(self._menu)
        self.choose(builtin_templates()[0])

    @property
    def current(self) -> SkeletonTemplate | None:
        return self._current

    def _populate(self) -> None:
        self._menu.clear()
        current = self._current.name if self._current else ""

        def section(title: str, templates: list[SkeletonTemplate], editable: bool) -> None:
            # A label rather than addSection: most Qt styles draw a section as
            # a bare separator, dropping the title.
            heading = QLabel(title.upper())
            heading.setProperty("textRole", "section")
            heading.setContentsMargins(10, 6, 10, 2)
            header = QWidgetAction(self._menu)
            header.setDefaultWidget(heading)
            header.setEnabled(False)
            self._menu.addAction(header)
            for t in templates:
                action = QWidgetAction(self._menu)
                action.setDefaultWidget(_TemplateItem(t, t.name == current, editable, self))
                self._menu.addAction(action)

        section("Default Templates", builtin_templates(), editable=False)
        customs = list_custom_templates()
        if customs:
            section("Custom Templates", customs, editable=True)
        self._menu.addSeparator()
        self._menu.addAction("+  New Template…").triggered.connect(lambda: self.edit(None))

    def choose(self, template: SkeletonTemplate) -> None:
        self._menu.close()
        self._current = template
        self.setText(f"{template.name}  ·  {len(template.keypoints)} keypoints   ▾")
        self.setToolTip(template.description)
        self.template_chosen.emit(template.copy(template.name))

    def edit(self, template: SkeletonTemplate | None) -> None:
        self._menu.close()
        saved = edit_template(self.window(), template)
        if saved is not None:
            self.choose(saved)

    def delete(self, template: SkeletonTemplate) -> None:
        self._menu.close()
        answer = QMessageBox.question(
            self.window(), "Delete template", f"Delete the custom template '{template.name}'?"
        )
        if answer == QMessageBox.StandardButton.Yes:
            delete_custom_template(template.name)
            if self._current is not None and self._current.name == template.name:
                self.choose(builtin_templates()[0])
