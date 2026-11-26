"""The project's images as a grid that reflows to its width, every frame, its labels drawn on.

Thumbnails are made on demand, the first time a tile is painted, and cached;
a project of a few thousand frames opens at once and only decodes what is on
screen. A label edit invalidates just that frame's thumbnail.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QPointF,
    QRect,
    QSize,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QListView, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from glider.gui.styles import colors
from glider.vision.training.labels import HIDDEN, MISSING
from glider.vision.training.project import FrameRecord, StudioProject

__all__ = ["GalleryModel", "GalleryView", "render_thumbnail"]

#: Narrowest a grid tile gets; the column count is whatever fits.
_MIN_TILE_W = 190
#: Filmstrip tile width.
_STRIP_W = 96
#: Thumbnail width the cache holds; tiles scale it to fit.
_THUMB_W = 360
_CAPTION_H = 22
_PAD = 5
_STATUS_COLOUR = {"labelled": "#4ade80", "todo": "#64748b", "skipped": "#f59e0b"}
_SPLIT_COLOUR = {"train": "#38bdf8", "val": "#c084fc", "test": "#fb923c"}


def frame_status(project: StudioProject, frame: FrameRecord) -> str:
    if frame.skipped:
        return "skipped"
    return "labelled" if project.is_labelled(frame) else "todo"


def _colour(project: StudioProject, index: int) -> QColor:
    from glider.gui.animal_annotator.canvas import keypoint_colour

    if index < len(project.keypoint_colours):
        return QColor(*project.keypoint_colours[index])
    return keypoint_colour(index)


def render_thumbnail(project: StudioProject, frame: FrameRecord, width: int = _THUMB_W) -> QPixmap:
    """The frame at ``width`` pixels wide, with its labelled keypoints drawn on."""
    try:
        data = np.fromfile(str(project.image_path(frame)), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        image = None
    if image is None:
        blank = QPixmap(width, width * 3 // 4)
        blank.fill(QColor(colors.SURFACE_2))
        return blank
    h, w = image.shape[:2]
    scale = width / w
    small = cv2.resize(image, (width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    sh, sw = rgb.shape[:2]
    pixmap = QPixmap.fromImage(QImage(rgb.data, sw, sh, 3 * sw, QImage.Format.Format_RGB888).copy())

    try:
        instances = project.read_instances(frame)
    except ValueError:
        instances = []
    if instances and project.is_segment:
        from glider.gui.animal_annotator.seg_canvas import class_colour

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for inst in instances:
            colour = class_colour(inst.class_id)
            fill = QColor(colour)
            fill.setAlpha(90)
            painter.setPen(QPen(colour, 1.2))
            painter.setBrush(fill)
            painter.drawPolygon(QPolygonF([QPointF(x * sw, y * sh) for x, y in inst.polygon]))
        painter.end()
    elif instances:
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for inst in instances:
            pts = inst.keypoints
            for k, (x, y, v) in enumerate(pts):
                if v == MISSING:
                    continue
                colour = _colour(project, k)
                painter.setPen(QPen(colour if v == HIDDEN else QColor(0, 0, 0, 170), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush if v == HIDDEN else colour)
                painter.drawEllipse(round(x * sw) - 3, round(y * sh) - 3, 6, 6)
        painter.end()
    return pixmap


class GalleryModel(QAbstractListModel):
    """One row per project frame; the thumbnail is the decoration."""

    StatusRole = Qt.ItemDataRole.UserRole + 1
    #: (frame number, video key) for the tile caption.
    CaptionRole = Qt.ItemDataRole.UserRole + 2
    SplitRole = Qt.ItemDataRole.UserRole + 3
    #: Instances (animals or outlines) on the frame, cached with the thumbnail.
    AnimalCountRole = Qt.ItemDataRole.UserRole + 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: StudioProject | None = None
        self._cache: dict[str, QPixmap] = {}
        self._counts: dict[str, int] = {}

    def set_project(self, project: StudioProject | None) -> None:
        self.beginResetModel()
        self._project = project
        self._cache.clear()
        self._counts.clear()
        self.endResetModel()

    def reload(self) -> None:
        """Frames were added or removed: rebuild, keeping cached thumbnails."""
        self.beginResetModel()
        if self._project is not None:
            names = {f.name for f in self._project.frames}
            self._cache = {k: v for k, v in self._cache.items() if k in names}
            self._counts = {k: v for k, v in self._counts.items() if k in names}
        self.endResetModel()

    def invalidate(self, row: int) -> None:
        """One frame's labels or status changed: redraw its tile."""
        if self._project is None or not 0 <= row < len(self._project.frames):
            return
        name = self._project.frames[row].name
        self._cache.pop(name, None)
        self._counts.pop(name, None)
        index = self.index(row)
        self.dataChanged.emit(index, index)

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: B008 - Qt's signature
        if parent.isValid() or self._project is None:
            return 0
        return len(self._project.frames)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if self._project is None or not index.isValid():
            return None
        frame = self._project.frames[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return frame.name
        if role == Qt.ItemDataRole.ToolTipRole:
            video = self._project.videos.get(frame.video, frame.video)
            return f"{frame.name}\nFrame {frame.frame} of {video}\nClick to label"
        if role == self.StatusRole:
            return frame_status(self._project, frame)
        if role == self.CaptionRole:
            return (frame.frame, frame.video)
        if role == self.SplitRole:
            return frame.split
        if role == self.AnimalCountRole:
            if frame.name not in self._counts:
                try:
                    self._counts[frame.name] = len(self._project.read_instances(frame))
                except ValueError:
                    self._counts[frame.name] = 0
            return self._counts[frame.name]
        if role == Qt.ItemDataRole.DecorationRole:
            pixmap = self._cache.get(frame.name)
            if pixmap is None:
                pixmap = self._cache[frame.name] = render_thumbnail(self._project, frame)
            return pixmap
        return None


class _TileDelegate(QStyledItemDelegate):
    """Image with a split badge and status chip, caption underneath, accent border when selected.

    In a filmstrip only the image and a status dot are drawn.
    """

    def __init__(self, view: GalleryView):
        super().__init__(view)
        self._view = view

    def sizeHint(self, option, index) -> QSize:
        return self._view.gridSize()

    @staticmethod
    def _chip(painter: QPainter, rect: QRect, text: str, colour: str, *, right: bool) -> None:
        width = painter.fontMetrics().horizontalAdvance(text) + 10
        box = QRect(0, rect.top() + 4, width, painter.fontMetrics().height() + 4)
        box.moveLeft(rect.right() - 4 - width if right else rect.left() + 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, 3, 3)
        painter.setPen(QColor(colour))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        strip = self._view.strip
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        cell = option.rect.adjusted(_PAD, _PAD, -_PAD, -_PAD)
        caption_h = 0 if strip else _CAPTION_H
        image_rect = QRect(cell.left(), cell.top(), cell.width(), cell.height() - caption_h)
        painter.fillRect(image_rect, QColor(colors.SURFACE_1))

        pixmap: QPixmap | None = index.data(Qt.ItemDataRole.DecorationRole)
        if pixmap is not None and not pixmap.isNull():
            scaled = pixmap.size().scaled(image_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRect(0, 0, scaled.width(), scaled.height())
            target.moveCenter(image_rect.center())
            painter.drawPixmap(target, pixmap)

        status = index.data(GalleryModel.StatusRole)
        if status == "skipped":
            painter.fillRect(image_rect, QColor(0, 0, 0, 120))

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            pen = QPen(QColor(colors.ACCENT if selected else colors.TEXT_SECONDARY))
            pen.setWidthF(2.0 if selected else 1.0)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(image_rect.adjusted(0, 0, -1, -1))

        font = QFont(option.font)
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.9))
        painter.setFont(font)

        if strip:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(_STATUS_COLOUR.get(status, colors.TEXT_SECONDARY)))
            painter.drawEllipse(image_rect.right() - 11, image_rect.top() + 4, 7, 7)
            painter.restore()
            return

        split = index.data(GalleryModel.SplitRole)
        if split:
            self._chip(
                painter, image_rect, split.upper(), _SPLIT_COLOUR.get(split, "#ffffff"), right=False
            )
        count = index.data(GalleryModel.AnimalCountRole)
        noun = "outline" if self._view.outlines else "animal"
        if status == "skipped":
            chip = "skipped"
        elif status == "labelled":
            chip = f"\u2713 {count} {noun}{'' if count == 1 else 's'}"
        else:
            chip = "\u25cb not yet"
        self._chip(painter, image_rect, chip, _STATUS_COLOUR.get(status, "#ffffff"), right=True)

        caption = QRect(cell.left(), image_rect.bottom() + 3, cell.width(), _CAPTION_H - 3)
        number, video = index.data(GalleryModel.CaptionRole)
        text = painter.fontMetrics().elidedText(
            f"{Path(video).name} \u00b7 f{number}", Qt.TextElideMode.ElideMiddle, caption.width()
        )
        painter.setPen(QColor(colors.TEXT_SECONDARY))
        painter.drawText(caption, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        painter.restore()


class GalleryView(QListView):
    """Every frame in a project as tiles that reflow to the width, or one filmstrip row.

    Signals:
        frame_activated: Row (index into ``project.frames``) plain-clicked or
            Enter pressed: open it for labelling. Cmd/Ctrl/Shift-click selects
            instead.
        selection_changed: The set of selected rows changed.
    """

    frame_activated = pyqtSignal(int)
    selection_changed = pyqtSignal()

    def __init__(self, parent=None, *, strip: bool = False):
        super().__init__(parent)
        self.strip = strip
        #: Segmentation project: chips say "outline" rather than "animal".
        self.outlines = False
        self._filter: tuple[str | None, str | None, str | None] = (None, None, None)
        self._press_row: int | None = None
        self.gallery_model = GalleryModel(self)
        self.setItemDelegate(_TileDelegate(self))
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(not strip)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.setMouseTracking(True)
        if strip:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setFixedHeight(
                _STRIP_W * 3 // 4 + 2 * _PAD + self.horizontalScrollBar().sizeHint().height() + 4
            )
        else:
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.empty_text = "No images yet.\n\nAdd videos and extract frames to start labelling."
        self._attach(self.gallery_model)
        self._fit_columns()

    def _attach(self, model: GalleryModel) -> None:
        self.setModel(model)
        model.modelReset.connect(self.apply_filter)
        self.selectionModel().selectionChanged.connect(lambda *_: self.selection_changed.emit())
        self.apply_filter()

    def share_model(self, other: GalleryView) -> None:
        """Show ``other``'s frames; this view's filter and selection stay its own."""
        self.gallery_model = other.gallery_model
        self._attach(self.gallery_model)

    def set_project(self, project: StudioProject | None) -> None:
        self.outlines = bool(project and project.is_segment)
        self.gallery_model.set_project(project)

    def select_row(self, row: int) -> None:
        if 0 <= row < self.gallery_model.rowCount():
            index = self.gallery_model.index(row)
            self.setCurrentIndex(index)
            self.scrollTo(index)

    # -- filtering and selection ---------------------------------------

    def set_filter(self, status: str | None, split: str | None, video: str | None) -> None:
        """Show only rows matching every given criterion; ``None`` is any."""
        self._filter = (status, split, video)
        self.apply_filter()

    @property
    def current_filter(self) -> tuple[str | None, str | None, str | None]:
        """(status, split, video) as last given to :meth:`set_filter`."""
        return self._filter

    def apply_filter(self, *_) -> None:
        """Re-apply the current filter. Label edits do not, so a frame being labelled stays put."""
        model = self.gallery_model
        status, split, video = self._filter
        for row in range(model.rowCount()):
            index = model.index(row)
            keep = (
                (status is None or index.data(GalleryModel.StatusRole) == status)
                and (split is None or index.data(GalleryModel.SplitRole) == split)
                and (video is None or index.data(GalleryModel.CaptionRole)[1] == video)
            )
            self.setRowHidden(row, not keep)

    def visible_rows(self) -> list[int]:
        return [r for r in range(self.gallery_model.rowCount()) if not self.isRowHidden(r)]

    def selected_rows(self) -> list[int]:
        """Selected rows the filter shows; a hidden selected row is never acted on."""
        return sorted(
            i.row()
            for i in self.selectionModel().selectedIndexes()
            if not self.isRowHidden(i.row())
        )

    def mousePressEvent(self, event) -> None:
        index = self.indexAt(event.position().toPoint())
        plain = (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        )
        self._press_row = index.row() if plain and index.isValid() else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        # Only a press and release on the same tile opens it: not the end of a
        # drag, and not the second half of a double-click (that has no press).
        row, self._press_row = self._press_row, None
        if row is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if event.modifiers() != Qt.KeyboardModifier.NoModifier:
            return
        index = self.indexAt(event.position().toPoint())
        if index.isValid() and index.row() == row:
            self.frame_activated.emit(row)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.currentIndex().isValid():
            self.frame_activated.emit(self.currentIndex().row())
            return
        super().keyPressEvent(event)

    # -- layout ----------------------------------------------------------

    def _fit_columns(self) -> None:
        if self.strip:
            h = _STRIP_W * 3 // 4 + 2 * _PAD
            self.setGridSize(QSize(_STRIP_W + 2 * _PAD, h))
            return
        # Size for the viewport *with* its scrollbar showing. The scrollbar
        # appears only once the tiles overflow, after they were sized; sized
        # without it, the last tile no longer fits and wraps to a new row.
        width = self.viewport().width()
        if not self.verticalScrollBar().isVisible():
            width -= self.style().pixelMetric(self.style().PixelMetric.PM_ScrollBarExtent)
        width = max(_MIN_TILE_W, width - 1)
        cell_w = width // max(1, width // _MIN_TILE_W)
        image_h = round((cell_w - 2 * _PAD) * 3 / 4)
        self.setGridSize(QSize(cell_w, image_h + _CAPTION_H + 2 * _PAD))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_columns()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.gallery_model.rowCount() == 0:
            painter = QPainter(self.viewport())
            painter.setPen(QColor(colors.TEXT_SECONDARY))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            painter.end()
