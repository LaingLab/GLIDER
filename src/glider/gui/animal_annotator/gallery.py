"""The project's images as a grid: four across, every frame, keypoints drawn on.


Thumbnails are made on demand, the first time a tile is painted, and cached;
a project of a few thousand frames opens at once and only decodes what is on
screen. A label edit invalidates just that frame's thumbnail.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QAbstractListModel, QModelIndex, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QListView, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from glider.gui.styles import colors
from glider.vision.training.labels import HIDDEN, MISSING
from glider.vision.training.project import FrameRecord, StudioProject

__all__ = ["COLUMNS", "GalleryModel", "GalleryView", "render_thumbnail"]

COLUMNS = 4
#: Thumbnail width the cache holds; tiles scale it to fit.
_THUMB_W = 360
_CAPTION_H = 22
_PAD = 5
_STATUS_COLOUR = {"labelled": "#4ade80", "todo": "#64748b", "skipped": "#f59e0b"}
_STATUS_TEXT = {"labelled": "Labelled", "todo": "To do", "skipped": "Skipped"}


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
    if instances:
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: StudioProject | None = None
        self._cache: dict[str, QPixmap] = {}

    def set_project(self, project: StudioProject | None) -> None:
        self.beginResetModel()
        self._project = project
        self._cache.clear()
        self.endResetModel()

    def reload(self) -> None:
        """Frames were added or removed: rebuild, keeping cached thumbnails."""
        self.beginResetModel()
        if self._project is not None:
            names = {f.name for f in self._project.frames}
            self._cache = {k: v for k, v in self._cache.items() if k in names}
        self.endResetModel()

    def invalidate(self, row: int) -> None:
        """One frame's labels or status changed: redraw its tile."""
        if self._project is None or not 0 <= row < len(self._project.frames):
            return
        self._cache.pop(self._project.frames[row].name, None)
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
            return f"{frame.name}\nFrame {frame.frame} of {video}\nDouble-click to label"
        if role == self.StatusRole:
            return frame_status(self._project, frame)
        if role == self.CaptionRole:
            return (frame.frame, frame.video)
        if role == Qt.ItemDataRole.DecorationRole:
            pixmap = self._cache.get(frame.name)
            if pixmap is None:
                pixmap = self._cache[frame.name] = render_thumbnail(self._project, frame)
            return pixmap
        return None


class _TileDelegate(QStyledItemDelegate):
    """Image on top, status and name underneath, accent border when selected."""

    def __init__(self, view: GalleryView):
        super().__init__(view)
        self._view = view

    def sizeHint(self, option, index) -> QSize:
        return self._view.gridSize()

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        cell = option.rect.adjusted(_PAD, _PAD, -_PAD, -_PAD)
        image_rect = QRect(cell.left(), cell.top(), cell.width(), cell.height() - _CAPTION_H)
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

        caption = QRect(cell.left(), image_rect.bottom() + 3, cell.width(), _CAPTION_H - 3)
        dot = QColor(_STATUS_COLOUR.get(status, colors.TEXT_SECONDARY))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot)
        painter.drawEllipse(caption.left() + 1, caption.center().y() - 3, 7, 7)
        font = QFont(option.font)
        font.setPointSizeF(max(7.0, font.pointSizeF() * 0.9))
        painter.setFont(font)
        number, video = index.data(GalleryModel.CaptionRole)
        text_rect = caption.adjusted(13, 0, 0, 0)
        title = f"Frame {number}"
        painter.setPen(QColor(colors.TEXT_PRIMARY))
        painter.drawText(
            text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, title
        )
        used = painter.fontMetrics().horizontalAdvance(title) + 10
        video_rect = text_rect.adjusted(used, 0, 0, 0)
        video_text = painter.fontMetrics().elidedText(
            video, Qt.TextElideMode.ElideMiddle, video_rect.width()
        )
        painter.setPen(QColor(colors.TEXT_SECONDARY))
        painter.drawText(
            video_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, video_text
        )
        painter.restore()


class GalleryView(QListView):
    """Every frame in a project, :data:`COLUMNS` across, as many rows as it takes.

    Signals:
        frame_activated: Row (index into ``project.frames``) double-clicked or
            Enter pressed: open it for labelling.
    """

    frame_activated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gallery_model = GalleryModel(self)
        self.setModel(self.gallery_model)
        self.setItemDelegate(_TileDelegate(self))
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.setMouseTracking(True)
        self.empty_text = "No images yet.\n\nAdd videos and extract frames to start labelling."
        self.activated.connect(lambda index: self.frame_activated.emit(index.row()))
        self.doubleClicked.connect(lambda index: self.frame_activated.emit(index.row()))
        self._fit_columns()

    def set_project(self, project: StudioProject | None) -> None:
        self.gallery_model.set_project(project)

    def select_row(self, row: int) -> None:
        if 0 <= row < self.gallery_model.rowCount():
            index = self.gallery_model.index(row)
            self.setCurrentIndex(index)
            self.scrollTo(index)

    def _fit_columns(self) -> None:
        # Size for the viewport *with* its scrollbar showing. The scrollbar
        # appears only once the tiles overflow, after they were sized; sized
        # without it, the fourth tile no longer fits and wraps to a new row.
        width = self.viewport().width()
        if not self.verticalScrollBar().isVisible():
            width -= self.style().pixelMetric(self.style().PixelMetric.PM_ScrollBarExtent)
        width = max(COLUMNS * 40, width - 1)
        cell_w = width // COLUMNS
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
