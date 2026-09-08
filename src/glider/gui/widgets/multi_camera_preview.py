"""
Multi-Camera Preview Widget - Grid layout for multiple camera previews.

Displays multiple camera feeds in an adaptive grid layout,
with visual indicators for the primary camera and recording status.
"""

import logging
import math

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from glider.gui.styles import colors, radius

logger = logging.getLogger(__name__)


def grid_columns(count: int) -> int:
    """Columns to lay *count* preview tiles out in.

    Square-ish, biased wide: monitors are wider than they are tall, and a tall
    grid shrinks every tile to fit the short axis. The small counts are spelled
    out because ``ceil(sqrt())`` would render two cameras as a single column,
    which reads as a mistake even though it tiles correctly.

    Grows without bound rather than pinning at 3x3 as it used to - past nine
    cameras the old layout left the extras out of the grid entirely, so they
    recorded but never appeared.
    """
    if count <= 1:
        return 1
    if count <= 4:
        return 2
    if count <= 9:
        return 3
    return math.ceil(math.sqrt(count))


class CameraPreviewTile(QFrame):
    """
    Single camera preview tile in the grid.

    Shows camera feed with label, primary indicator,
    and recording status.

    Thread Safety:
    - All methods must be called from the main Qt thread
    - update_frame() creates QPixmap which is not thread-safe
    """

    clicked = pyqtSignal(str)  # camera_id when clicked
    rename_requested = pyqtSignal(str)  # camera_id double-clicked

    def __init__(self, camera_id: str, is_primary: bool = False, parent=None):
        super().__init__(parent)
        self._camera_id = camera_id
        self._is_primary = is_primary
        self._is_recording = False
        self._is_focused = False

        self._setup_ui()
        self._update_style()

    def _setup_ui(self) -> None:
        """Set up the tile UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header with camera label and indicators
        header = QHBoxLayout()

        self._camera_label = QLabel(f"Camera {self._camera_id.replace('cam_', '')}")
        self._camera_label.setToolTip("Double-click the tile to rename this camera")
        self._camera_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        header.addWidget(self._camera_label)

        header.addStretch()

        # Primary indicator
        self._primary_indicator = QLabel("PRIMARY")
        self._primary_indicator.setStyleSheet(f"""
            QLabel {{
                background-color: {colors.ACCENT};
                color: white;
                padding: 2px 6px;
                border-radius: {radius.SMALL}px;
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        self._primary_indicator.setVisible(self._is_primary)
        header.addWidget(self._primary_indicator)

        # Recording indicator
        self._recording_indicator = QLabel("REC")
        self._recording_indicator.setStyleSheet(f"""
            QLabel {{
                background-color: {colors.ERROR};
                color: white;
                padding: 2px 6px;
                border-radius: {radius.SMALL}px;
                font-size: 9px;
                font-weight: bold;
            }}
        """)
        self._recording_indicator.hide()
        header.addWidget(self._recording_indicator)

        layout.addLayout(header)

        # Preview area
        self._preview = QLabel()
        self._preview.setMinimumSize(160, 120)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setScaledContents(False)
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setStyleSheet(f"""
            QLabel {{
                background-color: {colors.CANVAS};
                border-radius: {radius.SMALL}px;
            }}
        """)
        self._preview.setText("No Feed")
        layout.addWidget(self._preview, 1)

        # FPS label
        self._fps_label = QLabel("-- FPS")
        self._fps_label.setProperty("textRole", "muted")
        self._fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._fps_label)

        # Make entire tile clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _update_style(self) -> None:
        """Update tile style based on state."""
        if self._is_focused:
            self.setStyleSheet(f"""
                CameraPreviewTile {{
                    background-color: {colors.SURFACE_2};
                    border: 2px solid {colors.STATE_OK};
                    border-radius: {radius.MEDIUM}px;
                }}
            """)
        elif self._is_primary:
            self.setStyleSheet(f"""
                CameraPreviewTile {{
                    background-color: {colors.SURFACE_2};
                    border: 2px solid {colors.ACCENT};
                    border-radius: {radius.MEDIUM}px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                CameraPreviewTile {{
                    background-color: {colors.SURFACE_2};
                    border: 1px solid {colors.BORDER};
                    border-radius: {radius.MEDIUM}px;
                }}
                CameraPreviewTile:hover {{
                    border: 1px solid {colors.ACCENT};
                }}
            """)

    @property
    def camera_id(self) -> str:
        """Camera ID for this tile."""
        return self._camera_id

    def set_primary(self, is_primary: bool) -> None:
        """Update primary indicator."""
        self._is_primary = is_primary
        self._primary_indicator.setVisible(is_primary)
        self._update_style()

    def set_display_name(self, name: str) -> None:
        """Caption this tile with the operator's name for the arena.

        Falls back to the camera id rather than rendering an empty header,
        because a tile with no caption is worse than one labelled cam_3.
        """
        self._camera_label.setText(name or self._camera_id)

    def set_focused(self, is_focused: bool) -> None:
        """Mark this tile as the enlarged one."""
        self._is_focused = is_focused
        self._update_style()

    def set_recording(self, is_recording: bool) -> None:
        """Update recording indicator."""
        self._is_recording = is_recording
        self._recording_indicator.setVisible(is_recording)

    def update_frame(self, frame: np.ndarray) -> None:
        """Update preview with new frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

        # Scale to fit while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(
            self._preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def update_fps(self, fps: float) -> None:
        """Update FPS display."""
        self._fps_label.setText(f"{fps:.1f} FPS")

    def show_placeholder(self, text: str = "No Feed") -> None:
        """Show placeholder text."""
        self._preview.clear()
        self._preview.setText(text)

    def mousePressEvent(self, event):
        """Handle mouse click to select as primary."""
        self.clicked.emit(self._camera_id)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click renames -- the gesture people already try."""
        self.rename_requested.emit(self._camera_id)
        super().mouseDoubleClickEvent(event)


class MultiCameraPreviewWidget(QWidget):
    """
    Grid layout widget showing all connected cameras.

    Layout adapts based on camera count; see :func:`grid_columns`. It keeps
    growing past nine, which the fixed 3x3 it replaced did not - the extra
    tiles were built and fed frames but never placed, so a 16-camera rig
    recorded sixteen files and showed nine.

    Thread Safety:
    - All methods must be called from the main Qt thread
    - Frame updates from CameraPanel are marshaled to main thread via signals
    """

    primary_changed = pyqtSignal(str)  # camera_id of new primary
    focus_changed = pyqtSignal(str)  # camera_id now focused, or "" for the grid
    rename_requested = pyqtSignal(str)  # camera_id the operator wants to rename

    #: How many columns the focused tile spans. The rest of the cameras sit
    #: in a single column beside it, still live -- the point of focusing is
    #: to look closely at one arena *without* losing sight of the others,
    #: which is the whole reason to run eight at once.
    FOCUS_SPAN = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tiles: dict[str, CameraPreviewTile] = {}
        self._primary_id: str | None = None
        self._focused_id: str | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(8)

        # Container for grid
        self._grid_container = QWidget()
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setSpacing(8)

        self._main_layout.addWidget(self._grid_container)

        # Placeholder when no cameras
        self._placeholder = QLabel("No cameras connected")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setProperty("textRole", "disabled")
        self._main_layout.addWidget(self._placeholder)

    def add_camera(self, camera_id: str, is_primary: bool = False) -> CameraPreviewTile:
        """
        Add a camera tile to the grid.

        Args:
            camera_id: Unique camera identifier
            is_primary: Whether this is the primary camera

        Returns:
            The created CameraPreviewTile
        """
        if camera_id in self._tiles:
            return self._tiles[camera_id]

        tile = CameraPreviewTile(camera_id, is_primary)
        tile.clicked.connect(self._on_tile_clicked)
        tile.rename_requested.connect(self._on_tile_rename)
        self._tiles[camera_id] = tile

        if is_primary:
            self._primary_id = camera_id

        self._reflow_grid()
        self._update_placeholder_visibility()

        logger.debug(f"Added camera tile: {camera_id}")
        return tile

    def remove_camera(self, camera_id: str) -> None:
        """Remove a camera tile from the grid."""
        if camera_id not in self._tiles:
            return

        tile = self._tiles.pop(camera_id)
        self._grid_layout.removeWidget(tile)
        tile.deleteLater()

        # Update primary if needed
        if self._primary_id == camera_id and self._tiles:
            new_primary = next(iter(self._tiles.keys()))
            self.set_primary(new_primary)

        self._reflow_grid()
        self._update_placeholder_visibility()

        logger.debug(f"Removed camera tile: {camera_id}")

    def remove_all_cameras(self) -> None:
        """Remove all camera tiles."""
        for camera_id in list(self._tiles.keys()):
            self.remove_camera(camera_id)

    def update_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """Update specific camera's preview."""
        tile = self._tiles.get(camera_id)
        if tile:
            tile.update_frame(frame)

    def update_fps(self, camera_id: str, fps: float) -> None:
        """Update specific camera's FPS display."""
        tile = self._tiles.get(camera_id)
        if tile:
            tile.update_fps(fps)

    def set_primary(self, camera_id: str) -> None:
        """Set which camera is the primary."""
        if camera_id not in self._tiles:
            return

        # Update old primary
        if self._primary_id and self._primary_id in self._tiles:
            self._tiles[self._primary_id].set_primary(False)

        # Set new primary
        self._primary_id = camera_id
        self._tiles[camera_id].set_primary(True)

    def set_recording(self, recording: bool) -> None:
        """Update recording indicator on all tiles."""
        for tile in self._tiles.values():
            tile.set_recording(recording)

    def set_camera_recording(self, camera_id: str, recording: bool) -> None:
        """Update recording indicator for specific camera."""
        tile = self._tiles.get(camera_id)
        if tile:
            tile.set_recording(recording)

    def set_display_name(self, camera_id: str, name: str) -> None:
        """Caption one tile."""
        tile = self._tiles.get(camera_id)
        if tile is not None:
            tile.set_display_name(name)

    def _on_tile_rename(self, camera_id: str) -> None:
        self.rename_requested.emit(camera_id)

    def _on_tile_clicked(self, camera_id: str) -> None:
        """Handle tile click to change primary camera."""
        if camera_id != self._primary_id:
            self.set_primary(camera_id)
            self.primary_changed.emit(camera_id)

    @property
    def focused_camera_id(self) -> str | None:
        """The camera currently enlarged, or None when showing an even grid."""
        return self._focused_id

    def set_focus(self, camera_id: str | None) -> None:
        """Enlarge one camera, or pass None to go back to the even grid.

        Focusing a camera that is not there is a no-op rather than an error:
        the id can go stale between a click and a camera being unplugged.
        """
        if camera_id is not None and camera_id not in self._tiles:
            return
        if camera_id == self._focused_id:
            return
        self._focused_id = camera_id
        for cid, tile in self._tiles.items():
            tile.set_focused(cid == camera_id)
        self._reflow_grid()
        self.focus_changed.emit(camera_id or "")

    def toggle_focus(self, camera_id: str) -> None:
        """Focus this camera, or unfocus it if it is already focused."""
        self.set_focus(None if camera_id == self._focused_id else camera_id)

    def _reflow_grid(self) -> None:
        """Recalculate grid layout based on camera count."""
        # Remove all from grid
        for tile in self._tiles.values():
            self._grid_layout.removeWidget(tile)
        for column in range(self._grid_layout.columnCount()):
            self._grid_layout.setColumnStretch(column, 0)
        for row in range(self._grid_layout.rowCount()):
            self._grid_layout.setRowStretch(row, 0)

        count = len(self._tiles)
        if count == 0:
            return

        if self._focused_id in self._tiles and count > 1:
            self._reflow_focused()
            return

        cols = grid_columns(count)

        # Add tiles to grid
        camera_ids = list(self._tiles.keys())
        for i, camera_id in enumerate(camera_ids):
            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(self._tiles[camera_id], row, col)
            self._grid_layout.setColumnStretch(col, 1)
            self._grid_layout.setRowStretch(row, 1)

    def _reflow_focused(self) -> None:
        """One large tile, the rest in a strip beside it.

        The others keep receiving frames -- they are the same widgets, just
        placed differently -- so nothing stops while a camera is focused.
        """
        others = [cid for cid in self._tiles if cid != self._focused_id]
        rows = max(len(others), 1)

        self._grid_layout.addWidget(self._tiles[self._focused_id], 0, 0, rows, self.FOCUS_SPAN)
        for row, camera_id in enumerate(others):
            self._grid_layout.addWidget(self._tiles[camera_id], row, self.FOCUS_SPAN)
            self._grid_layout.setRowStretch(row, 1)

        for column in range(self.FOCUS_SPAN):
            self._grid_layout.setColumnStretch(column, 1)
        # The strip is deliberately narrow: it is for noticing that another
        # arena needs attention, not for watching it.
        self._grid_layout.setColumnStretch(self.FOCUS_SPAN, 1)

    def _update_placeholder_visibility(self) -> None:
        """Show/hide placeholder based on camera count."""
        has_cameras = len(self._tiles) > 0
        self._placeholder.setVisible(not has_cameras)
        self._grid_container.setVisible(has_cameras)

    @property
    def camera_count(self) -> int:
        """Number of camera tiles."""
        return len(self._tiles)

    @property
    def primary_camera_id(self) -> str | None:
        """ID of the primary camera."""
        return self._primary_id

    def get_tile(self, camera_id: str) -> CameraPreviewTile | None:
        """Get a specific camera tile."""
        return self._tiles.get(camera_id)
