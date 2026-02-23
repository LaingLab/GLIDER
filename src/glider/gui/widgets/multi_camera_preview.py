"""
Multi-Camera Preview Widget - Grid layout for multiple camera previews.

Displays multiple camera feeds in an adaptive grid layout,
with visual indicators for the primary camera and recording status.
"""

import logging

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

logger = logging.getLogger(__name__)


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

    def __init__(self, camera_id: str, is_primary: bool = False, parent=None):
        super().__init__(parent)
        self._camera_id = camera_id
        self._is_primary = is_primary
        self._is_recording = False

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
        self._camera_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        header.addWidget(self._camera_label)

        header.addStretch()

        # Primary indicator
        self._primary_indicator = QLabel("PRIMARY")
        self._primary_indicator.setStyleSheet("""
            QLabel {
                background-color: #3498db;
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 9px;
                font-weight: bold;
            }
        """)
        self._primary_indicator.setVisible(self._is_primary)
        header.addWidget(self._primary_indicator)

        # Recording indicator
        self._recording_indicator = QLabel("REC")
        self._recording_indicator.setStyleSheet("""
            QLabel {
                background-color: #c0392b;
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 9px;
                font-weight: bold;
            }
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
        self._preview.setStyleSheet("""
            QLabel {
                background-color: #0d0d1a;
                border-radius: 4px;
            }
        """)
        self._preview.setText("No Feed")
        layout.addWidget(self._preview, 1)

        # FPS label
        self._fps_label = QLabel("-- FPS")
        self._fps_label.setStyleSheet("color: #888; font-size: 10px;")
        self._fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(self._fps_label)

        # Make entire tile clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def _update_style(self) -> None:
        """Update tile style based on state."""
        if self._is_primary:
            self.setStyleSheet("""
                CameraPreviewTile {
                    background-color: #1a1a2e;
                    border: 2px solid #3498db;
                    border-radius: 8px;
                }
            """)
        else:
            self.setStyleSheet("""
                CameraPreviewTile {
                    background-color: #1a1a2e;
                    border: 1px solid #2d2d44;
                    border-radius: 8px;
                }
                CameraPreviewTile:hover {
                    border: 1px solid #3498db;
                }
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


class MultiCameraPreviewWidget(QWidget):
    """
    Grid layout widget showing all connected cameras.

    Layout adapts based on camera count:
    - 1 camera: Single view
    - 2 cameras: 1x2 horizontal
    - 3-4 cameras: 2x2 grid
    - 5-6 cameras: 2x3 grid
    - 7+ cameras: 3x3 grid

    Thread Safety:
    - All methods must be called from the main Qt thread
    - Frame updates from CameraPanel are marshaled to main thread via signals
    """

    primary_changed = pyqtSignal(str)  # camera_id of new primary

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tiles: dict[str, CameraPreviewTile] = {}
        self._primary_id: str | None = None

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
        self._placeholder.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 14px;
                padding: 40px;
            }
        """)
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

    def _on_tile_clicked(self, camera_id: str) -> None:
        """Handle tile click to change primary camera."""
        if camera_id != self._primary_id:
            self.set_primary(camera_id)
            self.primary_changed.emit(camera_id)

    def _reflow_grid(self) -> None:
        """Recalculate grid layout based on camera count."""
        # Remove all from grid
        for tile in self._tiles.values():
            self._grid_layout.removeWidget(tile)

        count = len(self._tiles)
        if count == 0:
            return

        # Determine grid dimensions
        if count == 1:
            cols, _rows = 1, 1
        elif count == 2:
            cols, _rows = 2, 1
        elif count <= 4:
            cols, _rows = 2, 2
        elif count <= 6:
            cols, _rows = 3, 2
        else:
            cols, _rows = 3, 3

        # Add tiles to grid
        camera_ids = list(self._tiles.keys())
        for i, camera_id in enumerate(camera_ids):
            row = i // cols
            col = i % cols
            self._grid_layout.addWidget(self._tiles[camera_id], row, col)

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
