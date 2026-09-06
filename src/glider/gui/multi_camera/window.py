"""Multi-Camera Recording - a tool window for running a camera array.

The camera panel already has a Multi-Camera checkbox, and it stays: for two to
four cameras, previewing inline beside the node graph is the right thing. This
window is for the other case. A sixteen-camera grid needs a whole window, often
on a second monitor, and it needs a status table beside it - which is the part
the panel has nowhere to put.

Follows the same shape as the other tool windows (see
:mod:`glider.gui.pose_batch.window`): a QMainWindow held on the MainWindow and
re-surfaced rather than rebuilt, so state survives closing it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from glider.gui.multi_camera.status_table import CameraStatusTable
from glider.gui.styles import colors
from glider.gui.widgets.multi_camera_preview import MultiCameraPreviewWidget
from glider.gui.widgets.tool_ui import Card, apply_tool_theme, hint, set_button_role

if TYPE_CHECKING:
    from glider.vision.multi_camera_manager import MultiCameraManager
    from glider.vision.multi_video_recorder import MultiVideoRecorder

logger = logging.getLogger(__name__)

#: How often the status table refreshes. Fast enough that a stalled camera is
#: obvious within a second or two, slow enough that sixteen rows of Qt item
#: updates never compete with the capture threads for the GIL.
_POLL_MS = 500


class MultiCameraWindow(QMainWindow):
    """Preview, record and monitor every camera in the rig at once."""

    def __init__(
        self,
        multi_camera_manager: MultiCameraManager,
        recorder: MultiVideoRecorder | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._manager = multi_camera_manager
        self._recorder = recorder

        self.setWindowTitle("Multi-Camera Recording")
        self.resize(1400, 900)
        self._build_ui()
        apply_tool_theme(self)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_status)
        self._timer.start(_POLL_MS)

        self.refresh_cameras()

    # ------------------------------------------------------------------
    # ui
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.record_button = QPushButton("Record All")
        self.record_button.setToolTip("Start recording on every connected camera")
        self.record_button.clicked.connect(self.start_recording)
        set_button_role(self.record_button, "primary")
        controls.addWidget(self.record_button)

        self.stop_button = QPushButton("Stop All")
        self.stop_button.clicked.connect(self.stop_recording)
        controls.addWidget(self.stop_button)

        refresh = QPushButton("Refresh Cameras")
        refresh.setToolTip("Re-read the camera list from the manager")
        refresh.clicked.connect(self.refresh_cameras)
        set_button_role(refresh, "ghost")
        controls.addWidget(refresh)

        controls.addStretch(1)
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet(f"color: {colors.ERROR}; font-weight: bold;")
        controls.addWidget(self.warning_label)
        layout.addLayout(controls)

        split = QSplitter(Qt.Orientation.Horizontal)

        preview_card = Card("Live preview")
        self.preview = MultiCameraPreviewWidget()
        preview_card.add(self.preview, 1)
        split.addWidget(preview_card)

        status_card = Card(
            "Per-camera status", "drops and stalls, while there is still time to act"
        )
        self.status_table = CameraStatusTable()
        status_card.add(self.status_table, 1)
        status_card.add(
            hint(
                "Dropped frames mean the writer could not keep up; that file is "
                "short. Zero FPS while recording means the capture thread stopped."
            )
        )
        split.addWidget(status_card)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)

    # ------------------------------------------------------------------
    # cameras
    # ------------------------------------------------------------------

    def camera_ids(self) -> list[str]:
        return sorted(getattr(self._manager, "cameras", {}) or {})

    def refresh_cameras(self) -> None:
        """Rebuild the preview tiles and status rows from the manager."""
        ids = self.camera_ids()
        primary = getattr(self._manager, "primary_camera_id", None)

        for camera_id in list(self.preview._tiles):
            self.preview.remove_camera(camera_id)
        for camera_id in ids:
            self.preview.add_camera(camera_id, is_primary=(camera_id == primary))

        self.status_table.set_cameras(ids)
        self._refresh_controls()

    def on_frame(self, camera_id: str, frame: np.ndarray, timestamp: float = 0.0) -> None:
        """Feed a frame to its tile. Unknown cameras are ignored.

        Frames arrive on capture threads; the caller is responsible for
        marshalling to the Qt thread, exactly as CameraPanel does.
        """
        if camera_id in self.preview._tiles:
            self.preview.update_frame(camera_id, frame)

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return bool(self._recorder is not None and getattr(self._recorder, "is_recording", False))

    def start_recording(self) -> None:
        if self._recorder is None or not self.camera_ids():
            return
        try:
            self._recorder.start_recording()
        except Exception:
            logger.exception("MultiCameraWindow: could not start recording")
        self._refresh_controls()

    def stop_recording(self) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.stop_recording()
        except Exception:
            logger.exception("MultiCameraWindow: could not stop recording")
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        recording = self.is_recording
        self.record_button.setEnabled(bool(self.camera_ids()) and not recording)
        self.stop_button.setEnabled(recording)

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    def _poll_status(self) -> None:
        """Refresh the status table.

        Never raises: this runs on a timer, and a camera unplugged mid-session
        or a recorder torn down between ticks must not take the window with it.
        """
        try:
            recorder = self._recorder
            recording = self.is_recording
            dropped_all = dict(getattr(recorder, "frames_dropped", {}) or {}) if recorder else {}
            for camera_id in self.camera_ids():
                fps = 0.0
                try:
                    fps = float(self._manager.get_camera_fps(camera_id) or 0.0)
                except Exception:
                    pass
                self.status_table.update_status(
                    camera_id,
                    fps=fps,
                    queue_depth=self._queue_depth(camera_id),
                    dropped=int(dropped_all.get(camera_id, 0)),
                    recording=recording,
                )
            flagged = self.status_table.flagged_cameras()
            self.warning_label.setText(f"⚠ losing frames: {', '.join(flagged)}" if flagged else "")
            self._refresh_controls()
        except Exception:
            logger.exception("MultiCameraWindow: status poll failed")

    def _queue_depth(self, camera_id: str) -> int:
        threads = getattr(self._recorder, "_writer_threads", None) or {}
        thread = threads.get(camera_id)
        try:
            return int(thread.queue_depth) if thread is not None else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 - Qt override
        """Stop polling, but leave recording alone.

        Closing this window must not end a run: it is a monitor, and the
        operator may well close it to free the screen while the session
        continues.
        """
        self._timer.stop()
        super().closeEvent(event)


__all__ = ["MultiCameraWindow"]
