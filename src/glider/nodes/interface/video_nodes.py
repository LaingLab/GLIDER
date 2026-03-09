"""
Video Nodes - Video playback for experiments.

Provides a VideoPlayback node that plays MP4/AVI/MOV files in a fullscreen
window on a user-selected monitor.  The window is created the first time
the node executes.  Execution blocks until the video finishes so that
sequential nodes play back-to-back correctly.

Video-only (no audio track) — cv2.VideoCapture does not handle audio.

Requires: opencv-python (cv2), already a project dependency.
"""

import asyncio
import logging

import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from glider.nodes.base_node import (
    GliderNode,
    NodeCategory,
    NodeDefinition,
    PortDefinition,
    PortType,
)

logger = logging.getLogger(__name__)


class VideoPlayerWindow(QWidget):
    """Fullscreen black window that plays video frames on demand.

    The window shows immediately as a black screen.  Call ``play()``
    to begin frame-by-frame playback of the loaded video file.
    Emits ``finished`` when the last frame has been displayed.
    """

    finished = pyqtSignal()

    def __init__(self, file_path: str, monitor_index: int = -1, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Playback")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: black;")

        self._file_path = file_path
        self._cap: cv2.VideoCapture | None = None
        self._fps = 30.0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._next_frame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet("background-color: black;")
        layout.addWidget(self._label)

        # Position on the selected monitor and go fullscreen
        self._go_fullscreen(monitor_index)

    def _go_fullscreen(self, monitor_index: int) -> None:
        """Move to the chosen monitor and fill it."""
        screens = QApplication.screens()
        if 0 <= monitor_index < len(screens):
            screen = screens[monitor_index]
        elif screens:
            screen = screens[0]
        else:
            return
        geo = screen.geometry()
        self.setGeometry(geo)
        self._display_w = geo.width()
        self._display_h = geo.height()
        self._label.setFixedSize(self._display_w, self._display_h)

    def play(self) -> None:
        """Open the video file and start frame playback.

        Safe to call multiple times — stops any in-progress playback first.
        Emits ``finished`` when the video ends.
        """
        if not self._file_path:
            self.finished.emit()
            return
        # Stop previous playback if still active
        if self._timer.isActive():
            self._timer.stop()
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = cv2.VideoCapture(self._file_path)
        if not self._cap.isOpened():
            logger.error(f"VideoPlayerWindow: cannot open '{self._file_path}'")
            self.finished.emit()
            return
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._timer.start(int(1000 / self._fps))

    def _next_frame(self) -> None:
        if self._cap is None:
            return
        ret, frame = self._cap.read()
        if not ret:
            self._timer.stop()
            self._cap.release()
            self._cap = None
            # Clear to black after video ends
            self._label.clear()
            self.finished.emit()
            return
        frame = cv2.resize(frame, (self._display_w, self._display_h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._label.setPixmap(QPixmap.fromImage(qimg))

    def stop(self) -> None:
        """Stop playback, release resources, and close the window."""
        if self._timer.isActive():
            self._timer.stop()
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        self.close()

    def closeEvent(self, event):  # noqa: N802
        self.stop()
        super().closeEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        """Allow Escape to close the window."""
        if event.key() == Qt.Key.Key_Escape:
            self.stop()
        else:
            super().keyPressEvent(event)


class VideoPlaybackNode(GliderNode):
    """Play a video file fullscreen on a selected monitor.

    The window is created the first time ``execute`` runs, plays the
    video, and waits for it to finish before firing ``next``.
    The window stays open (black) between plays and closes on ``stop``.
    """

    definition = NodeDefinition(
        name="VideoPlayback",
        category=NodeCategory.INTERFACE,
        description="Play a video file (MP4/AVI/MOV)",
        inputs=[
            PortDefinition("exec", PortType.EXEC, description="Execution input"),
        ],
        outputs=[
            PortDefinition("next", PortType.EXEC, description="Triggers after playback finishes"),
        ],
        color="#2d4a5a",
    )

    def __init__(self):
        super().__init__()
        self._state.setdefault("file_path", "")
        self._state.setdefault("monitor_index", -1)
        self._player: VideoPlayerWindow | None = None

    def update_event(self) -> None:
        """Called when inputs change."""
        pass

    def _ensure_player(self) -> None:
        """Create the fullscreen window if it doesn't exist yet."""
        if self._player is not None:
            return
        monitor_index = self._state.get("monitor_index", -1)
        self._player = VideoPlayerWindow(
            file_path=self._state.get("file_path", ""),
            monitor_index=monitor_index,
        )
        self._player.show()
        logger.info(f"VideoPlayback: window opened on monitor {monitor_index}")

    async def execute(self) -> None:
        """Play the video and fire next after it finishes."""
        file_path = self._state.get("file_path", "")
        if not file_path:
            logger.warning("VideoPlayback: no file path set")
            await self._fire_exec_output("next")
            return

        try:
            self._ensure_player()

            # Create a Future that will be resolved when the video finishes.
            loop = asyncio.get_event_loop()
            finished_future: asyncio.Future = loop.create_future()

            def _on_finished():
                if not finished_future.done():
                    loop.call_soon_threadsafe(finished_future.set_result, True)

            self._player.finished.connect(_on_finished)
            self._player.play()
            logger.info(f"VideoPlayback: playing '{file_path}'")

            # Wait for the video to finish before proceeding.
            await finished_future

            # Disconnect the one-shot slot to prevent duplicate triggers on replay.
            try:
                self._player.finished.disconnect(_on_finished)
            except RuntimeError:
                pass  # Already disconnected (e.g., window closed)
        except Exception as e:
            logger.error(f"VideoPlayback: playback error - {e}")
            self.set_error(str(e))

        await self._fire_exec_output("next")

    async def stop(self) -> None:
        """Close the video popup."""
        if self._player is not None:
            self._player.stop()
            self._player = None

    def exec_output(self, index: int = 0) -> None:
        """Trigger execution output."""
        for callback in self._update_callbacks:
            callback("next", True)


def register_video_nodes(flow_engine) -> None:
    """Register video nodes with the flow engine."""
    flow_engine.register_node("VideoPlayback", VideoPlaybackNode)
    logger.info("Registered video nodes")
