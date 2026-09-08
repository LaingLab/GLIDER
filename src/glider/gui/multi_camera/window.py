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

import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from glider.gui.multi_camera.camera_names import CameraLabels
from glider.gui.multi_camera.camera_properties import CameraPropertiesPanel
from glider.gui.multi_camera.status_table import CameraStatusTable
from glider.gui.styles import colors
from glider.gui.widgets.multi_camera_preview import MultiCameraPreviewWidget
from glider.gui.widgets.tool_ui import Card, apply_tool_theme, hint, set_button_role
from glider.vision.camera_manager import CameraSettings

if TYPE_CHECKING:
    from glider.vision.multi_camera_manager import MultiCameraManager
    from glider.vision.multi_video_recorder import MultiVideoRecorder

logger = logging.getLogger(__name__)

#: How often the status table refreshes. Fast enough that a stalled camera is
#: obvious within a second or two, slow enough that sixteen rows of Qt item
#: updates never compete with the capture threads for the GIL.
_POLL_MS = 500

#: Prefix for a run started from this window rather than from an experiment.
#: Files are "<name>_<timestamp>[_camN].mp4", so the timestamp already makes
#: them unique -- the name is what makes them findable a month later.
DEFAULT_EXPERIMENT_NAME = "multicam"


class MultiCameraWindow(QMainWindow):
    """Preview, record and monitor every camera in the rig at once."""

    #: Frames arrive on each camera's capture thread. Qt widgets may only be
    #: touched from the GUI thread, so they cross on a signal rather than being
    #: painted where they land - the same hop CameraPanel makes.
    _frame_received = pyqtSignal(str, object)

    def __init__(
        self,
        multi_camera_manager: MultiCameraManager,
        recorder: MultiVideoRecorder | None = None,
        base_settings: CameraSettings | None = None,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        labels: CameraLabels | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._manager = multi_camera_manager
        self._recorder = recorder
        # What the operator configured in the Settings dialog. Registering
        # cameras on bare defaults would silently record 640x480 at 30 fps on a
        # rig set up for something else, and a recording cannot be redone.
        self._base_settings = base_settings or CameraSettings()
        self._default_experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self._labels = labels if labels is not None else CameraLabels()
        self._subscribed: set[str] = set()
        self._selected_id: str | None = None

        self.setWindowTitle("Multi-Camera Recording")
        self.resize(1400, 900)
        self._build_ui()
        apply_tool_theme(self)

        self._frame_received.connect(self._show_frame)
        self.preview.primary_changed.connect(self.select_camera)
        self.preview.rename_requested.connect(self.rename_camera)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_status)
        self._timer.start(_POLL_MS)

        self.connect_cameras()

    # ------------------------------------------------------------------
    # ui
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        # Eight files land per run. Without a name they are distinguishable
        # only by timestamp, which is not what anyone remembers a session by.
        controls.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit(self._default_experiment_name)
        self.name_edit.setPlaceholderText(DEFAULT_EXPERIMENT_NAME)
        self.name_edit.setToolTip("Prefix for every file this run writes")
        self.name_edit.setMaximumWidth(220)
        controls.addWidget(self.name_edit)

        self.record_button = QPushButton("Record All")
        self.record_button.setToolTip("Start recording on every connected camera")
        self.record_button.clicked.connect(self.start_recording)
        set_button_role(self.record_button, "primary")
        controls.addWidget(self.record_button)

        self.stop_button = QPushButton("Stop All")
        self.stop_button.clicked.connect(self.stop_recording)
        controls.addWidget(self.stop_button)

        # Elapsed time. A recording with no clock on it is one nobody can
        # stop at the right moment, and every protocol here is timed.
        self.elapsed_label = QLabel("--:--")
        self.elapsed_label.setToolTip("Time recorded so far")
        self.elapsed_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        controls.addWidget(self.elapsed_label)

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

        properties_card = Card("Camera", "exposure, and what it is costing right now")
        self.properties = CameraPropertiesPanel()
        self.properties.settings_changed.connect(self._apply_camera_settings)
        self.properties.rename_requested.connect(self.rename_camera)
        properties_card.add(self.properties, 1)
        split.addWidget(properties_card)

        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 2)
        split.setStretchFactor(2, 2)
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
        self._refresh_names()
        # Something has to be selected for the properties panel to be useful,
        # and the primary is the camera the operator already thinks in terms of.
        if self._selected_id not in ids:
            self._selected_id = None
            self.properties.set_camera(None, None, "")
        if self._selected_id is None and ids:
            self.select_camera(primary if primary in ids else ids[0])
        self._refresh_controls()

    def connect_cameras(self) -> None:
        """Enumerate, register and start every camera, then show them.

        The window cannot assume someone else has done this. It shares the
        core's manager with CameraPanel, so the cameras may already be there -
        in which case adding them again would stack a second frame callback on
        each one - but nothing guarantees the panel has ever been opened.
        """
        try:
            existing = set(getattr(self._manager, "cameras", {}) or {})
            if not existing:
                self._add_enumerated_cameras()
            for camera_id in self.camera_ids():
                self._subscribe(camera_id)
            self._manager.start_all_streaming()
        except Exception:
            logger.exception("MultiCameraWindow: could not connect cameras")
        self.refresh_cameras()

    def _add_enumerated_cameras(self) -> None:
        """Register every camera the manager can see, at the configured settings.

        Each gets its own copy via ``replace``: the base settings object is
        shared with the camera panel, so stamping an index onto it in place
        would retarget that panel's camera too.
        """
        from dataclasses import replace

        for info in self._manager.enumerate_all_cameras() or []:
            index = getattr(info, "index", None)
            if index is None:
                continue
            camera_id = self._manager.camera_id_from_index(index)
            self._manager.add_camera(camera_id, replace(self._base_settings, camera_index=index))

    def _subscribe(self, camera_id: str) -> None:
        """Register the frame callback once per camera."""
        if camera_id in self._subscribed:
            return
        self._manager.on_frame(camera_id, self._on_camera_frame)
        self._subscribed.add(camera_id)

    def _on_camera_frame(self, camera_id: str, frame: np.ndarray, timestamp: float) -> None:
        """Capture-thread entry point. Copies and hands off to the GUI thread.

        The copy matters: the capture loop reuses its buffer, so painting the
        original from another thread races the next grab.
        """
        try:
            self._frame_received.emit(camera_id, frame.copy())
        except Exception:
            logger.exception("MultiCameraWindow: dropping a frame from %s", camera_id)

    def _show_frame(self, camera_id: str, frame: np.ndarray) -> None:
        """GUI-thread half of the hop. Unknown cameras are ignored."""
        if camera_id not in self.preview._tiles:
            return
        self.preview.update_frame(camera_id, frame)
        # Only the selected camera is measured. Reading all sixteen every frame
        # would compete with the capture threads for the GIL to answer a
        # question about a camera nobody is looking at.
        if camera_id == self._selected_id:
            self.properties.update_readout(frame)

    def on_frame(self, camera_id: str, frame: np.ndarray, timestamp: float = 0.0) -> None:
        """Push a frame in from outside, already on the GUI thread."""
        self._show_frame(camera_id, frame)

    # ------------------------------------------------------------------
    # selection, focus and naming
    # ------------------------------------------------------------------

    def select_camera(self, camera_id: str) -> None:
        """Point the properties panel at one camera.

        Clicking a tile selects it; clicking the selected one enlarges it. The
        two are deliberately separate: selecting to read a camera's exposure
        should not rearrange the whole window under the operator.
        """
        if camera_id not in self.preview._tiles:
            return
        if camera_id == self._selected_id:
            self.preview.toggle_focus(camera_id)
        self._selected_id = camera_id
        self.properties.set_camera(
            camera_id,
            self._manager.get_camera_settings(camera_id),
            self._labels.display_name(camera_id),
        )

    def rename_camera(self, camera_id: str) -> None:
        """Ask for a name for this camera and remember it.

        The name reaches the filenames, which is the point -- naming a camera
        in the window and still getting ``_cam5`` on disk would leave the same
        question unanswered a month later.
        """
        current = self._labels.label(camera_id)
        name, accepted = QInputDialog.getText(
            self,
            "Name this camera",
            f"Name for {camera_id} (used on the tile and in its filenames):",
            text=current,
        )
        if not accepted:
            return
        self._labels.set_label(camera_id, name)
        self._refresh_names()
        if camera_id == self._selected_id:
            self.properties.set_camera(
                camera_id,
                self._manager.get_camera_settings(camera_id),
                self._labels.display_name(camera_id),
            )

    def _refresh_names(self) -> None:
        """Push the current names onto the tiles and into the recorder."""
        for camera_id in self.camera_ids():
            self.preview.set_display_name(camera_id, self._labels.display_name(camera_id))
        setter = getattr(self._recorder, "set_camera_labels", None)
        if callable(setter):
            try:
                setter({cid: self._labels.file_fragment(cid) for cid in self.camera_ids()})
            except Exception:
                logger.exception("MultiCameraWindow: could not hand labels to the recorder")

    def _apply_camera_settings(self, camera_id: str, settings) -> None:
        """Push edited settings to one camera, live.

        Never raises: a camera that rejects a property -- and DirectShow
        devices reject plenty -- must not take the window down mid-session.
        """
        camera = self._manager.get_camera(camera_id)
        if camera is None:
            return
        try:
            camera.apply_settings(settings)
        except Exception:
            logger.exception("MultiCameraWindow: could not apply settings to %s", camera_id)

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return bool(self._recorder is not None and getattr(self._recorder, "is_recording", False))

    @property
    def _experiment_name(self) -> str:
        """What this run's files are named after.

        Falls back to the default when the field is blank rather than
        writing files that begin with an underscore.
        """
        typed = self.name_edit.text().strip() if hasattr(self, "name_edit") else ""
        return typed or self._default_experiment_name or DEFAULT_EXPERIMENT_NAME

    def start_recording(self) -> None:
        if self._recorder is None or not self.camera_ids():
            return
        self._schedule(
            self._recorder.start(self._experiment_name),
            "could not start recording",
        )

    def stop_recording(self) -> None:
        if self._recorder is None:
            return
        self._schedule(self._recorder.stop(), "could not stop recording")

    def _schedule(self, coro, description: str) -> None:
        """Run one of the recorder's coroutines on the Qt event loop.

        ``MultiVideoRecorder.start`` / ``stop`` are async -- they open and
        drain a writer thread per camera, which must not block the GUI thread
        while eight files are finalised. qasync runs the asyncio loop on the Qt
        main thread, so the completion callback is already where it needs to be
        to touch widgets.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop: headless, or a unit test driving the window
            # directly. Close the coroutine so it does not warn about never
            # being awaited, and leave the controls describing reality.
            coro.close()
            logger.debug("MultiCameraWindow: %s - no running event loop", description)
            self._refresh_controls()
            return
        task = loop.create_task(coro)
        task.add_done_callback(lambda finished: self._recorder_task_done(finished, description))

    def _recorder_task_done(self, task, description: str) -> None:
        """Report what the scheduled recorder call did, and resync the buttons.

        A bare ``create_task`` swallows the exception, which would leave Record
        greyed out and Stop enabled with nothing recording -- the window saying
        a run is in flight that never started.
        """
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("MultiCameraWindow: %s", description)
        self._refresh_controls()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """mm:ss, or h:mm:ss once a session runs past an hour."""
        seconds = max(0, int(seconds))
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"

    def _refresh_elapsed(self) -> None:
        """Show how long this run has been going.

        Holds the final time after Stop rather than resetting to zero: the last
        thing an operator wants to know when a run ends is how long it was.
        """
        recorder = self._recorder
        if recorder is None:
            return
        try:
            elapsed = float(getattr(recorder, "duration", 0.0) or 0.0)
        except Exception:
            return
        if self.is_recording:
            self.elapsed_label.setText(self._format_elapsed(elapsed))
            self.elapsed_label.setStyleSheet(
                f"font-weight: bold; font-size: 15px; color: {colors.ERROR};"
            )
        elif elapsed <= 0:
            self.elapsed_label.setText("--:--")
            self.elapsed_label.setStyleSheet("font-weight: bold; font-size: 15px;")

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
            self._refresh_elapsed()
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
