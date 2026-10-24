"""Multi-Camera Recording - a tool window for running a camera array.

The camera panel already has a Multi-Camera checkbox, and it stays: for two to
four cameras, previewing inline beside the node graph is the right thing. This
window is for the other case. A sixteen-camera grid needs a whole window, often
on a second monitor, and it needs a status table beside it - which is the part
the panel has nowhere to put.

Follows the same shape as the other tool windows (see
:mod:`glider.gui.pose_batch.window`): a QMainWindow held on the MainWindow and
re-surfaced rather than rebuilt, so state survives closing it.

Not every camera has to take part. Each one can be switched off in the status
table - it is then disconnected, so it costs no USB bandwidth - and each one
can be recorded on its own, alongside or instead of the rest.
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
from glider.gui.multi_camera.camera_selection import CameraSelection
from glider.gui.multi_camera.status_table import CameraStatusTable, format_elapsed
from glider.gui.styles import colors
from glider.gui.widgets.animal_picker import AnimalPicker
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

#: This window's streaming claim on the shared :class:`~glider.vision.multi_camera_manager.MultiCameraManager`.
#: Named rather than passed around so the acquire in ``connect_cameras``
#: and the release in ``closeEvent`` cannot drift apart -- a mismatch
#: would leak the claim and keep the cameras on for the rest of the session.
_STREAM_OWNER = "multi_camera_window"

#: Prefix for a run started from this window rather than from an experiment.
#: Files are "<name>_<timestamp>[_camN].mp4", so the timestamp already makes
#: them unique -- the name is what makes them findable a month later.
DEFAULT_EXPERIMENT_NAME = "multicam"


def camera_sort_key(camera_id: str) -> tuple[str, int]:
    """cam_2 before cam_10.

    Plain string sorting put a sixteen-camera rig in the order 0, 1, 10, 11 ...
    15, 2, 3, which is not the order anyone numbers their arenas in.
    """
    prefix, _, number = camera_id.rpartition("_")
    return (prefix, int(number)) if number.isdigit() else (camera_id, -1)


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
        selection: CameraSelection | None = None,
        session=None,
        parent=None,
    ):
        """``session``: a callable returning the experiment session, whose
        Animals list the run can be assigned from (#215). None hides the picker."""
        super().__init__(parent)
        self._session = session
        self._manager = multi_camera_manager
        self._recorder = recorder
        # What the operator configured in the Settings dialog. Registering
        # cameras on bare defaults would silently record 640x480 at 30 fps on a
        # rig set up for something else, and a recording cannot be redone.
        self._base_settings = base_settings or CameraSettings()
        self._default_experiment_name = experiment_name or DEFAULT_EXPERIMENT_NAME
        self._labels = labels if labels is not None else CameraLabels()
        self._selection = selection if selection is not None else CameraSelection()
        self._subscribed: set[str] = set()
        self._selected_id: str | None = None
        #: Every camera this window knows exists, connected or switched off,
        #: and the device index it was found at -- which a switched-off camera
        #: needs to be connected again, since the manager forgets it entirely.
        self._known: dict[str, int | None] = {}
        self._discovered = False

        self.setWindowTitle("Multi-Camera Recording")
        self.resize(1400, 900)
        self._build_ui()
        apply_tool_theme(self)

        self._frame_received.connect(self._show_frame)
        self.preview.primary_changed.connect(self.select_camera)
        self.preview.rename_requested.connect(self.rename_camera)
        # Queued: switching a camera rebuilds the table, which would delete
        # the very checkbox item whose change is still being delivered.
        self.status_table.enabled_toggled.connect(
            self.set_camera_enabled, Qt.ConnectionType.QueuedConnection
        )
        self.status_table.record_toggled.connect(self.toggle_camera_recording)

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

        # The animal from Experiment > Animals this run is of. Its ID joins the
        # file prefix, since these recordings are video only (#215).
        self.animal_picker = None
        if self._session is not None:
            controls.addWidget(QLabel("Animal"))
            self.animal_picker = AnimalPicker(self._session)
            controls.addWidget(self.animal_picker)

        self.record_button = QPushButton("Record All")
        self.record_button.setToolTip(
            "Start recording every camera that is switched on and not already recording"
        )
        self.record_button.clicked.connect(self.start_recording)
        set_button_role(self.record_button, "primary")
        controls.addWidget(self.record_button)

        self.stop_button = QPushButton("Stop All")
        self.stop_button.setToolTip("Stop every camera that is recording")
        self.stop_button.clicked.connect(self.stop_recording)
        controls.addWidget(self.stop_button)

        # Elapsed time. A recording with no clock on it is one nobody can
        # stop at the right moment, and every protocol here is timed.
        self.elapsed_label = QLabel("--:--")
        self.elapsed_label.setToolTip("Time recorded so far")
        self.elapsed_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        controls.addWidget(self.elapsed_label)

        refresh = QPushButton("Refresh Cameras")
        refresh.setToolTip("Look for cameras plugged in since the window opened")
        refresh.clicked.connect(self.rediscover_cameras)
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
                "Untick a camera to switch it off. Record on a row records that "
                "camera alone; the others carry on as they are."
            )
        )
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
        """The cameras switched on: connected, previewed, and recordable."""
        return sorted(getattr(self._manager, "cameras", {}) or {}, key=camera_sort_key)

    def all_camera_ids(self) -> list[str]:
        """Every camera this window knows of, switched on or not."""
        return sorted(set(self._known) | set(self.camera_ids()), key=camera_sort_key)

    def refresh_cameras(self) -> None:
        """Rebuild the preview tiles and status rows from the manager."""
        ids = self.camera_ids()
        everything = self.all_camera_ids()
        primary = getattr(self._manager, "primary_camera_id", None)

        for camera_id in list(self.preview._tiles):
            self.preview.remove_camera(camera_id)
        for camera_id in ids:
            self.preview.add_camera(camera_id, is_primary=(camera_id == primary))

        # A row is ticked when its camera is connected. Not simply when the
        # stored selection says "on": a camera that would not connect is not
        # on, and a ticked row with no picture would say it was.
        self.status_table.set_cameras(everything)
        connected = set(ids)
        for camera_id in everything:
            self.status_table.set_enabled(camera_id, camera_id in connected)
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
        """Find, register and start every camera that is switched on, then show them.

        The window cannot assume someone else has done this. It shares the
        core's manager with CameraPanel, so the cameras may already be there -
        in which case adding them again would stack a second frame callback on
        each one - but nothing guarantees the panel has ever been opened.

        Looks for cameras only the first time, or when the manager has none:
        probing every index again on each reopen would make reopening slow,
        and Refresh Cameras is there for a camera plugged in mid-session.
        """
        try:
            if not self._discovered or not (getattr(self._manager, "cameras", {}) or {}):
                self._discover_cameras()
                self._connect_enabled_cameras()
            for camera_id in self.camera_ids():
                self._subscribe(camera_id)
            self._manager.start_all_streaming(owner=_STREAM_OWNER)
        except Exception:
            logger.exception("MultiCameraWindow: could not connect cameras")
        self.refresh_cameras()

    def rediscover_cameras(self) -> None:
        """Look again for cameras, and connect the new ones that are switched on."""
        self._discovered = False
        self.connect_cameras()

    def _discover_cameras(self) -> None:
        """Learn which cameras exist, without disturbing the ones streaming.

        Connected cameras are not probed: a camera streaming here can refuse a
        second open and read as absent. They are known already, from the
        manager.
        """
        connected = list(getattr(self._manager, "cameras", {}) or {})
        for camera_id in connected:
            self._known[camera_id] = self._index_of(camera_id)
        in_use = {self._known[c] for c in connected if self._known[c] is not None}
        for info in self._manager.enumerate_all_cameras(skip_indices=in_use) or []:
            index = getattr(info, "index", None)
            if index is None:
                continue
            self._known[self._manager.camera_id_from_index(index)] = index
        self._discovered = True

    def _index_of(self, camera_id: str) -> int | None:
        """The device index a connected camera was opened at."""
        try:
            index = getattr(self._manager.get_camera_settings(camera_id), "camera_index", None)
        except Exception:
            index = None
        if isinstance(index, int):
            return index
        _, _, number = camera_id.rpartition("_")
        return int(number) if number.isdigit() else self._known.get(camera_id)

    def _connect_enabled_cameras(self) -> None:
        """Register every known camera that is switched on and not yet connected."""
        connected = set(self.camera_ids())
        for camera_id in self.all_camera_ids():
            if camera_id not in connected and self._selection.is_enabled(camera_id):
                self._add_camera(camera_id)

    def _add_camera(self, camera_id: str) -> bool:
        """Register one camera at the configured settings.

        Each gets its own copy via ``replace``: the base settings object is
        shared with the camera panel, so stamping an index onto it in place
        would retarget that panel's camera too.
        """
        from dataclasses import replace

        index = self._known.get(camera_id)
        if index is None:
            return False
        try:
            return bool(
                self._manager.add_camera(
                    camera_id, replace(self._base_settings, camera_index=index)
                )
            )
        except Exception:
            logger.exception("MultiCameraWindow: could not add %s", camera_id)
            return False

    def set_camera_enabled(self, camera_id: str, enabled: bool) -> None:
        """Switch one camera on (connect and stream it) or off (disconnect it).

        Off means disconnected, not merely hidden: the point of switching a
        camera off is usually the USB bandwidth it was taking from the rest.
        A camera with a file open cannot be switched off -- disconnecting it
        would cut that file off unfinished.
        """
        if not enabled and camera_id in self._recording_ids() | self._saving_ids():
            self.status_table.set_enabled(camera_id, True)
            self.statusBar().showMessage(
                f"Stop recording {self._labels.display_name(camera_id)} before switching it off",
                5000,
            )
            return

        self._selection.set_enabled(camera_id, enabled)
        if enabled:
            if camera_id not in self.camera_ids():
                if self._add_camera(camera_id):
                    self._subscribe(camera_id)
                    try:
                        self._manager.start_streaming(camera_id)
                    except Exception:
                        logger.exception("MultiCameraWindow: could not start %s", camera_id)
                else:
                    self.statusBar().showMessage(
                        f"{self._labels.display_name(camera_id)} would not connect", 5000
                    )
        else:
            # The manager drops its callbacks with the camera, so connecting it
            # again later has to subscribe again.
            self._subscribed.discard(camera_id)
            try:
                self._manager.remove_camera(camera_id)
            except Exception:
                logger.exception("MultiCameraWindow: could not remove %s", camera_id)
        self.refresh_cameras()

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
        """Push the current names onto the tiles, the table and into the recorder."""
        for camera_id in self.all_camera_ids():
            name = self._labels.display_name(camera_id)
            self.preview.set_display_name(camera_id, name)
            self.status_table.set_display_name(camera_id, name)
        setter = getattr(self._recorder, "set_camera_labels", None)
        if callable(setter):
            try:
                setter({cid: self._labels.file_fragment(cid) for cid in self.all_camera_ids()})
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
        name = typed or self._default_experiment_name or DEFAULT_EXPERIMENT_NAME
        animal = self.animal_picker.active_subject() if self.animal_picker else None
        tag = animal and (animal.subject_id or animal.name)
        return f"{name}_{tag}" if tag else name

    def _recording_ids(self) -> set[str]:
        """The cameras with a file open right now."""
        try:
            return set(getattr(self._recorder, "recording_camera_ids", None) or ())
        except Exception:
            return set()

    def _saving_ids(self) -> set[str]:
        """Cameras stopped on their own whose files are still being finished."""
        try:
            return set(getattr(self._recorder, "saving_camera_ids", None) or ())
        except Exception:
            return set()

    def start_recording(self) -> None:
        """Record every camera that is switched on.

        With nothing recording that is one run of them all. With some already
        going it adds the rest to that run, rather than doing nothing -- or
        restarting the ones whose takes are already under way.
        """
        if self._recorder is None or not self.camera_ids():
            return
        if not self.is_recording:
            self._schedule(
                self._recorder.start(self._experiment_name),
                "could not start recording",
            )
            return
        recording = self._recording_ids()
        for camera_id in self.camera_ids():
            if camera_id not in recording:
                self._schedule(
                    self._recorder.start_camera(camera_id, self._experiment_name),
                    f"could not start {camera_id}",
                )

    def stop_recording(self) -> None:
        if self._recorder is None:
            return
        self._schedule(self._recorder.stop(), "could not stop recording")

    def toggle_camera_recording(self, camera_id: str) -> None:
        """Start or stop one camera, leaving the rest as they are."""
        if self._recorder is None or camera_id not in self.camera_ids():
            return
        if camera_id in self._recording_ids():
            self._schedule(self._recorder.stop_camera(camera_id), f"could not stop {camera_id}")
        elif camera_id not in self._saving_ids():
            self._schedule(
                self._recorder.start_camera(camera_id, self._experiment_name),
                f"could not start {camera_id}",
            )

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

        Refreshes the whole table, not just the top buttons. Waiting for the
        next poll left a camera's row reading "Saving…", its button disabled,
        for up to half a second after its file was done, and a click on it in
        that time did nothing.
        """
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("MultiCameraWindow: %s", description)
        self._poll_status()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        """mm:ss, or h:mm:ss once a session runs past an hour."""
        return format_elapsed(seconds)

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
        ids = self.camera_ids()
        recording = self._recording_ids()
        # During a run, Record All stays live while any camera that is on is
        # not part of it yet.
        if self.is_recording:
            self.record_button.setEnabled(any(c not in recording for c in ids))
        else:
            self.record_button.setEnabled(bool(ids))
        self.stop_button.setEnabled(self.is_recording or bool(recording))

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
            recording = self._recording_ids()
            saving = self._saving_ids()
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
                    recording=camera_id in recording,
                    elapsed=self._camera_elapsed(camera_id),
                    saving=camera_id in saving,
                )
                self.preview.set_camera_recording(camera_id, camera_id in recording)
                self.preview.update_fps(camera_id, fps)
            flagged = self.status_table.flagged_cameras()
            self.warning_label.setText(f"⚠ losing frames: {', '.join(flagged)}" if flagged else "")
            self._refresh_controls()
        except Exception:
            logger.exception("MultiCameraWindow: status poll failed")

    def _camera_elapsed(self, camera_id: str) -> float | None:
        """How long this camera's current or last take has run."""
        getter = getattr(self._recorder, "camera_duration", None)
        if not callable(getter):
            return None
        try:
            return float(getter(camera_id))
        except Exception:
            return None

    def _queue_depth(self, camera_id: str) -> int:
        threads = getattr(self._recorder, "_writer_threads", None) or {}
        thread = threads.get(camera_id)
        try:
            return int(thread.queue_depth) if thread is not None else 0
        except Exception:
            return 0

    # ------------------------------------------------------------------

    def showEvent(self, event):  # noqa: N802 - Qt override
        """Take the cameras and restart polling, every time it is opened.

        Symmetrical with :meth:`closeEvent`, and it has to be: the window is
        kept on the main window and reused, so a close that tore things down
        with nothing to build them back left the reopened window showing dead
        tiles and a status table frozen at whatever it last read. Setting up
        here rather than only in ``__init__`` is what makes reopening work.

        Both calls are idempotent -- the timer restarts harmlessly and
        ``connect_cameras`` re-registers nothing it has already registered --
        so a ``show()`` on an already-open window costs nothing.
        """
        super().showEvent(event)
        self._timer.start(_POLL_MS)
        self.connect_cameras()
        if self.animal_picker is not None:
            self.animal_picker.refresh()

    def closeEvent(self, event):  # noqa: N802 - Qt override
        """Stop polling and let the cameras go, unless a recording needs them.

        Closing this window used to leave every camera streaming: it stopped
        its own poll timer and nothing else, so the capture threads ran and the
        camera lights stayed on until the app quit.

        It releases rather than stops, because this manager is shared with the
        camera panel's multi-camera preview -- see
        :meth:`~glider.vision.multi_camera_manager.MultiCameraManager.stop_all_streaming`.
        And it does not release at all while the recorder is running: closing a
        monitor must never end a run, which is the reason this window did
        nothing on close in the first place.
        """
        self._timer.stop()
        self._release_cameras()
        super().closeEvent(event)

    def _release_cameras(self) -> None:
        """Drop this window's streaming claim, if a run does not still need it."""
        if self.is_recording or self._recording_ids():
            logger.debug("MultiCameraWindow closed mid-recording; leaving cameras streaming")
            return
        try:
            self._manager.stop_all_streaming(owner=_STREAM_OWNER)
        except Exception:  # pragma: no cover - a close must not raise
            logger.exception("MultiCameraWindow: could not release cameras")


__all__ = ["MultiCameraWindow"]
