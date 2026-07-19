"""
GLIDER Core - The central orchestrator.

The main controller that initializes the event loop, loads plugins,
manages the ExperimentSession, and coordinates between hardware
and flow execution.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from glider.core.data_recorder import DataRecorder
from glider.core.event_logger import DeviceEventLogger
from glider.core.experiment_session import ExperimentSession, SessionState
from glider.core.flow_engine import FlowEngine
from glider.core.hardware_manager import HardwareManager
from glider.vision.audio_recorder import AudioRecorder, mux_audio_video
from glider.vision.calibration import CameraCalibration
from glider.vision.camera_manager import CameraManager
from glider.vision.cv_processor import CVProcessor, CVSettings
from glider.vision.multi_camera_manager import MultiCameraManager
from glider.vision.multi_video_recorder import MultiVideoRecorder
from glider.vision.tracking_logger import TrackingDataLogger
from glider.vision.video_recorder import VideoRecorder

if TYPE_CHECKING:
    from glider.plugins.plugin_manager import PluginManager

logger = logging.getLogger(__name__)


class GliderCore:
    """
    Central orchestrator for GLIDER.

    Responsibilities:
    - Initialize and manage the event loop
    - Load and manage plugins
    - Coordinate ExperimentSession, HardwareManager, and FlowEngine
    - Handle system signals and emergency shutdown
    """

    def __init__(self):
        """Initialize the GLIDER core."""
        self._session: ExperimentSession | None = None
        self._hardware_manager = HardwareManager()
        self._flow_engine = FlowEngine(self._hardware_manager)
        self._plugin_manager: PluginManager | None = None
        self._data_recorder = DataRecorder(self._hardware_manager)
        # Event logger captures per-edge / per-write device events, complementary
        # to the periodic state samples written by data_recorder. Together with
        # the frame-aligned data_recorder rows (camera-driven mode), this makes
        # video <-> tracking <-> device joins a single-key operation.
        self._event_logger = DeviceEventLogger(self._hardware_manager)
        # Fold runtime settings mutations (e.g. an HX711 tare offset) back
        # into the session as they happen, so they persist and the session
        # goes dirty — waiting until save_session would be too late for the
        # unsaved-changes prompt that decides whether a save happens at all.
        self._hardware_manager.register_device_settings_callback(self._on_device_settings_changed)

        # Vision components
        self._camera_manager = CameraManager()
        self._cv_processor = CVProcessor()
        self._video_recorder = VideoRecorder(self._camera_manager)
        self._tracking_logger = TrackingDataLogger()
        self._audio_recorder = AudioRecorder()
        self._calibration = CameraCalibration()

        # Multi-camera support
        self._multi_camera_manager = MultiCameraManager()
        self._multi_video_recorder = MultiVideoRecorder(self._multi_camera_manager)
        self._multi_camera_enabled = False

        self._initialized = False
        self._shutting_down = False
        self._experiment_lock = asyncio.Lock()
        # Strong references to fire-and-forget tasks (flow teardown, event
        # recording). asyncio only weakly references tasks; without retaining
        # them here, a task like _handle_flow_complete — which drives all
        # devices to a safe LOW state — could be garbage-collected before it
        # runs. Mirrors FlowEngine._running_tasks.
        self._background_tasks: set[asyncio.Task] = set()
        self._recording_enabled = True  # Auto-record experiments by default
        self._video_recording_enabled = True  # Auto-record video when camera connected
        self._annotated_video_enabled = True  # Also save annotated video with tracking overlays
        self._cv_processing_enabled = True  # Enable CV processing by default

        # Flow timing anchors (monotonic, immune to wall-clock drift / NTP).
        # ``_flow_start_monotonic`` is captured the moment the flow engine
        # is told to start (the last step of ``start_experiment``), and
        # ``_flow_end_monotonic`` is captured at the very FIRST line of
        # ``_handle_flow_complete`` / ``_stop_experiment_locked``, *before*
        # any teardown I/O. This decouples the reported flow duration from
        # the variable cost of stopping recorders / setting devices low /
        # transitioning state, which previously made a ``Delay(10s)`` flow
        # display as 10.11s-10.43s run-to-run.
        self._flow_start_monotonic: float | None = None
        self._flow_end_monotonic: float | None = None
        # Wall-clock siblings of the monotonic anchors. Captured at the
        # same instants but using time.time() so they can be cross-
        # referenced with output-file timestamps (which are all Unix
        # epoch). Used to:
        #   - emit ``flow_marker`` rows into the event log (analysts grep
        #     for the wall-clock boundaries of the flow), and
        #   - drive ``set_flow_anchor()`` on the tracking + data
        #     recorders so each frame/sample carries a flow-relative
        #     ``flow_elapsed_ms`` column (analysts plot ethograms /
        #     rasters with t=0 at flow start, no offset math).
        self._flow_start_wall: float | None = None
        self._flow_end_wall: float | None = None

        # Callbacks
        self._session_callbacks: list[Callable[[ExperimentSession], None]] = []
        self._state_callbacks: list[Callable[[SessionState], None]] = []
        self._error_callbacks: list[Callable[[str, Exception], None]] = []

        # Register hardware error handling
        self._hardware_manager.on_error(self._on_hardware_error)
        self._flow_engine.on_error(self._on_flow_error)
        self._flow_engine.on_flow_complete(self._on_flow_complete)
        self._flow_engine.on_node_update(self._on_node_update)

    @property
    def session(self) -> ExperimentSession | None:
        """Current experiment session."""
        return self._session

    @property
    def last_flow_duration_s(self) -> float | None:
        """Duration of the most recent flow, in seconds.

        Computed as ``_flow_end_monotonic - _flow_start_monotonic``.
        Returns ``None`` if no flow has completed yet, or while one is
        still running (only ``_flow_start_monotonic`` set).

        This is the operator's truth-of-record for "how long did the
        experiment take." It does NOT include pre-flow recorder setup
        or post-flow teardown — only the time between the engine being
        told to start and the engine signaling completion (or the user
        clicking STOP). Use this in the runner timer's final display
        and in output-file duration footers.
        """
        if self._flow_start_monotonic is None or self._flow_end_monotonic is None:
            return None
        return self._flow_end_monotonic - self._flow_start_monotonic

    @property
    def hardware_manager(self) -> HardwareManager:
        """Hardware manager instance."""
        return self._hardware_manager

    @property
    def flow_engine(self) -> FlowEngine:
        """Flow engine instance."""
        return self._flow_engine

    @property
    def data_recorder(self) -> DataRecorder:
        """Data recorder instance."""
        return self._data_recorder

    @property
    def camera_manager(self) -> CameraManager:
        """Camera manager instance."""
        return self._camera_manager

    @property
    def cv_processor(self) -> CVProcessor:
        """CV processor instance."""
        return self._cv_processor

    @property
    def video_recorder(self) -> VideoRecorder:
        """Video recorder instance."""
        return self._video_recorder

    @property
    def tracking_logger(self) -> TrackingDataLogger:
        """Tracking data logger instance."""
        return self._tracking_logger

    @property
    def event_logger(self) -> DeviceEventLogger:
        """Device event logger instance (per-edge / per-write event log)."""
        return self._event_logger

    @property
    def calibration(self) -> CameraCalibration:
        """Camera calibration instance."""
        return self._calibration

    @property
    def multi_camera_manager(self) -> MultiCameraManager:
        """Multi-camera manager instance."""
        return self._multi_camera_manager

    @property
    def multi_video_recorder(self) -> MultiVideoRecorder:
        """Multi-video recorder instance."""
        return self._multi_video_recorder

    @property
    def multi_camera_enabled(self) -> bool:
        """Whether multi-camera mode is enabled."""
        return self._multi_camera_enabled

    @multi_camera_enabled.setter
    def multi_camera_enabled(self, value: bool) -> None:
        """Enable or disable multi-camera mode."""
        self._multi_camera_enabled = value
        self._multi_camera_manager.enabled = value

    @property
    def video_recording_enabled(self) -> bool:
        """Whether automatic video recording is enabled."""
        return self._video_recording_enabled

    @video_recording_enabled.setter
    def video_recording_enabled(self, value: bool) -> None:
        """Enable or disable automatic video recording."""
        self._video_recording_enabled = value

    @property
    def annotated_video_enabled(self) -> bool:
        """Whether annotated video recording (with tracking overlays) is enabled."""
        return self._annotated_video_enabled

    @annotated_video_enabled.setter
    def annotated_video_enabled(self, value: bool) -> None:
        """Enable or disable annotated video recording."""
        self._annotated_video_enabled = value

    @property
    def cv_processing_enabled(self) -> bool:
        """Whether CV processing is enabled."""
        return self._cv_processing_enabled

    @cv_processing_enabled.setter
    def cv_processing_enabled(self, value: bool) -> None:
        """Enable or disable CV processing."""
        self._cv_processing_enabled = value

    @property
    def recording_enabled(self) -> bool:
        """Whether automatic recording is enabled."""
        return self._recording_enabled

    @recording_enabled.setter
    def recording_enabled(self, value: bool) -> None:
        """Enable or disable automatic recording."""
        self._recording_enabled = value

    def set_recording_directory(self, path: Path) -> None:
        """Set the directory for recording data files (CSV, video, tracking)."""
        self._data_recorder.set_output_directory(path)
        self._video_recorder.set_output_directory(path)
        self._multi_video_recorder.set_output_directory(path)
        self._tracking_logger.set_output_directory(path)
        self._event_logger.set_output_directory(path)
        self._audio_recorder.set_output_directory(path)

    def set_recording_interval(self, interval: float) -> None:
        """Set the sampling interval for recording (in seconds)."""
        self._data_recorder.sample_interval = interval

    @property
    def is_initialized(self) -> bool:
        """Whether the core is initialized."""
        return self._initialized

    @property
    def state(self) -> SessionState:
        """Current session state."""
        return self._session.state if self._session else SessionState.IDLE

    @property
    def is_experiment_busy(self) -> bool:
        """True while a start/stop is mid-flight (the experiment lock is held).

        Lets the GUI refuse a manual function run during the window where an
        experiment is starting but ``state`` has not yet flipped to RUNNING.
        """
        return self._experiment_lock.locked()

    def on_session_change(self, callback: Callable[[ExperimentSession], None]) -> None:
        """Register callback for session changes."""
        self._session_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[SessionState], None]) -> None:
        """Register callback for state changes."""
        self._state_callbacks.append(callback)

    def on_error(self, callback: Callable[[str, Exception], None]) -> None:
        """Register callback for errors."""
        self._error_callbacks.append(callback)

    def _notify_session_change(self) -> None:
        """Notify session change callbacks."""
        for callback in self._session_callbacks:
            try:
                callback(self._session)
            except Exception as e:
                logger.error(f"Session callback error: {e}")

    def _notify_state_change(self, state: SessionState) -> None:
        """Notify state change callbacks."""
        for callback in self._state_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"State callback error: {e}")

    def _notify_error(self, source: str, error: Exception) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(source, error)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

    def _on_hardware_error(self, source: str, error: Exception) -> None:
        """Handle hardware errors."""
        logger.error(f"Hardware error from {source}: {error}")
        self._notify_error(f"hardware:{source}", error)

    def _on_flow_error(self, source: str, error: Exception) -> None:
        """Handle flow errors."""
        logger.error(f"Flow error from {source}: {error}")
        self._notify_error(f"flow:{source}", error)

    def _on_node_update(self, node_id: str, output_name: str, value: Any) -> None:
        """Handle node output updates — record events for audio playback, etc."""
        if not self._data_recorder.is_recording:
            return

        node = self._flow_engine.get_node(node_id)
        if node is None:
            return

        # Record audio playback events
        if node.name == "AudioPlayback":
            if output_name == "playing" and value:
                import os

                filename = os.path.basename(str(value))
                self._create_background_task(
                    self._data_recorder.record_event("AudioPlayback", filename)
                )
            elif output_name == "error" and value:
                self._create_background_task(
                    self._data_recorder.record_event("AudioPlaybackError", str(value))
                )

    def _on_flow_complete(self) -> None:
        """Handle flow completion (EndExperiment reached).

        Capture ``_flow_end_monotonic`` HERE — synchronously, before the
        teardown task even gets scheduled — so the reported flow duration
        reflects the actual logical end of the flow (when EndExperiment
        fired) rather than when teardown happens to finish. This is what
        keeps ``Delay(10s)`` reporting exactly 10s instead of 10s plus
        the variable I/O cost of stopping recorders + setting devices
        low + atomic-renaming output files.

        Also captures the wall-clock end and writes the ``end`` flow
        marker to the event log NOW — before _stop_recorders() tears
        the event logger down. The marker's wall-clock matches the
        same instant the monotonic anchor sees, so the event-log
        boundary, the runner timer, and the recorder footer all agree.
        """
        if self._flow_start_monotonic is not None:
            self._flow_end_monotonic = time.monotonic()
            self._flow_end_wall = time.time()
            # Stamp the boundary into the event log before teardown
            # closes it. Safe to call when not recording — it no-ops.
            self._event_logger.record_flow_marker("end")
        logger.info("Flow completed - transitioning to READY state")
        # Schedule the async completion handler (retained in
        # _background_tasks so it cannot be GC'd before teardown runs)
        self._create_background_task(self._handle_flow_complete())

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        """Log exceptions from fire-and-forget tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Unhandled error in background task: {exc}", exc_info=exc)

    def _create_background_task(self, coro) -> asyncio.Task:
        """Create a fire-and-forget task, retaining a strong reference until done."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        task.add_done_callback(self._log_task_exception)
        return task

    async def _handle_flow_complete(self) -> None:
        """Async handler for flow completion."""
        # Stop the flow engine
        await self._flow_engine.stop()
        await self._stop_recorders()

        # Set all devices to safe state
        await self._set_all_devices_low()

        # Update session state
        if self._session:
            self._session.state = SessionState.READY

    async def initialize(self) -> None:
        """Initialize the GLIDER core."""
        if self._initialized:
            return

        logger.info("Initializing GLIDER Core...")

        # Register built-in nodes FIRST (before initializing flow engine)
        # This ensures _node_registry is populated before ryvencore registration
        self._register_builtin_nodes()

        # Initialize flow engine (registers nodes with ryvencore session)
        self._flow_engine.initialize()

        # Load plugins
        await self._load_plugins()

        # Load declarative custom devices from the device library (safe: data,
        # not code) and register them as device types.
        self._load_device_library()

        # Create new session
        self._session = ExperimentSession()
        self._session.on_state_change(self._notify_state_change)

        self._initialized = True
        logger.info("GLIDER Core initialized successfully")

    def _register_builtin_nodes(self) -> None:
        """Register built-in node types with the flow engine."""
        try:
            from glider.nodes.experiment_nodes import register_experiment_nodes

            register_experiment_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register experiment nodes: {e}")

        try:
            from glider.nodes.control_nodes import register_control_nodes

            register_control_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register control nodes: {e}")

        try:
            from glider.nodes.logic.flow_nodes import register_logic_nodes

            register_logic_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register logic nodes: {e}")

        try:
            from glider.nodes.flow_function_nodes import register_flow_function_nodes

            register_flow_function_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register flow function nodes: {e}")

        try:
            from glider.nodes.vision.zone_nodes import register_zone_nodes

            register_zone_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register zone nodes: {e}")

        try:
            from glider.nodes.interface.audio_nodes import register_audio_nodes

            register_audio_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register audio nodes: {e}")

        try:
            from glider.nodes.interface.video_nodes import register_video_nodes

            register_video_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register video nodes: {e}")

        try:
            from glider.nodes.interface import register_interface_nodes

            register_interface_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register interface (input + display) nodes: {e}")

        try:
            from glider.nodes.hardware import register_hardware_nodes

            register_hardware_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register hardware nodes: {e}")

        try:
            from glider.nodes.logic import (
                register_comparison_nodes,
                register_logic_control_nodes,
                register_math_nodes,
            )

            register_math_nodes(self._flow_engine)
            register_comparison_nodes(self._flow_engine)
            register_logic_control_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register math/comparison/logic-control nodes: {e}")

    async def _load_plugins(self) -> None:
        """Load plugins from the plugin directory."""
        try:
            from pathlib import Path

            from glider.core.config import get_config
            from glider.plugins.plugin_manager import PluginManager

            cfg = get_config().plugins
            extra_dirs = [Path(p) for p in cfg.plugin_dirs]
            self._plugin_manager = PluginManager(
                plugin_dirs=extra_dirs or None,
                enable_directory_plugins=cfg.enable_directory_plugins,
            )
            await self._plugin_manager.discover_plugins()
            await self._plugin_manager.load_plugins()
        except ImportError:
            logger.warning("Plugin manager not available")
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")

    def _load_device_library(self) -> None:
        """Register declarative custom devices from the device library."""
        try:
            from glider.core.config import get_config
            from glider.core.device_library import load_and_register_all

            load_and_register_all(get_config().paths.devices_dir)
        except Exception as e:
            logger.error(f"Error loading device library: {e}")

    def new_session(self) -> ExperimentSession:
        """Create a new experiment session."""
        if self._session and self._session.is_dirty:
            logger.warning("Discarding unsaved changes in current session")

        # Clear the flow engine to reset state
        self._flow_engine.clear()

        self._session = ExperimentSession()
        self._session.on_state_change(self._notify_state_change)
        self._notify_session_change()
        logger.info("Created new session")
        return self._session

    def load_session(self, file_path: str) -> ExperimentSession:
        """
        Load an experiment session from file.

        Args:
            file_path: Path to the session file

        Returns:
            Loaded session
        """
        logger.info(f"Loading session from {file_path}")

        # Clear the flow engine to reset state
        self._flow_engine.clear()

        self._session = ExperimentSession.load(file_path)
        self._session.on_state_change(self._notify_state_change)
        self._notify_session_change()
        return self._session

    async def load_experiment(self, file_path: Path) -> None:
        """
        Load an experiment from file using the serialization layer.

        Args:
            file_path: Path to the .glider experiment file
        """
        from glider.serialization import ExperimentSerializer

        serializer = ExperimentSerializer()
        schema = serializer.load(file_path)

        # Create new session if needed
        if self._session is None:
            self._session = ExperimentSession()
            self._session.on_state_change(self._notify_state_change)

        # Apply schema to session
        serializer.apply_to_session(
            schema,
            self._session,
            self._flow_engine,
            self._hardware_manager,
        )

        # Apply the vision block. The serializer carries it opaquely (it must
        # not import cv2), so the conversion back to CVSettings lives here.
        # Absent for files written before schema 1.1.0 — leave the processor's
        # current settings alone rather than stomping them with defaults.
        if schema.vision.settings:
            try:
                self._cv_processor.update_settings(CVSettings.from_dict(schema.vision.settings))
            except Exception:
                logger.exception(
                    "Could not apply vision settings from %s; keeping current CV configuration",
                    file_path,
                )

        self._notify_session_change()
        logger.info(f"Loaded experiment: {schema.metadata.name}")

    async def save_experiment(self, file_path: Path) -> None:
        """
        Save the current experiment to file.

        Args:
            file_path: Path to save the .glider file
        """
        if self._session is None:
            raise RuntimeError("No session to save")

        from glider.serialization import ExperimentSerializer

        serializer = ExperimentSerializer()
        serializer.save(
            file_path,
            self._session,
            self._flow_engine,
            self._hardware_manager,
            vision_settings=self._cv_processor.settings.to_dict(),
        )
        logger.info(f"Saved experiment to {file_path}")

    def _on_device_settings_changed(self, device_id: str, device) -> None:
        """Adopt a device's runtime settings mutation into the session."""
        if self._session is None:
            return
        if self._session.get_device(device_id) is None:
            return  # a device the session doesn't track; nothing to persist
        self._session.update_device(device_id, settings=dict(device.config.settings))

    def save_session(self, file_path: str | None = None) -> str:
        """
        Save the current session to file.

        Args:
            file_path: Path to save to (uses existing if None)

        Returns:
            Path to saved file
        """
        if self._session is None:
            raise RuntimeError("No session to save")

        # Runtime-mutated device settings (e.g. an HX711 tare offset) live on
        # the device instances; the session serializes its own DeviceConfig
        # copies. Sync any drift before saving so those mutations persist.
        for device_id, device in self._hardware_manager.devices.items():
            session_config = self._session.get_device(device_id)
            if session_config is not None and session_config.settings != device.config.settings:
                self._session.update_device(device_id, settings=dict(device.config.settings))

        return self._session.save(file_path)

    async def setup_hardware(self) -> bool:
        """
        Set up hardware from the current session.

        Creates board and device instances from session configuration.

        Returns:
            True if all hardware set up successfully
        """
        if self._session is None:
            raise RuntimeError("No session loaded")

        self._session.state = SessionState.INITIALIZING
        success = True

        try:
            # Create boards
            for board_config in self._session.hardware.boards:
                try:
                    await self._hardware_manager.create_board(board_config)
                except Exception as e:
                    logger.error(f"Failed to create board {board_config.id}: {e}")
                    success = False

            # Create devices
            for device_config in self._session.hardware.devices:
                try:
                    await self._hardware_manager.create_device(device_config)
                except Exception as e:
                    logger.error(f"Failed to create device {device_config.id}: {e}")
                    success = False

        except Exception as e:
            logger.error(f"Error setting up hardware: {e}")
            success = False

        return success

    async def connect_hardware(self) -> dict[str, bool]:
        """
        Connect to all configured hardware.

        Returns:
            Dictionary of board_id -> success
        """
        if self._session is None:
            raise RuntimeError("No session loaded")

        self._session.state = SessionState.INITIALIZING

        # Connect boards
        results = await self._hardware_manager.connect_all()

        # Initialize devices on connected boards
        if any(results.values()):
            device_results = await self._hardware_manager.initialize_all_devices()
            results.update({f"device:{k}": v for k, v in device_results.items()})

        # Update session state
        if all(results.values()):
            self._session.state = SessionState.READY
        else:
            self._session.state = SessionState.ERROR

        return results

    async def _ensure_devices_initialized(self) -> None:
        """Ensure all devices are initialized before starting experiment."""
        for device_id, device in self._hardware_manager.devices.items():
            if not getattr(device, "_initialized", False):
                logger.info(f"Initializing device: {device_id}")
                try:
                    await device.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize device {device_id}: {e}")

    def setup_flow(self) -> None:
        """Set up the flow graph from the current session."""
        if self._session is None:
            raise RuntimeError("No session loaded")

        self._flow_engine.load_from_session(self._session)

    async def start_experiment(self) -> None:
        """Start running the experiment."""
        async with self._experiment_lock:
            await self._start_experiment_locked()

    async def _start_experiment_locked(self) -> None:
        """Internal start implementation, called under _experiment_lock."""
        if self._session is None:
            raise RuntimeError("No session loaded")

        # Allow starting from IDLE state too (will connect hardware first)
        if self._session.state == SessionState.IDLE:
            logger.info("Connecting hardware before starting experiment...")
            await self.connect_hardware()

        # Always ensure devices are initialized before starting
        # (they may not be initialized if user manually connected hardware)
        await self._ensure_devices_initialized()

        if self._session.state not in (SessionState.READY, SessionState.PAUSED, SessionState.IDLE):
            raise RuntimeError(f"Cannot start experiment in state: {self._session.state}")

        logger.info("Starting experiment")

        # Set up flow from session if not resuming
        if self._session.state != SessionState.PAUSED:
            self.setup_flow()

        # Shared session epoch — captured once, before any recorder is
        # started, and propagated to all three recorders so their
        # `elapsed_ms` columns share t=0. Without this, each recorder
        # anchors elapsed_ms to its own `start_time`, which differs by
        # tens of milliseconds because the recorders are started
        # sequentially below. With it, joining the device CSV, the event
        # log, and the tracking CSV on `elapsed_ms` becomes exact.
        import time as _time

        session_epoch = _time.time()
        self._data_recorder.set_session_epoch(session_epoch)
        self._event_logger.set_session_epoch(session_epoch)
        self._tracking_logger.set_session_epoch(session_epoch)

        # Start data recording if enabled. When a camera is connected and
        # CV processing is on, the recorder switches to camera-driven mode
        # so each row of the device-state CSV is anchored to a camera frame
        # (one row per processed frame, with a `frame` column that matches
        # the tracking CSV and the MP4 frame index). Otherwise it falls
        # back to the periodic timer loop.
        if self._recording_enabled and not self._data_recorder.is_recording:
            experiment_name = self._session.metadata.name or "experiment"
            camera_driven = self._cv_processing_enabled and self._camera_manager.is_connected
            self._data_recorder.set_camera_driven(camera_driven)
            try:
                file_path = await self._data_recorder.start(experiment_name, self._session)
                mode = "camera-driven" if camera_driven else "timer-driven"
                logger.info(f"Recording data ({mode}) to: {file_path}")
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")

        # Start the device event logger if enabled. It subscribes to
        # per-board output callbacks (write_digital/analog/servo) and to
        # per-pin input callbacks for input-capable devices, so every state
        # transition shorter than the sampling interval is still captured.
        if self._recording_enabled and not self._event_logger.is_recording:
            experiment_name = self._session.metadata.name or "experiment"
            try:
                event_path = await self._event_logger.start(experiment_name, self._session)
                logger.info(f"Recording device events to: {event_path}")
            except Exception as e:
                logger.error(f"Failed to start event logger: {e}")

        # Start video recording if enabled and camera is connected
        experiment_name = self._session.metadata.name or "experiment"
        if self._video_recording_enabled:
            # Check for multi-camera mode
            if self._multi_camera_enabled and self._multi_camera_manager.camera_count > 0:
                try:
                    # Record annotated video with tracking overlays if CV is enabled (primary only)
                    record_annotated = self._annotated_video_enabled and self._cv_processing_enabled
                    video_paths = await self._multi_video_recorder.start(
                        experiment_name, record_annotated=record_annotated
                    )
                    for cam_id, path in video_paths.items():
                        logger.info(f"Recording {cam_id} video to: {path}")
                    if record_annotated:
                        logger.info("Also recording annotated video (primary camera only)")
                except Exception as e:
                    logger.error(f"Failed to start multi-camera video recording: {e}")
            elif self._camera_manager.is_connected:
                try:
                    # Single camera mode
                    record_annotated = self._annotated_video_enabled and self._cv_processing_enabled
                    video_path = await self._video_recorder.start(
                        experiment_name, record_annotated=record_annotated
                    )
                    logger.info(f"Recording video to: {video_path}")
                    if record_annotated:
                        logger.info("Also recording annotated video with tracking overlays")
                except Exception as e:
                    logger.error(f"Failed to start video recording: {e}")

        # Start audio recording if configured
        audio_device_name = self._session.camera.audio_device_name
        if audio_device_name:
            device_index = AudioRecorder.resolve_device_by_name(audio_device_name)
            if device_index is None:
                logger.warning(
                    f"Audio device '{audio_device_name}' not found — skipping audio recording"
                )
            else:
                # Update cached index
                self._session.camera.audio_device_index = device_index
                if not AudioRecorder.is_ffmpeg_available():
                    logger.warning(
                        "FFmpeg not found — audio will be saved as .wav but not muxed into video"
                    )
                try:
                    audio_path = await self._audio_recorder.start(
                        experiment_name, device_index=device_index
                    )
                    if audio_path:
                        logger.info(f"Recording audio to: {audio_path}")
                except Exception as e:
                    logger.error(f"Failed to start audio recording: {e}")

        # Start tracking logger if CV processing enabled and camera is connected
        # (separate from video recording so tracking works even if video recording is disabled)
        if self._cv_processing_enabled and self._camera_manager.is_connected:
            try:
                # Hand the operator-supplied bodypart names over before start()
                # so the keypoints CSV can label rows. Sourced from CVSettings
                # here rather than at the settings dialog so it holds however
                # the settings arrived — dialog, .glider load, or default.
                self._tracking_logger.set_keypoint_names(self._cv_processor.settings.keypoint_names)
                tracking_path = await self._tracking_logger.start(
                    experiment_name, session=self._session
                )
                logger.info(f"Tracking data to: {tracking_path}")
            except Exception as e:
                logger.error(f"Failed to start tracking logger: {e}")

        # Reset prior-run timing so ``last_flow_duration_s`` is None until
        # this flow completes (callers can detect "still running" cleanly).
        self._flow_start_monotonic = None
        self._flow_end_monotonic = None
        self._flow_start_wall = None
        self._flow_end_wall = None

        self._session.state = SessionState.RUNNING
        await self._flow_engine.start()
        # Capture the flow's logical start AFTER the engine is live —
        # i.e., the moment the StartExperiment node is about to be
        # scheduled. Anything that ran before this (recorder setup,
        # state transition, board init) is *pre*-flow and is correctly
        # excluded from the reported duration.
        self._flow_start_monotonic = time.monotonic()
        self._flow_start_wall = time.time()

        # Anchor the recorders' ``flow_elapsed_ms`` column and emit a
        # ``flow_marker`` row into the event log. Done immediately after
        # the monotonic capture so all three artifacts (event log,
        # tracking CSV, data CSV) reference the same wall-clock instant
        # as t=0 for the flow. Frames/samples captured during pre-flow
        # setup get an empty flow_elapsed_ms cell; everything from here
        # forward is flow-aligned.
        self._tracking_logger.set_flow_anchor(self._flow_start_wall)
        self._data_recorder.set_flow_anchor(self._flow_start_wall)
        self._event_logger.record_flow_marker("start")

    async def stop_experiment(self) -> None:
        """Stop the running experiment and set all devices to safe state."""
        async with self._experiment_lock:
            await self._stop_experiment_locked()

    async def _stop_experiment_locked(self) -> None:
        """Internal stop implementation, called under _experiment_lock."""
        if self._session is None:
            return

        # Capture the flow's logical end NOW, before any teardown I/O.
        # The operator clicked STOP at this instant — that's the
        # truth-of-record for "how long did the experiment run."
        # Stopping recorders, draining cancellations, and driving
        # devices low all happen on the operator's clock; they're
        # post-flow and must not inflate the reported duration.
        if self._flow_start_monotonic is not None and self._flow_end_monotonic is None:
            self._flow_end_monotonic = time.monotonic()
            self._flow_end_wall = time.time()
            # Stamp the boundary into the event log before _stop_recorders
            # tears it down, so a stopped-by-operator run still has a
            # locatable end-of-flow marker for post-hoc analysis.
            self._event_logger.record_flow_marker("end")

        logger.info("Stopping experiment")
        self._session.state = SessionState.STOPPING
        await self._flow_engine.stop()
        await self._stop_recorders()

        # Set all devices to LOW/safe state for safety
        await self._set_all_devices_low()

        self._session.state = SessionState.READY

    async def _stop_recorders(self) -> None:
        """Stop all active recorders (data, audio, video, tracking)."""
        # Stop audio recording first (need the WAV path for muxing)
        audio_path = None
        if self._audio_recorder.is_recording:
            try:
                audio_path = await self._audio_recorder.stop()
                logger.info(f"Audio saved to: {audio_path}")
            except Exception as e:
                logger.error(f"Failed to stop audio recording: {e}")

        # Stop the event logger BEFORE the data recorder so subscriptions
        # are torn down while boards/devices are still live; otherwise a
        # straggling input callback could try to write to a closed CSV.
        if self._event_logger.is_recording:
            try:
                event_path = await self._event_logger.stop()
                logger.info(f"Event log saved to: {event_path}")
            except Exception as e:
                logger.error(f"Failed to stop event logger: {e}")

        # Stop data recording
        if self._data_recorder.is_recording:
            try:
                file_path = await self._data_recorder.stop()
                logger.info(f"Recording saved to: {file_path}")
            except Exception as e:
                logger.error(f"Failed to stop recording: {e}")

        # Stop video recording and collect all video paths
        video_paths: list[Path] = []

        if self._multi_video_recorder.is_recording:
            try:
                multi_paths = await self._multi_video_recorder.stop()
                for cam_id, path in multi_paths.items():
                    logger.info(f"Video {cam_id} saved to: {path}")
                    video_paths.append(path)
                # Check for annotated video
                annotated = self._multi_video_recorder.annotated_file_path
                if annotated and annotated.exists():
                    video_paths.append(annotated)
            except Exception as e:
                logger.error(f"Failed to stop multi-camera video recording: {e}")
        elif self._video_recorder.is_recording:
            try:
                video_path = await self._video_recorder.stop()
                if video_path:
                    logger.info(f"Video saved to: {video_path}")
                    video_paths.append(video_path)
                # Check for annotated video
                annotated = self._video_recorder.annotated_file_path
                if annotated and annotated.exists():
                    video_paths.append(annotated)
            except Exception as e:
                logger.error(f"Failed to stop video recording: {e}")

        # Mux audio into each video file
        if audio_path and audio_path.exists() and video_paths:
            all_muxed = True
            for vpath in video_paths:
                try:
                    success = await mux_audio_video(vpath, audio_path)
                    if not success:
                        all_muxed = False
                except Exception as e:
                    logger.error(f"Failed to mux audio into {vpath}: {e}")
                    all_muxed = False

            # Only delete WAV if ALL muxes succeeded
            if all_muxed:
                try:
                    audio_path.unlink()
                    logger.info("Deleted intermediate audio WAV file")
                except Exception as e:
                    logger.error(f"Failed to delete WAV file: {e}")
            else:
                logger.warning(f"Some muxes failed — keeping {audio_path} for manual retry")

        # Stop tracking logger
        if self._tracking_logger.is_recording:
            try:
                # Pass the authoritative flow duration so the CSV footer's
                # ``# Duration (s)`` line reflects the flow's logical
                # duration, not recorder-start-to-recorder-stop wall-clock
                # (which includes pre-flow setup + post-flow teardown
                # latency — variable, and exactly the source of the
                # 10.11s / 10.43s drift the user reported).
                tracking_path = await self._tracking_logger.stop(
                    flow_duration_s=self.last_flow_duration_s
                )
                logger.info(f"Tracking data saved to: {tracking_path}")
            except Exception as e:
                logger.error(f"Failed to stop tracking logger: {e}")

    async def _set_all_devices_low(self) -> None:
        """Set all output devices to LOW/off state for safety."""
        for device_id, device in self._hardware_manager.devices.items():
            try:
                if hasattr(device, "shutdown"):
                    await device.shutdown()
                    logger.debug(f"Set device {device_id} to safe state")
            except Exception as e:
                logger.error(f"Error setting device {device_id} to safe state: {e}")

    async def pause_experiment(self) -> None:
        """Pause the running experiment."""
        if self._session is None or self._session.state != SessionState.RUNNING:
            return

        logger.info("Pausing experiment")
        await self._flow_engine.pause()

        # Pause recorders so audio/video stay in sync
        if self._audio_recorder.is_recording:
            self._audio_recorder.pause()
        if self._video_recorder.is_recording:
            await self._video_recorder.pause()
        if self._multi_video_recorder.is_recording:
            await self._multi_video_recorder.pause()

        self._session.state = SessionState.PAUSED

    async def resume_experiment(self) -> None:
        """Resume a paused experiment."""
        if self._session is None or self._session.state != SessionState.PAUSED:
            return

        logger.info("Resuming experiment")
        await self._flow_engine.resume()

        # Resume recorders
        if self._audio_recorder.is_recording:
            self._audio_recorder.resume()
        if self._video_recorder.is_recording:
            await self._video_recorder.resume()
        if self._multi_video_recorder.is_recording:
            await self._multi_video_recorder.resume()

        self._session.state = SessionState.RUNNING

    async def emergency_stop(self) -> None:
        """
        Trigger emergency stop.

        Stops all hardware and flow execution immediately.
        """
        logger.warning("EMERGENCY STOP triggered!")

        # Stop flow first
        if self._flow_engine.is_running:
            await self._flow_engine.stop()

        # Stop all recorders
        await self._stop_recorders()

        # Emergency stop hardware
        await self._hardware_manager.emergency_stop()

        # Update state
        if self._session:
            self._session.state = SessionState.ERROR

    async def shutdown(self) -> None:
        """Shutdown the GLIDER core."""
        if self._shutting_down:
            return

        self._shutting_down = True
        logger.info("Shutting down GLIDER Core...")

        # Stop experiment if running or paused
        if self._session and self._session.state in (SessionState.RUNNING, SessionState.PAUSED):
            await self.stop_experiment()

        # Disconnect cameras
        if self._camera_manager.is_connected:
            self._camera_manager.disconnect()
            logger.info("Camera disconnected")

        # Shutdown multi-camera manager
        self._multi_camera_manager.shutdown()

        # Drain fire-and-forget tasks BEFORE tearing down the flow engine /
        # hardware manager: a flow-complete teardown scheduled just before
        # shutdown (_handle_flow_complete drives all devices to a safe LOW
        # state) must be allowed to finish rather than race the shutdown
        # teardown over the same state. Wait first — do NOT cancel first —
        # then cancel and await any stragglers (mirrors FlowEngine.stop()).
        if self._background_tasks:
            # Snapshot: done-callbacks mutate the live set.
            pending = set(self._background_tasks)
            _done, still_pending = await asyncio.wait(pending, timeout=5.0)
            for task in still_pending:
                logger.warning("Background task did not finish before shutdown; cancelling")
                task.cancel()
            if still_pending:
                await asyncio.gather(*still_pending, return_exceptions=True)

        # Shutdown flow engine
        await self._flow_engine.shutdown()

        # Shutdown hardware
        await self._hardware_manager.shutdown()

        # Unload plugins
        if self._plugin_manager:
            await self._plugin_manager.unload_all()

        self._initialized = False
        logger.info("GLIDER Core shutdown complete")

    def get_available_board_types(self) -> list[dict[str, Any]]:
        """Get list of available board types."""
        board_types = []
        for driver_name in self._hardware_manager.get_available_drivers():
            driver_class = self._hardware_manager.get_driver_class(driver_name)
            if driver_class:
                info = {
                    "driver": driver_name,
                    "name": driver_class.__name__,
                }
                # Add board subtypes if available
                if hasattr(driver_class, "BOARD_CONFIGS"):
                    info["subtypes"] = list(driver_class.BOARD_CONFIGS.keys())
                board_types.append(info)
        return board_types

    def get_available_device_types(self) -> list[str]:
        """Get list of available device types."""
        from glider.hal.base_device import DEVICE_REGISTRY

        return list(DEVICE_REGISTRY.keys())

    def get_available_node_types(self) -> list[str]:
        """Get list of available node types."""
        return self._flow_engine.get_available_nodes()


# Convenience function to create and initialize core
async def create_core() -> GliderCore:
    """Create and initialize a GliderCore instance."""
    core = GliderCore()
    await core.initialize()
    return core
