"""
GLIDER Core - The central orchestrator.

The main controller that initializes the event loop, loads plugins,
manages the ExperimentSession, and coordinates between hardware
and flow execution.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from glider.core.data_recorder import DataRecorder
from glider.core.experiment_session import ExperimentSession, SessionState
from glider.core.flow_engine import FlowEngine
from glider.core.hardware_manager import HardwareManager
from glider.vision.calibration import CameraCalibration
from glider.vision.camera_manager import CameraManager
from glider.vision.cv_processor import CVProcessor
from glider.vision.multi_camera_manager import MultiCameraManager
from glider.vision.multi_video_recorder import MultiVideoRecorder
from glider.vision.tracking_logger import TrackingDataLogger
from glider.vision.video_recorder import VideoRecorder

if TYPE_CHECKING:
    from glider.agent.agent_controller import AgentController
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
        self._session: Optional[ExperimentSession] = None
        self._hardware_manager = HardwareManager()
        self._flow_engine = FlowEngine(self._hardware_manager)
        self._plugin_manager: Optional[PluginManager] = None
        self._data_recorder = DataRecorder(self._hardware_manager)

        # Vision components
        self._camera_manager = CameraManager()
        self._cv_processor = CVProcessor()
        self._video_recorder = VideoRecorder(self._camera_manager)
        self._tracking_logger = TrackingDataLogger()
        self._calibration = CameraCalibration()

        # Multi-camera support
        self._multi_camera_manager = MultiCameraManager()
        self._multi_video_recorder = MultiVideoRecorder(self._multi_camera_manager)
        self._multi_camera_enabled = False

        # Agent (lazy initialization)
        self._agent_controller: Optional[AgentController] = None

        self._initialized = False
        self._shutting_down = False
        self._recording_enabled = True  # Auto-record experiments by default
        self._video_recording_enabled = True  # Auto-record video when camera connected
        self._annotated_video_enabled = True  # Also save annotated video with tracking overlays
        self._cv_processing_enabled = True  # Enable CV processing by default

        # Callbacks
        self._session_callbacks: list[Callable[[ExperimentSession], None]] = []
        self._state_callbacks: list[Callable[[SessionState], None]] = []
        self._error_callbacks: list[Callable[[str, Exception], None]] = []

        # Register hardware error handling
        self._hardware_manager.on_error(self._on_hardware_error)
        self._flow_engine.on_error(self._on_flow_error)
        self._flow_engine.on_flow_complete(self._on_flow_complete)

    @property
    def session(self) -> Optional[ExperimentSession]:
        """Current experiment session."""
        return self._session

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
    def agent_controller(self) -> "AgentController":
        """
        Agent controller instance (lazy initialization).

        Creates the agent controller on first access.
        """
        if self._agent_controller is None:
            from glider.agent.agent_controller import AgentController

            self._agent_controller = AgentController(self)
        return self._agent_controller

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
        self._data_recorder._output_dir = path
        self._video_recorder.set_output_directory(path)
        self._multi_video_recorder.set_output_directory(path)
        self._tracking_logger.set_output_directory(path)

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

    def _on_flow_complete(self) -> None:
        """Handle flow completion (EndExperiment reached)."""
        logger.info("Flow completed - transitioning to READY state")
        # Schedule the async completion handler
        task = asyncio.create_task(self._handle_flow_complete())
        task.add_done_callback(self._log_task_exception)

    @staticmethod
    def _log_task_exception(task: asyncio.Task) -> None:
        """Log exceptions from fire-and-forget tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"Unhandled error in background task: {exc}", exc_info=exc)

    async def _handle_flow_complete(self) -> None:
        """Async handler for flow completion."""
        # Stop the flow engine
        await self._flow_engine.stop()

        # Stop data recording
        if self._data_recorder.is_recording:
            try:
                file_path = await self._data_recorder.stop()
                logger.info(f"Recording saved to: {file_path}")
            except Exception as e:
                logger.error(f"Failed to stop recording: {e}")

        # Stop video recording
        if self._multi_video_recorder.is_recording:
            try:
                video_paths = await self._multi_video_recorder.stop()
                for cam_id, path in video_paths.items():
                    logger.info(f"Video {cam_id} saved to: {path}")
            except Exception as e:
                logger.error(f"Failed to stop multi-camera video recording: {e}")
        elif self._video_recorder.is_recording:
            try:
                video_path = await self._video_recorder.stop()
                logger.info(f"Video saved to: {video_path}")
            except Exception as e:
                logger.error(f"Failed to stop video recording: {e}")

        # Stop tracking logger
        if self._tracking_logger.is_recording:
            try:
                tracking_path = await self._tracking_logger.stop()
                logger.info(f"Tracking data saved to: {tracking_path}")
            except Exception as e:
                logger.error(f"Failed to stop tracking logger: {e}")

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
            from glider.nodes.flow_function_nodes import register_flow_function_nodes

            register_flow_function_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register flow function nodes: {e}")

        try:
            from glider.nodes.vision.zone_nodes import register_zone_nodes

            register_zone_nodes(self._flow_engine)
        except Exception as e:
            logger.error(f"Failed to register zone nodes: {e}")

    async def _load_plugins(self) -> None:
        """Load plugins from the plugin directory."""
        try:
            from glider.plugins.plugin_manager import PluginManager

            self._plugin_manager = PluginManager()
            await self._plugin_manager.discover_plugins()
            await self._plugin_manager.load_plugins()
        except ImportError:
            logger.warning("Plugin manager not available")
        except Exception as e:
            logger.error(f"Error loading plugins: {e}")

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
        )
        logger.info(f"Saved experiment to {file_path}")

    def save_session(self, file_path: Optional[str] = None) -> str:
        """
        Save the current session to file.

        Args:
            file_path: Path to save to (uses existing if None)

        Returns:
            Path to saved file
        """
        if self._session is None:
            raise RuntimeError("No session to save")

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

        # Start data recording if enabled
        if self._recording_enabled and not self._data_recorder.is_recording:
            experiment_name = self._session.metadata.name or "experiment"
            try:
                file_path = await self._data_recorder.start(experiment_name, self._session)
                logger.info(f"Recording data to: {file_path}")
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")

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

        # Start tracking logger if CV processing enabled and camera is connected
        # (separate from video recording so tracking works even if video recording is disabled)
        if self._cv_processing_enabled and self._camera_manager.is_connected:
            try:
                tracking_path = await self._tracking_logger.start(
                    experiment_name, session=self._session
                )
                logger.info(f"Tracking data to: {tracking_path}")
            except Exception as e:
                logger.error(f"Failed to start tracking logger: {e}")

        self._session.state = SessionState.RUNNING
        await self._flow_engine.start()

    async def stop_experiment(self) -> None:
        """Stop the running experiment and set all devices to safe state."""
        if self._session is None:
            return

        logger.info("Stopping experiment")
        self._session.state = SessionState.STOPPING
        await self._flow_engine.stop()

        # Stop data recording
        if self._data_recorder.is_recording:
            try:
                file_path = await self._data_recorder.stop()
                logger.info(f"Recording saved to: {file_path}")
            except Exception as e:
                logger.error(f"Failed to stop recording: {e}")

        # Stop video recording
        if self._multi_video_recorder.is_recording:
            try:
                video_paths = await self._multi_video_recorder.stop()
                for cam_id, path in video_paths.items():
                    logger.info(f"Video {cam_id} saved to: {path}")
            except Exception as e:
                logger.error(f"Failed to stop multi-camera video recording: {e}")
        elif self._video_recorder.is_recording:
            try:
                video_path = await self._video_recorder.stop()
                logger.info(f"Video saved to: {video_path}")
            except Exception as e:
                logger.error(f"Failed to stop video recording: {e}")

        # Stop tracking logger
        if self._tracking_logger.is_recording:
            try:
                tracking_path = await self._tracking_logger.stop()
                logger.info(f"Tracking data saved to: {tracking_path}")
            except Exception as e:
                logger.error(f"Failed to stop tracking logger: {e}")

        # Set all devices to LOW/safe state for safety
        await self._set_all_devices_low()

        self._session.state = SessionState.READY

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
        self._session.state = SessionState.PAUSED

    async def resume_experiment(self) -> None:
        """Resume a paused experiment."""
        if self._session is None or self._session.state != SessionState.PAUSED:
            return

        logger.info("Resuming experiment")
        await self._flow_engine.resume()
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

        # Stop experiment if running
        if self._session and self._session.state == SessionState.RUNNING:
            await self.stop_experiment()

        # Disconnect cameras
        if self._camera_manager.is_connected:
            self._camera_manager.disconnect()
            logger.info("Camera disconnected")

        # Shutdown multi-camera manager
        self._multi_camera_manager.shutdown()

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
