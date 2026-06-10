"""
Device Event Logger - Records device state-change *events* to CSV.

Complementary to DataRecorder, which writes periodic state *samples* (one
row per camera frame in camera-driven mode, or one row every
sample_interval seconds otherwise). Periodic sampling can miss device
transitions shorter than the sample interval; this logger captures
*every* state change with full sub-frame timing precision.

Two event streams are written into a single CSV:

* ``input_change`` rows: fired by the existing per-pin input callbacks
  (``BaseBoard.register_callback``) when the board reports a pin
  transition (Telemetrix digital/analog input callbacks, gpiozero edge
  notifications, etc.). Subscribed per-pin for devices whose ``actions``
  dict includes a ``"read"`` entry.

* ``output_write`` rows: fired by the new per-board output callbacks
  (``BaseBoard.register_output_callback``) after a successful
  ``write_digital`` / ``write_analog`` / ``write_pwm`` / ``write_servo``.

Each row records ``(frame, timestamp, elapsed_ms, source, board_id,
device_id, device_type, pin, pin_type, value)``. ``frame`` is the
latest camera frame index seen via ``set_current_frame`` so events can
be joined to video frames; it is empty for events that occur before the
first frame arrives or in headless sessions.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from glider.hal.base_board import PinType

if TYPE_CHECKING:
    from glider.core.experiment_session import ExperimentSession
    from glider.core.hardware_manager import HardwareManager
    from glider.hal.base_board import BaseBoard
    from glider.hal.base_device import BaseDevice

logger = logging.getLogger(__name__)

# Periodic fsync interval, in number of rows written. Trades off durability
# against SD-card wear on the Raspberry Pi. Bursts of events between fsyncs
# are bounded to this many rows in a crash.
_FSYNC_INTERVAL_ROWS = 200


class DeviceEventLogger:
    """
    Records device state-change events to a CSV file.

    Output filename: ``<experiment>_<YYYYMMDD_HHMMSS>_events.csv``.

    Lifecycle: instantiate once, then call ``start(...)`` at experiment
    start and ``stop()`` at experiment end. Between start and stop, the
    camera pipeline should call ``set_current_frame(frame_no, frame_ts)``
    once per processed frame so events get stamped with the current frame
    index.
    """

    def __init__(
        self,
        hardware_manager: HardwareManager,
        output_dir: Path | None = None,
    ):
        """
        Args:
            hardware_manager: Source of boards/devices to subscribe to.
            output_dir: Directory for the CSV file (defaults to cwd).
        """
        self._hardware_manager = hardware_manager
        self._output_dir = Path(output_dir) if output_dir else Path.cwd()

        self._recording = False
        self._file: Any | None = None
        self._writer: csv.writer | None = None
        self._file_path: Path | None = None
        self._start_time: datetime | None = None
        # See DataRecorder.set_session_epoch — when set, the event log
        # measures elapsed_ms against the shared session epoch instead
        # of its own start_time so rows are joinable across recorders.
        self._start_timestamp: float = 0.0
        self._session_epoch_override: float | None = None

        # Most recent camera frame; pushed in by CameraPanel via
        # set_current_frame(). Empty/zero until the first frame arrives.
        self._current_frame: int | None = None
        self._current_frame_ts: float = 0.0

        # Subscriptions are tracked so stop() can cleanly tear them down
        # without leaving dangling references on boards that outlive a
        # session.
        self._input_subscriptions: list[tuple[BaseBoard, int, Any]] = []  # (board, pin, callback)
        self._output_subscriptions: list[tuple[BaseBoard, Any]] = []  # (board, callback)

        # fsync bookkeeping (mirrors TrackingDataLogger's pattern).
        self._rows_since_fsync = 0

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def file_path(self) -> Path | None:
        return self._file_path

    def set_output_directory(self, path: Path) -> None:
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def set_session_epoch(self, epoch: float) -> None:
        """
        Anchor ``elapsed_ms`` against a shared Unix timestamp instead of
        this logger's own ``start_time``. Must be called before ``start()``.
        See DataRecorder.set_session_epoch for the joinability rationale.
        """
        if self._recording:
            logger.warning("set_session_epoch called while recording; takes effect on next start()")
        self._session_epoch_override = float(epoch)

    # ------------------------------------------------------------------
    # Frame tracking
    # ------------------------------------------------------------------

    def set_current_frame(self, frame_no: int, frame_ts: float) -> None:
        """
        Update the most-recent camera frame index and timestamp.

        Called by the camera pipeline once per processed frame so any
        subsequent event rows are stamped with this frame index. No-op if
        the logger is not recording.

        Args:
            frame_no: Canonical camera frame index (matches
                TrackingDataLogger.frame_count for the same frame).
            frame_ts: Unix timestamp of the frame (time.time() captured at
                frame readback inside CameraManager).
        """
        if not self._recording:
            return
        self._current_frame = int(frame_no)
        self._current_frame_ts = float(frame_ts)

    # ------------------------------------------------------------------
    # Flow markers
    # ------------------------------------------------------------------

    def record_flow_marker(self, marker: str) -> None:
        """
        Record a flow boundary marker to the event log.

        Writes a row with ``source="flow_marker"`` and the marker name
        (typically ``"start"`` / ``"end"``) in the ``value`` cell, with
        the same wall-clock timestamp and ``elapsed_ms`` as any other
        event written at this instant. Analysts grep these rows to find
        the exact boundary times so post-hoc analysis (ethogram / raster
        plot / video sync) can trim and align against the same flow
        boundaries the runner reports — no offset detective work.

        Args:
            marker: Marker name to record (``"start"`` / ``"end"``).
        """
        if not self._recording or self._writer is None:
            return

        now = time.time()
        elapsed_ms = (now - self._start_timestamp) * 1000 if self._start_timestamp else 0.0
        iso = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
        frame_cell = "" if self._current_frame is None else str(self._current_frame)

        try:
            self._writer.writerow(
                [
                    frame_cell,
                    iso,
                    f"{elapsed_ms:.1f}",
                    "flow_marker",
                    "",  # board_id — flow markers are not tied to a board
                    "",  # device_id
                    "",  # device_type
                    "",  # pin
                    "",  # pin_type
                    str(marker),
                ]
            )
            # Flow markers are infrequent but load-bearing — force fsync
            # so they're guaranteed durable even if the machine loses
            # power before the next event/sample lands.
            self._fsync(force=True)
        except Exception:
            logger.exception("DeviceEventLogger: flow marker write failed")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _generate_filename(self, experiment_name: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in experiment_name)
        safe_name = safe_name.strip().replace(" ", "_") or "experiment"
        return f"{safe_name}_{timestamp}_events.csv"

    async def start(
        self,
        experiment_name: str = "experiment",
        session: ExperimentSession | None = None,
    ) -> Path:
        """
        Open the events CSV and subscribe to board/device event sources.

        Args:
            experiment_name: Base name for the output file.
            session: Optional experiment session for metadata header.

        Returns:
            Path to the events CSV being written.
        """
        if self._recording:
            logger.warning("DeviceEventLogger already recording")
            return self._file_path  # type: ignore[return-value]

        filename = self._generate_filename(experiment_name)
        self._file_path = self._output_dir / filename
        self._start_time = datetime.now()
        self._start_timestamp = (
            self._session_epoch_override
            if self._session_epoch_override is not None
            else self._start_time.timestamp()
        )
        self._current_frame = None
        self._current_frame_ts = 0.0
        self._rows_since_fsync = 0

        self._file = open(self._file_path, "w", newline="", encoding="utf-8")
        try:
            self._writer = csv.writer(self._file)
            self._write_header(experiment_name, session)

            self._recording = True
            self._subscribe_all()
        except Exception:
            self._file.close()
            self._file = None
            self._writer = None
            raise

        logger.info(f"Started device event log: {self._file_path}")
        return self._file_path

    async def stop(self) -> Path | None:
        """Unsubscribe from boards/devices and close the events CSV."""
        if not self._recording:
            return None

        # Stop receiving events first so no row is written after close().
        self._recording = False
        self._unsubscribe_all()

        if self._writer and self._file:
            end_time = datetime.now()
            duration = (end_time - self._start_time).total_seconds() if self._start_time else 0
            self._writer.writerow([])
            self._writer.writerow(["# End Time", end_time.isoformat()])
            self._writer.writerow(["# Duration (s)", f"{duration:.2f}"])
            self._fsync(force=True)

        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

        path = self._file_path
        logger.info(f"Stopped device event log. Saved to {path}")
        return path

    # ------------------------------------------------------------------
    # Header / metadata
    # ------------------------------------------------------------------

    def _write_header(self, experiment_name: str, session: ExperimentSession | None) -> None:
        assert self._writer is not None

        self._writer.writerow(["# GLIDER Device Event Log"])
        self._writer.writerow(["# Experiment", experiment_name])
        self._writer.writerow(["# Start Time", self._start_time.isoformat()])

        if session and getattr(session, "metadata", None):
            md = session.metadata
            if getattr(md, "protocol", None):
                self._writer.writerow(["# Protocol", md.protocol])
            if getattr(md, "experiment_type", None):
                self._writer.writerow(["# Experiment Type", md.experiment_type])
            if getattr(md, "experimenter", None):
                self._writer.writerow(["# Experimenter", md.experimenter])

        self._writer.writerow([])
        self._writer.writerow(
            [
                "frame",
                "timestamp",
                "elapsed_ms",
                "source",
                "board_id",
                "device_id",
                "device_type",
                "pin",
                "pin_type",
                "value",
            ]
        )
        self._fsync(force=True)

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    @staticmethod
    def _device_is_input_capable(device: BaseDevice) -> bool:
        """
        Heuristic: a device is treated as input-capable if its ``actions``
        dict advertises a ``"read"`` operation. Subscribing the per-pin
        input callback only for these devices avoids double-counting events
        on boards (like MockBoard) that fire ``_notify_callbacks`` from
        their write path for testing convenience.
        """
        actions = getattr(device, "actions", None)
        if actions is None:
            return False
        try:
            return "read" in actions
        except TypeError:
            return False

    def _subscribe_all(self) -> None:
        """Subscribe to per-board output callbacks and per-pin input callbacks."""

        # Output stream: one subscription per board. The callback receives
        # (pin, pin_type, value); we resolve the owning device by
        # (board, pin) at log time.
        for board in self._hardware_manager.boards.values():
            cb = self._on_output_change_factory(board)
            try:
                board.register_output_callback(cb)
                self._output_subscriptions.append((board, cb))
            except Exception:
                logger.exception(
                    "Failed to register output callback on board %s",
                    getattr(board, "id", "<unknown>"),
                )

        # Input stream: one subscription per pin per input-capable device.
        for device in self._hardware_manager.devices.values():
            if not self._device_is_input_capable(device):
                continue
            board = getattr(device, "_board", None) or getattr(device, "board", None)
            config = getattr(device, "_config", None)
            if board is None or config is None:
                continue
            for pin in config.pins.values():
                try:
                    cb = self._on_input_change_factory(board, device, pin)
                    board.register_callback(int(pin), cb)
                    self._input_subscriptions.append((board, int(pin), cb))
                except Exception:
                    logger.exception(
                        "Failed to register input callback on board %s pin %s",
                        getattr(board, "id", "<unknown>"),
                        pin,
                    )

    def _unsubscribe_all(self) -> None:
        for board, pin, cb in self._input_subscriptions:
            try:
                board.unregister_callback(pin, cb)
            except Exception:
                logger.debug("Failed to unregister input callback", exc_info=True)
        self._input_subscriptions.clear()

        for board, cb in self._output_subscriptions:
            try:
                board.unregister_output_callback(cb)
            except Exception:
                logger.debug("Failed to unregister output callback", exc_info=True)
        self._output_subscriptions.clear()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_output_change_factory(self, board: BaseBoard):
        board_id = board.id

        def _cb(pin: int, pin_type: PinType, value: Any) -> None:
            self._write_event(
                source="output_write",
                board_id=board_id,
                device=self._find_device_for_pin(board, int(pin)),
                pin=int(pin),
                pin_type=pin_type,
                value=value,
            )

        return _cb

    def _on_input_change_factory(self, board: BaseBoard, device: BaseDevice, pin: int):
        board_id = board.id

        def _cb(pin_arg: int, value: Any) -> None:
            self._write_event(
                source="input_change",
                board_id=board_id,
                device=device,
                pin=int(pin_arg),
                # Inputs don't carry pin_type through the legacy callback
                # signature; we record the device's expected type if
                # available, otherwise leave blank.
                pin_type=None,
                value=value,
            )

        return _cb

    # ------------------------------------------------------------------
    # Row writing
    # ------------------------------------------------------------------

    def _find_device_for_pin(self, board: BaseBoard, pin: int) -> BaseDevice | None:
        for device in self._hardware_manager.devices.values():
            d_board = getattr(device, "_board", None) or getattr(device, "board", None)
            d_config = getattr(device, "_config", None)
            if d_board is None or d_config is None:
                continue
            if getattr(d_board, "id", None) != board.id:
                continue
            if pin in {int(p) for p in d_config.pins.values()}:
                return device
        return None

    def _write_event(
        self,
        source: str,
        board_id: str,
        device: BaseDevice | None,
        pin: int,
        pin_type: PinType | None,
        value: Any,
    ) -> None:
        if not self._recording or self._writer is None:
            return

        now = time.time()
        elapsed_ms = (now - self._start_timestamp) * 1000 if self._start_timestamp else 0.0
        iso = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
        frame_cell = "" if self._current_frame is None else str(self._current_frame)

        device_id = device.id if device is not None else ""
        device_type = getattr(device, "device_type", "") if device is not None else ""
        pin_type_cell = pin_type.name if isinstance(pin_type, PinType) else ""

        # Format value uniformly with DataRecorder.
        if value is None:
            value_cell = ""
        elif isinstance(value, bool):
            value_cell = "1" if value else "0"
        elif isinstance(value, float):
            value_cell = f"{value:.4f}"
        else:
            value_cell = str(value)

        try:
            self._writer.writerow(
                [
                    frame_cell,
                    iso,
                    f"{elapsed_ms:.1f}",
                    source,
                    board_id,
                    device_id,
                    device_type,
                    pin,
                    pin_type_cell,
                    value_cell,
                ]
            )
            self._fsync()
        except Exception:
            logger.exception("DeviceEventLogger: row write failed")

    def _fsync(self, force: bool = False) -> None:
        if self._file is None:
            return
        try:
            self._file.flush()
        except Exception:
            logger.exception("DeviceEventLogger: flush failed")
            return
        self._rows_since_fsync += 1
        if force or self._rows_since_fsync >= _FSYNC_INTERVAL_ROWS:
            try:
                os.fsync(self._file.fileno())
            except OSError:
                logger.debug("DeviceEventLogger: os.fsync failed", exc_info=True)
            self._rows_since_fsync = 0
