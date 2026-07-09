"""
Telemetrix-AIO board implementation for Arduino communication.

Telemetrix is actively maintained and supports asynchronous operation natively.
It communicates with the Arduino over serial (USB) using a custom protocol
that allows for callback-based reporting.
"""

import asyncio
import contextlib
import io
import logging
import threading
from typing import Any, Optional

from glider.hal.base_board import (
    BaseBoard,
    BoardCapabilities,
    BoardConnectionState,
    PinCapability,
    PinMode,
    PinType,
)

logger = logging.getLogger(__name__)


class TelemetrixThread:
    """
    Runs telemetrix in a dedicated background thread with its own event loop.
    This isolates telemetrix's async operations from the Qt event loop.
    """

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._telemetrix = None
        self._ready = threading.Event()
        self._stop_event = threading.Event()

    def start(self, port: str | None, sleep_tune: float = 0.0001):
        """Start the telemetrix thread and connect to the board."""
        self._stop_event.clear()
        self._ready.clear()
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._run, args=(port, sleep_tune), daemon=True)
        self._thread.start()
        # Wait for connection to complete
        if not self._ready.wait(timeout=10.0):
            # If we time out, telemetrix may have partially initialised and is
            # still running in the background with an open serial handle. Tear
            # the thread down before raising so we don't leak the serial port
            # (and then fail subsequent reconnect attempts with "port busy").
            logger.warning("Telemetrix connection timed out after 10s; cleaning up worker thread")
            try:
                self.stop()
            except Exception as cleanup_exc:
                logger.warning(
                    f"Error during telemetrix cleanup after connect timeout: {cleanup_exc}"
                )
            raise RuntimeError("Telemetrix connection timed out")
        if self._error is not None:
            # _run() already set _telemetrix to None on error, and its finally
            # block closes the loop. Join the worker to avoid leaving a zombie.
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            raise self._error
        if self._telemetrix is None:
            # Best-effort cleanup: the worker should have exited on its own,
            # but join it so we don't leak.
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            raise RuntimeError("Failed to connect to Arduino")

    def _run(self, port: str | None, sleep_tune: float):
        """Thread entry point - creates its own event loop."""
        # Suppress telemetrix debug output via logging
        logging.getLogger("telemetrix_aio").setLevel(logging.WARNING)
        logging.getLogger("telemetrix_aio.telemetrix_aio").setLevel(logging.WARNING)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            # Suppress print statements from telemetrix only during connect
            with contextlib.redirect_stdout(io.StringIO()):
                self._loop.run_until_complete(self._connect(port, sleep_tune))

            self._ready.set()
            # Run the event loop to process tasks
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Telemetrix thread error: {e}")
            self._error = e
            self._telemetrix = None  # Mark as failed
            self._ready.set()  # Unblock waiting caller
        finally:
            if self._loop is not None:
                self._loop.close()

    async def _connect(self, port: str | None, sleep_tune: float):
        """Connect to telemetrix within the thread's event loop."""
        from telemetrix_aio import telemetrix_aio

        if port:
            self._telemetrix = telemetrix_aio.TelemetrixAIO(
                com_port=port,
                autostart=False,
                sleep_tune=sleep_tune,
                close_loop_on_shutdown=False,
            )
        else:
            self._telemetrix = telemetrix_aio.TelemetrixAIO(
                autostart=False,
                sleep_tune=sleep_tune,
                close_loop_on_shutdown=False,
            )
        await self._telemetrix.start_aio()

    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method on the telemetrix instance from the main thread.
        The method is executed in the telemetrix thread's event loop.
        """
        if self._loop is None or self._telemetrix is None:
            raise RuntimeError("Telemetrix not connected")

        # Check if the loop is still running
        if self._loop.is_closed():
            raise RuntimeError("Telemetrix event loop is closed - board may have disconnected")

        async def _call():
            method = getattr(self._telemetrix, method_name)
            # Use wait_for so the coroutine is cancelled on timeout, unblocking
            # the telemetrix event loop for other tasks (e.g. the serial reporter
            # that delivers analog callbacks).
            return await asyncio.wait_for(method(*args, **kwargs), timeout=5.0)

        try:
            future = asyncio.run_coroutine_threadsafe(_call(), self._loop)
            # Outer timeout is longer than inner wait_for to let cancellation propagate
            return future.result(timeout=10.0)
        except TimeoutError:
            future.cancel()
            raise RuntimeError(f"Telemetrix operation '{method_name}' timed out after 5s") from None
        except RuntimeError as e:
            if "Event loop is closed" in str(e):
                raise RuntimeError("Telemetrix connection lost - please reconnect the board") from e
            raise

    def stop(self):
        """Stop the telemetrix thread."""
        if self._telemetrix is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._telemetrix.shutdown(), self._loop)
                future.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"Error during telemetrix shutdown: {e}")

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._telemetrix = None
        self._loop = None
        self._thread = None

    @property
    def telemetrix(self):
        return self._telemetrix

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


# Standard Arduino Uno pin capabilities
ARDUINO_UNO_PINS = {
    0: PinCapability(0, {PinType.DIGITAL}, description="Digital (RX)"),
    1: PinCapability(1, {PinType.DIGITAL}, description="Digital (TX)"),
    2: PinCapability(2, {PinType.DIGITAL}, description="Digital/Interrupt"),
    3: PinCapability(3, {PinType.DIGITAL, PinType.PWM}, max_value=255, description="Digital/PWM"),
    4: PinCapability(4, {PinType.DIGITAL}, description="Digital"),
    5: PinCapability(5, {PinType.DIGITAL, PinType.PWM}, max_value=255, description="Digital/PWM"),
    6: PinCapability(6, {PinType.DIGITAL, PinType.PWM}, max_value=255, description="Digital/PWM"),
    7: PinCapability(7, {PinType.DIGITAL}, description="Digital"),
    8: PinCapability(8, {PinType.DIGITAL}, description="Digital"),
    9: PinCapability(
        9,
        {PinType.DIGITAL, PinType.PWM, PinType.SERVO},
        max_value=255,
        description="Digital/PWM/Servo",
    ),
    10: PinCapability(
        10,
        {PinType.DIGITAL, PinType.PWM, PinType.SERVO},
        max_value=255,
        description="Digital/PWM/Servo",
    ),
    11: PinCapability(11, {PinType.DIGITAL, PinType.PWM}, max_value=255, description="Digital/PWM"),
    12: PinCapability(12, {PinType.DIGITAL}, description="Digital"),
    13: PinCapability(13, {PinType.DIGITAL}, description="Digital (LED)"),
    # Analog pins (A0-A5)
    14: PinCapability(14, {PinType.DIGITAL, PinType.ANALOG}, max_value=1023, description="A0"),
    15: PinCapability(15, {PinType.DIGITAL, PinType.ANALOG}, max_value=1023, description="A1"),
    16: PinCapability(16, {PinType.DIGITAL, PinType.ANALOG}, max_value=1023, description="A2"),
    17: PinCapability(17, {PinType.DIGITAL, PinType.ANALOG}, max_value=1023, description="A3"),
    18: PinCapability(
        18, {PinType.DIGITAL, PinType.ANALOG, PinType.I2C}, max_value=1023, description="A4 (SDA)"
    ),
    19: PinCapability(
        19, {PinType.DIGITAL, PinType.ANALOG, PinType.I2C}, max_value=1023, description="A5 (SCL)"
    ),
}

# Arduino Mega pin capabilities (simplified)
ARDUINO_MEGA_PINS = {
    **{i: PinCapability(i, {PinType.DIGITAL}) for i in range(54)},
    # PWM pins on Mega: 2-13, 44-46
    **{
        i: PinCapability(i, {PinType.DIGITAL, PinType.PWM}, max_value=255)
        for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 44, 45, 46]
    },
    # Analog pins A0-A15
    **{
        i: PinCapability(i, {PinType.DIGITAL, PinType.ANALOG}, max_value=1023)
        for i in range(54, 70)
    },
}


class TelemetrixBoard(BaseBoard):
    """
    Arduino board implementation using Telemetrix-AIO.

    Telemetrix uses callbacks for data reporting. The Arduino pushes data
    when pins change, which is far more efficient for the event loop.
    """

    # Board type configurations
    BOARD_CONFIGS = {
        "uno": {
            "name": "Arduino Uno",
            "pins": ARDUINO_UNO_PINS,
            "analog_resolution": 10,
            "pwm_resolution": 8,
        },
        "mega": {
            "name": "Arduino Mega",
            "pins": ARDUINO_MEGA_PINS,
            "analog_resolution": 10,
            "pwm_resolution": 8,
        },
    }

    def __init__(
        self,
        port: str | None = None,
        board_type: str = "uno",
        auto_reconnect: bool = False,
    ):
        """
        Initialize the Telemetrix board interface.

        Args:
            port: COM port (e.g., COM3, /dev/ttyUSB0). If None, auto-detect.
            board_type: Type of Arduino board ("uno", "mega")
            auto_reconnect: Whether to auto-reconnect on failure
        """
        super().__init__(port, auto_reconnect)
        self._board_type = board_type
        self._board_config = self.BOARD_CONFIGS.get(board_type, self.BOARD_CONFIGS["uno"])
        self._telemetrix_thread: TelemetrixThread | None = None
        self._pin_modes: dict[int, PinMode] = {}
        self._pin_values: dict[int, Any] = {}
        self._pwm_pins_forced_low: set[int] = set()
        self._pin_values_lock = threading.Lock()  # Thread-safe access to _pin_values
        self._analog_map: dict[int, int] = {}  # Maps actual pin to Arduino analog number
        # Serializes connect()/disconnect(). Their bodies await
        # (asyncio.to_thread), so without this an auto-reconnect connect()
        # can interleave with a manual connect/disconnect and both mutate
        # _telemetrix_thread (stopping a mid-start thread, or clobbering a
        # freshly started one and leaking a live serial connection). Neither
        # method calls the other, so a non-reentrant lock is safe.
        self._connect_lock = asyncio.Lock()
        # Main event loop for marshalling callbacks from telemetrix thread.
        # Initialized to None here; updated in connect() to capture the correct
        # (qasync) event loop that is actually running.
        self._main_loop: asyncio.AbstractEventLoop | None = None

    @property
    def _telemetrix(self):
        """Access the underlying telemetrix instance."""
        if self._telemetrix_thread is None:
            return None
        return self._telemetrix_thread.telemetrix

    @property
    def name(self) -> str:
        return self._board_config["name"]

    @property
    def board_type(self) -> str:
        return "telemetrix"

    @property
    def is_connected(self) -> bool:
        """Check if board is connected AND telemetrix thread is alive."""
        # First check base state
        if self._state != BoardConnectionState.CONNECTED:
            return False

        # Also verify the telemetrix thread is still running
        if self._telemetrix_thread is None:
            return False

        if not self._telemetrix_thread.is_running:
            # Thread died but state wasn't updated - fix it now
            # Only log once by checking if we're transitioning state
            if self._state == BoardConnectionState.CONNECTED:
                logger.warning("Telemetrix thread died unexpectedly - marking as disconnected")
                self._set_state(BoardConnectionState.DISCONNECTED)
                # The cached pin state is stale the moment the link drops
                # (the Arduino resets on the next serial open); clear it so
                # post-reconnect reads don't serve pre-drop values as live.
                self._clear_pin_caches()
                if self._auto_reconnect:
                    # Passive disconnect (cable jostle, serial error): engage
                    # auto-reconnect. start_reconnect() is a no-op if a
                    # reconnect task is already running.
                    try:
                        self.start_reconnect()
                    except RuntimeError:
                        # Property polled from a context with no running event
                        # loop; a later explicit connect() will still work.
                        logger.warning("Auto-reconnect not started: no running event loop")
            return False

        return True

    @property
    def capabilities(self) -> BoardCapabilities:
        return BoardCapabilities(
            name=self.name,
            pins=self._board_config["pins"],
            supports_analog=True,
            analog_resolution=self._board_config["analog_resolution"],
            pwm_resolution=self._board_config["pwm_resolution"],
            pwm_frequency=490,
            i2c_buses=[0],
            spi_buses=[0],
        )

    async def connect(self) -> bool:
        """Establish connection to the Arduino via Telemetrix."""
        # Serialize against concurrent connect()/disconnect(): the body
        # awaits (asyncio.to_thread), so overlapping calls would otherwise
        # both mutate _telemetrix_thread mid-flight.
        async with self._connect_lock:
            # Check if already connected
            if (
                self._state == BoardConnectionState.CONNECTED
                and self._telemetrix_thread is not None
            ):
                logger.debug(f"{self.name} is already connected")
                return True

            try:
                self._set_state(BoardConnectionState.CONNECTING)
                logger.info(f"Connecting to {self.name} on port {self._port or 'auto'}...")

                # Capture the running event loop (qasync) for marshalling callbacks
                # from the telemetrix thread back to the Qt/main thread.
                try:
                    self._main_loop = asyncio.get_running_loop()
                except RuntimeError:
                    self._main_loop = None

                # Import telemetrix here to allow graceful failure if not installed
                try:
                    from telemetrix_aio import telemetrix_aio
                except ImportError:
                    logger.error(
                        "telemetrix-aio not installed. Install with: pip install telemetrix-aio"
                    )
                    self._set_state(BoardConnectionState.ERROR)
                    return False

                # Disconnect existing connection if any
                if self._telemetrix_thread is not None:
                    await asyncio.to_thread(self._telemetrix_thread.stop)
                    self._telemetrix_thread = None

                # Create and start telemetrix in a separate thread
                # This isolates its event loop from Qt's event loop.
                # start() blocks up to 10s on the connect handshake, so it must
                # run in a worker thread to keep the qasync loop responsive.
                self._telemetrix_thread = TelemetrixThread()
                await asyncio.to_thread(
                    self._telemetrix_thread.start, self._port, sleep_tune=0.0001
                )

                self._set_state(BoardConnectionState.CONNECTED)
                logger.info(f"Successfully connected to {self.name}")
                return True

            except Exception as e:
                logger.error(f"Failed to connect to {self.name}: {e}")
                self._set_state(BoardConnectionState.ERROR)
                self._telemetrix_thread = None
                self._notify_error(e)
                if self._auto_reconnect:
                    self.start_reconnect()
                return False

    async def disconnect(self) -> None:
        """Disconnect from the Arduino."""
        # Cancel any pending auto-reconnect BEFORE queueing on the lock so a
        # reconnect task waiting for the lock doesn't reconnect ahead of us.
        self.stop_reconnect()

        async with self._connect_lock:
            # Disable analog reporting before shutdown to prevent KeyError
            # in telemetrix's _analog_message when callbacks are torn down
            if self._telemetrix_thread is not None:
                for _actual_pin, analog_pin in self._analog_map.items():
                    try:
                        self._call_telemetrix("disable_analog_reporting", analog_pin)
                    except Exception:
                        pass

            if self._telemetrix_thread is not None:
                try:
                    # stop() blocks on future.result() + thread.join(); keep it
                    # off the event loop.
                    await asyncio.to_thread(self._telemetrix_thread.stop)
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")
                finally:
                    self._telemetrix_thread = None
            # The Arduino resets when the serial port reopens, so all cached pin
            # state is stale after a disconnect. Clearing it prevents read_digital
            # / read_analog fallbacks from returning pre-disconnect values as live.
            self._clear_pin_caches()
            self._set_state(BoardConnectionState.DISCONNECTED)
            logger.info(f"Disconnected from {self.name}")

    def _clear_pin_caches(self) -> None:
        """Drop all cached pin state (modes, values, PWM flags, analog map).

        Called on any disconnect — explicit or passively detected — because
        the Arduino resets when the serial port reopens, making every cached
        value/mode wrong.
        """
        self._analog_map.clear()
        self._pin_modes.clear()
        self._pwm_pins_forced_low.clear()
        with self._pin_values_lock:
            self._pin_values.clear()

    def _call_telemetrix(self, method_name: str, *args, **kwargs) -> Any:
        """Call a method on telemetrix in its thread."""
        if self._telemetrix_thread is None:
            raise RuntimeError("Board not connected")
        return self._telemetrix_thread.call_method(method_name, *args, **kwargs)

    async def set_pin_mode(
        self, pin: int, mode: PinMode, pin_type: PinType = PinType.DIGITAL
    ) -> None:
        """Configure a pin's mode."""
        if not self.is_connected or self._telemetrix_thread is None:
            raise RuntimeError("Board not connected")

        try:
            if pin_type == PinType.DIGITAL:
                if mode == PinMode.OUTPUT:
                    await asyncio.to_thread(
                        self._call_telemetrix, "set_pin_mode_digital_output", pin
                    )
                elif mode == PinMode.INPUT:
                    await asyncio.to_thread(
                        self._call_telemetrix,
                        "set_pin_mode_digital_input",
                        pin,
                        callback=self._digital_callback,
                    )
                elif mode == PinMode.INPUT_PULLUP:
                    await asyncio.to_thread(
                        self._call_telemetrix,
                        "set_pin_mode_digital_input_pullup",
                        pin,
                        callback=self._digital_callback,
                    )

            elif pin_type == PinType.ANALOG:
                # Convert pin number to analog pin number if needed
                analog_pin = pin - 14 if pin >= 14 else pin
                self._analog_map[pin] = analog_pin
                # Initialize pin value to 0 to prevent None values
                with self._pin_values_lock:
                    self._pin_values[pin] = 0

                try:
                    await asyncio.to_thread(
                        self._call_telemetrix,
                        "set_pin_mode_analog_input",
                        analog_pin,
                        differential=1,
                        callback=self._analog_callback,
                    )
                    logger.info(
                        f"Registered analog callback for pin {pin} (analog pin {analog_pin})"
                    )
                except Exception as e:
                    logger.error(f"Failed to set analog input mode for pin {pin}: {e}")
                    raise

            elif pin_type == PinType.PWM:
                await asyncio.to_thread(self._call_telemetrix, "set_pin_mode_analog_output", pin)

            elif pin_type == PinType.SERVO:
                await asyncio.to_thread(self._call_telemetrix, "set_pin_mode_servo", pin)

            self._pin_modes[pin] = mode
            logger.debug(f"Set pin {pin} to mode {mode} type {pin_type}")

        except Exception as e:
            logger.error(f"Failed to set pin mode: {e}")
            raise

    async def write_digital(self, pin: int, value: bool) -> None:
        """Write a digital value to a pin."""
        if not self.is_connected or self._telemetrix_thread is None:
            raise RuntimeError("Board not connected")

        await asyncio.to_thread(self._call_telemetrix, "digital_write", pin, 1 if value else 0)
        with self._pin_values_lock:
            self._pin_values[pin] = value
        self._notify_output_change(pin, PinType.DIGITAL, bool(value))

    async def read_digital(self, pin: int) -> bool:
        """Read a digital value from a pin."""
        if not self.is_connected or self._telemetrix_thread is None:
            raise RuntimeError("Board not connected")

        # Perform a live read from the hardware
        # telemetrix-aio's digital_read returns a list [pin, value, timestamp]
        try:
            result = await asyncio.to_thread(self._call_telemetrix, "digital_read", pin)
            if result and len(result) > 1:
                value = bool(result[1])
                with self._pin_values_lock:
                    self._pin_values[pin] = value  # Update cache
                return value
        except Exception as e:
            logger.error(f"Error during live digital read on pin {pin}: {e}")
            # Fallback to cached value on error
            with self._pin_values_lock:
                return bool(self._pin_values.get(pin, False))

        # Fallback to cached value if live read fails
        with self._pin_values_lock:
            return bool(self._pin_values.get(pin, False))

    async def write_analog(self, pin: int, value: int) -> None:
        """Write a PWM value to a pin.

        Retries up to 3 times on failure since serial communication can
        be unreliable, and a missed write (especially to 0) can leave
        hardware running unintentionally.
        """
        if not self.is_connected or self._telemetrix_thread is None:
            raise RuntimeError("Board not connected")

        value = max(0, min(255, value))
        last_error = None
        for attempt in range(3):
            try:
                if value == 0:
                    # Force pin fully LOW to overcome back-fed voltage from loads
                    # like ultrasonic atomizers driven through motor controllers.
                    # Step 1: Zero the PWM duty cycle to stop the timer output
                    await asyncio.to_thread(self._call_telemetrix, "analog_write", pin, 0)
                    # Step 2: Switch to digital output mode (disconnects PWM timer)
                    await asyncio.to_thread(
                        self._call_telemetrix, "set_pin_mode_digital_output", pin
                    )
                    # Step 3: Drive LOW with the stronger digital output driver
                    await asyncio.to_thread(self._call_telemetrix, "digital_write", pin, 0)
                    self._pwm_pins_forced_low.add(pin)
                    logger.info(f"PWM pin {pin} forced LOW via digital output mode")
                else:
                    if pin in self._pwm_pins_forced_low:
                        # Re-enable PWM mode before writing a non-zero value
                        await asyncio.to_thread(
                            self._call_telemetrix, "set_pin_mode_analog_output", pin
                        )
                        self._pwm_pins_forced_low.discard(pin)
                    await asyncio.to_thread(self._call_telemetrix, "analog_write", pin, value)
                with self._pin_values_lock:
                    self._pin_values[pin] = value
                self._notify_output_change(pin, PinType.PWM, value)
                return
            except Exception as e:
                last_error = e
                logger.warning(f"PWM write to pin {pin} failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    await asyncio.sleep(0.05)
        raise RuntimeError(
            f"Failed to write PWM value {value} to pin {pin} after 3 attempts: {last_error}"
        )

    async def read_analog(self, pin: int) -> int:
        """Read an analog value from a pin.

        Note: telemetrix-aio is callback-based and has no synchronous analog_read method.
        This returns the cached value that is continuously updated by the callback
        registered in set_pin_mode().
        """
        if not self.is_connected or self._telemetrix_thread is None:
            raise RuntimeError("Board not connected")

        analog_pin = self._analog_map.get(pin)
        if analog_pin is None:
            logger.warning(f"Pin {pin} is not configured as an analog input.")
            return 0

        # Return cached value (continuously updated by _analog_callback)
        with self._pin_values_lock:
            return self._pin_values.get(pin, 0)

    async def write_servo(self, pin: int, angle: int) -> None:
        """Write a servo angle."""
        if not self.is_connected or self._telemetrix_thread is None:
            raise RuntimeError("Board not connected")

        angle = max(0, min(180, angle))
        await asyncio.to_thread(self._call_telemetrix, "servo_write", pin, angle)
        with self._pin_values_lock:
            self._pin_values[pin] = angle
        self._notify_output_change(pin, PinType.SERVO, angle)

    async def _debug_callback_silent(self, data: list) -> None:
        """Silent callback to suppress debug output from telemetrix."""
        pass

    async def _digital_callback(self, data: list) -> None:
        """Callback for digital pin value changes (called from telemetrix thread)."""
        pin = data[1]
        value = bool(data[2])
        with self._pin_values_lock:
            self._pin_values[pin] = value
        # Marshal callback notification to main event loop
        if self._main_loop is not None and not self._main_loop.is_closed():
            self._main_loop.call_soon_threadsafe(self._notify_callbacks, pin, value)
        else:
            logger.warning(
                "Digital callback fired with no main loop available; "
                "notification skipped for pin %s",
                pin,
            )

    async def _analog_callback(self, data: list) -> None:
        """Callback for analog pin value changes (called from telemetrix thread).

        telemetrix-aio callback format: [AT_ANALOG (3), pin, value, timestamp]
        """
        try:
            pin = int(data[1])
            value = int(data[2])
            value = max(0, min(1023, value))

            # Map analog pin back to actual pin number
            for actual_pin, analog_pin in self._analog_map.items():
                if analog_pin == pin:
                    with self._pin_values_lock:
                        self._pin_values[actual_pin] = value
                    # Marshal callback notification to main event loop
                    if self._main_loop is not None and not self._main_loop.is_closed():
                        self._main_loop.call_soon_threadsafe(
                            self._notify_callbacks, actual_pin, value
                        )
                    else:
                        logger.warning(
                            "Analog callback fired with no main loop available; "
                            "notification skipped for pin %s",
                            actual_pin,
                        )
                    break
        except Exception as e:
            logger.error(f"Error in analog callback: {e}, data={data}", exc_info=True)

    async def emergency_stop(self) -> None:
        """Set all outputs to safe state."""
        if not self.is_connected or self._telemetrix_thread is None:
            return

        # Turn off all configured output pins, continuing even if individual pins fail
        for pin, mode in self._pin_modes.items():
            if mode == PinMode.OUTPUT:
                try:
                    cap = self._board_config["pins"].get(pin)
                    if cap and PinType.PWM in cap.supported_types:
                        # Zero PWM then force digital LOW
                        await asyncio.to_thread(self._call_telemetrix, "analog_write", pin, 0)
                        await asyncio.to_thread(
                            self._call_telemetrix, "set_pin_mode_digital_output", pin
                        )
                        await asyncio.to_thread(self._call_telemetrix, "digital_write", pin, 0)
                        self._pwm_pins_forced_low.add(pin)
                    else:
                        await asyncio.to_thread(self._call_telemetrix, "digital_write", pin, 0)
                except Exception as e:
                    logger.error(f"Error during emergency stop on pin {pin}: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize board configuration to dictionary."""
        data = super().to_dict()
        data["board_type"] = self._board_type
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TelemetrixBoard":
        """Create board instance from dictionary configuration."""
        instance = cls(
            port=data.get("port"),
            board_type=data.get("board_type", "uno"),
            auto_reconnect=data.get("auto_reconnect", False),
        )
        instance._id = data.get("id", instance._id)
        return instance
