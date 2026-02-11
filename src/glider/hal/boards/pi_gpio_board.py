"""
Raspberry Pi GPIO board implementation.

Supports two modes of operation:
- **Remote** (port is a hostname/IP): Uses the pigpio library to connect
  to the pigpiod daemon running on the remote Pi. This allows controlling
  GPIO pins from any machine (e.g., Windows, Mac, Linux).
- **Local** (port is None): Uses gpiozero or lgpio for direct GPIO control
  when running directly on the Pi.

All blocking calls are wrapped with asyncio.to_thread() to ensure
non-blocking operation.
"""

import asyncio
import logging
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


# Raspberry Pi GPIO pin capabilities (BCM numbering)
# Pins 0-1 are reserved for I2C EEPROM
# Pins 2-3 are I2C (SDA/SCL)
# Pins 14-15 are UART (TXD/RXD)
RPI_GPIO_PINS = {
    2: PinCapability(2, {PinType.DIGITAL, PinType.I2C}, description="GPIO2 (SDA)"),
    3: PinCapability(3, {PinType.DIGITAL, PinType.I2C}, description="GPIO3 (SCL)"),
    4: PinCapability(4, {PinType.DIGITAL}, description="GPIO4"),
    5: PinCapability(5, {PinType.DIGITAL}, description="GPIO5"),
    6: PinCapability(6, {PinType.DIGITAL}, description="GPIO6"),
    7: PinCapability(7, {PinType.DIGITAL, PinType.SPI}, description="GPIO7 (CE1)"),
    8: PinCapability(8, {PinType.DIGITAL, PinType.SPI}, description="GPIO8 (CE0)"),
    9: PinCapability(9, {PinType.DIGITAL, PinType.SPI}, description="GPIO9 (MISO)"),
    10: PinCapability(10, {PinType.DIGITAL, PinType.SPI}, description="GPIO10 (MOSI)"),
    11: PinCapability(11, {PinType.DIGITAL, PinType.SPI}, description="GPIO11 (SCLK)"),
    12: PinCapability(
        12, {PinType.DIGITAL, PinType.PWM}, max_value=100, description="GPIO12 (PWM0)"
    ),
    13: PinCapability(
        13, {PinType.DIGITAL, PinType.PWM}, max_value=100, description="GPIO13 (PWM1)"
    ),
    14: PinCapability(14, {PinType.DIGITAL}, description="GPIO14 (TXD)"),
    15: PinCapability(15, {PinType.DIGITAL}, description="GPIO15 (RXD)"),
    16: PinCapability(16, {PinType.DIGITAL}, description="GPIO16"),
    17: PinCapability(17, {PinType.DIGITAL}, description="GPIO17"),
    18: PinCapability(
        18, {PinType.DIGITAL, PinType.PWM}, max_value=100, description="GPIO18 (PWM0)"
    ),
    19: PinCapability(
        19, {PinType.DIGITAL, PinType.PWM}, max_value=100, description="GPIO19 (PWM1)"
    ),
    20: PinCapability(20, {PinType.DIGITAL}, description="GPIO20"),
    21: PinCapability(21, {PinType.DIGITAL}, description="GPIO21"),
    22: PinCapability(22, {PinType.DIGITAL}, description="GPIO22"),
    23: PinCapability(23, {PinType.DIGITAL}, description="GPIO23"),
    24: PinCapability(24, {PinType.DIGITAL}, description="GPIO24"),
    25: PinCapability(25, {PinType.DIGITAL}, description="GPIO25"),
    26: PinCapability(26, {PinType.DIGITAL}, description="GPIO26"),
    27: PinCapability(27, {PinType.DIGITAL}, description="GPIO27"),
}


class PiGPIOBoard(BaseBoard):
    """
    Raspberry Pi GPIO board implementation.

    Supports remote operation via the pigpio daemon (pigpiod) when a
    hostname/IP is provided as port, or local operation via gpiozero
    when running directly on the Pi.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        auto_reconnect: bool = True,
    ):
        """
        Initialize the Raspberry Pi GPIO board.

        Args:
            port: Hostname or IP address of the Raspberry Pi for remote
                  connection (e.g., "192.168.1.100", "raspberrypi.local").
                  Set to None for local GPIO access when running on the Pi.
            auto_reconnect: Whether to auto-reconnect on failure
        """
        super().__init__(port, auto_reconnect)
        self._gpiozero_available = False
        self._lgpio_available = False
        self._pigpio_available = False
        self._pigpio_pi: Any = None  # pigpio.pi instance for remote connections
        self._devices: dict[int, Any] = {}  # gpiozero device instances
        self._pin_modes: dict[int, PinMode] = {}
        self._pin_types: dict[int, PinType] = {}
        self._pin_values: dict[int, Any] = {}
        self._pigpio_callbacks: dict[int, Any] = {}  # pigpio callback handles
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def name(self) -> str:
        return "Raspberry Pi GPIO"

    @property
    def board_type(self) -> str:
        return "pigpio"

    @property
    def capabilities(self) -> BoardCapabilities:
        return BoardCapabilities(
            name=self.name,
            pins=RPI_GPIO_PINS,
            supports_analog=False,  # Pi doesn't have built-in ADC
            analog_resolution=0,
            pwm_resolution=8,  # Software PWM
            pwm_frequency=100,
            i2c_buses=[1],  # Default I2C bus on Pi
            spi_buses=[0, 1],
        )

    async def connect(self) -> bool:
        """Initialize GPIO access.

        If a port (hostname/IP) was provided, connects to the remote Pi's
        pigpiod daemon. Otherwise, uses local gpiozero/lgpio.
        """
        try:
            self._set_state(BoardConnectionState.CONNECTING)

            # Store event loop for thread-safe callbacks
            self._event_loop = asyncio.get_running_loop()

            if self._port:
                # Remote connection via pigpio daemon
                logger.info(f"Connecting to Raspberry Pi at {self._port}...")
                try:
                    import pigpio

                    self._pigpio_pi = await asyncio.to_thread(pigpio.pi, self._port)
                    if not self._pigpio_pi.connected:
                        logger.error(
                            f"Failed to connect to pigpiod on {self._port}. "
                            "Ensure pigpiod is running on the Pi (sudo pigpiod)."
                        )
                        self._pigpio_pi = None
                        self._set_state(BoardConnectionState.ERROR)
                        return False
                    self._pigpio_available = True
                    logger.info(f"Connected to pigpiod on {self._port}")
                except ImportError:
                    logger.error(
                        "pigpio library not available. "
                        "Install with: pip install pigpio"
                    )
                    self._set_state(BoardConnectionState.ERROR)
                    return False
            else:
                # Local GPIO access
                logger.info("Initializing local Raspberry Pi GPIO...")
                try:
                    import gpiozero

                    self._gpiozero_available = True
                    logger.info("Using gpiozero for GPIO control")
                except ImportError:
                    logger.warning("gpiozero not available")

                # Try lgpio as fallback
                if not self._gpiozero_available:
                    try:
                        import lgpio

                        self._lgpio_available = True
                        logger.info("Using lgpio for GPIO control")
                    except ImportError:
                        logger.warning("lgpio not available")

                if not self._gpiozero_available and not self._lgpio_available:
                    logger.error(
                        "No GPIO library available. Install gpiozero or lgpio, "
                        "or provide a hostname to connect remotely via pigpio."
                    )
                    self._set_state(BoardConnectionState.ERROR)
                    return False

            self._set_state(BoardConnectionState.CONNECTED)
            logger.info("Raspberry Pi GPIO initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize GPIO: {e}")
            self._set_state(BoardConnectionState.ERROR)
            self._notify_error(e)
            return False

    async def disconnect(self) -> None:
        """Clean up GPIO resources."""
        self.stop_reconnect()

        if self._pigpio_available and self._pigpio_pi:
            # Cancel pigpio callbacks
            for pin, cb in self._pigpio_callbacks.items():
                try:
                    cb.cancel()
                except Exception as e:
                    logger.warning(f"Error cancelling callback on pin {pin}: {e}")
            self._pigpio_callbacks.clear()

            # Stop all PWM and reset pins
            for pin in list(self._pin_modes.keys()):
                try:
                    pin_type = self._pin_types.get(pin)
                    if pin_type == PinType.PWM:
                        await asyncio.to_thread(self._pigpio_pi.set_PWM_dutycycle, pin, 0)
                    elif pin_type == PinType.SERVO:
                        await asyncio.to_thread(self._pigpio_pi.set_servo_pulsewidth, pin, 0)
                    else:
                        await asyncio.to_thread(self._pigpio_pi.write, pin, 0)
                except Exception as e:
                    logger.warning(f"Error resetting pin {pin}: {e}")

            # Disconnect from pigpiod
            try:
                await asyncio.to_thread(self._pigpio_pi.stop)
            except Exception as e:
                logger.warning(f"Error stopping pigpio connection: {e}")
            self._pigpio_pi = None
            self._pigpio_available = False
        else:
            # Close all gpiozero devices
            for pin, device in self._devices.items():
                try:
                    if hasattr(device, "close"):
                        await asyncio.to_thread(device.close)
                except Exception as e:
                    logger.warning(f"Error closing device on pin {pin}: {e}")

        self._devices.clear()
        self._pin_modes.clear()
        self._pin_types.clear()
        self._pin_values.clear()
        self._set_state(BoardConnectionState.DISCONNECTED)
        logger.info("Raspberry Pi GPIO disconnected")

    async def set_pin_mode(
        self, pin: int, mode: PinMode, pin_type: PinType = PinType.DIGITAL
    ) -> None:
        """Configure a pin's mode."""
        if not self.is_connected:
            raise RuntimeError("Board not connected")

        try:
            if self._pigpio_available and self._pigpio_pi:
                await self._set_pin_mode_pigpio(pin, mode, pin_type)
            elif self._gpiozero_available:
                await self._set_pin_mode_gpiozero(pin, mode, pin_type)
            else:
                raise RuntimeError("No GPIO library available")

            self._pin_modes[pin] = mode
            self._pin_types[pin] = pin_type
            logger.debug(f"Set GPIO pin {pin} to mode {mode} type {pin_type}")

        except Exception as e:
            logger.error(f"Failed to set pin mode: {e}")
            raise

    async def _set_pin_mode_pigpio(
        self, pin: int, mode: PinMode, pin_type: PinType
    ) -> None:
        """Configure pin mode using pigpio (remote)."""
        import pigpio

        pi = self._pigpio_pi

        if pin_type == PinType.DIGITAL:
            if mode == PinMode.OUTPUT:
                await asyncio.to_thread(pi.set_mode, pin, pigpio.OUTPUT)
            elif mode == PinMode.INPUT:
                await asyncio.to_thread(pi.set_mode, pin, pigpio.INPUT)
                await asyncio.to_thread(pi.set_pull_up_down, pin, pigpio.PUD_OFF)
                self._setup_pigpio_input_callback(pin)
            elif mode == PinMode.INPUT_PULLUP:
                await asyncio.to_thread(pi.set_mode, pin, pigpio.INPUT)
                await asyncio.to_thread(pi.set_pull_up_down, pin, pigpio.PUD_UP)
                self._setup_pigpio_input_callback(pin)
            elif mode == PinMode.INPUT_PULLDOWN:
                await asyncio.to_thread(pi.set_mode, pin, pigpio.INPUT)
                await asyncio.to_thread(pi.set_pull_up_down, pin, pigpio.PUD_DOWN)
                self._setup_pigpio_input_callback(pin)

        elif pin_type == PinType.PWM:
            # pigpio handles PWM configuration automatically on first write
            await asyncio.to_thread(pi.set_mode, pin, pigpio.OUTPUT)

        elif pin_type == PinType.SERVO:
            # pigpio handles servo configuration automatically on first write
            await asyncio.to_thread(pi.set_mode, pin, pigpio.OUTPUT)

    def _setup_pigpio_input_callback(self, pin: int) -> None:
        """Set up a pigpio callback for input pin changes."""
        import pigpio

        # Cancel existing callback if any
        if pin in self._pigpio_callbacks:
            self._pigpio_callbacks[pin].cancel()

        def on_change(gpio, level, tick):
            self._pin_values[pin] = bool(level)
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(
                    lambda: self._notify_callbacks(pin, bool(level))
                )

        cb = self._pigpio_pi.callback(pin, pigpio.EITHER_EDGE, on_change)
        self._pigpio_callbacks[pin] = cb

    async def _set_pin_mode_gpiozero(
        self, pin: int, mode: PinMode, pin_type: PinType
    ) -> None:
        """Configure pin mode using gpiozero (local)."""
        import gpiozero

        # Close existing device if any
        if pin in self._devices:
            await asyncio.to_thread(self._devices[pin].close)

        if pin_type == PinType.DIGITAL:
            if mode == PinMode.OUTPUT:
                device = await asyncio.to_thread(lambda: gpiozero.DigitalOutputDevice(pin))
                self._devices[pin] = device
            elif mode == PinMode.INPUT:
                device = await asyncio.to_thread(
                    lambda: gpiozero.DigitalInputDevice(pin, pull_up=False)
                )
                self._devices[pin] = device
                self._setup_input_callback(pin, device)
            elif mode == PinMode.INPUT_PULLUP:
                device = await asyncio.to_thread(
                    lambda: gpiozero.DigitalInputDevice(pin, pull_up=True)
                )
                self._devices[pin] = device
                self._setup_input_callback(pin, device)
            elif mode == PinMode.INPUT_PULLDOWN:
                device = await asyncio.to_thread(
                    lambda: gpiozero.DigitalInputDevice(pin, pull_up=False)
                )
                self._devices[pin] = device
                self._setup_input_callback(pin, device)

        elif pin_type == PinType.PWM:
            device = await asyncio.to_thread(lambda: gpiozero.PWMOutputDevice(pin))
            self._devices[pin] = device

        elif pin_type == PinType.SERVO:
            device = await asyncio.to_thread(lambda: gpiozero.Servo(pin))
            self._devices[pin] = device

    def _setup_input_callback(self, pin: int, device: Any) -> None:
        """Set up callbacks for input devices."""

        def on_change():
            value = device.value
            self._pin_values[pin] = value
            # Use call_soon_threadsafe to marshal back to main event loop
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(lambda: self._notify_callbacks(pin, value))

        device.when_activated = on_change
        device.when_deactivated = on_change

    async def write_digital(self, pin: int, value: bool) -> None:
        """Write a digital value to a pin."""
        if not self.is_connected:
            raise RuntimeError("Board not connected")

        if self._pigpio_available and self._pigpio_pi:
            await asyncio.to_thread(self._pigpio_pi.write, pin, int(value))
        else:
            device = self._devices.get(pin)
            if device is None:
                raise ValueError(f"Pin {pin} not configured")
            if value:
                await asyncio.to_thread(device.on)
            else:
                await asyncio.to_thread(device.off)

        self._pin_values[pin] = value

    async def read_digital(self, pin: int) -> bool:
        """Read a digital value from a pin."""
        if not self.is_connected:
            raise RuntimeError("Board not connected")

        if self._pigpio_available and self._pigpio_pi:
            value = await asyncio.to_thread(self._pigpio_pi.read, pin)
            value = bool(value)
        else:
            device = self._devices.get(pin)
            if device is None:
                raise ValueError(f"Pin {pin} not configured")
            value = bool(await asyncio.to_thread(lambda: device.value))

        self._pin_values[pin] = value
        return value

    async def write_analog(self, pin: int, value: int) -> None:
        """Write a PWM value to a pin (0-255)."""
        if not self.is_connected:
            raise RuntimeError("Board not connected")

        value = max(0, min(255, value))

        if self._pigpio_available and self._pigpio_pi:
            # pigpio uses 0-255 natively
            await asyncio.to_thread(self._pigpio_pi.set_PWM_dutycycle, pin, value)
        else:
            device = self._devices.get(pin)
            if device is None:
                raise ValueError(f"Pin {pin} not configured")
            # Convert 0-255 to 0-1 for gpiozero
            pwm_value = value / 255.0
            await asyncio.to_thread(lambda: setattr(device, "value", pwm_value))

        self._pin_values[pin] = value

    async def read_analog(self, pin: int) -> int:
        """Read analog value - not supported on Pi without external ADC."""
        raise NotImplementedError(
            "Raspberry Pi does not have built-in ADC. Use an external ADC like MCP3008."
        )

    async def write_servo(self, pin: int, angle: int) -> None:
        """Write a servo angle (0-180)."""
        if not self.is_connected:
            raise RuntimeError("Board not connected")

        angle = max(0, min(180, angle))

        if self._pigpio_available and self._pigpio_pi:
            # Map 0-180 degrees to 500-2500 microsecond pulse width
            pulsewidth = 500 + int((angle / 180.0) * 2000)
            await asyncio.to_thread(
                self._pigpio_pi.set_servo_pulsewidth, pin, pulsewidth
            )
        else:
            device = self._devices.get(pin)
            if device is None:
                raise ValueError(f"Pin {pin} not configured as servo")
            # Map 0-180 to -1 to 1 for gpiozero Servo
            servo_value = (angle / 90.0) - 1.0
            servo_value = max(-1.0, min(1.0, servo_value))
            await asyncio.to_thread(lambda: setattr(device, "value", servo_value))

        self._pin_values[pin] = angle

    async def emergency_stop(self) -> None:
        """Set all outputs to safe state."""
        if self._pigpio_available and self._pigpio_pi:
            for pin in list(self._pin_modes.keys()):
                try:
                    pin_type = self._pin_types.get(pin)
                    if pin_type == PinType.PWM:
                        await asyncio.to_thread(
                            self._pigpio_pi.set_PWM_dutycycle, pin, 0
                        )
                    elif pin_type == PinType.SERVO:
                        await asyncio.to_thread(
                            self._pigpio_pi.set_servo_pulsewidth, pin, 0
                        )
                    elif self._pin_modes.get(pin) == PinMode.OUTPUT:
                        await asyncio.to_thread(self._pigpio_pi.write, pin, 0)
                except Exception as e:
                    logger.error(f"Error during emergency stop on pin {pin}: {e}")
        else:
            for pin, device in self._devices.items():
                try:
                    if hasattr(device, "off"):
                        await asyncio.to_thread(device.off)
                    elif hasattr(device, "value"):
                        await asyncio.to_thread(lambda d=device: setattr(d, "value", 0))
                except Exception as e:
                    logger.error(f"Error during emergency stop on pin {pin}: {e}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize board configuration to dictionary."""
        return {
            "id": self._id,
            "name": self.name,
            "port": self._port,
            "auto_reconnect": self._auto_reconnect,
            "board_type": "raspberry_pi",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PiGPIOBoard":
        """Create board instance from dictionary configuration."""
        instance = cls(
            port=data.get("port"),
            auto_reconnect=data.get("auto_reconnect", True),
        )
        instance._id = data.get("id", instance._id)
        return instance
