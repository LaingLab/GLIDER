"""
Mock Board - Simulated board for testing without hardware.

Provides a virtual board that logs operations but doesn't require
actual hardware to be connected.
"""

import logging
from typing import Any

from glider.hal.base_board import (
    BaseBoard,
    BoardCapabilities,
    BoardConnectionState,
    PinCapability,
    PinMode,
    PinType,
)

logger = logging.getLogger(__name__)


class MockBoard(BaseBoard):
    """
    Mock board implementation for testing.

    Simulates hardware operations by logging them and maintaining
    virtual pin states in memory.
    """

    def __init__(self, port: str = "MOCK", auto_reconnect: bool = False):
        super().__init__(port, auto_reconnect)
        self._pin_states: dict[int, Any] = {}
        self._pin_modes: dict[int, PinMode] = {}
        self._set_state(BoardConnectionState.CONNECTED)

    @property
    def board_type(self) -> str:
        return "mock"

    @property
    def name(self) -> str:
        return "Mock Board"

    @property
    def capabilities(self) -> BoardCapabilities:
        # Mock board supports all common pins
        pins = {}
        for i in range(54):  # Arduino Mega-like pin count
            pins[i] = PinCapability(
                pin=i,
                supported_types={PinType.DIGITAL, PinType.ANALOG, PinType.PWM},
                max_value=255,
                description=f"Mock Pin {i}",
            )
        return BoardCapabilities(
            name="Mock Board",
            pins=pins,
            supports_analog=True,
            analog_resolution=10,
            pwm_resolution=8,
        )

    async def connect(self) -> bool:
        logger.info("MockBoard: Connected (simulated)")
        self._set_state(BoardConnectionState.CONNECTED)
        return True

    async def disconnect(self) -> None:
        logger.info("MockBoard: Disconnected (simulated)")
        self._set_state(BoardConnectionState.DISCONNECTED)

    async def set_pin_mode(
        self, pin: int, mode: PinMode, pin_type: PinType = PinType.DIGITAL
    ) -> None:
        logger.info(f"MockBoard: Set pin {pin} mode to {mode.name} ({pin_type.name})")
        self._pin_modes[pin] = mode

    async def write_digital(self, pin: int, value: bool) -> None:
        logger.info(f"MockBoard: Write digital pin {pin} = {'HIGH' if value else 'LOW'}")
        self._pin_states[pin] = value
        self._notify_callbacks(pin, value)

    async def read_digital(self, pin: int) -> bool:
        value = self._pin_states.get(pin, False)
        logger.info(f"MockBoard: Read digital pin {pin} = {'HIGH' if value else 'LOW'}")
        return value

    async def write_analog(self, pin: int, value: int) -> None:
        logger.info(f"MockBoard: Write analog pin {pin} = {value}")
        self._pin_states[pin] = value
        self._notify_callbacks(pin, value)

    async def read_analog(self, pin: int) -> int:
        value = self._pin_states.get(pin, 0)
        logger.info(f"MockBoard: Read analog pin {pin} = {value}")
        return value

    def get_pin_state(self, pin: int) -> Any:
        """Get the current state of a pin."""
        return self._pin_states.get(pin)
