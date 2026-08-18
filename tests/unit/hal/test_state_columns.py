"""Devices may declare multiple CSV sub-columns."""

from collections.abc import Callable
from typing import Any

from glider.hal.base_board import BaseBoard
from glider.hal.base_device import BaseDevice, DeviceConfig, DigitalOutputDevice
from glider.hal.mock_board import MockBoard


class MultiDevice(BaseDevice):
    """A device that contributes two sub-columns instead of one."""

    @property
    def device_type(self) -> str:
        return "multi"

    @property
    def required_pins(self) -> list[str]:
        return []

    @property
    def actions(self) -> dict[str, Callable]:
        return {}

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: BaseBoard) -> "MultiDevice":
        return cls(board, DeviceConfig(**data["config"]), data.get("name"))

    def state_columns(self) -> list[str]:
        return ["alpha", "beta"]


def test_default_state_columns_is_none():
    """Existing devices keep single-column behaviour."""
    device = DigitalOutputDevice(MockBoard(), DeviceConfig(pins={"output": 5}), name="relay")
    assert device.state_columns() is None


def test_device_may_declare_sub_columns():
    """A device that overrides state_columns reports its names."""
    device = MultiDevice(MockBoard(), DeviceConfig())
    assert device.state_columns() == ["alpha", "beta"]
