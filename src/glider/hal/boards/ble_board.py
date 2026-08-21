"""
BLE (Bluetooth Low Energy) board driver.

Represents the host machine's Bluetooth adapter as a GLIDER "board". Unlike
pin-based boards (Arduino, Pi GPIO), a BLE adapter has no GPIO -- it is a
transport over which one or more BLE peripheral *devices* (e.g. BLEWriteDevice)
each open their own connection. This board therefore:

- verifies the ``bleak`` stack is importable on connect()
- offers ``scan()`` to discover nearby peripherals (used by the hardware
  panel's "Scan" button)
- raises a clear error for any pin operation, since BLE has no pins

Cross-platform via bleak: Windows (WinRT), macOS (CoreBluetooth), Linux (BlueZ).
"""

import logging
from dataclasses import dataclass

from glider.hal.base_board import (
    BaseBoard,
    BoardCapabilities,
    BoardConnectionState,
    PinMode,
    PinType,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveredPeripheral:
    """One peripheral seen by a scan.

    ``name`` is empty when the peripheral advertised none, which is common and
    not a defect: a Zephyr device puts its name in the scan response, and
    Windows drops that often enough that a working device routinely appears
    nameless.
    """

    address: str
    name: str = ""
    rssi: int | None = None
    service_uuids: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """What to show a human picking one out of a list."""
        base = self.name or self.address
        return f"{base} ({self.rssi} dBm)" if self.rssi is not None else base

    def advertises(self, service_uuid: str) -> bool:
        return str(service_uuid).strip().lower() in self.service_uuids


class BLEBoard(BaseBoard):
    """Host Bluetooth LE adapter. Peripherals connect per-device via bleak."""

    def __init__(self, port: str | None = None, auto_reconnect: bool = False):
        # `port` is unused for BLE (kept for the BaseBoard / driver-factory
        # signature); the adapter is implicit.
        super().__init__(port, auto_reconnect)

    @property
    def board_type(self) -> str:
        return "bluetooth"

    @property
    def name(self) -> str:
        return "Bluetooth LE"

    @property
    def capabilities(self) -> BoardCapabilities:
        # No GPIO pins on a BLE transport.
        return BoardCapabilities(name="Bluetooth LE", pins={})

    async def connect(self) -> bool:
        """Verify the bleak stack is importable and mark the adapter ready."""
        try:
            import bleak  # noqa: F401
        except ImportError as e:
            self._set_state(BoardConnectionState.ERROR)
            raise RuntimeError(
                "bleak not installed. Run: pip install bleak (or reinstall GLIDER)."
            ) from e
        self._set_state(BoardConnectionState.CONNECTED)
        logger.info("BLEBoard: adapter ready")
        return True

    async def disconnect(self) -> None:
        # Peripheral connections live on the individual devices; nothing to do
        # at the adapter level beyond updating state.
        self._set_state(BoardConnectionState.DISCONNECTED)
        logger.info("BLEBoard: adapter released")

    @staticmethod
    async def scan(timeout: float = 8.0) -> list[DiscoveredPeripheral]:
        """Discover nearby BLE peripherals.

        Reads the name from the advertisement data (``local_name``) rather than
        ``device.name``: many peripherals (e.g. Zephyr devices) send their name
        in the SCAN RESPONSE, which an active scan captures into ``local_name``
        even when ``device.name`` comes back empty on Windows. Static so callers
        can scan the host adapter without needing a board instance.

        Keeps the signal strength and the advertised service UUIDs, because a
        peripheral whose name did not come through is otherwise a bare MAC in a
        list of bare MACs. The services say *what* it is and the RSSI says which
        one is on the bench in front of you -- and a device that knows its own
        service UUID can be matched without a name at all.

        Sorted strongest-first, so the peripheral you are holding is near the
        top rather than wherever the adapter happened to enumerate it.
        """
        from bleak import BleakScanner

        # return_adv=True -> {address: (BLEDevice, AdvertisementData)}
        discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)
        results = []
        for dev, adv in discovered.values():
            name = (getattr(adv, "local_name", None) or getattr(dev, "name", None) or "").strip()
            results.append(
                DiscoveredPeripheral(
                    name=name,
                    address=dev.address,
                    rssi=getattr(adv, "rssi", None),
                    service_uuids=tuple(
                        str(u).lower() for u in (getattr(adv, "service_uuids", None) or ())
                    ),
                )
            )
        results.sort(key=lambda p: (p.rssi if p.rssi is not None else -999), reverse=True)
        logger.info(
            "BLEBoard: scan found %d peripheral(s): %s",
            len(results),
            ", ".join(p.label for p in results) or "none",
        )
        return results

    # --- pin operations are not applicable to BLE ---

    @staticmethod
    def _no_pins(op: str) -> None:
        raise NotImplementedError(
            f"BLEBoard has no GPIO pins ({op} is not supported). "
            "Use a BLE device (e.g. BLEWrite) to talk to a peripheral."
        )

    async def set_pin_mode(
        self, pin: int, mode: PinMode, pin_type: PinType = PinType.DIGITAL
    ) -> None:
        self._no_pins("set_pin_mode")

    async def write_digital(self, pin: int, value: bool) -> None:
        self._no_pins("write_digital")

    async def read_digital(self, pin: int) -> bool:
        self._no_pins("read_digital")

    async def write_analog(self, pin: int, value: int) -> None:
        self._no_pins("write_analog")

    async def read_analog(self, pin: int) -> int:
        self._no_pins("read_analog")
