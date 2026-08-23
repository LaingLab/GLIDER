"""Full-featured generic BLE peripheral device.

Talks to any Bluetooth LE peripheral by address (or advertised name) and one or
more GATT characteristic UUIDs, without a dedicated class per gadget. Extends
what ``BLEWriteDevice`` does (write one characteristic) with the two halves it
lacks:

- **read**: read a value characteristic on demand (``read`` action).
- **notify/subscribe**: most BLE sensors *push* data via GATT notifications;
  when ``notify`` is enabled this device subscribes on connect and caches the
  latest decoded value, so ``read``/``get_state`` never block and the
  DataRecorder can log a free-running BLE sensor at its own interval (same
  cached-``get_state`` pattern as HX711Device / GenericSerialDevice).

Cross-platform via bleak (Windows WinRT, macOS CoreBluetooth, Linux BlueZ).

**Address portability (Mac ↔ Windows).** CoreBluetooth (macOS) hides the real
MAC and hands out a *per-host* UUID, while Windows/Linux expose the MAC. A
``.glider`` file with a hardcoded address is therefore NOT portable across
machines. To survive that, set ``name`` (the advertised local name): when
``address`` is blank the device resolves the current host's address by scanning
for that name at connect time, so the same file works on every OS.

``BLEWriteDevice`` (``device_type == "BLEWrite"``) is retained unchanged for
existing files; this device (``"BLE"``) is the superset for new work.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import BaseDevice, DeviceConfig

if TYPE_CHECKING:
    from glider.hal.base_board import BaseBoard

logger = logging.getLogger(__name__)

# A cached notification older than this is not a reading (mirrors HX711/serial).
MAX_SAMPLE_AGE_S = 5.0
# How long a ``read`` on a notify device waits for the first push before failing.
READ_WAIT_S = 2.0
# Discovery scan timeout when resolving an address from an advertised name.
RESOLVE_SCAN_S = 8.0

_VALUE_FORMATS = ("text", "int", "int_be", "hex", "bytes")


class BLEDevice(BaseDevice):
    """A BLE peripheral: write / read / subscribe across named characteristics.

    Settings:
    - address: peripheral address (MAC on Windows/Linux, UUID on macOS). Leave
        blank and set ``name`` for cross-host portability.
    - name: advertised local name; used to resolve ``address`` when it is blank.
    - service_uuid: optional service UUID (reference / future filtering).
    - write_char_uuid: characteristic for the ``write`` action.
    - read_char_uuid: characteristic for the ``read`` action (and notify).
    - notify: subscribe to ``read_char_uuid`` notifications on connect (default
        False). When True, ``read``/``get_state`` return the latest push.
    - value_format: how to decode read/notify bytes -- one of "text" (default),
        "int" (unsigned LE), "int_be" (unsigned BE), "hex", "bytes".
    - encoding: text codec for write commands and "text" decoding (default utf-8).
    - write_response: acknowledged write when the characteristic supports both
        modes (default False).
    """

    SETTINGS_SCHEMA = [
        {
            "key": "address",
            "label": "Address / UUID",
            "type": "str",
            "default": "",
            "help": "MAC (Windows/Linux) or UUID (macOS). Leave blank and set Name for portability.",
        },
        {
            "key": "name",
            "label": "Advertised name",
            "type": "str",
            "default": "",
            "help": "Resolve the address by scanning for this name when Address is blank.",
        },
        {
            "key": "read_char_uuid",
            "label": "Read/Notify characteristic",
            "type": "str",
            "default": "",
        },
        {"key": "write_char_uuid", "label": "Write characteristic", "type": "str", "default": ""},
        {
            "key": "notify",
            "label": "Subscribe (stream notifications)",
            "type": "bool",
            "default": False,
            "help": "Push-based sensors: subscribe on connect and record each notification.",
        },
        {
            "key": "value_format",
            "label": "Value format",
            "type": "enum",
            "default": "text",
            "choices": [
                ["text", "Text"],
                ["int", "Integer (LE)"],
                ["int_be", "Integer (BE)"],
                ["hex", "Hex string"],
                ["bytes", "Raw bytes"],
            ],
        },
        {"key": "service_uuid", "label": "Service UUID (optional)", "type": "str", "default": ""},
        {"key": "encoding", "label": "Encoding", "type": "str", "default": "utf-8"},
        {"key": "write_response", "label": "Acknowledged write", "type": "bool", "default": False},
    ]

    def __init__(self, board: "BaseBoard", config: DeviceConfig, name: str | None = None):
        super().__init__(board, config, name)
        self._resolved_address: str | None = None
        self._apply_setting_caches(config.settings)  # validates; sets the caches

        self._client = None  # BleakClient, created in initialize()
        self._lock = asyncio.Lock()  # serialize connect/write/read
        # Latest notification as (decoded_value, perf_counter_ts). bleak fires
        # the notify handler on the event loop, so a plain attribute is safe --
        # no thread lock needed (unlike the serial/HX711 sampler threads).
        self._latest: tuple[Any, float] | None = None
        # The tracked link. Unlike the BaseDevice default this is stored, not
        # derived: the board here is the host adapter, which is "connected"
        # from the moment bleak imports and knows nothing about whether this
        # peripheral is actually answering.
        self._link = ConnectionState.DISCONNECTED

    def _apply_setting_caches(self, s: dict[str, Any]) -> None:
        """Set the settings-derived cache attributes (validates value_format first,
        before mutating anything, so a rejected edit leaves the caches untouched)."""
        value_format = str(s.get("value_format", "text"))
        if value_format not in _VALUE_FORMATS:
            raise ValueError(f"value_format must be one of {_VALUE_FORMATS}, got {value_format!r}")
        self._address = str(s.get("address", "")).strip()
        self._adv_name = str(s.get("name", "")).strip()
        self._service_uuid = str(s.get("service_uuid", "")).strip() or None
        self._write_char = str(s.get("write_char_uuid", "")).strip()
        self._read_char = str(s.get("read_char_uuid", "")).strip()
        self._notify = bool(s.get("notify", False))
        self._value_format = value_format
        self._encoding = s.get("encoding", "utf-8")
        self._write_response = bool(s.get("write_response", False))
        self._resolved_address = None  # re-resolve the address on next connect

    def apply_settings(self, settings: dict[str, Any]) -> None:
        """Adopt edited settings (validated first).

        BLE settings are connection/subscription/decode parameters; re-resolving
        the address or re-subscribing can't be done safely under a live link, so
        while the device is initialized the edit is only recorded to
        ``config.settings`` (for the saved file) and takes effect on the next
        initialize(). Before init, the caches are refreshed immediately. This is
        the drift guard ``BLEWriteDevice`` and the other transport devices carry.
        """
        merged = {**self._config.settings, **settings}
        if self._initialized:
            if str(merged.get("value_format", "text")) not in _VALUE_FORMATS:
                raise ValueError("value_format is invalid")
            self._config.settings.update(settings)
            logger.info("BLE %s: settings saved; reconnect to apply", self._name)
            return
        self._apply_setting_caches(merged)  # validates before mutating
        self._config.settings.update(settings)

    # --- identity ---

    @property
    def device_type(self) -> str:
        return "BLE"

    @property
    def required_pins(self) -> list[str]:
        return []

    @property
    def address(self) -> str:
        """Configured peripheral address (may be blank if resolved by name)."""
        return self._address

    @property
    def is_streaming(self) -> bool:
        """Whether the device subscribes to notifications."""
        return self._notify

    @property
    def owns_link(self) -> bool:
        return True

    @property
    def link_state(self) -> ConnectionState:
        return self._link

    def _set_link(self, state: ConnectionState) -> None:
        """Move the tracked link state, notifying only on a real change.

        The no-change guard is what lets the supervisor poll every two
        seconds forever without repainting the GUI on every tick.
        """
        if state is self._link:
            return
        self._link = state
        logger.info("BLE %s: link -> %s", self._name, state.name.lower())
        self._notify_link_state()

    def _on_disconnected(self, client: Any) -> None:
        """bleak's disconnect callback: the peripheral went away.

        Called on the event loop. Ignores a client we have already replaced --
        an old client's teardown fires this too, and acting on it would report
        a live link as dead.
        """
        if client is not self._client:
            return
        self._set_link(ConnectionState.DISCONNECTED)

    async def poll_link(self) -> None:
        """Reconcile against ``client.is_connected``.

        The backstop for a disconnect callback that never fired, which
        CoreBluetooth and WinRT both do often enough to be the reported
        symptom rather than an edge case.
        """
        if not self._initialized:
            return
        client = self._client
        live = client is not None and client.is_connected
        if not live and self._link is ConnectionState.CONNECTED:
            self._set_link(ConnectionState.DISCONNECTED)

    @property
    def actions(self) -> dict[str, Callable]:
        return {"write": self.write, "read": self.read}

    # --- lifecycle ---

    async def initialize(self) -> None:
        """Connect (resolving the address if needed) and subscribe if requested."""
        if not self._address and not self._adv_name:
            raise ValueError("BLE: set an 'address' or a 'name' to resolve")
        async with self._lock:
            await self._ensure_connected()
            if self._notify:
                if not self._read_char:
                    raise ValueError("BLE: 'read_char_uuid' is required when notify is enabled")
                self._latest = None
                await self._client.start_notify(self._read_char, self._on_notify)
            self._initialized = True
        logger.info(
            "BLE initialized: %s (notify=%s)", self._address or self._adv_name, self._notify
        )

    async def shutdown(self) -> None:
        """Unsubscribe (best-effort) and disconnect.

        Clears ``_initialized`` FIRST so queued actions refuse to run, then does
        teardown inside try/finally so a failing stop/disconnect still clears
        state and drops the cache.
        """
        self._initialized = False
        try:
            async with self._lock:
                client = self._client
                self._client = None
                if client is not None:
                    if self._notify and self._read_char:
                        try:
                            await client.stop_notify(self._read_char)
                        except Exception:  # pragma: no cover - best-effort
                            pass
                    try:
                        await client.disconnect()
                    except Exception as e:  # pragma: no cover - best-effort
                        logger.warning("BLE: error during disconnect: %s", e)
        finally:
            self._initialized = False
            self._latest = None
            self._set_link(ConnectionState.DISCONNECTED)

    async def _resolve_address(self) -> str:
        """Return a usable address, scanning for the advertised name if needed."""
        if self._address:
            return self._address
        if self._resolved_address:
            return self._resolved_address
        return await self._find_by_name()

    async def _find_by_service(self) -> str | None:
        """Address of a peripheral advertising ``service_uuid``, if exactly one is.

        The identifier of last resort, and the sturdiest available: a service
        UUID neither rotates the way a private address does nor depends on a
        scan response surviving the trip -- which is how a Zephyr device ends up
        nameless on Windows while a phone app sees it fine.

        Returns None rather than guessing when several peripherals advertise the
        service. With six identical stimulators on a bench that is the normal
        case, and picking one would connect to the wrong animal's.
        """
        if not self._service_uuid:
            return None
        from glider.hal.boards.ble_board import BLEBoard

        found = await BLEBoard.scan(timeout=RESOLVE_SCAN_S)
        matches = [p for p in found if p.advertises(self._service_uuid)]
        if not matches:
            return None
        if len(matches) > 1:
            logger.warning(
                "BLE %s: %d peripherals advertise service %s (%s); set an address "
                "or an advertised name to say which one",
                self._name,
                len(matches),
                self._service_uuid,
                ", ".join(p.label for p in matches),
            )
            return None
        logger.info(
            "BLE %s: matched service %s -> %s", self._name, self._service_uuid, matches[0].address
        )
        return matches[0].address

    async def _find_by_name(self) -> str:
        """Scan for ``name`` and return the address it is advertising *now*."""
        from bleak import BleakScanner

        dev = await BleakScanner.find_device_by_name(self._adv_name, timeout=RESOLVE_SCAN_S)
        if dev is None:
            raise RuntimeError(
                f"BLE: no peripheral advertising name {self._adv_name!r} found "
                f"within {RESOLVE_SCAN_S:.0f}s. It may be connected to something "
                f"else -- a peripheral with a central attached usually stops "
                f"advertising -- or out of range."
            )
        self._resolved_address = dev.address
        logger.info("BLE: resolved name %r -> %s", self._adv_name, dev.address)
        return dev.address

    async def _ensure_connected(self) -> None:
        """Open (or reopen) the BleakClient connection if not already up."""
        if self._client is not None and self._client.is_connected:
            return
        try:
            from bleak import BleakClient
        except ImportError as e:
            raise RuntimeError(
                "bleak not installed. Run: pip install bleak (or reinstall GLIDER)."
            ) from e
        address = await self._resolve_address()
        try:
            client = BleakClient(address, disconnected_callback=self._on_disconnected)
            await client.connect()
        except Exception as exc:
            # A stored address goes stale. Many peripherals advertise a
            # *resolvable private address* that rotates every few minutes, so
            # the address the Scan button captured may name nothing by the time
            # anyone presses Connect -- bleak reports that as plainly "was not
            # found", which reads like the device is off.
            #
            # An advertised name does not rotate, so when one is configured it
            # is worth a rescan before giving up. The Scan button fills the
            # address and the operator often fills the name too, which makes
            # this the common case rather than an exotic one.
            if not self._adv_name and not self._service_uuid:
                raise
            logger.info(
                "BLE %s: no answer at %s (%s); re-resolving by name %r",
                self._name,
                address,
                exc,
                self._adv_name,
            )
            self._resolved_address = None
            # Name first when there is one -- it names *this* unit. The service
            # UUID is shared by every device of the type, so it can only help
            # when exactly one is in range.
            fresh = await self._find_by_name() if self._adv_name else None
            if fresh is None:
                fresh = await self._find_by_service()
            if fresh is None or fresh == address:
                # The name resolved to the address that just failed, so this is
                # not a rotation. Report the original failure rather than a
                # second identical one.
                raise
            client = BleakClient(fresh, disconnected_callback=self._on_disconnected)
            await client.connect()
            address = fresh

        self._client = client
        logger.info("BLE: connected to %s", address)
        self._set_link(ConnectionState.CONNECTED)

    # --- value decoding ---

    def _decode_value(self, data: bytes) -> Any:
        """Decode a raw GATT payload per ``value_format``."""
        if self._value_format == "text":
            return data.decode(self._encoding, errors="replace").strip()
        if self._value_format == "int":
            return int.from_bytes(data, "little", signed=False)
        if self._value_format == "int_be":
            return int.from_bytes(data, "big", signed=False)
        if self._value_format == "hex":
            return data.hex()
        return bytes(data)  # "bytes"

    def _on_notify(self, _characteristic: Any, data: bytearray) -> None:
        """bleak notification handler: cache the latest decoded value."""
        if not self._initialized:
            # A notification scheduled (call_soon_threadsafe) just before
            # shutdown must not repopulate _latest after shutdown cleared it --
            # get_state() deliberately ignores _initialized, so a resurrected
            # sample would be logged as a fabricated post-shutdown reading.
            return
        try:
            value = self._decode_value(bytes(data))
        except Exception:
            logger.exception("BLE %s: failed to decode notification", self._name)
            return
        self._latest = (value, time.perf_counter())

    def _fresh_sample(self) -> tuple[Any, float] | None:
        sample = self._latest
        if sample is None:
            return None
        if time.perf_counter() - sample[1] > MAX_SAMPLE_AGE_S:
            return None
        return sample

    async def get_state(self) -> Any | None:
        """Latest notification value for the DataRecorder poll (None if stale/absent)."""
        sample = self._fresh_sample()
        return sample[0] if sample is not None else None

    # --- actions ---

    @staticmethod
    def _format_arg(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    def _char_properties(self, char_uuid: str) -> set | None:
        """The connected characteristic's GATT properties, or None if unknown."""
        client = self._client
        if client is None or not char_uuid:
            return None
        try:
            char = client.services.get_characteristic(char_uuid)
            return set(char.properties) if char is not None else None
        except Exception:
            return None

    def _effective_response(self) -> bool:
        """Pick the write mode, auto-detecting from the characteristic.

        A characteristic that supports only one write mode is written that way;
        when it supports both (or its properties can't be read) the configured
        ``write_response`` preference is honored. (Ported from BLEWriteDevice.)
        """
        props = self._char_properties(self._write_char)
        if props is None:
            return self._write_response
        has_with = "write" in props
        has_without = "write-without-response" in props
        if has_with and not has_without:
            return True
        if has_without and not has_with:
            return False
        return self._write_response

    async def _with_retry(self, op: Callable) -> Any:
        """Run a GATT op, reconnecting once and retrying on a dropped link.

        Skips the retry if a shutdown ran in between (``_initialized`` is False),
        so a transient failure never re-arms a device that was just stopped.
        (Ported from BLEWriteDevice.write's retry.)
        """
        try:
            await self._ensure_connected()
            return await op()
        except Exception:
            if not self._initialized:
                raise
            self._client = None
            await self._ensure_connected()
            return await op()

    async def write(self, *args: Any) -> None:
        """Write a command to ``write_char_uuid`` (comma-joins multiple args).

        Auto-detects the write mode from the characteristic and retries once
        after a reconnect if the link dropped.
        """
        if not args or all(a is None for a in args):
            raise ValueError("BLE.write: command is required")
        if not self._write_char:
            raise ValueError("BLE.write: 'write_char_uuid' is not configured")
        command = ",".join(self._format_arg(a) for a in args if a is not None)
        data = command.encode(self._encoding)
        async with self._lock:
            if not self._initialized:
                raise RuntimeError(f"BLE device {self._name} is not initialized")
            await self._with_retry(
                lambda: self._client.write_gatt_char(
                    self._write_char, data, response=self._effective_response()
                )
            )
        logger.info("BLE: wrote %r to %s", command, self._write_char)

    async def read(self) -> Any:
        """Read the value characteristic.

        On a notify device this returns the latest pushed value (waiting briefly
        for the first one); otherwise it reads ``read_char_uuid`` on demand,
        retrying once after a reconnect if the link dropped.
        """
        if self._notify:
            return await self._await_fresh_sample()
        if not self._read_char:
            raise ValueError("BLE.read: 'read_char_uuid' is not configured")
        async with self._lock:
            if not self._initialized:
                raise RuntimeError(f"BLE device {self._name} is not initialized")
            data = await self._with_retry(lambda: self._client.read_gatt_char(self._read_char))
        return self._decode_value(bytes(data))

    async def _await_fresh_sample(self) -> Any:
        deadline = time.perf_counter() + READ_WAIT_S
        while True:
            sample = self._fresh_sample()
            if sample is not None:
                return sample[0]
            if not self._initialized:
                raise RuntimeError(f"BLE device {self._name} is not initialized")
            if time.perf_counter() >= deadline:
                raise RuntimeError(
                    f"BLE {self._name}: no notification within {READ_WAIT_S:.1f}s - "
                    "is the peripheral sending data?"
                )
            await asyncio.sleep(0.02)

    # --- serialization ---

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: "BaseBoard") -> "BLEDevice":
        config = DeviceConfig(
            pins=data["config"].get("pins", {}),
            settings=data["config"].get("settings", {}),
        )
        instance = cls(board, config, data.get("name"))
        instance._id = data.get("id", instance._id)
        return instance
