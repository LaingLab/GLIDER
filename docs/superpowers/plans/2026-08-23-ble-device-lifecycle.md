# BLE Device Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a BLE peripheral's real connection state visible everywhere GLIDER
claims to show it, reconnect automatically without ever silently re-arming a
stimulator, and make argument-taking device actions (Maimu's `pulse`) drivable
by hand on both control surfaces.

**Architecture:** `BaseDevice` grows a link-state interface — `owns_link`,
`link_state`, `poll_link()` — whose default derives from the board, so pin-based
devices are unaffected. `BLEDevice` overrides it with a real tracked state fed by
bleak's `disconnected_callback`, by I/O failures, and by a polled backstop, plus a
bounded reconnect task that re-subscribes notifications and calls an
`_on_reconnected()` hook that `MaimuDevice` uses to write `off`. `HardwareManager`
relays device state changes on a channel mirroring its board channel. Separately,
`ACTION_ARGS_SCHEMA` lets a device declare an action's arguments in the field
vocabulary `SETTINGS_SCHEMA` already uses, so both panels can render them.

**Tech Stack:** Python 3.11+, asyncio, bleak (>=0.21), PyQt6, pytest +
pytest-asyncio (`asyncio_mode = "auto"`) + pytest-qt.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-23-ble-device-lifecycle-design.md`. Read it before Task 1.
- **Test command** (works from the repo root, verified):
  `QT_QPA_PLATFORM=offscreen uv run --no-sync pytest <paths> -q`
  If `pytest` is not found, run `uv sync --extra dev` then
  `uv pip install -e ./plugins/glider-maimu` once, first.
- **Baseline:** `tests/unit/hal/test_ble_device_full.py` + `plugins/glider-maimu/tests` = 110 passing before Task 1. Never finish a task with fewer passing.
- **Lint/format gate** (CI runs these; run them before each commit):
  `uv run ruff check src tests plugins` and `uv run black --check src tests plugins`
- **Line length:** 100 (`[tool.black] line-length`). Match it.
- **Commit messages:** Conventional Commits (`feat(hal):`, `fix(gui):`, `test(hal):`). **No `Co-Authored-By` trailer and no "Generated with Claude Code" footer** — the repository does not use them.
- **GitHub Actions does not run.** CI credits are exhausted and checks sit pending forever. Local `pytest` is the only gate that exists; never report a task as verified on the strength of a CI badge.
- **Enum vocabulary:** device link states reuse `BoardConnectionState`'s five members. Do not introduce a second enum.
- **Naming:** `link_state` (not `connection_state`), `owns_link` (not `has_link`), `ACTION_ARGS_SCHEMA` (not `ACTIONS_SCHEMA`). Later tasks depend on these exact spellings.
- **Never write these** into any docstring or comment: "TODO", "TBD", "for now".

---

### Task 1: A device knows its own link state

Spec §4. Pure HAL. Adds the interface every later task consumes; changes no
behaviour for any existing device.

**Files:**
- Modify: `src/glider/hal/base_board.py` (add alias after the `BoardConnectionState` definition, currently ending line 70)
- Modify: `src/glider/hal/base_device.py` (`BaseDevice.__init__` ~line 70; new members after the `is_enabled` property ~line 118)
- Test: `tests/unit/hal/test_device_link_state.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `glider.hal.base_board.ConnectionState` — alias of `BoardConnectionState`.
  - `BaseDevice.owns_link -> bool` — property, `False`.
  - `BaseDevice.link_state -> ConnectionState` — property.
  - `async BaseDevice.poll_link() -> None` — no-op.
  - `BaseDevice.set_link_state_callback(cb: Callable[["BaseDevice"], None] | None) -> None`
  - `BaseDevice._notify_link_state() -> None` — protected; calls the callback, swallowing and logging any exception it raises.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/hal/test_device_link_state.py`:

```python
"""A device's link state, separate from whether it has been set up.

``_initialized`` was doing two jobs — "has been configured" and "is
reachable" — and the second was a lie the moment a link dropped. These cover
the default derivation: a pin-based device has no link of its own and is
exactly as connected as its board.
"""

import pytest

from glider.hal.base_board import BoardConnectionState, ConnectionState
from glider.hal.base_device import BaseDevice, DeviceConfig


class _FakeBoard:
    def __init__(self, connected=True):
        self.id = "fake_board"
        self.is_connected = connected


class _PinDevice(BaseDevice):
    """A device with no link of its own — the default case."""

    @property
    def device_type(self):
        return "PinThing"

    @property
    def actions(self):
        return {}

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False


def _device(connected=True):
    return _PinDevice(_FakeBoard(connected), DeviceConfig())


def test_connection_state_is_the_board_enum():
    """One vocabulary, not two: the strip's mapping already speaks it."""
    assert ConnectionState is BoardConnectionState


def test_pin_device_owns_no_link():
    assert _device().owns_link is False


def test_uninitialized_device_is_disconnected():
    assert _device().link_state is ConnectionState.DISCONNECTED


async def test_initialized_device_on_connected_board_is_connected():
    device = _device(connected=True)
    await device.initialize()
    assert device.link_state is ConnectionState.CONNECTED


async def test_initialized_device_on_dead_board_is_disconnected():
    device = _device(connected=True)
    await device.initialize()
    device.board.is_connected = False
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_shutdown_returns_to_disconnected():
    device = _device()
    await device.initialize()
    await device.shutdown()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_poll_link_is_a_no_op_by_default():
    """The supervisor calls this on every device; the default must be cheap."""
    device = _device()
    assert await device.poll_link() is None


def test_link_state_callback_fires_with_the_device():
    device = _device()
    seen = []
    device.set_link_state_callback(seen.append)
    device._notify_link_state()
    assert seen == [device]


def test_link_state_callback_can_be_cleared():
    device = _device()
    seen = []
    device.set_link_state_callback(seen.append)
    device.set_link_state_callback(None)
    device._notify_link_state()
    assert seen == []


def test_a_raising_callback_does_not_escape():
    """A broken GUI listener must not take a hardware state change down."""
    device = _device()

    def _boom(_dev):
        raise RuntimeError("listener exploded")

    device.set_link_state_callback(_boom)
    device._notify_link_state()  # must not raise
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_device_link_state.py -q
```

Expected: collection error — `ImportError: cannot import name 'ConnectionState' from 'glider.hal.base_board'`.

- [ ] **Step 3: Add the `ConnectionState` alias**

In `src/glider/hal/base_board.py`, immediately after the `BoardConnectionState`
class body (which ends with `RECONNECTING = auto()`), add:

```python
#: The same five states, under a name that does not say "Board".
#:
#: A BLE peripheral holds a link of its own that its board knows nothing about
#: (the "board" is the host adapter), so devices need this vocabulary too. It is
#: an alias rather than a second enum on purpose: the status strip's
#: DEVICE_STATE_BY_BOARD_STATE mapping and _board_state_text already render
#: these members, and a parallel enum meaning the same five things would need a
#: translation table whose only job is to be kept in sync.
ConnectionState = BoardConnectionState
```

- [ ] **Step 4: Add the link-state members to `BaseDevice`**

In `src/glider/hal/base_device.py`, at the end of `BaseDevice.__init__` (after
the `self._settings_changed_cb = None` line), add:

```python
        # Fired when this device's own link changes state. HardwareManager
        # wires it so the GUI can repaint; see set_link_state_callback.
        self._link_state_cb: Callable[[BaseDevice], None] | None = None
```

Then, immediately after the `is_enabled` property, add:

```python
    # --- link state ---

    @property
    def owns_link(self) -> bool:
        """Whether this device holds a connection of its own.

        False for a pin-based device: a DigitalOutput has no link separate
        from its board's, and giving it its own status dot would only
        duplicate the board's. True for a transport device (BLE, and serial
        when it adopts this) that opens and owns a socket.
        """
        return False

    @property
    def link_state(self) -> "ConnectionState":
        """Where this device's own link stands, right now.

        Derived rather than stored, which is what makes the default correct
        for every device that has no link to track: it is exactly as
        connected as its board, and DISCONNECTED before setup and after
        teardown. Transport devices override this with a real tracked state.

        Distinct from ``is_initialized``, which answers "has this been set
        up" and keeps its existing job of gating ``execute_action``. It was
        never able to answer "is this reachable", which is what every status
        readout was asking it.
        """
        from glider.hal.base_board import ConnectionState

        if not self._initialized:
            return ConnectionState.DISCONNECTED
        return (
            ConnectionState.CONNECTED
            if self._board.is_connected
            else ConnectionState.DISCONNECTED
        )

    async def poll_link(self) -> None:
        """Reconcile ``link_state`` against the transport, if that is possible.

        A no-op here: a derived ``link_state`` is already current every time
        it is read, so there is nothing to reconcile. Transport devices
        override this to catch a drop their disconnect callback missed, and
        HardwareManager's supervisor calls it on a timer.
        """
        return None

    def set_link_state_callback(
        self, callback: "Callable[[BaseDevice], None] | None"
    ) -> None:
        """Listen for changes to this device's link state.

        HardwareManager wires this in ``_track_device`` and re-broadcasts on
        its own device channel. Pass None to clear.
        """
        self._link_state_cb = callback

    def _notify_link_state(self) -> None:
        """Tell the listener the link moved.

        Never raises: this is called from a bleak disconnect callback and
        from a background reconnect task, where an exception has nowhere to
        go and would take the transport's state machine with it.
        """
        callback = self._link_state_cb
        if callback is None:
            return
        try:
            callback(self)
        except Exception:
            logger.exception("Link-state callback failed for device %s", self._name)
```

The `ConnectionState` import is function-local in `link_state` to avoid a
circular import (`base_board` does not import `base_device`, but the
`TYPE_CHECKING` block in `base_device` already imports from `base_board`, and
keeping the runtime import local matches how `BaseDevice` already defers).

- [ ] **Step 5: Run the test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_device_link_state.py -q
```

Expected: 10 passed.

- [ ] **Step 6: Confirm nothing else moved**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit plugins/glider-maimu/tests -q -m "not slow"
```

Expected: all pass. `owns_link`/`link_state` are additive; no existing device reads them yet.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/hal/base_board.py src/glider/hal/base_device.py tests/unit/hal/test_device_link_state.py
git commit -m "feat(hal): let a device report its own link state

_initialized answered 'has been set up' and was being read as 'is
reachable' by every status readout in the GUI. Split them: link_state is
derived from the board by default, so a pin-based device is unaffected,
and a transport device can override it with something it actually tracks."
```

---

### Task 2: `BLEDevice` tracks its real link

Spec §5. The disconnect callback and the polled backstop. No reconnect yet —
this task makes the drop *visible*, Task 3 makes it *recoverable*.

**Files:**
- Modify: `src/glider/hal/devices/ble_device.py`
- Test: `tests/unit/hal/test_ble_device_link.py` (create)

**Interfaces:**
- Consumes: `ConnectionState`, `BaseDevice.owns_link`, `BaseDevice.link_state`, `BaseDevice.poll_link`, `BaseDevice._notify_link_state` (Task 1).
- Produces:
  - `BLEDevice.owns_link` → `True`
  - `BLEDevice.link_state` → the tracked `self._link`
  - `BLEDevice._set_link(state: ConnectionState) -> None` — assigns and notifies **only on an actual change**
  - `BLEDevice._on_disconnected(client) -> None` — bleak callback; ignores a stale client
  - `async BLEDevice.poll_link() -> None` — override

- [ ] **Step 1: Write the failing test**

Create `tests/unit/hal/test_ble_device_link.py`:

```python
"""BLEDevice's tracked link state.

The complaint this answers: GLIDER kept calling a peripheral connected after
it had gone. Nothing passed bleak's disconnected_callback, and
_ensure_connected only consulted client.is_connected lazily at I/O time, so
the drop was genuinely unknown to the process until the next write failed.
"""

import sys

import pytest

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _FakeClient:
    """A BleakClient stand-in that can be told to drop."""

    def __init__(self, address, disconnected_callback=None):
        self.address = address
        self.is_connected = False
        self.written = []
        self.read_value = bytearray(b"42")
        self.disconnected_callback = disconnected_callback
        self._handler = None

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append((char, bytes(data), response))

    async def read_gatt_char(self, char):
        return bytearray(self.read_value)

    async def start_notify(self, char, handler):
        self._handler = handler

    async def stop_notify(self, char):
        self._handler = None

    def drop(self, *, notify=True):
        """Peripheral goes away. ``notify=False`` models a missed callback."""
        self.is_connected = False
        if notify and self.disconnected_callback is not None:
            self.disconnected_callback(self)


@pytest.fixture
def fake_bleak(monkeypatch):
    from unittest.mock import MagicMock

    created = {}

    def make_client(address, *a, disconnected_callback=None, **k):
        client = _FakeClient(address, disconnected_callback)
        created["client"] = client
        created.setdefault("clients", []).append(client)
        return client

    module = MagicMock(name="bleak")
    module.BleakClient = make_client
    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


async def _initialized(**settings):
    settings.setdefault("address", "11:22:33:44:55:66")
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings=settings), name="ble")
    await device.initialize()
    return device


def test_ble_owns_its_link():
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings={"address": "AA"}), name="ble")
    assert device.owns_link is True


def test_link_is_disconnected_before_initialize():
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings={"address": "AA"}), name="ble")
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_initialize_reports_connected(fake_bleak):
    device = await _initialized()
    assert device.link_state is ConnectionState.CONNECTED


async def test_the_client_is_built_with_a_disconnect_callback(fake_bleak):
    _module, created = fake_bleak
    await _initialized()
    assert created["client"].disconnected_callback is not None


async def test_a_drop_moves_the_state(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_a_drop_fires_the_listener_once(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    seen = []
    device.set_link_state_callback(lambda dev: seen.append(dev.link_state))
    created["client"].drop()
    assert seen == [ConnectionState.DISCONNECTED]


async def test_repeating_a_state_does_not_re-notify(fake_bleak):
    """The 2s poll runs forever; it must not spam the GUI with no news."""
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    seen = []
    device.set_link_state_callback(lambda dev: seen.append(dev.link_state))
    await device.poll_link()
    await device.poll_link()
    assert seen == []


async def test_poll_catches_a_drop_the_callback_missed(fake_bleak):
    """CoreBluetooth and WinRT both lose the callback often enough that this
    backstop is the mechanism, not a belt on top of one."""
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop(notify=False)
    assert device.link_state is ConnectionState.CONNECTED  # nobody has looked yet
    await device.poll_link()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_shutdown_reports_disconnected(fake_bleak):
    device = await _initialized()
    await device.shutdown()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_a_stale_client_callback_is_ignored(fake_bleak):
    """The old client's teardown callback must not knock over a live link."""
    _module, created = fake_bleak
    device = await _initialized()
    stale = created["client"]
    device._client = _FakeClient("other")
    device._client.is_connected = True
    device._set_link(ConnectionState.CONNECTED)
    stale.drop()
    assert device.link_state is ConnectionState.CONNECTED


async def test_poll_before_initialize_is_quiet(fake_bleak):
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings={"address": "AA"}), name="ble")
    await device.poll_link()
    assert device.link_state is ConnectionState.DISCONNECTED
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_ble_device_link.py -q
```

Expected: FAIL — `AttributeError: 'BLEDevice' object has no attribute '_set_link'`, and `owns_link` is False.

- [ ] **Step 3: Track the link in `BLEDevice`**

In `src/glider/hal/devices/ble_device.py`:

**(a)** Add to the imports at the top:

```python
from glider.hal.base_board import ConnectionState
```

**(b)** In `__init__`, after `self._latest: tuple[Any, float] | None = None`, add:

```python
        # The tracked link. Unlike the BaseDevice default this is stored, not
        # derived: the board here is the host adapter, which is "connected"
        # from the moment bleak imports and knows nothing about whether this
        # peripheral is actually answering.
        self._link = ConnectionState.DISCONNECTED
```

**(c)** After the `is_streaming` property, add:

```python
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
```

**(d)** In `_ensure_connected`, pass the callback to **both** `BleakClient(...)`
constructions. Change:

```python
            client = BleakClient(address)
            await client.connect()
```

to:

```python
            client = BleakClient(address, disconnected_callback=self._on_disconnected)
            await client.connect()
```

and the second one, in the re-resolve branch:

```python
            client = BleakClient(fresh)
            await client.connect()
```

to:

```python
            client = BleakClient(fresh, disconnected_callback=self._on_disconnected)
            await client.connect()
```

**(e)** At the end of `_ensure_connected`, after `logger.info("BLE: connected to %s", address)`, add:

```python
        self._set_link(ConnectionState.CONNECTED)
```

**(f)** In `shutdown`, inside the `finally` block, after `self._latest = None`, add:

```python
            self._set_link(ConnectionState.DISCONNECTED)
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_ble_device_link.py -q
```

Expected: 12 passed.

- [ ] **Step 5: Confirm the existing BLE and Maimu suites still pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal plugins/glider-maimu/tests -q -m "not slow"
```

Expected: all pass. Both existing fakes accept `(address, *a, **k)`, so the new
keyword argument is absorbed.

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/hal/devices/ble_device.py tests/unit/hal/test_ble_device_link.py
git commit -m "feat(hal): make a BLE device notice its own link dropping

Passes bleak's disconnected_callback, which was never wired, and adds a
poll_link backstop for the platforms that lose it. A dropped peripheral now
moves to DISCONNECTED instead of staying 'connected' until the next write
happens to fail."
```

---

### Task 3: Bounded reconnect, notify re-subscribe, safe-state hook

Spec §6 and §1.2. The largest task. Three coupled changes that cannot be
tested apart: the reconnect loop, the subscription it must restore, and the
hook it must call.

**Files:**
- Modify: `src/glider/hal/devices/ble_device.py`
- Test: `tests/unit/hal/test_ble_device_reconnect.py` (create)

**Interfaces:**
- Consumes: `BLEDevice._set_link`, `BLEDevice._on_disconnected`, `BLEDevice.link_state` (Task 2).
- Produces:
  - `BLEDevice.RECONNECT_BASE_S: float = 5.0`
  - `BLEDevice.MAX_RECONNECT_ATTEMPTS: int = 12`
  - `BLEDevice.RECONNECT_MAX_BACKOFF_S: float = 60.0`
  - `async BLEDevice._on_reconnected() -> None` — no-op hook, **the extension point Task 4 overrides**
  - `BLEDevice._start_reconnect() -> None`
  - `async BLEDevice._reconnect_loop() -> None`
  - `async BLEDevice._resubscribe() -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/hal/test_ble_device_reconnect.py`:

```python
"""BLEDevice's bounded reconnect.

Two behaviours carry the weight here.

The first is that a reconnect re-subscribes. _with_retry rebuilt the client on
a failed GATT op but only initialize() ever called start_notify, so a notify
device silently lost its subscription on the first blip and get_state()
returned None for the rest of the session with nothing logged.

The second is the _on_reconnected hook, and specifically that it fires on the
supervised reconnect and NOT on the reconnect inside a write's retry. A Maimu
uses it to write 'off'; running it on the retry path would cancel the exact
command the caller had just issued.
"""

import asyncio
import sys

import pytest

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _FakeClient:
    def __init__(self, address, disconnected_callback=None):
        self.address = address
        self.is_connected = False
        self.written = []
        self.disconnected_callback = disconnected_callback
        self._handler = None
        self.connect_error = None

    async def connect(self):
        if self.connect_error:
            raise self.connect_error
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append(bytes(data))

    async def read_gatt_char(self, char):
        return bytearray(b"1")

    async def start_notify(self, char, handler):
        self._handler = handler

    async def stop_notify(self, char):
        self._handler = None

    @property
    def subscribed(self):
        return self._handler is not None

    def push(self, data: bytes):
        assert self._handler is not None, "not subscribed"
        self._handler(object(), bytearray(data))

    def drop(self, *, notify=True):
        self.is_connected = False
        if notify and self.disconnected_callback is not None:
            self.disconnected_callback(self)


@pytest.fixture
def fake_bleak(monkeypatch):
    from unittest.mock import MagicMock

    created = {"clients": []}

    def make_client(address, *a, disconnected_callback=None, **k):
        client = _FakeClient(address, disconnected_callback)
        client.connect_error = created.get("connect_error")
        created["client"] = client
        created["clients"].append(client)
        return client

    module = MagicMock(name="bleak")
    module.BleakClient = make_client
    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    """Collapse the backoff so the suite runs in milliseconds, not minutes.

    The delays are asserted separately in test_backoff_doubles_to_the_cap,
    which reads them off a recorded sleep log instead of living through them.
    """
    monkeypatch.setattr(BLEDevice, "RECONNECT_BASE_S", 0.001)
    monkeypatch.setattr(BLEDevice, "RECONNECT_MAX_BACKOFF_S", 0.001)


async def _initialized(**settings):
    settings.setdefault("address", "11:22:33:44:55:66")
    device = BLEDevice(_FakeBoard(), DeviceConfig(settings=settings), name="ble")
    await device.initialize()
    return device


async def _settle(device, tries=200):
    """Let the reconnect task run to a resting state."""
    for _ in range(tries):
        if device.link_state is not ConnectionState.RECONNECTING:
            return
        await asyncio.sleep(0)
    raise AssertionError("reconnect never settled")


# --- the loop ----------------------------------------------------------------


async def test_a_drop_starts_a_reconnect(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    assert device.link_state is ConnectionState.CONNECTED
    await device.shutdown()


async def test_reconnect_builds_a_fresh_client(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    first = created["client"]
    first.drop()
    await _settle(device)
    assert device._client is not first
    assert device._client.is_connected
    await device.shutdown()


async def test_giving_up_lands_in_error(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("peripheral is gone")
    created["client"].drop()
    for _ in range(5000):
        if device.link_state is ConnectionState.ERROR:
            break
        await asyncio.sleep(0)
    assert device.link_state is ConnectionState.ERROR
    await device.shutdown()


async def test_attempts_are_bounded(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("gone")
    before = len(created["clients"])
    created["client"].drop()
    for _ in range(5000):
        if device.link_state is ConnectionState.ERROR:
            break
        await asyncio.sleep(0)
    attempts = len(created["clients"]) - before
    assert attempts == BLEDevice.MAX_RECONNECT_ATTEMPTS
    await device.shutdown()


async def test_backoff_doubles_to_the_cap(monkeypatch):
    """5 -> 10 -> 20 -> 40 -> 60 -> 60, matching BaseBoard._attempt_reconnect."""
    monkeypatch.setattr(BLEDevice, "RECONNECT_BASE_S", 5.0)
    monkeypatch.setattr(BLEDevice, "RECONNECT_MAX_BACKOFF_S", 60.0)
    delays = [BLEDevice._backoff_for(BLEDevice, attempt) for attempt in range(6)]
    assert delays == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]


async def test_shutdown_during_backoff_cancels_the_task(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("gone")
    created["client"].drop()
    await asyncio.sleep(0)
    await device.shutdown()
    task = device._reconnect_task
    assert task is None or task.done()
    assert device.link_state is ConnectionState.DISCONNECTED


async def test_only_one_reconnect_task_at_a_time(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["connect_error"] = OSError("gone")
    client = created["client"]
    client.drop()
    first = device._reconnect_task
    client.drop()
    assert device._reconnect_task is first
    await device.shutdown()


async def test_no_reconnect_after_shutdown(fake_bleak):
    """A drop reported during teardown must not resurrect the device."""
    _module, created = fake_bleak
    device = await _initialized()
    client = created["client"]
    await device.shutdown()
    client.drop()
    assert device._reconnect_task is None
    assert device.link_state is ConnectionState.DISCONNECTED


# --- the subscription --------------------------------------------------------


async def test_reconnect_restores_the_notify_subscription(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized(read_char_uuid="beef", notify=True)
    created["client"].drop()
    await _settle(device)
    assert device._client.subscribed
    device._client.push(b"7")
    assert await device.get_state() == "7"
    await device.shutdown()


async def test_a_non_notify_device_does_not_subscribe(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized(write_char_uuid="cafe")
    created["client"].drop()
    await _settle(device)
    assert not device._client.subscribed
    await device.shutdown()


# --- the hook ----------------------------------------------------------------


async def test_the_hook_runs_on_a_supervised_reconnect(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    ran = []
    device._on_reconnected = lambda: _record(ran)
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        if ran:
            break
        await asyncio.sleep(0)
    assert ran == ["hook"]
    await device.shutdown()


def _record(log):
    async def _run():
        log.append("hook")

    return _run()


async def test_the_hook_does_not_run_on_a_write_retry(fake_bleak):
    """The retry path is a caller's own command; an 'off' there would cancel it."""
    _module, created = fake_bleak
    device = await _initialized(write_char_uuid="cafe")
    ran = []
    device._on_reconnected = lambda: _record(ran)

    original = created["client"]
    failing = {"first": True}

    async def _flaky(char, data, response=False):
        if failing["first"]:
            failing["first"] = False
            raise OSError("link dropped mid-write")
        original.written.append(bytes(data))

    device._client.write_gatt_char = _flaky
    device._client.is_connected = False  # force _ensure_connected to rebuild

    await device.write("hello")
    await asyncio.sleep(0)
    assert ran == []
    assert device._client.written == [b"hello"]
    await device.shutdown()


async def test_a_failing_hook_leaves_the_link_up(fake_bleak):
    """The link genuinely reconnected; a failed safe-state write does not undo that."""
    _module, created = fake_bleak
    device = await _initialized()

    async def _boom():
        raise OSError("could not send off")

    device._on_reconnected = _boom
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        await asyncio.sleep(0)
    assert device.link_state is ConnectionState.CONNECTED
    await device.shutdown()
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_ble_device_reconnect.py -q
```

Expected: FAIL — `AttributeError: type object 'BLEDevice' has no attribute 'RECONNECT_BASE_S'`.

- [ ] **Step 3: Implement the reconnect**

In `src/glider/hal/devices/ble_device.py`:

**(a)** In `__init__`, after the `self._link = ...` line from Task 2, add:

```python
        self._reconnect_task: asyncio.Task | None = None
```

**(b)** After the `_set_link` / `_on_disconnected` / `poll_link` block from
Task 2, add the whole reconnect section:

```python
    # --- reconnect ---

    #: Backoff base, doubling per attempt. Deliberately the same numbers as
    #: BaseBoard._attempt_reconnect: a peripheral and a board that drop for the
    #: same reason should not retry on two different rhythms.
    RECONNECT_BASE_S: float = 5.0
    RECONNECT_MAX_BACKOFF_S: float = 60.0
    MAX_RECONNECT_ATTEMPTS: int = 12

    def _backoff_for(self, attempt: int) -> float:
        """Seconds to wait before ``attempt`` (0-based): 5, 10, 20, 40, 60, 60…"""
        return min(self.RECONNECT_BASE_S * (2**attempt), self.RECONNECT_MAX_BACKOFF_S)

    async def _on_reconnected(self) -> None:
        """Hook: the link is back up and any subscription has been restored.

        A no-op for a generic peripheral, which has no state worth asserting
        on reconnect. Overridden by devices that do -- a stimulator that runs
        its pattern in firmware came back mid-train, and this is the first
        chance anyone has had to stop it.

        Called with the device lock RELEASED, because an override will want to
        write, and ``write()`` takes that lock. Called only on the supervised
        reconnect path, never on ``_with_retry``'s reconnect-inside-a-write.
        """
        return None

    def _start_reconnect(self) -> None:
        """Begin retrying, unless a retry is already running or we are down.

        Safe to call from a bleak callback: it only schedules.
        """
        if not self._initialized:
            return
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        self._set_link(ConnectionState.RECONNECTING)
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Retry with bounded exponential backoff; give up into ERROR.

        Mirrors BaseBoard._attempt_reconnect, including clearing the task
        handle in ``finally`` so a later drop can start a fresh one.
        """
        try:
            for attempt in range(self.MAX_RECONNECT_ATTEMPTS):
                try:
                    await asyncio.sleep(self._backoff_for(attempt))
                except asyncio.CancelledError:
                    return
                if not self._initialized:
                    return
                try:
                    async with self._lock:
                        if not self._initialized:
                            return
                        self._client = None
                        await self._ensure_connected()
                        await self._resubscribe()
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - retried, then reported
                    logger.info(
                        "BLE %s: reconnect attempt %d/%d failed (%s)",
                        self._name,
                        attempt + 1,
                        self.MAX_RECONNECT_ATTEMPTS,
                        e,
                    )
                    self._set_link(ConnectionState.RECONNECTING)
                    continue

                # The link is up. Say so before running the hook: it is true,
                # and an override that writes needs write() to work.
                self._set_link(ConnectionState.CONNECTED)
                try:
                    await self._on_reconnected()
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - the link is still up
                    logger.warning(
                        "BLE %s: post-reconnect hook failed (%s); the link is up "
                        "but the device may not be in its safe state",
                        self._name,
                        e,
                    )
                return

            logger.error(
                "BLE %s: gave up reconnecting after %d attempts",
                self._name,
                self.MAX_RECONNECT_ATTEMPTS,
            )
            self._set_link(ConnectionState.ERROR)
        finally:
            self._reconnect_task = None

    async def _resubscribe(self) -> None:
        """Restore the notify subscription on a freshly built client.

        ``initialize()`` is the only other place ``start_notify`` is called, so
        without this a reconnect leaves a notify device permanently silent --
        ``_latest`` never refreshes again and ``get_state()`` returns None for
        the rest of the session. Caller holds ``self._lock``.
        """
        if not self._notify or not self._read_char or self._client is None:
            return
        self._latest = None
        await self._client.start_notify(self._read_char, self._on_notify)
```

**(c)** Make `_on_disconnected` kick off the retry. Change its body's last line
from:

```python
        self._set_link(ConnectionState.DISCONNECTED)
```

to:

```python
        self._set_link(ConnectionState.DISCONNECTED)
        self._start_reconnect()
```

**(d)** Make the poll backstop kick it off too. In `poll_link`, change:

```python
        if not live and self._link is ConnectionState.CONNECTED:
            self._set_link(ConnectionState.DISCONNECTED)
```

to:

```python
        if not live and self._link is ConnectionState.CONNECTED:
            self._set_link(ConnectionState.DISCONNECTED)
            self._start_reconnect()
```

**(e)** Cancel the task in `shutdown`. `shutdown` currently begins:

```python
        self._initialized = False
        try:
            async with self._lock:
```

Change it to:

```python
        self._initialized = False
        # Cancel BEFORE taking the lock: the reconnect loop holds it while it
        # connects, and an emergency stop must not queue behind a retry. The
        # _initialized guard above already stops a retry from re-arming a
        # device that was just stopped; this stops it from running at all.
        task = self._reconnect_task
        self._reconnect_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - teardown must not raise
                logger.debug("BLE %s: reconnect task errored on cancel", self._name)
        try:
            async with self._lock:
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_ble_device_reconnect.py -q
```

Expected: 14 passed.

If `test_backoff_doubles_to_the_cap` fails with a `TypeError`, note that it
calls `BLEDevice._backoff_for(BLEDevice, attempt)` — the unbound form, passing
the class as `self` so the class attributes resolve. That is intentional; the
method reads only class attributes.

- [ ] **Step 5: Confirm the whole HAL and plugin suites still pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit plugins/glider-maimu/tests -q -m "not slow"
```

Expected: all pass.

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/hal/devices/ble_device.py tests/unit/hal/test_ble_device_reconnect.py
git commit -m "feat(hal): reconnect a dropped BLE link, and restore its subscription

Bounded exponential backoff on the same schedule BaseBoard already uses,
into ERROR when it gives up. A reconnect now re-subscribes notifications --
_with_retry rebuilt the client but only initialize() ever called
start_notify, so a notify device went permanently silent after the first
blip with nothing logged.

Adds the _on_reconnected hook, which fires only on the supervised path. On
the reconnect inside a write's retry it would cancel the caller's own
command."
```

---

### Task 4: A Maimu comes back off

Spec §6.1. Small and plugin-side, but it is the whole safety argument for §6.
Also fixes a `testpaths` gap that hides this plugin's suite from a bare local
`pytest` run.

**Files:**
- Modify: `plugins/glider-maimu/src/glider_maimu/device.py`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options] testpaths`, line 179)
- Test: `plugins/glider-maimu/tests/test_device_reconnect.py` (create)

**Interfaces:**
- Consumes: `BLEDevice._on_reconnected`, `BLEDevice._start_reconnect`, `BLEDevice.RECONNECT_BASE_S`, `BLEDevice.RECONNECT_MAX_BACKOFF_S`, `BLEDevice.link_state` (Task 3).
- Produces: `MaimuDevice._on_reconnected()` override. Nothing later depends on it.

- [ ] **Step 1: Write the failing test**

Create `plugins/glider-maimu/tests/test_device_reconnect.py`:

```python
"""A Maimu comes back off.

The firmware runs a pulse autonomously, so a link that died mid-train left a
stimulator running with nothing attached to stop it. The reconnect is the
first chance anyone has had to say otherwise, and it takes it -- the device
never silently resumes whatever it was doing.

Same reasoning as shutdown(), which writes 'off' before it disconnects.
"""

import asyncio
import sys

import pytest

from glider.hal.base_board import ConnectionState
from glider.hal.base_device import DeviceConfig
from glider.hal.devices.ble_device import BLEDevice
from glider_maimu.device import MaimuDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _FakeClient:
    def __init__(self, address, disconnected_callback=None):
        self.address = address
        self.is_connected = False
        self.written = []
        self.disconnected_callback = disconnected_callback

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def write_gatt_char(self, char, data, response=False):
        self.written.append(bytes(data))

    def drop(self):
        self.is_connected = False
        if self.disconnected_callback is not None:
            self.disconnected_callback(self)


@pytest.fixture
def fake_bleak(monkeypatch):
    from unittest.mock import MagicMock

    created = {"clients": []}

    def make_client(address, *a, disconnected_callback=None, **k):
        client = _FakeClient(address, disconnected_callback)
        created["client"] = client
        created["clients"].append(client)
        return client

    module = MagicMock(name="bleak")
    module.BleakClient = make_client
    monkeypatch.setitem(sys.modules, "bleak", module)
    return module, created


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch):
    monkeypatch.setattr(BLEDevice, "RECONNECT_BASE_S", 0.001)
    monkeypatch.setattr(BLEDevice, "RECONNECT_MAX_BACKOFF_S", 0.001)


async def _initialized():
    config = DeviceConfig(settings={"address": "11:22:33:44:55:66"})
    device = MaimuDevice(_FakeBoard(), config, name="maimu")
    await device.initialize()
    return device


async def _settle(device):
    for _ in range(500):
        if device.link_state is ConnectionState.CONNECTED:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(0)
    raise AssertionError("never reconnected")


async def test_reconnect_writes_off(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        if device._client.written:
            break
        await asyncio.sleep(0)
    assert device._client.written == [b"off"]
    await device.shutdown()


async def test_off_is_written_exactly_once(fake_bleak):
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    for _ in range(100):
        await asyncio.sleep(0)
    assert device._client.written.count(b"off") == 1
    await device.shutdown()


async def test_off_is_the_first_thing_on_the_new_link(fake_bleak):
    """A pulse issued after the reconnect must land after the safe state."""
    _module, created = fake_bleak
    device = await _initialized()
    created["client"].drop()
    await _settle(device)
    for _ in range(50):
        if device._client.written:
            break
        await asyncio.sleep(0)
    await device.pulse(500, 10)
    assert device._client.written == [b"off", b"500,10"]
    await device.shutdown()


async def test_a_write_retry_does_not_write_off(fake_bleak):
    """The retry path carries the caller's own command; 'off' would cancel it."""
    _module, created = fake_bleak
    device = await _initialized()
    client = device._client
    failing = {"first": True}
    real_write = client.write_gatt_char

    async def _flaky(char, data, response=False):
        if failing["first"]:
            failing["first"] = False
            raise OSError("link dropped mid-write")
        await real_write(char, data, response)

    client.write_gatt_char = _flaky
    client.is_connected = False

    await device.pulse(500, 10)
    await asyncio.sleep(0)
    assert b"off" not in device._client.written
    assert device._client.written == [b"500,10"]
    await device.shutdown()
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest plugins/glider-maimu/tests/test_device_reconnect.py -q
```

Expected: FAIL — `assert [] == [b'off']`. `MaimuDevice` has not overridden the hook.

- [ ] **Step 3: Override the hook**

In `plugins/glider-maimu/src/glider_maimu/device.py`, in the `# --- lifecycle ---`
section, immediately **before** `async def shutdown`, add:

```python
    async def _on_reconnected(self) -> None:
        """Come back off.

        The firmware runs a pulse autonomously, so a link that dropped
        mid-train left the stimulator running with nothing attached to stop
        it. Whatever it is doing, it has been doing it unsupervised, and this
        is the first moment anyone can say otherwise -- so the device is put
        in a known state rather than resumed in an unknown one.

        Same reasoning as :meth:`shutdown`, and the same best-effort
        treatment: BLEDevice logs a failure here and leaves the link up,
        because the link genuinely did reconnect.
        """
        await self.write("off")
```

Also extend the module docstring. Find the paragraph beginning
`**Stopping matters here.**` and append to it:

```
The same reasoning covers a *dropped* link: :meth:`MaimuDevice._on_reconnected`
writes ``off`` when the automatic reconnect succeeds, so a stimulator that came
back mid-pattern is put in a known state instead of left running one nobody
asked for.
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest plugins/glider-maimu/tests/test_device_reconnect.py -q
```

Expected: 4 passed.

- [ ] **Step 5: Close the `testpaths` gap**

`pyproject.toml` line 179 lists `plugins/glider-harp/tests` but not
`plugins/glider-maimu/tests`, so a bare `pytest` has never collected this
plugin's suite — even though CI's explicit command line does, and its comment
claims the two match. With Actions not running, a bare local `pytest` is the
only gate there is. Change:

```toml
testpaths = ["tests/unit", "tests/integration", "plugins/glider-harp/tests"]
```

to:

```toml
testpaths = [
    "tests/unit",
    "tests/integration",
    "plugins/glider-harp/tests",
    "plugins/glider-maimu/tests",
]
```

- [ ] **Step 6: Verify a bare run now collects the Maimu suite**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest --collect-only -q 2>&1 | grep -c "glider-maimu"
```

Expected: a number greater than 0 (it was 0 before this step).

- [ ] **Step 7: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Expected: all pass.

- [ ] **Step 8: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add plugins/glider-maimu/src/glider_maimu/device.py plugins/glider-maimu/tests/test_device_reconnect.py pyproject.toml
git commit -m "feat(maimu): write 'off' when a dropped link comes back

The firmware runs a pulse on its own, so a link that died mid-train left a
stimulator running unsupervised. The reconnect puts it in a known state
rather than resuming an unknown one.

Also adds plugins/glider-maimu/tests to testpaths. CI passed the path
explicitly, so a bare local pytest had never collected this plugin's suite."
```

---

### Task 5: `HardwareManager` relays device link changes

Spec §4.1 and §5's supervisor.

**Files:**
- Modify: `src/glider/core/hardware_manager.py`
- Test: `tests/unit/core/test_hardware_manager_device_links.py` (create)

**Interfaces:**
- Consumes: `BaseDevice.owns_link`, `BaseDevice.link_state`, `BaseDevice.poll_link`, `BaseDevice.set_link_state_callback` (Task 1).
- Produces:
  - `glider.core.hardware_manager.LINK_POLL_INTERVAL_S: float = 2.0`
  - `HardwareManager.on_device_connection_change(cb: Callable[[str, ConnectionState], None]) -> None`
  - `HardwareManager.start_link_supervisor() -> None`
  - `async HardwareManager.stop_link_supervisor() -> None`
  - `async HardwareManager.poll_device_links() -> None` — one sweep; the supervisor's body, exposed so it can be tested without a timer.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_hardware_manager_device_links.py`:

```python
"""HardwareManager's device link channel.

Mirrors the board channel (on_connection_change) so the GUI has one shape to
learn, and wires each device in _track_device -- the same chokepoint that
already wires the settings hook, so no creation path can register a device
without it.
"""

import asyncio

import pytest

from glider.core.hardware_manager import LINK_POLL_INTERVAL_S, HardwareManager
from glider.hal.base_board import ConnectionState
from glider.hal.base_device import BaseDevice, DeviceConfig


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _LinkDevice(BaseDevice):
    """A device that owns a link, so the supervisor polls it."""

    def __init__(self, board, config, name=None):
        super().__init__(board, config, name)
        self._link = ConnectionState.DISCONNECTED
        self.polls = 0

    @property
    def device_type(self):
        return "FakeLink"

    @property
    def actions(self):
        return {}

    @property
    def owns_link(self):
        return True

    @property
    def link_state(self):
        return self._link

    async def poll_link(self):
        self.polls += 1

    def set_link(self, state):
        self._link = state
        self._notify_link_state()

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False


class _PinDevice(BaseDevice):
    """No link of its own; the supervisor must skip it."""

    def __init__(self, board, config, name=None):
        super().__init__(board, config, name)
        self.polls = 0

    @property
    def device_type(self):
        return "FakePin"

    @property
    def actions(self):
        return {}

    async def poll_link(self):
        self.polls += 1

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False


def _manager_with(device, device_id="dev1"):
    manager = HardwareManager()
    manager._track_device(device_id, device)
    return manager


def test_the_poll_interval_is_two_seconds():
    assert LINK_POLL_INTERVAL_S == 2.0


def test_a_device_link_change_reaches_a_listener():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    seen = []
    manager.on_device_connection_change(lambda dev_id, state: seen.append((dev_id, state)))
    device.set_link(ConnectionState.RECONNECTING)
    assert seen == [("dev1", ConnectionState.RECONNECTING)]


def test_every_listener_is_told():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    first, second = [], []
    manager.on_device_connection_change(lambda d, s: first.append(s))
    manager.on_device_connection_change(lambda d, s: second.append(s))
    device.set_link(ConnectionState.ERROR)
    assert first == second == [ConnectionState.ERROR]


def test_a_raising_listener_does_not_stop_the_others():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    survived = []

    def _boom(_dev_id, _state):
        raise RuntimeError("listener exploded")

    manager.on_device_connection_change(_boom)
    manager.on_device_connection_change(lambda d, s: survived.append(s))
    device.set_link(ConnectionState.CONNECTED)
    assert survived == [ConnectionState.CONNECTED]


def test_a_device_tracked_before_the_listener_is_still_wired():
    """Registration order must not decide whether a device is heard."""
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)  # tracked first
    seen = []
    manager.on_device_connection_change(lambda d, s: seen.append(s))  # listener second
    device.set_link(ConnectionState.DISCONNECTED)
    assert seen == [ConnectionState.DISCONNECTED]


async def test_a_sweep_polls_a_link_owning_device():
    device = _LinkDevice(_FakeBoard(), DeviceConfig(), name="ble")
    manager = _manager_with(device)
    await manager.poll_device_links()
    assert device.polls == 1


async def test_a_sweep_skips_a_pin_device():
    """Polling a derived link_state is pure overhead on every rig with GPIO."""
    device = _PinDevice(_FakeBoard(), DeviceConfig(), name="led")
    manager = _manager_with(device)
    await manager.poll_device_links()
    assert device.polls == 0


async def test_a_failing_poll_does_not_stop_the_sweep():
    good = _LinkDevice(_FakeBoard(), DeviceConfig(), name="good")
    bad = _LinkDevice(_FakeBoard(), DeviceConfig(), name="bad")

    async def _boom():
        raise OSError("adapter went away")

    bad.poll_link = _boom
    manager = HardwareManager()
    manager._track_device("bad", bad)
    manager._track_device("good", good)
    await manager.poll_device_links()
    assert good.polls == 1


async def test_the_supervisor_starts_and_stops():
    manager = HardwareManager()
    manager.start_link_supervisor()
    assert manager._link_supervisor is not None
    await manager.stop_link_supervisor()
    assert manager._link_supervisor is None


async def test_starting_twice_keeps_one_task():
    manager = HardwareManager()
    manager.start_link_supervisor()
    first = manager._link_supervisor
    manager.start_link_supervisor()
    assert manager._link_supervisor is first
    await manager.stop_link_supervisor()


async def test_stopping_when_not_started_is_quiet():
    await HardwareManager().stop_link_supervisor()  # must not raise


async def test_manager_shutdown_stops_the_supervisor():
    manager = HardwareManager()
    manager.start_link_supervisor()
    await manager.shutdown()
    assert manager._link_supervisor is None
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/core/test_hardware_manager_device_links.py -q
```

Expected: collection error — `ImportError: cannot import name 'LINK_POLL_INTERVAL_S'`.

- [ ] **Step 3: Implement the channel and the supervisor**

In `src/glider/core/hardware_manager.py`:

**(a)** After the `DEVICE_IO_TIMEOUT_S = 2.0` constant, add:

```python
# How often the link supervisor asks each transport device to reconcile its
# link state. The disconnect callback is the primary signal; this is the
# backstop for the platforms that lose it, so it only has to be fast enough
# that a human does not notice the lag.
LINK_POLL_INTERVAL_S = 2.0
```

**(b)** In `__init__`, after `self._connection_callbacks: ... = []`, add:

```python
        self._device_connection_callbacks: list[
            Callable[[str, BoardConnectionState], None]
        ] = []
        self._link_supervisor: asyncio.Task | None = None
```

**(c)** In `_wire_device_settings`'s neighbourhood, add a sibling method after it:

```python
    def _wire_device_link(self, device_id: str, device: "BaseDevice") -> None:
        """Point a device's link-state hook at our device channel."""
        device.set_link_state_callback(
            lambda dev, _id=device_id: self._notify_device_connection_change(
                _id, dev.link_state
            )
        )
```

**(d)** In `_track_device`, after `self._wire_device_settings(device_id, device)`, add:

```python
        self._wire_device_link(device_id, device)
```

and extend that method's docstring to say "and its link-state hook" where it
says "and wires its settings-changed hook".

**(e)** After `on_connection_change`, add:

```python
    def on_device_connection_change(
        self, callback: Callable[[str, BoardConnectionState], None]
    ) -> None:
        """Register a callback for *device* link state changes.

        The peripheral-level sibling of :meth:`on_connection_change`. A BLE
        board is the host adapter, which is "connected" from the moment bleak
        imports; only the device knows whether the peripheral is answering.
        """
        self._device_connection_callbacks.append(callback)
```

**(f)** After `_notify_connection_change`, add:

```python
    def _notify_device_connection_change(
        self, device_id: str, state: BoardConnectionState
    ) -> None:
        """Notify device link-state callbacks."""
        for callback in self._device_connection_callbacks:
            try:
                callback(device_id, state)
            except Exception as e:
                logger.error(f"Device connection callback failed: {e}")
```

**(g)** Before `async def shutdown`, add the supervisor:

```python
    # --- device link supervision ---

    async def poll_device_links(self) -> None:
        """Ask every link-owning device to reconcile its state once.

        Skips devices whose ``link_state`` is derived: there is nothing to
        reconcile, and polling them would be pure overhead on every rig with
        GPIO on it. One device's failure never stops the sweep.
        """
        for device_id, device in list(self._devices.items()):
            if not device.owns_link:
                continue
            try:
                await device.poll_link()
            except Exception as e:  # noqa: BLE001 - one bad device, not a sweep
                logger.debug("Link poll failed for device %s: %s", device_id, e)

    def start_link_supervisor(self) -> None:
        """Begin polling device links on a timer. Idempotent."""
        if self._link_supervisor is not None and not self._link_supervisor.done():
            return
        self._link_supervisor = asyncio.create_task(self._link_supervisor_loop())

    async def _link_supervisor_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(LINK_POLL_INTERVAL_S)
            except asyncio.CancelledError:
                return
            await self.poll_device_links()

    async def stop_link_supervisor(self) -> None:
        """Cancel the supervisor and wait for it. Safe when never started."""
        task = self._link_supervisor
        self._link_supervisor = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
```

**(h)** In `async def shutdown`, make the supervisor the first thing stopped:

```python
    async def shutdown(self) -> None:
        """Shutdown the hardware manager."""
        logger.info("Shutting down hardware manager")
        await self.stop_link_supervisor()
        await self.emergency_stop()
        await self.disconnect_all()
        self._boards.clear()
        self._devices.clear()
        self._pin_managers.clear()
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/core/test_hardware_manager_device_links.py -q
```

Expected: 13 passed.

- [ ] **Step 5: Confirm nothing else moved**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Expected: all pass.

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/core/hardware_manager.py tests/unit/core/test_hardware_manager_device_links.py
git commit -m "feat(core): relay device link state, and poll it as a backstop

A device channel mirroring the board one, wired in _track_device so no
creation path can register a device without it, plus a 2s supervisor sweep
for the platforms that lose bleak's disconnect callback."
```

---

### Task 6: The tree and the Device Control panel tell the truth

Spec §7, first two surfaces. Introduces the shared vocabulary module both use.

**Files:**
- Create: `src/glider/gui/device_status.py`
- Modify: `src/glider/gui/panels/hardware_panel.py:165-172` (the device tree item's status column)
- Modify: `src/glider/gui/panels/device_control_panel.py:255-266` (the status line) and `_run_device_action`
- Test: `tests/unit/gui/test_device_link_status.py` (create)

**Interfaces:**
- Consumes: `BaseDevice.link_state`, `ConnectionState` (Task 1).
- Produces:
  - `glider.gui.device_status.link_status_text(state: ConnectionState) -> str` — `"Ready"` / `"Connecting…"` / `"Reconnecting…"` / `"Disconnected"` / `"Error"`
  - `glider.gui.device_status.link_strip_state(state: ConnectionState) -> str` — one of the strip's `DEVICE_STATES` (`"ok"` / `"warn"` / `"error"` / `"unknown"`)
  - `glider.gui.device_status.link_is_usable(state: ConnectionState) -> bool` — True only for `CONNECTED`

  Task 7 consumes `link_strip_state`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/test_device_link_status.py`:

```python
"""Device link state as the GUI renders it.

One vocabulary module, because the tree, the Device Control panel and the
status strip were each inventing their own word for the same state -- which
is how the status bar came to read "Connected" beside a red dot.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton

from glider.gui.device_status import link_is_usable, link_status_text, link_strip_state
from glider.gui.panels.device_control_panel import DeviceControlPanel
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


# --- the vocabulary -----------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [
        (ConnectionState.CONNECTED, "Ready"),
        (ConnectionState.CONNECTING, "Connecting…"),
        (ConnectionState.RECONNECTING, "Reconnecting…"),
        (ConnectionState.DISCONNECTED, "Disconnected"),
        (ConnectionState.ERROR, "Error"),
    ],
)
def test_status_text(state, expected):
    assert link_status_text(state) == expected


@pytest.mark.parametrize(
    "state,expected",
    [
        (ConnectionState.CONNECTED, "ok"),
        (ConnectionState.CONNECTING, "warn"),
        (ConnectionState.RECONNECTING, "warn"),
        (ConnectionState.DISCONNECTED, "error"),
        (ConnectionState.ERROR, "error"),
    ],
)
def test_strip_state(state, expected):
    assert link_strip_state(state) == expected


def test_an_unknown_state_is_never_green():
    """A state nobody recognises does not get the benefit of the doubt."""
    assert link_strip_state(object()) == "unknown"
    assert link_status_text(object()) == "Unknown"


@pytest.mark.parametrize(
    "state,usable",
    [
        (ConnectionState.CONNECTED, True),
        (ConnectionState.RECONNECTING, False),
        (ConnectionState.DISCONNECTED, False),
        (ConnectionState.ERROR, False),
        (ConnectionState.CONNECTING, False),
    ],
)
def test_usable_only_when_connected(state, usable):
    assert link_is_usable(state) is usable


# --- the Device Control panel -------------------------------------------------


class _Device:
    device_type = "Maimu"
    name = "Stimulator"
    _initialized = True

    def __init__(self, state=ConnectionState.CONNECTED):
        self.link_state = state
        self.owns_link = True
        self.board = SimpleNamespace(is_connected=True)

    async def on(self):
        pass

    async def off(self):
        pass

    @property
    def actions(self):
        return {"on": self.on, "off": self.off}


def _panel(qtbot, device):
    manager = SimpleNamespace(
        devices={"dev1": device},
        get_device=lambda dev_id: device if dev_id == "dev1" else None,
    )
    panel = DeviceControlPanel(manager, lambda coro: coro.close())
    qtbot.addWidget(panel)
    panel.refresh_devices()
    panel._device_combo.setCurrentIndex(1)
    return panel


def test_panel_status_reads_the_link(qtbot):
    panel = _panel(qtbot, _Device(ConnectionState.RECONNECTING))
    assert "Reconnecting" in panel._device_status_label.text()


def test_panel_status_says_ready_when_up(qtbot):
    panel = _panel(qtbot, _Device(ConnectionState.CONNECTED))
    assert "Ready" in panel._device_status_label.text()


def test_action_buttons_are_dead_while_the_link_is_down(qtbot):
    """Offering a press that is certain to fail is worse than not offering it."""
    panel = _panel(qtbot, _Device(ConnectionState.DISCONNECTED))
    buttons = panel._actions_widget.findChildren(QPushButton)
    assert buttons, "expected on/off buttons to be built"
    assert all(not b.isEnabled() for b in buttons)


def test_action_buttons_are_live_when_the_link_is_up(qtbot):
    panel = _panel(qtbot, _Device(ConnectionState.CONNECTED))
    buttons = panel._actions_widget.findChildren(QPushButton)
    assert buttons and all(b.isEnabled() for b in buttons)
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_device_link_status.py -q
```

Expected: collection error — `ModuleNotFoundError: No module named 'glider.gui.device_status'`.

- [ ] **Step 3: Create the vocabulary module**

Create `src/glider/gui/device_status.py`:

```python
"""How a device's link state is worded and coloured.

One module because three surfaces render the same five states -- the hardware
tree, the Device Control panel, and the status strip -- and each of them
inventing its own word is how the status bar came to read "Connected" beside a
red dot. The strip's own board-level mapping (DEVICE_STATE_BY_BOARD_STATE in
main_window) is the same idea for boards; this is its device sibling.
"""

from __future__ import annotations

from glider.hal.base_board import ConnectionState

#: What each state is called in a status line.
_TEXT = {
    ConnectionState.CONNECTED: "Ready",
    ConnectionState.CONNECTING: "Connecting…",
    ConnectionState.RECONNECTING: "Reconnecting…",
    ConnectionState.DISCONNECTED: "Disconnected",
    ConnectionState.ERROR: "Error",
}

#: What each state is on the status strip's four-colour scale.
_STRIP = {
    ConnectionState.CONNECTED: "ok",
    ConnectionState.CONNECTING: "warn",
    ConnectionState.RECONNECTING: "warn",
    ConnectionState.DISCONNECTED: "error",
    ConnectionState.ERROR: "error",
}


def link_status_text(state: object) -> str:
    """The word for ``state`` in a status line.

    "Ready" rather than "Connected" because that is the word the hardware tree
    already used for a device that was good to go, and the tree is where most
    people read it.
    """
    return _TEXT.get(state, "Unknown")


def link_strip_state(state: object) -> str:
    """``state`` as one of the status strip's DEVICE_STATES.

    An unrecognised state renders neutral rather than green: a state nobody
    mapped is not evidence that anything is healthy.
    """
    return _STRIP.get(state, "unknown")


def link_is_usable(state: object) -> bool:
    """Whether a command sent right now has a link to travel over.

    Only CONNECTED. RECONNECTING is honest about trying, but a button pressed
    during one fails, and offering a press that is certain to fail is worse
    than grey.
    """
    return state is ConnectionState.CONNECTED
```

- [ ] **Step 4: Point the hardware tree at it**

In `src/glider/gui/panels/hardware_panel.py`, add to the imports:

```python
from glider.gui.device_status import link_status_text
```

Then replace the `QTreeWidgetItem` construction's third column. Change:

```python
                    device_item = QTreeWidgetItem(
                        [
                            getattr(device, "name", device_id),
                            f"{getattr(device, 'device_type', 'unknown')} ({pin_str})",
                            (
                                "Ready"
                                if getattr(device, "_initialized", False)
                                else "Not initialized"
                            ),
                        ]
                    )
```

to:

```python
                    # link_state, not _initialized. The old check answered
                    # "has this been set up", which a peripheral that walked
                    # out of range never stops answering yes to.
                    device_item = QTreeWidgetItem(
                        [
                            getattr(device, "name", device_id),
                            f"{getattr(device, 'device_type', 'unknown')} ({pin_str})",
                            link_status_text(getattr(device, "link_state", None)),
                        ]
                    )
```

- [ ] **Step 5: Point the Device Control panel at it**

In `src/glider/gui/panels/device_control_panel.py`, add to the imports:

```python
from glider.gui.device_status import link_is_usable, link_status_text
```

In `_on_device_selected`, replace:

```python
        device_type = getattr(device, "device_type", "unknown")
        board = getattr(device, "board", None)
        connected = board.is_connected if board else False
        initialized = getattr(device, "_initialized", False)

        status = "Connected" if connected else "Disconnected"
        if connected and initialized:
            status = "Ready"
        elif connected and not initialized:
            status = "Not initialized"

        self._device_status_label.setText(f"Status: {status} | Type: {device_type}")
```

with:

```python
        device_type = getattr(device, "device_type", "unknown")
        link = getattr(device, "link_state", None)
        self._device_status_label.setText(
            f"Status: {link_status_text(link)} | Type: {device_type}"
        )
```

Then in `_build_action_buttons`, in the `else:` branch that enables a
no-argument button, gate it on the link. Replace:

```python
            else:
                button.setToolTip(f"Run {name} on this device")
                button.clicked.connect(
                    lambda _checked=False, action=name: self._run_device_action(action)
                )
```

with:

```python
            elif not link_is_usable(getattr(device, "link_state", None)):
                # The link is down or coming back. A press now is certain to
                # fail, and a grey button says so before it is pressed.
                button.setEnabled(False)
                button.setToolTip(
                    f"{name} is unavailable while the device is "
                    f"{link_status_text(getattr(device, 'link_state', None)).lower()}"
                )
            else:
                button.setToolTip(f"Run {name} on this device")
                button.clicked.connect(
                    lambda _checked=False, action=name: self._run_device_action(action)
                )
```

- [ ] **Step 6: Run the new test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_device_link_status.py -q
```

Expected: 20 passed.

- [ ] **Step 7: Run the existing GUI suites, which assert the old strings**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui -q
```

`tests/unit/gui/test_device_action_buttons.py`'s `_Device` has `_initialized`
but no `link_state`, so `getattr(device, "link_state", None)` returns `None`,
`link_is_usable(None)` is False, and every action button is disabled — which
will fail its assertions. Fix the fixture rather than the production code: add
to that file's `_Device` class body, next to `_initialized = True`:

```python
    from glider.hal.base_board import ConnectionState as _CS

    link_state = _CS.CONNECTED
    owns_link = True
```

Prefer a plain module-level import at the top of the test file
(`from glider.hal.base_board import ConnectionState`) and `link_state =
ConnectionState.CONNECTED` in the class body — the nested-import form above is
only shown to be unambiguous about where it goes.

Also check `tests/unit/gui/test_open_restores_board_drivers.py` and any other
test asserting the literal string `"Not initialized"`:

```bash
grep -rn "Not initialized" tests/ plugins/
```

Update each to the new vocabulary (`"Disconnected"` for a device that was never
initialized on a live board).

- [ ] **Step 8: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Expected: all pass.

- [ ] **Step 9: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/gui/device_status.py src/glider/gui/panels/hardware_panel.py src/glider/gui/panels/device_control_panel.py tests/unit/gui/
git commit -m "fix(gui): read a device's link, not whether it was ever set up

The tree and the Device Control panel both showed _initialized, which is set
once and cleared only on shutdown -- so a peripheral that walked out of range
kept reading 'Ready'. Both now read link_state through one shared vocabulary
module, and action buttons grey out while the link is down."
```

---

### Task 7: The status strip shows peripherals

Spec §7, third surface, plus the drop notification.

**Files:**
- Modify: `src/glider/gui/main_window.py` — `_refresh_strip_devices` (~line 1019), `_connect_signals` (~line 1627), and a new slot beside `_on_hardware_connection_change` (~line 1767)
- Test: `tests/unit/gui/test_strip_device_chips.py` (create)

**Interfaces:**
- Consumes: `link_strip_state`, `link_status_text` (Task 6); `HardwareManager.on_device_connection_change` (Task 5); `BaseDevice.owns_link`, `BaseDevice.link_state` (Task 1).
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/test_strip_device_chips.py`:

```python
"""Peripherals on the status strip.

A BLE board is the host adapter: it goes green when bleak imports and stays
there. Its peripherals are the things that actually come and go, so the strip
-- the one piece of chrome that cannot be dismissed -- has to show them, or a
drop is invisible unless you happen to have the Hardware panel open.
"""

from types import SimpleNamespace

import pytest

from glider.gui.main_window import _device_chips
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


def _device(name, state, owns=True):
    return SimpleNamespace(
        name=name,
        owns_link=owns,
        link_state=state,
        device_type="Maimu",
    )


def test_a_pin_device_gets_no_chip():
    """It would only duplicate its board's dot."""
    devices = {"led": _device("LED", ConnectionState.CONNECTED, owns=False)}
    assert _device_chips(devices) == []


def test_a_peripheral_gets_a_chip():
    devices = {"d1": _device("Stimulator", ConnectionState.CONNECTED)}
    assert _device_chips(devices) == [("Stimulator", "ok", "Maimu · Ready")]


def test_a_dropped_peripheral_is_red():
    devices = {"d1": _device("Stimulator", ConnectionState.DISCONNECTED)}
    name, state, _detail = _device_chips(devices)[0]
    assert (name, state) == ("Stimulator", "error")


def test_reconnecting_is_amber_and_says_so_in_the_tooltip():
    """The four-colour mapping cannot tell 'retrying' from 'gone'. The tooltip can."""
    devices = {"d1": _device("Stimulator", ConnectionState.RECONNECTING)}
    _name, state, detail = _device_chips(devices)[0]
    assert state == "warn"
    assert "Reconnecting" in detail


def test_chips_keep_the_devices_order():
    devices = {
        "a": _device("Left", ConnectionState.CONNECTED),
        "b": _device("Right", ConnectionState.DISCONNECTED),
    }
    assert [name for name, _s, _d in _device_chips(devices)] == ["Left", "Right"]


def test_a_nameless_device_falls_back_to_its_id():
    devices = {"dev_7": SimpleNamespace(owns_link=True, link_state=ConnectionState.CONNECTED)}
    assert _device_chips(devices)[0][0] == "dev_7"


def test_an_awkward_device_does_not_take_the_strip_down():
    """A plugin device with a raising property must not blank the strip."""

    class _Awkward:
        owns_link = True

        @property
        def link_state(self):
            raise RuntimeError("plugin exploded")

    devices = {
        "bad": _Awkward(),
        "good": _device("Stimulator", ConnectionState.CONNECTED),
    }
    chips = _device_chips(devices)
    assert ("Stimulator", "ok", "Maimu · Ready") in chips
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_strip_device_chips.py -q
```

Expected: collection error — `ImportError: cannot import name '_device_chips'`.

- [ ] **Step 3: Add the chip builder**

In `src/glider/gui/main_window.py`, add to the imports:

```python
from glider.gui.device_status import link_status_text, link_strip_state
```

Then, next to `_board_detail` (module level, around line 137), add:

```python
def _device_chips(devices: dict) -> list[tuple[str, str, str]]:
    """Strip chips for the devices that hold a link of their own.

    Gated on ``owns_link`` because a pin-based device's state is its board's,
    and a second dot saying the same thing is noise on a rig with twenty LEDs.

    Never raises. A plugin device with an awkward ``link_state`` is skipped
    rather than allowed to blank the strip for everything beside it.
    """
    chips: list[tuple[str, str, str]] = []
    for device_id, device in devices.items():
        try:
            if not getattr(device, "owns_link", False):
                continue
            state = device.link_state
            name = getattr(device, "name", None) or device_id
            device_type = getattr(device, "device_type", "") or "device"
            detail = f"{device_type} · {link_status_text(state)}"
            chips.append((str(name), link_strip_state(state), detail))
        except Exception:
            logger.warning("Could not read link state from device %s", device_id, exc_info=True)
    return chips
```

- [ ] **Step 4: Put the chips on the strip**

In `_refresh_strip_devices`, replace the final `strip.set_devices(devices)` line
with:

```python
        # Peripherals after boards: the adapter, then what is attached to it.
        devices.extend(_device_chips(self._core.hardware_manager.devices))
        strip.set_devices(devices)
```

and extend the method's docstring with:

```
    Peripherals that hold their own link (``owns_link``) get a chip of their
    own after the boards. A BLE board is the host adapter -- green from the
    moment bleak imports -- so without this the strip has nothing to say about
    the thing that actually came and went.
```

- [ ] **Step 5: Wire the device channel**

In `_connect_signals`, beside the existing board wiring, add:

```python
        self._device_link_changed.connect(self._on_device_link_change)
        self._core.hardware_manager.on_device_connection_change(
            lambda device_id, state: self._device_link_changed.emit(device_id, state)
        )
```

Declare the signal in the `MainWindow` class body immediately after
`_hardware_connection_changed = pyqtSignal(str, object)` (line 254):

```python
    # Marshals a device link change onto the Qt thread. Devices fire from a
    # bleak callback and from a background reconnect task, neither of which is
    # on this thread's call stack.
    _device_link_changed = pyqtSignal(str, object)
```

Then add the slot immediately after `_on_hardware_connection_change`:

```python
    @pyqtSlot(str, object)
    def _on_device_link_change(self, device_id: str, state) -> None:
        """A peripheral's own link moved.

        Repaints both readouts, and says so out loud if it dropped while an
        experiment was recording. Deliberately does *not* pause the run: a
        ten-second BLE dropout should not end a two-hour session, and
        _show_hardware_disconnection_dialog is modal and stays reserved for a
        board going away.
        """
        self._refresh_hardware_readouts()

        if state not in (BoardConnectionState.DISCONNECTED, BoardConnectionState.ERROR):
            return

        device = self._core.hardware_manager.get_device(device_id)
        label = getattr(device, "name", device_id) if device is not None else device_id
        logger.warning("Device %s link %s", label, link_status_text(state).lower())

        if not hasattr(self._core, "state"):
            return
        from glider.core.glider_core import SessionState

        if self._core.state != SessionState.RUNNING:
            return
        self._notify_user(
            f"{label} disconnected",
            f"{label} lost its connection during the run. GLIDER is retrying; "
            "the experiment has not been paused.",
            level="warning",
        )
```

- [ ] **Step 6: Start the supervisor**

`MainWindow.__init__` calls `_connect_signals()` at line 398, which runs
**before** qasync starts the event loop — so `asyncio.create_task` there raises
`RuntimeError: no running event loop`. Two call sites, one guarded and one not.

In `main_window.py`, at the very end of `_connect_signals`, add:

```python
        # Backstop for a bleak disconnect callback that never fires (spec §5).
        # __init__ runs before qasync starts the loop, so this normally raises
        # and the hardware-connect path below starts it instead. Attempted here
        # anyway for the case where the window is built inside a running loop.
        try:
            self._core.hardware_manager.start_link_supervisor()
        except RuntimeError:
            logger.debug("Link supervisor deferred: no running event loop yet")
```

In `src/glider/gui/panels/hardware_panel.py`, in `_connect_hardware_async`
(line 1350), add the call after `self.refresh_tree()` inside the `try`:

```python
            results = await self._hardware_manager.connect_all()
            self.refresh_tree()
            # On the loop by now, so this is the call that actually takes.
            # Idempotent: a second connect does not start a second sweep.
            self._hardware_manager.start_link_supervisor()
```

- [ ] **Step 7: Run the new test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_strip_device_chips.py -q
```

Expected: 7 passed.

- [ ] **Step 8: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Expected: all pass.

- [ ] **Step 9: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/gui/main_window.py src/glider/gui/panels/hardware_panel.py tests/unit/gui/test_strip_device_chips.py
git commit -m "feat(gui): put BLE peripherals on the status strip

A BLE board is the host adapter and is green from the moment bleak imports,
so the strip had nothing to say about the thing that actually came and went.
Devices that own a link now get their own chip, and a drop during a run
raises a warning notification -- without pausing the run."
```

---

### Task 8: Devices declare their actions' arguments

Spec §8. Pure HAL plus the Maimu declaration. No GUI.

**Files:**
- Modify: `src/glider/hal/base_device.py`
- Modify: `plugins/glider-maimu/src/glider_maimu/device.py`
- Test: `tests/unit/hal/test_action_args_schema.py` (create)
- Test: `plugins/glider-maimu/tests/test_pulse_schema.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `BaseDevice.ACTION_ARGS_SCHEMA: dict[str, list[dict]] = {}` — class attribute
  - `BaseDevice.action_args_schema(action: str) -> list[dict]`
  - `BaseDevice.action_needs_args(action: str) -> bool`

  Tasks 9 and 10 consume both methods.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/hal/test_action_args_schema.py`:

```python
"""A device declaring what its actions' arguments are.

Until now an action with required arguments was dead on both control
surfaces -- disabled in the Builder's Device Control panel, and a TypeError in
the Runner's manual controls, which called it with none. A declared schema is
what lets either one render fields and pass real values.

Same field vocabulary as SETTINGS_SCHEMA, so schema_form renders it with no
new widget code.
"""

import pytest

from glider.hal.base_device import BaseDevice, DeviceConfig


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


class _Device(BaseDevice):
    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {"key": "period_ms", "label": "Period (ms)", "type": "int", "default": 500},
            {"key": "duration_s", "label": "Duration (s)", "type": "int", "default": 10},
        ],
    }

    @property
    def device_type(self):
        return "Fake"

    @property
    def actions(self):
        return {
            "on": self.on,
            "pulse": self.pulse,
            "fade": self.fade,
            "write": self.write,
        }

    async def on(self):
        pass

    async def pulse(self, period_ms, duration_s):
        pass

    async def fade(self, level=5):
        pass

    async def write(self, *args):
        pass

    async def initialize(self):
        self._initialized = True

    async def shutdown(self):
        self._initialized = False


def _device():
    return _Device(_FakeBoard(), DeviceConfig())


def test_the_default_schema_is_empty():
    assert BaseDevice.ACTION_ARGS_SCHEMA == {}


def test_a_declared_action_returns_its_fields():
    schema = _device().action_args_schema("pulse")
    assert [f["key"] for f in schema] == ["period_ms", "duration_s"]


def test_an_undeclared_action_returns_an_empty_list():
    assert _device().action_args_schema("on") == []


def test_an_unknown_action_returns_an_empty_list():
    assert _device().action_args_schema("nonsense") == []


def test_a_no_argument_action_needs_none():
    assert _device().action_needs_args("on") is False


def test_a_required_argument_action_needs_them():
    assert _device().action_needs_args("pulse") is True


def test_a_defaulted_argument_does_not_count_as_required():
    """fade(level=5) is pressable as-is; it must not be greyed out."""
    assert _device().action_needs_args("fade") is False


def test_varargs_do_not_count_as_required():
    """write(*args) validates its own emptiness and reports a real error."""
    assert _device().action_needs_args("write") is False


def test_an_unknown_action_needs_nothing():
    assert _device().action_needs_args("nonsense") is False


def test_declared_keys_match_the_real_parameters():
    """The schema is passed positionally, so a typo would swap the arguments."""
    import inspect

    device = _device()
    for action, schema in type(device).ACTION_ARGS_SCHEMA.items():
        params = [
            p.name
            for p in inspect.signature(device.actions[action]).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert [f["key"] for f in schema] == params[: len(schema)]
```

Create `plugins/glider-maimu/tests/test_pulse_schema.py`:

```python
"""The Maimu's pulse arguments, as the control panels will render them.

period_ms is a period in milliseconds, not a frequency, and the firmware
atoi()s both fields -- so the declared bounds are the same whole-number,
at-least-1 contract MaimuDevice._whole_number enforces at call time.
"""

import inspect

from glider.hal.base_device import DeviceConfig
from glider_maimu.device import MaimuDevice


class _FakeBoard:
    def __init__(self):
        self.id = "fake_board"
        self.is_connected = True


def _device():
    return MaimuDevice(_FakeBoard(), DeviceConfig(), name="maimu")


def test_pulse_declares_both_arguments():
    assert [f["key"] for f in _device().action_args_schema("pulse")] == [
        "period_ms",
        "duration_s",
    ]


def test_pulse_is_the_only_declared_action():
    """on/off/write need nothing; declaring them would only add empty forms."""
    assert set(MaimuDevice.ACTION_ARGS_SCHEMA) == {"pulse"}


def test_the_declared_order_matches_the_signature():
    """The panels pass these positionally; a swap would invert period and duration."""
    params = list(inspect.signature(MaimuDevice.pulse).parameters)[1:]
    assert [f["key"] for f in _device().action_args_schema("pulse")] == params


def test_the_defaults_are_the_node_s_defaults():
    """A researcher moving between the node and the panel should see one number."""
    from glider_maimu.node import DEFAULT_DURATION_S, DEFAULT_PERIOD_MS

    fields = {f["key"]: f for f in _device().action_args_schema("pulse")}
    assert fields["period_ms"]["default"] == DEFAULT_PERIOD_MS
    assert fields["duration_s"]["default"] == DEFAULT_DURATION_S


def test_the_bounds_reject_zero():
    """_whole_number raises below 1; the spin box should not offer it."""
    fields = {f["key"]: f for f in _device().action_args_schema("pulse")}
    assert fields["period_ms"]["min"] == 1
    assert fields["duration_s"]["min"] == 1


def test_every_field_is_a_whole_number():
    """The firmware atoi()s both, so a float widget would silently truncate."""
    assert all(f["type"] == "int" for f in _device().action_args_schema("pulse"))
```

- [ ] **Step 2: Run them and watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_action_args_schema.py plugins/glider-maimu/tests/test_pulse_schema.py -q
```

Expected: FAIL — `AttributeError: type object 'BaseDevice' has no attribute 'ACTION_ARGS_SCHEMA'`.

- [ ] **Step 3: Add the schema to `BaseDevice`**

In `src/glider/hal/base_device.py`, add a class attribute in the `BaseDevice`
class body, immediately before `def __init__`:

```python
    #: Declares an action's arguments, keyed by action name.
    #:
    #: Same field vocabulary as SETTINGS_SCHEMA (key / label / type / default /
    #: min / max / help), so glider.gui.widgets.schema_form renders it with no
    #: new widget code. Fields are passed to the action POSITIONALLY in the
    #: order declared, so each ``key`` must name the real parameter and the
    #: order must match the signature.
    #:
    #: Declaring this is what makes an argument-taking action pressable: both
    #: control surfaces will otherwise refuse it, one by greying the button and
    #: one by not offering it at all.
    ACTION_ARGS_SCHEMA: dict[str, list[dict[str, Any]]] = {}
```

Then, after the `actions` abstract property, add:

```python
    def action_args_schema(self, action_name: str) -> list[dict[str, Any]]:
        """The declared argument fields for ``action_name`` (empty if none)."""
        return list(type(self).ACTION_ARGS_SCHEMA.get(action_name, ()))

    def action_needs_args(self, action_name: str) -> bool:
        """Whether ``action_name`` cannot be called with no arguments.

        Only *required* positional parameters count. A defaulted one
        (``fade(level=5)``) is pressable as-is, and ``*args`` is not a
        requirement -- an action taking it validates its own emptiness and
        reports a better error than a greyed-out button could.

        An unintrospectable callable is reported as needing nothing: offering
        it and letting it report its own failure beats hiding a working
        action.
        """
        import inspect

        try:
            func = self.actions[action_name]
        except (KeyError, TypeError):
            return False
        try:
            parameters = inspect.signature(func).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            param.default is inspect.Parameter.empty
            and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            for param in parameters
        )
```

- [ ] **Step 4: Declare the Maimu's pulse arguments**

In `plugins/glider-maimu/src/glider_maimu/device.py`, add after the
`SETTINGS_SCHEMA` list, inside the class body:

```python
    # What the control panels put in front of a pulse button. The bounds are
    # the same contract _whole_number enforces at call time -- whole numbers,
    # at least 1 -- because the firmware atoi()s both fields, so a spin box
    # that offered 0 or a fraction would only produce a legible error later.
    # The defaults match MaimuNode's, so the number a researcher sees does not
    # change when they move between the graph and the panel.
    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {
                "key": "period_ms",
                "label": "Period (ms)",
                "type": "int",
                "default": 500,
                "min": 1,
                "max": 3_600_000,
                "help": (
                    "On/off toggle period in milliseconds -- a period, not a "
                    "frequency. 500 ms toggles about once a second."
                ),
            },
            {
                "key": "duration_s",
                "label": "Duration (s)",
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 86_400,
                "help": "How long the train runs. The firmware stops on its own.",
            },
        ],
    }
```

- [ ] **Step 5: Run them and watch them pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/hal/test_action_args_schema.py plugins/glider-maimu/tests/test_pulse_schema.py -q
```

Expected: 16 passed.

- [ ] **Step 6: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Expected: all pass. Nothing reads the schema yet.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/hal/base_device.py plugins/glider-maimu/src/glider_maimu/device.py tests/unit/hal/test_action_args_schema.py plugins/glider-maimu/tests/test_pulse_schema.py
git commit -m "feat(hal): let a device declare its actions' arguments

ACTION_ARGS_SCHEMA, in the same field vocabulary SETTINGS_SCHEMA uses, so
schema_form renders it unchanged. Maimu declares pulse's period and
duration. Nothing reads it yet; the two control surfaces follow."
```

---

### Task 9: The Builder can pulse

Spec §8.1.

**Files:**
- Modify: `src/glider/gui/panels/device_control_panel.py` — `_setup_ui` (the `_actions_widget` block, ~line 126), `_clear_action_buttons`, `_build_action_buttons`, `_run_device_action`
- Test: `tests/unit/gui/test_device_action_args.py` (create)

**Interfaces:**
- Consumes: `BaseDevice.action_args_schema`, `BaseDevice.action_needs_args` (Task 8); `link_is_usable` (Task 6); `build_schema_widgets`, `read_schema_widget` (existing, `glider.gui.widgets.schema_form`).
- Produces:
  - `DeviceControlPanel._action_arg_widgets: dict[str, dict[str, tuple]]` — action name → `{field key: (widget, ftype)}`
  - `DeviceControlPanel._action_args(action: str) -> list` — the current field values, in schema order

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/test_device_action_args.py`:

```python
"""Pressing Pulse in the Builder's Device Control panel.

pulse(period_ms, duration_s) has two required arguments and the panel had
nowhere to put them, so _build_action_buttons rendered it disabled with a
tooltip pointing at a node. A device that declares ACTION_ARGS_SCHEMA now gets
real fields and a live button.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton, QSpinBox

from glider.gui.panels.device_control_panel import DeviceControlPanel
from glider.hal.base_board import ConnectionState

pytestmark = pytest.mark.usefixtures("qtbot")


class _Device:
    device_type = "Maimu"
    name = "Stimulator"
    _initialized = True
    owns_link = True
    link_state = ConnectionState.CONNECTED

    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {"key": "period_ms", "label": "Period (ms)", "type": "int",
             "default": 500, "min": 1, "max": 3_600_000},
            {"key": "duration_s", "label": "Duration (s)", "type": "int",
             "default": 10, "min": 1, "max": 86_400},
        ],
    }

    def __init__(self):
        self.calls = []
        self.board = SimpleNamespace(is_connected=True)

    async def on(self):
        self.calls.append(("on", ()))

    async def pulse(self, period_ms, duration_s):
        self.calls.append(("pulse", (period_ms, duration_s)))

    async def fade(self, level):
        self.calls.append(("fade", (level,)))

    async def execute_action(self, name, *args):
        return await self.actions[name](*args)

    @property
    def actions(self):
        return {"on": self.on, "pulse": self.pulse, "fade": self.fade}

    def action_args_schema(self, action):
        return list(self.ACTION_ARGS_SCHEMA.get(action, ()))

    def action_needs_args(self, action):
        import inspect

        func = self.actions.get(action)
        if func is None:
            return False
        return any(
            p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in inspect.signature(func).parameters.values()
        )


def _panel(qtbot, device):
    def _runner(coro):
        # These are sync tests, so there is no loop to schedule onto. Run the
        # coroutine to completion inline; asyncio.run rather than
        # get_event_loop, which is deprecated outside a running loop.
        import asyncio

        asyncio.run(coro)

    manager = SimpleNamespace(
        devices={"dev1": device},
        get_device=lambda dev_id: device if dev_id == "dev1" else None,
    )
    panel = DeviceControlPanel(manager, _runner)
    qtbot.addWidget(panel)
    panel.refresh_devices()
    panel._device_combo.setCurrentIndex(1)
    return panel


def _button(panel, label):
    found = panel._actions_widget.findChildren(QPushButton)
    for btn in found:
        if btn.text() == label:
            return btn
    raise AssertionError(f"no {label!r} button; found {[b.text() for b in found]}")


def test_a_declared_action_is_pressable(qtbot):
    panel = _panel(qtbot, _Device())
    assert _button(panel, "pulse").isEnabled()


def test_its_fields_are_rendered_with_their_defaults(qtbot):
    panel = _panel(qtbot, _Device())
    widgets = panel._action_arg_widgets["pulse"]
    assert widgets["period_ms"][0].value() == 500
    assert widgets["duration_s"][0].value() == 10


def test_the_fields_are_spin_boxes_with_the_declared_bounds(qtbot):
    panel = _panel(qtbot, _Device())
    period = panel._action_arg_widgets["pulse"]["period_ms"][0]
    assert isinstance(period, QSpinBox)
    assert period.minimum() == 1
    assert period.maximum() == 3_600_000


def test_args_are_read_in_schema_order(qtbot):
    """Swapped, pulse would run a 10 ms train for 500 seconds."""
    panel = _panel(qtbot, _Device())
    panel._action_arg_widgets["pulse"]["period_ms"][0].setValue(250)
    panel._action_arg_widgets["pulse"]["duration_s"][0].setValue(30)
    assert panel._action_args("pulse") == [250, 30]


def test_pressing_it_calls_the_action_with_both_values(qtbot):
    device = _Device()
    panel = _panel(qtbot, device)
    panel._action_arg_widgets["pulse"]["period_ms"][0].setValue(250)
    panel._action_arg_widgets["pulse"]["duration_s"][0].setValue(30)
    _button(panel, "pulse").click()
    assert device.calls == [("pulse", (250, 30))]


def test_an_undeclared_argument_action_stays_disabled(qtbot):
    """fade(level) declares nothing; the old tooltip is still the right answer."""
    panel = _panel(qtbot, _Device())
    fade = _button(panel, "fade")
    assert not fade.isEnabled()
    assert "Device Action node" in fade.toolTip()


def test_a_no_argument_action_gets_no_fields(qtbot):
    panel = _panel(qtbot, _Device())
    assert "on" not in panel._action_arg_widgets


def test_switching_device_clears_the_fields(qtbot):
    """Stale widgets from the previous device would be read on the next press."""
    panel = _panel(qtbot, _Device())
    assert "pulse" in panel._action_arg_widgets
    panel._device_combo.setCurrentIndex(0)  # "-- Select Device --"
    assert panel._action_arg_widgets == {}


def test_a_down_link_disables_it_too(qtbot):
    device = _Device()
    device.link_state = ConnectionState.RECONNECTING
    panel = _panel(qtbot, device)
    assert not _button(panel, "pulse").isEnabled()
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_device_action_args.py -q
```

Expected: FAIL — `AttributeError: 'DeviceControlPanel' object has no attribute '_action_arg_widgets'`.

- [ ] **Step 3: Give the panel a place to put the fields**

In `src/glider/gui/panels/device_control_panel.py`:

**(a)** Add to the imports:

```python
from PyQt6.QtWidgets import QFormLayout
```

(add `QFormLayout` to the existing `from PyQt6.QtWidgets import (...)` block,
keeping it alphabetical) and:

```python
from glider.gui.widgets.schema_form import build_schema_widgets, read_schema_widget
```

**(b)** In `__init__`, before `self._setup_ui()`, add:

```python
        # action name -> {field key: (widget, ftype)} for the currently
        # selected device. Cleared with the buttons; a stale widget from the
        # previous device would be read on the next press.
        self._action_arg_widgets: dict[str, dict[str, tuple]] = {}
```

**(c)** In `_setup_ui`, immediately after the block that creates
`self._actions_widget` and adds it to `self._control_group_layout`, add a second
container for the fields:

```python
        # Argument fields for the actions that declare them. A row of buttons
        # has nowhere to put a period and a duration; this is where they go.
        self._action_args_widget = QWidget()
        self._action_args_layout = QFormLayout(self._action_args_widget)
        self._action_args_layout.setContentsMargins(0, 4, 0, 0)
        self._control_group_layout.addWidget(self._action_args_widget)
```

**(d)** Rewrite `_clear_action_buttons` to clear both:

```python
    def _clear_action_buttons(self) -> None:
        self._action_arg_widgets.clear()
        for layout in (self._actions_layout, self._action_args_layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    # setParent(None) first: deleteLater only *queues*
                    # destruction, so without this the previous device's
                    # buttons remain children -- and remain clickable -- until
                    # the event loop turns.
                    widget.setParent(None)
                    widget.deleteLater()
```

Note `QFormLayout.takeAt` returns label and field as separate items, so the
loop handles both.

**(e)** Replace `_build_action_buttons` entirely. Task 6 already added a link
gate to its `else:` branch; this version supersedes that edit rather than
adding to it, so paste over the whole method.

```python
    def _build_action_buttons(self, device) -> bool:
        """One button per action the device declares, plus any argument fields.

        This is how a Maimu gets its ``on`` and ``off`` -- which is how you
        tell which of six stimulators on the bench is the one you just added --
        and how any plugin device gets manual control without core knowing its
        type.

        An action that takes arguments is pressable when the device declares
        them in ``ACTION_ARGS_SCHEMA``: the fields render beneath the button
        row and the values are passed positionally in the declared order. An
        action that takes arguments and declares none stays disabled, because
        the panel genuinely has nothing to send; the tooltip says where to
        drive it from instead.

        Buttons are dead while the link is down. Offering a press that is
        certain to fail is worse than greying it.

        Returns whether any button was built. Never raises: a device with an
        awkward ``actions`` property must not take the panel down mid-session.
        """
        self._clear_action_buttons()
        try:
            actions = dict(getattr(device, "actions", {}) or {})
        except Exception:
            logger.warning("Could not read actions from %s", device, exc_info=True)
            return False

        usable = link_is_usable(getattr(device, "link_state", None))
        built = False
        for name in actions:
            needs_args = self._needs_args(device, name)
            schema = self._args_schema(device, name) if needs_args else []

            button = QPushButton(name)
            button.setMinimumHeight(32)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            if needs_args and not schema:
                button.setEnabled(False)
                button.setToolTip(
                    f"{name} takes arguments this device does not declare; "
                    "drive it from a Device Action node or a node for this device."
                )
            elif not usable:
                state_word = link_status_text(getattr(device, "link_state", None)).lower()
                button.setEnabled(False)
                button.setToolTip(f"{name} is unavailable while the device is {state_word}")
            else:
                button.setToolTip(f"Run {name} on this device")
                button.clicked.connect(
                    lambda _checked=False, action=name: self._run_device_action(action)
                )

            if schema:
                fields: dict[str, tuple] = {}
                build_schema_widgets(self._action_args_layout, schema, fields)
                for widget, _ftype in fields.values():
                    widget.setEnabled(button.isEnabled())
                self._action_arg_widgets[name] = fields

            self._actions_layout.addWidget(button)
            built = True

        self._action_args_widget.setVisible(bool(self._action_arg_widgets))
        return built

    @staticmethod
    def _needs_args(device, action: str) -> bool:
        """Whether ``action`` cannot be called with no arguments.

        Asks the device, which is the thing that knows. Falls back to
        signature introspection for a device predating ``action_needs_args``
        (a plugin pinned to an older core), so an unknown device is never
        offered a press that would raise TypeError.
        """
        asker = getattr(device, "action_needs_args", None)
        if callable(asker):
            try:
                return bool(asker(action))
            except Exception:
                logger.debug("action_needs_args failed for %s", action, exc_info=True)
        import inspect

        try:
            func = device.actions[action]
            parameters = inspect.signature(func).parameters.values()
        except (KeyError, TypeError, ValueError):
            return False
        return any(
            param.default is inspect.Parameter.empty
            and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
            for param in parameters
        )

    @staticmethod
    def _args_schema(device, action: str) -> list[dict]:
        """The device's declared argument fields for ``action`` (empty if none)."""
        asker = getattr(device, "action_args_schema", None)
        if not callable(asker):
            return []
        try:
            return list(asker(action) or [])
        except Exception:
            logger.debug("action_args_schema failed for %s", action, exc_info=True)
            return []

    def _action_args(self, action: str) -> list:
        """Current values of ``action``'s argument fields, in declared order.

        Order is the schema's, because they are passed positionally: swapped,
        a pulse would run a 10 ms train for 500 seconds.
        """
        fields = self._action_arg_widgets.get(action, {})
        return [read_schema_widget(widget, ftype) for widget, ftype in fields.values()]
```

`dict` preserves insertion order, and `build_schema_widgets` writes `out[key]`
in schema order, so `_action_args` is in declared order by construction.

**(f)** Make `_run_device_action` pass them:

```python
    def _run_device_action(self, action: str) -> None:
        """Run one action on the selected device, with any declared arguments."""
        device = self._get_selected_device()
        if device is None:
            return
        args = self._action_args(action)

        async def _run():
            try:
                await device.execute_action(action, *args)
                self._device_status_label.setText(f"Status: ran {action!r}")
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                logger.exception("Device action %s failed", action)
                self._device_status_label.setText(f"Status: {action!r} failed - {exc}")

        self._run_async(_run())
```

**(g)** In `_on_device_selected`, the early-return branches (`device_id is None`
and `device is None`) return before `_build_action_buttons` runs, leaving the
previous device's fields in place. Add `self._clear_action_buttons()` to both,
immediately before their `return`.

- [ ] **Step 4: Run the test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_device_action_args.py -q
```

Expected: 9 passed.

- [ ] **Step 5: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Expected: all pass. `tests/unit/gui/test_device_action_buttons.py`'s `_Device`
declares no `ACTION_ARGS_SCHEMA`, so its `pulse` still renders disabled — but
its tooltip text changed. If that test asserts the old wording, update the
assertion to match the new message.

- [ ] **Step 6: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/gui/panels/device_control_panel.py tests/unit/gui/
git commit -m "feat(gui): make argument-taking device actions pressable in the Builder

pulse(period_ms, duration_s) rendered as a disabled button with a tooltip
pointing at a node, because the panel had nowhere to put two numbers. A
device that declares ACTION_ARGS_SCHEMA now gets real fields and a live
button; one that declares nothing keeps the old tooltip."
```

---

### Task 10: The Runner can pulse, and stops crashing

Spec §8.2. The `TypeError` fix.

**Files:**
- Modify: `src/glider/gui/runner/device_controls.py` — `_controls_for`, `_make_device_section`, a new `_make_action_args`, and a new signal
- Modify: `src/glider/gui/main_window.py` — two `action_call_requested` connections (beside the existing `action_fire_requested` ones at ~line 617 and ~line 740)
- Test: `tests/unit/gui/test_runner_action_args.py` (create)

**Interfaces:**
- Consumes: `BaseDevice.action_args_schema`, `BaseDevice.action_needs_args` (Task 8); `build_schema_widgets`, `read_schema_widget` (existing).
- Produces:
  - `RunnerDeviceControls.action_call_requested = pyqtSignal(str, str, object)` — `(device_id, action, [args])`
  - `RunnerDeviceControls._make_action_args(dev_id, action, schema) -> QWidget`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/test_runner_action_args.py`:

```python
"""Argument-taking actions in the Runner's touch controls.

The bug this closes is a crash, not a cosmetic one: pulse has no value_spec,
so _controls_for classified it as a plain fire button, and pressing it called
execute_action("pulse") with no arguments -- TypeError, every time.
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QPushButton, QSpinBox

from glider.gui.runner.device_controls import RunnerDeviceControls

pytestmark = pytest.mark.usefixtures("qtbot")


class _Device:
    device_type = "Maimu"
    name = "Stimulator"

    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {"key": "period_ms", "label": "Period (ms)", "type": "int",
             "default": 500, "min": 1, "max": 3_600_000},
            {"key": "duration_s", "label": "Duration (s)", "type": "int",
             "default": 10, "min": 1, "max": 86_400},
        ],
    }

    async def on(self):
        pass

    async def off(self):
        pass

    async def pulse(self, period_ms, duration_s):
        pass

    async def fade(self, level):
        pass

    @property
    def actions(self):
        return {"on": self.on, "off": self.off, "pulse": self.pulse, "fade": self.fade}

    def value_spec(self, action):
        return None

    def action_args_schema(self, action):
        return list(self.ACTION_ARGS_SCHEMA.get(action, ()))

    def action_needs_args(self, action):
        import inspect

        func = self.actions.get(action)
        if func is None:
            return False
        return any(
            p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in inspect.signature(func).parameters.values()
        )


def _controls(qtbot, device):
    manager = SimpleNamespace(devices={"dev1": device})
    widget = RunnerDeviceControls(manager, session_fn=lambda: None)
    qtbot.addWidget(widget)
    widget.refresh()
    return widget


def test_a_declared_action_gets_an_args_control(qtbot):
    device = _Device()
    kinds = {action: kind for kind, action, _spec in _controls(qtbot, device)._controls_for(device)}
    assert kinds["pulse"] == "action_args"


def test_a_no_argument_action_is_still_a_button(qtbot):
    device = _Device()
    kinds = {action: kind for kind, action, _spec in _controls(qtbot, device)._controls_for(device)}
    assert kinds["on"] == "button"


def test_an_undeclared_argument_action_is_not_a_bare_button(qtbot):
    """This is the crash: a bare fire button called fade() with no level."""
    device = _Device()
    kinds = {action: kind for kind, action, _spec in _controls(qtbot, device)._controls_for(device)}
    assert kinds.get("fade") != "button"


def test_the_fields_render_with_their_defaults(qtbot):
    widget = _controls(qtbot, _Device())
    values = sorted(s.value() for s in widget.findChildren(QSpinBox))
    assert values == [10, 500]


def test_pressing_it_emits_both_values(qtbot):
    device = _Device()
    widget = _controls(qtbot, device)
    fields = widget._widgets[("dev1", "pulse")]["args"]
    fields["period_ms"][0].setValue(250)
    fields["duration_s"][0].setValue(30)

    emitted = []
    widget.action_call_requested.connect(
        lambda dev_id, action, args: emitted.append((dev_id, action, list(args)))
    )
    button = widget._widgets[("dev1", "pulse")]["button"]
    button.click()
    assert emitted == [("dev1", "pulse", [250, 30])]


def test_the_args_are_in_schema_order(qtbot):
    """Swapped, a pulse runs a 10 ms train for 500 seconds."""
    device = _Device()
    widget = _controls(qtbot, device)
    emitted = []
    widget.action_call_requested.connect(
        lambda dev_id, action, args: emitted.append(list(args))
    )
    widget._widgets[("dev1", "pulse")]["button"].click()
    assert emitted == [[500, 10]]


def test_a_failure_does_not_try_to_revert_a_slider(qtbot):
    """The args control has no committed value to snap back to."""
    device = _Device()
    widget = _controls(qtbot, device)
    widget.on_action_failed("dev1", "pulse", "pulse failed: link down")
    assert widget._status.isVisible()
```

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_runner_action_args.py -q
```

Expected: FAIL — `assert 'button' == 'action_args'`.

- [ ] **Step 3: Add the control kind**

In `src/glider/gui/runner/device_controls.py`:

**(a)** Add to the imports:

```python
from glider.gui.widgets.schema_form import build_schema_widgets, read_schema_widget
```

and add `QFormLayout` to the existing `from PyQt6.QtWidgets import (...)` block.

**(b)** Beside the other signals in the class body, add:

```python
    # (device_id, action_name, [positional args]) -- an action driven with the
    # values from its declared argument fields.
    action_call_requested = pyqtSignal(str, str, object)
```

**(c)** In `_controls_for`, replace the final `else:` branch. Change:

```python
            elif action in _SWITCH_REDUNDANT and has_switch:
                continue
            else:
                controls.append(("button", action, None))
        return controls
```

to:

```python
            elif action in _SWITCH_REDUNDANT and has_switch:
                continue
            elif _needs_args(device, action):
                # An action with required arguments must never become a plain
                # fire button: that button called execute_action(action) with
                # none of them, which is a TypeError every time it is pressed.
                schema = _args_schema(device, action)
                if schema:
                    controls.append(("action_args", action, schema))
                # No declared schema means there is nothing to send. Omitted
                # rather than shown disabled: the runner is a touchscreen with
                # no tooltips, so a dead control explains nothing.
            else:
                controls.append(("button", action, None))
        return controls
```

**(d)** Add the two module-level helpers next to `_title` at the bottom of the file:

```python
def _needs_args(device, action: str) -> bool:
    """Whether ``action`` cannot be called with no arguments.

    Asks the device first; falls back to signature introspection for a plugin
    device predating ``action_needs_args``, so an unknown device never gets a
    button that raises TypeError on the first press.
    """
    asker = getattr(device, "action_needs_args", None)
    if callable(asker):
        try:
            return bool(asker(action))
        except Exception:
            pass
    import inspect

    try:
        parameters = inspect.signature(device.actions[action]).parameters.values()
    except (KeyError, TypeError, ValueError):
        return False
    return any(
        param.default is inspect.Parameter.empty
        and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        for param in parameters
    )


def _args_schema(device, action: str) -> list[dict]:
    """The device's declared argument fields for ``action`` (empty if none)."""
    asker = getattr(device, "action_args_schema", None)
    if not callable(asker):
        return []
    try:
        return list(asker(action) or [])
    except Exception:
        return []
```

**(e)** In `_make_device_section`, add the builder to the dispatch dict:

```python
            builder = {
                "switch": self._make_switch,
                "slider": self._make_slider,
                "read": self._make_read,
                "button": self._make_button,
                "action_args": self._make_action_args,
            }[kind]
```

**(f)** After `_make_button`, add:

```python
    def _make_action_args(self, dev_id: str, action: str, schema) -> QWidget:
        """Labelled fields plus a fire button, for an action with arguments.

        Stored under an ``args`` key rather than ``spin``/``slider`` on
        purpose: those keys drive the optimistic-revert path in
        on_action_failed, and there is no single committed value to snap an
        argument form back to.
        """
        block = QWidget()
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._labeled(action))

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        fields: dict[str, tuple] = {}
        build_schema_widgets(form, schema, fields)
        for widget, _ftype in fields.values():
            widget.setMinimumHeight(_CONTROL_MIN_HEIGHT)
        layout.addLayout(form)

        btn = QPushButton(_title(action))
        btn.setMinimumHeight(_CONTROL_MIN_HEIGHT)
        btn.clicked.connect(
            lambda _=False: self.action_call_requested.emit(
                dev_id,
                action,
                [read_schema_widget(w, t) for w, t in fields.values()],
            )
        )
        layout.addWidget(btn)

        self._widgets[(dev_id, action)] = {"args": fields, "button": btn}
        return block
```

- [ ] **Step 4: Wire the signal in `main_window`**

`_drive_action(self, dev_id, action, *value)` already takes varargs, so it
needs no change. Beside **each** of the two existing
`action_fire_requested.connect(...)` blocks (one in the dashboard view, one in
the runner view), add:

```python
        self._runner_device_controls.action_call_requested.connect(
            lambda dev_id, action, args: self._run_async(
                self._drive_action(dev_id, action, *args)
            )
        )
```

- [ ] **Step 5: Run the test and watch it pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/gui/test_runner_action_args.py -q
```

Expected: 7 passed.

- [ ] **Step 6: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Expected: all pass.

- [ ] **Step 7: Lint, format, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

```bash
git add src/glider/gui/runner/device_controls.py src/glider/gui/main_window.py tests/unit/gui/test_runner_action_args.py
git commit -m "fix(gui): stop the Runner calling an action without its arguments

pulse has no value_spec, so _controls_for made it a plain fire button and
pressing it called execute_action('pulse') with nothing -- TypeError, every
press. An action with required arguments now renders its declared fields and
fires with them, and one that declares none is not offered at all."
```

---

### Task 11: Document it

The reference page describes the Maimu and the BLE device to researchers; both
gained behaviour a user can see.

**Files:**
- Modify: `docs-site/reference/devices.md`
- Modify: `CHANGELOG.md`
- Modify: `plugins/glider-maimu/README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing.

- [ ] **Step 1: Read what is there now**

```bash
grep -n "Maimu" -A 30 docs-site/reference/devices.md | head -60
```

```bash
head -40 CHANGELOG.md
```

- [ ] **Step 2: Update the device reference**

In `docs-site/reference/devices.md`, in the BLE and Maimu sections, add a
subsection matching the surrounding heading level:

```markdown
### When the link drops

A BLE peripheral that goes out of range, loses power, or is claimed by another
central is noticed straight away rather than at the next write. The device's
row in the Hardware panel and its dot on the status strip both move to
**Disconnected**, and GLIDER starts retrying on its own: 5 s, then 10, 20, 40
and 60, up to twelve attempts before it gives up and shows **Error**.

The Bluetooth *board* stays green throughout. That is the host adapter, and it
is genuinely fine — it is the peripheral that went away, which is why
peripherals have dots of their own.

A run is not paused by a dropped peripheral. You get a warning notification and
the retries continue underneath it.

**A Maimu comes back off.** The stimulator runs a pulse train in its own
firmware, so a link that died mid-train left it stimulating with nothing
attached to stop it. When the reconnect succeeds GLIDER writes `off` before
anything else, so the device returns in a known state instead of resuming a
pattern nobody is watching. Re-issue the pulse yourself if you still want it.
```

Then, in the Maimu section, add:

```markdown
### Pulsing by hand

`pulse` takes a period and a duration, so it renders as two number fields and a
button — in the Builder's Device Control panel and in the Runner's manual
controls alike. Period is in **milliseconds and is a period, not a frequency**:
500 ms toggles about once a second. Duration is in seconds. The firmware runs
the train and stops on its own, so the button returns as soon as the command
lands.
```

- [ ] **Step 3: Add a CHANGELOG entry**

Follow the existing format at the top of `CHANGELOG.md` (match its heading
style and its Added/Fixed grouping exactly — check with `head -40`):

```markdown
### Added

- BLE peripherals report their own connection state, separately from the host
  adapter's. A dropped link shows as Disconnected in the Hardware panel and on
  the status strip, and GLIDER reconnects on its own with bounded backoff.
- A Maimu writes `off` when a dropped link comes back, so a stimulator that
  reconnected mid-pulse is put in a known state rather than left running.
- Devices can declare an action's arguments (`ACTION_ARGS_SCHEMA`), which is
  what makes Maimu's `pulse` pressable from both control panels.

### Fixed

- A BLE device with notifications enabled lost its subscription on the first
  reconnect and stayed silent for the rest of the session, with nothing logged.
- The Runner's manual controls called an argument-taking action with no
  arguments, raising `TypeError` on every press.
- The Hardware panel and the Device Control panel reported `_initialized`
  rather than the link, so a peripheral that walked out of range kept reading
  "Ready".
- `plugins/glider-maimu/tests` was missing from `testpaths`, so a bare `pytest`
  never collected that plugin's suite.
```

- [ ] **Step 4: Update the plugin README**

In `plugins/glider-maimu/README.md`, add a short paragraph under the actions or
usage section:

```markdown
`pulse` takes a period in milliseconds and a duration in seconds. Both are
declared in `ACTION_ARGS_SCHEMA`, so the Device Control panel and the Runner's
manual controls each render two number fields beside the button.

If the BLE link drops, GLIDER reconnects with bounded backoff and writes `off`
as soon as it is back — the firmware runs the pulse train itself, so a
reconnected device would otherwise still be stimulating.
```

- [ ] **Step 5: Check the docs build**

```bash
uv run mkdocs build --strict 2>&1 | tail -20
```

Expected: no warnings. If `mkdocs` is not installed, skip this step and say so
in the task report rather than reporting it as passing.

- [ ] **Step 6: Final full run**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```

Expected: all pass, clean lint.

- [ ] **Step 7: Commit**

```bash
git add docs-site/reference/devices.md CHANGELOG.md plugins/glider-maimu/README.md
git commit -m "docs: BLE reconnect behaviour and pulse's argument fields"
```

Do **not** run `mkdocs gh-deploy` as part of this plan — publishing the docs
site is a separate, deliberate step the maintainer takes.

---

## Verification checklist

Before calling the whole plan done, confirm each of these by running it —
not by reasoning about it.

- [ ] `QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"` — all pass, and the total is higher than the 110 baseline.
- [ ] `uv run ruff check src tests plugins` — clean.
- [ ] `uv run black --check src tests plugins` — clean.
- [ ] `uv run --no-sync pytest --collect-only -q 2>&1 | grep -c "glider-maimu"` — greater than 0.
- [ ] No commit carries a Claude attribution trailer or footer:
  `git log origin/main..HEAD --format='%B' | grep -c "Co-Authored-By\|Generated with Claude"` — must print `0`.
- [ ] Manual, with real hardware if one is on the bench: connect a Maimu, press **pulse** in the Device Control panel and confirm it runs; power the device off and confirm its strip dot goes red within ~2 s; power it back on and confirm it reconnects and does *not* resume the pulse.
