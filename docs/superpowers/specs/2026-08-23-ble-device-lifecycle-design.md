# BLE device lifecycle

**Status:** approved 2026-08-23
**Prompted by:** a Maimu stimulator that GLIDER still calls connected after the
link has dropped, and a Pulse control that cannot be pressed.

---

## 1. The problem

Two symptoms, one root cause and one unrelated bug.

### 1.1 "Connected" outlives the connection

Nothing in GLIDER learns that a BLE peripheral has gone away. Every surface that
claims to show a Maimu's connection is showing something else:

| Surface | What it actually reads | Why it lies |
|---|---|---|
| Status strip dot | `board.state` — [main_window.py:1019](../../../src/glider/gui/main_window.py) | The BLE "board" is the host adapter. [`BLEBoard.connect`](../../../src/glider/hal/boards/ble_board.py) sets `CONNECTED` when `import bleak` succeeds and nothing moves it back. Permanently green. |
| Status bar line | same | same |
| Hardware tree status column | `device._initialized` — [hardware_panel.py:169](../../../src/glider/gui/panels/hardware_panel.py) | Set once in `initialize()`, cleared only in `shutdown()`. A peripheral walking out of range never touches it. |
| Device Control panel status | `board.is_connected` + `device._initialized` — [device_control_panel.py:259](../../../src/glider/gui/panels/device_control_panel.py) | Both of the above, combined. |

Underneath all four: [`BLEDevice`](../../../src/glider/hal/devices/ble_device.py)
never passes bleak's `disconnected_callback`, and `_ensure_connected` only
consults `client.is_connected` lazily, at I/O time. The drop is genuinely
unknown to the process until the next write fails.

`_initialized` is doing two jobs — "has been set up" and "is reachable" — and
the second one is a lie the moment the link drops.

### 1.2 A reconnect silently loses the notify subscription

`_with_retry` reconnects on a failed GATT op by clearing `_client` and calling
`_ensure_connected` again. `initialize()` is where `start_notify` happens, so
the new client is never subscribed. On a notify device `_latest` then goes stale
forever and `get_state()` returns `None` for the rest of the session, with
nothing logged. Independent of the Maimu, but the same lifecycle gap, so it is
fixed here.

### 1.3 Pulse cannot be pressed

`MaimuDevice.pulse(period_ms, duration_s)` takes two required arguments, and
neither control surface can supply them:

- **Builder — Device Control panel.** `_build_action_buttons` introspects the
  signature, finds required positional parameters, and renders the button
  **disabled** with a tooltip pointing at a node
  ([device_control_panel.py:320](../../../src/glider/gui/panels/device_control_panel.py)).
- **Runner — manual controls.** `pulse` has no `value_spec`, so `_controls_for`
  classifies it as a plain fire button
  ([device_controls.py:172](../../../src/glider/gui/runner/device_controls.py)).
  Pressing it calls `execute_action("pulse")` with no arguments and raises
  `TypeError`.

`on` and `off` work on both surfaces. This is not a Maimu problem — *any*
argument-taking action on *any* device is dead in one panel and crashes in the
other. The Maimu is simply the first device to have one.

## 2. What this is

1. A device that knows whether its own link is up, separately from whether it
   has been set up.
2. A BLE device that notices a drop, says so, and retries — without ever
   silently re-arming a stimulator.
3. Argument-taking actions that are drivable by hand on both surfaces.

## 3. Non-goals

- **Pausing the run on a drop.** The board-level handler does this
  (`_on_hardware_connection_change`); a peripheral drop will log and notify, not
  pause. A ten-second dropout should not end a two-hour experiment.
- **`serial_device.py`.** It has the same stale-state problem. The base-class
  interface in §4 is shaped so it can adopt this later; this work does not touch
  it.
- **Deriving `MaimuNode.PROPERTIES_SCHEMA` from the device.** The node and the
  device would each declare period and duration. The duplication is three
  fields, the node's help text is written for the node, and coupling them buys
  little.
- **An opt-in switch for auto-reconnect.** Bounded retries are the behaviour;
  there is no setting.

## 4. A device knows its own link state

`BaseDevice` grows two members.

```python
@property
def owns_link(self) -> bool:
    """Whether this device holds a connection of its own.

    False for a pin-based device: a DigitalOutput has no link separate from
    its board's. True for a transport device that opens its own socket.
    """
    return False

@property
def link_state(self) -> ConnectionState:
    """Where this device's own link stands, right now.

    The default is derived, not stored: a device with no link of its own is
    exactly as connected as its board, and is DISCONNECTED before setup.
    """
    if not self._initialized:
        return ConnectionState.DISCONNECTED
    return (
        ConnectionState.CONNECTED
        if self._board.is_connected
        else ConnectionState.DISCONNECTED
    )
```

`ConnectionState` is `BoardConnectionState`, aliased in `base_board` so the name
stops saying "Board". Reused rather than duplicated: the strip's
`DEVICE_STATE_BY_BOARD_STATE` mapping and `_board_state_text` already render
this vocabulary, and a second five-member enum meaning the same five things
would need a mapping table whose only purpose is to be kept in sync.

`_initialized` keeps its existing meaning and its existing job of gating
`execute_action`. It is no longer asked what it never knew.

### 4.1 Reporting a change

`HardwareManager` gains a device channel mirroring the board one:

```python
def on_device_connection_change(
    self, callback: Callable[[str, ConnectionState], None]
) -> None: ...
```

The device's own hook is wired in `_track_device` — the same chokepoint that
already wires `set_settings_changed_callback`, so no creation path can register
a device without it.

`MainWindow` marshals it through a `pyqtSignal` (devices fire from the bleak
callback and from a background task, neither of which is the Qt thread's own
call stack) into `_refresh_hardware_readouts`, exactly as
`_hardware_connection_changed` does today.

## 5. The BLE device notices drops

Three mechanisms, because the first is not reliable on its own.

**Primary — `disconnected_callback`.** `_ensure_connected` constructs
`BleakClient(address, disconnected_callback=self._on_disconnected)`. Near
instant on a clean drop.

**Secondary — I/O failure.** `_with_retry` already reconnects on a failed GATT
op. It now also moves `link_state` while it does, so a failure that is repaired
in-band is still visible as a blip rather than as nothing.

**Backstop — a poll.** `BaseDevice.poll_link()` is a no-op coroutine by default;
`BLEDevice` overrides it to compare `client.is_connected` against the state it
believes. `HardwareManager` runs one supervisor task that polls every device
with `owns_link` on a fixed interval (2 s).

The backstop is not belt-and-braces. CoreBluetooth and WinRT both drop the
disconnect callback often enough that "GLIDER says connected when it isn't" is
reproducible without it, which is the exact complaint this work exists to
answer. A callback that usually fires plus a poll that always does is one
mechanism, not two.

## 6. Auto-reconnect, bounded, never re-arming

An unsolicited drop while `_initialized` starts `BLEDevice._reconnect_task`,
modelled on `BaseBoard._attempt_reconnect` and sharing its shape and its
constants: a 5 s base interval doubling per attempt to a 60 s cap
(5 → 10 → 20 → 40 → 60 → 60 …), `MAX_RECONNECT_ATTEMPTS = 12`, `RECONNECTING`
throughout, `ERROR` and stop when it gives up. The task handle is cleared in
`finally` so re-entry is allowed.

Before `initialize()` and after `shutdown()`, `BLEDevice.link_state` is
`DISCONNECTED` — the override tracks a real link, and there is no real link to
be optimistic about.

On a successful reconnect, in this order:

1. **Re-subscribe** `start_notify(read_char)` if `_notify` — the §1.2 fix. This
   is why reconnect is a method rather than a bare `_ensure_connected`.
2. `await self._on_reconnected()` — a no-op hook on `BLEDevice`.
3. `link_state` → `CONNECTED`, callback fires.

### 6.1 The safe-state hook

```python
class MaimuDevice(BLEDevice):
    async def _on_reconnected(self) -> None:
        """Come back off.

        The firmware runs a pulse autonomously, so a link that dropped
        mid-train left a stimulator running with nothing attached to stop
        it. Whatever it was doing, it is doing it without supervision, and
        the reconnect is the first chance anyone has had to say otherwise.
        Same reasoning as shutdown(), which writes 'off' before it
        disconnects.
        """
        await self.write("off")
```

**The hook fires only on the background reconnect path.** Not on
`_with_retry`'s reconnect-inside-a-write: an `off` there would cancel the exact
command the caller just issued. The two paths reconnect through the same
`_ensure_connected`; only the supervised one runs the hook.

### 6.2 Teardown

`shutdown()` cancels `_reconnect_task` and awaits it before touching anything
else. The existing `_initialized` guard already stops a retry from re-arming a
device an emergency stop just stopped; cancelling the task first means the retry
does not even run. Both stay — the flag guards the in-flight case the
cancellation cannot reach.

## 7. What the operator sees

**Status strip.** One chip per peripheral beside the board chips, gated on
`owns_link` so a pin device does not get a dot that duplicates its board's.
Labelled by device name, coloured from `link_state` through the existing
`DEVICE_STATE_BY_BOARD_STATE`, with the device's own word in the `detail` slot
`set_devices` already accepts. The mapping collapses `RECONNECTING` and
`DISCONNECTED` onto the same amber/red decision; "retrying" and "gone" are not
the same news, and the tooltip is where that survives.

**Hardware tree.** The status column reads `link_state`: `Ready`,
`Reconnecting…`, `Disconnected`, `Error` — not `Ready` / `Not initialized`.

**Device Control panel.** Same status line, and action buttons disable while the
link is down rather than offering a press that will fail.

**During a run.** A `logger.warning` and one `MainWindow._notify_user(...,
level="warning")` on the drop — the same call the core-error path uses, so a
peripheral that goes away while an experiment is recording is on the record and
on the screen. No pause (§3), and no dialog: `_show_hardware_disconnection_dialog`
is modal and stays reserved for a board.

## 8. Pulse gets real controls

```python
ACTION_ARGS_SCHEMA: dict[str, list[dict]] = {}
```

A class attribute on `BaseDevice`, empty by default, declaring an action's
arguments in the field vocabulary `SETTINGS_SCHEMA` already uses — so
`schema_form.build_schema_widgets` and `read_schema_widget` render and read it
with no new widget code.

```python
class MaimuDevice(BLEDevice):
    ACTION_ARGS_SCHEMA = {
        "pulse": [
            {"key": "period_ms", "label": "Period (ms)", "type": "int",
             "default": 500, "min": 1, "max": 3_600_000,
             "help": "On/off toggle period — a period, not a frequency."},
            {"key": "duration_s", "label": "Duration (s)", "type": "int",
             "default": 10, "min": 1, "max": 86_400,
             "help": "How long the train runs. The firmware stops on its own."},
        ],
    }
```

Arguments are passed **positionally in schema order**, so `pulse` receives
`(period_ms, duration_s)` the right way round. Keys must match the parameter
names; §9 asserts it by introspection rather than by convention.

### 8.1 Builder — Device Control panel

`_build_action_buttons` gains a third case. Today it has two:

| Action | Today | After |
|---|---|---|
| No required args | Enabled button | unchanged |
| Required args, **schema declared** | Disabled + tooltip | **Enabled**, with the schema's fields rendered inline beneath the button row |
| Required args, no schema | Disabled + tooltip | unchanged |

Clicking reads the fields and calls `execute_action("pulse", 500, 10)`.

### 8.2 Runner — manual controls

A new control kind in `_controls_for`: an action with an `ACTION_ARGS_SCHEMA`
entry yields `("action_args", action, schema)`, rendering labelled spin boxes
and a fire button at `_CONTROL_MIN_HEIGHT`.

The crash goes with it. An action with required arguments and *no* schema stops
being classified as a plain fire button — the `else` branch that produces one
today is what makes `execute_action("pulse")` reachable with no arguments.

## 9. Testing

The Maimu plugin already exercises its protocol entirely against a fake bleak
client. Extend that fake with a `drop()`.

**Lifecycle**

- Drop → `link_state` is `RECONNECTING`, the device callback fires once,
  and the tree / strip / panel text all change.
- Successful reconnect → `link_state` is `CONNECTED`.
- Maimu reconnect → `off` is written exactly once, before any other write.
- Reconnect *inside* a `write` retry → `off` is **not** written, and the
  original command still lands.
- `shutdown()` during backoff → the task is cancelled and no reconnect
  follows.
- Attempts exhausted → `ERROR`, no further retries.
- Notify device reconnect → the subscription is live again and `get_state()`
  returns a fresh value (§1.2).
- Callback missed entirely (fake never fires it) → the 2 s poll still moves
  the state.
- Pin-based device → `owns_link` is False, `link_state` tracks its board, and
  it gets no strip chip.

**Controls**

- Builder: the Pulse button is enabled, reads its spin boxes, and calls
  `execute_action("pulse", 500, 10)`.
- Builder: an argument-taking action with no schema stays disabled.
- Runner: an argument-taking action with no schema is not rendered as a bare
  fire button.
- Runner: the Pulse control fires with both values.
- For every device class with an `ACTION_ARGS_SCHEMA`, each declared key
  matches a real parameter of the action, in order.

## 10. Files

| File | Change |
|---|---|
| `src/glider/hal/base_board.py` | `ConnectionState` alias |
| `src/glider/hal/base_device.py` | `owns_link`, `link_state`, `poll_link`, `ACTION_ARGS_SCHEMA`, link-state callback hook |
| `src/glider/hal/devices/ble_device.py` | disconnect callback, tracked state, reconnect task, `_on_reconnected` hook, notify re-subscribe |
| `src/glider/core/hardware_manager.py` | `on_device_connection_change`, wiring in `_track_device`, poll supervisor |
| `src/glider/gui/main_window.py` | device signal, strip chips for `owns_link` devices |
| `src/glider/gui/panels/hardware_panel.py` | status column from `link_state` |
| `src/glider/gui/panels/device_control_panel.py` | status from `link_state`, third case in `_build_action_buttons`, buttons disabled while down |
| `src/glider/gui/runner/device_controls.py` | `action_args` control kind, no bare button for argument-taking actions |
| `plugins/glider-maimu/src/glider_maimu/device.py` | `ACTION_ARGS_SCHEMA`, `_on_reconnected` |
