# glider-maimu

Maimu BLE stimulator support for [GLIDER](https://github.com/LaingLab/glider).

Adds a **Maimu** device type and a **Maimu** node, so a stimulator can be added
from **Add Device → Maimu → Scan** and driven from the flow graph without
knowing the wire protocol.

> **Alpha.** The protocol is exercised entirely against a fake BLE client.
> Nothing here has been run against a physical stimulator.

## Install

```bash
pip install glider-maimu
```

Or from the **Tools → Plugins…** window inside GLIDER. A newly installed plugin
loads without a restart.

## The device

A Nordic/Zephyr peripheral with one writable characteristic that accepts three
UTF-8 commands:

| Command | Meaning |
| --- | --- |
| `on` | Full intensity, continuous |
| `off` | Stop the train, output dark |
| `<period_ms>,<width_ms>,<count>,<intensity_pct>` | Run a pulse train |

`50,4,4,100` is four 4 ms pulses at 20 Hz at full intensity. Note the first field is a
**period in milliseconds, not a frequency** — 50 ms is 20 Hz. `count` of `0` runs until
stopped; a width equal to the period is continuous light.

Intensity is **relative** — a percentage of whatever peak the board delivers, not
calibrated optical power. Per-unit brightness is trimmed in firmware.

### Settings

| Setting | Notes |
| --- | --- |
| **Address / UUID** | Filled by the **Scan** button. MAC on Windows/Linux, per-host UUID on macOS. |
| **Advertised name** | Set this *instead of* an address and the unit is found by name at connect time, so the same `.glider` file opens on any machine. |
| **Command characteristic** | Pre-filled. Advanced. |
| **Service UUID** | Pre-filled. Reference only. |

Because macOS hides the real MAC and hands out a per-host UUID, a file with a
hardcoded address is not portable between machines. Setting the advertised name
is what makes it portable.

### Actions

`on`, `off`, `pulse(period_ms, pulse_width_ms, count, intensity_pct)`, and the
inherited `write` as an escape hatch for anything a future firmware adds.
There is no `read`: the peripheral has no read characteristic, so offering one
would only allow selecting an action that fails at runtime.

`pulse` rejects zero (except where 0 is legitimate — see below), negative and
fractional values with a legible error rather than writing a command the
firmware's strict parser would silently reject.

`period_ms`, `pulse_width_ms`, `count` and `intensity_pct` are declared in
`ACTION_ARGS_SCHEMA`, which is what lets `pulse` render as four number fields
beside its button — in the Builder's Device Control panel and the Runner's
manual controls alike — instead of needing a Device Action node with
hand-typed arguments.

## The node

One `exec` in, one `exec` out, with **Mode** (On / Off / Pulse), **Period**,
**Pulse width**, **Pulses** and **Intensity** in its properties.

**Pulse is fire-and-continue.** `exec` out fires as soon as the write lands and
the stimulator runs the pattern itself — which is what `exec` out means
everywhere else in GLIDER. Put a Delay node after it to hold the flow for the
duration.

## Closed-loop use

The node's `exec` input accepts anything that fires execution, including the
**Behavior Input** node, so a stimulus can be delivered because of what the
animal is doing:

```
Behavior Input (freezing) ──On Enter──▶ Maimu (4 ms @ 20 Hz, 20 pulses, 100%)
```

Any behavior the loaded model recognises works — darting, grooming, whatever it
was trained on. `tests/test_behavior_triggers_stimulus.py` drives that whole
chain and asserts on the bytes reaching the peripheral.

## Stopping

The firmware runs a pulse **autonomously**, so dropping the BLE link mid-pulse
would leave the device stimulating with nothing connected to stop it. The device
therefore writes `off` before it disconnects, bounded so it cannot eat GLIDER's
emergency-stop budget. Emergency stop, End Experiment and app quit all stop the
device through that one path.

The same reasoning covers a link that drops on its own — out of range, out of
power, claimed by another central. GLIDER reconnects it with bounded backoff
(5, 10, 20, 40 and 60 seconds, up to twelve attempts) and, the moment it is
back, writes `off` before anything else touches the device. A stimulator that
reconnected mid-pattern comes back in a known state instead of resuming one
nobody is watching — re-issue the pulse yourself if you still want it.

## Development

```bash
PYTHONPATH="src;plugins/glider-maimu/src" QT_QPA_PLATFORM=offscreen \
    pytest plugins/glider-maimu/tests/
```

The plugin's source must be on `PYTHONPATH` (or installed editable), or an
older installed copy shadows it.
