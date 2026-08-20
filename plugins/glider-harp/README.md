# glider-harp

Harp binary-protocol support for [GLIDER](https://github.com/LaingLab/glider),
a laboratory automation platform.

[Harp](https://harp-tech.org/) is an open hardware standard for behavioural
neuroscience instruments — lickometers, behaviour boards, olfactometers — that
speak a common binary protocol over a serial link. This package lets GLIDER
record from them alongside its cameras and its other hardware.

## Install

```
pip install glider-harp
```

Or, from GLIDER 1.1 onwards, through **Tools → Plugins…** in the application
itself.

Requires Python 3.11–3.13 and an existing GLIDER installation. GLIDER is
deliberately *not* declared as a dependency: this package is installed into an
environment that already has it.

## What it provides

| Entry point | Kind | Name |
|---|---|---|
| `glider.driver` | Board driver | `harp` |
| `glider.device` | Device type | `HarpDevice` |

Add a Harp board in GLIDER's hardware map, attach a device to it, and its
registers appear as columns in the recording CSV.

## Recorded columns

A device contributes several columns, prefixed with its GLIDER device id. A
LicketySplit configured as `harp1` produces:

| Column | Type | Meaning |
|---|---|---|
| `harp1:lick_state` | 0/1 | Register value at poll time |
| `harp1:lick_count` | int | Events since the previous poll, cleared on read |
| `harp1:lick_last_ms` | float | Device timestamp of the most recent event, ms |

Sub-columns are joined with `_` rather than `:`, because the recorder splits the
device prefix off with a single `partition(":")` — a colon inside a device's own
column name would be ambiguous.

## Timing

**GLIDER keeps the master clock.** Harp devices carry their own hardware
timestamps, and this package records them *alongside* GLIDER's row timestamps
rather than in place of them. It does not adopt Harp's clock, and it does not
attempt to synchronise GLIDER to it.

The device timestamp is `seconds + micros × 32 µs`, as the Harp protocol
defines. Intervals between events on one device are reliable. Aligning
`*_last_ms` against another clock means knowing what epoch the device is
counting from, which is device- and configuration-dependent — check your
device's documentation before relying on it.

## Supported devices

The package ships one profile:

- **LicketySplit** lickometer (`WhoAmI` 1400)

Other Harp devices work by supplying a profile — a small JSON file naming the
device's registers and which of them should be recorded. Core registers
(`WhoAmI`, hardware and firmware versions, `TimestampSecond`, `OperationControl`,
`ClockConfig`) are never recorded as data columns.

### Adding your own profile

Put it in `~/.glider/harp_profiles/`, as `<name>.json`. It appears in the
**Recording profile** dropdown when you add a Harp device, alongside the shipped
ones, and it survives upgrades of this package.

```json
{
  "schema_version": "1.0",
  "name": "My Behaviour Board",
  "who_am_i": 1216,
  "record": [
    { "register": "DigitalInputState", "as": "poke" },
    { "register": "AnalogData", "as": "analog" }
  ]
}
```

`register` names a register in the board's own `device.yml`; `as` is the base
name of the three columns it contributes. `who_am_i` is checked against the
schema, so a profile written for one board cannot be pointed at another.

A profile of your own with the same name as a shipped one **replaces** it —
that is how you correct a shipped profile that does not match your firmware.
The dropdown marks it as yours so the substitution is never a mystery, and the
log says so when it is loaded.

### Which register types can be recorded

| Payload type | Recorded | Written |
|---|---|---|
| `U8` `U16` `U32` `U64` | yes | yes |
| `S8` `S16` `S32` `S64` | yes, decoded signed | yes, packed signed |
| `Float` | yes, IEEE-754 single | no — refused, not coerced |
| Arrays (`length` > 1) | no — refused | untested; see below |

A profile naming a register that cannot be recorded is **refused when the
device initializes**, with a message naming the register and its type. It does
not record a wrong number: the only failure mode is a device that will not
start until the profile is corrected.

Arrays are refused because how several values should appear in one CSV cell is
undecided, not merely unimplemented — one column per element, a delimited
string, and the first element only are all defensible and mean different
things. If you need one, say which.

Writes have always used the register's declared width and signedness, so a
signed register has always been usable as an action even when it could not be
recorded. Writing a `Float` register is refused outright rather than pushed
through a lossy integer conversion that would look like it worked. Writing an
**array** register is the one gap that is not a refusal: the write is packed to
a single element's width, which is not what an array register expects. Do not
drive an array register from a `DeviceAction` node.

## Status and limitations

This package has **never been run against real Harp hardware.** Everything in
its test suite exercises `MockHarpDevice`, which fakes the serial port. The frame
codec, the reader thread, register derivation, and the recording path are all
covered; what is unverified is every assumption about how a physical device
behaves — how quickly it answers, what it does on a mid-session unplug, and
whether the shipped LicketySplit profile matches the firmware you have.

Treat a first run as a bring-up exercise, not a deployment.

Known gaps beyond that:

- Array registers cannot be recorded and `Float` registers cannot be written —
  both refusals, not wrong numbers. Writing an array register is neither: it
  packs one element's width and does not complain. See the table above.
- Devices boot in Standby and emit no events until set Active; the driver does
  this and reads the register back to confirm, but only that path is tested.

## Licence

MIT — see `LICENSE`.

## Links

- GLIDER: https://github.com/LaingLab/glider
- Harp technology: https://harp-tech.org/
- Issues: https://github.com/LaingLab/glider/issues
