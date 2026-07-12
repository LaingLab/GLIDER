# Devices & Hardware

A node graph decides *what* your experiment does; **devices** are *how* it
touches the real world. This page covers adding boards, adding devices, binding
a device to a node, and how GLIDER decides the range a value is allowed to take.

## Boards vs. devices

GLIDER separates hardware into two layers:

- A **board** is the controller you plug into — an Arduino, a Raspberry Pi, or a
  Bluetooth LE link. It owns the physical pins and the connection.
- A **device** is a meaningful thing wired to that board — an LED, a button, a
  servo, an ADC chip. A device turns raw pins into named **actions** like
  `on`, `off`, `set`, or `read`, so your nodes never have to think in pin
  numbers.

You add a board first, then add devices onto it.

## Adding a board

Open **Hardware → Add Board...**. Pick a board type:

| Option in the dialog | Hardware | Notes |
|---|---|---|
| **telemetrix** | Arduino | Needs the Telemetrix4Arduino firmware flashed on the board. Choose the serial port (or let GLIDER auto-detect it). |
| **pigpio** | Raspberry Pi | Uses the Pi's own GPIO. No serial port needed. |
| **bluetooth** | Bluetooth LE | Connects to a BLE peripheral. No serial port — the address is set on the device. |

Give the board an ID (or accept the default), pick a serial port where it
applies, and confirm. Once boards are added, connect them all with
**Hardware → Connect All** (and **Disconnect All** to release them).

!!! note "No hardware? Design offline."
    You can build and even run a graph with nothing connected. Add real boards
    and devices when you're ready — the graph doesn't change.

## Adding a device

Open **Hardware → Add Device...** (you need at least one board first). Choose a
device type, give it a **Device ID** and pick the **Board** it lives on, then
fill in the pins and settings the dialog shows for that type.

### The built-in device catalog

| Device type | Dialog name | What it does | Pins / key settings |
|---|---|---|---|
| `DigitalOutput` | Digital Output (LED, Relay) | Drive a pin HIGH or LOW | `output` pin |
| `DigitalInput` | Digital Input (Button, Sensor) | Read a pin as on/off | `input` pin (optional pull-up) |
| `AnalogInput` | Analog Input (Potentiometer) | Read a voltage as a number | `input` pin |
| `PWMOutput` | PWM Output (Dimmable LED, Motor) | Drive a pin at a variable level | `output` pin |
| `Servo` | Servo Motor | Move a servo to an angle | `signal` pin; min/max angle |
| `MotorGovernor` | Motor Governor | Step a motorized position up/down | `up`, `down`, `signal` pins |
| `StepperA4988` | Stepper Motor (A4988) | Drive a stepper via an A4988 driver | `step`, `dir`, `enable`, `ms1`–`ms3` |
| `ADS1115` | ADS1115 (I2C ADC) | 16-bit analog readings over I2C | I2C address, gain, data rate |
| `GenericI2C` | Generic I2C Device | Talk to any I2C chip by address | I2C bus, address, register |
| `BLEWrite` | BLE Device (write characteristic) | Send string commands to a BLE peripheral | BLE address, characteristic UUID |

!!! tip "The full device reference"
    Each device exposes its own set of actions (for example, a `PWMOutput` has
    `set`, `set_percent`, and `off`). For the complete list of actions per
    device, see the [Device Catalog](../reference/devices.md).

### I2C, ADC, and Bluetooth specifics

- **ADS1115** and **Generic I2C** don't use GPIO pins — on a Raspberry Pi the
  I2C data and clock lines are fixed. You configure them by **address** (and, for
  the generic device, a register) instead of pins.
- **BLE Device** connects to a Bluetooth peripheral by address and writes string
  commands (like `on`, `off`, or `20,10`) to a characteristic. The Add Device
  dialog can **Scan** for nearby peripherals so you don't have to type the
  address by hand.

## Binding a device to a node

A hardware node is inert until you tell it *which* device to act on. This is
**binding**.

1. Add a hardware node to your graph — for example an `Output` node.
2. Select it, and in the **Properties** panel find the **Device** dropdown.
3. Pick your device from the list (it shows each device's name and type).

The device dropdown appears for these nodes: `Output`, `Input`, `WaitForInput`,
`MotorGovernor`, and `Device Action`.

!!! note "One node, many kinds of hardware"
    The generic `Output` node adapts to whatever it's bound to: bound to a
    digital device it writes HIGH/LOW; bound to a PWM device it writes a level.
    Likewise `Input` reads digital or analog depending on its device. That's why
    most experiments use a handful of `Output` and `Input` nodes rather than a
    different node per pin.

### Device Action: calling a named action

The `Device Action` node is the general-purpose way to invoke any action a
device offers. After binding a device, its **Action** dropdown lists that
device's available actions (like `set`, `read`, `write`, `up`, `down`). You can
also type constant **Arguments** — split on commas, with numbers detected
automatically — which are passed to the action when no data inputs are wired.
This is how you'd send a BLE `write` command such as `on`, `off`, or `20,10`.

## How a value's range is decided

When a node writes a value to a device — a PWM level, a servo angle, a pump rate
— GLIDER needs to know what range that value can legally take. Rather than
hard-coding a range, **each device declares the range for each of its actions**,
and every layer (the property editors, the Runner controls, and the write itself)
reads that single declaration. This is called the value's *spec*.

The practical upshots:

- **PWM depends on the board's resolution.** On an 8-bit board a PWM value runs
  `0–255`; on a 12-bit board it runs `0–4095`. The same `Output` or `PWM Write`
  node adapts to whichever board its device is on — it's never silently capped at
  8-bit.
- **A servo is bounded by its configured angle.** If you set a servo's min/max to
  `0–180`, commanding `200` lands at `180`.
- **On/off devices are a 0–1 switch.** A digital `set` action takes just on or
  off.

If you write a value outside the allowed range, GLIDER **clamps** it to the
nearest legal value rather than letting it wrap around to a wrong number. A
non-numeric or infinite value is rejected outright.

!!! warning "Out-of-range saved values are flagged at load"
    If you saved a value into a node (say, an `Output` set to `1000`) and later
    changed the bound device so that value no longer fits its range, GLIDER warns
    you when the experiment loads that the stored value will be clamped — so a
    range change never silently rewrites your experiment mid-run.

## Keeping hardware safe

- **Connect All / Disconnect All** (Hardware menu) bring every board online or
  release them.
- **Emergency Stop** (Run → Emergency Stop, `Ctrl+Shift+Esc`) immediately drives
  devices to a safe state — outputs off, motors stopped — without waiting for the
  graph to finish.

## Where to go next

- [The Node Graph](node-graph.md) — wire your bound devices into a running flow.
- [Custom Devices & Plugins](plugins.md) — add hardware GLIDER doesn't ship with.
- [Device Catalog](../reference/devices.md) — every device type and its actions.
