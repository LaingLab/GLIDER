# Device Catalog

GLIDER talks to hardware through a **board** (the controller) and the **devices**
attached to it. This page lists the built-in board drivers and device types. For
how to add and bind them, see [Devices & Hardware](../building/devices.md).

## Boards

| Board | Connection | Notes |
|---|---|---|
| **Arduino** | USB serial | Uses the Telemetrix firmware protocol. The most common desktop setup. |
| **Raspberry Pi** | On-board GPIO | Requires the `rpi` extra (gpiozero + lgpio). Great for kiosk rigs. |
| **Bluetooth (BLE)** | Bluetooth Low Energy | Cross-platform (Windows/macOS/Linux) via `bleak`. |
| **Serial (UART)** | Host serial ports | Cross-platform via `pyserial`. A transport with no pins of its own — each serial device opens its own port. A Bluetooth-Classic (SPP) pairing shows up here as an ordinary serial port. |

## Device types

| Device | What it is | Typical use |
|---|---|---|
| **DigitalOutput** | A single on/off output pin | LED, valve, relay, TTL trigger |
| **DigitalInput** | A single on/off input pin | Button, lever, lick sensor, beam-break |
| **AnalogInput** | An analog voltage reading | Potentiometer, light/temperature sensor |
| **PWMOutput** | A variable (pulse-width) output | LED brightness, fan/motor speed |
| **Servo** | An angle-controlled servo | Positioning arms, gates, feeders |
| **MotorGovernor** | Up / down / stop motor control | Simple motorized stage |
| **StepperA4988** | A stepper motor via an A4988 driver | Precise, repeatable positioning |
| **ADS1115** | A 4-channel, 16-bit I²C analog-to-digital converter | High-resolution analog inputs |
| **GenericI2C** | Register-level read/write to any I²C device | Custom or uncommon I²C sensors |
| **BLEWrite** | Writes to a Bluetooth LE characteristic | Wireless actuators / custom BLE peripherals |

!!! tip "Maimu stimulators come from a plugin"
    The **Maimu** device and node ship in `glider-maimu`, installable from
    **Tools → Plugins…**. It bakes in the stimulator's UUIDs and command
    protocol, so adding one is **Add Device → Maimu → Scan** rather than
    pasting UUIDs into a generic BLE device.

!!! tip "Pulsing a Maimu by hand"
    `pulse` takes a period and a duration, so it renders as two number fields
    beside its button — in the Builder's Device Control panel and the Runner's
    manual controls alike. **Period is in milliseconds and is a period, not a
    frequency**: 500 ms toggles about once a second. Duration is in seconds.
    Both must be whole numbers of at least 1 — the firmware `atoi`s them, so a
    fractional or zero value would only fail later. The firmware runs the
    train and stops on its own, so the button returns as soon as the command
    lands.

!!! note "I²C is Linux/Pi at runtime"
    I²C devices (**ADS1115**, **GenericI2C**) need the `i2c` extra and run on
    Linux or a Raspberry Pi. They install fine on any OS, but the I²C bus itself
    is only available on Linux/Pi.

## When a BLE link drops

A BLE peripheral that goes out of range, loses power, or is claimed by another
central is noticed straight away, not at the next write. Its row in the
[Hardware panel](../building/devices.md) and its own dot on the status strip
(shown beside the board dots for any peripheral that tracks its own link) both
move to **Disconnected**, and GLIDER starts retrying on its own: 5 seconds,
then 10, 20, 40 and 60, up to twelve attempts, before it gives up and shows
**Error**.

- The Bluetooth **board**'s own dot stays green throughout. That is the host
  adapter, and it is genuinely fine — it is the peripheral that went away,
  which is why a peripheral that owns its link gets a dot of its own.
- A run is not paused by a dropped peripheral. You get a warning notification
  and the retries continue underneath it; only **Error** — retries exhausted —
  needs you to go and look.
- **A Maimu comes back off.** The stimulator runs a pulse train in its own
  firmware, so a link that died mid-train left it stimulating with nothing
  attached to stop it. When the reconnect succeeds, GLIDER writes `off` before
  anything else, so the device returns in a known state instead of resuming a
  pattern nobody is watching. Re-issue the pulse yourself if you still want it.

!!! tip "Two stimulators with the same name"
    A strip dot is labelled with the device's name, since that is usually more
    meaningful than its id. If two devices on the rig share a name — six
    still-default-named stimulators, say — GLIDER appends the id to tell their
    dots apart. A uniquely named device keeps its plain name.

## Value ranges

Each device declares the valid range for every action it supports, and GLIDER
uses that range everywhere — the controls it generates, the values a node writes,
and the clamping that protects the hardware. For example:

- a **PWMOutput** on an 8-bit board accepts `0–255`; on a 12-bit board, `0–4095`;
- a **Servo** accepts its configured minimum-to-maximum angle;
- an **AnalogInput**'s reading spans `0` to the board's ADC resolution.

You never have to hard-code these limits — bind the device and GLIDER fills them
in.

!!! tip "Need a device that isn't listed?"
    You can define your own without touching GLIDER's source using the
    [custom device / plugin system](../building/plugins.md).
