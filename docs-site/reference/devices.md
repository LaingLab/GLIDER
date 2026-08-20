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
| **Maimu** | A Maimu BLE stimulator, protocol built in | Wireless stimulation — on, off, or a timed pulse train |

!!! note "I²C is Linux/Pi at runtime"
    I²C devices (**ADS1115**, **GenericI2C**) need the `i2c` extra and run on
    Linux or a Raspberry Pi. They install fine on any OS, but the I²C bus itself
    is only available on Linux/Pi.

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
