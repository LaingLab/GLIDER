# Rotary Encoder plugin

A GLIDER **device plugin** that adds a `RotaryEncoder` device for the AS5600
magnetic rotary encoder (I2C). It tracks cumulative **revolutions** in a
background poll loop and exposes angle/degrees/counts with gear-ratio and
rounding conversions — all inside the device, so downstream nodes just read a
finished number.

This folder doubles as the reference template for writing your own GLIDER
device plugins.

## What it provides

| Read / action | Meaning |
|---|---|
| `read()` (DeviceRead) | cumulative **revolutions** (signed, ÷ gear ratio, rounded) |
| `angle` | current raw angle, `0 .. counts_per_turn-1` |
| `degrees` | current angle in degrees, `0 .. 360` |
| `revolutions` | cumulative revolutions |
| `total_counts` | raw signed cumulative counts |
| `reset` | zero the count at the current position |

**Settings:** `i2c_bus`, `i2c_address` (0x36), `angle_register` (0x0E),
`counts_per_turn` (4096), `gear_ratio` (1.0), `decimals` (2),
`poll_interval` (0.02 s). These render automatically in the Add Device dialog
from the device's `SETTINGS_SCHEMA`.

## Install

1. Copy this folder into your plugin directory:

   ```
   ~/.glider/plugins/rotary_encoder/
   ```

2. Enable directory plugins in `~/.glider/config.json` (off by default, since
   directory plugins run arbitrary code):

   ```json
   {
     "plugins": { "enable_directory_plugins": true }
   }
   ```

3. Install the I2C dependency on the Pi if needed: `pip install 'GLIDER[i2c]'`.

4. Restart GLIDER. `RotaryEncoder` now appears under **Add Device → Plugin
   Devices**, with its settings form rendered from the schema.

## Writing your own device plugin

Use this folder as a template. The minimum a device plugin needs:

- `manifest.json` with `name`, `plugin_type: "device"`, and a non-empty
  `entry_point` (so the module is imported).
- `__init__.py` defining a `BaseDevice` subclass and
  `DEVICE_TYPES = {"YourDevice": YourDeviceClass}`.
- Optionally a `SETTINGS_SCHEMA` class attribute for an auto-rendered settings
  form, and `setup()` / `teardown()` hooks.
