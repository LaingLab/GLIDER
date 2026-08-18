# Custom Devices & Plugins

GLIDER ships with a solid set of [built-in devices](devices.md), but labs use
all kinds of hardware. When you need something that isn't in the list, GLIDER
gives you two ways to add it:

1. **The no-code Custom Device builder** — describe a device with a form, no
   programming. Best for straightforward I2C or GPIO gadgets.
2. **Python plugins** — a folder of code that registers new device types (and
   more). Best for hardware with real logic behind it.

Start with the builder; reach for a plugin when you outgrow it.

## No-code custom devices

Open **Hardware → New Custom Device Type...** to open the Custom Device builder.
You describe the device as data — no Python — and GLIDER saves it to your device
library and registers it, so it shows up in **Add Device** like any built-in.

Because a custom device is pure description (not executable code), loading one is
safe by default.

### Building one

In the dialog you:

1. **Name** the device (and optionally describe it).
2. Pick a **Transport** — how it talks to the board:
      - **I2C** — a chip on the I2C bus, addressed by number.
      - **GPIO** — a plain pin.
3. Add one or more **Actions**. Each action has a name and maps to a primitive
   **operation**. The available operations depend on the transport:

    | Transport | Operations |
    |---|---|
    | I2C | Read byte, Read word (16-bit), Write byte, Write word (16-bit) |
    | GPIO | Set HIGH, Set LOW, Read digital, Read analog, Write PWM |

   I2C read/write operations act on a **register** you specify. The write
   operations (write byte/word, write PWM) take their value at run time — from a
   `Device Action` node's argument or a wired input.

4. Mark one action as **Primary** — that's the one a `Device Read` node uses by
   default.

When you save, the device appears under **Add Device** with the standard settings
for its transport already filled in (I2C bus and address, or a GPIO pin).

!!! tip "Give a value-writing action a range"
    On a value-writing operation you can declare the range the value is allowed to
    take (a minimum, maximum, step, and unit). GLIDER then clamps writes to that
    range and uses it to build the right control in the Runner — the same
    [value-range system](devices.md#how-a-values-range-is-decided) the built-in
    devices use.

### Rotary encoders (revolution tracking)

I2C custom devices have an optional **Track revolutions** mode for rotary
encoders. Turn it on and GLIDER continuously reads the encoder's angle register
and unwraps it into a running total of turns, adding read operations for
revolutions, angle, and total counts, plus settings for the angle register and
counts-per-turn. Downstream nodes then just read a finished number.

## Python plugins

When a device needs real behavior — a background poll loop, unit conversions,
custom wait conditions — write a **directory plugin**: a folder of Python that
GLIDER loads at startup. The bundled `rotary_encoder` example under
`examples/plugins/` is a complete, working template.

!!! warning "Directory plugins run code — they're off by default"
    Unlike no-code custom devices, a directory plugin executes arbitrary Python.
    For that reason GLIDER does **not** load directory plugins unless you turn
    them on. Only enable plugins you trust.

### Installing a plugin

1. Copy the plugin folder into your plugin directory:

    ```text
    ~/.glider/plugins/rotary_encoder/
    ```

2. Enable directory plugins in `~/.glider/config.json`:

    ```json
    {
      "plugins": { "enable_directory_plugins": true }
    }
    ```

3. Install any dependencies the plugin lists (the rotary encoder needs I2C
   support: `pip install 'GLIDER[i2c]'`).

4. Restart GLIDER. The new device type now appears in **Add Device**, with its
   settings form rendered automatically.

### What's in a plugin

A device plugin is a folder with, at minimum, two files:

**`manifest.json`** — metadata GLIDER reads to find and load the plugin:

```json
{
  "name": "rotary_encoder",
  "version": "1.0.0",
  "description": "Rotary encoder (AS5600) device over I2C.",
  "author": "GLIDER",
  "plugin_type": "device",
  "entry_point": "rotary_encoder:setup",
  "requirements": ["smbus2>=0.4.0"]
}
```

**`__init__.py`** — the code, which must expose the plugin contract:

- `DEVICE_TYPES` — a dictionary mapping a type name to your device class, e.g.
  `DEVICE_TYPES = {"RotaryEncoder": RotaryEncoderDevice}`. This is what registers
  the device.
- The device class itself subclasses GLIDER's `BaseDevice` and implements its
  actions.
- Optionally a `SETTINGS_SCHEMA` on the class, so **Add Device** renders a
  settings form for it automatically.
- Optional `setup()` and `teardown()` lifecycle hooks.

### The settings schema

A `SETTINGS_SCHEMA` is a list of field descriptions. GLIDER turns each one into a
labeled input in the Add Device dialog. Supported field types are `int`, `float`,
`hex`, `bool`, and `str`, with optional keys `min`, `max`, `decimals` (for
floats), and `help`:

```python
SETTINGS_SCHEMA = [
    {"key": "i2c_bus", "label": "I2C Bus", "type": "int", "default": 1, "min": 0, "max": 1},
    {"key": "i2c_address", "label": "I2C Address", "type": "hex", "default": 0x36,
     "help": "AS5600 default is 0x36"},
    {"key": "counts_per_turn", "label": "Counts/Turn", "type": "int", "default": 4096},
]
```

!!! note "Plugins can add more than devices"
    The same folder-plus-contract mechanism can expose board drivers
    (`BOARD_DRIVERS`) and node types (`NODE_TYPES`), not just devices. Devices
    are the most common thing to add.

!!! warning "`entry_point` names a function, not a class"
    The attribute after the colon is **called** during load, and registration
    happens separately, by reading the `DEVICE_TYPES` / `BOARD_DRIVERS` /
    `NODE_TYPES` dictionaries off the module. So `my_plugin:setup` is correct and
    `my_plugin:MyDeviceClass` is not — the latter constructs your class with no
    arguments, discards it, finds no dictionaries, and registers nothing. The
    plugin loads without raising, which is what makes this worth stating: the
    only symptom is that your device never appears in **Add Device**.

    GLIDER's own Arduino, Raspberry Pi, Bluetooth, and serial drivers are *not*
    registered this way. They are registered explicitly in
    `HardwareManager`, and the `glider.driver` entry points declared in the
    project's own `pyproject.toml` have the `module:Class` shape above and
    register nothing.

### Advanced: custom wait behaviors

A plugin device can also advertise named **input behaviors** — conditions the
[`Wait For Input`](node-graph.md#useful-flow-control-nodes) node can wait on. The
rotary encoder plugin, for example, adds a *Revolution (Turns)* behavior (wait
until the shaft completes a set number of turns) and a *Move Counts* behavior
(wait until it moves a set distance), each with an optional motor ramp-down so the
shaft coasts to a clean stop. When such a device is bound to a `WaitForInput`
node, its behaviors show up as choices in the node's properties.

## Where to go next

- [Devices & Hardware](devices.md) — add and bind your new device once it's
  registered.
- [The Node Graph](node-graph.md) — put the device to work in a flow.
- [Device Catalog](../reference/devices.md) — the built-in devices for
  comparison.
