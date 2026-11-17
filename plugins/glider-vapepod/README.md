# glider-vapepod

GLIDER support for the vape box **side pod**: the second pump and atomizer on
the far side of the glass box from the PiBox. The pod has its own 12 V supply
and a Seeed Studio XIAO nRF52840 that GLIDER drives over Bluetooth, so only a
mains cord crosses over, or over a USB cable from the Pi (see
[Wired (USB)](#wired-usb)). This package holds both halves:

- `firmware/vapepod/`: the XIAO nRF52840 sketch the pod is built with.
  `firmware/vapepod_esp32/` is the same firmware for an ESP32 DevKitC.
- `src/glider_vapepod/`: the GLIDER device types `VapePod` (Bluetooth) and
  `VapePodUSB` (USB cable), and the `VapePod` node that drives either.

## Safety rules (in the firmware)

The pod's microcontroller enforces these itself, so a crashed or out-of-range host cannot leave
the coil heating:

- Both outputs turn **off the moment the Bluetooth link drops**, and the
  moment a wired host **closes the port or goes quiet for 3 s**
  (`HOST_TIMEOUT_MS`). Losing either link turns everything off, so drive a pod
  over one link at a time.
- The atomizer never runs above **50 % duty** (`MAX_ATOM_PERCENT`): full power
  burns the coils out. A puff asking for more is refused, not turned down.
- The atomizer only runs for a stated time, **at most 10 s** (`MAX_PUFF_MS`),
  then turns itself off. There is no "atomizer on" command. A longer puff is
  refused, not shortened.
- If the firmware's main loop ever hangs, the watchdog resets the chip, and a
  reset leaves both outputs low.

GLIDER adds two more on top:

- **Shutdown writes `off` first.** Emergency stop, End Experiment and quitting
  stop the pod at once, without waiting for the link to time out.
- **A missed puff is an error, never a late puff.** A puff asked for while the
  link is down raises, and the node shows the error. One cut short by a drop is
  logged. Neither is ever resent.

The pump is different. After a dropped link it goes back to the speed the
experiment last set. While the link is down, pump commands return at once, and
the newest one is applied on reconnect.

## Flashing the XIAO nRF52840

In the Arduino IDE:
1. In File > Preferences > Additional boards manager URLs, add
   `https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json`.
2. In the Boards Manager, install **Seeed nRF52 Boards** (1.1.x). Not "Seeed nRF52
   mbed-enabled Boards": the sketch needs that core's Bluefruit library and
   stops with an error saying so on the mbed one.
3. Open `firmware/vapepod/vapepod.ino`, select **Seeed XIAO nRF52840** and its USB
   port.
4. Upload. If an upload stops part-way or the port disappears, double-tap the
   XIAO's reset button so it starts its bootloader, then upload again.

**Check it before wiring the loads.** Open the serial monitor at 115200 baud.
It should print `# VapePod 0.3.6, max puff 10000 ms`. Type `pump,100` and
press Enter: it answers `atom=0 pump=100`. (Close the monitor, or send `off`:
three seconds without a command turns it all off again.) Then use a phone BLE app
(e.g. nRF Connect) to connect to `VapePod` and write `puff,1000` to the command
characteristic. The atomizer module's LED should light for one second.

**Building more than one pod:** give each its own `DEVICE_NAME` in the sketch
(`VapePod-2`, ...) and enter the same name in GLIDER.

**With an ESP32 DevKitC instead:** install **esp32 by Espressif, version 3.3.x**
(core 2.x should also work; core 4.x will not, as it replaced the Bluetooth
API), open `firmware/vapepod_esp32/vapepod_esp32.ino` and select **ESP32 Dev
Module**. Or run `pio run -t upload` in `firmware/`. It speaks the same protocol,
so GLIDER can't tell the two apart.

## Wiring

| XIAO nRF52840 | To |
|---|---|
| USB-C | Power (and, wired, the commands): the Pi, or any USB supply for Bluetooth-only use |
| GND | Both MOSFET modules' signal GND |
| D5 | Atomizer MOSFET module, signal input (IN / PWM / TRIG) |
| D6 | Pump MOSFET module, signal input |
| A0 (D0) | Formaldehyde (HCHO) sensor output, 0-3.3 V |
| A1 (D1) | Beam sensor output, 0-3.3 V |

On an ESP32 DevKitC the same go to its USB port, GND, GPIO25 (atomizer) and
GPIO26 (pump).

- **Signal current:** the firmware puts D5 and D6 in high-drive mode. That is
  plenty for a module whose input goes straight to a MOSFET gate. If a module
  has an optocoupler input and its LED stays dim or the output won't switch
  fully, it needs more current than the pin gives: drive it through a small
  NPN transistor.

- **MOSFET modules:** 12 V supply in. Their output goes to the 510 connector
  (atomizer) and to the pump. Put the 15 A fuse on the atomizer's +12 V lead.
  If the pump module has no flyback diode, fit a 15SQ045 across the pump
  (stripe to +).
- **Inputs must default to off.** Most modules have a pull-down on the signal
  input. If yours doesn't, add 10 kΩ from each signal pin to GND. Without one,
  the inputs float while the XIAO boots or resets.
- **Grounds:** run the atomizer's and the pump's return wires straight to the
  supply's −V terminal, never through the XIAO's ground. Whether the XIAO's
  GND connects to −V at all depends on the modules: an optocoupled input is
  isolated from it, a direct-to-gate input is not (see
  [Wired (USB)](#wired-usb)).
- **Wire:** 10 AWG for the atomizer pair, 18 AWG for the pump and the supply,
  22 AWG for the signals.
- If the link drops every time a puff starts, check the MOSFET modules' ground
  wiring first: the coil's current is upsetting the XIAO's ground.

## Wired (USB)

The same firmware takes the same commands over the XIAO's USB port, so a pod
can be driven from the Pi with a cable instead of Bluetooth. Nothing to change
on the pod; in GLIDER:

1. Add a board of type **serial**.
2. Add a device of type **VapePodUSB (plugin)** on it. Leave the port blank and
   it finds the XIAO by its USB IDs (`2886:8045`); set it (`/dev/ttyACM0`, or a
   `/dev/serial/by-id/...` path) only if two are plugged in, or for an ESP32
   pod, whose USB-UART bridge has no IDs of its own.
3. The **VapePod** node and the actions below work on it unchanged.

The device sends `status` every second as a heartbeat, from its own thread. If
GLIDER closes or crashes, the heartbeat stops and the pod switches itself off.
A pulled cable is not reconnected automatically: the next command raises (a
puff is never lost silently), and the device needs reconnecting from the
Hardware panel.

**Grounds.** The XIAO runs off the USB cable, so its GND *is* the Pi's GND. The
pod and the PiBox are on separate 12 V supplies on purpose (two Pi 5s died
with a shared 12 V ground), so check that the MOSFET modules keep them apart:
with everything unplugged, measure from the XIAO's GND pin to the pod supply's
−V.

- **Open circuit** (optocoupled module inputs): the modules isolate the Pi from
  the 12 V side. A plain cable is fine.
- **Near 0 Ω** (inputs straight to the gate): the cable joins the Pi's ground
  to the 12 V return. Put an ADuM3160-based USB isolator in the cable (full
  speed, 12 Mbps, which is all the XIAO uses) with an isolated 5 V output; the
  XIAO draws tens of mA, well within it.

**Analog inputs.** Only A0-A5 (D0-D5) can read analog on the XIAO; pins
D6-D10 sit on the nRF52840's second port, which has no ADC. A sensor whose
output can go above 3.3 V needs a divider (e.g. 10 kOhm in series, 20 kOhm to
GND for a 5 V signal). Over USB every status carries both readings in mV
(`atom=0 pump=0 hcho=812 beam=3301`), so the recorder logs them too;
Bluetooth does not carry them.

## Protocol

There is one GATT service, `70c4957d-11e9-41a8-83fc-8783938e1e90`, with two
characteristics.

**Command:** `…1e91`, write with response. Send one UTF-8 command per write.
Commands are case-insensitive and their fields are comma-separated.

| Command | Does |
|---|---|
| `off` | Atomizer and pump off |
| `pump,<percent>` | Pump at 0-100 %. It stays at that speed until changed. 100 is a solid output. Below 100 it switches at 1 kHz, which makes the pump whine. |
| `puff,<ms>[,<percent>]` | Atomizer at `percent` (1-50, default 50) for `ms` (1-10000), then off. A new puff replaces one already running. |
| `puff,0` | Stop a puff early |
| `status` | Send the status now |

The pod ignores anything else and prints it as `# refused: ...` on the USB
serial port.

**Over USB** each command is one line ending in `\n` (up to 31 characters; a
longer line is refused whole). Every line the pod sends back is either the
status or a note starting with `# `. Once a host has sent a command, it must
send another (GLIDER sends `status`) at least every 3 s, or the pod treats it
as gone and turns everything off. On the XIAO, closing the port (DTR dropping)
does the same at once.

**Status:** `…1e92`, read and notify. It reads `atom=<percent> pump=<percent>`,
and the pod pushes it on every change and once a second.

## In GLIDER

Install the plugin into GLIDER's environment:

```bash
uv pip install -e ./plugins/glider-vapepod
```

Then add a device of type **VapePod** (Bluetooth; for a cable see
[Wired (USB)](#wired-usb)):
- The advertised name is pre-filled as `VapePod`, so the address can stay
  blank.
- The UUIDs are pre-filled too.

The device subscribes to the status, so the recorder logs the pod's own account
of its outputs (`atom=100 pump=100`) alongside each sample.

| Action | Arguments | Notes |
|---|---|---|
| `puff` | duration_ms, power (%, at most 50) | Returns once the command lands; the pod times the puff |
| `pump_on` | | Full speed |
| `pump` | percent | 0 is off |
| `pump_off` | | |
| `off` | | Atomizer and pump |
| `read_hcho` | | HCHO sensor now, in mV (USB only) |
| `read_beam` | | Beam sensor now, in mV (USB only) |
| `read` | | The status text |
| `write` | text | Raw command, for anything not modelled |

The **VapePod** node, in the Hardware section, does the same with a Mode of
Puff, Pump on, Pump off or All off. A puff is fire-and-continue: the node's
`exec` output fires as soon as the command lands. To hold the flow for the
puff, follow the node with a Delay.

## Tests

The Bluetooth driver tests run against a fake bleak client and the USB ones
against a fake pyserial port. Others check that the UUIDs, `MAX_PUFF_MS`, the
status format and the wired timeout match between both sketches and the
driver.

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src pytest plugins/glider-vapepod/tests
```
