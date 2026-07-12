# Troubleshooting

The problems people hit most often, and how to fix them.

## Nothing happens when I press Start

Almost always a missing **execution wire**. Your experiment runs by following
exec wires from `StartExperiment` to `EndExperiment` — data wires alone won't
trigger anything.

- Trace an unbroken exec path from `StartExperiment` all the way to
  `EndExperiment`.
- Make sure you connected the **exec** ports (the "go" pulse), not just data
  ports.

See [How GLIDER Works](../getting-started/concepts.md#two-kinds-of-wires-exec-and-data).

## My device isn't listed / won't connect

- Confirm the **board** is connected first (Arduino over USB, the Pi's GPIO, or a
  Bluetooth device), then add devices to it. See
  [Devices & Hardware](../building/devices.md).
- **Arduino:** check the USB cable and that the correct serial port is selected.
- **Raspberry Pi:** install with the `rpi` extra and make sure your user is in
  the `gpio` group.
- **I²C devices (ADS1115, GenericI2C):** these need the `i2c` extra and run on
  Linux/Pi only — the I²C bus isn't available on Windows/macOS.

## A value I set isn't being written to the device

Devices clamp writes to their declared range. If you asked for a value outside
what the device supports (e.g. `1000` on an 8-bit PWM output that maxes at
`255`), GLIDER clamps it. Check the device's range — see
[Value ranges](devices.md#value-ranges).

## Video won't record

Video recording needs **FFmpeg** installed and on your PATH. Install it (see
[Recording prerequisites](../getting-started/installation.md#recording-prerequisites-ffmpeg))
and restart GLIDER.

## Tracking / Behavior Analysis is greyed out

These are **optional** features behind extras:

- **Tracking** needs the `vision` extra (`uv sync --extra pc --extra vision`).
- **Behavior Analysis** needs the `behavior` extra
  (`uv sync --extra pc --extra behavior`). The menu entry stays disabled — with
  an install hint in its tooltip — until those packages are present.

Confirm your machine's compute setup with **Tools → GPU / Device Check**.

## macOS won't open the app

An unnotarized app can be blocked on first launch. **Right-click the app →
Open**, then confirm. You only need to do this once. See
[Installation](../getting-started/installation.md#prebuilt-app).

## The Runner is stuck in kiosk mode

Runner mode is one-mode-per-process by design. To get back to the desktop
editor, use the **Setup** tab's switch-to-desktop control, or relaunch without
the `--runner` flag. On a Raspberry Pi kiosk you can log in as the `pi` user over
SSH to recover. See [Raspberry Pi Kiosk](../runner/pi-kiosk.md).

## Still stuck?

Open an issue on [GitHub](https://github.com/LaingLab/glider/issues) with your OS,
how you installed GLIDER, and what you were doing when the problem happened.
