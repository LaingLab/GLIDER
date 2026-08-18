# Raspberry Pi Kiosk

The most common home for GLIDER's [Runner screen](runner.md) is a Raspberry Pi
with a small touchscreen, bolted to the rig and left running. This page covers
the two ways to get there: flashing the **prebuilt GLIDER Pi image** (the easy
path), and understanding the **systemd service and update script** that keep it
running and up to date.

## The prebuilt Pi image (recommended)

GLIDER ships a bootable Raspberry Pi OS image with everything already installed
and configured to launch straight into Runner mode. You flash it to an SD card
with the standard **Raspberry Pi Imager** tool and boot the Pi — no manual
setup.

### What's on the image

The image is built on **Raspberry Pi OS Bookworm (64-bit)** with the desktop
included (the GUI needs it), and it's preconfigured with:

- A dedicated **`glider`** user, added to the `gpio`, `video`, `dialout`, `i2c`,
  `spi`, and `plugdev` groups so it can reach hardware.
- GLIDER cloned to `/opt/glider/repo` and installed into a Python virtual
  environment at `/opt/glider/venv` (built with `--system-site-packages` so the
  system's `python3-pyqt6` and `python3-opencv` are used instead of being
  rebuilt on the Pi).
- A **`glider.service`** systemd unit that auto-starts GLIDER in Runner mode
  once the graphical session is up.
- **lightdm autologin** for the `glider` user, so the Pi boots straight to the
  kiosk with no login prompt.
- An **update helper script** (`/opt/glider/scripts/glider-update.sh`) and a
  tightly-scoped permission rule for it.
- A data directory at **`/home/glider/data/`** for your experiment output.

The default **`pi`** user is left intact for recovery — if the kiosk ever breaks,
you can still SSH in as `pi` and debug.

### Flashing it with Raspberry Pi Imager

The GLIDER image is published as a *custom list* that Raspberry Pi Imager can
load:

1. Install and open **Raspberry Pi Imager**.
2. Choose **Operating System → Use custom list** and paste the GLIDER OS-list
   URL (published with each release; for LaingLab this is
   `https://lainglab.github.io/glider/os-list.json`).
3. Pick **GLIDER** from the list, choose your SD card, and flash.
4. Put the card in the Pi and power on. It boots, autologs in the `glider` user,
   and GLIDER launches in Runner mode.

!!! tip "Use a fast card, or an SSD for long runs"
    Multi-hour recordings write a lot of data. For long experiments, attach a USB
    SSD and point the data directory (`/home/glider/data/`) at it to avoid SD
    card wear.

!!! note "Camera hardware"
    Bookworm uses `libcamera`; the image includes the `libcamera-apps`
    compatibility shims for GLIDER's camera support. Test with the specific USB
    or CSI camera you'll use in the lab before relying on it.

## How GLIDER auto-starts: the systemd service

The image installs a systemd unit, `glider.service`, that owns the kiosk. You
generally never touch it, but knowing how it works makes troubleshooting far
easier.

- It waits for the graphical session (lightdm's autologin brings that up) and
  the network before starting.
- It runs GLIDER as the `glider` user on the Pi's display, launching:

  ```bash
  /opt/glider/venv/bin/glider --runner
  ```

  The `--runner` flag forces the [touchscreen layout](runner.md) regardless of
  the detected screen size.
- If GLIDER **crashes**, systemd restarts it after a few seconds. If GLIDER
  **exits cleanly** (you quit it on purpose), systemd leaves it closed rather
  than fighting you.
- All of GLIDER's log output goes to the system journal.

### Handy commands

If you SSH into the Pi (as `pi`, or `glider`), these help you check on the
kiosk:

```bash
# See GLIDER's live logs
journalctl -u glider -f

# Restart the kiosk
sudo systemctl restart glider.service

# Stop / start it
sudo systemctl stop glider.service
sudo systemctl start glider.service

# Is it running?
systemctl status glider.service
```

!!! warning "If you quit the kiosk from a terminal"
    lightdm autologins the `glider` user once, then hands the display to the
    service. If you kill GLIDER from a TTY (for example with Ctrl-C), lightdm
    will **not** automatically re-launch it — reboot the Pi to get the kiosk
    back.

## Keeping GLIDER updated

The image ships an **update helper script**,
`/opt/glider/scripts/glider-update.sh`, so you can move the kiosk to a newer
GLIDER release without reflashing the card. There is **no in-app update action
yet** — the only updater inside GLIDER is the release check, which can open the
GitHub Releases page in a browser. To update the kiosk, SSH in and run the
script:

```bash
sudo /opt/glider/scripts/glider-update.sh
```

The script does the work:

1. Fetches the latest tagged GLIDER release.
2. Checks it out in `/opt/glider/repo`.
3. Reinstalls it into the existing `/opt/glider/venv` (picking up any new Python
   dependencies).
4. Restarts `glider.service`, so the kiosk relaunches on the new version.

The script runs with a narrowly-scoped permission: the `glider` user is allowed
to run **only that one script** with `sudo` and no password, and nothing else.
Its log is written to `/var/log/glider-update.log`.

!!! warning "The update script only updates GLIDER"
    It refreshes GLIDER itself, not the operating system. It does **not** run
    `apt upgrade` or update the kernel. To pick up OS-level updates, reflash the
    SD card with a newer image.

!!! note "Machine-learning tracking (YOLO / ultralytics)"
    The optional `ultralytics` package used by the CV-tracking nodes is **not**
    bundled in the base image. GLIDER detects it's missing the first time you use
    a tracking node and offers to install it into the venv on demand. See
    [Tracking](../camera-behavior/tracking.md).

## Installing on a Pi by hand (no prebuilt image)

If you'd rather set up your own Pi instead of flashing the image, install GLIDER
manually. On Raspberry Pi, PyQt6 comes from the system package manager (it is
intentionally not a bundled dependency), and the virtual environment is created
with access to those system packages:

```bash
sudo apt install python3-pyqt6
uv venv --system-site-packages
uv sync --extra rpi --extra i2c
```

The `rpi` extra pulls in the Raspberry Pi GPIO libraries and `i2c` adds support
for I2C devices. See [Installation](../getting-started/installation.md) for the
full walkthrough and the complete list of optional extras. Once installed, start
the touchscreen layout with:

```bash
glider --runner
```

To make a hand-built Pi auto-launch on boot like the prebuilt image, you can
adapt the systemd service and autologin setup described above; the packaging
files under `packaging/pi/` in the repository are a working reference.
