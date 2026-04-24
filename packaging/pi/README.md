# Raspberry Pi SD image — build notes

This directory scaffolds a [pi-gen](https://github.com/RPi-Distro/pi-gen)
build that produces a bootable Raspberry Pi OS image with GLIDER pre-
installed and set to auto-launch in Runner (kiosk) mode.

## What ends up on the SD card

- **Raspberry Pi OS Bookworm (64-bit)** with the desktop stage (`stage4`) —
  needed for the GUI; the lite stage would require us to pull Qt + dozens of
  dependencies ourselves.
- A dedicated `glider` user in the `gpio`, `video`, `dialout`, `i2c`, `spi`,
  `plugdev` groups.
- GLIDER cloned into `/opt/glider/repo`, installed into
  `/opt/glider/venv` (created with `--system-site-packages` so the
  apt-installed `python3-pyqt6` and `python3-opencv` are picked up).
- A `glider.service` systemd unit that launches `glider --runner` after
  the graphical target is up.
- lightdm configured to autologin `glider`.
- The in-app updater (`/opt/glider/scripts/glider-update.sh`) plus a narrow
  sudoers rule.
- A user data directory at `/home/glider/data/`.

The default `pi` user is kept intact for SSH recovery — if the kiosk breaks,
you can still log in as `pi` and debug.

## Directory layout

```
packaging/pi/
├── pi-gen-config            # Sourced by pi-gen's build.sh
├── os-list.json             # Pi Imager manifest (rewritten at release time)
├── README.md                # You are here
└── stage-glider/
    ├── 00-packages          # apt packages installed before 00-run.sh
    ├── 00-run.sh            # Chroot-time provisioning script
    ├── EXPORT_IMAGE         # Marker — pi-gen exports a .img from this stage
    └── files/
        ├── glider.service   # systemd unit
        ├── glider-update.sh # In-app updater helper (runs as root via sudo)
        └── sudoers-glider   # Grants glider NOPASSWD on the updater only
```

## Building locally (optional — CI is the canonical builder)

```bash
# Clone pi-gen (once).
git clone https://github.com/RPi-Distro/pi-gen
cd pi-gen

# Copy this directory's config and stage into pi-gen.
cp /path/to/glider/packaging/pi/pi-gen-config ./config
cp -r /path/to/glider/packaging/pi/stage-glider ./

# Point the stage at the release tag you want baked in. Omit for main.
export GLIDER_REF=v1.0.0

# Run. Takes ~45-60 min on a 16-core AMD; 2-3× longer on a laptop.
# Requires root (pi-gen uses a chroot).
sudo -E ./build.sh
```

The resulting `.img` and compressed `.img.xz` land in `pi-gen/deploy/`.

## Publishing to Raspberry Pi Imager

1. Release tag triggers the Pi workflow
   (`.github/workflows/release-pi.yml`), which uploads `GLIDER-<version>-pi64.img.xz`
   and an updated `os-list.json` as release assets.
2. Publish `os-list.json` to a stable URL (e.g. GitHub Pages on this repo at
   `https://lainglab.github.io/glider/os-list.json`).
3. In Raspberry Pi Imager, users choose *Operating System → Use custom list*
   and paste the URL. GLIDER appears as a choice, Imager fetches the
   `.img.xz`, and flashes it.

The manifest template (`os-list.json`) has placeholder `__VERSION__` strings
and zeroed hashes; the CI workflow replaces these with the real values after
the image is built.

## YOLO / ultralytics

Not included in the base image. The CV-tracking nodes detect missing
`ultralytics` at first use and prompt the user to install it into the venv
on demand — same AGPL-avoidance strategy as the Windows build.

## Gotchas

- **Camera stack.** Bookworm uses `libcamera` by default; the app's V4L2
  calls should work with `libcamera-apps` providing the compatibility
  shims. Test with the specific USB or CSI camera used in the lab before
  calling the image "done".
- **Autologin + X.** lightdm's autologin-user-timeout=0 logs in the glider
  user immediately; the systemd unit then takes over the display. If a
  user dismisses the unit (e.g. by Ctrl-C from a TTY), lightdm will *not*
  automatically re-exec — they'd need to reboot.
- **SD card wear.** Tracking logs fsync is already bounded in code. For
  multi-hour experiments recommend attaching a USB SSD for `/home/glider/data/`.
- **In-app updater scope.** The updater only refreshes GLIDER itself; it
  does not run `apt upgrade`. OS-level updates require reflashing the SD.
