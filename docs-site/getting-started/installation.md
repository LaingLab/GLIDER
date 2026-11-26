# Installation

GLIDER installs from source with [uv](https://docs.astral.sh/uv/), on macOS,
Windows, Linux and the Raspberry Pi. The [one-command installer](#one-command)
does it all for you, and running it again updates GLIDER. On a Pi you can
instead flash the [prebuilt SD-card image](../runner/pi-kiosk.md).

!!! note "No more macOS and Windows installers"
    GLIDER no longer publishes `.dmg` or Windows installer builds. If you have
    one, it keeps working, but it won't get new versions: install from source
    with the one command below to stay up to date.

!!! info "Supported Python"
    GLIDER runs on **Python 3.12**, the version CI tests. `uv` installs
    Python 3.12 for you, so you don't need to manage Python yourself. The Raspberry Pi is the
    exception: it uses the OS's own Python (3.11 on Bookworm) so it can see the
    system's PyQt6.

## From source

### One command

This installs uv and git if they are missing, downloads GLIDER into `~/glider`
(set `GLIDER_DIR` to put it elsewhere), installs the right extras for your
platform, and finishes by running [`glider doctor`](#verify-your-install).
Run it again at any time to update.

=== "macOS / Linux / Raspberry Pi"

    ```bash
    curl -LsSf https://raw.githubusercontent.com/LaingLab/glider/main/install.sh | sh
    ```

    To add [extras](#optional-extras), pass them after `sh -s --`:

    ```bash
    curl -LsSf https://raw.githubusercontent.com/LaingLab/glider/main/install.sh | sh -s -- --extra vision --extra behavior
    ```

=== "Windows"

    In PowerShell:

    ```powershell
    irm https://raw.githubusercontent.com/LaingLab/glider/main/install.ps1 | iex
    ```

    To add [extras](#optional-extras):

    ```powershell
    & ([scriptblock]::Create((irm https://raw.githubusercontent.com/LaingLab/glider/main/install.ps1))) -Extra vision,behavior
    ```

!!! note "Name every extra each time"
    Re-running the installer installs exactly the extras you pass, plus the
    platform's own. An extra you added last time and leave out now is removed.

When it finishes, it prints the command that starts GLIDER.

### By hand

GLIDER uses [uv](https://docs.astral.sh/uv/) to manage its virtual environment
and Python version.

#### 1. Install uv

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    winget install astral-sh.uv
    ```

#### 2. Get GLIDER and install it

=== "macOS / Windows / Linux"

    ```bash
    git clone https://github.com/LaingLab/glider.git
    cd glider
    uv venv
    uv sync --extra pc
    uv run glider
    ```

=== "Raspberry Pi"

    On the Pi, PyQt6 comes from the system package manager (it's intentionally
    *not* a core dependency), so the virtual environment must be the system
    Python's, with access to its packages. Install with `uv pip` and start the
    venv's own `glider`: `uv sync` and `uv run` would swap the venv for one on
    the Python that `.python-version` pins, which cannot see apt's PyQt6.

    ```bash
    sudo apt install python3-pyqt6 python3-pyqt6.qtsvg python3-pyqt6.qtopengl
    git clone https://github.com/LaingLab/glider.git
    cd glider
    uv venv --system-site-packages --python /usr/bin/python3
    uv pip install -e ".[rpi,i2c]"
    .venv/bin/glider
    ```

`uv run glider` launches the app. The first time you run it, GLIDER offers a
short interactive tour — take it or skip it.

### Open GLIDER by searching for it

The install script adds GLIDER to your applications, so you can open it
from **Spotlight** (macOS), the **Start Menu** search (Windows) or your
desktop's app menu (Linux), like any other app. If you installed by hand,
or the step was skipped, add the launcher yourself from the GLIDER folder:

```bash
uv run glider shortcut
```

On macOS this creates `GLIDER.app` in your home folder's **Applications**;
on Windows, a **GLIDER** Start Menu shortcut. Both start GLIDER from the
install they were made from.

!!! tip "Launch straight into Runner mode"
    Add the `--runner` flag to start in the touchscreen
    [Runner](../runner/index.md): `uv run glider --runner`.

## Optional extras

The `--extra` flags decide which capabilities are installed. Add any of them to
the `uv sync` command, e.g. `uv sync --extra pc --extra vision`.

| Extra | What it adds |
|---|---|
| `pc` | PyQt6 + audio — the standard desktop install |
| `rpi` | Raspberry Pi GPIO (gpiozero, lgpio) |
| `vision` | Pose/object [tracking](../camera-behavior/tracking.md) (YOLO / ultralytics) |
| `audio` | Audio recording/playback |
| `i2c` | I²C devices (ADS1x15, smbus2) — Linux/Pi only at runtime |
| `behavior` | [Behavior analysis](../camera-behavior/behavior.md) (UMAP, HDBSCAN, scikit-learn, LightGBM) |
| `dev` | Test/lint tooling plus the full `pc` + `vision` + `i2c` + `behavior` stack |

!!! note "You can add extras later"
    Re-run `uv sync` with more `--extra` flags at any time — for example, add
    tracking after the fact with `uv sync --extra pc --extra vision`.

## Recording prerequisites (FFmpeg)

Video recording relies on **FFmpeg**. Install it once:

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Windows"

    ```powershell
    winget install "FFmpeg (Essentials Build)"
    ```

=== "Linux / Raspberry Pi"

    ```bash
    sudo apt install ffmpeg
    ```

## Updating

=== "From source"

    GLIDER compares your copy with the `main` branch on GitHub at launch and
    offers to pull any new changes (**Help → Check for Updates…**). If an
    update changes GLIDER's dependencies, it says so: re-run the installer.

    Or re-run the [one-command installer](#one-command) with the same extras, or
    by hand:

    ```bash
    cd glider
    git pull
    uv sync --extra pc      # repeat any extras you use
    ```

## Verify your install

From source, `glider doctor` reports whether this machine can run GLIDER: the
Python version, Qt and the core libraries, each optional extra, FFmpeg, and
which device tracking will run on. Anything wrong comes with the command that
fixes it, and an extra you never installed is listed as such, not as a fault.

```bash
uv run glider doctor
```

(On a Pi installed by hand as above: `.venv/bin/glider doctor`.)

Then launch GLIDER (`uv run glider`, or open the app). You should see the graph
editor with a **File / Edit / Experiment / View / Hardware / Run / Tools / Help**
menu bar. To confirm your machine's compute capabilities (useful before
[tracking](../camera-behavior/tracking.md)), open the **Tools** tab and
choose **GPU / Device Check**.

Next: [build your first experiment](first-experiment.md).
