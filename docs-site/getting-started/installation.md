# Installation

There are two ways to install GLIDER:

- **Prebuilt app** — the easiest path for most people. Download an installer and
  run it. Best if you just want to *use* GLIDER.
- **From source** — installs the Python project with [uv](https://docs.astral.sh/uv/).
  Best if you want the very latest changes, are on a Raspberry Pi, or plan to
  develop.

!!! info "Supported Python"
    GLIDER runs on **Python 3.11–3.13** (3.14+ is not yet supported). If you
    install from source with `uv`, a compatible Python is installed for you —
    you don't need to manage Python yourself.

## Prebuilt app

Prebuilt installers are published on the project's
[**GitHub Releases**](https://github.com/LaingLab/glider/releases) page.

=== "macOS"

    1. Download the `.dmg` for your Mac from the Releases page. There is one per
       architecture — **Apple Silicon** (`arm64`) and **Intel** (`x86_64`).
       If you're unsure, choose Apple Silicon for M1/M2/M3/M4 Macs.
    2. Open the `.dmg` and drag **GLIDER** into your **Applications** folder.
    3. Launch it from Applications.

    !!! tip "“GLIDER can't be opened” on first launch"
        Because the app isn't notarized, macOS may block the first launch.
        Right-click the app → **Open**, then confirm. You only need to do this
        once.

=== "Windows"

    1. Download the Windows build from the Releases page.
    2. Run the executable.

    !!! note "FFmpeg for recording"
        To record video you also need FFmpeg — see [Recording prerequisites](#recording-prerequisites-ffmpeg)
        below.

=== "Raspberry Pi"

    For a Pi, the easiest option is the **prebuilt SD-card image**, which boots
    straight into GLIDER's touchscreen [Runner](../runner/index.md). See
    [Raspberry Pi Kiosk](../runner/pi-kiosk.md) for flashing and setup. To
    install onto an existing Pi instead, use the [from-source](#from-source)
    steps.

## From source

GLIDER uses [uv](https://docs.astral.sh/uv/) to manage its virtual environment
and Python version.

### 1. Install uv

=== "macOS / Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    winget install astral-sh.uv
    ```

### 2. Get GLIDER and install it

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
    *not* a core dependency), so the virtual environment must be allowed to see
    system packages:

    ```bash
    sudo apt install python3-pyqt6
    git clone https://github.com/LaingLab/glider.git
    cd glider
    uv venv --system-site-packages
    uv sync --extra rpi --extra i2c
    uv run glider
    ```

`uv run glider` launches the app. The first time you run it, GLIDER offers a
short interactive tour — take it or skip it.

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

=== "Prebuilt app"

    Download the newer installer from the
    [Releases](https://github.com/LaingLab/glider/releases) page and reinstall.
    GLIDER also checks for updates and will let you know when a new version is
    available (**Help → Check for Updates…**).

=== "From source"

    ```bash
    cd glider
    git pull
    uv sync --extra pc      # repeat any extras you use
    ```

## Verify your install

Launch GLIDER (`uv run glider`, or open the app). You should see the graph
editor with a **File / Edit / Experiment / View / Hardware / Run / Tools / Help**
menu bar. To confirm your machine's compute capabilities (useful before
[tracking](../camera-behavior/tracking.md)), open **Tools → GPU / Device Check**.

Next: [build your first experiment](first-experiment.md).
