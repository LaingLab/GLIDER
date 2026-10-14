<div align="center">

<img src="docs-site/assets/logo.png" alt="GLIDER logo" width="140">

# GLIDER

**General Laboratory Interface for Design, Experimentation, and Recording**

Build, run, and record reproducible experiments without writing code.

[![Documentation](https://img.shields.io/badge/docs-online-2f81f7?style=flat-square)](https://lainglab.github.io/GLIDER/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat-square)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux%20%7C%20Raspberry%20Pi-lightgrey?style=flat-square)](#quick-start)
[![Status](https://img.shields.io/badge/status-beta-orange?style=flat-square)](https://github.com/LaingLab/GLIDER/releases)

[**Documentation**](https://lainglab.github.io/GLIDER/) ·
[**Quick start**](#quick-start) ·
[**Contributing**](CONTRIBUTING.md)

</div>

macOS / Linux / Raspberry Pi:

```bash
curl -LsSf https://raw.githubusercontent.com/LaingLab/glider/main/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/LaingLab/glider/main/install.ps1 | iex
```

The installer puts GLIDER in `~/glider` and finishes by running `glider doctor`.
For extras or a manual install, see [Quick start](#quick-start).

---

<div align="center">
  <img src="docs-site/assets/readme/hero-tracking.gif" alt="Live tracking in GLIDER: a zebrafish larva, a mouse and a fly, each followed by its keypoints and trail">
  <p><em>Live tracking in GLIDER: zebrafish · mouse · fly</em></p>
</div>

GLIDER is a desktop application for building, running, and recording laboratory
experiments. You wire a **visual node graph** that drives hardware (Arduino,
Raspberry Pi, Bluetooth), track animals live on camera, record synchronized
video and data, and review every session afterwards. Experiment files are
portable across Windows, macOS, Linux and the Raspberry Pi.

## Prerequisites

1. [uv](https://docs.astral.sh/uv/getting-started/installation/): manages the virtual environment and Python for you
   - Windows: `winget install astral-sh.uv`
   - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. FFmpeg (for video recording)
   - Windows: `winget install "FFmpeg (Essentials Build)"`
   - macOS: `brew install ffmpeg`
   - Linux/Raspberry Pi: `sudo apt install ffmpeg`

> GLIDER requires Python 3.11–3.13 (3.14+ is not supported). **3.12 is the
> version to use**: it is what the release builds ship, and `.python-version`
> pins it so `uv venv` picks it for you. Without that pin `uv` takes the newest
> interpreter on the machine, and a fresh install on 3.13 has been seen to fail
> at startup with `DLL load failed while importing QtCore` on Windows.

## Quick start

One command installs uv and git if they are missing, puts GLIDER in `~/glider`
(or `GLIDER_DIR`), and finishes by running `glider doctor`. Run it again to update.

macOS / Linux / Raspberry Pi:

```bash
curl -LsSf https://raw.githubusercontent.com/LaingLab/glider/main/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/LaingLab/glider/main/install.ps1 | iex
```

Add extras with `| sh -s -- --extra vision --extra behavior`, or on Windows
`& ([scriptblock]::Create((irm https://raw.githubusercontent.com/LaingLab/glider/main/install.ps1))) -Extra vision,behavior`.

`glider doctor` checks an install at any time: Python, Qt, each extra, FFmpeg
and the inference device, each with the fix for whatever is wrong.

### By hand: Windows / macOS / Linux

```bash
git clone https://github.com/LaingLab/GLIDER.git
cd GLIDER
uv venv
uv sync --extra pc
uv run glider
```

### By hand: Raspberry Pi

PyQt6 comes from apt on the Pi (it is intentionally not a core dependency),
so the venv must be the system Python's, with access to its packages. Install
with `uv pip` and start the venv's own `glider`: `uv sync` and `uv run` would
swap the venv for one on the Python that `.python-version` pins, which cannot
see apt's PyQt6.

```bash
sudo apt install python3-pyqt6 python3-pyqt6.qtsvg python3-pyqt6.qtopengl
git clone https://github.com/LaingLab/GLIDER.git
cd GLIDER
uv venv --system-site-packages --python /usr/bin/python3
uv pip install -e ".[rpi,i2c]"
.venv/bin/glider
```

## Optional extras

Add any of these to the sync command with additional `--extra` flags,
e.g. `uv sync --extra pc --extra vision`:

| Extra      | What it adds                                                  |
| ---------- | ------------------------------------------------------------- |
| `pc`       | PyQt6 + audio — the standard desktop install                  |
| `rpi`      | Raspberry Pi GPIO (gpiozero, lgpio)                           |
| `vision`   | YOLO tracking (ultralytics)                                   |
| `audio`    | Audio recording/playback                                      |
| `i2c`      | I2C devices (ADS1x15, smbus2) — Linux/Pi only at runtime      |
| `behavior` | Behavior analysis (UMAP, HDBSCAN, scikit-learn, LightGBM)     |
| `dev`      | Test/lint tooling plus the full pc+vision+i2c+behavior stack  |

## Documentation

Full guides live at **[lainglab.github.io/GLIDER](https://lainglab.github.io/GLIDER/)**:

| Guide | What it covers |
| ----- | -------------- |
| [Getting Started](https://lainglab.github.io/GLIDER/getting-started/) | Install GLIDER and build your first experiment |
| [Building Experiments](https://lainglab.github.io/GLIDER/building/) | The node graph, wiring devices, packaging routines |
| [Camera & Behavior](https://lainglab.github.io/GLIDER/camera-behavior/) | Cameras, synchronized recording, tracking, analysis |
| [Runner Mode](https://lainglab.github.io/GLIDER/runner/) | Operating experiments from a touchscreen or Pi kiosk |

## Development

```bash
uv sync --extra dev

# Tests (Qt needs a virtual display when headless)
QT_QPA_PLATFORM=offscreen uv run pytest tests/

# Lint & format
uv run ruff check .
uv run black .
```

## Citation

If you use GLIDER in your research, please cite it:

```bibtex
@software{glider,
  title  = {GLIDER: General Laboratory Interface for Design, Experimentation, and Recording},
  author = {Laing Lab},
  year   = {2026},
  doi    = {10.17632/vd2h64x7wg},
  url    = {https://github.com/LaingLab/GLIDER}
}
```

The data, analysis code and results are archived at Mendeley Data:
<https://doi.org/10.17632/vd2h64x7wg>

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for development setup, coding standards, and the review process.

## License

Released under the [MIT License](LICENSE).
