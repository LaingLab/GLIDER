#!/bin/sh
# Install GLIDER from source, or update an install this made, on macOS, Linux
# or a Raspberry Pi. (Windows: install.ps1.)
#
#   curl -LsSf https://raw.githubusercontent.com/LaingLab/glider/main/install.sh | sh
#   curl -LsSf https://raw.githubusercontent.com/LaingLab/glider/main/install.sh | sh -s -- --extra vision
#
# It goes in ~/glider unless GLIDER_DIR says otherwise. Run it again to update;
# pass every --extra you want each time, since extras not named are removed.
set -eu

REPO=https://github.com/LaingLab/glider.git
DIR=${GLIDER_DIR:-$HOME/glider}

EXTRAS=""
while [ $# -gt 0 ]; do
    case $1 in
        --extra) EXTRAS="$EXTRAS ${2:?--extra needs a name}"; shift 2 ;;
        --extra=*) EXTRAS="$EXTRAS ${1#--extra=}"; shift ;;
        *) echo "Unknown option: $1 (the only option is --extra NAME)" >&2; exit 2 ;;
    esac
done

say() { printf '\n==> %s\n' "$*"; }
has() { command -v "$1" >/dev/null 2>&1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then SUDO=sudo; fi

PI=false
if grep -aqs "Raspberry Pi" /proc/device-tree/model; then PI=true; fi

if ! has uv; then
    say "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Where uv's installer puts it; this shell has not re-read its profile.
    PATH="${XDG_BIN_HOME:-$HOME/.local/bin}:$PATH"
fi

if $PI; then
    # PyQt6 comes from apt on the Pi; pip has no wheel for it there. The Qt
    # modules are the set the Pi image installs.
    say "Installing git, FFmpeg and PyQt6 from apt"
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q git ffmpeg python3-pyqt6 python3-pyqt6.qtsvg python3-pyqt6.qtopengl
elif [ "$(uname -s)" = Darwin ]; then
    # /usr/bin/git exists on every Mac, but it is only a stub until Apple's
    # command line tools are installed, and that installer is a dialog.
    if ! xcode-select -p >/dev/null 2>&1; then
        xcode-select --install 2>/dev/null || true
        echo "Finish installing Apple's command line tools (they include git), then run this again." >&2
        exit 1
    fi
elif ! has git; then
    if ! has apt-get; then
        echo "Install git with your package manager, then run this again." >&2
        exit 1
    fi
    say "Installing git"
    $SUDO apt-get update -q
    $SUDO apt-get install -y -q git
fi

if [ -d "$DIR/.git" ]; then
    say "Updating $DIR"
    git -C "$DIR" pull --ff-only
elif [ -e "$DIR" ]; then
    echo "$DIR already exists and is not a GLIDER checkout. Set GLIDER_DIR to install somewhere else." >&2
    exit 1
else
    say "Downloading GLIDER into $DIR"
    git clone "$REPO" "$DIR"
fi
cd "$DIR"

say "Installing GLIDER's Python packages"
if $PI; then
    # The venv must be the system Python's, or apt's PyQt6 is invisible to it.
    # uv sync and uv run would replace it with one on the Python that
    # .python-version pins, so the Pi installs with uv pip and starts the
    # venv's own glider -- as the Pi image does.
    extras=rpi,i2c
    for extra in $EXTRAS; do extras="$extras,$extra"; done
    rm -rf .venv
    uv venv --system-site-packages --python /usr/bin/python3
    uv pip install -e ".[$extras]"
    GLIDER=.venv/bin/glider
else
    set -- --extra pc
    for extra in $EXTRAS; do set -- "$@" --extra "$extra"; done
    uv sync "$@"
    GLIDER="uv run glider"
fi

# A launcher where Spotlight (or the desktop's app menu) looks, so GLIDER can be
# opened by searching for it. Not on the Pi, whose image ships its own.
if ! $PI; then
    say "Adding GLIDER to your applications"
    $GLIDER shortcut || echo "Could not add the launcher; GLIDER still runs from the command below."
fi

say "Checking the install"
if $GLIDER doctor; then
    say "Done. Start GLIDER with:"
else
    say "Installed, but doctor found problems (above). Once they are fixed, start GLIDER with:"
fi
echo "    cd $DIR && $GLIDER"
$PI || echo "or search for GLIDER in Spotlight / your app menu."
