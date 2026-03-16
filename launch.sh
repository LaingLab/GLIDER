#!/usr/bin/env bash
# GLIDER Launcher for macOS and Linux
# Installs uv if needed, creates venv, syncs dependencies, and launches GLIDER.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect platform
IS_PI=false
if [ -f /proc/device-tree/model ] && grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
    IS_PI=true
fi

# --- Install uv if not found ---
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add to PATH for this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installed but not found on PATH."
        echo "Restart your terminal and run this script again."
        exit 1
    fi
    echo "uv installed successfully."
fi

# --- Create venv and sync dependencies ---
if [ "$IS_PI" = true ]; then
    echo "Raspberry Pi detected - using system site-packages..."
    if [ ! -d ".venv" ]; then
        uv venv --system-site-packages
    fi
    uv sync
else
    echo "Desktop detected - installing with PC extras..."
    if [ ! -d ".venv" ]; then
        uv venv
    fi
    uv sync --extra pc
fi

# --- Launch GLIDER ---
echo "Launching GLIDER..."
if [ "$IS_PI" = true ]; then
    uv run glider --runner "$@"
else
    uv run glider "$@"
fi
