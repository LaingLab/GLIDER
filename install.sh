#!/usr/bin/env bash
# GLIDER One-Click Installer for macOS and Linux
# Clones the repo, installs uv, syncs dependencies, launches GLIDER, and creates a desktop shortcut.
set -euo pipefail

REPO_URL="https://github.com/LaingLab/glider.git"
INSTALL_DIR="$HOME/GLIDER"

# Detect platform
OS="$(uname -s)"
IS_PI=false
if [ -f /proc/device-tree/model ] && grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
    IS_PI=true
fi

echo "============================================"
echo "  GLIDER Installer"
echo "============================================"
echo ""

# --- Install git if missing ---
if ! command -v git &>/dev/null; then
    echo "Installing git..."
    if [ "$OS" = "Darwin" ]; then
        xcode-select --install 2>/dev/null || true
        echo "Please complete the Xcode Command Line Tools prompt, then re-run this script."
        exit 1
    elif command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y git
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y git
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm git
    else
        echo "ERROR: git is required. Please install it and re-run."
        exit 1
    fi
fi

# --- Install uv if missing ---
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installed but not found on PATH."
        echo "Restart your terminal and re-run this script."
        exit 1
    fi
fi

# --- Install system PyQt6 on Raspberry Pi ---
if [ "$IS_PI" = true ]; then
    echo "Raspberry Pi detected - installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-pyqt6
fi

# --- Clone or update repo ---
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "Updating existing GLIDER installation..."
    cd "$INSTALL_DIR"
    git pull --ff-only || echo "Warning: could not fast-forward, using existing version."
else
    echo "Cloning GLIDER..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# --- Create venv and sync ---
echo "Setting up environment..."
if [ "$IS_PI" = true ]; then
    [ -d ".venv" ] || uv venv --system-site-packages
    uv sync
else
    [ -d ".venv" ] || uv venv
    uv sync --extra pc
fi

# --- Create desktop shortcut ---
echo "Creating desktop shortcut..."

DESKTOP_DIR=""
if [ "$OS" = "Darwin" ]; then
    DESKTOP_DIR="$HOME/Desktop"
else
    DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
fi
mkdir -p "$DESKTOP_DIR"

if [ "$OS" = "Darwin" ]; then
    # macOS: create a .command file
    SHORTCUT="$DESKTOP_DIR/GLIDER.command"
    cat > "$SHORTCUT" << 'SCRIPT'
#!/usr/bin/env bash
cd "$HOME/GLIDER" && ./launch.sh
SCRIPT
    chmod +x "$SHORTCUT"

elif [ "$IS_PI" = true ]; then
    # Raspberry Pi: .desktop file launching in runner mode
    cat > "$DESKTOP_DIR/glider.desktop" << EOF
[Desktop Entry]
Name=GLIDER
Comment=General Laboratory Interface for Design, Experimentation, and Recording
Exec=bash -c 'cd $INSTALL_DIR && ./launch.sh'
Icon=utilities-terminal
Terminal=false
Type=Application
Categories=Science;Education;
EOF
    chmod +x "$DESKTOP_DIR/glider.desktop"

else
    # Linux desktop: .desktop file
    cat > "$DESKTOP_DIR/glider.desktop" << EOF
[Desktop Entry]
Name=GLIDER
Comment=General Laboratory Interface for Design, Experimentation, and Recording
Exec=bash -c 'cd $INSTALL_DIR && ./launch.sh'
Icon=utilities-terminal
Terminal=false
Type=Application
Categories=Science;Education;
EOF
    chmod +x "$DESKTOP_DIR/glider.desktop"
    # Mark as trusted on GNOME-based desktops
    command -v gio &>/dev/null && gio set "$DESKTOP_DIR/glider.desktop" metadata::trusted true 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "  GLIDER installed successfully!"
echo "  Location: $INSTALL_DIR"
echo "  Shortcut: Desktop"
echo "============================================"
echo ""

# --- Launch ---
echo "Launching GLIDER..."
if [ "$IS_PI" = true ]; then
    uv run glider --runner
else
    uv run glider
fi
