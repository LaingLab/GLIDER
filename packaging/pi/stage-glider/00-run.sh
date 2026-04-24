#!/bin/bash
# pi-gen stage script — runs inside the image chroot as root.
#
# Responsibilities:
#   1. Create the dedicated `glider` user and add it to hardware groups.
#   2. Clone GLIDER into /opt/glider/repo at the requested release tag.
#   3. Install GLIDER into a venv with system-site-packages (so the apt-
#      installed python3-pyqt6 is visible).
#   4. Register the systemd unit that launches Runner mode on boot.
#   5. Configure lightdm autologin for the glider user.
#   6. Install the in-app updater helper and its sudoers rule.
#
# All paths are inside the chroot rootfs — no references to the build host.
set -euo pipefail

# Version/ref to clone. CI overrides this via pi-gen's environment.
# Default to ``main`` so a manual local build still works.
GLIDER_REF="${GLIDER_REF:-main}"
GLIDER_REPO_URL="${GLIDER_REPO_URL:-https://github.com/LaingLab/glider.git}"

# --- User and groups ---------------------------------------------------------

if ! id -u glider >/dev/null 2>&1; then
    useradd --create-home \
            --shell /bin/bash \
            --groups gpio,video,dialout,i2c,spi,plugdev \
            glider
fi

# --- Clone + install ---------------------------------------------------------

install -d -o glider -g glider /opt/glider

sudo -u glider git clone --depth 1 --branch "$GLIDER_REF" \
    "$GLIDER_REPO_URL" /opt/glider/repo

# --system-site-packages so the apt-installed python3-pyqt6 and python3-opencv
# are visible. Those are the two heaviest deps and building them from source
# on the Pi is a non-starter.
sudo -u glider python3 -m venv --system-site-packages /opt/glider/venv
sudo -u glider /opt/glider/venv/bin/pip install --upgrade pip wheel
sudo -u glider /opt/glider/venv/bin/pip install -e /opt/glider/repo

# --- Systemd unit ------------------------------------------------------------

install -m 0644 files/glider.service /etc/systemd/system/glider.service
systemctl enable glider.service

# --- Autologin ---------------------------------------------------------------
#
# lightdm runs before our systemd unit (which has After=graphical.target), so
# we set it to autologin the glider user. The kiosk UI owns the display from
# the moment the user is logged in.
mkdir -p /etc/lightdm
cat > /etc/lightdm/lightdm.conf.d/10-glider-autologin.conf <<'EOF'
[Seat:*]
autologin-user=glider
autologin-user-timeout=0
EOF

# --- Updater helper + sudoers -----------------------------------------------

install -d /opt/glider/scripts
install -m 0755 files/glider-update.sh /opt/glider/scripts/glider-update.sh

# Narrow sudoers entry: the glider user can run *only* the updater script
# without a password. Nothing else.
install -m 0440 files/sudoers-glider /etc/sudoers.d/glider

# --- Desktop integration ----------------------------------------------------
#
# Install the .desktop entry and icon so the app shows in the Pi OS menu and
# in Alt-Tab. Sourced from packaging/linux/ in the repo we just cloned —
# keeps a single source of truth.

install -D -m 0644 /opt/glider/repo/packaging/linux/glider.desktop \
    /usr/share/applications/glider.desktop

install -D -m 0644 /opt/glider/repo/packaging/linux/glider.png \
    /usr/share/icons/hicolor/512x512/apps/glider.png

# Refresh the icon cache so the icon resolves immediately.
gtk-update-icon-cache -q /usr/share/icons/hicolor || true

# --- Data directory ---------------------------------------------------------

install -d -o glider -g glider /home/glider/data

# --- Log tail for debugging --------------------------------------------------

echo "stage-glider: build complete. Glider ref: $GLIDER_REF"
