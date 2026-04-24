#!/bin/bash
# In-app updater for the Raspberry Pi build.
#
# Called via sudo (see /etc/sudoers.d/glider) from the GLIDER app when the
# user chooses "Update GLIDER" from Settings. Pulls the latest tagged
# release, reinstalls into the existing venv, and restarts the systemd
# service.
#
# OS-level updates (kernel, apt packages) still require a full SD reflash —
# that's the trade-off for a baked kiosk image. If we ever need online
# OS updates we'd invoke `apt-get upgrade` here, but that's a much bigger
# blast radius and we'd want a recovery story first.
set -euo pipefail

REPO_DIR="/opt/glider/repo"
VENV_DIR="/opt/glider/venv"
LOG_FILE="/var/log/glider-update.log"

# Route both stdout and stderr to the log for later troubleshooting, while
# still letting the caller see them.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== glider-update.sh at $(date -Iseconds) ==="

cd "$REPO_DIR"

# Fetch tags first — "latest" is a tag, not a branch.
sudo -u glider git fetch --tags --prune origin

# Resolve "latest" to the most recent annotated tag. Fall back to main if no
# tags exist yet (shouldn't happen in production but keeps manual testing
# sane).
LATEST_TAG="$(sudo -u glider git describe --tags --abbrev=0 2>/dev/null || echo main)"
echo "Checking out $LATEST_TAG"
sudo -u glider git checkout "$LATEST_TAG"

# Reinstall. -e keeps it editable so future `git checkout` swaps are picked
# up without another pip install — but we run pip install anyway to refresh
# any new Python deps added in the release.
sudo -u glider "$VENV_DIR/bin/pip" install --upgrade -e "$REPO_DIR"

# Restart the kiosk. The user's display session will show GLIDER re-launching.
systemctl restart glider.service

echo "=== update complete ==="
