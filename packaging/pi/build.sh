#!/usr/bin/env bash
#
# Local Raspberry Pi SD-image build: wires stage-glider into pi-gen and runs
# it. One-command equivalent of release-pi.yml for building on your own
# ARM64 Debian/Raspberry Pi OS machine (a Pi 4/5 with a fast SD or SSD works;
# expect roughly an hour and ~15 GB of scratch space).
#
# Run from anywhere on the build machine:
#     sudo ./packaging/pi/build.sh
#
# Output: pi-gen/deploy/GLIDER-*.img.xz (flash with Raspberry Pi Imager).
#
# NOTE: release-pi.yml carries extra workarounds for GitHub's Ubuntu ARM
# runners (Debian keyring injection, stripping unpublished rpi-* packages).
# On a real Raspberry Pi OS box those usually aren't needed; if the build
# fails in stage0 with NO_PUBKEY or on rpi-swap/rpi-loop-utils, borrow the
# corresponding step from that workflow.

set -euo pipefail

if [ "$(uname -m)" != "aarch64" ]; then
    echo "error: pi-gen must run on an ARM64 (aarch64) host; this is $(uname -m)." >&2
    echo "Build on a Raspberry Pi / ARM64 Debian box." >&2
    exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "error: pi-gen needs root for chroots and loop devices — run with sudo." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${PI_GEN_DIR:-${REPO_ROOT}/../pi-gen}"

echo ">> Installing pi-gen build dependencies"
apt-get update
apt-get install -y --no-install-recommends \
    coreutils quilt parted qemu-user-static debootstrap zerofree zip dosfstools \
    libarchive-tools libcap2-bin grep rsync xz-utils file git curl bc arch-test \
    binfmt-support ca-certificates kmod pigz

if [ ! -d "${WORK_DIR}" ]; then
    echo ">> Cloning pi-gen (arm64 branch) into ${WORK_DIR}"
    git clone --depth 1 --branch arm64 https://github.com/RPi-Distro/pi-gen.git "${WORK_DIR}"
fi

echo ">> Wiring stage-glider into pi-gen"
cp "${REPO_ROOT}/packaging/pi/pi-gen-config" "${WORK_DIR}/config"
rm -rf "${WORK_DIR}/stage-glider"
cp -r "${REPO_ROOT}/packaging/pi/stage-glider" "${WORK_DIR}/"
chmod +x "${WORK_DIR}/stage-glider/00-run.sh" \
         "${WORK_DIR}/stage-glider/files/glider-update.sh"

echo ">> Running pi-gen (this takes a while — desktop stage + GLIDER install)"
cd "${WORK_DIR}"
./build.sh

echo ">> Done. Images in ${WORK_DIR}/deploy:"
ls -lh deploy/ | grep -E "img|zip" || ls -lh deploy/
