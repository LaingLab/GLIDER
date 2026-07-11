#!/usr/bin/env bash
#
# Local macOS build: freeze GLIDER.app with PyInstaller and package a .dmg.
#
# Run on a Mac, from anywhere, inside an env that has the app deps + PyInstaller
# (e.g. `uv sync --extra pc --extra dev` then `source .venv/bin/activate`, or
# prefix this script with that env's bin on PATH):
#
#     ./packaging/macos/build.sh
#
# Output: dist/GLIDER.app and dist/GLIDER-<version>-<arch>.dmg
#
# The .dmg is UNSIGNED — Gatekeeper will warn on first open (right-click →
# Open, or `xattr -dr com.apple.quarantine /Applications/GLIDER.app`). See
# README.md for the signing/notarization path.

set -euo pipefail

# Repo root, regardless of where the script is invoked from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Version from the single source of truth; arch from the host (arm64 / x86_64).
VERSION="$(python -c "import sys; sys.path.insert(0, 'src'); from glider._version import __version__; print(__version__)")"
ARCH="$(uname -m)"

echo ">> Building GLIDER ${VERSION} (${ARCH})"

python -m PyInstaller packaging/macos/glider.spec --clean --noconfirm

echo ">> Smoke-testing the frozen app"
# --version exits inside argparse, before any Qt window init.
dist/GLIDER.app/Contents/MacOS/GLIDER --version

echo ">> Packaging .dmg"
DMG="dist/GLIDER-${VERSION}-${ARCH}.dmg"
STAGING="$(mktemp -d)"
cp -R "dist/GLIDER.app" "${STAGING}/"
# Drag-to-install target: users drop GLIDER.app onto Applications.
ln -s /Applications "${STAGING}/Applications"
rm -f "${DMG}"
hdiutil create -volname "GLIDER ${VERSION}" \
  -srcfolder "${STAGING}" -ov -format UDZO "${DMG}" >/dev/null
rm -rf "${STAGING}"

echo ">> Done: ${DMG} ($(du -h "${DMG}" | cut -f1))"
