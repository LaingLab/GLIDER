#!/usr/bin/env bash
# Render the launch splash to src/glider/assets/splash.gif.
# Needs Node, ffmpeg, and a Python with Pillow (the repo venv has it).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../../.venv/bin/python}"

"$PY" make_art.py
[ -d node_modules ] || npm ci
rm -rf out && npx remotion render src/index.ts Splash out/frames --sequence --image-format=png
# One palette for the whole clip, 1-bit alpha for the rounded corners, play once (-loop -1).
ffmpeg -y -loglevel error -framerate 30 -i out/frames/element-%03d.png \
  -vf "split[a][b];[a]palettegen=max_colors=256:reserve_transparent=1:stats_mode=diff[p];[b][p]paletteuse=dither=none:alpha_threshold=128" \
  -loop -1 ../../src/glider/assets/splash.gif
ls -la ../../src/glider/assets/splash.gif
