# macOS `.dmg` — build notes

This directory builds a macOS application bundle and packages it into a
disk image for distribution.

1. **PyInstaller** freezes the app into `dist/GLIDER.app` (one-folder bundle),
   using [`glider.spec`](glider.spec).
2. The [`release-macos.yml`](../../.github/workflows/release-macos.yml) workflow
   wraps that `.app` into `GLIDER-<version>-<arch>.dmg` with an `/Applications`
   drag target, **one per architecture** (Apple Silicon `arm64` on `macos-14`,
   Intel `x86_64` on `macos-13`).

## How releases are cut

- **Tag push** (`git tag vX.Y.Z && git push origin vX.Y.Z`) → both DMGs are
  built and attached to a **draft** GitHub Release for that tag. A human
  publishes the release after a smoke test.
- **Manual** `workflow_dispatch` (Actions → *release-macos* → *Run workflow*)
  → the DMGs upload as downloadable **artifacts**, no release created. Use this
  to test spec changes without cutting a tag.

## Build locally (the preferred path)

Requires macOS with Xcode command-line tools (for `hdiutil`). One command
freezes the app, smoke-tests it, and packages the `.dmg`:

```bash
uv sync --extra pc --extra dev       # once: app deps + PyInstaller
source .venv/bin/activate
./packaging/macos/build.sh           # -> dist/GLIDER-<version>-<arch>.dmg
```

Or drive PyInstaller directly if you only want the `.app`:

```bash
uv run pyinstaller packaging/macos/glider.spec --clean --noconfirm
# -> dist/GLIDER.app
```

The build is per-architecture — run it on an Apple Silicon Mac for the `arm64`
DMG and an Intel Mac for `x86_64`. `release-macos.yml` runs the same steps on
CI runners if you ever want that, but local builds are the default here.

The bundle icon is [`../icons/glider.icns`](../icons), generated from
`glider_source_original.png` (2048²) with `sips` + `iconutil` and committed so
local and CI builds don't need a generation step. Regenerate only if the source
art changes.

## Known gaps (tracked, not blockers)

| Item | Status | Notes |
|---|---|---|
| **Code signing + notarization** | **Not done** | The DMG is unsigned. Gatekeeper shows *"Apple cannot check it for malicious software"*; users right-click → **Open** once, or run `xattr -dr com.apple.quarantine /Applications/GLIDER.app`. Fixing needs an Apple Developer ID (~$99/yr) — the `release-macos.yml` codesign/notarize steps are stubbed with a TODO. |
| **Universal binary** | Per-arch instead | PyQt6/opencv don't all ship `universal2` wheels, so we build one DMG per arch rather than a single fat binary. Intel Macs use the `x86_64` DMG; Apple Silicon uses `arm64`. |
| **Camera / microphone permissions** | Handled | `Info.plist` declares `NSCameraUsageDescription` + `NSMicrophoneUsageDescription`; without these macOS kills the app on camera access. |
