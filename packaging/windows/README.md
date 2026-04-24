# Windows installer — build notes

This directory scaffolds a two-stage Windows build:

1. **PyInstaller** freezes the Python application into `dist\GLIDER\` (one-folder
   layout, ~300-400MB with Qt, OpenCV, and telemetrix-aio).
2. **Inno Setup** wraps that folder into `Output\glider-setup-<version>.exe`,
   adding Start Menu entries, an uninstaller, and upgrade handling.

The installer is currently **unsigned**. See the "Code signing — TODO" section
below.

## Prerequisites (for local builds)

- Windows 10 or 11 x64.
- Python 3.12 with the project installed as `uv sync --extra pc --extra dev`.
- [Inno Setup 6](https://jrsoftware.org/isdl.php) on PATH.

CI builds run on `windows-2022` runners — see
`.github/workflows/release-windows.yml` for the canonical pipeline.

## Build steps

```powershell
# From the repo root:
pyinstaller packaging\windows\glider.spec --clean --noconfirm

# Produce the installer. /DMyAppVersion injects the version; if omitted, it
# falls back to 0.0.0-dev.
ISCC.exe /DMyAppVersion=1.0.0 packaging\windows\installer.iss
```

The installer lands at `Output\glider-setup-<version>.exe`.

## Placeholders before first real build

| File | Current state | TODO |
|---|---|---|
| `..\icons\glider.ico` | Present | Already a multi-resolution 16/32/… `.ico`. No action. |
| `assets\wizard.bmp` / `sidebar.bmp` | Missing | Optional Inno Setup wizard graphics. Default Inno visuals work fine if omitted — keep unless/until we want branded installer screens. |
| `version_info.txt` | Missing | Windows executable version metadata (company, product, copyright). PyInstaller can generate it via `pyi-set_version` from a template; CI is expected to produce it from `_version.py`. |

The `version_info.txt` reference in `glider.spec` is optional — delete the
`version=` line from the spec while iterating locally if you haven't
generated the file yet.

## Code signing — TODO

The installer ships unsigned in v1. Every first-time downloader will see
Windows SmartScreen's full-screen "Windows protected your PC" warning and must
click *More info → Run anyway*. Acceptable for internal/alpha distribution;
unacceptable for the eventual non-technical-user goal.

When we're ready to sign, replace the empty step in
`.github/workflows/release-windows.yml` with either:

**Azure Trusted Signing** (recommended): ~$10/mo, GitHub-Actions-native,
keys stay in Azure. Uses the
[`azure/trusted-signing-action`](https://github.com/Azure/trusted-signing-action)
action. Requires an Azure tenant and a completed identity verification.

**EV cert via cloud HSM** (e.g. DigiCert KeyLocker, SSL.com eSigner):
$400-700/yr. Installs `signtool.exe` in CI, authenticates to the cert
vendor's cloud HSM, signs the produced `.exe`. Wire via each vendor's
documented GitHub Actions pattern.

Both give instant SmartScreen trust — the OV-cert "reputation ramp" path is
**not recommended** because our download volume is too low to ever build
enough reputation to clear the warning.

## YOLO / ultralytics

Deliberately **excluded** from the bundle (`excludes=[...]` in `glider.spec`).
`ultralytics` is AGPL-3.0 and redistributing it imposes source-availability
obligations on any downstream user. The app's CV-tracking nodes detect its
absence at first use and offer to download it into the bundled Python
environment. This keeps the base installer permissively-licensed.

The download-on-demand helper is not yet implemented in `src/glider/vision/`.
See follow-up work.

## Known gotchas

- **OpenCV DLL loading.** PyInstaller's OpenCV hook occasionally misses the
  secondary DLLs (`opencv_videoio_ffmpeg*`, etc.). `glider.spec` calls
  `collect_dynamic_libs("cv2")` explicitly to catch them.
- **Per-user install.** The installer runs without UAC and writes under
  `%LocalAppData%\Programs\GLIDER`. Shared-lab machines where different
  users log in each need their own install. Flip to per-machine by changing
  `PrivilegesRequired=admin` in `installer.iss` and replacing `{autopf}` —
  start per-user and revisit if needed.
- **Windows 10 minimum.** `MinVersion=10.0.17763` in `installer.iss` rejects
  anything older than Windows 10 1809. If a lab still runs Windows 7,
  Python 3.12 and modern PyQt6 don't support it anyway.
