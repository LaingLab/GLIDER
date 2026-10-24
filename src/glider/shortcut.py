"""``glider shortcut``: make GLIDER findable by name from the OS's own search.

A source install (``install.sh`` / ``install.ps1``) used to end with "cd ~/glider
&& uv run glider", which nobody remembers a week later. This puts a launcher
where the OS search looks (#216):

* macOS: ``~/Applications/GLIDER.app``, which Spotlight and Launchpad index.
* Windows: a Start Menu shortcut, which Windows Search finds.
* Linux: a ``.desktop`` entry in ``~/.local/share/applications``.

Each launches *this* interpreter with ``-m glider``, so it starts the same
environment the command was run from. The packaged DMG and installer builds do
not need this: they are already apps.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

#: The repo's icons; present in a source checkout, which is what this is for.
_ICONS = Path(__file__).resolve().parents[2] / "packaging" / "icons"


def _mac_app(home: Path, python: Path) -> Path:
    app = home / "Applications" / "GLIDER.app"
    contents = app / "Contents"
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(parents=True, exist_ok=True)
    launcher = contents / "MacOS" / "GLIDER"
    launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(str(python))} -m glider "$@"\n')
    launcher.chmod(0o755)
    icon = _ICONS / "glider.icns"
    if icon.exists():
        shutil.copyfile(icon, contents / "Resources" / "glider.icns")
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(
            {
                "CFBundleName": "GLIDER",
                "CFBundleDisplayName": "GLIDER",
                "CFBundleExecutable": "GLIDER",
                # Not the DMG build's identifier, so the two never collide.
                "CFBundleIdentifier": "com.lainglab.glider.source",
                "CFBundleIconFile": "glider.icns",
                "CFBundlePackageType": "APPL",
                "NSHighResolutionCapable": True,
            },
            f,
        )
    return app


def _windows_shortcut(python: Path, run=subprocess.run) -> Path:
    programs = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    link = programs / "GLIDER.lnk"
    # pythonw: no console window behind the app.
    pythonw = python.with_name("pythonw.exe")
    target = pythonw if pythonw.exists() else python
    # Values travel as environment variables, so no path needs PowerShell quoting.
    env = dict(
        os.environ,
        GLIDER_LNK=str(link),
        GLIDER_TARGET=str(target),
        GLIDER_WORKDIR=str(python.parents[2]),  # .venv\Scripts\python.exe -> checkout
        GLIDER_ICON=str(_ICONS / "glider.ico"),
    )
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:GLIDER_LNK);"
        "$s.TargetPath = $env:GLIDER_TARGET; $s.Arguments = '-m glider';"
        "$s.WorkingDirectory = $env:GLIDER_WORKDIR;"
        "if (Test-Path $env:GLIDER_ICON) { $s.IconLocation = $env:GLIDER_ICON };"
        "$s.Save()"
    )
    run(["powershell", "-NoProfile", "-Command", script], env=env, check=True)
    return link


def _linux_entry(home: Path, python: Path) -> Path:
    entry = home / ".local" / "share" / "applications" / "glider.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    icon = _ICONS.parent / "linux" / "glider.png"
    entry.write_text(
        "[Desktop Entry]\nType=Application\nName=GLIDER\n"
        "Comment=General Laboratory Interface for Design, Experimentation, and Recording\n"
        f"Exec={shlex.quote(str(python))} -m glider\n"
        f"Icon={icon if icon.exists() else 'glider'}\n"
        "Terminal=false\nCategories=Science;Education;\nStartupWMClass=GLIDER\n"
    )
    return entry


def create_shortcut(platform: str = sys.platform, home: Path | None = None, run=subprocess.run):
    """Create the launcher for *platform*; return where it was put."""
    python = Path(sys.executable)
    home = home or Path.home()
    if platform == "darwin":
        return _mac_app(home, python)
    if platform == "win32":
        return _windows_shortcut(python, run=run)
    return _linux_entry(home, python)


def main() -> int:
    try:
        where = create_shortcut()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not create the GLIDER shortcut: {exc}", file=sys.stderr)
        return 1
    print(f"GLIDER can now be opened by searching for it. Shortcut: {where}")
    return 0
