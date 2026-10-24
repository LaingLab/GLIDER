"""`glider shortcut` puts a launcher where the OS search looks (#216)."""

from __future__ import annotations

import plistlib
import sys

from glider.shortcut import create_shortcut


def test_macos_gets_an_app_in_the_users_applications(tmp_path):
    app = create_shortcut("darwin", home=tmp_path)

    assert app == tmp_path / "Applications" / "GLIDER.app"
    with open(app / "Contents" / "Info.plist", "rb") as f:
        info = plistlib.load(f)
    assert info["CFBundleExecutable"] == "GLIDER"
    launcher = app / "Contents" / "MacOS" / "GLIDER"
    assert launcher.stat().st_mode & 0o111  # executable
    assert sys.executable in launcher.read_text()
    assert "-m glider" in launcher.read_text()


def test_making_it_twice_just_refreshes_it(tmp_path):
    create_shortcut("darwin", home=tmp_path)
    create_shortcut("darwin", home=tmp_path)


def test_windows_gets_a_start_menu_shortcut_to_this_python(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    calls = []
    link = create_shortcut("win32", run=lambda args, env, check: calls.append((args, env)))

    assert link == tmp_path / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "GLIDER.lnk"
    args, env = calls[0]
    assert args[0] == "powershell" and "CreateShortcut" in args[-1]
    assert env["GLIDER_LNK"] == str(link)
    assert "python" in env["GLIDER_TARGET"].lower()


def test_linux_gets_a_desktop_entry(tmp_path):
    entry = create_shortcut("linux", home=tmp_path)

    assert entry == tmp_path / ".local" / "share" / "applications" / "glider.desktop"
    assert "-m glider" in entry.read_text()
