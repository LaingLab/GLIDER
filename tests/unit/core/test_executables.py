"""Finding a tool a GUI launch cannot see.

``shutil.which`` searches PATH, and an app launched from the Dock does not
inherit the shell's. Measured on the machine this was written on, a
GUI-launched GLIDER had a PATH of exactly one directory -- Flutter's -- so uv
at ``~/.local/bin/uv`` was invisible, and since ``uv venv`` installs no pip,
plugins could not be installed at all. From a terminal everything worked, which
is why it survived.
"""

from __future__ import annotations

import os
import sys

import pytest

from glider.core.executables import find_executable


@pytest.fixture(autouse=True)
def _no_cache():
    """The lookup is cached for the process; these tests change the answer."""
    find_executable.cache_clear()
    yield
    find_executable.cache_clear()


def _make_tool(directory, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    tool = directory / (f"{name}.exe" if sys.platform == "win32" else name)
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    return tool


def test_path_wins(tmp_path, monkeypatch):
    """A tool the user has deliberately put in front stays in front."""
    on_path = _make_tool(tmp_path / "bin", "widget")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))

    assert find_executable("widget") == str(on_path)


def test_it_looks_where_the_dock_cannot(tmp_path, monkeypatch):
    """The bug: PATH has nothing, but the tool is installed in ~/.local/bin."""
    home = tmp_path / "home"
    fallback = _make_tool(home / ".local" / "bin", "widget")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    # EXTRA_BIN_DIRS is computed at import; rebuild it against the fake home.
    import glider.core.executables as module

    monkeypatch.setattr(module, "EXTRA_BIN_DIRS", module._extra_bin_dirs())

    assert find_executable("widget") == str(fallback)


def test_a_tool_that_really_is_absent_is_still_absent(tmp_path, monkeypatch):
    """The fallbacks must not turn 'missing' into a false positive -- the whole
    point of the probe is to report honestly."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    import glider.core.executables as module

    monkeypatch.setattr(module, "EXTRA_BIN_DIRS", module._extra_bin_dirs())

    assert find_executable("definitely-not-a-real-tool") is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_a_non_executable_file_is_not_a_tool(tmp_path, monkeypatch):
    home = tmp_path / "home"
    directory = home / ".local" / "bin"
    directory.mkdir(parents=True)
    (directory / "widget").write_text("not executable")
    os.chmod(directory / "widget", 0o644)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    import glider.core.executables as module

    monkeypatch.setattr(module, "EXTRA_BIN_DIRS", module._extra_bin_dirs())

    assert find_executable("widget") is None
