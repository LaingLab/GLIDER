"""Find a command-line tool, including where a GUI launch cannot see it.

``shutil.which`` searches ``PATH``, and that is the whole problem: an app
launched from the Dock, Finder or a ``.app`` bundle does not inherit the shell's
``PATH``. It gets launchd's, which on a normal Mac is roughly
``/usr/bin:/bin:/usr/sbin:/sbin`` -- and can be narrower still. Measured on the
machine that produced this module, a GUI-launched GLIDER had a ``PATH`` of one
directory, and it was Flutter's.

Everything a lab installs lands outside that set. ``uv``'s own installer writes
to ``~/.local/bin``; Homebrew uses ``/opt/homebrew/bin`` on Apple Silicon and
``/usr/local/bin`` on Intel. So ``shutil.which`` answers "not installed" for
tools that are installed, and only when GLIDER is started the way users
actually start it -- which is why this survived: from a terminal, everything
works.

Two things went wrong because of it, and the second is the one that shows the
class:

* **Plugins could not be installed at all on macOS.** ``uv venv`` does not
  install pip, so on GLIDER's *documented* setup uv is the only installer there
  is -- and it was invisible.
* **ffmpeg went missing**, taking audio recording and the camera manager's
  ffmpeg paths with it, silently, on the same machines.

This is deliberately not a ``PATH`` fix-up. Rewriting the process environment
would change what every subprocess GLIDER ever launches can see, to fix a
question two modules ask; answering the question honestly is smaller and does
not have side effects on things that are working.
"""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

__all__ = ["EXTRA_BIN_DIRS", "find_executable"]


def _extra_bin_dirs() -> tuple[Path, ...]:
    """Directories a GUI launch misses, most-likely first.

    Ordered by how a lab machine actually gets these tools rather than
    alphabetically: uv's installer default, then cargo, then Homebrew's two
    prefixes, then the traditional local prefix. ``~/.local/bin`` also covers
    uv on Windows, whose installer uses the same directory.
    """
    home = Path.home()
    return (
        home / ".local" / "bin",
        home / ".cargo" / "bin",
        Path("/opt/homebrew/bin"),
        Path("/usr/local/bin"),
        Path("/opt/local/bin"),
    )


#: Exposed so a caller can say *where* it looked when it reports a tool missing.
EXTRA_BIN_DIRS = _extra_bin_dirs()


@lru_cache(maxsize=32)
def find_executable(name: str) -> str | None:
    """Absolute path to ``name``, or ``None`` if it genuinely is not installed.

    ``PATH`` first, so a shell-launched GLIDER and anything the user has
    deliberately put in front keep winning. Only then the fallbacks above.

    Args:
        name: The command, without an extension. ``.exe`` is appended on
            Windows for the fallback lookup; ``shutil.which`` already handles
            it for the ``PATH`` pass.

    Cached because both callers ask repeatedly -- the installer on every
    catalogue build, the recorder on every check -- and the answer cannot
    change without a restart in any way that matters. A test that installs a
    tool mid-run should call ``find_executable.cache_clear()``.
    """
    found = shutil.which(name)
    if found:
        return found

    filename = f"{name}.exe" if sys.platform == "win32" else name
    for directory in EXTRA_BIN_DIRS:
        candidate = directory / filename
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except OSError:  # pragma: no cover - unreadable directory
            continue
    return None
