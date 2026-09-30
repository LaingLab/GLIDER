"""Small file-writing helpers shared by the core modules."""

import contextlib
import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` so a failure never leaves a truncated file.

    Serialize before calling this: the old file is untouched until the new
    one is complete and fsynced beside it, then ``os.replace`` swaps it in.
    Keeps the existing file's mode (mkstemp would otherwise make it 0600).
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def unique_path(path: Path) -> Path:
    """``path``, or ``stem_1.ext``, ``stem_2.ext``... if it already exists."""
    candidate, n = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate
