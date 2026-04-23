"""GLIDER application assets (icons, brand imagery).

This subpackage ships the runtime icon alongside the Python code so it's
available after ``pip install`` or inside a PyInstaller bundle. The higher-
resolution master lives at ``packaging/icons/glider_source.png`` and is *not*
shipped — it's the source for derived variants under this directory and for
``packaging/icons/glider.ico`` (Windows installer).

Public API:

- :func:`get_app_icon` — multi-resolution :class:`QIcon` for ``QApplication``
  and ``QMainWindow``.
- :func:`get_icon_path` — filesystem path to a specific size, useful when a
  caller needs a file (e.g. ``tray.setIcon(QIcon(str(path)))``).
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon

# Sizes actually shipped under this directory. Keep in sync with the PNG files
# on disk; Qt picks the best match for the current display DPI.
_ICON_SIZES: tuple[int, ...] = (256, 512)


def get_icon_path(size: int = 512) -> Path:
    """Return a filesystem path to the sized icon PNG.

    The file is guaranteed to exist for sizes in :data:`_ICON_SIZES`; other
    sizes will raise :class:`FileNotFoundError` on first read.
    """
    return Path(str(files(__package__) / f"icon_{size}.png"))


def get_app_icon() -> QIcon:
    """Build a multi-resolution :class:`QIcon` for the GLIDER application.

    The returned icon contains every size shipped with the package, so Qt
    can render crisp at 16×16 (system tray) through 512×512 (macOS dock at
    Retina) without blurring.
    """
    icon = QIcon()
    for size in _ICON_SIZES:
        path = str(files(__package__) / f"icon_{size}.png")
        icon.addFile(path, QSize(size, size))
    return icon


__all__ = ["get_app_icon", "get_icon_path"]
