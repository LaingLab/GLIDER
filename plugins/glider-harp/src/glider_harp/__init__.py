"""Harp binary-protocol support for GLIDER.

The package's public names are re-exported here so callers depend on
``glider_harp`` rather than on which module a class currently lives in --
``RegisterCache`` sits beside ``HarpReader`` for the obvious reason that the
reader fills it, and importing a cache from a module called ``reader`` reads
like a mistake every time.

Two layers, and they are separable on purpose:

* ``frames`` -- wire format only. ``decode`` and ``FrameSplitter`` turn bytes
  into frames and are the only place ``harp.protocol`` is named, so an upstream
  change touches that file and no other.
* ``reader`` -- what the device reported. ``HarpReader`` owns the port on a
  background thread and fills a ``RegisterCache``, which is what becomes CSV
  columns.
"""

from glider_harp.frames import (
    ChecksumError,
    FrameError,
    FrameSplitter,
    HarpFrame,
    TruncatedFrameError,
    decode,
)
from glider_harp.reader import HarpReader, RegisterCache

__all__ = [
    "ChecksumError",
    "FrameError",
    "FrameSplitter",
    "HarpFrame",
    "HarpReader",
    "RegisterCache",
    "TruncatedFrameError",
    "decode",
]
