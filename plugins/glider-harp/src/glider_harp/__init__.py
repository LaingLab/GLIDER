"""Harp binary-protocol support for GLIDER.

The package's public names are re-exported here so callers depend on
``glider_harp`` rather than on which module a class currently lives in --
``RegisterCache`` sits beside ``HarpReader`` for the obvious reason that the
reader fills it, and importing a cache from a module called ``reader`` reads
like a mistake every time.

Four layers, and they are separable on purpose:

* ``frames`` -- wire format only. ``decode`` and ``FrameSplitter`` turn bytes
  into frames.
* ``reader`` -- what the device reported. ``HarpReader`` owns the port on a
  background thread and fills a ``RegisterCache``, which is what becomes CSV
  columns.
* ``schema`` -- what the device *has*. ``build_registers`` compiles a
  ``device.yml``-shaped dict into typed register classes.
* ``derivation`` -- what this experiment wants of it. ``derive`` decides which
  registers become columns and which become actions.

``frames`` and ``schema`` are the only two modules that name ``harp.protocol``,
and they name disjoint halves of it -- the message codec and the register DSL.
An upstream change lands in one of those two files and no other.
"""

from glider_harp.derivation import CORE_REGISTERS, Derived, derive, load_profile
from glider_harp.frames import (
    ChecksumError,
    FrameError,
    FrameSplitter,
    HarpFrame,
    TruncatedFrameError,
    decode,
)
from glider_harp.reader import HarpReader, RegisterCache
from glider_harp.schema import SchemaError, build_registers, load_schema

__all__ = [
    "CORE_REGISTERS",
    "ChecksumError",
    "Derived",
    "FrameError",
    "FrameSplitter",
    "HarpFrame",
    "HarpReader",
    "RegisterCache",
    "SchemaError",
    "TruncatedFrameError",
    "build_registers",
    "decode",
    "derive",
    "load_profile",
    "load_schema",
]
