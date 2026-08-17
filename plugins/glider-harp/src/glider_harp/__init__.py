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

Three modules name ``harp.protocol``, and an upstream change lands in those
three files and no other:

* ``frames`` -- the message codec (``HarpMessage``, ``HarpParseError``).
* ``schema`` -- the register DSL. Disjoint from ``frames``' half.
* ``board`` -- ``HarpMessage`` again, but not to use it: it is imported as a
  *canary*, to tell a correctly resolved ``harp-protocol`` from the
  incompatible 0.4.0 that ``harp``'s unbounded requirement also accepts. So an
  upstream **rename** of ``HarpMessage`` does not merely break ``frames``; it
  silently turns that guard into a false alarm, reporting a bad install to
  anyone with a good one. Retarget the canary along with the codec.

``board`` is also the only module here that imports ``glider``, which is why it
is not re-exported below: ``import glider_harp`` pulls in no ``glider`` modules
at all, while ``import glider_harp.board`` pulls in 36, including all of
``glider.vision``. Import it as ``from glider_harp.board import HarpBoard``.
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
