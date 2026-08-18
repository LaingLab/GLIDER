"""Harp binary-protocol support for GLIDER.

The package's public names are re-exported here so callers depend on
``glider_harp`` rather than on which module a class currently lives in --
``RegisterCache`` sits beside ``HarpReader`` for the obvious reason that the
reader fills it, and importing a cache from a module called ``reader`` reads
like a mistake every time.

Five layers, and they are separable on purpose:

* ``frames`` -- wire format only. ``decode`` and ``FrameSplitter`` turn bytes
  into frames.
* ``reader`` -- what the device reported. ``HarpReader`` owns the port on a
  background thread and fills a ``RegisterCache``, which is what becomes CSV
  columns.
* ``schema`` -- what the device *has*. ``build_registers`` compiles a
  ``device.yml``-shaped dict into typed register classes.
* ``derivation`` -- what this experiment wants of it. ``derive`` decides which
  registers become columns and which become actions.
* ``device`` -- the four above, composed. ``HarpDevice`` owns the port and the
  order the others are driven in; ``mock`` is the same device with a fake
  handle underneath it.

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

``board``, ``device`` and ``mock`` are the modules here that import ``glider``,
which is why none of them is re-exported below: ``import glider_harp`` pulls in
no ``glider`` modules at all, while importing any of those three pulls in 36,
including all of ``glider.vision``. Import them by module --
``from glider_harp.board import HarpBoard``,
``from glider_harp.device import HarpDevice``.

``BOARD_DRIVERS`` and ``DEVICE_TYPES`` are the exception, and they are lazy for
exactly that reason. ``PluginManager._register_plugin_components`` reads them
off a plugin module with ``hasattr``/``getattr``, so they must *resolve* from
this package -- but writing them as top-level assignments would mean
``import glider_harp`` imports ``board`` and ``device``, and the property above
would be gone. PEP 562's module ``__getattr__`` satisfies ``hasattr`` and
``getattr`` identically while deferring the import to first access, so the cost
is paid only by a caller who actually wants the drivers. The resolved value is
written back into ``globals()``, so the second access is a plain dict lookup
and never re-enters this function.

There is no ``NODE_TYPES``: this package ships no node classes, and an empty
dict would only make ``PluginManager`` walk a loop over nothing.
"""

from typing import Any

from glider_harp.derivation import CORE_REGISTERS, Derived, derive, load_profile
from glider_harp.frames import (
    ChecksumError,
    FrameError,
    FrameSplitter,
    HarpFrame,
    TruncatedFrameError,
    decode,
    encode,
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
    "encode",
    "load_profile",
    "load_schema",
]

# Deliberately NOT in ``__all__``: ``from glider_harp import *`` would evaluate
# every name in it, which would pull ``glider`` in through the back door and
# undo the whole point of the laziness below. They are part of the plugin-host
# contract, not of the star-import surface. ``__dir__`` still advertises them.
_LAZY_PLUGIN_ATTRS = frozenset({"BOARD_DRIVERS", "DEVICE_TYPES"})


def __getattr__(name: str) -> Any:
    """Resolve the plugin-host tables on first access (PEP 562).

    ``PluginManager`` reaches for these with ``hasattr``/``getattr``, both of
    which route through here, so the deferral is invisible to it.
    """
    if name == "BOARD_DRIVERS":
        from glider_harp.board import HarpBoard

        # Key is the driver name: what ``HardwareManager._driver_registry`` is
        # keyed by, matching ``HarpBoard.board_type`` and the entry point name.
        value: Any = {"harp": HarpBoard}
    elif name == "DEVICE_TYPES":
        from glider_harp.device import HarpDevice

        # Key is ``HarpDevice.device_type``, which is what
        # ``create_device_from_dict`` looks up in ``DEVICE_REGISTRY``.
        value = {"Harp": HarpDevice}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Cache into the module namespace: a second access never re-enters here.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | _LAZY_PLUGIN_ATTRS)
