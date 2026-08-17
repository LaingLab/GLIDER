"""Decide which registers become CSV columns and which become actions.

A Harp device offers everything it has; an experiment wants almost none of it.
This module is where that choice is made, and it is made from data -- a device
schema plus a *profile*, a small JSON file naming the registers worth
recording. Keeping it apart from ``schema`` is the point: building a register
class from a type name and choosing what a trial writes to disk fail in
completely different ways, and only one of them is a judgement call.

Three rules, and the third is the one that matters:

* **Core registers are never columns.** Eight specific addresses below 15 --
  see ``CORE_REGISTERS``, and note it is eight of those fifteen, not all of
  them -- are identity and lifecycle: who the device is, what firmware it runs,
  what time it thinks it is. Recording them writes the same number in every row
  of every trial.
* **``Write`` and ``Read`` registers become actions**, reachable from a
  ``DeviceAction`` node. That is the whole control surface of the device.
* **Without a profile, nothing is recorded.** A Behavior board declares about
  thirty registers; deriving columns from "everything that emits events" would
  drop thirty columns into a CSV that a person then has to explain. Widening
  the record is always a deliberate act, so an unrecognised device is silent
  until somebody writes a profile for it.

This module also owns the invariant that column names are well formed, and it
owns it because it is the first point where those names stop being Python
literals and start being JSON a user can edit. ``RegisterCache.__init__``
checks the same things and is the last line of defence, but by then the name
has lost its provenance -- all it can say is "column names must be unique",
while this can say which profile entry is at fault.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Identity and lifecycle, fixed by the Harp specification: 0 WhoAmI,
# 1-2 hardware version, 6-7 firmware version, 8 TimestampSecond,
# 10 OperationControl, 14 ClockConfiguration. Eight addresses, not a range --
# 3, 4, 5, 9, 11, 12 and 13 are ordinary registers a device may use for data.
CORE_REGISTERS = frozenset({0, 1, 2, 6, 7, 8, 10, 14})

PROFILE_DIR = Path(__file__).parent / "profiles"

# Access modes that make a register something the flow graph can invoke.
# ``Event`` is the device talking to us and is not among them.
_ACTION_ACCESS = frozenset({"Read", "Write"})

# What a ``record`` entry may say. ``mode`` is **reserved and currently
# ignored**: it is carried in the shipped profile as a decoding hint for a
# later task, and nothing reads it today. Unknown keys are rejected rather
# than ignored, because a profile is hand-edited JSON and a key typed as
# ``"az"`` would otherwise be a change that silently does nothing.
_RECORD_KEYS = frozenset({"register", "as", "mode"})

# A profile name selects a file inside the package, so it is a name and not a
# path: anything else lets a device setting read a file we never shipped.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class Derived:
    """What a schema and a profile decided between them.

    ``recorded`` maps a register address to the **base** name of its columns --
    ``lick``, not ``lick_state``. ``RegisterCache`` expands each into the three
    columns a register contributes, so this side never spells a column name in
    full and the two cannot drift apart.

    ``actions`` maps a register name to its address, keyed by name because that
    is what a person picks in the node editor.
    """

    recorded: dict[int, str] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)


def load_profile(name: str) -> dict[str, Any]:
    """Load a profile shipped inside the package.

    A profile is ``{"schema_version", "name", "who_am_i", "record"}``, where
    each ``record`` entry is ``{"register", "as", "mode"}``. ``mode`` is
    reserved and read by nothing today; see ``_RECORD_KEYS``.
    """
    if not _PROFILE_NAME.match(name):
        raise ValueError(f"Profile name {name!r} is not a plain profile name")
    path = PROFILE_DIR / f"{name}.json"
    if not path.is_file():
        available = sorted(p.stem for p in PROFILE_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"No shipped profile named {name!r}"
            + (f"; available: {', '.join(available)}" if available else "")
        )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Profile {name!r} is not a JSON object")
    return loaded


def derive(schema: Mapping[str, Any], profile: Mapping[str, Any] | None) -> Derived:
    """Work out this device's columns and actions.

    ``schema`` is a ``device.yml``-shaped mapping; ``profile`` is a loaded
    profile, or ``None`` for a device nobody has written one for -- which
    records nothing and exposes everything writable.

    Raises ``ValueError``, naming the profile entry at fault, for a profile
    that cannot produce a valid CSV header: an unknown or core register, or a
    column base name that is empty, contains ``:``, or collides with another
    on the same device.
    """
    registers = schema.get("registers") or {}
    if not isinstance(registers, Mapping):
        raise ValueError("Schema 'registers' must be a mapping of name to register entry")

    result = Derived()
    for name, meta in registers.items():
        address = _address_of(str(name), meta)
        if address in CORE_REGISTERS:
            continue
        if _access_of(meta) & _ACTION_ACCESS:
            result.actions[str(name)] = address

    if not profile:
        return result

    _check_who_am_i(schema, profile)

    by_name = {str(name): meta for name, meta in registers.items()}
    claimed: dict[str, str] = {}
    for entry in profile.get("record") or []:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Profile record entry is not a mapping: {entry!r}")
        if unknown := sorted(str(key) for key in entry if key not in _RECORD_KEYS):
            raise ValueError(f"Profile record entry has unknown keys: {', '.join(unknown)}")
        register = entry.get("register")
        column = entry.get("as")

        # Checked for being a string before being looked up, because a JSON
        # list or object here is unhashable: the membership test would raise
        # TypeError, which is the one malformed entry a hand-editing user
        # could produce that did not come back as a ValueError like the rest.
        if not isinstance(register, str):
            raise ValueError(f"Profile record entry has a non-string 'register': {register!r}")
        if register not in by_name:
            raise ValueError(f"Profile entry names unknown register {register!r}")
        address = _address_of(register, by_name[register])
        if address in CORE_REGISTERS:
            # Silently dropping it would leave a profile that looks like it
            # asked for a column and produced none.
            raise ValueError(
                f"Profile entry {register!r} names core register {address}, which is "
                "identity and lifecycle, not data"
            )
        if address in result.recorded:
            raise ValueError(f"Profile records register {register!r} more than once")

        if not isinstance(column, str) or not column:
            raise ValueError(f"Profile entry {register!r} has no 'as' column name")
        if ":" in column:
            # The recorder builds its header as "{device_id}:{sub_column}" and
            # recovers the sub-column by partitioning on the first colon, so a
            # colon here yields a header nothing downstream can take apart.
            raise ValueError(f"Profile entry {register!r} column name {column!r} contains ':'")
        if column in claimed:
            raise ValueError(
                f"Profile entries {claimed[column]!r} and {register!r} both use "
                f"column name {column!r}"
            )

        if "Event" not in _access_of(by_name[register]):
            # Not fatal -- a schema may simply be incomplete -- but the column
            # would sit at its initial value for the whole session, and nothing
            # further down can tell that apart from a device that never licked.
            logger.warning(
                "Harp profile records register %r, which is not an Event register; "
                "its columns will never change",
                register,
            )

        claimed[column] = register
        result.recorded[address] = column

    return result


def _address_of(name: str, meta: Any) -> int:
    """The address of one register entry."""
    if not isinstance(meta, Mapping) or "address" not in meta:
        raise ValueError(f"Register {name!r} has no address")
    address = meta["address"]
    if not isinstance(address, int) or isinstance(address, bool):
        raise ValueError(f"Register {name!r} has non-integer address {address!r}")
    return address


def _access_of(meta: Any) -> frozenset[str]:
    """A register's access modes.

    Returned as a set because the schema allows either a single mode or a list
    -- a register that is both written and reported back is ordinary, and
    reading only the first entry would drop it out of the actions.
    """
    if not isinstance(meta, Mapping):
        return frozenset()
    access = meta.get("access")
    if isinstance(access, str):
        return frozenset({access})
    if isinstance(access, (list, tuple, set, frozenset)):
        return frozenset(str(item) for item in access)
    return frozenset()


def _check_who_am_i(schema: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    """Refuse a profile written for a different device.

    Only checkable when both sides say who they are; a schema that omits
    ``whoAmI`` is taken at its word. The failure this prevents is a profile
    whose register names happen to exist on another board, which derives
    cleanly and records the wrong thing.
    """
    declared = schema.get("whoAmI")
    expected = profile.get("who_am_i")
    if declared is None or expected is None:
        return
    if _who_am_i(declared) != _who_am_i(expected):
        # Both sides shown as reprs. Compared raw, a schema quoting its WhoAmI
        # produced "is for WhoAmI 1400, but this schema declares 1400" -- a
        # mismatch message naming two identical numbers, which sends the reader
        # looking for a problem that is not there.
        raise ValueError(
            f"Profile {profile.get('name', '?')!r} is for WhoAmI {expected!r}, "
            f"but this schema declares {declared!r}"
        )


def _who_am_i(raw: Any) -> Any:
    """A WhoAmI as a number where it can be read as one.

    A schema hand-copied from a datasheet may quote it, or write it in hex --
    ``"1400"`` and ``0x578`` are the same device as ``1400``. Anything that is
    not a number at all comes back unchanged, so it still fails the comparison
    and still prints recognisably.
    """
    if isinstance(raw, bool) or raw is None:
        return raw
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw), 0)
    except ValueError:
        return raw
