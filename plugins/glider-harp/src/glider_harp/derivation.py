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

Profiles come from two places, and the second is the point of the first
being data at all: the ones shipped in this package, and the ones a lab writes
into ``~/.glider/harp_profiles``. A second Harp device in a lab needs a
profile, and before that directory existed the only way to add one was to edit
a file inside an installed package, which the next upgrade overwrites. See
``available_profiles`` for the precedence between them.

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

# Where a lab keeps profiles of its own. Only the leaf is fixed here; the root
# is resolved at the call, by ``user_profile_dir``.
USER_PROFILE_SUBDIR = "harp_profiles"

# Access modes that make a register something the flow graph can invoke.
# ``Event`` is the device talking to us and is not among them.
_ACTION_ACCESS = frozenset({"Read", "Write"})

# What a ``record`` entry may say. ``mode`` is **reserved and currently
# ignored**: it is carried in the shipped profile as a decoding hint for a
# later task, and nothing reads it today. Unknown keys are rejected rather
# than ignored, because a profile is hand-edited JSON and a key typed as
# ``"az"`` would otherwise be a change that silently does nothing. (A
# ``device.yml`` is treated the opposite way, and rightly: it is vendor-
# authored and full of keys we legitimately do not consume.)
_RECORD_KEYS = frozenset({"register", "as", "mode"})

# The profile format this module understands. Rejecting unknown keys is only
# safe with a version gate in front of it: without one, a 1.1 profile adding a
# key fails with "unknown keys: scale", which points at the key rather than at
# the version, and reads like a typo in a file that is perfectly correct for a
# newer GLIDER. Major only -- a minor bump is by definition additive, and a
# profile that omits the field is assumed to be of this major version.
_PROFILE_MAJOR_VERSION = 1

# A profile name selects a file inside one of two known directories, so it is
# a name and not a path: anything else lets a device setting read a file we
# never shipped and the user never wrote.
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

# Payload types a recorded register may have. Not a policy: it is exactly what
# ``RegisterCache`` can decode, which is every *scalar* Harp payload type,
# each read as its declared type. See ``_check_recordable``.
_RECORDABLE_TYPES = frozenset({"U8", "U16", "U32", "U64", "S8", "S16", "S32", "S64", "Float"})


@dataclass
class Derived:
    """What a schema and a profile decided between them.

    ``recorded`` maps a register address to the **base** name of its columns --
    ``lick``, not ``lick_state``. ``RegisterCache`` expands each into the three
    columns a register contributes, so this side never spells a column name in
    full and the two cannot drift apart.

    ``recorded_types`` maps the same addresses to the payload type the schema
    declares for each -- ``"S16"``, ``"Float"``. It is here because this is the
    only place that has read the schema, and ``RegisterCache`` cannot decode a
    payload by its type without being told what that type is. Kept beside
    ``recorded`` rather than folded into it so the values of ``recorded`` stay
    the column base names every other caller reads them as; the two are filled
    on the same line and ``RegisterCache`` refuses a type for an address it was
    not given, which is what stops them drifting.

    ``actions`` maps a register name to its address, keyed by name because that
    is what a person picks in the node editor.

    ``access`` carries each non-core register's access modes, and it is here
    because an address alone cannot answer the question every caller
    downstream has to ask: is this action a read or a write? Without it a
    device holding only ``actions`` must send a Read to a write-only register
    to find out, which on real hardware is a round-trip that times out --
    and a GUI cannot tell which control to draw at all.

    ``warnings`` are things the record will not say that it looks like it
    says, in a form a caller can write into a CSV. They are *also* logged as
    they are found, but a log line during a long unattended run reaches
    nobody, so the finding is carried rather than only emitted. Kept here
    rather than rebuilt by the caller so the predicate and the message have
    one home: a second copy in a caller is a second copy that can be subtly
    wrong about, say, a register whose ``access`` is a list.
    """

    recorded: dict[int, str] = field(default_factory=dict)
    recorded_types: dict[int, str] = field(default_factory=dict)
    actions: dict[str, int] = field(default_factory=dict)
    access: dict[str, frozenset[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _load_glider_config_dir() -> Path:
    """GLIDER's user configuration root, asked of GLIDER itself.

    Imported **inside the call**, and that is not a style choice: importing
    ``glider.core.config`` at module scope executes ``glider/__init__``, which
    pulls in the whole core engine. ``glider_harp`` importing no ``glider`` at
    all is a property the packaging suite pins, because it is what keeps this
    package's plugin tables lazy; a convenience import here would erase it
    silently. By the time anything calls this, ``glider.core.config`` is
    already in ``sys.modules`` and the import is a dict lookup.
    """
    from glider.core.config import get_config

    return Path(get_config().paths.user_config_dir)


def _glider_config_dir() -> Path:
    """The configuration root, or the conventional one if GLIDER cannot say.

    Every failure falls back rather than propagating -- broadly, and on
    purpose. This is reached from ``HarpDevice``'s class body, where the
    exception would not misconfigure a device but stop the plugin importing at
    all; and the fallback is not a guess, it is the same path GLIDER's own
    default config would have produced.
    """
    try:
        return _load_glider_config_dir()
    except Exception:
        logger.debug("Harp profiles: GLIDER config unavailable, using ~/.glider", exc_info=True)
        return Path.home() / ".glider"


def user_profile_dir() -> Path:
    """Where a lab's own profiles live: ``~/.glider/harp_profiles``.

    Resolved at the call rather than at import, so it follows a relocated
    ``user_config_dir`` and costs no ``glider`` import to anyone who never asks.
    """
    return _glider_config_dir() / USER_PROFILE_SUBDIR


def available_profiles() -> dict[str, Path]:
    """Every profile name a device may be configured with, and the file it loads.

    Both directories, name-sorted, **user last** -- so a user profile with the
    same stem as a shipped one is the file that comes back. That precedence is
    what lets a lab correct a shipped profile that does not match the firmware
    it actually has, which is otherwise only fixable by editing a file inside
    an installed package that the next upgrade overwrites.

    It is also the one way this feature can go quietly wrong: a stale local
    copy silently outliving the shipped fix. So the shadowing is never only
    implied -- ``load_profile`` logs it, and the hardware panel's dropdown
    labels it.
    """
    found: dict[str, Path] = {}
    for directory in (PROFILE_DIR, user_profile_dir()):
        for path in _profile_files(directory):
            found[path.stem] = path
    return dict(sorted(found.items()))


def _profile_files(directory: Path) -> list[Path]:
    """The ``*.json`` files in one profile directory, or none.

    A missing directory is the ordinary case -- nobody has written a profile
    yet -- and an unreadable one must not be worse than a missing one: this
    runs while ``HarpDevice``'s class body is executing, so an ``OSError``
    escaping here takes the whole plugin down rather than one device.
    """
    try:
        return sorted(directory.glob("*.json"))
    except OSError:
        logger.warning("Harp profiles: could not read %s", directory, exc_info=True)
        return []


def load_profile(name: str) -> dict[str, Any]:
    """Load a profile by name, from the user's directory or the package.

    A profile is ``{"schema_version", "name", "who_am_i", "record"}``, where
    each ``record`` entry is ``{"register", "as", "mode"}``. ``mode`` is
    reserved and read by nothing today; see ``_RECORD_KEYS``.

    A user profile is *data*, not an escape hatch: it goes through the same
    version gate as a shipped one, and ``derive`` applies the same rules to
    what it asks for. What differs is the reporting -- every failure here names
    the file, because unlike a shipped profile it is a file the person reading
    the message can open and fix.
    """
    if not _PROFILE_NAME.match(name):
        raise ValueError(f"Profile name {name!r} is not a plain profile name")
    path = _resolve_profile(name)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Profile file {path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Profile file {path} could not be read: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Profile file {path} does not contain a JSON object")
    try:
        _check_schema_version(loaded)
    except ValueError as exc:
        raise ValueError(f"{exc} (in {path})") from None
    return loaded


def _resolve_profile(name: str) -> Path:
    """Which file a profile name loads. User directory first; see ``available_profiles``."""
    user = user_profile_dir() / f"{name}.json"
    shipped = PROFILE_DIR / f"{name}.json"
    if user.is_file():
        if shipped.is_file():
            # Not a warning about a mistake -- overriding is the feature. It is
            # a warning because the override is invisible everywhere else once
            # the device is configured, and a local copy that quietly survives
            # an upgrade of the shipped one is the failure this precedence buys.
            logger.warning(
                "Harp profile %r is being read from %s, which overrides the profile "
                "shipped with glider-harp",
                name,
                user,
            )
        return user
    if shipped.is_file():
        return shipped
    available = sorted(available_profiles())
    raise FileNotFoundError(
        f"No profile named {name!r}"
        + (f"; available: {', '.join(available)}" if available else "")
        + f". Add one of your own as {user_profile_dir() / f'{name}.json'}"
    )


def _check_schema_version(profile: Mapping[str, Any]) -> None:
    """Refuse a profile written to a format this GLIDER does not know.

    Checked here *and* from ``derive``, so the gate holds however the profile
    arrived: ``load_profile`` reads both the shipped and the user directory,
    but a caller may equally hand ``derive`` a profile it built or parsed
    itself, and gating in one place would leave the other open.
    """
    declared = profile.get("schema_version")
    if declared is None:
        return
    major = str(declared).split(".")[0].strip()
    try:
        version = int(major)
    except ValueError:
        raise ValueError(
            f"Profile {profile.get('name', '?')!r} has an unreadable "
            f"schema_version {declared!r}"
        ) from None
    if version != _PROFILE_MAJOR_VERSION:
        raise ValueError(
            f"Profile {profile.get('name', '?')!r} declares schema_version "
            f"{declared!r}; this GLIDER reads profile format "
            f"{_PROFILE_MAJOR_VERSION}.x"
        )


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
        modes = _access_of(meta)
        result.access[str(name)] = modes
        if modes & _ACTION_ACCESS:
            result.actions[str(name)] = address

    if not profile:
        return result

    _check_schema_version(profile)
    _check_who_am_i(schema, profile)

    by_name = {str(name): meta for name, meta in registers.items()}
    claimed: dict[str, str] = {}
    records = profile.get("record") or []
    if not isinstance(records, (list, tuple)):
        # A mapping is the plausible mistake -- writing the block as
        # {"LickState": "lick"} -- and it is the worst shape to let through:
        # iterating it yields the register *names*, each of which then fails
        # as "not a mapping" while naming a register that does exist, which
        # points the reader at the wrong half of the file entirely.
        raise ValueError(
            f"Profile 'record' must be a list of entries, got {type(records).__name__}"
        )
    for entry in records:
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

        _check_recordable(register, by_name[register])

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
            #
            # Carried on the result as well as logged, because logging it is
            # not the same as reporting it: the run this matters to is the
            # unattended one, whose log nobody opens. The caller writes this
            # into the CSV metadata, which is what outlives the session.
            warning = (
                f"register {register} (column {column}) is not an Event register; "
                "its columns will never change"
            )
            logger.warning("Harp profile records a register that cannot report: %s", warning)
            result.warnings.append(warning)

        claimed[column] = register
        result.recorded[address] = column
        # Filled on the same pass as the name, so the cache is never handed a
        # column whose type nobody looked up. ``_check_recordable`` has already
        # refused anything the cache cannot decode, so this is a known type.
        result.recorded_types[address] = str(by_name[register].get("type"))

    return result


def _check_recordable(name: str, meta: Any) -> None:
    """Refuse a register whose payload the record would decode wrongly.

    The gate is exactly the set ``RegisterCache`` can decode, and it moves only
    when the cache does. It used to be the four unsigned widths, because the
    cache read every payload as one unsigned little-endian integer; the cache
    now decodes by the register's declared type, so every *scalar* type passes
    and the two halves stayed together, as this docstring said they had to.

    What still fails, and why each is a refusal rather than a warning -- every
    one of them writes a file that opens cleanly, plots, and is wrong, and
    nothing downstream can tell a bad number from a real reading:

    * **Array registers** (``length`` above 1). Not a decoding gap but a
      design question nobody has answered: one CSV cell cannot hold eight
      values, and inventing a serialization for them -- one column per
      element, a delimited string, the first element only -- is a decision
      about what the record *means*, not an implementation detail. Until
      somebody makes it, the whole payload would collapse into one implausible
      number.
    * **A type the cache has never heard of**, including a register whose
      schema declares none. ``Uint16`` for ``U16`` is a plausible hand-edit,
      and there is no decoding at all behind it.

    Only the *recorded* side is gated, and always was. Writes go out with the
    register's declared width and signedness (``HarpDevice._pack``), so a
    signed register has always been usable as an action.
    """
    if not isinstance(meta, Mapping):
        return
    declared = meta.get("type")
    if declared not in _RECORDABLE_TYPES:
        raise ValueError(
            f"Profile records register {name!r}, whose type is {declared!r}; only "
            f"{', '.join(sorted(_RECORDABLE_TYPES))} can be recorded, because those are "
            "the payload types the register cache knows how to decode"
        )
    length = meta.get("length", 1)
    if isinstance(length, int) and not isinstance(length, bool) and length > 1:
        raise ValueError(
            f"Profile records register {name!r}, which has {length} elements; the register "
            "cache reads the whole payload as one value, and how an array should appear in "
            "a CSV is not yet decided, so it would be recorded as a single meaningless number"
        )


def _address_of(name: str, meta: Any) -> int:
    """The address of one register entry.

    Deliberately not range-checked. ``schema.build_registers`` is the single
    authority on what a frame header can carry (0..255) and rejects anything
    else; repeating the bound here would put the wire's limit in two files that
    could disagree. So ``derive`` will happily route address 999 into
    ``actions`` -- a caller that only ever derives, and never builds registers
    from the same schema, gets no protection from that and should not expect
    any. In GLIDER both run against the same schema.
    """
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
