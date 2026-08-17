"""Compile a Harp ``device.yml`` into register classes, at runtime.

``harp``'s README claims it builds a register interface from a ``device.yml``.
It does not: there is no YAML handling anywhere in the package, and harp-tech's
own workflow is C# code generation -- ``harp/device/_registers.py`` opens with
"This file was automatically generated". What the package does provide is the
half that is genuinely hard, and it provides it at runtime: dtype layout, mask
shifting, enum decoding and strided bulk views, reachable through metaclasses
that manufacture a *class* from an address (``RegisterU8(32)`` is a class, not
an instance). This module is the front end that was missing -- schema dict in,
register classes out, no code generation step.

Second of the two places ``harp.protocol`` is named, and the other half of it:
``frames`` adapts the message codec, this adapts the register DSL. They are
disjoint imports, so an upstream change lands in one file or the other.

What a register entry may say, and what each key turns into:

* ``type`` -- one of the nine Harp payload types. Anything else raises; a
  register built as the wrong width decodes every event wrongly and silently.
* ``length`` -- above 1, an array register whose ``parse`` returns an ndarray.
* ``maskType`` -- names an entry in the schema's ``bitMasks`` (decoded to
  ``IntFlag``) or ``groupMasks`` (decoded to ``IntEnum``).
* ``payloadSpec`` -- named members at element offsets, each optionally masked,
  decoded to a payload object with one attribute per member.

Everything raises ``SchemaError`` at build time rather than deferring to the
first frame. A register class is built once, when a device is configured, and
then used against every event for the rest of the session: a fault here that
waits until decode time costs one counted ``processing_errors`` per frame in
``HarpReader`` and an experiment record that is empty for no visible reason.
"""

from __future__ import annotations

import enum
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harp.protocol import (
    AnonymousPayload,
    BitMask,
    Field,
    GroupMask,
    IdentityConverter,
    PayloadType,
    RegisterBase,
    RegisterFloat,
    RegisterFloatArray,
    RegisterS8,
    RegisterS8Array,
    RegisterS16,
    RegisterS16Array,
    RegisterS32,
    RegisterS32Array,
    RegisterS64,
    RegisterS64Array,
    RegisterU8,
    RegisterU8Array,
    RegisterU16,
    RegisterU16Array,
    RegisterU32,
    RegisterU32Array,
    RegisterU64,
    RegisterU64Array,
    StructPayload,
)

# A Harp address is one byte of the header, so this is the wire's own limit
# rather than a policy of ours.
_MAX_ADDRESS = 0xFF


class SchemaError(ValueError):
    """A device schema could not be turned into register classes.

    A ``ValueError`` because that is what a caller validating configuration
    already catches, and because ``HarpReader`` treats ``ValueError`` from the
    ingest path as a counted processing error rather than a dead port -- so a
    schema fault degrades the same way wherever it happens to surface.
    """


# The nine payload types the wire format defines, each with its scalar and
# array register constructor. Keyed by the spelling ``device.yml`` uses, which
# is also ``PayloadType``'s own member name -- so the payload type and element
# dtype come from ``PayloadType[name]`` rather than from a second table that
# could disagree with this one.
_REGISTER_TYPES: dict[str, tuple[Any, Any]] = {
    "U8": (RegisterU8, RegisterU8Array),
    "U16": (RegisterU16, RegisterU16Array),
    "U32": (RegisterU32, RegisterU32Array),
    "U64": (RegisterU64, RegisterU64Array),
    "S8": (RegisterS8, RegisterS8Array),
    "S16": (RegisterS16, RegisterS16Array),
    "S32": (RegisterS32, RegisterS32Array),
    "S64": (RegisterS64, RegisterS64Array),
    "Float": (RegisterFloat, RegisterFloatArray),
}


def load_schema(path: str | Path) -> dict[str, Any]:
    """Read a ``device.yml`` from disk.

    Deliberately thin, and deliberately separate from ``build_registers``:
    keeping the parse in the caller's hands is what lets every test here build
    a schema as a literal instead of as a fixture on disk. ``PyYAML`` is
    imported here rather than at module scope so that a caller who already has
    the schema as a dict -- which is every caller inside GLIDER -- needs no
    YAML parser installed at all.
    """
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise SchemaError(f"{path} does not contain a device schema mapping")
    return loaded


def build_registers(schema: Mapping[str, Any]) -> dict[str, type[RegisterBase[Any]]]:
    """Build one register class per entry in ``schema['registers']``.

    Returns them keyed by register name, in schema order. Each class carries
    the ``address`` it was built for, so a caller wanting the address map that
    a dispatcher needs builds it with ``{cls.address: cls for cls in ...}`` --
    which is safe precisely because duplicate addresses are rejected here.

    Takes a mapping, not a path: see ``load_schema``.
    """
    registers = schema.get("registers") or {}
    if not isinstance(registers, Mapping):
        raise SchemaError("Schema 'registers' must be a mapping of name to register entry")

    bit_masks = _mask_section(schema, "bitMasks")
    group_masks = _mask_section(schema, "groupMasks")

    built: dict[str, type[RegisterBase[Any]]] = {}
    seen: dict[int, str] = {}
    for name, entry in registers.items():
        if not str(name):
            # A class named '' and a key of '' -- unreachable by name from
            # anything, and it looks like a register that simply isn't there.
            raise SchemaError("Schema has a register with an empty name")
        register = _build_register(str(name), entry, bit_masks, group_masks)
        # Two registers at one address is a broken schema, and one that hides
        # well: the loser simply never appears in an address-keyed map, so its
        # events are silently attributed to the winner for the whole session.
        if register.address in seen:
            raise SchemaError(
                f"Registers {seen[register.address]!r} and {name!r} share "
                f"address {register.address}"
            )
        seen[register.address] = str(name)
        built[str(name)] = register
    return built


def _build_register(
    name: str,
    entry: Any,
    bit_masks: Mapping[str, Any],
    group_masks: Mapping[str, Any],
) -> type[RegisterBase[Any]]:
    """Build the one register class ``entry`` describes."""
    if not isinstance(entry, Mapping):
        raise SchemaError(f"Register {name!r} must be a mapping, got {type(entry).__name__}")

    address = _address_of(name, entry)
    type_name = entry.get("type")
    if type_name not in _REGISTER_TYPES:
        raise SchemaError(
            f"Register {name!r} has unsupported type {type_name!r}; "
            f"expected one of {', '.join(sorted(_REGISTER_TYPES))}"
        )
    payload_type = PayloadType[str(type_name)]
    element = payload_type.numpy_dtype
    length = _length_of(name, entry)

    payload_spec = entry.get("payloadSpec")
    mask_type = entry.get("maskType")

    if payload_spec is not None and mask_type is not None:
        # The register-level mask has nowhere to go once the payload is a
        # struct -- a payloadSpec member carries its own maskType -- so taking
        # the payloadSpec branch and returning would drop it in silence, and
        # the built class would then decode every event of the session through
        # the wrong lens. The same argument as the masked-array refusal below:
        # a schema declaring both is one its author misunderstood, and guessing
        # which half they meant is worse than saying so.
        raise SchemaError(
            f"Register {name!r} declares both payloadSpec and maskType {mask_type!r}; "
            "a payloadSpec member carries its own maskType, so the register-level "
            "one would be silently dropped"
        )

    if payload_spec is not None:
        payload = _struct_payload(name, payload_spec, element, length, bit_masks, group_masks)
        return _register_class(name, address, payload_type, payload)

    if mask_type is not None:
        if length > 1:
            # Representable in principle -- an array of flag words -- but not
            # by the descriptors upstream offers, and a mask quietly applied to
            # only the first element would be worse than refusing.
            raise SchemaError(
                f"Register {name!r} has both maskType {mask_type!r} and length {length}; "
                "a masked array register is not supported"
            )
        descriptor = _mask_descriptor(name, str(mask_type), element, bit_masks, group_masks)
        payload = _root_payload(name, descriptor, element)
        return _register_class(name, address, payload_type, payload)

    scalar, array = _REGISTER_TYPES[str(type_name)]
    register = array(address, length=length) if length > 1 else scalar(address)
    register.__name__ = name
    register.__qualname__ = name
    return register


def _mask_section(schema: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """The ``bitMasks`` or ``groupMasks`` block, checked before it is indexed.

    Without this, a section written as a bare string passes the ``in`` test a
    ``maskType`` lookup does -- ``"Bits" in "Bits"`` is substring membership --
    and then dies on the subscript with ``TypeError: string indices must be
    integers``, which is neither a ``SchemaError`` nor a hint about the file.
    """
    section = schema.get(key) or {}
    if not isinstance(section, Mapping):
        raise SchemaError(
            f"Schema {key!r} must be a mapping of mask name to definition, "
            f"got {type(section).__name__}"
        )
    return section


def _address_of(name: str, entry: Mapping[str, Any]) -> int:
    """The register's address, checked against what a frame header can carry."""
    if "address" not in entry:
        raise SchemaError(f"Register {name!r} has no address")
    address = entry["address"]
    if not isinstance(address, int) or isinstance(address, bool):
        raise SchemaError(f"Register {name!r} has non-integer address {address!r}")
    if not 0 <= address <= _MAX_ADDRESS:
        raise SchemaError(f"Register {name!r} has address {address}, outside 0..{_MAX_ADDRESS}")
    return address


def _length_of(name: str, entry: Mapping[str, Any]) -> int:
    """Element count. Absent means one; a scalar register is length 1."""
    length = entry.get("length", 1)
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise SchemaError(f"Register {name!r} has invalid length {length!r}")
    return length


def _register_class(
    name: str,
    address: int,
    payload_type: PayloadType,
    payload: type[Any],
) -> type[RegisterBase[Any]]:
    """Assemble a register class around a payload class we built ourselves.

    The scalar and array constructors are metaclasses that take an address and
    nothing else -- ``RegisterU8(32, payload_class=...)`` is a ``TypeError`` --
    so a register whose payload carries masks or named members has to be
    assembled here instead. Upstream's own generated code hand-assembles these
    the same way.

    ``types.new_class`` here is **consistency, not necessity**: the base is a
    bare ``RegisterBase``, so a plain ``type()`` call would work. It is
    necessary in ``_root_payload`` and ``_struct_payload``, whose bases are
    subscripted generics that need ``__mro_entries__`` resolved and
    ``__orig_bases__`` recorded -- upstream reads the latter to derive the
    payload's base element width. Kept the same in all three so that changing
    one of them is not a decision about which mechanism it was using.
    """
    return types.new_class(
        name,
        (RegisterBase,),
        {},
        lambda namespace: namespace.update(
            {
                "address": address,
                "payload_type": payload_type,
                "payload_class": payload,
            }
        ),
    )


def _root_payload(name: str, descriptor: Any, element: Any) -> type[Any]:
    """A payload that *is* one decoded value, held in the reserved ``__value__``.

    ``__value__`` is upstream's name, not ours, and it is load-bearing: an
    ``AnonymousPayload`` subclass declaring exactly one descriptor field with
    that exact name is how the package spells "this payload is a single value"
    (their docstring compares it to pydantic's ``__root__``). Any other field
    name is a definition-time ``TypeError``. The effect is that ``parse``
    unwraps the field and hands back the ``IntFlag`` or ``IntEnum`` itself,
    rather than a wrapper object the caller would have to reach through.
    """
    try:
        return types.new_class(
            f"{name}Payload",
            (AnonymousPayload[element],),
            {},
            lambda namespace: namespace.update({"__value__": descriptor}),
        )
    except TypeError as exc:  # pragma: no cover - upstream layout rejection
        raise SchemaError(f"Register {name!r} has an unbuildable masked payload: {exc}") from exc


def _struct_payload(
    name: str,
    spec: Any,
    element: Any,
    length: int,
    bit_masks: Mapping[str, Any],
    group_masks: Mapping[str, Any],
) -> type[Any]:
    """A payload with one named member per ``payloadSpec`` entry.

    ``offset`` is in base elements, matching the schema, and each member is a
    whole-element view unless it declares a ``maskType`` or a bare ``mask``.
    """
    if not isinstance(spec, Mapping) or not spec:
        raise SchemaError(f"Register {name!r} has an empty or non-mapping payloadSpec")

    members: dict[str, Any] = {}
    for member_name, member in spec.items():
        if not isinstance(member, Mapping):
            raise SchemaError(
                f"Register {name!r} payloadSpec member {member_name!r} is not a mapping"
            )
        offset = member.get("offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise SchemaError(
                f"Register {name!r} payloadSpec member {member_name!r} has invalid offset {offset!r}"
            )
        member_mask_type = member.get("maskType")
        if member_mask_type is not None:
            members[str(member_name)] = _mask_descriptor(
                f"{name}.{member_name}",
                str(member_mask_type),
                element,
                bit_masks,
                group_masks,
                offset=offset,
            )
            continue
        raw_mask = member.get("mask")
        if raw_mask is not None:
            members[str(member_name)] = Field(
                IdentityConverter(element),
                mask=_mask_value(f"{name}.{member_name}", raw_mask),
                offset=offset,
            )
            continue
        members[str(member_name)] = Field(IdentityConverter(element), offset=offset)

    kwds: dict[str, Any] = {"length": length} if length > 1 else {}
    try:
        return types.new_class(
            f"{name}Payload",
            (StructPayload[element],),
            kwds,
            lambda namespace: namespace.update(members),
        )
    except TypeError as exc:
        # Overlapping offsets, a member wider than the register, or a member
        # named after something PayloadBase reserves.
        raise SchemaError(f"Register {name!r} has an unbuildable payloadSpec: {exc}") from exc


def _mask_descriptor(
    name: str,
    mask_type: str,
    element: Any,
    bit_masks: Mapping[str, Any],
    group_masks: Mapping[str, Any],
    *,
    offset: int = 0,
) -> Any:
    """The descriptor a ``maskType`` reference resolves to.

    ``bitMasks`` and ``groupMasks`` are different things and decode
    differently: bits are independent and combine, so they become an
    ``IntFlag`` read in place; group values are alternatives, so they become an
    ``IntEnum`` shifted down to bit 0.
    """
    if mask_type in bit_masks:
        flags = _flag_enum(mask_type, _mask_members(name, mask_type, bit_masks[mask_type], "bits"))
        return BitMask(enum=flags, offset=offset)
    if mask_type in group_masks:
        values = _mask_members(name, mask_type, group_masks[mask_type], "values")
        return GroupMask(
            enum=_int_enum(mask_type, values),
            mask=(1 << (element.itemsize * 8)) - 1,
            offset=offset,
        )
    raise SchemaError(
        f"Register {name!r} names mask {mask_type!r}, which is in neither "
        "bitMasks nor groupMasks"
    )


def _mask_members(name: str, mask_type: str, definition: Any, key: str) -> dict[str, int]:
    """The member name to value map inside a ``bitMasks``/``groupMasks`` entry.

    Both spellings the published schemas use are accepted: a bare number, and a
    mapping carrying a ``value`` alongside a description.
    """
    if not isinstance(definition, Mapping) or key not in definition:
        raise SchemaError(f"Mask {mask_type!r} (used by {name!r}) has no {key!r} section")
    members = definition[key]
    if not isinstance(members, Mapping) or not members:
        raise SchemaError(f"Mask {mask_type!r} has an empty or non-mapping {key!r} section")

    resolved: dict[str, int] = {}
    for member_name, value in members.items():
        raw = value.get("value") if isinstance(value, Mapping) else value
        resolved[str(member_name)] = _mask_value(f"{mask_type}.{member_name}", raw)
    return resolved


def _mask_value(name: str, raw: Any) -> int:
    """One mask constant, written as a number or as a ``0x``-prefixed string."""
    if isinstance(raw, bool) or raw is None:
        raise SchemaError(f"{name} has no usable mask value ({raw!r})")
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw), 0)
    except ValueError as exc:
        raise SchemaError(f"{name} has a non-numeric mask value {raw!r}") from exc


def _flag_enum(name: str, members: Mapping[str, int]) -> type[enum.IntFlag]:
    """Bits that combine -- ``IntFlag``, so ``Channel0 | Channel1`` is a value."""
    try:
        return enum.IntFlag(name, dict(members))
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Mask {name!r} is not a usable bit mask: {exc}") from exc


def _int_enum(name: str, members: Mapping[str, int]) -> type[enum.IntEnum]:
    """Alternatives -- ``IntEnum``, and upstream rejects a value not listed."""
    try:
        return enum.IntEnum(name, dict(members))
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Mask {name!r} is not a usable group mask: {exc}") from exc
