"""Building register classes from a device schema.

The register classes are what turn payload bytes into values, so every test
here decodes something: a class that is built but decodes wrongly is exactly
the failure this module exists to prevent, and it looks identical to a correct
one from the outside.
"""

import enum

import numpy as np
import pytest

from glider_harp.schema import SchemaError, build_registers, load_schema

# One entry per payload type, with a value whose bytes differ across widths
# and byte orders -- a single-byte value would pass under any endianness and
# any width, which is how a wrong register type escapes a test suite.
SWEEP = {
    "U8": (200, 1),
    "U16": (0xBEEF, 2),
    "U32": (0xDEADBEEF, 4),
    "U64": (0x0123456789ABCDEF, 8),
    "S8": (-100, 1),
    "S16": (-30000, 2),
    "S32": (-2000000000, 4),
    "S64": (-4000000000, 8),
}


def _schema(**registers):
    return {"registers": registers}


def _pack(value, size, signed):
    return int(value).to_bytes(size, "little", signed=signed)


def test_a_u8_event_register_decodes_its_payload():
    """The LicketySplit case: one byte in, one value out."""
    registers = build_registers(_schema(LickState={"address": 32, "type": "U8", "access": "Event"}))

    assert registers["LickState"].address == 32
    assert int(registers["LickState"].parse(b"\x01")) == 1


def test_a_register_is_a_class_not_an_instance():
    """``RegisterU8(32)`` returns a class; a caller that got an instance would
    find ``.parse`` bound to nothing useful."""
    registers = build_registers(_schema(LickState={"address": 32, "type": "U8"}))

    assert isinstance(registers["LickState"], type)


def test_a_u16_register_decodes_little_endian_above_one_byte():
    """The byte order the wire actually uses. 0x0201, not 0x0102."""
    registers = build_registers(_schema(Threshold={"address": 33, "type": "U16"}))

    assert int(registers["Threshold"].parse(b"\x01\x02")) == 0x0201


@pytest.mark.parametrize("type_name", sorted(SWEEP))
def test_every_integer_type_round_trips_at_its_own_width(type_name):
    """Swept rather than sampled: a table that maps U32 to a 16-bit register
    passes any test that only ever checks small values."""
    value, size = SWEEP[type_name]
    registers = build_registers(_schema(Reg={"address": 40, "type": type_name}))

    payload = _pack(value, size, signed=type_name.startswith("S"))
    assert len(payload) == size
    assert int(registers["Reg"].parse(payload)) == value


def test_a_float_register_decodes_a_float():
    registers = build_registers(_schema(Reg={"address": 40, "type": "Float"}))

    payload = np.float32(1.5).tobytes()
    assert float(registers["Reg"].parse(payload)) == 1.5


@pytest.mark.parametrize("length", [2, 3, 6])
def test_an_array_register_decodes_that_many_elements(length):
    registers = build_registers(_schema(Analog={"address": 44, "type": "U16", "length": length}))

    values = np.arange(length, dtype="<u2") * 1000
    parsed = registers["Analog"].parse(values.tobytes())

    assert isinstance(parsed, np.ndarray)
    assert len(parsed) == length
    assert list(parsed) == list(values)


def test_length_one_is_a_scalar_register():
    """``length: 1`` is how a schema spells "one element", not "an array of one"."""
    registers = build_registers(_schema(Reg={"address": 40, "type": "U8", "length": 1}))

    assert not isinstance(registers["Reg"].parse(b"\x07"), np.ndarray)
    assert int(registers["Reg"].parse(b"\x07")) == 7


def test_a_bit_mask_register_decodes_to_an_int_flag():
    """Bits combine, so the decoded value has to be a flag set, not a number."""
    registers = build_registers(
        {
            "registers": {
                "LickState": {
                    "address": 32,
                    "type": "U8",
                    "access": "Event",
                    "maskType": "LickChannels",
                }
            },
            "bitMasks": {"LickChannels": {"bits": {"Channel0": 0x1, "Channel1": 0x2}}},
        }
    )

    parsed = registers["LickState"].parse(b"\x03")

    assert isinstance(parsed, enum.IntFlag)
    assert parsed.Channel0 in parsed and parsed.Channel1 in parsed
    assert int(parsed) == 3
    assert int(registers["LickState"].parse(b"\x02")) == 2


def test_a_bit_mask_may_be_written_with_descriptions():
    """Published schemas use both spellings for a mask member's value."""
    registers = build_registers(
        {
            "registers": {"Reg": {"address": 32, "type": "U8", "maskType": "M"}},
            "bitMasks": {
                "M": {
                    "bits": {
                        "A": {"value": 0x1, "description": "first"},
                        "B": {"value": "0x2"},
                    }
                }
            },
        }
    )

    assert int(registers["Reg"].parse(b"\x03")) == 3


def test_a_bit_mask_on_a_wide_register_reads_the_whole_element():
    """The mask defaults to the base element, so a 16-bit flag word keeps its
    high byte instead of being truncated to the first."""
    registers = build_registers(
        {
            "registers": {"Reg": {"address": 32, "type": "U16", "maskType": "M"}},
            "bitMasks": {"M": {"bits": {"Low": 0x1, "High": 0x100}}},
        }
    )

    assert int(registers["Reg"].parse(b"\x01\x01")) == 0x101


def test_a_group_mask_register_decodes_to_an_int_enum():
    """Group values are alternatives, not bits."""
    registers = build_registers(
        {
            "registers": {"Mode": {"address": 33, "type": "U8", "maskType": "ModeValues"}},
            "groupMasks": {"ModeValues": {"values": {"Off": 0, "Slow": 1, "Fast": 2}}},
        }
    )

    parsed = registers["Mode"].parse(b"\x02")

    assert isinstance(parsed, enum.IntEnum)
    assert parsed.name == "Fast"


def test_a_payload_spec_register_decodes_named_members():
    registers = build_registers(
        _schema(
            AnalogData={
                "address": 44,
                "type": "S16",
                "payloadSpec": {
                    "AnalogInput0": {"offset": 0},
                    "Encoder": {"offset": 1},
                },
            }
        )
    )

    parsed = registers["AnalogData"].parse(np.array([1000, -2000], dtype="<i2").tobytes())

    assert int(parsed.AnalogInput0) == 1000
    assert int(parsed.Encoder) == -2000


# --- everything that must raise rather than build something wrong ---


def test_an_unknown_type_raises():
    with pytest.raises(SchemaError, match="U12"):
        build_registers(_schema(Reg={"address": 32, "type": "U12"}))


def test_a_missing_type_raises():
    with pytest.raises(SchemaError, match="Reg"):
        build_registers(_schema(Reg={"address": 32}))


def test_a_missing_address_raises():
    with pytest.raises(SchemaError, match="address"):
        build_registers(_schema(Reg={"type": "U8"}))


@pytest.mark.parametrize("address", [-1, 256, 1000, "32", None])
def test_an_address_no_frame_could_carry_raises(address):
    with pytest.raises(SchemaError, match="Reg"):
        build_registers(_schema(Reg={"address": address, "type": "U8"}))


def test_two_registers_at_one_address_raise():
    """An address-keyed map would silently keep one and lose the other."""
    with pytest.raises(SchemaError, match="32"):
        build_registers(
            _schema(
                First={"address": 32, "type": "U8"},
                Second={"address": 32, "type": "U8"},
            )
        )


def test_an_unknown_mask_type_raises():
    with pytest.raises(SchemaError, match="Nope"):
        build_registers(_schema(Reg={"address": 32, "type": "U8", "maskType": "Nope"}))


def test_a_masked_array_register_raises_rather_than_masking_one_element():
    with pytest.raises(SchemaError, match="maskType"):
        build_registers(
            {
                "registers": {"Reg": {"address": 32, "type": "U8", "length": 4, "maskType": "M"}},
                "bitMasks": {"M": {"bits": {"A": 0x1}}},
            }
        )


@pytest.mark.parametrize("length", [0, -2, 1.5, "3"])
def test_an_impossible_length_raises(length):
    with pytest.raises(SchemaError, match="length"):
        build_registers(_schema(Reg={"address": 32, "type": "U8", "length": length}))


def test_a_mask_with_no_members_raises():
    with pytest.raises(SchemaError, match="M"):
        build_registers(
            {
                "registers": {"Reg": {"address": 32, "type": "U8", "maskType": "M"}},
                "bitMasks": {"M": {"description": "no bits section"}},
            }
        )


def test_overlapping_payload_spec_members_raise():
    with pytest.raises(SchemaError, match="Reg"):
        build_registers(
            _schema(
                Reg={
                    "address": 44,
                    "type": "U16",
                    "payloadSpec": {"A": {"offset": 0}, "B": {"offset": 0}},
                }
            )
        )


# --- the surrounding shape ---


def test_a_schema_with_no_registers_builds_nothing():
    assert build_registers({}) == {}


def test_registers_keep_schema_order():
    """The order a person reads in the yml is the order a caller iterates."""
    registers = build_registers(
        _schema(
            Third={"address": 34, "type": "U8"},
            First={"address": 32, "type": "U8"},
            Second={"address": 33, "type": "U8"},
        )
    )

    assert list(registers) == ["Third", "First", "Second"]


def test_load_schema_reads_a_yaml_file(tmp_path):
    """Thin on purpose: the parse belongs to the caller, so this is the only
    test that needs a file on disk at all."""
    path = tmp_path / "device.yml"
    path.write_text(
        "device: LicketySplit\nwhoAmI: 1400\nregisters:\n"
        "  LickState:\n    address: 32\n    type: U8\n    access: Event\n",
        encoding="utf-8",
    )

    schema = load_schema(path)

    assert schema["whoAmI"] == 1400
    assert build_registers(schema)["LickState"].address == 32


def test_load_schema_rejects_a_file_that_is_not_a_mapping(tmp_path):
    path = tmp_path / "device.yml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(SchemaError):
        load_schema(path)
