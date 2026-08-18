"""Which registers become CSV columns, and which become actions.

Every rule here decides what a trial writes to disk, and every way of getting
one wrong produces a file that is readable, plausible and wrong: a column that
never changes, a column missing entirely, or thirty columns nobody asked for.
"""

import json
import logging

import pytest

from glider_harp import derivation
from glider_harp.derivation import CORE_REGISTERS, Derived, derive, load_profile

SCHEMA = {
    "registers": {
        "WhoAmI": {"address": 0, "type": "U16", "access": "Read"},
        "OperationControl": {"address": 10, "type": "U8", "access": "Write"},
        "LickState": {"address": 32, "type": "U8", "access": "Event"},
        "Channel0TriggerThreshold": {"address": 33, "type": "U8", "access": "Write"},
        "Channel0UntriggerThreshold": {"address": 34, "type": "U8", "access": "Write"},
    }
}

LICK = [{"register": "LickState", "as": "lick", "mode": "boolean"}]


def _schema_of(*addresses):
    """A schema with one Event register per address, named R{address}."""
    return {
        "registers": {
            f"R{address}": {"address": address, "type": "U8", "access": ["Write", "Event"]}
            for address in addresses
        }
    }


# --- core registers: identity and lifecycle, never data ---

# Spelled out rather than imported, and that is the whole point. Sweeping over
# ``CORE_REGISTERS`` itself tests nothing: drop an address from the constant and
# the sweep quietly stops covering it, so the parametrised cases below pass
# while that register leaks straight into the CSV. Caught by exactly that
# mutant. This literal is the specification -- 0 WhoAmI, 1-2 hardware version,
# 6-7 firmware version, 8 TimestampSecond, 10 OperationControl, 14 ClockConfig.
CORE = frozenset({0, 1, 2, 6, 7, 8, 10, 14})


def test_the_core_register_set_is_the_one_the_protocol_specifies():
    assert CORE_REGISTERS == CORE


def test_no_core_register_becomes_an_action():
    """Swept over every core address rather than the two this schema happens to
    carry: a rule tested against one address is a rule tested against nothing."""
    result = derive(_schema_of(*CORE), None)

    assert result.actions == {}


@pytest.mark.parametrize("address", sorted(CORE))
def test_a_profile_may_not_record_any_core_register(address):
    """Dropping it silently would leave a profile that asked for a column and
    produced none.

    Swept for the same reason the action side is, and it needed it: pinned by
    address 0 alone, narrowing this guard to ``{0}`` survived the whole suite,
    leaving a profile recording 10 or 14 unconstrained."""
    schema = _schema_of(address, 32)
    profile = {"record": [{"register": f"R{address}", "as": "who"}]}

    with pytest.raises(ValueError, match=f"R{address}"):
        derive(schema, profile)


def test_a_profile_may_record_a_non_core_register_below_fifteen():
    """The guard is eight addresses, not everything under 15: 3, 4, 5, 9, 11,
    12 and 13 are ordinary registers a device may report data on."""
    schema = _schema_of(*(sorted(set(range(15)) - CORE)))
    profile = {"record": [{"register": "R9", "as": "nine"}]}

    assert derive(schema, profile).recorded == {9: "nine"}


@pytest.mark.parametrize("address", sorted(CORE))
def test_each_core_address_is_excluded_individually(address):
    result = derive(_schema_of(address, 32), None)

    assert result.actions == {"R32": 32}


# --- no profile: silent, but fully controllable ---


def test_no_profile_records_nothing():
    """A Behavior board has ~30 registers. Widening a CSV is deliberate."""
    assert derive(SCHEMA, None).recorded == {}


def test_an_empty_profile_records_nothing():
    assert derive(SCHEMA, {}).recorded == {}


def test_a_profile_with_an_empty_record_list_records_nothing():
    assert derive(SCHEMA, {"record": []}).recorded == {}


def test_no_profile_still_exposes_every_action():
    result = derive(SCHEMA, None)

    assert result.actions == {
        "Channel0TriggerThreshold": 33,
        "Channel0UntriggerThreshold": 34,
    }


# --- actions: the control surface ---


def test_write_registers_become_actions():
    result = derive(SCHEMA, None)

    assert result.actions["Channel0TriggerThreshold"] == 33
    assert "OperationControl" not in result.actions


def test_read_registers_become_actions():
    schema = {"registers": {"Serial": {"address": 20, "type": "U16", "access": "Read"}}}

    assert derive(schema, None).actions == {"Serial": 20}


def test_event_only_registers_are_not_actions():
    """An Event register is the device talking to us; there is nothing to call."""
    schema = {"registers": {"LickState": {"address": 32, "type": "U8", "access": "Event"}}}

    assert derive(schema, None).actions == {}


def test_a_register_that_is_both_written_and_reported_is_an_action():
    """The schema allows a list of access modes, and reading only the first
    would drop a writable register out of the node editor."""
    schema = {
        "registers": {"Threshold": {"address": 33, "type": "U8", "access": ["Write", "Event"]}}
    }

    assert derive(schema, None).actions == {"Threshold": 33}


def test_a_register_with_no_access_is_not_an_action():
    schema = {"registers": {"Reg": {"address": 33, "type": "U8"}}}

    assert derive(schema, None).actions == {}


# --- a profile selects, by address, under a base name ---


def test_a_profile_selects_the_named_registers():
    assert derive(SCHEMA, {"record": LICK}).recorded == {32: "lick"}


def test_recorded_names_are_base_names_not_column_names():
    """RegisterCache expands ``lick`` into lick_state/lick_count/lick_last_ms;
    spelling a full column name here would give two places to get it wrong."""
    recorded = derive(SCHEMA, {"record": LICK}).recorded

    assert list(recorded.values()) == ["lick"]


def test_recording_a_register_does_not_remove_it_from_the_actions():
    schema = {"registers": {"Reg": {"address": 32, "type": "U8", "access": ["Write", "Event"]}}}

    result = derive(schema, {"record": [{"register": "Reg", "as": "reg"}]})

    assert result.recorded == {32: "reg"}
    assert result.actions == {"Reg": 32}


def test_a_profile_records_only_what_it_names():
    schema = _schema_of(32, 33, 34)

    result = derive(schema, {"record": [{"register": "R33", "as": "b"}]})

    assert result.recorded == {33: "b"}


def test_a_profile_may_record_several_registers():
    schema = _schema_of(32, 33)

    result = derive(
        schema, {"record": [{"register": "R32", "as": "a"}, {"register": "R33", "as": "b"}]}
    )

    assert result.recorded == {32: "a", 33: "b"}


def test_a_profile_naming_an_unknown_register_raises():
    with pytest.raises(ValueError, match="Nope"):
        derive(SCHEMA, {"record": [{"register": "Nope", "as": "x", "mode": "boolean"}]})


def test_recording_a_register_that_emits_no_events_warns(caplog):
    """Not fatal, but the column would sit at its initial value forever and
    look exactly like a device that never fired."""
    with caplog.at_level(logging.WARNING):
        result = derive(SCHEMA, {"record": [{"register": "Channel0TriggerThreshold", "as": "t"}]})

    assert result.recorded == {33: "t"}
    assert "Channel0TriggerThreshold" in caplog.text


# --- name validation: the invariant that starts being user-editable here ---


@pytest.mark.parametrize("column", ["", None, 0, [], {"a": 1}])
def test_a_missing_or_empty_column_name_raises(column):
    """``_state`` is unique and non-empty and tells a reader nothing."""
    with pytest.raises(ValueError, match="LickState"):
        derive(SCHEMA, {"record": [{"register": "LickState", "as": column}]})


@pytest.mark.parametrize("column", ["lick:state", ":lick", "lick:", "a:b:c"])
def test_a_column_name_containing_a_colon_raises(column):
    """The recorder partitions "{device_id}:{sub_column}" on the first colon."""
    with pytest.raises(ValueError, match="LickState"):
        derive(SCHEMA, {"record": [{"register": "LickState", "as": column}]})


def test_two_entries_sharing_a_column_name_raise():
    schema = _schema_of(32, 33)
    profile = {"record": [{"register": "R32", "as": "lick"}, {"register": "R33", "as": "lick"}]}

    with pytest.raises(ValueError, match="lick"):
        derive(schema, profile)


def test_a_collision_names_both_offending_entries():
    """The reason this validation lives here and not in RegisterCache: only
    this side still knows which profile entries produced the clash."""
    schema = _schema_of(32, 33)
    profile = {"record": [{"register": "R32", "as": "lick"}, {"register": "R33", "as": "lick"}]}

    with pytest.raises(ValueError) as caught:
        derive(schema, profile)

    assert "R32" in str(caught.value) and "R33" in str(caught.value)


def test_recording_one_register_twice_raises():
    profile = {
        "record": [
            {"register": "LickState", "as": "lick"},
            {"register": "LickState", "as": "lick2"},
        ]
    }

    with pytest.raises(ValueError, match="LickState"):
        derive(SCHEMA, profile)


def test_a_record_entry_that_is_not_a_mapping_raises():
    with pytest.raises(ValueError):
        derive(SCHEMA, {"record": ["LickState"]})


@pytest.mark.parametrize("register", [["LickState"], {"name": "LickState"}, 32, None])
def test_a_non_string_register_raises_a_value_error_like_everything_else(register):
    """A JSON list or object here is unhashable, so the membership test used to
    raise TypeError -- the one hand-editing mistake that came back as a
    different exception type from all the others."""
    with pytest.raises(ValueError, match="register"):
        derive(SCHEMA, {"record": [{"register": register, "as": "x"}]})


@pytest.mark.parametrize("key", ["az", "Mode", "column", "registers"])
def test_an_unknown_key_in_a_record_entry_raises(key):
    """``mode`` is reserved; anything else is a typo that would silently do
    nothing, which is what a hand-edited profile produces most often."""
    entry = {"register": "LickState", "as": "lick", key: "boolean"}

    with pytest.raises(ValueError, match=key):
        derive(SCHEMA, {"record": [entry]})


@pytest.mark.parametrize("records", [5, "LickState", {"LickState": "lick"}])
def test_a_record_block_that_is_not_a_list_raises(records):
    """An object keyed by register name is the plausible mistake, and the
    worst shape to let through: iterating it yields the register names, which
    then fail as malformed entries naming registers that do exist."""
    with pytest.raises(ValueError, match="list of entries"):
        derive(SCHEMA, {"record": records})


def test_the_reserved_mode_key_is_accepted_and_ignored():
    with_mode = derive(SCHEMA, {"record": [{"register": "LickState", "as": "lick", "mode": "x"}]})
    without = derive(SCHEMA, {"record": [{"register": "LickState", "as": "lick"}]})

    assert with_mode.recorded == without.recorded == {32: "lick"}


def test_a_profile_for_another_device_raises():
    """Register names overlap across boards; WhoAmI is what does not."""
    schema = dict(SCHEMA, whoAmI=1216)

    with pytest.raises(ValueError, match="1400"):
        derive(schema, load_profile("licketysplit"))


def test_a_matching_who_am_i_derives_normally():
    schema = dict(SCHEMA, whoAmI=1400)

    assert derive(schema, load_profile("licketysplit")).recorded == {32: "lick"}


@pytest.mark.parametrize("declared", ["1400", 0x578, "0x578", 1400])
def test_a_who_am_i_written_another_way_still_matches(declared):
    """A schema hand-copied from a datasheet quotes it, or writes it in hex.
    Compared raw, that reported a mismatch between two identical numbers."""
    schema = dict(SCHEMA, whoAmI=declared)

    assert derive(schema, load_profile("licketysplit")).recorded == {32: "lick"}


def test_a_who_am_i_mismatch_shows_both_values_as_written():
    """A value that is not a number at all still has to print recognisably."""
    schema = dict(SCHEMA, whoAmI="not-a-number")

    with pytest.raises(ValueError) as caught:
        derive(schema, load_profile("licketysplit"))

    assert "'not-a-number'" in str(caught.value)


# --- the shipped profile ---


def test_the_shipped_licketysplit_profile_loads_and_derives():
    assert derive(SCHEMA, load_profile("licketysplit")).recorded == {32: "lick"}


def test_the_shipped_licketysplit_profile_declares_the_real_who_am_i():
    """0x0578 == 1400, from the device.yml of the published board."""
    assert load_profile("licketysplit")["who_am_i"] == 0x0578


def test_the_shipped_profile_is_valid_json_with_the_expected_shape():
    profile = load_profile("licketysplit")

    assert profile["schema_version"] == "1.0"
    assert profile["name"] == "LicketySplit"
    assert profile["record"] == [{"register": "LickState", "as": "lick", "mode": "boolean"}]


@pytest.mark.parametrize("version", ["2.0", "0.9", 2, "11.0"])
def test_a_profile_from_another_format_version_raises(version):
    """The gate that makes strict record keys safe: without it a 1.1 profile
    carrying a new key fails with "unknown keys: scale", which names the key
    rather than the version and reads like a typo in a correct file."""
    profile = {"schema_version": version, "record": LICK}

    with pytest.raises(ValueError, match="schema_version"):
        derive(SCHEMA, profile)


@pytest.mark.parametrize("version", ["1.0", "1.4", 1, "1"])
def test_any_minor_version_of_this_format_is_accepted(version):
    """A minor bump is additive by definition, so only the major is gated."""
    assert derive(SCHEMA, {"schema_version": version, "record": LICK}).recorded == {32: "lick"}


def test_a_profile_with_no_declared_version_is_assumed_current():
    assert derive(SCHEMA, {"record": LICK}).recorded == {32: "lick"}


def test_an_unreadable_schema_version_raises():
    with pytest.raises(ValueError, match="schema_version"):
        derive(SCHEMA, {"schema_version": "one point oh", "record": LICK})


def test_load_profile_gates_the_version_too(tmp_path, monkeypatch):
    """``load_profile`` is only the shipped-profile path, and Task 11 will read
    user files of its own -- so the gate has to hold at both entry points, and
    checking it only in ``derive`` would leave this one open."""
    monkeypatch.setattr(derivation, "PROFILE_DIR", tmp_path)
    (tmp_path / "future.json").write_text(
        json.dumps({"schema_version": "2.0", "name": "Future", "record": []}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="schema_version"):
        derivation.load_profile("future")


def test_an_unknown_profile_raises():
    with pytest.raises(FileNotFoundError, match="nosuchdevice"):
        load_profile("nosuchdevice")


@pytest.mark.parametrize("name", ["../secrets", "a/b", "..", "", "a.b"])
def test_a_profile_name_that_is_a_path_raises(name):
    """The name comes from a device setting, so it is a name, not a path."""
    with pytest.raises(ValueError):
        load_profile(name)


def test_load_profile_returns_a_plain_dict():
    assert json.dumps(load_profile("licketysplit"))


# --- the shape of the result ---


def test_derived_defaults_to_empty():
    empty = Derived()

    assert empty.recorded == {} and empty.actions == {}


def test_a_schema_with_no_registers_derives_nothing():
    result = derive({}, None)

    assert result.recorded == {} and result.actions == {}


# --- what the record can actually decode ---------------------------------


@pytest.mark.parametrize(
    "entry, fragment",
    [
        ({"address": 32, "type": "S16", "access": "Event"}, "'S16'"),
        ({"address": 32, "type": "Float", "access": "Event"}, "'Float'"),
        ({"address": 32, "type": "U8", "access": "Event", "length": 4}, "4 elements"),
    ],
    ids=["signed", "float", "array"],
)
def test_recording_a_register_the_cache_cannot_decode_is_refused(entry, fragment):
    """``RegisterCache`` reads every payload as one unsigned little-endian
    integer, so an S16 of -1 is recorded as 65535, a Float of 1.5 as
    1069547520, and a four-element array as one implausible number.

    Refused rather than warned because every one of those files opens cleanly,
    plots, and is wrong: nothing downstream can tell 65535 from a reading. And
    ``HarpDevice`` packs *writes* by the declared type, so a signed register
    written as -1 and read back through the record returns a different number
    than it was given, inside one program.
    """
    schema = {"registers": {"Reg": entry}}
    with pytest.raises(ValueError, match="Reg") as excinfo:
        derive(schema, {"record": [{"register": "Reg", "as": "r"}]})
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize("declared", ["U8", "U16", "U32", "U64"])
def test_every_unsigned_width_may_be_recorded(declared):
    """The gate is exactly what the cache can decode, not a narrower guess."""
    schema = {"registers": {"Reg": {"address": 32, "type": declared, "access": "Event"}}}
    assert derive(schema, {"record": [{"register": "Reg", "as": "r"}]}).recorded == {32: "r"}


def test_a_signed_register_is_still_usable_as_an_action():
    """Only the recorded side is gated. Writes go out with the correct width
    and signedness already, so refusing the action too would take away
    something that works."""
    schema = {"registers": {"Offset": {"address": 32, "type": "S16", "access": "Write"}}}
    assert derive(schema, None).actions == {"Offset": 32}
