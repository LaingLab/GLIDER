"""Which registers become CSV columns, and which become actions.

Every rule here decides what a trial writes to disk, and every way of getting
one wrong produces a file that is readable, plausible and wrong: a column that
never changes, a column missing entirely, or thirty columns nobody asked for.
"""

import json
import logging

import pytest

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


def test_core_registers_are_never_recorded():
    assert all(address not in derive(SCHEMA, None).recorded for address in CORE)


def test_no_core_register_becomes_an_action():
    """Swept over every core address rather than the two this schema happens to
    carry: a rule tested against one address is a rule tested against nothing."""
    result = derive(_schema_of(*CORE), None)

    assert result.actions == {}


def test_a_profile_may_not_record_a_core_register():
    """Dropping it silently would leave a profile that asked for a column and
    produced none."""
    schema = _schema_of(0, 32)
    profile = {"record": [{"register": "R0", "as": "who"}]}

    with pytest.raises(ValueError, match="R0"):
        derive(schema, profile)


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


def test_a_profile_for_another_device_raises():
    """Register names overlap across boards; WhoAmI is what does not."""
    schema = dict(SCHEMA, whoAmI=1216)

    with pytest.raises(ValueError, match="1400"):
        derive(schema, load_profile("licketysplit"))


def test_a_matching_who_am_i_derives_normally():
    schema = dict(SCHEMA, whoAmI=1400)

    assert derive(schema, load_profile("licketysplit")).recorded == {32: "lick"}


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
