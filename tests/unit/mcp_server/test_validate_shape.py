"""Validator: content that would be dropped, misspelled fields, and settings."""

import copy

import pytest

pytest.importorskip("mcp")

from glider.core.flow_engine import FlowEngine  # noqa: E402
from glider.mcp import experiments  # noqa: E402
from glider.nodes.base_node import ExecNode, NodeCategory, NodeDefinition  # noqa: E402

BASE = {
    "metadata": {"name": "t"},
    "hardware": {"boards": [], "devices": []},
    "flow": {
        "nodes": [
            {"id": "start", "node_type": "StartExperiment", "position": [0, 0]},
            {
                "id": "wait",
                "node_type": "Delay",
                "position": [250, 0],
                "state": {"duration": 500, "unit": "milliseconds"},
            },
            {"id": "end", "node_type": "EndExperiment", "position": [500, 0]},
        ],
        "connections": [
            {
                "id": "c1",
                "from_node": "start",
                "from_output": 0,
                "to_node": "wait",
                "to_input": 0,
                "connection_type": "exec",
            },
            {
                "id": "c2",
                "from_node": "wait",
                "from_output": 0,
                "to_node": "end",
                "to_input": 0,
                "connection_type": "exec",
            },
        ],
    },
}


def edited(edit):
    data = copy.deepcopy(BASE)
    edit(data)
    return data


def paths(findings):
    return [f["path"] for f in findings]


async def test_base_is_valid():
    report = await experiments.validate(BASE)
    assert report["valid"], report
    assert report["warnings"] == []


async def test_top_level_nodes_is_an_error_naming_the_fix():
    data = edited(lambda d: d.__setitem__("nodes", d["flow"].pop("nodes")))
    report = await experiments.validate(data)
    err = next(e for e in report["errors"] if e["path"] == "nodes")
    assert "flow.nodes" in err["message"]


async def test_unknown_top_level_key_is_a_warning():
    report = await experiments.validate(edited(lambda d: d.__setitem__("notes", "hi")))
    assert report["valid"]
    assert "notes" in paths(report["warnings"])


@pytest.mark.parametrize(
    "edit, path, fragment",
    [
        (
            lambda d: d["flow"]["nodes"][1].update(type=d["flow"]["nodes"][1].pop("node_type")),
            "flow.nodes[1]",
            "did you mean 'node_type'",
        ),
        (
            lambda d: d["flow"]["connections"][0].update(
                from_port=d["flow"]["connections"][0].pop("from_output")
            ),
            "flow.connections[0]",
            "did you mean 'from_output'",
        ),
    ],
)
async def test_missing_field_suggests_the_right_name(edit, path, fragment):
    report = await experiments.validate(edited(edit))
    err = next(e for e in report["errors"] if e["path"] == path)
    assert fragment in err["message"]


async def test_missing_field_never_suggests_a_legitimate_field():
    report = await experiments.validate(
        edited(lambda d: d["flow"]["connections"][0].pop("to_node"))
    )
    err = next(e for e in report["errors"] if e["path"] == "flow.connections[0]")
    assert "missing to_node" in err["message"]
    assert "did you mean" not in err["message"]


async def test_node_settings_outside_state_is_an_error():
    def edit(d):
        d["flow"]["nodes"][1]["settings"] = d["flow"]["nodes"][1].pop("state")

    report = await experiments.validate(edited(edit))
    err = next(e for e in report["errors"] if e["path"] == "flow.nodes[1].settings")
    assert "state" in err["message"]


async def test_unknown_item_field_is_a_warning():
    report = await experiments.validate(edited(lambda d: d["flow"]["nodes"][1].update(pin=11)))
    assert report["valid"]
    assert "flow.nodes[1].pin" in paths(report["warnings"])


@pytest.mark.parametrize(
    "state, path",
    [
        ({"duration": 500, "unit": "msec"}, "flow.nodes[1].state.unit"),
        ({"duration": -1}, "flow.nodes[1].state.duration"),
        ({"duration": "500"}, "flow.nodes[1].state.duration"),
    ],
)
async def test_bad_setting_values_are_errors(state, path):
    report = await experiments.validate(
        edited(lambda d: d["flow"]["nodes"][1].__setitem__("state", state))
    )
    assert path in paths(report["errors"]), report["errors"]


async def test_misspelled_setting_is_a_warning_with_suggestion():
    report = await experiments.validate(
        edited(lambda d: d["flow"]["nodes"][1].__setitem__("state", {"durration": 5}))
    )
    assert report["valid"]
    warn = next(w for w in report["warnings"] if w["path"] == "flow.nodes[1].state.durration")
    assert "did you mean 'duration'" in warn["hint"]


@pytest.mark.parametrize("state", [[], "x", 3])
async def test_non_object_state_is_an_error(state):
    report = await experiments.validate(
        edited(lambda d: d["flow"]["nodes"][1].__setitem__("state", state))
    )
    assert "flow.nodes[1].state" in paths(report["errors"])


def _with_device(device_type, settings):
    def edit(d):
        d["hardware"]["boards"].append({"id": "uno", "driver_type": "arduino"})
        d["hardware"]["devices"].append(
            {
                "id": "dev",
                "device_type": device_type,
                "name": "dev",
                "board_id": "uno",
                "pins": {},
                "settings": settings,
            }
        )

    return edited(edit)


async def test_bad_device_setting_is_an_error():
    report = await experiments.validate(_with_device("GenericSerial", {"baudrate": "fast"}))
    assert "hardware.devices[0].settings.baudrate" in paths(report["errors"])


async def test_device_without_schema_gets_no_settings_warnings():
    report = await experiments.validate(_with_device("DigitalOutput", {"anything": 1}))
    assert not any(".settings." in p for p in paths(report["warnings"]))


@pytest.mark.parametrize("settings", [[], "x"])
async def test_non_object_device_settings_is_an_error(settings):
    report = await experiments.validate(_with_device("GenericSerial", settings))
    assert "hardware.devices[0].settings" in paths(report["errors"])


class _Schemaed(ExecNode):
    definition = NodeDefinition(name="Plug", category=NodeCategory.LOGIC)
    PROPERTIES_SCHEMA = [{"key": "level", "label": "Level", "type": "int", "default": 1}]

    async def execute(self) -> None:
        pass


class _Bare(ExecNode):
    definition = NodeDefinition(name="Bare", category=NodeCategory.LOGIC)

    async def execute(self) -> None:
        pass


async def test_plugin_node_settings_are_checked_and_bare_plugins_ignored(monkeypatch):
    await experiments.get_core()
    monkeypatch.setitem(FlowEngine._node_registry, "Plug", _Schemaed)
    monkeypatch.setitem(FlowEngine._node_registry, "Bare", _Bare)

    def edit(d):
        d["flow"]["nodes"] += [
            {"id": "p", "node_type": "Plug", "position": [0, 200], "state": {"level": 1.5}},
            {"id": "b", "node_type": "Bare", "position": [0, 400], "state": {"whatever": 1}},
        ]

    report = await experiments.validate(edited(edit))
    assert "flow.nodes[3].state.level" in paths(report["errors"])
    assert not any(p.startswith("flow.nodes[4].state") for p in paths(report["warnings"]))


async def test_nodes_using_type_are_told_to_use_node_type():
    def edit(d):
        for n in d["flow"]["nodes"]:
            n["type"] = n.pop("node_type")

    report = await experiments.validate(edited(edit))
    assert not report["valid"]
    err = report["errors"][0]
    assert err["message"].startswith("nodes use 'node_type', not 'type'")
    # Still offered, as the second sentence, for a file an old GLIDER really wrote.
    assert err["hint"].index("rename") < err["hint"].index("re-save")


async def test_legacy_boards_alone_keep_the_legacy_format_message():
    legacy = {"hardware": {"boards": [{"id": "b", "type": "arduino"}]}}
    report = await experiments.validate(legacy)
    assert "legacy serializer format" in report["errors"][0]["message"]


def _call_flow(state, with_function=True):
    def edit(d):
        extra = [{"id": "call", "node_type": "FunctionCall", "position": [0, 200]}]
        if state is not None:
            extra[0]["state"] = state
        if with_function:
            extra.append(
                {
                    "id": "fn",
                    "node_type": "StartFunction",
                    "position": [0, 400],
                    "state": {"function_name": "Beep"},
                }
            )
        d["flow"]["nodes"] += extra

    return edited(edit)


CALL_ID = "flow.nodes[3].state.function_start_id"


@pytest.mark.parametrize(
    "state", [None, {}, {"function_name": "Beep"}, {"function_start_id": None}]
)
async def test_function_call_without_function_start_id_is_an_error(state):
    report = await experiments.validate(_call_flow(state))
    err = next(e for e in report["errors"] if e["path"] == CALL_ID)
    assert "calls nothing" in err["message"]
    assert "StartFunction" in err["hint"]


async def test_function_call_naming_a_start_function_is_clean():
    report = await experiments.validate(_call_flow({"function_start_id": "fn"}))
    assert CALL_ID not in paths(report["errors"] + report["warnings"])
    assert not any("function_start_id" in w["message"] for w in report["warnings"])


async def test_function_call_with_an_unknown_function_id_names_the_ids_present():
    report = await experiments.validate(_call_flow({"function_start_id": "gone"}))
    err = next(e for e in report["errors"] if e["path"] == CALL_ID)
    assert "'gone'" in err["message"]
    assert "'fn'" in err["hint"]


async def test_function_call_id_of_another_node_type_is_an_error():
    report = await experiments.validate(_call_flow({"function_start_id": "wait"}))
    assert CALL_ID in paths(report["errors"])


async def test_function_call_with_no_start_function_in_the_flow_says_to_add_one():
    report = await experiments.validate(
        _call_flow({"function_start_id": "fn"}, with_function=False)
    )
    err = next(e for e in report["errors"] if e["path"] == CALL_ID)
    assert "add a StartFunction" in err["hint"]


@pytest.mark.parametrize("state", [["x"], {"function_start_id": 5}, {"function_start_id": ["fn"]}])
async def test_function_call_with_malformed_state_does_not_raise(state):
    report = await experiments.validate(_call_flow(state))
    assert not report["valid"]
