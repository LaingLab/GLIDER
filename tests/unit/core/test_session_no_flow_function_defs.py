"""Flow-function *definitions* have been removed from the session model.

The inline graph-function system (StartFunction/EndFunction nodes, manual
controls) is unaffected and covered by its own tests.
"""

from glider.core.experiment_session import ExperimentSession


def test_to_dict_has_no_flow_functions_key():
    assert "flow_functions" not in ExperimentSession().to_dict()


def test_from_dict_ignores_legacy_flow_functions_without_crashing():
    data = ExperimentSession().to_dict()
    data["flow_functions"] = {"abc": {"id": "abc", "name": "Legacy"}}
    # Must load cleanly; the legacy definitions are simply dropped.
    session = ExperimentSession.from_dict(data)
    assert "flow_functions" not in session.to_dict()


def test_session_has_no_flow_function_definition_api():
    session = ExperimentSession()
    for attr in (
        "flow_function_definitions",
        "add_flow_function_definition",
        "get_flow_function_definition",
        "remove_flow_function_definition",
    ):
        assert not hasattr(session, attr), f"{attr} should be removed"
