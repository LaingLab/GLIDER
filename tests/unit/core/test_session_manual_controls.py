from glider.core.experiment_session import ExperimentSession


def test_new_session_has_empty_manual_controls():
    assert ExperimentSession().manual_controls == []


def test_set_and_get_manual_controls_marks_dirty():
    session = ExperimentSession()
    session._mark_clean()
    entries = [{"slot": 0, "start_node_id": "s1", "label": "Purge"}]
    session.set_manual_controls(entries)
    assert session.manual_controls == entries
    assert session.is_dirty is True


def test_manual_controls_round_trip_through_dict():
    session = ExperimentSession()
    entries = [
        {"slot": 0, "start_node_id": "s1", "label": "Purge"},
        {"slot": 1, "start_node_id": "s2", "label": "Flush"},
    ]
    session.set_manual_controls(entries)
    restored = ExperimentSession.from_dict(session.to_dict())
    assert restored.manual_controls == entries


def test_old_file_without_field_loads_empty():
    data = ExperimentSession().to_dict()
    data.pop("manual_controls", None)
    restored = ExperimentSession.from_dict(data)
    assert restored.manual_controls == []


def test_empty_manual_controls_omitted_from_dict():
    assert "manual_controls" not in ExperimentSession().to_dict()
