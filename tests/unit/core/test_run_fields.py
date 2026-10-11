from glider.core.experiment_session import RunField, SessionMetadata


def _md():
    md = SessionMetadata()
    md.run_fields = [
        RunField("dam_id", "Dam ID", True),
        RunField("litter", "Litter size", False, "6"),
    ]
    return md


def test_run_fields_round_trip():
    back = SessionMetadata.from_dict(_md().to_dict())
    assert back.run_fields == _md().run_fields


def test_run_values_never_serialized():
    md = _md()
    md.run_values = {"dam_id": "D14"}
    d = md.to_dict()
    assert "run_values" not in d
    assert SessionMetadata.from_dict(d).run_values == {"dam_id": "", "litter": "6"}


def test_old_file_loads_without_run_fields():
    d = SessionMetadata().to_dict()
    d.pop("run_fields")
    md = SessionMetadata.from_dict(d)
    assert md.run_fields == [] and md.run_values == {}


def test_reset_run_values_uses_defaults():
    md = _md()
    md.run_values = {"dam_id": "D14", "stale": "x"}
    md.reset_run_values()
    assert md.run_values == {"dam_id": "", "litter": "6"}
