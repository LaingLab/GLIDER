"""Writes that must never destroy what was there (review 2026-09-28: A2, A11, A-suspects)."""

import os
from pathlib import Path

import pytest

from glider.core import device_library
from glider.core.data_recorder import DataRecorder
from glider.core.experiment_session import ExperimentSession, NodeConfig
from glider.core.hardware_manager import HardwareManager
from glider.core.project import Project, ProjectError


def test_a_session_that_cannot_serialize_leaves_the_old_file_intact(tmp_path):
    path = tmp_path / "exp.glider"
    session = ExperimentSession()
    session.save(str(path))
    before = path.read_text()

    session.add_node(NodeConfig(id="n", node_type="Delay", state={"bad": {1, 2}}))
    with pytest.raises(TypeError):
        session.save(str(path))

    assert path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["exp.glider"]  # no temp left


def test_an_unreadable_manifest_is_an_error_not_an_empty_project(tmp_path, monkeypatch):
    (tmp_path / "glider_project.json").write_text("{}")

    def denied(*_a, **_k):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(ProjectError, match="denied"):
        Project.load(tmp_path)


def test_a_non_integer_schema_version_is_refused(tmp_path):
    (tmp_path / "glider_project.json").write_text('{"schema_version": "99"}')
    with pytest.raises(ProjectError, match="schema_version"):
        Project.load(tmp_path)


def test_a_custom_device_cannot_take_a_built_in_name(tmp_path):
    from glider.hal.base_device import DEVICE_REGISTRY

    taken = next(iter(DEVICE_REGISTRY))
    with pytest.raises(ValueError, match="already a device type"):
        device_library._check_name_free(taken)


async def test_a_rerun_in_the_same_second_does_not_overwrite_the_csv(tmp_path):
    first = DataRecorder(HardwareManager())
    first.set_output_directory(tmp_path)
    second = DataRecorder(HardwareManager())
    second.set_output_directory(tmp_path)

    a = await first.start("run")
    b = await second.start("run")
    await first.stop()
    await second.stop()

    assert a != b
    assert os.path.exists(a) and os.path.exists(b)
