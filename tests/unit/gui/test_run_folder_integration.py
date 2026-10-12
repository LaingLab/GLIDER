from datetime import date
from pathlib import Path

import pytest

from glider.core.experiment_session import RunField

pytestmark = pytest.mark.usefixtures("qtbot")


@pytest.fixture
def window(main_window_factory, tmp_path):
    w = main_window_factory()
    w._core.session.metadata.recording_directory = str(tmp_path)
    return w


def test_no_run_fields_keeps_flat_folder(window):
    assert window._prepare_run_folder() is None


def test_run_fields_make_a_numbered_folder(window, tmp_path):
    md = window._core.session.metadata
    md.experiment_type = "Pup retrieval"
    md.run_fields = [RunField("dam_id", "Dam ID", True), RunField("notes2", "Extra")]
    md.run_values = {"dam_id": "D14", "notes2": "ignored"}
    folder = window._prepare_run_folder()
    stamp = f"{date.today():%Y%m%d}"
    assert folder == tmp_path / "pup-retrieval" / f"d14_{stamp}_run01"
    assert folder.is_dir()
    assert window._prepare_run_folder().name == f"d14_{stamp}_run02"


def test_run_info_shows_next_number(window, tmp_path):
    md = window._core.session.metadata
    md.experiment_type = "OFT"
    md.run_fields = [RunField("animal_id", "Animal ID", True)]
    md.run_values = {"animal_id": "M031"}
    window._refresh_run_info()
    assert window._run_header._run_label.text() == "Run 1 today"
    assert "m031_" in window._run_header._tags._trailing.text()


def test_values_reset_after_a_run(window):
    md = window._core.session.metadata
    md.run_fields = [RunField("animal_id", "Animal ID", True)]
    md.run_values = {"animal_id": "M031"}

    class S:
        def __init__(self, n):
            self.name = n

    window._on_core_state_change(S("RUNNING"))
    window._on_core_state_change(S("STOPPED"))
    assert md.run_values == {"animal_id": ""}


def test_flat_run_after_preset_run_does_not_reuse_run_folder(window, tmp_path, monkeypatch):
    md = window._core.session.metadata
    md.run_fields = [RunField("animal_id", "Animal ID", True)]
    md.run_values = {"animal_id": "M031"}
    folder = window._apply_recording_directory()
    assert window._core.data_recorder._output_dir == folder

    md.run_fields = []
    md.recording_directory = ""
    monkeypatch.chdir(tmp_path)
    assert window._apply_recording_directory() is None
    assert window._core.data_recorder._output_dir == Path.cwd()


def test_error_after_run_keeps_values(window):
    md = window._core.session.metadata
    md.run_fields = [RunField("animal_id", "Animal ID", True)]
    md.run_values = {"animal_id": "M031"}

    class S:
        def __init__(self, n):
            self.name = n

    window._on_core_state_change(S("RUNNING"))
    window._on_core_state_change(S("ERROR"))
    assert md.run_values == {"animal_id": "M031"}


def test_failed_start_removes_empty_run_folder(window, tmp_path):
    import asyncio

    md = window._core.session.metadata
    md.experiment_type = "OFT"
    md.run_fields = [RunField("animal_id", "Animal ID", True)]
    md.run_values = {"animal_id": "M031"}

    async def boom():
        raise RuntimeError("no start")

    window._core.start_experiment = boom
    window._notify_user = lambda *a, **k: None
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(window._start_async())
    finally:
        loop.close()
    assert list((tmp_path / "oft").iterdir()) == []
