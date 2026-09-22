"""Finding the sessions under a folder, and what group each belongs to."""

from __future__ import annotations

import json
from pathlib import Path

from glider.analysis.cohort import discover_sessions, session_id_for


def _tracking(folder: Path, video_stem: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rec_tracking.csv").write_text(
        "# GLIDER Tracking Data\n\nframe,timestamp,elapsed_ms,object_id,behavioral_state\n"
        "1,2026-05-25T14:00:30.000,0.0,0,rest\n",
        encoding="utf-8",
    )
    if video_stem:
        (folder / f"{video_stem}.mp4").write_bytes(b"")
    return folder


def _ethogram(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "ethogram_raw.csv").write_text("frame,behavior\n0,rest\n", encoding="utf-8")
    return folder / "ethogram_raw.csv"


def test_an_ethogram_claims_the_recording_above_it(tmp_path):
    recording = _tracking(tmp_path / "rec")
    ethogram = _ethogram(recording / "v")
    sources, warning = discover_sessions(tmp_path)
    assert warning is None
    assert [(s.path.resolve(), s.recording) for s in sources] == [
        (ethogram.resolve(), recording.resolve())
    ]


def test_an_unclaimed_recording_is_its_own_session(tmp_path):
    recording = _tracking(tmp_path / "rec")
    sources, _ = discover_sessions(tmp_path)
    assert [(s.path, s.recording) for s in sources] == [(recording.resolve(), recording.resolve())]


def test_session_ids(tmp_path):
    assert session_id_for(_ethogram(tmp_path / "sessions" / "s1" / "analysis")) == "s1"
    assert session_id_for(_ethogram(tmp_path / "out" / "M14_day3")) == "M14_day3"
    assert session_id_for(_tracking(tmp_path / "raw", video_stem="M15_day3")) == "M15_day3"
    assert session_id_for(_tracking(tmp_path / "bare")) == "bare"


def test_a_distant_sessions_folder_is_not_mistaken_for_the_layout(tmp_path):
    """~/data/sessions/cohortA/outputs/t0/ is animal t0, not animal cohortA."""
    ethogram = _ethogram(tmp_path / "sessions" / "cohortA" / "outputs" / "t0")
    assert session_id_for(ethogram) == "t0"


def test_groups_come_from_the_project_manifest(tmp_path):
    _ethogram(tmp_path / "sessions" / "s1" / "analysis")
    _ethogram(tmp_path / "sessions" / "s2" / "analysis")
    (tmp_path / "glider_project.json").write_text(
        json.dumps({"sessions": {"s1": {"group": "ChR2"}}}), encoding="utf-8"
    )
    sources, warning = discover_sessions(tmp_path)
    assert warning is None
    assert {s.session_id: s.group for s in sources} == {"s1": "ChR2", "s2": ""}


def test_a_broken_manifest_warns_and_loads_ungrouped(tmp_path):
    _ethogram(tmp_path / "sessions" / "s1" / "analysis")
    (tmp_path / "glider_project.json").write_text("{not json", encoding="utf-8")
    sources, warning = discover_sessions(tmp_path)
    assert [s.group for s in sources] == [""]
    assert warning and "ungrouped" in warning


def _not_utf8(folder: Path) -> Path:
    """A lab's own CSV saved from Excel on Windows: cp1252, not UTF-8."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "notes.csv"
    path.write_bytes("dose,µg\n".encode("cp1252"))
    return path


def test_a_non_utf8_csv_beside_a_recording_is_ignored(tmp_path):
    from glider.analysis._io import discover

    recording = _tracking(tmp_path / "rec")
    _not_utf8(recording)
    assert discover(recording).tracking == recording / "rec_tracking.csv"


def test_a_non_utf8_csv_does_not_stop_discovery(tmp_path):
    recording = _tracking(tmp_path / "rec")
    _not_utf8(recording)
    _not_utf8(tmp_path / "notes_only")
    sources, _ = discover_sessions(tmp_path)
    assert [s.path for s in sources] == [recording.resolve()]


def _manifest(root: Path, raw: bytes) -> None:
    _ethogram(root / "sessions" / "s1" / "analysis")
    (root / "glider_project.json").write_bytes(raw)


def test_a_misshapen_manifest_warns_and_loads_ungrouped(tmp_path):
    for i, raw in enumerate(
        [
            b'{"sessions": ["a"]}',
            b'{"sessions": {"a": "ctrl"}}',
            b'{"subjects": ["x"]}',
            '{"name": "µg"}'.encode("cp1252"),
        ]
    ):
        root = tmp_path / str(i)
        _manifest(root, raw)
        sources, warning = discover_sessions(root)
        assert [s.group for s in sources] == [""], raw
        assert warning and "ungrouped" in warning, raw
