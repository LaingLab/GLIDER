"""MCP analysis tools over the synthetic recordings this folder's conftest builds."""

import pytest

pytest.importorskip("mcp")

from glider.mcp import analysis  # noqa: E402


def test_session_summary_lists_what_can_be_asked(synthetic_recording):
    s = analysis.session_summary(str(synthetic_recording))
    assert s["has"]["tracking"] and s["has"]["events"]
    assert s["states"] == ["active", "resting", "unknown"]  # unknown = pre-flow frames
    assert s["zones"] == ["zone1"]
    assert s["object_ids"] == [0]
    assert s["frame_rate"] == pytest.approx(30.0, rel=0.05)
    assert s["duration_s"] == pytest.approx(4.0, abs=0.2)
    assert s["event_sources"]


def test_ethogram_totals_and_transitions(synthetic_recording):
    result = analysis.ethogram(str(synthetic_recording))
    totals = {t["state"]: t for t in result["state_totals"]}
    assert totals["resting"]["n_bouts"] == 2
    assert totals["active"]["n_bouts"] == 1
    assert sum(t["percent"] for t in totals.values()) == pytest.approx(100.0)
    assert result["columns"] == ["object_id", "state", "start_ms", "end_ms", "duration_ms"]

    transitions = analysis.ethogram(str(synthetic_recording), kind="transitions")
    assert transitions["columns"] == ["from_state", "to_state", "count"]
    with pytest.raises(ValueError, match="kind"):
        analysis.ethogram(str(synthetic_recording), kind="bouts")


def test_trajectory_kinds(synthetic_recording):
    dwell = analysis.trajectory(str(synthetic_recording))
    assert dwell["rows"][0]["zone"] == "zone1"
    positions = analysis.trajectory(str(synthetic_recording), kind="positions")
    assert positions["total_rows"] > 100
    assert "center_x" in positions["columns"]


def test_kinematics_summary(synthetic_recording):
    result = analysis.kinematics(str(synthetic_recording))
    assert result["speed_units"] == "px/s"
    assert result["max_speed"] > 0
    assert result["columns"] == ["bin_center", "count"]


def test_events_and_event_triggered(synthetic_recording):
    s = analysis.session_summary(str(synthetic_recording))
    source = s["event_sources"][0]
    found = analysis.events(str(synthetic_recording), source=source)
    assert found["total_rows"] >= 1
    triggered = analysis.events(str(synthetic_recording), source=source, window_s=[0.5, 1.0])
    assert triggered["columns"] == ["trial_id", "event_time_ms", "time_offset_ms", "value"]
    with pytest.raises(ValueError, match="source"):
        analysis.events(str(synthetic_recording), window_s=[0.5, 1.0])


def test_out_csv_writes_full_table_and_previews(synthetic_recording, tmp_path):
    out = tmp_path / "positions.csv"
    result = analysis.trajectory(str(synthetic_recording), kind="positions", out_csv=str(out))
    assert result["csv"] == str(out)
    assert len(result["rows"]) == analysis.PREVIEW_ROWS
    assert result["truncated"] is True
    assert sum(1 for _ in out.open()) == result["total_rows"] + 1  # header
    with pytest.raises(ValueError, match="already exists"):
        analysis.trajectory(str(synthetic_recording), kind="positions", out_csv=str(out))


def test_missing_tracking_says_what_to_do(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="session_summary"):
        analysis.ethogram(str(empty))
    with pytest.raises(ValueError, match="not a folder"):
        analysis.session_summary(str(tmp_path / "nope"))


def test_find_recordings_and_doctor(synthetic_recording):
    root = synthetic_recording.parent
    found = analysis.find_recordings(str(root))
    assert [s["recording"] for s in found["sessions"]] == [str(synthetic_recording.resolve())]
    report = analysis.project_doctor(str(root))
    assert isinstance(report["findings"], list)
    assert "finding" in report["summary"]


@pytest.mark.parametrize("kind", ["ethogram", "trajectory", "occupancy", "zone_dwell", "velocity"])
def test_plot_writes_png(synthetic_recording, tmp_path, kind):
    out = tmp_path / f"{kind}.png"
    result = analysis.plot(str(synthetic_recording), kind, str(out))
    assert result == out
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_rejects_unknown_kind_without_writing(synthetic_recording, tmp_path):
    out = tmp_path / "x.png"
    with pytest.raises(ValueError, match="kind"):
        analysis.plot(str(synthetic_recording), "pie", str(out))
    assert not out.exists()


def test_unknown_object_id_names_the_tracked_ones(synthetic_recording, tmp_path):
    with pytest.raises(ValueError, match=r"object_id 7.*\[0\]"):
        analysis.ethogram(str(synthetic_recording), object_id=7)
    with pytest.raises(ValueError, match="object_id 7"):
        analysis.kinematics(str(synthetic_recording), object_id=7)
    with pytest.raises(ValueError, match="object_id 7"):
        analysis.plot(str(synthetic_recording), "trajectory", str(tmp_path / "p.png"), object_id=7)
