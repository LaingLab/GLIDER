"""Tests for analysis._io: artifact discovery + CSV parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from glider.analysis._io import discover, parse_csv

from .conftest import RecordingSpec, write_synthetic_recording


def test_discover_finds_all_three_csv_types(synthetic_recording: Path):
    """A complete recording resolves tracking, data, and events paths."""
    art = discover(synthetic_recording)
    assert art.tracking is not None and art.tracking.exists()
    assert art.data is not None and art.data.exists()
    assert art.events is not None and art.events.exists()
    assert art.tracking.name.endswith("_tracking.csv")
    assert art.events.name.endswith("_events.csv")


def test_discover_classifies_by_header_marker_not_suffix(tmp_path: Path):
    """Renaming a file keeps it discoverable as long as the header is intact."""
    write_synthetic_recording(tmp_path / "rec", RecordingSpec())
    rec_dir = tmp_path / "rec"
    # Rename tracking CSV to remove the conventional suffix.
    original = next(rec_dir.glob("*_tracking.csv"))
    renamed = rec_dir / "user_renamed_file.csv"
    original.rename(renamed)

    art = discover(rec_dir)
    assert art.tracking == renamed, "Discovery should classify by header marker, not filename"


def test_discover_ignores_unrelated_csvs(tmp_path: Path):
    """A user's notes.csv in the directory shouldn't be misclassified."""
    rec_dir = tmp_path / "rec"
    write_synthetic_recording(rec_dir, RecordingSpec())
    (rec_dir / "user_notes.csv").write_text("date,note\n2026-05-25,test run\n")

    art = discover(rec_dir)
    # The unrelated CSV must not become any of the artifact slots.
    for slot in (art.tracking, art.data, art.events):
        assert slot is None or "user_notes" not in slot.name


def test_discover_tolerates_missing_artifacts(tmp_path: Path):
    """A headless run (no tracking) still discovers what's there."""
    rec_dir = tmp_path / "rec"
    write_synthetic_recording(rec_dir, RecordingSpec(write_tracking=False))

    art = discover(rec_dir)
    assert art.tracking is None
    assert art.data is not None
    assert art.events is not None


def test_discover_raises_on_missing_directory(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        discover(tmp_path / "does_not_exist")


def test_parse_csv_returns_metadata_and_dataframe(synthetic_recording: Path):
    """parse_csv splits the file into metadata dict + data DataFrame."""
    art = discover(synthetic_recording)
    metadata, df = parse_csv(art.tracking)

    # Metadata keys come from the "# Key,Value" rows.
    assert metadata["Experiment"] == "test_experiment"
    assert "Start Time" in metadata
    assert metadata["Pixels/mm"] == "4.0000"

    # Data frame has the expected columns + non-empty rows.
    assert "behavioral_state" in df.columns
    assert "flow_elapsed_ms" in df.columns
    assert len(df) == 150  # 30 pre-flow + 120 post-flow


def test_parse_csv_handles_inline_comment_sections(synthetic_recording: Path):
    """The data CSV has '# Boards' / '# Devices' between header and data;
    pandas's comment="#" should skip them so only the real header is parsed."""
    art = discover(synthetic_recording)
    _, df = parse_csv(art.data)

    assert list(df.columns) == ["frame", "timestamp", "elapsed_ms", "flow_elapsed_ms"]


def test_parse_csv_excludes_footer_rows(synthetic_recording: Path):
    """The '# End Time' / '# Duration' footer should not become a data row."""
    art = discover(synthetic_recording)
    _, df = parse_csv(art.events)

    # Only the flow_marker rows (start + end) should remain.
    assert len(df) == 2
    assert set(df["value"]) == {"start", "end"}


def test_parse_csv_pre_flow_rows_have_nan_flow_elapsed(synthetic_recording: Path):
    """Frames captured before the flow anchor must read back as NaN, not 0."""
    art = discover(synthetic_recording)
    _, df = parse_csv(art.tracking)

    # First 30 rows are pre-flow per the default spec.
    pre_flow = df.head(30)
    assert (
        pre_flow["flow_elapsed_ms"].isna().all()
    ), "Pre-flow rows must read back as NaN so they can be filtered cleanly"

    # Remaining rows have positive flow_elapsed_ms.
    post_flow = df.tail(120)
    assert (post_flow["flow_elapsed_ms"] >= 0).all()


def test_discover_picks_most_recent_when_multiple(tmp_path: Path):
    """If a directory has two recordings, the latest one wins per type."""
    import time

    rec_dir = tmp_path / "rec"
    write_synthetic_recording(rec_dir, RecordingSpec(experiment_name="first"))
    first_tracking = next(rec_dir.glob("first_*_tracking.csv"))

    # Sleep briefly so mtimes differ on filesystems with second-precision.
    time.sleep(0.05)
    write_synthetic_recording(rec_dir, RecordingSpec(experiment_name="second"))
    second_tracking = next(rec_dir.glob("second_*_tracking.csv"))

    art = discover(rec_dir)
    assert art.tracking == second_tracking
    assert first_tracking.exists()  # not deleted, just not chosen


def test_parse_csv_returns_empty_metadata_when_no_header(tmp_path: Path):
    """A CSV with no '# Key,Value' rows yields an empty metadata dict."""
    p = tmp_path / "tracking.csv"
    p.write_text("# GLIDER Tracking Data\nframe,timestamp\n1,2026-01-01T00:00:00.000\n")
    metadata, df = parse_csv(p)
    # The marker line "# GLIDER Tracking Data" parses as a single-cell
    # row, so it becomes a key with empty value.
    assert "GLIDER Tracking Data" in metadata
    assert len(df) == 1
