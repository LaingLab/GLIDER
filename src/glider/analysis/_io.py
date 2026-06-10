"""
Artifact discovery + CSV parsing for GLIDER recordings.

A recording directory contains one or more of:

  - ``<name>_<ts>.csv``           — DataRecorder output (sensor traces)
  - ``<name>_<ts>_tracking.csv``  — TrackingDataLogger output (per-frame CV)
  - ``<name>_<ts>_events.csv``    — DeviceEventLogger output (pin edges +
                                    output writes + flow_marker rows)
  - ``<name>_<ts>.mp4``           — raw video
  - ``<name>_<ts>_annotated.mp4`` — annotated video (tracking overlays)

Discovery is via the ``# GLIDER ...`` header marker on row 1 of each
CSV rather than filename suffix, so renamed/copied files still classify
correctly. The actual markers are:

  - ``# GLIDER Tracking Data``       → tracking
  - ``# GLIDER Experiment Data``     → data
  - ``# GLIDER Device Event Log``    → events

CSVs that don't start with a recognized marker are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

ArtifactType = Literal["tracking", "data", "events"]

# First-row marker → artifact type. Keep this in lock-step with the
# strings written by event_logger / data_recorder / tracking_logger.
_HEADER_MARKERS: dict[str, ArtifactType] = {
    "# GLIDER Tracking Data": "tracking",
    "# GLIDER Experiment Data": "data",
    "# GLIDER Device Event Log": "events",
}


@dataclass
class Artifacts:
    """Paths to the files comprising a recording. Any may be None if
    that artifact wasn't written for the session."""

    directory: Path
    tracking: Path | None = None
    data: Path | None = None
    events: Path | None = None
    video: Path | None = None
    annotated_video: Path | None = None


def discover(directory: Path) -> Artifacts:
    """Scan a directory and classify each CSV by its header marker.

    Videos are matched by extension and the ``_annotated`` suffix on
    the stem. If multiple files of the same type are present (e.g.,
    two recordings in the same directory), the most recently modified
    one wins — most users will keep a clean directory per recording.

    Args:
        directory: Path to a recording directory.

    Returns:
        Artifacts dataclass with each field populated to the path found
        (or None if absent).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    art = Artifacts(directory=directory)

    # CSV classification by header marker. Track latest mtime per type
    # so the newest recording wins if multiple are present.
    csv_mtimes: dict[ArtifactType, float] = {}
    for csv_path in sorted(directory.glob("*.csv")):
        kind = _classify_csv(csv_path)
        if kind is None:
            continue
        mtime = csv_path.stat().st_mtime
        if kind not in csv_mtimes or mtime > csv_mtimes[kind]:
            csv_mtimes[kind] = mtime
            setattr(art, kind, csv_path)

    # Video discovery by stem suffix.
    for mp4_path in sorted(directory.glob("*.mp4")):
        if mp4_path.stem.endswith("_annotated"):
            if (
                art.annotated_video is None
                or mp4_path.stat().st_mtime > art.annotated_video.stat().st_mtime
            ):
                art.annotated_video = mp4_path
        else:
            if art.video is None or mp4_path.stat().st_mtime > art.video.stat().st_mtime:
                art.video = mp4_path

    return art


def _classify_csv(path: Path) -> ArtifactType | None:
    """Identify a CSV by reading its first non-empty line and matching
    against the known ``# GLIDER ...`` headers. Returns None for files
    that aren't GLIDER outputs (e.g., a user's notes.csv in the same
    directory).
    """
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\r\n")
                if not stripped:
                    continue
                return _HEADER_MARKERS.get(stripped)
    except OSError:
        return None
    return None


def parse_csv(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Parse a GLIDER CSV into (metadata, data).

    The metadata dict is built from the ``# Key, Value`` header rows
    (single-cell rows like ``# Active Subject`` become keys with empty
    string values). The data frame is parsed from the column-header row
    onward, with all ``#``-prefixed rows and blank rows skipped — so
    the same call handles the inline ``# Boards`` / ``# Devices``
    sections in data CSVs and the trailing ``# End Time`` footer.

    Args:
        path: Path to a GLIDER CSV.

    Returns:
        (metadata, dataframe). For an empty CSV the dataframe will have
        zero rows but the columns inferred from the header row.
    """
    metadata = _parse_metadata_header(path)
    df = pd.read_csv(
        path,
        comment="#",
        skip_blank_lines=True,
        # Keep zone_ids, behavioral_state etc. as strings rather than
        # letting pandas guess (it sometimes parses zone IDs as int).
        dtype=(
            {"behavioral_state": "string", "zone_ids": "string"}
            if "tracking" in path.name
            else None
        ),
    )
    return metadata, df


def _parse_metadata_header(path: Path) -> dict[str, str]:
    """Walk the top-of-file comment block and pull out key/value pairs.

    Stops at the first non-``#``, non-blank line — that's the column
    header row, and everything after it is data. Header rows like
    ``# Active Subject`` (single cell) are recorded with an empty
    string value.
    """
    metadata: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            if not stripped.startswith("#"):
                break
            # Split on first comma to separate "# Key" from "Value".
            parts = stripped.split(",", 1)
            key = parts[0].lstrip("#").strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if key:
                metadata[key] = value
    return metadata
