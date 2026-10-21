"""Which sessions live under a folder, and which treatment group each is in.

Session Review opens a folder as a cohort. A cohort mixes three kinds of session:
apply-run ethograms (``<output>/<video stem>/ethogram_raw.csv``), live
recordings (a folder of GLIDER tracking/events/data CSVs) and videos, found
by the ``_pups.npz`` the pup pass leaves beside them. An ethogram usually
sits beneath the recording it was scored from, and a pup video inside one,
so each *claims* that recording, and only unclaimed recordings become
sessions of their own -- otherwise one animal would appear twice in every
cohort table.

Qt-free, like the rest of :mod:`glider.analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from glider.analysis._io import discover, raw_video_stem
from glider.analysis.behavior.session_view import (
    _SEARCH_LEVELS,
    VIDEO_SUFFIXES,
    find_session_video,
    is_video,
)

__all__ = [
    "SessionSource",
    "discover_sessions",
    "is_recording",
    "project_groups",
    "recording_candidates",
    "session_id_for",
]


@dataclass(frozen=True)
class SessionSource:
    """One session to load: an ethogram CSV, a recording folder or a bare video."""

    path: Path
    recording: Path | None
    session_id: str
    group: str = ""


def is_recording(folder: Path) -> bool:
    """Whether ``folder`` holds a GLIDER recording (any classified CSV)."""
    try:
        artifacts = discover(Path(folder))
    except OSError:
        return False
    return any(p is not None for p in (artifacts.tracking, artifacts.events, artifacts.data))


def recording_candidates(ethogram_csv: Path, video_path: Path | None = None) -> list[Path]:
    """Where an ethogram's recording might be, nearest first.

    An apply run writes ``<output>/<video stem>/ethogram_raw.csv``, so the
    ethogram's own folder is a level *below* the recording, and the canonical
    layout buries it further under ``sessions/<id>/analysis/``. The video the
    session view found -- via ``run.json`` where there is one -- sits in the
    recording folder itself, so it leads.
    """
    folders = [Path(video_path).parent] if video_path is not None else []
    folder = Path(ethogram_csv).parent
    for _ in range(_SEARCH_LEVELS + 1):
        folders.append(folder)
        if folder == folder.parent:
            break
        folder = folder.parent
    return list(dict.fromkeys(f.resolve() for f in folders))


def session_id_for(path: Path) -> str:
    """The name a session goes by in tables, exports and ``glider_project.json``.

    The first of: the folder directly under a ``sessions/`` directory -- the
    session's own folder or its parent, which is the canonical layout
    (``sessions/<id>/`` and ``sessions/<id>/analysis/``); an ethogram's folder
    name (an apply run names it after the video); a bare video's stem; a
    recording's video stem; the folder name.
    """
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    # Only the nearest level: a lab whose data lives under ~/sessions/ must
    # not have every animal named after its cohort folder.
    for candidate in (folder, folder.parent):
        if candidate.parent.name == "sessions":
            return candidate.name
    if path.is_file() and is_video(path):
        return path.stem
    if not path.is_dir():
        return folder.name
    try:
        artifacts = discover(folder)
    except OSError:
        return folder.name
    video = artifacts.video or artifacts.annotated_video
    if video is not None:
        return raw_video_stem(video)
    return folder.name


def project_groups(root: Path) -> tuple[dict[str, str], str | None]:
    """Session id -> group from ``glider_project.json``, plus a warning if unreadable.

    A bad manifest must never block loading: the cohort still opens, ungrouped.
    """
    from glider.core.project import MANIFEST_NAME, Project, ProjectError

    if not (Path(root) / MANIFEST_NAME).is_file():
        return {}, None
    try:
        project = Project.load(root)
    except ProjectError as e:
        return {}, f"{e}. Sessions are shown ungrouped."
    return {sid: record.group for sid, record in project.sessions.items() if record.group}, None


def discover_sessions(root: Path) -> tuple[list[SessionSource], str | None]:
    """Every session beneath ``root``, grouped, and a warning to show if any."""
    # Imported here: the pups module pulls in cv2 and scipy, which a plain
    # cohort scan of ethograms and recordings has no other use for.
    from glider.vision.pose.pups import PUPS_SUFFIX

    root = Path(root)
    sources: list[SessionSource] = []
    claimed: set[Path] = set()
    for ethogram in sorted(root.rglob("ethogram_raw.csv")):
        candidates = recording_candidates(ethogram, find_session_video(ethogram))
        recording = next((f for f in candidates if is_recording(f)), None)
        if recording is not None:
            claimed.add(recording)
        sources.append(SessionSource(ethogram, recording, session_id_for(ethogram)))
    # A video is a session once the batch pup pass has left its pups beside
    # it: the pups, not the video, are what make it reviewable. One recorded
    # by GLIDER claims its recording, as an ethogram does, and goes by the
    # recording's name -- the one glider_project.json knows it by. An
    # ethogram of the same video is not merged with it: both stay.
    pup_videos: list[SessionSource] = []
    for npz in sorted(root.rglob(f"*{PUPS_SUFFIX}")):
        stem = npz.name[: -len(PUPS_SUFFIX)]
        found = (npz.with_name(stem + suffix) for suffix in sorted(VIDEO_SUFFIXES))
        video = next((v.resolve() for v in found if v.is_file()), None)
        if video is None:
            continue
        candidates = recording_candidates(video, video)
        recording = next((f for f in candidates if is_recording(f)), None)
        if recording is not None:
            claimed.add(recording)
        pup_videos.append(SessionSource(video, recording, session_id_for(recording or video)))
    # Only folders that hold a CSV are classified, so a tree of raw videos is
    # not header-sniffed file by file.
    for folder in sorted({p.parent.resolve() for p in root.rglob("*.csv")}):
        if folder not in claimed and is_recording(folder):
            sources.append(SessionSource(folder, folder, session_id_for(folder)))
    sources += pup_videos
    groups, warning = project_groups(root)
    return [replace(s, group=groups.get(s.session_id, "")) for s in sources], warning
