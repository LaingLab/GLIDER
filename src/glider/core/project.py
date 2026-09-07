"""Project - the directory, its sessions, and everything cohort-scoped.

:class:`~glider.core.session.Session` answers "where is this session's pose
CSV". It cannot answer "which mouse is this, and what were they given", because
nothing on disk records that. Recovering it for one cohort meant correlating
pose tracks against AnyMaze trajectories; an earlier attempt to infer it from
distance rank agreed with the truth on 0 of 15 videos and would have
mislabelled the whole experiment. That is what the manifest is for.

Phase 3 of the project-structure work (see
``docs/superpowers/specs/2026-09-06-project-structure-design.md``).

Two properties shape the design:

**Subjects outlive sessions.** One mouse is recorded on day 1 and day 2, so
subject identity lives in ``subjects`` and sessions reference it. Duplicating a
mouse's strain into each session is how two records of one animal come to
disagree.

**Treatment does not.** In the counterbalanced crossover this was written for,
the same mouse is saline on one day and drug on the other, so ``group`` and
``treatment`` belong to the *session*. Hanging them off the subject would make
a crossover unrepresentable, and - worse - representable incorrectly.

A manifest is optional throughout. A folder with no ``glider_project.json``
still loads, with every session anonymous. Refusing to read unlabelled data
would make the tool useless for exactly the cohorts that need it most.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from glider.core.experiment_session import Subject
from glider.core.session import Session

__all__ = [
    "MANIFEST_NAME",
    "PROJECT_SCHEMA_VERSION",
    "Project",
    "ProjectError",
    "Provenance",
    "SessionRecord",
]

#: The manifest filename. Its presence is what makes a folder a project.
MANIFEST_NAME = "glider_project.json"

PROJECT_SCHEMA_VERSION = 1

#: Fields of :class:`Subject` that describe the animal rather than what was
#: done to it on a given day. The rest (``solution``, ``dose``, ``route``,
#: ``concentration``, ``group``) move to the session, because in a crossover
#: they change between one subject's own recordings.
_SUBJECT_IDENTITY_FIELDS = (
    "subject_id",
    "name",
    "age",
    "sex",
    "weight",
    "strain",
    "notes",
)

#: Treatment keys carried per session.
_TREATMENT_FIELDS = ("solution", "concentration", "dose", "route")

#: Recording extensions, matching what Session will resolve.
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}

#: Subfolders that hold outputs rather than recordings. Scanning them for
#: videos would turn an annotated copy into a session of its own.
_NON_MEDIA_DIRS = {
    "analysis",
    "calibration",
    "final_outputs",
    "heatmap",
    "sessions",
    "superseded",
    "zones",
}


class ProjectError(ValueError):
    """Raised when a manifest is malformed in a way that cannot be ignored."""


@dataclass(frozen=True)
class Provenance:
    """What produced a session's current analysis outputs.

    Exists so a stale re-run is detectable. Mid-analysis on one cohort a
    ``train-6`` pose CSV replaced a ``train-2-4`` one and nothing recorded that
    the ethograms beside it predated the swap; every number in them was derived
    under a model that was no longer there.
    """

    pose_model: str = ""
    classifier: str = ""
    thresholds: dict[str, Any] = field(default_factory=dict)
    analysed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.pose_model:
            data["pose_model"] = self.pose_model
        if self.classifier:
            data["classifier"] = self.classifier
        if self.thresholds:
            data["thresholds"] = dict(self.thresholds)
        if self.analysed_at:
            data["analysed_at"] = self.analysed_at
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Provenance:
        data = data or {}
        thresholds = data.get("thresholds")
        return cls(
            pose_model=str(data.get("pose_model", "") or ""),
            classifier=str(data.get("classifier", "") or ""),
            thresholds=dict(thresholds) if isinstance(thresholds, dict) else {},
            analysed_at=str(data.get("analysed_at", "") or ""),
        )

    def __bool__(self) -> bool:
        return bool(self.to_dict())


@dataclass(frozen=True)
class SessionRecord:
    """What the manifest knows about one recording.

    ``subject`` is a key into :attr:`Project.subjects`, not an embedded copy,
    so a mouse recorded twice has one description rather than two that can
    drift apart.
    """

    session_id: str
    subject: str = ""
    group: str = ""
    treatment: dict[str, str] = field(default_factory=dict)
    day: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.subject:
            data["subject"] = self.subject
        if self.group:
            data["group"] = self.group
        if self.treatment:
            data["treatment"] = dict(self.treatment)
        if self.day:
            data["day"] = self.day
        if self.provenance:
            data["provenance"] = self.provenance.to_dict()
        if self.notes:
            data["notes"] = self.notes
        return data

    @classmethod
    def from_dict(cls, session_id: str, data: dict[str, Any] | None) -> SessionRecord:
        data = data or {}
        treatment = data.get("treatment")
        return cls(
            session_id=session_id,
            subject=str(data.get("subject", "") or ""),
            group=str(data.get("group", "") or ""),
            treatment=(
                {str(k): str(v) for k, v in treatment.items()}
                if isinstance(treatment, dict)
                else {}
            ),
            day=str(data.get("day", "") or ""),
            provenance=Provenance.from_dict(data.get("provenance")),
            notes=str(data.get("notes", "") or ""),
        )


def _subject_identity(subject: Subject) -> dict[str, Any]:
    """The animal, without the day's treatment hung off it."""
    full = subject.to_dict()
    return {k: full[k] for k in _SUBJECT_IDENTITY_FIELDS if full.get(k)}


@dataclass
class Project:
    """A folder of sessions, plus what the manifest records about them."""

    root: Path
    name: str = ""
    sessions: dict[str, SessionRecord] = field(default_factory=dict)
    subjects: dict[str, Subject] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def __repr__(self) -> str:
        return f"Project({self.name or self.root.name!r}, {len(self.sessions)} sessions)"

    # ------------------------------------------------------------------
    # loading and saving
    # ------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @classmethod
    def load(cls, root: str | Path) -> Project:
        """Read the project at *root*.

        A folder with no manifest is a project with no manifest, not an error:
        its sessions are discovered from disk and every one of them is
        anonymous. That is the state every existing cohort is in.
        """
        root = Path(root)
        path = root / MANIFEST_NAME
        try:
            raw = path.read_text()
        except OSError:
            return cls(root=root)
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise ProjectError(f"{path} is not valid JSON: {e}") from e
        if not isinstance(data, dict):
            raise ProjectError(f"{path} must hold a JSON object at the top level")

        version = data.get("schema_version", PROJECT_SCHEMA_VERSION)
        if isinstance(version, int) and version > PROJECT_SCHEMA_VERSION:
            raise ProjectError(
                f"{path} was written by a newer GLIDER (schema {version}, this one "
                f"understands {PROJECT_SCHEMA_VERSION}). Reading it would silently "
                f"drop whatever that version added."
            )

        subjects: dict[str, Subject] = {}
        for key, value in (data.get("subjects") or {}).items():
            if not isinstance(value, dict):
                continue
            payload = dict(value)
            payload.setdefault("subject_id", key)
            subjects[str(key)] = Subject.from_dict(payload)

        sessions: dict[str, SessionRecord] = {}
        for key, value in (data.get("sessions") or {}).items():
            sessions[str(key)] = SessionRecord.from_dict(str(key), value)

        return cls(
            root=root,
            name=str(data.get("name", "") or ""),
            sessions=sessions,
            subjects=subjects,
            created_at=str(data.get("created_at", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROJECT_SCHEMA_VERSION,
            "name": self.name,
            "created_at": self.created_at or datetime.now().isoformat(timespec="seconds"),
            "subjects": {key: _subject_identity(s) for key, s in sorted(self.subjects.items())},
            "sessions": {key: record.to_dict() for key, record in sorted(self.sessions.items())},
        }

    def save(self) -> Path:
        """Write the manifest. Returns the path written.

        Writes to a temporary file and renames, because a manifest truncated by
        an interrupted write is worse than one that is a version out of date:
        it takes the subject mapping with it, and that mapping cannot be
        recovered from the recordings.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        self.created_at = payload["created_at"]
        target = self.manifest_path
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(target)
        return target

    # ------------------------------------------------------------------
    # sessions
    # ------------------------------------------------------------------

    def session(self, session_id: str) -> Session:
        """The :class:`Session` for *session_id*, listed or not.

        Deliberately does not refuse an unlisted id. A recording that arrived
        after the manifest was written is a normal state, and a project that
        would not resolve it would send the operator straight back to building
        paths by hand.
        """
        return Session(self.root, session_id)

    def session_ids(self) -> list[str]:
        """Every session in this project: manifest and disk, unioned.

        Manifest-only ids are kept because a session whose files have not
        arrived yet is exactly what doctor should be reporting, not hiding.
        """
        found = set(self.sessions)
        found |= set(self.discover_session_ids())
        return sorted(found)

    def discover_session_ids(self) -> list[str]:
        """Session ids visible on disk, from the canonical layout or videos.

        Discovery is by *recording*, not by any file that happens to carry a
        session-shaped name. A stray ``Test 3_zone.json`` beside no video is a
        loose file, and inventing a session for it would put an empty row in
        every cohort table.
        """
        found: set[str] = set()
        sessions_dir = self.root / "sessions"
        if sessions_dir.is_dir():
            found |= {p.name for p in sessions_dir.iterdir() if p.is_dir()}
        for folder in self._video_folders():
            try:
                entries = list(folder.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_file() and entry.suffix.lower() in _VIDEO_SUFFIXES:
                    found.add(entry.stem)
        return sorted(found)

    def _video_folders(self) -> list[Path]:
        """The root and its immediate subfolders, mirroring Session lookup.

        A cohort split by sex keeps recordings in ``males/`` and ``females/``;
        those are the same sessions, not a different project.
        """
        folders = [self.root]
        try:
            folders += sorted(
                p for p in self.root.iterdir() if p.is_dir() and p.name not in _NON_MEDIA_DIRS
            )
        except OSError:
            pass
        return folders

    # ------------------------------------------------------------------
    # subjects and groups
    # ------------------------------------------------------------------

    def record(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    def subject_for(self, session_id: str) -> Subject | None:
        """The animal recorded in *session_id*, or None if unrecorded.

        None means "the manifest does not say", which is a state worth
        surfacing rather than papering over. Guessing the mapping is precisely
        the move that scored 0 of 15.
        """
        record = self.sessions.get(session_id)
        if record is None or not record.subject:
            return None
        return self.subjects.get(record.subject)

    def group_for(self, session_id: str) -> str | None:
        """This session's treatment arm, or None if unrecorded.

        Read from the session, not the subject: in a crossover one mouse is in
        both arms, on different days.
        """
        record = self.sessions.get(session_id)
        if record is None or not record.group:
            return None
        return record.group

    def sessions_in_group(self, group: str) -> list[str]:
        return sorted(k for k, r in self.sessions.items() if r.group == group)

    def groups(self) -> dict[str, list[str]]:
        """Group name -> session ids, for every group that names one."""
        out: dict[str, list[str]] = {}
        for key, record in sorted(self.sessions.items()):
            if record.group:
                out.setdefault(record.group, []).append(key)
        return out

    def sessions_for_subject(self, subject_id: str) -> list[str]:
        """Every recording of one animal, which is what makes a crossover
        analysable: the day-1 and day-2 rows have to be pairable."""
        return sorted(k for k, r in self.sessions.items() if r.subject == subject_id)

    def set_session(
        self,
        session_id: str,
        *,
        subject: str | Subject | None = None,
        group: str | None = None,
        treatment: dict[str, str] | None = None,
        day: str | None = None,
        provenance: Provenance | None = None,
        notes: str | None = None,
    ) -> SessionRecord:
        """Record what is known about a session, leaving the rest untouched.

        Passing a :class:`Subject` registers it under its ``subject_id`` and
        references it, so callers never have to keep the two dicts in step by
        hand -- which is the kind of bookkeeping that produces two subjects for
        one mouse.
        """
        existing = self.sessions.get(session_id) or SessionRecord(session_id=session_id)

        subject_key = existing.subject
        if isinstance(subject, Subject):
            subject_key = subject.subject_id or subject.id
            self.subjects[subject_key] = subject
            # A crossover's treatment varies per session, so anything the
            # Subject carries about the day's drug is taken as this session's
            # unless the caller said otherwise.
            if treatment is None:
                carried = {
                    f: getattr(subject, f) for f in _TREATMENT_FIELDS if getattr(subject, f, "")
                }
                treatment = carried or None
            if group is None and subject.group:
                group = subject.group
        elif subject is not None:
            subject_key = subject

        record = SessionRecord(
            session_id=session_id,
            subject=subject_key,
            group=existing.group if group is None else group,
            treatment=existing.treatment if treatment is None else dict(treatment),
            day=existing.day if day is None else day,
            provenance=existing.provenance if provenance is None else provenance,
            notes=existing.notes if notes is None else notes,
        )
        self.sessions[session_id] = record
        return record
