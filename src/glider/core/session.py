"""Session - one recording and everything derived from it.

Callers currently find artifacts by rebuilding filenames: a stem here, a suffix
there, a folder convention somewhere else. That is how a pose CSV and its
sidecar came to live in different folders, which cost a cohort its freezing
scores without raising anything - the sidecar carries the resolution, and with
no resolution ``classify`` computes no speed axis at all.

Resolving both from one object is the fix. ``pose_csv`` and ``pose_meta`` are
properties of the same session, so separating them takes deliberate effort
rather than inattention.

This is phase 1 of the project-structure work (see
``docs/superpowers/specs/2026-09-06-project-structure-design.md``): resolution
only. It reads the canonical layout *and* the flat folders that already exist,
because real cohorts cannot be asked to reorganise before they can be read.

An artifact that is not there resolves to ``None``, never to a path that does
not exist. A non-existent path reads as an answer and fails later, somewhere
else, which is the failure mode this module exists to remove.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Session"]

#: Where a session's own files live under the canonical layout.
_SESSIONS_DIR = "sessions"

#: Folders that have held analysis outputs, in preference order. "analysis" is
#: canonical; "final_outputs" is what the reorganised TRH cohort uses.
_ANALYSIS_DIRS = ("analysis", "final_outputs")

#: Immediate subfolders are searched for media as well as the root, because a
#: cohort split by sex keeps videos in males/ and females/ while the analysis
#: outputs sit beside those folders. Bounded to one level: a filename match two
#: levels down is more likely a copy than the thing being looked for.
_MEDIA_SEARCH_DEPTH = 1


def _first_existing(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _spelling_variants(session_id: str) -> list[str]:
    """The ways one session id has been written in existing cohorts.

    ``Test 1`` also appears as ``test1`` (summaries, zones) and ``Test_1``
    (anything that sanitised the space). Recognising them is not an endorsement
    - the canonical layout uses one spelling - but a flat folder full of the
    others still has to be readable.
    """
    variants = [session_id]
    squashed = session_id.replace(" ", "")
    variants += [squashed, squashed.lower(), session_id.replace(" ", "_")]
    return list(dict.fromkeys(variants))


@dataclass(frozen=True)
class Session:
    """One recording, and the artifacts derived from it.

    Args:
        root: The project directory, or - for a folder that has not been
            adopted yet - the folder the recordings sit in.
        session_id: The canonical name for this recording, which is also the
            session folder name under the canonical layout.
    """

    root: Path
    session_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def __repr__(self) -> str:
        return f"Session({self.session_id!r} in {self.root})"

    # ------------------------------------------------------------------
    # folders
    # ------------------------------------------------------------------

    @property
    def folder(self) -> Path:
        """Where this session's files live.

        The canonical ``sessions/<id>/`` when it exists, otherwise the root
        itself - which is what a flat cohort folder looks like.
        """
        canonical = self.root / _SESSIONS_DIR / self.session_id
        if canonical.is_dir():
            return canonical
        beside = self.root / self.session_id
        if beside.is_dir() and (beside / f"{self.session_id}.mp4").exists():
            return beside
        return self.root

    def analysis_candidates(self) -> list[Path]:
        """Folders that might hold this session's analysis outputs.

        Deliberately a list rather than one answer: a folder existing is not
        evidence the outputs are in it, so the file is looked for in each
        rather than a folder being picked and then trusted.
        """
        folders: list[Path] = []
        for base in dict.fromkeys([self.folder, self.root / self.session_id]):
            folders += [base / name for name in _ANALYSIS_DIRS]
            folders.append(base)
        folders.append(self.folder)
        return list(dict.fromkeys(folders))

    @property
    def analysis_dir(self) -> Path | None:
        """Where this session's analysis outputs actually are.

        Derived from finding one, not from a folder merely existing.
        """
        found = self.stats or self.run_manifest or self.ethogram or self.bouts
        return found.parent if found is not None else None

    def _media_folders(self) -> list[Path]:
        """Where a recording or its pose track might sit.

        The session folder and the root first, then the root's immediate
        subfolders - which is where a cohort split by sex keeps its videos.
        """
        folders = [self.folder, self.root]
        if _MEDIA_SEARCH_DEPTH:
            try:
                folders += sorted(p for p in self.root.iterdir() if p.is_dir())
            except OSError:
                pass
        return list(dict.fromkeys(folders))

    # ------------------------------------------------------------------
    # the recording
    # ------------------------------------------------------------------

    @property
    def video(self) -> Path | None:
        for suffix in (".mp4", ".avi", ".mov", ".mkv"):
            found = _first_existing(
                *(folder / f"{self.session_id}{suffix}" for folder in self._media_folders())
            )
            if found is not None:
                return found
        return None

    # ------------------------------------------------------------------
    # pose
    # ------------------------------------------------------------------

    @property
    def pose_csv(self) -> Path | None:
        """The pose track for this session.

        Delegates to :func:`glider.vision.pose.batch.find_pose_csv`, which
        already knows the ``<stem>DLC_<model>`` naming, skips the ``_raw`` and
        ``_annotations`` companions, and prefers the most recent when several
        models have been run.
        """
        from glider.vision.pose.batch import find_pose_csv

        for folder in self._media_folders():
            found = find_pose_csv(self.session_id, folder)
            if found is not None:
                return found
        return None

    @property
    def pose_meta(self) -> Path | None:
        """The sidecar belonging to :attr:`pose_csv`.

        Resolved *from the CSV*, so the two cannot be looked up independently
        and end up pointing at different folders.
        """
        csv = self.pose_csv
        if csv is None:
            return None
        from glider.vision.pose.dlc import meta_path

        sidecar = meta_path(csv)
        return sidecar if sidecar.exists() else None

    @property
    def resolution(self) -> tuple[int, int] | None:
        """Frame size the pose was tracked at, from the sidecar.

        ``None`` when there is no sidecar - which is a real state worth
        surfacing, not a reason to guess. Guessing is what turns a missing
        sidecar into a silently empty speed axis.
        """
        sidecar = self.pose_meta
        if sidecar is None:
            return None
        try:
            data = json.loads(sidecar.read_text())
            width, height = data["resolution"]
            return (int(width), int(height))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------
    # arena and zones
    # ------------------------------------------------------------------

    @property
    def arena(self) -> Path | None:
        """Drawn floor corners, scale and residuals for this session."""
        return self._resolve("_arena.json")

    @property
    def zone(self) -> Path | None:
        """The zone configuration derived from the arena."""
        return self._resolve("_zone.json")

    @property
    def zones_dir(self) -> Path | None:
        """Where ``zone_events.csv`` / ``zone_occupancy.csv`` live."""
        return _first_existing(
            self.folder / "zones",
            self.root / f"{self.session_id}_zones",
        )

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    @property
    def subject(self):
        """The animal recorded here, from the project manifest, or ``None``.

        ``None`` means the manifest does not say - not that the session is
        unlabelled in some recoverable way. Inferring the mapping from the
        recordings themselves is what agreed with the truth on 0 of 15 videos.
        """
        from glider.core.project import Project

        return Project.load(self.root).subject_for(self.session_id)

    @property
    def group(self) -> str | None:
        """This session's treatment arm, or ``None`` if unrecorded.

        A property of the session rather than the subject: in a crossover the
        same mouse is in both arms, on different days.
        """
        from glider.core.project import Project

        return Project.load(self.root).group_for(self.session_id)

    # ------------------------------------------------------------------
    # analysis outputs
    # ------------------------------------------------------------------

    @property
    def ethogram(self) -> Path | None:
        return self._in_analysis("ethogram_raw.csv")

    @property
    def bouts(self) -> Path | None:
        return self._in_analysis("bouts.csv")

    @property
    def stats(self) -> Path | None:
        return self._in_analysis("stats.csv")

    @property
    def run_manifest(self) -> Path | None:
        return self._in_analysis("run.json")

    # ------------------------------------------------------------------

    def _in_analysis(self, name: str) -> Path | None:
        return _first_existing(*(folder / name for folder in self.analysis_candidates()))

    def _resolve(self, suffix: str) -> Path | None:
        """Find ``<session><suffix>``, tolerating the spellings in the wild."""
        candidates: list[Path] = []
        for spelling in _spelling_variants(self.session_id):
            for folder in dict.fromkeys([self.folder, self.root]):
                candidates.append(folder / f"{spelling}{suffix}")
        found = _first_existing(*candidates)
        if found is not None:
            return found
        # A trailing-number id also appears with the prefix dropped entirely -
        # "t15_d2" written as "15_d2". Only tried once the direct spellings
        # fail, so it can never shadow an exact match.
        match = re.match(r"^[A-Za-z]+(\d.*)$", self.session_id)
        if match:
            for folder in dict.fromkeys([self.folder, self.root]):
                found = _first_existing(folder / f"{match.group(1)}{suffix}")
                if found is not None:
                    return found
        return None
