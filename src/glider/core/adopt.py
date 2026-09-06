"""Adopt - move an existing folder into the canonical layout, safely.

Existing cohorts are flat, inconsistently named, and are real data. Adoption
plans the whole move first, shows it, and only then touches the disk.

Phase 4 of the project-structure work (see
``docs/superpowers/specs/2026-09-06-project-structure-design.md``). The
constraints below were each learned from doing this by hand:

**Verify the whole plan before moving one file.** A collision found halfway
through leaves a folder that is neither shape, and the operator now has to
work out which half moved.

**Use :func:`os.rename`, never :func:`shutil.move`.** On a locked file - a
video still open in a player, which is routine - ``shutil.move`` falls back to
copy-then-delete and leaves the data in both places. That happened, and the
copies were byte-identical, so nothing but timestamps said which was real.
``os.rename`` fails having changed nothing.

**Be resumable.** A move whose source is gone and whose destination is present
is already done, not an error. These folders live on network shares and the
share will drop mid-run.

**Never overwrite.** A destination that already holds something different is a
collision, and the plan refuses. Superseded artifacts are moved aside, not
deleted, so a result computed against them stays reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from glider.core.project import _VIDEO_SUFFIXES, Project
from glider.core.session import _spelling_variants

__all__ = [
    "AdoptPlan",
    "AdoptResult",
    "Move",
    "apply_plan",
    "plan_adopt",
    "revert",
]

#: Where the reversal manifest is written, relative to the project root.
REVERSAL_NAME = "adopt_reversal.json"

#: Files that belong to the whole cohort, not to any one session. They stay at
#: the root; the point of the layout is that this distinction is visible.
_COHORT_FILES = {
    "arena_calibration.json",
    "cohort_speed.json",
    "glider_project.json",
    "pose_calibration.json",
}

#: Suffix -> subfolder of the session directory. Anything not listed lands in
#: the session folder itself.
_SUBFOLDER_FOR_SUFFIX = {
    "_heatmap": "heatmap",
    "_zones": "zones",
}

#: Names that mark a file as an analysis output wherever it is found.
_ANALYSIS_NAMES = {
    "bouts.csv",
    "ethogram_raw.csv",
    "run.json",
    "stats.csv",
    "transitions.csv",
}

#: Folder names that hold a session's *own* current outputs, as opposed to one
#: batch run among several.
_OWN_ANALYSIS_DIRS = ("analysis", "final_outputs")

#: Where a batch's files are kept, under the session folder. One subfolder
#: per batch, so alternate derivations stay whole and stay apart.
_RUNS_DIR = "runs"

#: Artifacts GLIDER finds by name, and so the only ones worth renaming onto the
#: canonical session id. Everything else keeps the name it has: extracted
#: frames are indexed by relative path inside a labelling project, and a rename
#: that is merely tidier is not worth breaking somebody's DLC dataset for.
_CANONICAL_SUFFIXES = ("_arena.json", "_zone.json", "_summary.csv")


@dataclass(frozen=True)
class Move:
    """One rename. ``source`` and ``destination`` are both absolute."""

    source: Path
    destination: Path
    session_id: str = ""
    why: str = ""

    @property
    def is_rename(self) -> bool:
        """True when the file is also being renamed, not just relocated.

        Worth showing separately in a plan: relocating a file is reversible in
        the operator's head, renaming it is the part they will want to read.
        """
        return self.source.name != self.destination.name


@dataclass
class AdoptPlan:
    """A complete source-to-destination mapping. Nothing has moved yet."""

    root: Path
    moves: list[Move] = field(default_factory=list)
    unclassified: list[Path] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Whether this plan may be applied.

        A plan with any problem is refused whole. Applying the safe part of an
        unsafe plan is how a folder ends up in two shapes at once.
        """
        return not self.problems

    def describe(self) -> str:
        lines = [f"{len(self.moves)} file{'s' if len(self.moves) != 1 else ''} to move"]
        by_session: dict[str, list[Move]] = {}
        for move in self.moves:
            by_session.setdefault(move.session_id, []).append(move)
        for session_id in sorted(by_session):
            lines.append("")
            lines.append(session_id or "(cohort)")
            for move in sorted(by_session[session_id], key=lambda m: m.source.name):
                rel_src = _relative(move.source, self.root)
                rel_dst = _relative(move.destination, self.root)
                lines.append(f"  {rel_src}  ->  {rel_dst}")
        if self.unclassified:
            lines.append("")
            lines.append(f"{len(self.unclassified)} file(s) left where they are:")
            for path in sorted(self.unclassified):
                lines.append(f"  {_relative(path, self.root)}")
        if self.problems:
            lines.append("")
            lines.append("REFUSED - nothing will move:")
            for problem in self.problems:
                lines.append(f"  ! {problem}")
        return "\n".join(lines)


@dataclass
class AdoptResult:
    """What actually happened."""

    moved: list[Move] = field(default_factory=list)
    already_done: list[Move] = field(default_factory=list)
    reversal_path: Path | None = None
    failed: Move | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.failed is None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _session_for(name: str, session_ids: list[str]) -> tuple[str, str] | None:
    """Which session a filename belongs to, and what follows its id.

    Returns ``(session_id, remainder)``, or None when nothing matches.

    The remainder check is what keeps ``Test 15_zone.json`` away from
    ``Test 1``: a leftover starting with a digit means the id was a prefix of a
    longer one, not the whole of it. Getting that wrong would file half a
    cohort under the wrong animal.
    """
    best: tuple[str, str] | None = None
    best_matched = -1
    for session_id in session_ids:
        for spelling in _spelling_variants(session_id):
            if not name.lower().startswith(spelling.lower()):
                continue
            remainder = name[len(spelling) :]
            if remainder[:1].isdigit():
                continue
            # Longest matched prefix wins, so a file belongs to the most
            # specific session id that can claim it.
            if len(spelling) > best_matched:
                best, best_matched = (session_id, remainder), len(spelling)
    return best


def _session_folder(path: Path, root: Path, session_ids: list[str]) -> Path | None:
    """The session-named folder *path* sits in, if any.

    Only folders strictly below the root are considered - the root itself is
    named for the cohort, and letting it match would sweep every loose file
    into one arbitrary session.
    """
    for parent in path.parents:
        if parent == root:
            return None
        try:
            parent.relative_to(root)
        except ValueError:
            return None
        if parent.name in _OWN_ANALYSIS_DIRS:
            continue
        match = _session_for(parent.name, session_ids)
        if match is not None and not match[1]:
            return parent
    return None


def _media_folders(root: Path) -> set[str]:
    """Top-level folders that hold recordings, by name.

    A cohort split by sex keeps its videos in ``males/`` and ``females/``.
    Those hold the canonical files for their sessions, so what is in them is
    not an alternate derivation - it is the data.
    """
    found: set[str] = set()
    try:
        entries = list(root.iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.is_dir():
            continue
        try:
            if any(p.suffix.lower() in _VIDEO_SUFFIXES for p in entry.iterdir() if p.is_file()):
                found.add(entry.name)
        except OSError:
            continue
    return found


def _batch_name(path: Path, root: Path, session_ids: list[str], media: set[str]) -> str:
    """The batch folder *path* came from, or ``""`` when it is canonical.

    Real cohorts keep several derivations of the same sessions side by side -
    ``rescored_filtered/``, ``output_2_to_7v3/``, ``jump_zones/`` - deliberately,
    so a published number stays reproducible against the files it came from.
    Collapsing them would not be a tidier layout; it would be several answers
    to one question with all but one destroyed. Worse, adoption would refuse
    the whole cohort over collisions between files that were never meant to be
    the same file: two of these folders alone accounted for 62 refusals.

    Canonical means: at the root, in a media folder, or in the session's own
    folder. Anything else is a batch, named for the folder it sits in.
    """
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    if len(relative.parts) < 2:
        return ""
    top = relative.parts[0]
    if top in media:
        return ""
    match = _session_for(top, session_ids)
    if match is not None and not match[1]:
        return ""  # the session's own folder
    return top


def _destination(
    path: Path, root: Path, session_ids: list[str], media: set[str]
) -> tuple[Path, str, str] | None:
    """Where *path* belongs, or None to leave it alone.

    Returns ``(destination, session_id, why)``.
    """
    name = path.name
    if name in _COHORT_FILES or name == REVERSAL_NAME:
        return None

    suffix = path.suffix.lower()

    # Attribute by filename first, then by the folder the file sits in.
    # Analysis outputs are named for what they are (``stats.csv``), never for
    # the session, so ``Test 1/final_outputs/stats.csv`` is only reachable
    # through its folder. Getting this wrong leaves every output unclassified,
    # which is the state that made a cohort take five folders to read.
    named = _session_for(path.stem, session_ids)
    session_folder = _session_folder(path, root, session_ids)
    if named is not None:
        session_id, remainder = named
    elif session_folder is not None:
        session_id, remainder = session_folder.name, ""
    else:
        return None
    session_dir = root / "sessions" / session_id

    # A batch's files are that batch's, kept whole and kept apart. Merging them
    # into the session's own folder is what turned two alternate derivations
    # into a collision.
    batch = _batch_name(path, root, session_ids, media)
    if batch:
        return session_dir / _RUNS_DIR / batch / name, session_id, f"batch {batch}"

    if name in _ANALYSIS_NAMES or path.parent.name in _OWN_ANALYSIS_DIRS:
        return session_dir / "analysis" / name, session_id, "analysis output"

    if named is None:
        # Named for neither its session nor a known output: keep the name, but
        # file it under the session whose folder it was already in.
        return session_dir / name, session_id, "session artifact"

    if suffix in _VIDEO_SUFFIXES and not remainder:
        return session_dir / f"{session_id}{suffix}", session_id, "recording"

    for marker, folder in _SUBFOLDER_FOR_SUFFIX.items():
        if marker in remainder.lower():
            return session_dir / folder / name, session_id, folder

    # Canonicalise the spelling only for the artifacts GLIDER resolves by name.
    # The pose CSV and its sidecar share a stem, so they are treated
    # identically and cannot land in different folders - which is the whole
    # point - but neither is renamed, because a labelling project indexes its
    # frames by path and a tidier name is not worth breaking that.
    if f"{remainder}{path.suffix}".lower() in _CANONICAL_SUFFIXES:
        canonical = f"{session_id}{remainder}{path.suffix}"
        return session_dir / canonical, session_id, "session artifact"
    return session_dir / name, session_id, "session artifact"


def plan_adopt(root: str | Path, *, session_ids: list[str] | None = None) -> AdoptPlan:
    """Work out how to move *root* into the canonical layout.

    Reads only. The returned plan is complete: if it says a file moves, the
    checks that could refuse it have already run.
    """
    root = Path(root)
    plan = AdoptPlan(root=root)
    if not root.is_dir():
        plan.problems.append(f"{root} is not a folder")
        return plan

    ids = list(session_ids) if session_ids is not None else Project.load(root).session_ids()
    if not ids:
        plan.problems.append(f"no sessions found in {root} - nothing to adopt")
        return plan
    # Longest first, so "Test 15" is considered before "Test 1".
    ids.sort(key=len, reverse=True)

    media = _media_folders(root)
    destinations: dict[Path, Move] = {}
    for path in _walk(root):
        result = _destination(path, root, ids, media)
        if result is None:
            plan.unclassified.append(path)
            continue
        destination, session_id, why = result
        if destination == path:
            continue  # already where it belongs

        try:
            destination.resolve().relative_to(root.resolve())
        except ValueError:
            plan.problems.append(
                f"{_relative(path, root)} would move outside the project, to {destination}"
            )
            continue

        move = Move(source=path, destination=destination, session_id=session_id, why=why)
        clash = destinations.get(destination)
        if clash is not None:
            plan.problems.append(
                f"{_relative(path, root)} and {_relative(clash.source, root)} would both "
                f"become {_relative(destination, root)}"
            )
            continue
        if destination.exists():
            plan.problems.append(
                f"{_relative(destination, root)} already exists, so moving "
                f"{_relative(path, root)} onto it would destroy it"
            )
            continue
        destinations[destination] = move
        plan.moves.append(move)

    plan.moves.sort(key=lambda m: (m.session_id, m.source.name))
    return plan


def _walk(root: Path) -> list[Path]:
    """Files worth classifying: the root, its subfolders, and their analysis
    folders. Not an unbounded walk - a deep tree under a cohort folder is
    somebody's archive, and adopting it would be a surprise."""
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        folder, depth = stack.pop()
        if folder.name == "sessions" and folder.parent == root:
            continue  # already canonical
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if depth < 2 and entry.name not in {"superseded"}:
                    stack.append((entry, depth + 1))
            elif entry.is_file():
                found.append(entry)
    return found


def apply_plan(plan: AdoptPlan, *, write_reversal: bool = True) -> AdoptResult:
    """Carry out *plan*, stopping at the first failure.

    Refuses an unsafe plan outright. Writes the reversal manifest *before*
    moving anything, so an interrupted run is still undoable.
    """
    result = AdoptResult()
    if not plan.is_safe:
        result.error = "the plan was refused; nothing moved"
        result.failed = plan.moves[0] if plan.moves else None
        return result

    if write_reversal and plan.moves:
        result.reversal_path = _write_reversal(plan)

    for move in plan.moves:
        # Resumable: a move already done is not a failure. Checked before
        # anything else, because after an interrupted run this is the common
        # case, not the exception.
        if not move.source.exists() and move.destination.exists():
            result.already_done.append(move)
            continue
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            # os.rename, never shutil.move: on a locked file the latter falls
            # back to copy-then-delete and leaves the data in two places.
            os.rename(move.source, move.destination)
        except OSError as e:
            result.failed = move
            result.error = f"could not move {move.source.name}: {e}"
            return result
        result.moved.append(move)
    return result


def _write_reversal(plan: AdoptPlan) -> Path:
    path = plan.root / REVERSAL_NAME
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "written_at": datetime.now().isoformat(timespec="seconds"),
                "root": str(plan.root),
                "moves": [{"from": str(m.source), "to": str(m.destination)} for m in plan.moves],
            },
            indent=2,
        )
        + "\n"
    )
    return path


def revert(reversal_path: str | Path) -> AdoptResult:
    """Replay a reversal manifest backwards.

    Same rules as forward: already-reverted entries are skipped, and a
    destination that has been recreated in the meantime stops the run rather
    than being overwritten.
    """
    reversal_path = Path(reversal_path)
    result = AdoptResult()
    try:
        data = json.loads(reversal_path.read_text())
    except (OSError, ValueError) as e:
        result.error = f"could not read {reversal_path}: {e}"
        result.failed = Move(source=reversal_path, destination=reversal_path)
        return result

    for entry in reversed(data.get("moves", [])):
        source = Path(entry["to"])
        destination = Path(entry["from"])
        move = Move(source=source, destination=destination)
        if not source.exists() and destination.exists():
            result.already_done.append(move)
            continue
        if destination.exists():
            result.failed = move
            result.error = (
                f"{destination} exists again, so putting {source.name} back would " f"destroy it"
            )
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(source, destination)
        except OSError as e:
            result.failed = move
            result.error = f"could not put {source.name} back: {e}"
            return result
        result.moved.append(move)
    return result


def supersede(path: str | Path, root: str | Path) -> Path | None:
    """Move an artifact aside rather than deleting it.

    A result computed against a superseded file stays reproducible only while
    the file exists, so nothing here deletes.
    """
    path = Path(path)
    root = Path(root)
    if not path.exists():
        return None
    folder = root / "superseded" / datetime.now().strftime("%Y-%m-%d")
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / path.name
    counter = 1
    while destination.exists():
        destination = folder / f"{path.stem}.{counter}{path.suffix}"
        counter += 1
    os.rename(path, destination)
    return destination
