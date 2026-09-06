"""Doctor - report what is wrong with a project before it costs a result.

Every failure that motivated the project-structure work was mechanically
detectable and none of it was detected. A pose sidecar sat in a different
folder from its CSV, so ``classify`` computed no speed axis, so freezing and
darting were never scored for eleven recordings. Nothing raised. It surfaced
weeks later, from an unrelated check that noticed 21,576 empty speed cells.

Phase 4 of the project-structure work (see
``docs/superpowers/specs/2026-09-06-project-structure-design.md``).

Two rules keep this useful:

**Findings are warnings, never errors.** A cohort mid-analysis is legitimately
inconsistent - poses tracked, nothing classified yet - and a checker that
refuses to run on real data is a checker nobody runs.

**Every finding names what it costs.** "No sidecar" is a fact about the disk;
"no sidecar, so freezing and darting will not be scored" is the reason to go
and fix it. A report of the first kind trains operators to scroll past it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from glider.core.project import Project
from glider.core.session import Session

__all__ = ["Finding", "check_session", "doctor", "format_report"]

#: How far one session's pixel scale may sit from the cohort median before it
#: is worth a second look. Deliberately loose: one real cohort spans 17% across
#: its cameras and that spread is physical, not drawing error, so a tighter
#: bound would flag thirty correct arenas and teach the operator to ignore it.
#: A corner clicked on the wrong feature misses by far more than this.
_MAX_SCALE_DEVIATION = 0.25

#: Below this many calibrated sessions a median is not a cohort, it is an
#: opinion, and the outlier check says nothing.
_MIN_SESSIONS_FOR_SCALE_CHECK = 4


@dataclass(frozen=True)
class Finding:
    """One thing worth an operator's attention.

    ``session_id`` is empty for cohort-scoped findings, which do not belong to
    any one recording.
    """

    check: str
    message: str
    session_id: str = ""
    severity: str = "warning"

    def __str__(self) -> str:
        where = f"{self.session_id}: " if self.session_id else ""
        return f"{where}{self.message}"


def _pose_model(pose_csv: Path) -> str:
    """The model name out of a ``<stem>DLC_<model>.csv`` filename.

    Empty when the name does not carry one, which is not an error - a hand-made
    or converted CSV is perfectly usable, it just cannot be checked for
    staleness.
    """
    stem = pose_csv.stem
    marker = "DLC_"
    index = stem.rfind(marker)
    return stem[index + len(marker) :] if index >= 0 else ""


#: Cohort-level calibration files. A project holding one of these is
#: calibrated, even where no session has its own drawn arena.
_CALIBRATION_MASTERS = ("arena_calibration.json", "pose_calibration.json")


def _calibration_master(root: Path) -> Path | None:
    for name in _CALIBRATION_MASTERS:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _arena_document(arena_json: Path) -> dict | None:
    try:
        data = json.loads(arena_json.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _calibration(document: dict):
    """The :class:`ArenaCalibration` inside an arena file, whichever shape it is.

    ``tools/arena_zones.py`` writes a document that *wraps* the calibration
    under ``arena`` and carries its derived numbers alongside; a bare
    calibration is also valid. Reading only the inner shape reported all thirty
    correctly-drawn arenas in one cohort as unreadable, which is worse than not
    checking - it is thirty warnings that are wrong.
    """
    from glider.vision.arena import ArenaCalibration

    for candidate in (document.get("arena"), document):
        if not isinstance(candidate, dict) or "corners" not in candidate:
            continue
        try:
            return ArenaCalibration.from_dict(candidate)
        except Exception:  # noqa: BLE001 - a diagnostic must not fail on bad input
            continue
    return None


def _arena_scale(arena_json: Path) -> float | None:
    """Centre pixels-per-cm from a saved arena, or None if unreadable."""
    document = _arena_document(arena_json)
    if document is None:
        return None
    recorded = document.get("px_per_cm_centre")
    if isinstance(recorded, (int, float)) and recorded > 0:
        return float(recorded)
    calibration = _calibration(document)
    if calibration is None:
        return None
    try:
        return calibration.px_per_cm_centre
    except Exception:  # noqa: BLE001 - a diagnostic must not fail on bad input
        return None


def _arena_suspect(arena_json: Path) -> str | None:
    """A description of why an arena looks mis-drawn, or None if it looks fine."""
    document = _arena_document(arena_json)
    if document is None:
        return "could not be read"

    residuals = document.get("residuals")
    if not isinstance(residuals, dict):
        calibration = _calibration(document)
        if calibration is None:
            return "could not be interpreted"
        try:
            residuals = calibration.residuals()
        except Exception:  # noqa: BLE001 - a diagnostic must not fail on bad input
            return "could not be interpreted"

    if not residuals.get("suspect"):
        return None
    edge = residuals.get("edge_ratio")
    scale = residuals.get("scale_ratio")
    if not isinstance(edge, (int, float)) or not isinstance(scale, (int, float)):
        return "its recorded residuals say so"
    return (
        f"opposite edges differ by {edge:.2f}x and the local scale by "
        f"{scale:.2f}x across the floor"
    )


def check_session(session: Session, project: Project | None = None) -> list[Finding]:
    """Everything detectably wrong with one session.

    Split out from :func:`doctor` so the analysis window can check the session
    in front of the operator without scanning the whole cohort.
    """
    findings: list[Finding] = []
    sid = session.session_id
    record = project.record(sid) if project is not None else None

    def warn(check: str, message: str, severity: str = "warning") -> None:
        findings.append(Finding(check=check, message=message, session_id=sid, severity=severity))

    video = session.video
    if video is None:
        warn("no_video", "no recording found for this session")

    pose_csv = session.pose_csv
    if pose_csv is None:
        if video is not None:
            warn("no_pose", "not tracked yet - no pose CSV", severity="info")
    else:
        # The one that cost a cohort its freezing scores.
        if session.pose_meta is None:
            warn(
                "pose_sidecar_missing",
                f"{pose_csv.name} has no .meta.json beside it, so nothing downstream "
                f"knows the frame size or rate. Without the frame size there is no "
                f"px-to-cm conversion, and freezing and darting are silently not "
                f"scored at all",
            )
        elif session.resolution is None:
            warn(
                "no_resolution",
                f"{session.pose_meta.name} records no resolution, so speed cannot be "
                f"converted to cm/s and freezing and darting will not be scored",
            )

    # Naming drift: the artifact was found, but under a spelling that is not
    # the session id. Harmless today, and the reason one session needed a
    # regex per artifact type to gather.
    for label, path in (("arena", session.arena), ("zone", session.zone)):
        if path is not None and not path.name.startswith(sid):
            warn(
                "naming_drift",
                f"the {label} file is named {path.name}, which does not match the "
                f"session id {sid!r}. Every tool that finds it has to guess the "
                f"spelling",
                severity="info",
            )

    arena = session.arena
    if arena is None:
        # A cohort calibrated from one master is calibrated. Saying otherwise
        # for every session in it - 31 warnings in one real cohort - buries the
        # findings that are real.
        master = _calibration_master(session.root)
        if master is not None:
            warn(
                "no_arena",
                f"no per-session arena; scale comes from {master.name}, so distances "
                f"are in cm but not perspective-corrected",
                severity="info",
            )
        else:
            warn(
                "no_arena",
                "no arena calibration, so distances stay in pixels and cannot be "
                "compared with any other session",
            )
    else:
        why = _arena_suspect(arena)
        if why is not None:
            warn("suspect_arena", f"the drawn arena looks off: {why}")

    # Stale outputs: the analysis beside this session was derived from a pose
    # CSV that is no longer the current one. Mid-analysis on one cohort a
    # train-6 CSV replaced a train-2-4 one and nothing recorded that every
    # number beside it predated the swap.
    manifest = session.run_manifest
    if manifest is not None and pose_csv is not None:
        try:
            run = json.loads(manifest.read_text())
        except (OSError, ValueError):
            run = {}
        recorded = run.get("pose_csv")
        if recorded:
            recorded_model = _pose_model(Path(str(recorded)))
            current_model = _pose_model(pose_csv)
            if recorded_model and current_model and recorded_model != current_model:
                warn(
                    "stale_analysis",
                    f"the outputs here were produced from a {recorded_model} pose track, "
                    f"but the current one is {current_model}. Every number in them "
                    f"predates the swap",
                )

    if project is not None:
        if record is None or not record.subject:
            warn(
                "no_subject",
                "the manifest does not say which animal this is, so it cannot be "
                "pooled by group or paired across days",
            )
        elif record.subject not in project.subjects:
            warn(
                "unknown_subject",
                f"references subject {record.subject!r}, which the manifest does not " f"describe",
            )
        if record is not None and record.subject and not record.group:
            warn(
                "no_group",
                "has a subject but no group, so it will be dropped from any "
                "between-group comparison without saying so",
            )

    return findings


def doctor(project: Project) -> list[Finding]:
    """Everything detectably wrong with a project, session by session.

    Ordered by session so a report reads as a checklist to work through rather
    than a pile sorted by machinery.
    """
    findings: list[Finding] = []
    scales: dict[str, float] = {}

    for session_id in project.session_ids():
        session = project.session(session_id)
        findings += check_session(session, project)
        arena = session.arena
        if arena is not None:
            scale = _arena_scale(arena)
            if scale is not None and scale > 0:
                scales[session_id] = scale

    findings += _scale_outliers(scales)
    findings += _competing_analyses(project)
    return findings


def _run_manifests(root: Path, max_depth: int = 3) -> list[tuple[Path, dict]]:
    """Every ``run.json`` in the project, with what it says produced it.

    Bounded rather than an unbounded walk: a deep tree under a cohort folder is
    somebody's archive, and reporting on it would bury the findings that matter.
    """
    found: list[tuple[Path, dict]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        folder, depth = stack.pop()
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir() and depth < max_depth:
                stack.append((entry, depth + 1))
            elif entry.is_file() and entry.name == "run.json":
                try:
                    data = json.loads(entry.read_text())
                except (OSError, ValueError):
                    continue
                if isinstance(data, dict):
                    found.append((entry, data))
    return found


def _competing_analyses(project: Project) -> list[Finding]:
    """Sessions with more than one set of outputs claiming to be current.

    Found in a real cohort: ``females/Test 1/`` and ``Test 1/`` both held a
    full set of outputs, both derived from ``males/Test 1.mp4``, and they
    disagreed - different bout counts, different stats. Whichever a tool
    happened to find first became the answer. Nothing said the other existed.
    """
    by_video: dict[str, tuple[str, list[Path]]] = {}
    for path, data in _run_manifests(project.root):
        video = data.get("video")
        if not video:
            continue
        name = Path(str(video)).name
        _, folders = by_video.setdefault(name.lower(), (name, []))
        folders.append(path.parent)

    findings: list[Finding] = []
    for _, (video, folders) in sorted(by_video.items()):
        if len(folders) < 2:
            continue
        session_id = Path(video).stem
        where = ", ".join(sorted(_relative(f, project.root) for f in folders))
        # Two sets in folders GLIDER would both resolve is an ambiguity that
        # bites today: whichever is found first silently becomes the answer.
        # A set kept in its own named batch folder is a deliberate alternate,
        # worth knowing about but not worth shouting about - and shouting once
        # per session is how a report gets scrolled past.
        resolvable = set(project.session(session_id).analysis_candidates())
        competing = [f for f in folders if f in resolvable]
        ambiguous = len(competing) > 1
        findings.append(
            Finding(
                check="competing_analyses",
                session_id=session_id,
                severity="warning" if ambiguous else "info",
                message=(
                    f"{len(folders)} sets of outputs were all produced from {video}: "
                    f"{where}. "
                    + (
                        "Whichever a tool finds first becomes the answer, and nothing "
                        "records that the others exist"
                        if ambiguous
                        else "Only one is where GLIDER looks; the others are kept "
                        "aside, so this is a note rather than an ambiguity"
                    )
                ),
            )
        )
    return findings


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _scale_outliers(scales: dict[str, float]) -> list[Finding]:
    """Sessions whose pixel scale disagrees with the rest of the cohort.

    Compared against the median rather than the spread, because a real cohort's
    cameras genuinely differ - one spans 17% and every arena in it is correctly
    drawn. What that band does not contain is a corner clicked on the wrong
    feature, which misses by much more.
    """
    if len(scales) < _MIN_SESSIONS_FOR_SCALE_CHECK:
        return []
    ordered = sorted(scales.values())
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    if median <= 0:
        return []

    findings: list[Finding] = []
    for session_id, scale in sorted(scales.items()):
        deviation = abs(scale - median) / median
        if deviation > _MAX_SCALE_DEVIATION:
            findings.append(
                Finding(
                    check="scale_outlier",
                    session_id=session_id,
                    message=(
                        f"its arena measures {scale:.2f} px/cm against a cohort median "
                        f"of {median:.2f} ({deviation * 100:.0f}% out). That lands "
                        f"directly in speed_cm_s, so identical movement scores "
                        f"differently here than in the rest of the cohort"
                    ),
                )
            )
    return findings


def format_report(findings: list[Finding], *, project: Project | None = None) -> str:
    """A report an operator can read top to bottom.

    Grouped by session, warnings before info, because the point is to be worked
    through - not to be complete.
    """
    if not findings:
        name = project.name or project.root.name if project is not None else "this project"
        return f"{name}: nothing to report."

    by_session: dict[str, list[Finding]] = {}
    for finding in findings:
        by_session.setdefault(finding.session_id, []).append(finding)

    lines: list[str] = []
    warnings = sum(1 for f in findings if f.severity == "warning")
    infos = len(findings) - warnings
    summary = f"{warnings} warning{'s' if warnings != 1 else ''}"
    if infos:
        summary += f", {infos} note{'s' if infos != 1 else ''}"
    lines.append(summary)

    for session_id in sorted(by_session, key=lambda s: (s == "", s)):
        lines.append("")
        lines.append(session_id or "(cohort)")
        entries = sorted(by_session[session_id], key=lambda f: (f.severity != "warning", f.check))
        for finding in entries:
            marker = "!" if finding.severity == "warning" else "-"
            lines.append(f"  {marker} [{finding.check}] {finding.message}")
    return "\n".join(lines)
