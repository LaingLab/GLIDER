"""Project manifest: one file that ties a set of sessions + shared
settings together so ``annotate`` / ``train`` don't have to re-specify
directories and flags every run.

The manifest is directory-based: give ``videos_dir`` (+ optional
``poses_dir``) and sessions are auto-discovered by matching stems
(``videos/t1.mp4`` ↔ ``poses/t1.csv``). All paths are resolved relative
to the manifest file's own location, so the file is portable.

This module has no Qt / Typer dependency — the CLI is a thin adapter
that fills command arguments from a :class:`Project`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


class ProjectError(ValueError):
    """Raised when a project manifest is malformed or inconsistent."""


@dataclass
class Project:
    """A parsed, path-resolved project manifest."""

    root: Path  # directory containing the manifest (for relative resolution)
    videos_dir: Path
    poses_dir: Path
    vocab: Path | None = None
    fps: float = 30.0
    window: int = 30
    body_axis: tuple[int, int] = (0, -1)
    holdout: list[str] = field(default_factory=list)  # session stems
    merge: dict[str, list[str]] = field(default_factory=dict)  # target -> members
    # Optional tuning knobs (None = "not set in the manifest").
    classifier: str | None = None
    mirror_augment: bool | None = None
    n_clips: int | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Project:
        import yaml

        path = Path(path)
        try:
            data = yaml.safe_load(path.read_text()) or {}
        except OSError as e:
            raise ProjectError(f"can't read project file {path}: {e}") from e
        if not isinstance(data, dict):
            raise ProjectError(f"{path} must be a YAML mapping at the top level")

        root = path.parent
        if "videos_dir" not in data or not data["videos_dir"]:
            raise ProjectError(f"{path} must set 'videos_dir'")

        def _resolve(p: str | None) -> Path | None:
            if p is None:
                return None
            p = Path(p)
            return p if p.is_absolute() else (root / p).resolve()

        videos_dir = _resolve(str(data["videos_dir"]))
        poses_dir = _resolve(str(data["poses_dir"])) if data.get("poses_dir") else videos_dir
        vocab = _resolve(str(data["vocab"])) if data.get("vocab") else None

        body_axis = _parse_body_axis(data.get("body_axis", [0, -1]))
        merge = _parse_merge(data.get("merge", {}) or {})

        return cls(
            root=root,
            videos_dir=videos_dir,
            poses_dir=poses_dir,
            vocab=vocab,
            fps=float(data.get("fps", 30.0)),
            window=int(data.get("window", 30)),
            body_axis=body_axis,
            holdout=[str(s) for s in (data.get("holdout") or [])],
            merge=merge,
            classifier=data.get("classifier"),
            mirror_augment=data.get("mirror_augment"),
            n_clips=data.get("n_clips"),
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve_sessions(self) -> list[tuple[Path, Path]]:
        """Return ``(video, pose_csv)`` pairs discovered by stem.

        Every video in ``videos_dir`` is paired with the expected
        ``poses_dir/<stem>.csv``. Existence of the pose CSV is the
        caller's concern (train requires it; review tolerates absence).
        """
        if not self.videos_dir.is_dir():
            raise ProjectError(f"videos_dir not found: {self.videos_dir}")
        videos = sorted(p for p in self.videos_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        return [(v, self.poses_dir / f"{v.stem}.csv") for v in videos]

    def resolve_holdout(self) -> list[Path]:
        """Map each holdout stem to its pose CSV. Raises if a stem matches
        no discovered session."""
        by_stem = {v.stem: pose for v, pose in self.resolve_sessions()}
        out: list[Path] = []
        for stem in self.holdout:
            if stem not in by_stem:
                raise ProjectError(
                    f"holdout stem {stem!r} matches no session in "
                    f"{self.videos_dir} (have: {sorted(by_stem)})"
                )
            out.append(by_stem[stem])
        return out

    def merge_specs(self) -> list[str]:
        """Render the ``merge:`` mapping as ``target=m1,m2`` strings — the
        format the CLI's ``--merge`` parser already validates."""
        return [f"{target}={','.join(members)}" for target, members in self.merge.items()]


def _parse_body_axis(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProjectError(f"body_axis must be a 2-element list, got {value!r}")
    return (int(value[0]), int(value[1]))


def _parse_merge(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ProjectError(f"merge must be a mapping target -> [members], got {value!r}")
    out: dict[str, list[str]] = {}
    for target, members in value.items():
        if not isinstance(members, (list, tuple)) or not members:
            raise ProjectError(f"merge target {target!r} must list one or more members")
        out[str(target)] = [str(m) for m in members]
    return out
