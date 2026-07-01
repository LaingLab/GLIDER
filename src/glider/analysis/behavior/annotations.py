"""Behavior-zone data model + CSV I/O.

A **zone** is one annotated behavior episode: ``(behavior, start_frame,
end_frame)``. The :class:`AnnotationStore` is a collection of zones for
**one video** with CRUD and overlap detection.

CSV schema
----------

One annotations CSV per video, conventionally named
``<video_stem>_annotations.csv``. Columns::

    behavior, start_frame, end_frame, created_at, note

* ``behavior`` — string, must exist in the vocabulary
* ``start_frame`` — int, inclusive
* ``end_frame`` — int, exclusive (Python-range convention)
* ``created_at`` — ISO 8601 timestamp, written at zone creation
* ``note`` — free-text, optional

Frame semantics
---------------

``start_frame`` is the first frame where the behavior is present.
``end_frame`` is the first frame where the behavior is **no longer**
present. So a zone of length 1 (only frame N) is ``(N, N+1)``.

Overlap rule
------------

Two zones of the **same** behavior overlap if their frame ranges
intersect. Two zones of **different** behaviors are allowed to overlap
(a mouse can locomote and rear simultaneously). :meth:`AnnotationStore.add`
raises :class:`OverlapError` when a same-behavior overlap is detected.

This module has no Qt dependency — it's testable in isolation.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class OverlapError(ValueError):
    """Raised when a new zone would overlap an existing one of the same behavior."""


@dataclass(frozen=False)
class BehaviorZone:
    """One annotated behavior episode."""

    behavior: str
    start_frame: int
    end_frame: int
    created_at: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.behavior or not self.behavior.strip():
            raise ValueError("behavior must be a non-empty string")
        if not isinstance(self.start_frame, int) or not isinstance(self.end_frame, int):
            raise TypeError(
                f"frame indices must be ints; got "
                f"{type(self.start_frame).__name__}, {type(self.end_frame).__name__}"
            )
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be >= 0, got {self.start_frame}")
        if self.end_frame <= self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) must be > start_frame "
                f"({self.start_frame}); zones are half-open intervals"
            )
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat(timespec="seconds")

    @property
    def duration_frames(self) -> int:
        return int(self.end_frame - self.start_frame)

    def covers(self, frame: int) -> bool:
        """True if ``frame`` falls inside this zone (start inclusive, end exclusive)."""
        return self.start_frame <= int(frame) < self.end_frame

    def overlaps(self, other: BehaviorZone) -> bool:
        """True iff the two zones have intersecting frame ranges.

        Behavior name is *not* considered here — :class:`AnnotationStore`
        owns the "same-behavior overlap is forbidden, cross-behavior is
        allowed" policy.
        """
        return not (self.end_frame <= other.start_frame or other.end_frame <= self.start_frame)

    def to_row(self) -> dict[str, str]:
        return {
            "behavior": self.behavior,
            "start_frame": str(self.start_frame),
            "end_frame": str(self.end_frame),
            "created_at": self.created_at,
            "note": self.note,
        }

    @classmethod
    def from_row(cls, row: dict[str, str]) -> BehaviorZone:
        return cls(
            behavior=row["behavior"],
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            created_at=row.get("created_at", ""),
            note=row.get("note", ""),
        )


def merge_behavior_zones(
    zones: Iterable[BehaviorZone],
    sources: Iterable[str],
    target: str,
) -> list[BehaviorZone]:
    """Fold ``sources`` into ``target``, unioning overlaps.

    Every zone whose behavior is in ``sources`` is renamed to ``target``.
    Renaming can leave two ``target`` zones overlapping (which the store
    forbids), so strictly-overlapping ``target`` zones are coalesced into
    one (earliest start, latest end). Adjacent/touching zones stay
    separate. Zones of other behaviors are returned unchanged.

    Pure function — does not mutate the input zones or any store.
    """
    src = set(sources) - {target}
    others: list[BehaviorZone] = []
    targets: list[BehaviorZone] = []
    for z in zones:
        behavior = target if z.behavior in src else z.behavior
        clone = BehaviorZone(
            behavior=behavior,
            start_frame=z.start_frame,
            end_frame=z.end_frame,
            created_at=z.created_at,
            note=z.note,
        )
        (targets if behavior == target else others).append(clone)

    targets.sort(key=lambda z: (z.start_frame, z.end_frame))
    unioned: list[BehaviorZone] = []
    for z in targets:
        if unioned and z.start_frame < unioned[-1].end_frame:  # strict overlap
            prev = unioned[-1]
            prev.end_frame = max(prev.end_frame, z.end_frame)
            notes = [n for n in (prev.note, z.note) if n]
            prev.note = "; ".join(dict.fromkeys(notes))
        else:
            unioned.append(z)
    return others + unioned


class AnnotationStore:
    """Collection of behavior zones for a single video.

    Operations
    ----------
    * :meth:`add` — append a zone (rejects same-behavior overlaps)
    * :meth:`remove` — delete a zone by identity
    * :meth:`replace` — overwrite the contents (used by ``load_csv``)
    * :meth:`zones_at_frame` — every zone covering the given frame
    * :meth:`zones_for_behavior` — every zone of a named behavior
    * :meth:`counts_by_behavior` — ``{behavior_name: n_zones}``
    * :meth:`save_csv` / :meth:`load_csv` — disk round-trip
    """

    FIELDNAMES = ("behavior", "start_frame", "end_frame", "created_at", "note")

    def __init__(self, zones: Iterable[BehaviorZone] | None = None):
        self._zones: list[BehaviorZone] = []
        if zones:
            for z in zones:
                self.add(z)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, zone: BehaviorZone) -> None:
        """Append a zone. Raises :class:`OverlapError` on same-behavior overlap.

        Different behaviors may overlap freely — that's a deliberate design
        choice; many lab behaviors aren't mutually exclusive (locomoting
        while rearing, sniffing while grooming).
        """
        for existing in self._zones:
            if existing.behavior == zone.behavior and existing.overlaps(zone):
                raise OverlapError(
                    f"new {zone.behavior!r} zone "
                    f"[{zone.start_frame}, {zone.end_frame}) overlaps with "
                    f"existing zone [{existing.start_frame}, {existing.end_frame})"
                )
        self._zones.append(zone)

    def remove(self, zone: BehaviorZone) -> bool:
        """Remove a zone by identity. Returns True if anything was removed."""
        try:
            self._zones.remove(zone)
            return True
        except ValueError:
            return False

    def replace(self, zones: Iterable[BehaviorZone]) -> None:
        """Atomically replace the contents (used by :meth:`load_csv`)."""
        # Build into a fresh list first so a malformed iterable doesn't
        # leave the store half-populated.
        new = list(zones)
        # Re-run the overlap check against the new list to validate the
        # caller's data.
        seen: list[BehaviorZone] = []
        for z in new:
            for s in seen:
                if s.behavior == z.behavior and s.overlaps(z):
                    raise OverlapError(
                        f"replace() rejected: {z.behavior!r} zones "
                        f"[{s.start_frame},{s.end_frame}) and "
                        f"[{z.start_frame},{z.end_frame}) overlap"
                    )
            seen.append(z)
        self._zones = new

    def clear(self) -> None:
        self._zones.clear()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[BehaviorZone]:
        return iter(self._zones)

    def __len__(self) -> int:
        return len(self._zones)

    def __bool__(self) -> bool:
        return bool(self._zones)

    def zones(self) -> list[BehaviorZone]:
        """Return a copy of all zones in insertion order."""
        return list(self._zones)

    def zones_at_frame(self, frame: int) -> list[BehaviorZone]:
        """Every zone covering ``frame`` (zero or more — cross-behavior overlap allowed)."""
        return [z for z in self._zones if z.covers(frame)]

    def zones_for_behavior(self, behavior: str) -> list[BehaviorZone]:
        return [z for z in self._zones if z.behavior == behavior]

    def counts_by_behavior(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for z in self._zones:
            out[z.behavior] = out.get(z.behavior, 0) + 1
        return out

    def total_frames_by_behavior(self) -> dict[str, int]:
        """Total annotated frames per behavior (sum of zone durations)."""
        out: dict[str, int] = {}
        for z in self._zones:
            out[z.behavior] = out.get(z.behavior, 0) + z.duration_frames
        return out

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save_csv(self, path: str | Path) -> Path:
        """Write the zones to ``path``. Writes the header even when empty."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Sort by start_frame for deterministic output; behavior order within
        # a tie is preserved (Python sort is stable).
        zones = sorted(self._zones, key=lambda z: (z.start_frame, z.end_frame))
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for z in zones:
                writer.writerow(z.to_row())
        return path

    @classmethod
    def load_csv(cls, path: str | Path) -> AnnotationStore:
        """Read a CSV produced by :meth:`save_csv`.

        Missing files yield an empty store (the convention for opening a
        new annotation session on a video that hasn't been touched yet).
        """
        path = Path(path)
        store = cls()
        if not path.exists():
            return store
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            missing = set(cls.FIELDNAMES[:3]) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {sorted(missing)}; "
                    f"found {reader.fieldnames}"
                )
            zones = [BehaviorZone.from_row(row) for row in reader]
        store.replace(zones)
        return store
