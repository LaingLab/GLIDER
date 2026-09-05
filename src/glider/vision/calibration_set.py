"""Per-video pixel-to-distance calibration for a batch, plus its master file.

Composes :class:`glider.vision.calibration.CameraCalibration` — one per video —
so scale values are always derived from the operator's drawn lines and never
stored as an independent, driftable number.

Qt-free on purpose: the batch GUI drives this, but a script or notebook can
build and save a master file with no Qt import.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from glider.vision.arena import ArenaCalibration, DegenerateArenaError
from glider.vision.calibration import CameraCalibration

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "CalibrationSet", "CalibrationSetError"]


class CalibrationSetError(ValueError):
    """A master calibration file could not be understood."""


@dataclass
class CalibrationSet:
    """Calibrations for the videos of one batch, keyed by resolved video path.

    A video can carry a drawn *line* (:class:`CameraCalibration`), a drawn
    *arena* (:class:`~glider.vision.arena.ArenaCalibration`), or both. They are
    kept in separate maps because they answer different questions - only the
    arena knows where the floor is, so only it can place a zone - but where
    both exist the arena sets the scale. See :meth:`px_per_mm`.
    """

    entries: dict[Path, CameraCalibration] = field(default_factory=dict)
    arenas: dict[Path, ArenaCalibration] = field(default_factory=dict)

    #: Arenas stamped by a copy and not yet checked against their own video.
    #: A copied arena that does not fit shows no residual warning -- residuals
    #: are computed from the corners alone -- so it must not satisfy the Run
    #: gate until an operator has seen the overlay on that video's floor.
    _unconfirmed: set[Path] = field(default_factory=set)

    # ------------------------------------------------------------------
    # map
    # ------------------------------------------------------------------

    @staticmethod
    def _key(video: Path | str) -> Path:
        """Resolved path, so two spellings of one file share an entry.

        Falls back to the unresolved path when the file is absent or the OS
        refuses to resolve it — a calibration must survive its video going
        temporarily offline.
        """
        path = Path(video)
        try:
            return path.resolve()
        except (OSError, ValueError):
            return path

    def set(self, video: Path | str, calibration: CameraCalibration) -> None:
        self.entries[self._key(video)] = calibration

    def get(self, video: Path | str) -> CameraCalibration | None:
        return self.entries.get(self._key(video))

    def discard(self, video: Path | str) -> None:
        self.entries.pop(self._key(video), None)

    def set_arena(
        self, video: Path | str, arena: ArenaCalibration, *, confirmed: bool = True
    ) -> None:
        key = self._key(video)
        self.arenas[key] = arena
        if confirmed:
            self._unconfirmed.discard(key)
        else:
            self._unconfirmed.add(key)

    def get_arena(self, video: Path | str) -> ArenaCalibration | None:
        return self.arenas.get(self._key(video))

    def is_arena_confirmed(self, video: Path | str) -> bool:
        return self._key(video) not in self._unconfirmed

    def discard_arena(self, video: Path | str) -> None:
        key = self._key(video)
        self.arenas.pop(key, None)
        self._unconfirmed.discard(key)

    def videos(self) -> list[Path]:
        """Every video this set knows anything about, line or arena."""
        return sorted(set(self.entries) | set(self.arenas))

    def subset(self, videos: Iterable[Path | str]) -> CalibrationSet:
        """A new set holding only the entries for *videos*.

        Lets a caller describe exactly one batch without having to reason about
        which unrelated videos happen to be in this set from earlier work.
        Videos with no entry are simply absent; the calibrations themselves are
        shared, not copied, so this is a view for saving, not a fork to edit.
        """
        picked = CalibrationSet()
        for video in videos:
            key = self._key(video)
            calibration = self.entries.get(key)
            if calibration is not None:
                picked.entries[key] = calibration
            arena = self.arenas.get(key)
            if arena is not None:
                picked.arenas[key] = arena
                if key in self._unconfirmed:
                    picked._unconfirmed.add(key)
        return picked

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def px_per_mm(self, video: Path | str) -> float | None:
        """Scale for *video*, or None when absent or carrying no usable scale.

        A drawn arena wins over a drawn line. Four corners of a known square
        are a better-conditioned measurement than one freehand segment - the
        square constrains itself, a line has nothing to check it against - and
        the arena reports the scale at the centre of the floor rather than
        wherever the line happened to be drawn, which on a tilted view is not
        the same number.

        An arena that does not describe a usable quadrilateral is ignored
        rather than fatal: a half-drawn perimeter must not knock out a line
        scale that already works.
        """
        arena = self.get_arena(video)
        if arena is not None:
            try:
                ppm = arena.px_per_mm_centre
                if ppm > 0:
                    return ppm
            except DegenerateArenaError:
                logger.warning("ignoring unusable arena for %s", video)

        calibration = self.get(video)
        if calibration is None:
            return None
        ppm = calibration.pixels_per_mm
        return ppm if ppm > 0 else None

    def missing(self, videos: Sequence[Path]) -> list[Path]:
        """Videos still needing calibration, in the order given.

        An entry that exists but yields no scale counts as missing: a stored
        calibration the operator never drew a usable line on is not one.
        """
        return [v for v in videos if self.px_per_mm(v) is None]

    def missing_arenas(self, videos: Sequence[Path]) -> list[Path]:
        """Videos still needing a usable, confirmed arena, in the order given.

        Parallel to :meth:`missing`, which asks the weaker question "is there a
        scale". An arena that will not fit a homography is ignored the same way
        ``px_per_mm`` ignores it, but here that makes the video missing rather
        than merely falling back to a line.
        """
        out = []
        for video in videos:
            arena = self.get_arena(video)
            if arena is None or not self.is_arena_confirmed(video):
                out.append(video)
                continue
            try:
                arena.homography()
            except DegenerateArenaError:
                out.append(video)
        return out

    def is_complete(self, videos: Sequence[Path]) -> bool:
        return not self.missing(videos)

    # ------------------------------------------------------------------
    # master file
    # ------------------------------------------------------------------

    def to_dict(self, *, model: Path | None = None) -> dict:
        """The master-file payload.

        ``px_per_mm`` / ``mm_per_px`` are written for analysts joining this to
        the DLC CSVs; they are *derived* and are ignored on load, so the file
        can never contradict its own lines.
        """
        videos = []
        # Sorted by path (not insertion order) so two runs over the same batch
        # produce diffable files regardless of the order the operator happened
        # to calibrate videos in.
        for video in self.videos():
            calibration = self.entries.get(video) or CameraCalibration()
            arena = self.arenas.get(video)
            # The written scale is the one px_per_mm() would answer with, so
            # an analyst joining this file to the CSVs sees what the pipeline
            # actually used rather than only what the lines say.
            ppm = self.px_per_mm(video) or 0.0

            width = calibration.calibration_width
            height = calibration.calibration_height
            if not width and arena is not None:
                width, height = arena.frame_size

            entry = {
                "video": str(video),
                "resolution": [width, height],
                "px_per_mm": ppm,
                "mm_per_px": (1.0 / ppm) if ppm > 0 else None,
                "calibration": calibration.to_dict(),
            }
            # Optional, and only when drawn. Absent keys keep files written by
            # line-only batches byte-identical to what earlier builds produced.
            if arena is not None:
                entry["arena"] = arena.to_dict()
                # Only when unconfirmed: absent means drawn-and-checked, so
                # files written by earlier builds keep their meaning and files
                # written by this one stay diffable against them.
                if video in self._unconfirmed:
                    entry["arena_confirmed"] = False
            videos.append(entry)
        return {
            "schema_version": SCHEMA_VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "model": str(model) if model is not None else None,
            "videos": videos,
        }

    def save(self, path: Path, *, model: Path | None = None) -> None:
        """Write the master calibration file. Raises OSError or ValueError if it cannot.

        Writes to a sibling temp file and ``os.replace``s it onto *path* —
        atomic on both Windows and POSIX — so a crash, full disk, or network
        hiccup mid-write can never leave a half-written master file behind;
        the previous good file (if any) survives untouched.

        ValueError, not just OSError, because *path* often comes from a free-text
        field: an embedded null byte is rejected below the OSError layer.
        """
        path = Path(path)
        payload = json.dumps(self.to_dict(model=model), indent=2) + "\n"
        tmp: Path | None = None
        try:
            tmp = path.with_name(path.name + ".tmp")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)
        except (OSError, ValueError):
            if tmp is not None:
                # A path the OS rejected outright left nothing behind, and
                # unlinking it would raise the same error over the real one.
                with suppress(OSError, ValueError):
                    tmp.unlink(missing_ok=True)
            raise
        logger.info("wrote calibration master for %d video(s) to %s", len(self.entries), path)

    @classmethod
    def load(cls, path: Path, *, known_videos: Iterable[Path] | None = None) -> CalibrationSet:
        """Read a master calibration file.

        ``known_videos`` enables filename recovery: when a stored path no longer
        exists (data copied to another drive) the entry is re-keyed to the video
        in *known_videos* with the same filename — but only when exactly one
        candidate matches. Every recovery is logged; nothing is matched
        silently, and an ambiguous filename is skipped rather than guessed.

        Caution: a stored path is matched by ``Path.exists()`` — presence,
        not identity. If a video is re-recorded at the same path, the old
        calibration is silently inherited by the new recording; this module
        has no way to detect that the file's content changed underneath it.

        Raises CalibrationSetError if the file cannot be understood. A
        malformed entry anywhere in the file aborts the whole load — nothing
        is applied — so a batch run never operates on a half-read map.
        """
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise CalibrationSetError(f"cannot read calibration file {path}: {e}") from e

        if not isinstance(data, dict):
            raise CalibrationSetError(f"{path} is not a calibration file")

        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise CalibrationSetError(
                f"{path} has schema_version {version!r}; this build understands {SCHEMA_VERSION}"
            )

        videos = data.get("videos", [])
        if not isinstance(videos, list):
            raise CalibrationSetError(f"{path} has a non-list 'videos'")

        by_name: dict[str, list[Path]] = {}
        for video in known_videos or ():
            key = cls._key(video)
            by_name.setdefault(key.name.lower(), []).append(key)

        cal_set = cls()
        for entry in videos:
            try:
                if not isinstance(entry, dict):
                    raise TypeError(f"video entry is a {type(entry).__name__}, not an object")
                calibration_data = entry["calibration"]
                if not isinstance(calibration_data, dict):
                    raise TypeError(
                        f"'calibration' is a {type(calibration_data).__name__}, not an object"
                    )
                stored = Path(entry["video"])
                calibration = CameraCalibration.from_dict(calibration_data)
                arena_data = entry.get("arena")
                if arena_data is not None and not isinstance(arena_data, dict):
                    raise TypeError(f"'arena' is a {type(arena_data).__name__}, not an object")
                arena = ArenaCalibration.from_dict(arena_data) if arena_data else None
                key = cls._key(stored)
                try:
                    key.stat()
                    key_exists = True
                except OSError:
                    # Missing, unreadable, or otherwise unconfirmable: treat as
                    # absent so filename recovery gets a chance below. Unlike
                    # Path.exists(), Path.stat() does not also swallow
                    # ValueError (e.g. an embedded null byte in the stored
                    # path), so that still falls through to the outer guard.
                    key_exists = False
            except (KeyError, TypeError, ValueError, AttributeError) as e:
                raise CalibrationSetError(f"{path} has a malformed video entry: {e}") from e

            if not key_exists:
                candidates = by_name.get(key.name.lower(), [])
                if len(candidates) == 1:
                    logger.info("calibration for %s recovered as %s", stored, candidates[0])
                    key = candidates[0]
                elif len(candidates) > 1:
                    logger.warning(
                        "calibration for %s not applied: %d batch videos share that filename",
                        stored,
                        len(candidates),
                    )
                    continue
            cal_set.entries[key] = calibration
            if arena is not None:
                cal_set.arenas[key] = arena
                if entry.get("arena_confirmed") is False:
                    cal_set._unconfirmed.add(key)
        return cal_set
