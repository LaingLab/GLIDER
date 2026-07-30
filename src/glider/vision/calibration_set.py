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

from glider.vision.calibration import CameraCalibration

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "CalibrationSet", "CalibrationSetError"]


class CalibrationSetError(ValueError):
    """A master calibration file could not be understood."""


@dataclass
class CalibrationSet:
    """Calibrations for the videos of one batch, keyed by resolved video path."""

    entries: dict[Path, CameraCalibration] = field(default_factory=dict)

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
        return picked

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------

    def px_per_mm(self, video: Path | str) -> float | None:
        """Scale for *video*, or None when absent or carrying no usable scale."""
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
        for video, calibration in sorted(self.entries.items(), key=lambda kv: kv[0]):
            ppm = calibration.pixels_per_mm
            videos.append(
                {
                    "video": str(video),
                    "resolution": [
                        calibration.calibration_width,
                        calibration.calibration_height,
                    ],
                    "px_per_mm": ppm,
                    "mm_per_px": (1.0 / ppm) if ppm > 0 else None,
                    "calibration": calibration.to_dict(),
                }
            )
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
        return cal_set
