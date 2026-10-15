"""A retrieval session's setup, and its cohort's protocol, as JSON beside the data.

Two files, because two kinds of setting:

* ``<stem>.retrieval.json`` beside each video -- what is true of *that* video:
  where the nest is (one outline per period, since a dam drags her nest), when
  the trial starts, and the scale.
* ``retrieval_protocol.json`` at the cohort root -- the rules every video is
  scored by: the cutoff, which keypoint is the nose, and the thresholds. Kept
  once so two sessions cannot be scored under different rules by accident.

A missing file is defaults. A file that exists but cannot be read raises
:class:`SidecarError`, and the caller must then never write over it -- the
same rule ``review_markers.json`` follows. So does one that parses but holds
a value GLIDER never writes (a NaN, a negative time or scale): read anyway,
it would score every frame NaN or raise from a Qt slot. Unknown keys survive
a save.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

from glider.analysis.behavior.retrieval import RetrievalSettings
from glider.core.fileio import atomic_write_text
from glider.vision.zones import Zone

__all__ = [
    "PROTOCOL_FILE",
    "SETUP_SUFFIX",
    "VERSION",
    "NestPeriod",
    "RetrievalProtocol",
    "RetrievalSetup",
    "SidecarError",
    "load_protocol",
    "load_setup",
    "protocol_path",
    "save_protocol",
    "save_setup",
    "setup_path",
]

SETUP_SUFFIX = ".retrieval.json"
PROTOCOL_FILE = "retrieval_protocol.json"
VERSION = 1


class SidecarError(ValueError):
    """A settings file exists but cannot be read; it must not be overwritten."""


def setup_path(video: Path | str) -> Path:
    video = Path(video)
    return video.with_name(f"{video.stem}{SETUP_SUFFIX}")


def protocol_path(folder: Path | str) -> Path:
    return Path(folder) / PROTOCOL_FILE


def _number(value, key: str, *, positive: bool = False) -> float:
    """*value* as a finite float, never negative (above 0 if *positive*); ValueError if not."""
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(f"{key} cannot be {value!r}")
    return number


def _opt_number(value, key: str, *, positive: bool = False) -> float | None:
    return None if value is None else _number(value, key, positive=positive)


@dataclass(frozen=True)
class NestPeriod:
    """The nest outline in force from *from_s* until the next period starts."""

    from_s: float
    zone: Zone


def _nest_period(entry: dict) -> NestPeriod:
    """A nest period from the file, its vertices checked: the viewer paints them."""
    zone = Zone.from_dict(entry["zone"])
    vertices = [(float(x), float(y)) for x, y in zone.vertices]
    if not all(math.isfinite(c) for vertex in vertices for c in vertex):
        raise ValueError(f"nest {zone.name!r} has a vertex that is not a number")
    return NestPeriod(_number(entry["from_s"], "from_s"), replace(zone, vertices=vertices))


@dataclass
class RetrievalSetup:
    trial_start_s: float | None = None
    nests: list[NestPeriod] = field(default_factory=list)
    px_per_mm: float | None = None
    extra: dict = field(default_factory=dict)

    def nest_at(self, t_s: float) -> NestPeriod | None:
        current = self.nests[0] if self.nests else None
        for period in self.nests:
            if period.from_s <= t_s:
                current = period
        return current

    def set_nest(self, from_s: float, zone: Zone) -> None:
        """Add a period from *from_s*, replacing one that starts there.

        The first period always starts at 0, so every frame has a nest.
        """
        from_s = 0.0 if not self.nests else max(0.0, float(from_s))
        kept = [p for p in self.nests if abs(p.from_s - from_s) > 1e-6]
        self.nests = sorted([*kept, NestPeriod(from_s, zone)], key=lambda p: p.from_s)

    def remove_nest(self, index: int) -> None:
        del self.nests[index]
        if self.nests and self.nests[0].from_s != 0.0:
            self.nests[0] = replace(self.nests[0], from_s=0.0)

    def to_dict(self) -> dict:
        return {
            **self.extra,
            "version": VERSION,
            "trial_start_s": self.trial_start_s,
            "nests": [{"from_s": p.from_s, "zone": p.zone.to_dict()} for p in self.nests],
            "px_per_mm": self.px_per_mm,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RetrievalSetup:
        known = {"version", "trial_start_s", "nests", "px_per_mm"}
        # In time order, the first from 0 -- the rules set_nest and remove_nest
        # keep -- however the file was edited.
        nests = sorted((_nest_period(n) for n in data.get("nests", [])), key=lambda p: p.from_s)
        if nests and nests[0].from_s != 0.0:
            nests[0] = replace(nests[0], from_s=0.0)
        return cls(
            trial_start_s=_opt_number(data.get("trial_start_s"), "trial_start_s"),
            nests=nests,
            px_per_mm=_opt_number(data.get("px_per_mm"), "px_per_mm", positive=True),
            extra={k: v for k, v in data.items() if k not in known},
        )


_DEFAULTS = RetrievalSettings()


@dataclass
class RetrievalProtocol:
    cutoff_s: float = 300.0
    nose_keypoint: str = "nose"
    in_nest_s: float = _DEFAULTS.in_nest_s
    contact_cm: float = _DEFAULTS.contact_cm
    carry_speed_cm_s: float = _DEFAULTS.carry_speed_cm_s
    extra: dict = field(default_factory=dict)

    def settings(self) -> RetrievalSettings:
        return replace(
            _DEFAULTS,
            in_nest_s=self.in_nest_s,
            contact_cm=self.contact_cm,
            carry_speed_cm_s=self.carry_speed_cm_s,
        )

    def to_dict(self) -> dict:
        return {
            **self.extra,
            "version": VERSION,
            "cutoff_s": self.cutoff_s,
            "nose_keypoint": self.nose_keypoint,
            "in_nest_s": self.in_nest_s,
            "contact_cm": self.contact_cm,
            "carry_speed_cm_s": self.carry_speed_cm_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RetrievalProtocol:
        base = cls()
        known = {
            "version",
            "cutoff_s",
            "nose_keypoint",
            "in_nest_s",
            "contact_cm",
            "carry_speed_cm_s",
        }
        number = {
            key: _number(data.get(key, getattr(base, key)), key, positive=True)
            for key in ("cutoff_s", "in_nest_s", "contact_cm", "carry_speed_cm_s")
        }
        return cls(
            nose_keypoint=str(data.get("nose_keypoint", base.nose_keypoint)),
            extra={k: v for k, v in data.items() if k not in known},
            **number,
        )


def _read(path: Path) -> dict | None:
    """The file's JSON object, None if there is no file; SidecarError otherwise."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as e:
        raise SidecarError(f"{path.name} could not be read: {e}") from e
    try:
        data = json.loads(text)
    # JSONDecodeError is a ValueError, as is an int past Python's 4300-digit
    # limit; deep nesting runs out of stack.
    except (ValueError, RecursionError) as e:
        raise SidecarError(f"{path.name} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise SidecarError(f"{path.name} does not hold a settings object")
    version = data.get("version")
    if not isinstance(version, int) or version > VERSION:
        raise SidecarError(
            f"{path.name} has version {version!r}; this GLIDER reads up to {VERSION}"
        )
    return data


def _parse(path: Path, cls, data: dict | None):
    if data is None:
        return cls()
    try:
        return cls.from_dict(data)
    # A valid JSON shape can still hold a malformed entry: a zone that is not an
    # object raises AttributeError, an absurdly large integer OverflowError.
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as e:
        raise SidecarError(f"{path.name} has an unreadable entry: {e}") from e


def load_setup(path: Path | str) -> RetrievalSetup:
    path = Path(path)
    return _parse(path, RetrievalSetup, _read(path))


def save_setup(path: Path | str, setup: RetrievalSetup) -> None:
    atomic_write_text(path, json.dumps(setup.to_dict(), indent=2))


def load_protocol(folder: Path | str) -> RetrievalProtocol:
    path = protocol_path(folder)
    return _parse(path, RetrievalProtocol, _read(path))


def save_protocol(folder: Path | str, protocol: RetrievalProtocol) -> None:
    atomic_write_text(protocol_path(folder), json.dumps(protocol.to_dict(), indent=2))
