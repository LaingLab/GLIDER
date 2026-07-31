"""A named, ordered keypoint layout — the thing a pose model actually agrees on.

Keypoint names are positional: the Nth name binds to the pose model's Nth
output, and a behavior model then looks its features up by those names. Order
is therefore load-bearing and not merely cosmetic — with
``FeatureSpec.auto_angles`` the angle columns are generated from keypoint
triplets *by index* and bake the names at those indices into the column names,
so the same names in a different order produce columns no model has seen.

Getting that wrong is silent: the features arrive as NaN, every prediction
comes back blank, and nothing raises. So the layout is worth capturing as a
file you can save, reload, and share, rather than a comma-separated string
retyped from memory each run.

Positions are normalised 0-1 against a reference figure (x left-to-right,
y nose-to-tail). They carry no analytical meaning — nothing downstream reads
them — but they make a schema self-documenting: you can see at a glance which
body part index 3 is meant to be.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "Keypoint", "KeypointSchema", "KeypointSchemaError"]


class KeypointSchemaError(ValueError):
    """A keypoint schema file could not be understood."""


@dataclass
class Keypoint:
    """One named point, positioned on the reference figure."""

    name: str
    x: float  # 0-1, left to right
    y: float  # 0-1, nose to tail

    def to_dict(self) -> dict:
        return {"name": self.name, "x": float(self.x), "y": float(self.y)}

    @classmethod
    def from_dict(cls, data: dict) -> Keypoint:
        return cls(name=str(data["name"]), x=float(data["x"]), y=float(data["y"]))


# The seven-point mouse layout this lab uses, in the order the cohort models
# were trained with. Positions are eyeballed onto a top-view figure; only the
# order and the names are load-bearing.
_DEFAULT_MOUSE: tuple[tuple[str, float, float], ...] = (
    ("left_ear", 0.40, 0.16),
    ("right_ear", 0.60, 0.16),
    ("nose", 0.50, 0.05),
    ("body_center", 0.50, 0.45),
    ("left_hip", 0.39, 0.66),
    ("right_hip", 0.61, 0.66),
    ("tail_base", 0.50, 0.78),
)


@dataclass
class KeypointSchema:
    """An ordered list of named keypoints, savable and reloadable."""

    keypoints: list[Keypoint] = field(default_factory=list)

    @classmethod
    def default_mouse(cls) -> KeypointSchema:
        """The seven-point top-view mouse layout, in trained order."""
        return cls([Keypoint(n, x, y) for n, x, y in _DEFAULT_MOUSE])

    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """The names, in order — what the keypoint-names field wants."""
        return [k.name for k in self.keypoints]

    def problem(self) -> str | None:
        """Why this schema is unusable, or None.

        Checked before a schema can be applied: duplicate or blank names
        silently break the feature lookup rather than raising anywhere.
        """
        if not self.keypoints:
            return "a schema needs at least one keypoint"
        names = self.names
        blank = [i for i, n in enumerate(names) if not n.strip()]
        if blank:
            return f"keypoint {blank[0]} has no name"
        seen: dict[str, int] = {}
        for i, n in enumerate(names):
            if n in seen:
                return f"'{n}' is used twice (positions {seen[n]} and {i})"
            seen[n] = i
        return None

    def move(self, index: int, delta: int) -> int:
        """Shift the keypoint at *index* by *delta* places; return its new index.

        Order is the whole point of this object, so reordering is a first-class
        operation rather than something the caller does to the list behind its
        back.
        """
        if not 0 <= index < len(self.keypoints):
            return index
        target = max(0, min(len(self.keypoints) - 1, index + delta))
        if target != index:
            self.keypoints.insert(target, self.keypoints.pop(index))
        return target

    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "keypoints": [k.to_dict() for k in self.keypoints],
        }

    def save(self, path: Path | str) -> None:
        """Write the schema. Raises OSError if it cannot be written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        logger.info("wrote keypoint schema (%d points) to %s", len(self.keypoints), path)

    @classmethod
    def from_dict(cls, data: dict) -> KeypointSchema:
        if not isinstance(data, dict):
            raise KeypointSchemaError("not a keypoint schema")
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise KeypointSchemaError(
                f"schema_version {version!r}; this build understands {SCHEMA_VERSION}"
            )
        raw = data.get("keypoints")
        if not isinstance(raw, list):
            raise KeypointSchemaError("'keypoints' must be a list")
        try:
            points = [Keypoint.from_dict(k) for k in raw]
        except (KeyError, TypeError, ValueError) as e:
            raise KeypointSchemaError(f"malformed keypoint entry: {e}") from e
        return cls(points)

    @classmethod
    def load(cls, path: Path | str) -> KeypointSchema:
        """Read a schema file. Raises KeypointSchemaError if unreadable."""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise KeypointSchemaError(f"cannot read {path}: {e}") from e
        return cls.from_dict(data)
