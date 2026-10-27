"""Skeleton templates: named keypoints, where they sit, and their colours.

Modelled on the Ultralytics Platform's skeleton templates. A template is what a
pose project is labelled with: the keypoint order the model learns, and a
reference layout -- the shape a whole skeleton is dropped in, and how the
labeller sees which body part is which. Keypoints are not joined by lines.

Built-in templates ship with GLIDER; custom ones are JSON files in
:func:`custom_templates_dir`, shared by every project on this computer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from glider.core.fileio import atomic_write_bytes

logger = logging.getLogger(__name__)

__all__ = [
    "POSE_PALETTE",
    "TEMPLATE_VERSION",
    "SkeletonTemplate",
    "TemplateError",
    "TemplateKeypoint",
    "builtin_templates",
    "custom_templates_dir",
    "delete_custom_template",
    "list_custom_templates",
    "palette_colour",
    "save_custom_template",
]

TEMPLATE_VERSION = 1

#: Ultralytics' pose palette (RGB), in order: keypoint *i* defaults to entry *i*,
#: which is what the Platform annotator shows.
POSE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 128, 0),
    (255, 153, 51),
    (255, 178, 102),
    (230, 230, 0),
    (255, 153, 255),
    (153, 204, 255),
    (255, 102, 255),
    (255, 51, 255),
    (102, 178, 255),
    (51, 153, 255),
    (255, 153, 153),
    (255, 102, 102),
    (255, 51, 51),
    (153, 255, 153),
    (102, 255, 102),
    (51, 255, 51),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (255, 255, 255),
)


def palette_colour(index: int) -> tuple[int, int, int]:
    return POSE_PALETTE[index % len(POSE_PALETTE)]


class TemplateError(ValueError):
    """A template could not be read, or is not usable."""


@dataclass
class TemplateKeypoint:
    """One keypoint: name, reference position (0-1) and colour (RGB)."""

    name: str
    x: float
    y: float
    colour: tuple[int, int, int] = (255, 128, 0)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "color": list(self.colour),
        }

    @classmethod
    def from_dict(cls, data: dict, index: int = 0) -> TemplateKeypoint:
        colour = data.get("color") or palette_colour(index)
        return cls(
            name=str(data["name"]),
            x=float(data["x"]),
            y=float(data["y"]),
            colour=tuple(int(c) for c in colour[:3]),
        )


@dataclass
class SkeletonTemplate:
    """An ordered set of named, coloured keypoints in a reference layout."""

    name: str
    keypoints: list[TemplateKeypoint] = field(default_factory=list)
    description: str = ""
    builtin: bool = False

    @property
    def names(self) -> list[str]:
        return [k.name for k in self.keypoints]

    @property
    def colours(self) -> list[tuple[int, int, int]]:
        return [k.colour for k in self.keypoints]

    def problem(self) -> str | None:
        """Why this template can't be used, or None."""
        if not self.name.strip():
            return "the template needs a name"
        if not self.keypoints:
            return "a template needs at least one keypoint"
        seen: dict[str, int] = {}
        for i, name in enumerate(self.names):
            if not name.strip():
                return f"keypoint {i} has no name"
            if name in seen:
                return f"'{name}' is used twice (keypoints {seen[name]} and {i})"
            seen[name] = i
        return None

    # -- editing ------------------------------------------------------------

    def add_keypoint(self, x: float, y: float, name: str | None = None) -> int:
        """Append a keypoint and return its index. Names default to ``point_N``."""
        index = len(self.keypoints)
        taken = set(self.names)
        if name is None:
            n = index
            while f"point_{n}" in taken:
                n += 1
            name = f"point_{n}"
        self.keypoints.append(TemplateKeypoint(name, x, y, palette_colour(index)))
        return index

    def remove_keypoint(self, index: int) -> None:
        """Delete a keypoint; the ones after it move up one place."""
        if 0 <= index < len(self.keypoints):
            del self.keypoints[index]

    def copy(self, name: str | None = None) -> SkeletonTemplate:
        return SkeletonTemplate(
            name=self.name if name is None else name,
            keypoints=[TemplateKeypoint(k.name, k.x, k.y, k.colour) for k in self.keypoints],
            description=self.description,
            builtin=False,
        )

    # -- files --------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "template_version": TEMPLATE_VERSION,
            "name": self.name,
            "description": self.description,
            "keypoints": [k.to_dict() for k in self.keypoints],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SkeletonTemplate:
        if not isinstance(data, dict):
            raise TemplateError("not a skeleton template")
        version = data.get("template_version")
        if version != TEMPLATE_VERSION:
            raise TemplateError(
                f"template_version {version!r}; this build understands {TEMPLATE_VERSION}"
            )
        try:
            keypoints = [TemplateKeypoint.from_dict(k, i) for i, k in enumerate(data["keypoints"])]
        except (KeyError, TypeError, ValueError) as e:
            raise TemplateError(f"malformed template: {e}") from e
        # A "skeleton" key, from files saved while templates had connections,
        # is ignored.
        return cls(
            name=str(data.get("name", "Untitled")),
            keypoints=keypoints,
            description=str(data.get("description", "")),
        )

    @classmethod
    def load(cls, path: Path | str) -> SkeletonTemplate:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise TemplateError(f"cannot read {path}: {e}") from e
        return cls.from_dict(data)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, (json.dumps(self.to_dict(), indent=2) + "\n").encode("utf-8"))


# ----------------------------------------------------------------------
# built-in templates
# ----------------------------------------------------------------------


def _template(name, description, points) -> SkeletonTemplate:
    keypoints = [TemplateKeypoint(n, x, y, palette_colour(i)) for i, (n, x, y) in enumerate(points)]
    return SkeletonTemplate(name=name, description=description, keypoints=keypoints, builtin=True)


def _mouse() -> SkeletonTemplate:
    return _template(
        "Mouse",
        "Top-view mouse, the lab's seven-point layout",
        [
            ("left_ear", 0.40, 0.22),
            ("right_ear", 0.60, 0.22),
            ("nose", 0.50, 0.08),
            ("body_center", 0.50, 0.48),
            ("left_hip", 0.39, 0.68),
            ("right_hip", 0.61, 0.68),
            ("tail_base", 0.50, 0.82),
        ],
    )


def _person() -> SkeletonTemplate:
    # COCO order. Facing the viewer, so the person's
    # left is on the right of the picture.
    return _template(
        "Person",
        "COCO human body pose estimation",
        [
            ("nose", 0.50, 0.10),
            ("left_eye", 0.53, 0.08),
            ("right_eye", 0.47, 0.08),
            ("left_ear", 0.57, 0.09),
            ("right_ear", 0.43, 0.09),
            ("left_shoulder", 0.63, 0.24),
            ("right_shoulder", 0.37, 0.24),
            ("left_elbow", 0.70, 0.40),
            ("right_elbow", 0.30, 0.40),
            ("left_wrist", 0.74, 0.55),
            ("right_wrist", 0.26, 0.55),
            ("left_hip", 0.58, 0.54),
            ("right_hip", 0.42, 0.54),
            ("left_knee", 0.59, 0.73),
            ("right_knee", 0.41, 0.73),
            ("left_ankle", 0.60, 0.92),
            ("right_ankle", 0.40, 0.92),
        ],
    )


def _hand() -> SkeletonTemplate:
    # The 21-point hand (Ultralytics hand-keypoints order): wrist, then four
    # joints per finger from the palm out. Palm towards the viewer, fingers up.
    joints = {
        "thumb": (("cmc", 0.36, 0.80), ("mcp", 0.26, 0.70), ("ip", 0.19, 0.60), ("tip", 0.13, 0.51)),
        "index": (("mcp", 0.40, 0.55), ("pip", 0.38, 0.40), ("dip", 0.37, 0.29), ("tip", 0.36, 0.19)),
        "middle": (("mcp", 0.50, 0.53), ("pip", 0.50, 0.36), ("dip", 0.50, 0.24), ("tip", 0.50, 0.13)),
        "ring": (("mcp", 0.60, 0.55), ("pip", 0.62, 0.40), ("dip", 0.63, 0.29), ("tip", 0.64, 0.20)),
        "pinky": (("mcp", 0.69, 0.60), ("pip", 0.73, 0.49), ("dip", 0.75, 0.40), ("tip", 0.77, 0.32)),
    }  # fmt: skip
    points = [("wrist", 0.50, 0.92)]
    for finger, chain in joints.items():
        points += [(f"{finger}_{joint}", x, y) for joint, x, y in chain]
    return _template("Hand", "21-point hand keypoints", points)


def _quadruped() -> SkeletonTemplate:
    # AP-10K order: side view, head to the left.
    return _template(
        "Quadruped",
        "AP-10K animal pose (17 keypoints)",
        [
            ("left_eye", 0.18, 0.22),
            ("right_eye", 0.22, 0.20),
            ("nose", 0.08, 0.30),
            ("neck", 0.30, 0.32),
            ("root_of_tail", 0.80, 0.34),
            ("left_shoulder", 0.34, 0.45),
            ("left_elbow", 0.33, 0.62),
            ("left_front_paw", 0.32, 0.84),
            ("right_shoulder", 0.39, 0.45),
            ("right_elbow", 0.40, 0.62),
            ("right_front_paw", 0.41, 0.84),
            ("left_hip", 0.74, 0.45),
            ("left_knee", 0.71, 0.63),
            ("left_back_paw", 0.70, 0.84),
            ("right_hip", 0.79, 0.45),
            ("right_knee", 0.80, 0.63),
            ("right_back_paw", 0.81, 0.84),
        ],
    )


def _box() -> SkeletonTemplate:
    return _template(
        "Box",
        "Four corner keypoints",
        [
            ("top_left", 0.2, 0.2),
            ("top_right", 0.8, 0.2),
            ("bottom_right", 0.8, 0.8),
            ("bottom_left", 0.2, 0.8),
        ],
    )


def builtin_templates() -> list[SkeletonTemplate]:
    """The templates GLIDER ships, Mouse first. Fresh copies on every call."""
    return [_mouse(), _person(), _hand(), _quadruped(), _box()]


# ----------------------------------------------------------------------
# custom templates
# ----------------------------------------------------------------------


def custom_templates_dir() -> Path:
    """Where custom templates are kept: ``~/.glider/skeleton_templates``."""
    return Path.home() / ".glider" / "skeleton_templates"


def _file_for(name: str, folder: Path) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name.strip().lower()).strip("-") or "template"
    return folder / f"{slug}.json"


def list_custom_templates(folder: Path | None = None) -> list[SkeletonTemplate]:
    """Every readable custom template, by name. Unreadable files are skipped."""
    folder = folder or custom_templates_dir()
    found = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            found.append(SkeletonTemplate.load(path))
        except TemplateError as e:
            logger.warning("skipping skeleton template %s: %s", path.name, e)
    return sorted(found, key=lambda t: t.name.lower())


def save_custom_template(
    template: SkeletonTemplate, folder: Path | None = None, replacing: str | None = None
) -> Path:
    """Save a custom template, keyed by its name.

    Args:
        replacing: The name it had before an edit; that file is removed when the
            name changed, so a rename does not leave the old one behind.

    Raises:
        TemplateError: The template is unusable, or a built-in or another
            custom template already has its name.
    """
    folder = folder or custom_templates_dir()
    problem = template.problem()
    if problem:
        raise TemplateError(problem)
    name = template.name.strip()
    if name.lower() in {t.name.lower() for t in builtin_templates()}:
        raise TemplateError(f"'{name}' is a built-in template; choose another name")
    path = _file_for(name, folder)
    clash = path.exists() and (replacing is None or _file_for(replacing, folder) != path)
    if clash:
        raise TemplateError(f"a custom template called '{name}' already exists")
    template.save(path)
    if replacing is not None and _file_for(replacing, folder) != path:
        _file_for(replacing, folder).unlink(missing_ok=True)
    return path


def delete_custom_template(name: str, folder: Path | None = None) -> None:
    _file_for(name, folder or custom_templates_dir()).unlink(missing_ok=True)
