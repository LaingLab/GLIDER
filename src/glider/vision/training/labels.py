"""YOLO pose labels: one text file per image, one line per animal.

A line is ``class cx cy w h  x1 y1 v1  x2 y2 v2 ...``, every coordinate
normalised 0-1 against the image. Visibility follows the COCO convention
Ultralytics trains on: 0 = not labelled, 1 = labelled but hidden, 2 = visible.

The file's presence is what marks a frame as labelled. An image with no label
file has not been looked at yet; that distinction is :class:`StudioProject`'s
to keep, since an empty file means something else to Ultralytics (a frame with
no animals in it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from glider.core.fileio import atomic_write_bytes

__all__ = [
    "HIDDEN",
    "MISSING",
    "VISIBLE",
    "LabelFormatError",
    "PoseInstance",
    "format_pose_line",
    "parse_pose_line",
    "read_labels",
    "write_labels",
]

MISSING, HIDDEN, VISIBLE = 0, 1, 2

#: Smallest box side, as a fraction of the image. A lone labelled point would
#: otherwise give a zero-size box, which Ultralytics drops as degenerate.
_MIN_BOX = 0.02


class LabelFormatError(ValueError):
    """A label line could not be understood."""


@dataclass
class PoseInstance:
    """One animal in one image.

    Args:
        keypoints: ``(x, y, v)`` per keypoint, in schema order, normalised 0-1.
            Unlabelled keypoints are ``(0, 0, MISSING)``.
        class_id: Index into the project's class names.
    """

    keypoints: list[tuple[float, float, int]] = field(default_factory=list)
    class_id: int = 0

    @classmethod
    def empty(cls, n_keypoints: int, class_id: int = 0) -> PoseInstance:
        return cls([(0.0, 0.0, MISSING)] * n_keypoints, class_id)

    @property
    def labelled(self) -> list[tuple[float, float, int]]:
        """The keypoints that have been placed (visible or hidden)."""
        return [k for k in self.keypoints if k[2] != MISSING]

    def box(self, padding: float = 0.2) -> tuple[float, float, float, float] | None:
        """``(cx, cy, w, h)`` around the placed keypoints, or None if none are.

        Keypoints sit inside the body, so the box is grown by ``padding`` times
        its longer side on every edge to take in the rest of the animal, then
        clipped to the image.
        """
        placed = self.labelled
        if not placed:
            return None
        xs = [k[0] for k in placed]
        ys = [k[1] for k in placed]
        pad = padding * max(max(xs) - min(xs), max(ys) - min(ys), _MIN_BOX)
        x1, x2 = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
        y1, y2 = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
        return ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, _MIN_BOX), max(y2 - y1, _MIN_BOX))


def format_pose_line(instance: PoseInstance, padding: float = 0.2) -> str | None:
    """The label line for one animal, or None if it has no keypoints placed."""
    box = instance.box(padding)
    if box is None:
        return None
    parts = [str(instance.class_id), *(f"{v:.6f}" for v in box)]
    for x, y, v in instance.keypoints:
        if v == MISSING:
            parts += ["0", "0", "0"]
        else:
            parts += [f"{x:.6f}", f"{y:.6f}", str(int(v))]
    return " ".join(parts)


def parse_pose_line(line: str, n_keypoints: int) -> PoseInstance:
    """Read one label line written for ``n_keypoints`` keypoints."""
    fields = line.split()
    expected = 5 + 3 * n_keypoints
    if len(fields) != expected:
        raise LabelFormatError(
            f"expected {expected} values for {n_keypoints} keypoints, got {len(fields)}"
        )
    try:
        class_id = int(fields[0])
        values = [float(f) for f in fields[5:]]
    except ValueError as e:
        raise LabelFormatError(str(e)) from e
    keypoints = []
    for i in range(n_keypoints):
        x, y, v = values[3 * i : 3 * i + 3]
        v = int(v)
        keypoints.append((0.0, 0.0, MISSING) if v == MISSING else (x, y, v))
    return PoseInstance(keypoints, class_id)


def write_labels(path: Path | str, instances: list[PoseInstance], padding: float = 0.2) -> bool:
    """Write the animals in one image; returns whether a file was written.

    Animals with no keypoints placed are left out, and if that leaves none the
    label file is removed: an image whose last point was deleted goes back to
    "not labelled" rather than becoming a deliberate "no animals here".
    """
    path = Path(path)
    lines = [line for i in instances if (line := format_pose_line(i, padding)) is not None]
    if not lines:
        path.unlink(missing_ok=True)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return True


def read_labels(path: Path | str, n_keypoints: int) -> list[PoseInstance]:
    """Every animal in one image's label file; empty if there is no file."""
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [parse_pose_line(line, n_keypoints) for line in text.splitlines() if line.strip()]
