"""Arena calibration - perspective-correct mapping between pixels and the floor.

:class:`~glider.vision.calibration.CameraCalibration` answers "how many pixels
is a centimetre" with a single number drawn from one line. That is enough for a
camera looking straight down and nothing else: tilt it over the arena and the
far wall is genuinely fewer pixels per centimetre than the near wall, so one
scalar is wrong at both ends and right only somewhere in between.

Drawing the whole floor perimeter instead fixes a homography, which carries the
perspective rather than averaging it away. Two things fall out of that:

**Zones that mean the same thing in every video.** A centred 10x10 cm square is
defined in centimetres and mapped back into the image, so it is the same region
of floor in every session regardless of where the camera sat. It lands as a
quadrilateral, not an axis-aligned rectangle - :class:`glider.vision.zones.Zone`
stores that as a ``polygon``.

**Scale as a function of position.** :meth:`ArenaCalibration.px_per_cm_at` reads
the local scale off the Jacobian, so near-wall and far-wall pixels are no longer
quietly treated as equal.

A closed square is also a better-conditioned measurement than a single line. A
line has no redundancy - nothing checks it. Four corners give four edges plus the
knowledge that the arena is square, so :meth:`ArenaCalibration.residuals` can say
when a corner was misplaced.

Corners may legitimately fall outside the frame. Cameras get mounted close, and
a projected corner is a well-defined point whether or not the sensor saw it, so
normalized coordinates outside 0-1 are kept rather than clamped.

Qt-free on purpose: a dialog drives this, but a script or notebook can build one
from four numbers with no Qt import.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

__all__ = ["SCHEMA_VERSION", "ArenaCalibration", "DegenerateArenaError"]

#: Corner order, as the operator is asked to click them.
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")

#: How far opposite edges may disagree before a quad is called a probable
#: misclick. Perspective alone moves this ratio - on this cohort's rigs the far
#: wall runs about 6% shorter than the near one - so the threshold sits well
#: above any plausible mounting and catches only gross errors.
_MAX_EDGE_RATIO = 1.30

#: How much the local pixels-per-centimetre may vary across the floor before
#: the same warning fires. 1.0 is a camera looking straight down; the rigs here
#: sit near 1.15. Past 1.6 the implied view is more oblique than a mounting
#: that films a whole open field can be, so a corner is likelier misplaced.
_MAX_SCALE_RATIO = 1.60


class DegenerateArenaError(ValueError):
    """The four corners do not describe a usable quadrilateral."""


def _as_array(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


@dataclass
class ArenaCalibration:
    """The floor of a rectangular arena, located in one camera's image.

    Args:
        corners: Four image points in normalized (0-1) coordinates, in the order
            given by :data:`CORNER_NAMES`. Values outside 0-1 are allowed and
            meaningful - see the module docstring.
        width_cm: Real width of the arena floor.
        height_cm: Real height of the arena floor.
        frame_size: ``(width, height)`` in pixels of the frame the corners were
            clicked on. Only :meth:`px_per_cm_at` needs it, since everything
            else works in normalized coordinates and is resolution-independent.
    """

    corners: list[tuple[float, float]]
    width_cm: float = 30.0
    height_cm: float = 30.0
    frame_size: tuple[int, int] = (640, 480)

    _homography: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.corners = [(float(x), float(y)) for x, y in self.corners]
        if len(self.corners) != 4:
            raise ValueError(f"An arena needs exactly four corners, got {len(self.corners)}")
        if self.width_cm <= 0 or self.height_cm <= 0:
            raise ValueError("Arena dimensions must be positive")

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @property
    def _arena_corners(self) -> np.ndarray:
        """The same four corners in centimetres, origin at the top-left."""
        w, h = self.width_cm, self.height_cm
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)

    def _check_simple(self) -> None:
        """Reject quads that are collapsed or self-intersecting.

        ``getPerspectiveTransform`` happily returns a matrix for a bow-tie, and
        every method downstream then returns plausible-looking nonsense. Cheaper
        to refuse here than to debug a zone that came out inside-out.
        """
        pts = _as_array(self.corners)
        signs = [_cross(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) for i in range(4)]
        if any(abs(s) < 1e-12 for s in signs):
            raise DegenerateArenaError(
                "Arena corners are collinear or coincident - three of the four "
                "points must not lie on one line"
            )
        if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
            raise DegenerateArenaError(
                "Arena corners are self-intersecting - click them in order "
                f"({', '.join(CORNER_NAMES)}), going around the floor"
            )

    def homography(self) -> np.ndarray:
        """3x3 matrix taking normalized image coordinates to arena centimetres.

        Solved directly rather than through OpenCV. Both ``findHomography`` and
        ``getPerspectiveTransform`` leave about 1e-6 of residual on an exact
        four-point correspondence - the first because it runs a least-squares
        fit that does not bottom out, the second because it takes float32. With
        four points the system is exactly determined, so fixing ``h33 = 1`` and
        solving the resulting 8x8 in float64 is both simpler and lands within
        1e-15. That margin is not needed for the zone itself, but it means a
        round-trip through the matrix is exactly the identity, so a later bug
        can never hide behind calibration noise.
        """
        if self._homography is None:
            self._check_simple()
            rows, rhs = [], []
            for (x, y), (u, v) in zip(self.corners, self._arena_corners, strict=True):
                rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
                rhs.append(u)
                rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
                rhs.append(v)
            try:
                solved = np.linalg.solve(
                    np.array(rows, dtype=np.float64), np.array(rhs, dtype=np.float64)
                )
            except np.linalg.LinAlgError as exc:
                raise DegenerateArenaError(
                    f"Could not fit a homography to these corners: {exc}"
                ) from exc
            self._homography = np.append(solved, 1.0).reshape(3, 3)
        return self._homography

    def inverse(self) -> np.ndarray:
        """3x3 matrix taking arena centimetres to normalized image coordinates."""
        return np.linalg.inv(self.homography())

    @staticmethod
    def _apply(matrix: np.ndarray, points) -> list[tuple[float, float]]:
        pts = _as_array(points)
        if len(pts) == 0:
            return []
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in out]

    def to_arena_cm(self, points) -> list[tuple[float, float]]:
        """Map normalized image points onto the floor, in centimetres."""
        return self._apply(self.homography(), points)

    def to_image(self, points) -> list[tuple[float, float]]:
        """Map floor points in centimetres back to normalized image coordinates."""
        return self._apply(self.inverse(), points)

    # ------------------------------------------------------------------
    # zones
    # ------------------------------------------------------------------

    def centre_zone_vertices(self, size_cm: float = 10.0) -> list[tuple[float, float]]:
        """A centred square of *size_cm*, as normalized image vertices.

        Returned in the same winding as :attr:`corners`, so it can go straight
        into a ``polygon`` :class:`glider.vision.zones.Zone`. Under perspective
        this is a quadrilateral rather than an axis-aligned rectangle; that is
        the correction, not an artefact.
        """
        if size_cm <= 0:
            raise ValueError("Zone size must be positive")
        if size_cm > self.width_cm or size_cm > self.height_cm:
            raise ValueError(
                f"A {size_cm} cm zone is larger than the arena "
                f"({self.width_cm} x {self.height_cm} cm)"
            )
        cx, cy = self.width_cm / 2, self.height_cm / 2
        half = size_cm / 2
        square = [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
        return self.to_image(square)

    # ------------------------------------------------------------------
    # scale
    # ------------------------------------------------------------------

    def px_per_cm_at(self, x_cm: float, y_cm: float) -> float:
        """Local image scale at a point on the floor.

        The Jacobian of the inverse homography maps a centimetre step on the
        floor to a pixel step in the image; its two singular values are the
        scale along each principal direction, and their geometric mean is the
        scale that preserves area. Under perspective this genuinely varies
        across the floor, which is the whole reason for this class.
        """
        w, h = self.frame_size
        eps = min(self.width_cm, self.height_cm) * 1e-4
        origin, dx, dy = self.to_image([(x_cm, y_cm), (x_cm + eps, y_cm), (x_cm, y_cm + eps)])
        jacobian = np.array(
            [
                [(dx[0] - origin[0]) * w / eps, (dy[0] - origin[0]) * w / eps],
                [(dx[1] - origin[1]) * h / eps, (dy[1] - origin[1]) * h / eps],
            ]
        )
        return float(math.sqrt(abs(np.linalg.det(jacobian))))

    @property
    def px_per_cm_centre(self) -> float:
        """Local scale at the centre of the floor.

        The single number to quote when one is needed - a defensible choice for
        a whole-arena scale, since it is where the animal spends most of its
        time and sits between the near-wall and far-wall extremes.
        """
        return self.px_per_cm_at(self.width_cm / 2, self.height_cm / 2)

    @property
    def px_per_mm_centre(self) -> float:
        """Centre scale in the units :mod:`glider.analysis.behavior.units` uses."""
        return self.px_per_cm_centre / 10.0

    # ------------------------------------------------------------------
    # quality
    # ------------------------------------------------------------------

    @property
    def is_clipped(self) -> bool:
        """Whether any corner falls outside the frame."""
        return any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in self.corners)

    def residuals(self) -> dict[str, Any]:
        """How far the clicked quad departs from a plausible view of this rig.

        A single drawn line has no redundancy - nothing contradicts it if it is
        wrong. Four corners of a known square do, so a misclick is detectable
        rather than silent.

        The check cannot be "is this quad a square", because it never is and
        need not be: a homography maps the unit square onto *any* convex quad,
        so every convex quad is a valid perspective view of some square. What
        can be judged is whether the camera pose it implies is a plausible one.
        Both measures below are 1.0 for a camera looking straight down and grow
        with obliqueness, so they separate a tilted mounting from a dragged
        corner by degree rather than by kind.

        Returns ``edge_ratio`` (the worse of the two opposite-edge length
        ratios), ``scale_ratio`` (how much the local pixels-per-centimetre
        varies across the floor) and ``suspect``.
        """
        # Same refusal as homography(): a quad with no area has no residuals to
        # report either, and callers should not have to catch two error types
        # to ask two questions about the same four points.
        self._check_simple()

        pts = _as_array(self.corners)
        edges = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
        pairs = [(edges[0], edges[2]), (edges[1], edges[3])]
        edge_ratio = max(max(a, b) / min(a, b) for a, b in pairs if min(a, b) > 0)

        try:
            w, h = self.width_cm, self.height_cm
            scales = [
                self.px_per_cm_at(x, y) for x, y in [(0, 0), (w, 0), (w, h), (0, h), (w / 2, h / 2)]
            ]
            scale_ratio = max(scales) / min(scales) if min(scales) > 0 else float("inf")
        except (DegenerateArenaError, np.linalg.LinAlgError):
            scale_ratio = float("inf")

        return {
            "edge_ratio": edge_ratio,
            "scale_ratio": scale_ratio,
            "clipped": self.is_clipped,
            "suspect": edge_ratio > _MAX_EDGE_RATIO or scale_ratio > _MAX_SCALE_RATIO,
        }

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "corners": [list(c) for c in self.corners],
            "width_cm": self.width_cm,
            "height_cm": self.height_cm,
            "frame_size": list(self.frame_size),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArenaCalibration:
        return cls(
            corners=[tuple(c) for c in data["corners"]],
            width_cm=float(data.get("width_cm", 30.0)),
            height_cm=float(data.get("height_cm", 30.0)),
            frame_size=tuple(data.get("frame_size", (640, 480))),
        )
