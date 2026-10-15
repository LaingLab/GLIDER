"""How far each animal is from the nest, every frame.

A pup-retrieval session asks one spatial question over and over: how far is
this animal from the nest? The nest is an ordinary zone drawn in the zone
editor, so "in or out" was already answerable. "How far" was not.

The answer here is a *signed* distance to the nest's edge: positive outside,
negative inside (how deep), zero on the boundary. One column then carries both
the distance and the membership, and it does not depend on where the centre of
an irregular nest happens to fall.

The nest does not move within a trial, so the whole frame's distances are
computed once, as a field, and every animal in every frame is then a lookup.
A pup mask is a ``min`` over its pixels: the nearest part of the pup, so a pup
touching the nest reads 0 however far its centroid is.

Everything is in pixels. Converting to centimetres is the caller's, because
the scale lives on the session, not the zone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ["NestField", "nest_field"]


@dataclass(frozen=True)
class NestField:
    """Signed pixel distance to the nest edge for every pixel of a frame."""

    signed: np.ndarray  # (H, W) float32; > 0 outside, < 0 inside
    centre: tuple[float, float]  # nest centroid, pixels
    # (M, 2) int boundary pixels, for drawing a line from an animal to the nest
    edge: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2), int))

    def at(self, xy: np.ndarray) -> np.ndarray:
        """Signed distance at each ``(..., 2)`` pixel position.

        A non-finite position (a dropped frame) gives NaN, never a distance:
        a missing pup is not "far from the nest". Positions off the frame are
        clamped to its edge, which is where the field was last measured.
        """
        xy = np.asarray(xy, dtype=float)
        out = np.full(xy.shape[:-1], np.nan)
        ok = np.isfinite(xy).all(axis=-1)
        h, w = self.signed.shape
        cols = np.clip(np.rint(xy[..., 0][ok]), 0, w - 1).astype(int)
        rows = np.clip(np.rint(xy[..., 1][ok]), 0, h - 1).astype(int)
        out[ok] = self.signed[rows, cols]
        return out

    def nearest_edge(self, xy: np.ndarray) -> np.ndarray:
        """The nest-boundary pixel nearest *xy*, for drawing a line to the nest."""
        d = np.hypot(self.edge[:, 0] - xy[0], self.edge[:, 1] - xy[1])
        return self.edge[int(np.argmin(d))].astype(float)

    def from_centre(self, xy: np.ndarray) -> np.ndarray:
        """Unsigned pixel distance from the nest centroid, NaN where *xy* is."""
        xy = np.asarray(xy, dtype=float)
        return np.hypot(xy[..., 0] - self.centre[0], xy[..., 1] - self.centre[1])

    def of_polygon(self, polygon: np.ndarray) -> float:
        """Signed distance of the nearest pixel inside a mask *polygon*.

        ``polygon`` is ``(N, 2)`` pixels, the shape Ultralytics hands back in
        ``result.masks.xy``. Filled rather than sampled at its vertices: a nest
        corner can poke into a pup without touching any vertex of its outline.
        """
        polygon = np.asarray(polygon, dtype=float)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            return float("nan")
        h, w = self.signed.shape
        x0, y0 = np.clip(np.floor(polygon.min(axis=0)).astype(int), 0, [w - 1, h - 1])
        x1, y1 = np.clip(np.ceil(polygon.max(axis=0)).astype(int), 0, [w - 1, h - 1])
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon - [x0, y0]).astype(np.int32)], 1)
        inside = self.signed[y0 : y1 + 1, x0 : x1 + 1][mask.astype(bool)]
        return float(inside.min()) if inside.size else float("nan")


def nest_field(zone, width: int, height: int) -> NestField:
    """The signed distance field of *zone* on a ``width`` × ``height`` frame.

    Accurate to about a pixel: ``cv2.distanceTransform`` measures between
    pixel centres, so a point just inside the edge reads -1 rather than -0.3.
    At 20 px/cm that is half a millimetre, well under a pup's mask noise.
    """
    from glider.vision.zones import ZoneShape

    mask = np.zeros((height, width), dtype=np.uint8)
    verts = [(vx * width, vy * height) for vx, vy in zone.vertices]
    if zone.shape == ZoneShape.CIRCLE and len(verts) >= 2:
        (cx, cy), (rx, ry) = verts[0], verts[1]
        cv2.circle(mask, (round(cx), round(cy)), round(np.hypot(rx - cx, ry - cy)), 1, -1)
    elif zone.shape == ZoneShape.RECTANGLE and len(verts) >= 2:
        (x1, y1), (x2, y2) = verts[0], verts[1]
        cv2.rectangle(mask, (round(x1), round(y1)), (round(x2), round(y2)), 1, -1)
    elif zone.shape == ZoneShape.POLYGON and len(verts) >= 3:
        cv2.fillPoly(mask, [np.rint(verts).astype(np.int32)], 1)
    if not mask.any():
        raise ValueError(f"zone {zone.name!r} covers no pixels of a {width}x{height} frame")

    outside = cv2.distanceTransform(1 - mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    m = cv2.moments(mask, binaryImage=True)
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.concatenate([c[:, 0, :] for c in found]) if found else np.zeros((0, 2), int)
    return NestField(
        signed=(outside - inside).astype(np.float32),
        centre=(m["m10"] / m["m00"], m["m01"] / m["m00"]),
        edge=edge,
    )
