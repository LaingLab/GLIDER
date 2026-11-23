"""Live multi-animal identity: keep a stable name on each animal, frame by frame.

Pure numpy, Qt-free. One slot per named animal (plus, in continuity mode, one
per clicked-but-unnamed animal). Each frame the slots are matched to the
frame's detections with ``linear_sum_assignment``; a match beyond the travel
limit (``max_travel_px`` per frame since the slot was last observed) is
refused. Beyond the one-frame reach (``max_travel_px``), a hidden slot may
only take a detection that newly appeared this frame (none in the previous
frame within ``max_travel_px``), so a hidden animal's name never jumps to an
animal that was already in view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from glider.vision.cv_processor import Detection, TrackedObject

FLAG_ORDER = ("close", "stitched", "gap", "relabeled")
# Appearance mode: cost of a detection whose class disagrees with the slot's.
# One box diagonal of motion, so a single mislabelled frame is outvoted.
CLASS_MISMATCH_COST = 1.0
# Added to a hidden slot's row: when slots compete for too few detections, the
# one observed last frame wins a tie over one held at a stale position.
_HIDDEN_COST = 1.0
_BLOCKED = 1e9
_KP_MIN_CONF = 0.3


@dataclass
class _Slot:
    name: str | None
    cls: str | None = None  # appearance mode: the class this name should carry
    pos: np.ndarray | None = None  # last observed centroid
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    box: tuple[int, int, int, int] = (0, 0, 0, 0)
    kps: np.ndarray | None = None
    det_cls: str = ""
    conf: float = 0.0
    hidden: int = 0  # frames since last observed
    age: int = 0
    mismatch: int = 0  # consecutive frames matched to the wrong class


def _kps(d: Detection) -> np.ndarray | None:
    k = getattr(d, "_keypoints", None)
    return None if k is None else np.asarray(k, dtype=float)


class IdentityAssigner:
    def __init__(
        self,
        names: list[str],
        *,
        mode: str,
        class_names: list[str | None] | None = None,
        max_travel_px: float = 40.0,
        separation_px: float = 60.0,
        relabel_frames: int = 15,
    ):
        self.names = list(names)
        self.mode = mode
        self.class_names = list(class_names) if class_names else [None] * len(self.names)
        self.max_travel_px = max_travel_px
        self.separation_px = separation_px
        self.relabel_frames = relabel_frames
        self.reset()

    def reset(self) -> None:
        self._prev: list[tuple[int, int]] = []  # last frame's detection centroids
        if self.mode == "appearance":
            self._slots = [_Slot(n, c) for n, c in zip(self.names, self.class_names, strict=True)]
        else:
            self._slots = []

    @property
    def ready(self) -> bool:
        return self.mode == "appearance" or bool(self._slots)

    def seed(
        self,
        detections: list[Detection],
        placements: dict[str, tuple[float, float]],
        anonymous: list[tuple[float, float]] = (),
    ) -> list[str]:
        """Bind each named / anonymous click to its nearest detection.

        Returns the names that could not be bound; on any, nothing is seeded.
        """
        clicks = [placements.get(n) for n in self.names] + list(anonymous)
        rows = [i for i, c in enumerate(clicks) if c is not None]
        bound: dict[int, Detection] = {}
        if rows and detections:
            here = np.array([d.centroid for d in detections], dtype=float)
            there = np.array([clicks[i] for i in rows], dtype=float)
            cost = np.linalg.norm(there[:, None, :] - here[None, :, :], axis=2)
            for r, c in zip(*linear_sum_assignment(cost), strict=True):
                bound[rows[r]] = detections[c]
        unbound = [n for i, n in enumerate(self.names) if i not in bound]
        if unbound:
            self._slots = []
            return unbound
        # Unbound anonymous clicks (more clicks than detections) are dropped.
        n = len(self.names)
        self._slots = []
        self._prev = []
        for i in sorted(bound):
            slot = _Slot(self.names[i] if i < n else None)
            self._observe(slot, bound[i], moved=False)
            self._slots.append(slot)
        return []

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        n = len(self.names)
        if not self.ready:
            return [self._extra(d, n + j) for j, d in enumerate(detections)]

        reach = self.max_travel_px
        new = [all(math.dist(d.centroid, p) > reach for p in self._prev) for d in detections]
        self._prev = [d.centroid for d in detections]
        slots = self._slots
        raw = np.array([[self._cost(s, d) for d in detections] for s in slots]).reshape(
            len(slots), len(detections)
        )
        cost = raw.copy()
        for i, s in enumerate(slots):
            if s.pos is None:
                cost[i] += _HIDDEN_COST
                continue
            limit = reach * (s.hidden + 1)
            pred = self._predict(s)
            for j, d in enumerate(detections):
                far = math.dist(d.centroid, pred)
                if far > limit or (far > reach and not new[j]):
                    cost[i, j] = _BLOCKED
            if s.hidden:
                cost[i] += _HIDDEN_COST
        match = {
            int(r): int(c)
            for r, c in zip(*linear_sum_assignment(cost), strict=True)
            if cost[r, c] < _BLOCKED
        }

        relabeled = set()
        if self.mode == "appearance":
            for i, s in enumerate(slots):
                j = match.get(i)
                wrong = s.cls is not None and j is not None and detections[j].class_name != s.cls
                s.mismatch = s.mismatch + 1 if wrong else 0
                if s.mismatch <= self.relabel_frames:
                    continue
                taken = {c for k, c in match.items() if k != i}
                own = [
                    j for j, d in enumerate(detections) if d.class_name == s.cls and j not in taken
                ]
                if own:
                    match[i] = min(own, key=lambda j: raw[i, j])
                    relabeled.add(i)
                    s.mismatch = 0

        out: list[TrackedObject] = []
        stitched = set()
        for i, s in enumerate(slots):
            if i in match:
                if s.pos is not None and s.hidden:
                    stitched.add(i)
                self._observe(s, detections[match[i]], moved=i not in relabeled)
            else:
                s.hidden += 1
            out.append(self._slot_object(s, i))
        used = set(match.values())
        extras = [d for j, d in enumerate(detections) if j not in used]
        out += [self._extra(d, len(slots) + k) for k, d in enumerate(extras)]

        # ponytail: O(n^2) proximity scan; fine for a handful of animals.
        for i, o in enumerate(out[: len(slots)]):
            causes = set()
            if o.visible:
                if any(
                    p is not o
                    and p.age  # skip never-seen slots
                    and math.dist(p.centroid, o.centroid) <= self.separation_px
                    for p in out
                ):
                    causes.add("close")
                if i in stitched:
                    causes.add("stitched")
                if i in relabeled:
                    causes.add("relabeled")
            else:
                causes.add("gap")
            o.identity_flag = "+".join(c for c in FLAG_ORDER if c in causes)
        return out

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _predict(s: _Slot) -> np.ndarray:
        # Hidden: freeze at the last observation (OC-SORT), never extrapolate.
        return s.pos if s.hidden else s.pos + s.vel

    def _cost(self, s: _Slot, d: Detection) -> float:
        mismatch = (
            CLASS_MISMATCH_COST
            if self.mode == "appearance" and s.cls is not None and d.class_name != s.cls
            else 0.0
        )
        if s.pos is None:
            return mismatch
        pred = self._predict(s)
        dist = math.dist(d.centroid, pred)
        kd = _kps(d)
        if s.kps is not None and kd is not None and kd.shape == s.kps.shape:
            ok = np.isfinite(kd[:, :2]).all(1) & np.isfinite(s.kps[:, :2]).all(1)
            if kd.shape[1] >= 3:
                ok &= (kd[:, 2] >= _KP_MIN_CONF) & (s.kps[:, 2] >= _KP_MIN_CONF)
            if ok.any():
                shifted = s.kps[ok, :2] + (pred - s.pos)
                dist = float(np.linalg.norm(kd[ok, :2] - shifted, axis=1).mean())
        _, _, w, h = s.box
        _, _, dw, dh = d.bbox
        size = abs(math.log(max(dw * dh, 1) / max(w * h, 1)))
        return dist / max(math.hypot(w, h), 1.0) + size + mismatch

    @staticmethod
    def _observe(s: _Slot, d: Detection, *, moved: bool) -> None:
        c = np.asarray(d.centroid, dtype=float)
        s.vel = c - s.pos if moved and s.pos is not None and not s.hidden else np.zeros(2)
        s.pos, s.box, s.kps = c, d.bbox, _kps(d)
        s.det_cls, s.conf = d.class_name, d.confidence
        s.hidden = 0
        s.age += 1

    def _slot_object(self, s: _Slot, track_id: int) -> TrackedObject:
        pos = (0, 0) if s.pos is None else (int(s.pos[0]), int(s.pos[1]))
        return TrackedObject(
            track_id=track_id,
            class_name=s.det_cls or (s.cls or ""),
            bbox=s.box,
            confidence=s.conf,
            centroid=pos,
            age=s.age,
            disappeared=s.hidden,
            velocity=float(np.hypot(*s.vel)) if not s.hidden else 0.0,
            keypoints=s.kps,
            animal=s.name,
            visible=not s.hidden,
        )

    @staticmethod
    def _extra(d: Detection, track_id: int) -> TrackedObject:
        return TrackedObject(
            track_id=track_id,
            class_name=d.class_name,
            bbox=d.bbox,
            confidence=d.confidence,
            centroid=d.centroid,
            age=1,
            keypoints=_kps(d),
        )
