"""Which keypoints the 3-D lift reads, found by name.

The prototype indexed exp-16's keypoints by position (ears 0-1, nose 2, body centre 3,
hips 4-5, tail base 6). exp-6 has the same seven parts in a different order (nose 0,
ears 1-2, ... ), so a positional port would silently read the wrong points on half the
lab's models. Every role here is looked up by name instead, with the spellings GLIDER
already meets in the wild (``center`` / ``body_center`` / ``center_body`` ...).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# exp-16's order, the model the prototype was built and tuned on.
EXP16_KEYPOINTS = [
    "left_ear",
    "right_ear",
    "nose",
    "body_center",
    "left_hip",
    "right_hip",
    "tail_base",
]

# Role -> accepted names, most specific first. Matching ignores case, '-' and ' '.
_ALIASES: dict[str, tuple[str, ...]] = {
    "nose": ("nose", "snout"),
    "left_ear": ("left_ear", "l_ear", "ear_left", "earl"),
    "right_ear": ("right_ear", "r_ear", "ear_right", "earr"),
    "center": ("body_center", "center_body", "centre_body", "body_centre", "center", "centre"),
    "left_hip": ("left_hip", "l_hip", "hip_left", "l_body", "left_body"),
    "right_hip": ("right_hip", "r_hip", "hip_right", "r_body", "right_body"),
    "tail_base": ("tail_base", "tailbase", "base_tail", "tail_root", "tail"),
}

# Spine points only the 10-keypoint model has (slots 7-9 after exp-16's seven).
# Optional for the lift; the neck is required for the back-arch measure.
_SPINE_ALIASES: dict[str, tuple[str, ...]] = {
    "neck": ("neck", "nape"),
    "withers": ("withers", "shoulders", "spine_front", "spine1"),
    "lumbar": ("lumbar", "spine_back", "spine2", "lower_back"),
}


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class KeypointRoles:
    """Indices into a model's keypoint array for each role the lift uses."""

    nose: int
    left_ear: int
    right_ear: int
    center: int
    left_hip: int
    right_hip: int
    tail_base: int
    neck: int | None = None
    withers: int | None = None
    lumbar: int | None = None

    @classmethod
    def from_names(cls, names: list[str]) -> KeypointRoles:
        normed = [_norm(n) for n in names]
        found: dict[str, int] = {}
        for role, aliases in _ALIASES.items():
            for alias in aliases:
                if alias in normed:
                    found[role] = normed.index(alias)
                    break
        missing = [r for r in _ALIASES if r not in found]
        if missing:
            raise ValueError(
                "the 3-D lift needs keypoints for "
                f"{', '.join(missing)}, but the model's names are {', '.join(names)}. "
                "Name them in the keypoint list (any of: "
                + "; ".join(f"{r}: {'/'.join(_ALIASES[r])}" for r in missing)
                + ")."
            )
        for role, aliases in _SPINE_ALIASES.items():
            for alias in aliases:
                if alias in normed:
                    found[role] = normed.index(alias)
                    break
        return cls(**found)

    def spine(self, xy: np.ndarray) -> np.ndarray | None:
        """Neck -> withers -> body centre -> lumbar -> tail base, ``(n, 2)``.

        Withers and lumbar are used when the model has them. None without a neck.
        """
        if self.neck is None:
            return None
        order = [self.neck, self.withers, self.center, self.lumbar, self.tail_base]
        return np.asarray(xy, float)[[i for i in order if i is not None], :2]

    def canonical(self, xy: np.ndarray) -> np.ndarray:
        """``(K, 2+)`` keypoints -> ``(7, 2)`` in exp-16 order, the order the maths uses."""
        order = [
            self.left_ear,
            self.right_ear,
            self.nose,
            self.center,
            self.left_hip,
            self.right_hip,
            self.tail_base,
        ]
        return np.asarray(xy, float)[order, :2]


# Indices into the canonical (exp-16) array.
L_EAR, R_EAR, NOSE, CENTER, L_HIP, R_HIP, TAIL = range(7)

ANCHOR_NAMES = ("tail_base", "hips", "body_center", "ears", "nose")


def anchor_points(xy: np.ndarray) -> np.ndarray:
    """Canonical keypoints -> the five spine anchors, tail base to nose."""
    return np.array(
        [
            xy[TAIL],
            (xy[L_HIP] + xy[R_HIP]) / 2,
            xy[CENTER],
            (xy[L_EAR] + xy[R_EAR]) / 2,
            xy[NOSE],
        ]
    )
