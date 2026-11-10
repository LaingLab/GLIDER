"""Mouse 3-D posture from one top-down camera (work in progress).

Ported from the 3D-latent prototype. Given 2-D keypoints and the frame, it estimates a
height surface of the mouse above the floor and five spine anchor heights, plus a rear
weight from keypoint geometry. Qt-free; cv2 / ultralytics / torch load lazily.

Status: estimates, not measurements. Held-out rear F1 0.66 (180 labelled clips); wall
rears mostly missed; depth alone under-reads height, so the surface leans on an outline
dome and keypoint geometry. See ``posture`` for the numbers behind each constant.
"""

from .keypoints import ANCHOR_NAMES, EXP16_KEYPOINTS, KeypointRoles
from .pipeline import Lift3D, Lift3DConfig, Lift3DFrame

__all__ = [
    "ANCHOR_NAMES",
    "EXP16_KEYPOINTS",
    "KeypointRoles",
    "Lift3D",
    "Lift3DConfig",
    "Lift3DFrame",
]
