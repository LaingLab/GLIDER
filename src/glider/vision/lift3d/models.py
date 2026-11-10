"""Finding and loading the mask and depth models the 3-D lift uses.

Neither model ships with GLIDER. Put them in ``~/.glider/models/lift3d`` (or the folder
named by ``$GLIDER_LIFT3D_MODELS``), or pass explicit paths:

* ``mouse_seg.pt``      YOLO26n-seg mouse mask, distilled from SAM 2.1 (IoU 0.93 vs SAM)
* ``yolo26x-depth.pt``  Ultralytics monocular depth model

The canonical copies are in the 3D-latent prototype's ``models`` folder on the lab share.
"""

from __future__ import annotations

import os
from pathlib import Path

MASK_MODEL = "mouse_seg.pt"
DEPTH_MODEL = "yolo26x-depth.pt"
# The first Ultralytics known to unpickle yolo26x-depth.pt (it needs DepthModel);
# GLIDER's floor (8.4.136) cannot.
DEPTH_MIN_ULTRALYTICS = (8, 4, 192)


def models_dir() -> Path:
    env = os.environ.get("GLIDER_LIFT3D_MODELS")
    return Path(env) if env else Path.home() / ".glider" / "models" / "lift3d"


def find_model(name: str, explicit: str | Path | None = None) -> Path | None:
    """``explicit`` if given (must exist), else ``models_dir()/name`` if present, else None."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"3-D lift model not found: {p}")
        return p
    p = models_dir() / name
    return p if p.exists() else None


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split(".")[:3]:
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out)


def check_depth_support() -> None:
    """Raise with an install hint when Ultralytics is too old for the depth model."""
    import ultralytics

    have = _version_tuple(ultralytics.__version__)
    if have < DEPTH_MIN_ULTRALYTICS:
        need = ".".join(map(str, DEPTH_MIN_ULTRALYTICS))
        raise RuntimeError(
            f"the depth model needs ultralytics>={need}; this environment has "
            f'{ultralytics.__version__}. Upgrade with `uv pip install "ultralytics>={need}"`, '
            "or run the lift without depth (keypoint + outline surface only)."
        )


def load_yolo(path: Path):
    from ultralytics import YOLO

    return YOLO(str(path))
