"""Finding and loading the mask and depth models the 3-D lift uses.

Neither model ships with GLIDER. They live in ``~/.glider/models/lift3d`` (or the folder
named by ``$GLIDER_LIFT3D_MODELS``), or explicit paths are passed:

* ``mouse_seg.pt``      YOLO26n-seg mouse mask, distilled from SAM 2.1 (IoU 0.93 vs SAM).
  Copy it there; the canonical copy is in the 3D-latent prototype's ``models`` folder on
  the lab share.
* ``yolo26x-depth.pt``  Ultralytics' own monocular depth model (110 MB). Session Review
  and the command line download it there on first use (:func:`fetch_depth_model`).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MASK_MODEL = "mouse_seg.pt"
DEPTH_MODEL = "yolo26x-depth.pt"
# The first Ultralytics known to unpickle yolo26x-depth.pt (it needs DepthModel).
# GLIDER's own floor is this version too since the lift3d release; kept for older hosts.
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


def fetch_depth_model() -> Path | None:
    """The depth model in :func:`models_dir`, downloaded from Ultralytics if missing.

    None when it cannot be had (offline, or Ultralytics too old to know the asset); the
    lift then runs without depth. Not called by the pipeline itself, so tests and live
    runs never start a 110 MB download behind the caller's back.
    """
    target = models_dir() / DEPTH_MODEL
    if target.exists():
        return target
    try:
        check_depth_support()
        from ultralytics.utils.downloads import attempt_download_asset

        target.parent.mkdir(parents=True, exist_ok=True)
        got = Path(attempt_download_asset(target))
    except Exception:  # noqa: BLE001 - no depth is a supported mode, not an error
        logger.warning("could not download %s; the 3-D lift runs without depth", DEPTH_MODEL)
        return None
    return got if got.exists() else None


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
