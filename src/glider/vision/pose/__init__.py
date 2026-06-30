"""Qt-free YOLO-pose inference, smoothing, and DLC-CSV interchange.

Ported from the yolo2pose project. cv2 / ultralytics / torch are lazy-imported
inside functions, so importing this package is cheap and dependency-light.
"""

from . import dlc, viz
from .core import PoseData, infer_video, pose_from_array
from .device import (
    format_gpu_info,
    gpu_info,
    require_gpu_or_raise,
    resolve_device,
)
from .filtering import (
    interpolate_gaps,
    mask_low_confidence,
    median_filter,
    smooth,
)

__all__ = [
    "PoseData",
    "infer_video",
    "pose_from_array",
    "resolve_device",
    "require_gpu_or_raise",
    "gpu_info",
    "format_gpu_info",
    "interpolate_gaps",
    "mask_low_confidence",
    "median_filter",
    "smooth",
    "dlc",
    "viz",
]
