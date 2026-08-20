"""Qt-free YOLO-pose inference, smoothing, and DLC-CSV interchange.

Ported from the yolo2pose project. cv2 / ultralytics / torch are lazy-imported
inside functions, so importing this package is cheap and dependency-light.
"""

from . import backend, batch, decode, dlc, spec, viz
from .backend import (
    OnnxPoseBackend,
    PoseBackend,
    UltralyticsBackend,
    load_pose_backend,
    preprocess_frame,
)
from .batch import (
    BatchEvent,
    BatchResult,
    EventKind,
    FilterSettings,
    discover_videos,
    dlc_output_path,
    raw_output_path,
    run_batch,
)
from .core import PoseCancelledError, PoseData, infer_video, pose_from_array
from .decode import decode_dlc_locref, decode_sleap_confmaps
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
from .spec import (
    PoseModelError,
    PoseModelMeta,
    PoseModelSpec,
    ensure_onnxruntime,
    identify_pose_model,
    is_onnx_model,
    read_pose_model_meta,
)

__all__ = [
    "PoseData",
    "PoseCancelledError",
    "infer_video",
    "pose_from_array",
    "batch",
    "BatchEvent",
    "BatchResult",
    "EventKind",
    "FilterSettings",
    "discover_videos",
    "dlc_output_path",
    "raw_output_path",
    "run_batch",
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
    # Pose backends: YOLO, DeepLabCut and SLEAP behind one protocol.
    "backend",
    "decode",
    "spec",
    "PoseBackend",
    "UltralyticsBackend",
    "OnnxPoseBackend",
    "load_pose_backend",
    "preprocess_frame",
    "decode_dlc_locref",
    "decode_sleap_confmaps",
    "PoseModelSpec",
    "PoseModelMeta",
    "PoseModelError",
    "identify_pose_model",
    "read_pose_model_meta",
    "is_onnx_model",
    "ensure_onnxruntime",
]
