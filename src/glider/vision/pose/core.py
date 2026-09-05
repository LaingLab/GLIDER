"""Core PoseData container + Ultralytics YOLO inference.

PoseData is the format-agnostic representation that every converter consumes.
It holds:
    xy            — float array, shape (n_frames, n_keypoints, 2)
    confidence    — float array, shape (n_frames, n_keypoints), in [0, 1]
    keypoint_names — ordered list of body part names (length n_keypoints)
    fps           — frames per second (used by movement / NWB)
    source        — string label propagated to "scorer" in DLC output
    metadata      — free-form dict for provenance (model path, video path, ...)

Single animal in v1. The shape leaves room for a future
(n_individuals, n_frames, n_keypoints, 3) layout, but every public function
in this module operates on one mouse per video.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from glider.vision.arena import ArenaCalibration
    from glider.vision.arena_gate import ArenaGateSettings

logger = logging.getLogger(__name__)


class PoseCancelledError(RuntimeError):
    """Raised when a caller's ``cancel_cb`` asks inference to stop early.

    Distinct from a failure: callers treat this as "the operator stopped it",
    not "this video is broken".
    """


@dataclass
class PoseData:
    """Frame-aligned pose container (single animal)."""

    xy: np.ndarray
    confidence: np.ndarray
    keypoint_names: list[str]
    fps: float = 30.0
    source: str = "ultralytics_yolo"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xy = np.asarray(self.xy, dtype=float)
        self.confidence = np.asarray(self.confidence, dtype=float)
        self.keypoint_names = list(self.keypoint_names)
        self._validate()

    def _validate(self) -> None:
        if self.xy.ndim != 3 or self.xy.shape[-1] != 2:
            raise ValueError(f"xy must have shape (n_frames, n_keypoints, 2); got {self.xy.shape}")
        if self.confidence.shape != self.xy.shape[:2]:
            raise ValueError(
                f"confidence shape {self.confidence.shape} must match xy[:, :, 0] "
                f"shape {self.xy.shape[:2]}"
            )
        if len(self.keypoint_names) != self.xy.shape[1]:
            raise ValueError(
                f"keypoint_names length ({len(self.keypoint_names)}) must equal "
                f"n_keypoints ({self.xy.shape[1]})"
            )
        if len(set(self.keypoint_names)) != len(self.keypoint_names):
            raise ValueError("keypoint_names must be unique")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive; got {self.fps}")

    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]

    @property
    def n_keypoints(self) -> int:
        return self.xy.shape[1]

    def kp_index(self, name: str) -> int:
        """Look up a body part by name."""
        try:
            return self.keypoint_names.index(name)
        except ValueError as e:
            raise KeyError(f"unknown keypoint {name!r}; known: {self.keypoint_names}") from e

    def slice_frames(self, start: int, stop: int) -> PoseData:
        """Return a PoseData restricted to [start, stop)."""
        return PoseData(
            xy=self.xy[start:stop].copy(),
            confidence=self.confidence[start:stop].copy(),
            keypoint_names=list(self.keypoint_names),
            fps=self.fps,
            source=self.source,
            metadata=dict(self.metadata),
        )

    def copy(self) -> PoseData:
        return PoseData(
            xy=self.xy.copy(),
            confidence=self.confidence.copy(),
            keypoint_names=list(self.keypoint_names),
            fps=self.fps,
            source=self.source,
            metadata=dict(self.metadata),
        )


def pose_from_array(
    xy: np.ndarray,
    confidence: np.ndarray,
    keypoint_names: Iterable[str],
    *,
    fps: float = 30.0,
    source: str = "ultralytics_yolo",
    metadata: dict[str, Any] | None = None,
) -> PoseData:
    """Wrap raw numpy arrays into a PoseData (convenience constructor)."""
    return PoseData(
        xy=np.asarray(xy),
        confidence=np.asarray(confidence),
        keypoint_names=list(keypoint_names),
        fps=fps,
        source=source,
        metadata=dict(metadata) if metadata else {},
    )


def _video_fps(video_path: str | Path) -> float:
    """Read FPS from a video file via OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError(f"unable to read FPS from {video_path}")
    return fps


def _infer_video_backend(
    spec,
    video_path: Path,
    *,
    conf: float,
    fps: float | None,
    source: str | None,
    progress: bool,
    progress_cb: Callable[[int, int], None] | None,
    cancel_cb: Callable[[], bool] | None,
) -> PoseData:
    """Frame-by-frame inference for DeepLabCut / SLEAP ONNX models.

    Ultralytics streams a whole video itself; these backends take one frame at
    a time, so decoding happens here with OpenCV. Progress and cancellation
    semantics match the ultralytics path exactly: ``cancel_cb`` is polled
    before each frame is decoded, and ``progress_cb`` fires once per frame with
    a total of ``0`` when the container reports no frame count.
    """
    import cv2

    from glider.vision.pose.backend import load_pose_backend
    from glider.vision.video_source import video_resolution

    backend = load_pose_backend(spec)
    keypoint_names = list(backend.keypoint_names)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    if fps is None:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not np.isfinite(fps):
            cap.release()
            raise ValueError(f"unable to read FPS from {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=total_frames or None, desc=f"{spec.kind.upper()} inference")
        except Exception:
            bar = None

    xy_rows: list[np.ndarray] = []
    conf_rows: list[np.ndarray] = []
    try:
        while True:
            if cancel_cb is not None and cancel_cb():
                raise PoseCancelledError(f"inference cancelled after {len(xy_rows)} frames")
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            kp_xy, kp_conf = backend.predict(frame)
            # Same masking rule as the YOLO path: below-threshold keypoints are
            # NaN so downstream feature maths never uses them.
            kp_xy = np.asarray(kp_xy, dtype=float).copy()
            kp_xy[np.asarray(kp_conf, dtype=float) < conf] = np.nan
            xy_rows.append(kp_xy)
            conf_rows.append(np.asarray(kp_conf, dtype=float))
            if bar is not None:
                bar.update(1)
            if progress_cb is not None:
                progress_cb(len(xy_rows), total_frames)
    finally:
        cap.release()
        if bar is not None:
            bar.close()
        backend.close()

    n_kpts = len(keypoint_names)
    xy_arr = np.stack(xy_rows, axis=0) if xy_rows else np.zeros((0, n_kpts, 2))
    conf_arr = np.stack(conf_rows, axis=0) if conf_rows else np.zeros((0, n_kpts))

    return PoseData(
        xy=xy_arr,
        confidence=conf_arr,
        keypoint_names=keypoint_names,
        fps=fps,
        source=source or spec.source_label,
        metadata={
            "model_path": str(spec.model_path),
            "video_path": str(video_path),
            "conf_threshold": conf,
            "device": "cpu",
            "backend": spec.kind,
            "resolution": video_resolution(video_path),
        },
    )


def _pick_candidate(result, confidences, arena, settings, resolution) -> int:
    """Index of the detection to keep.

    Plain ``argmax`` when there is no arena, so that path stays exactly what it
    was. With one, the highest-confidence candidate whose keypoints clear
    ``min_inside_fraction`` wins -- and if none do, ``argmax`` again.

    That fallback is what makes this a re-ranking rather than a filter: it can
    replace a bad pick with a good one but can never turn a frame that had a
    usable detection into a dropout. Blanking is the gate's job, downstream.

    Keypoint confidences are passed to :func:`inside_fraction`, not just ``xy``,
    for the reason its docstring gives: raw Ultralytics output pads unlocalized
    keypoints with ``(0, 0)`` at confidence 0, a finite pixel in the frame's
    top-left corner that is outside every arena. Judging on ``xy`` alone would
    score a correct detection with a few pads below quorum and hand the frame
    back to the bench-floor blob -- the exact trap this exists to close.
    """
    if arena is None:
        return int(confidences.argmax())

    from glider.vision.arena_gate import ArenaGateSettings, inside_fraction

    # run_batch gates on `gate is not None and arena is not None`, so an arena
    # can arrive here with no settings.
    settings = settings or ArenaGateSettings()
    # video_resolution returns None for a header it cannot read. Fall back to
    # the frame the corners were clicked on -- the same last resort
    # arena_gate._resolve_resolution uses. Re-ranking one candidate must never
    # be the thing that kills a multi-hour batch.
    resolution = resolution or arena.frame_size
    keypoint_conf = result.keypoints.conf
    for index in np.argsort(confidences)[::-1]:
        xy = result.keypoints.xy[index].cpu().numpy()
        cf = keypoint_conf[index].cpu().numpy() if keypoint_conf is not None else np.ones(len(xy))
        if inside_fraction(arena, xy, cf, resolution, settings) >= settings.min_inside_fraction:
            return int(index)
    return int(confidences.argmax())


def infer_video(
    model_path: str | Path,
    video_path: str | Path,
    keypoint_names: Iterable[str] | None = None,
    *,
    conf: float = 0.25,
    fps: float | None = None,
    source: str | None = None,
    device: str | int | None = None,
    require_gpu: bool = False,
    verbose: bool = False,
    progress: bool = True,
    echo_device: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
    arena: ArenaCalibration | None = None,
    gate_settings: ArenaGateSettings | None = None,
) -> PoseData:
    """Run an Ultralytics YOLO-pose model over a video and return PoseData.

    Parameters
    ----------
    model_path
        Path to the trained ``.pt`` weights.
    video_path
        Path to the video file.
    keypoint_names
        Ordered list of body part names matching the model's training order.
        **The order must match exactly** — see README warning.
    conf
        Detection confidence threshold (passed to Ultralytics).
    fps
        Frames per second. If ``None``, read from the video.
    source
        Label written into the "scorer" field of the DLC output. Defaults to
        ``"yolo_<modelstem>"``.
    device
        Torch device. Accepts ``None`` / ``"auto"`` (best available),
        ``"cpu"``, ``"cuda"``, ``"cuda:N"``, an int index, or ``"mps"``.
        Validated up front via :func:`glider.vision.pose.device.resolve_device`, so
        a misconfigured CUDA install raises immediately rather than silently
        running on CPU.
    require_gpu
        If True, raise ``RuntimeError`` when the resolved device is CPU.
        Use this for production runs where a silent CPU fallback would mean
        a multi-hour job.
    verbose
        Forwarded to Ultralytics.
    progress
        Show a tqdm progress bar over frames.
    echo_device
        Print the resolved device + (for CUDA) device name before inference
        starts. Helpful for confirming GPU usage in logs.
    progress_cb
        Called once per decoded frame as ``progress_cb(frames_done, total)``.
        ``total`` is ``0`` when OpenCV cannot report a frame count, which
        callers must render as indeterminate rather than as "zero frames".
        Intended for GUI progress bars; pass ``progress=False`` alongside it.
    cancel_cb
        Polled once per frame. Returning ``True`` aborts the run and raises
        :class:`PoseCancelledError`. Because Ultralytics streams results per frame,
        a cancel lands within one frame even on a multi-hour video.
    arena, gate_settings
        Re-rank multi-detection frames so an in-arena candidate beats a more
        confident one outside it. A re-ranking, not a filter — see
        :func:`_pick_candidate`. No-op on the DLC/SLEAP backends, which emit
        one detection per frame. Blanking remains the gate's job, downstream.

    Returns
    -------
    PoseData

    Raises
    ------
    PoseCancelledError
        If ``cancel_cb`` returned ``True`` before the video finished.

    Notes
    -----
    Frames where no animal is detected are written as NaN xy with confidence 0.
    For multi-detection frames, the highest box-confidence detection is kept
    (single-animal v1).
    """
    from glider.vision.pose.spec import PoseModelError, identify_pose_model

    model_path = Path(model_path)
    video_path = Path(video_path)

    # DeepLabCut / SLEAP take a different route: they run one frame at a time
    # through onnxruntime and carry their own keypoint names, so neither the
    # ultralytics streaming loop nor the caller's names apply.
    spec = identify_pose_model(model_path)
    if spec.kind != "yolo":
        # Logged here, not inside _infer_video_backend, which is never handed
        # the arena: one detection per frame means there is nothing to re-rank.
        if arena is not None:
            logger.info(
                "%s yields one detection per frame, so there are no candidates "
                "to re-rank; the arena still gates downstream",
                spec.kind,
            )
        return _infer_video_backend(
            spec,
            video_path,
            conf=conf,
            fps=fps,
            source=source,
            progress=progress,
            progress_cb=progress_cb,
            cancel_cb=cancel_cb,
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "Ultralytics is required for infer_video(). " "Install with: pip install glider[vision]"
        ) from e

    if not keypoint_names:
        raise PoseModelError(
            f"{model_path.name} is a YOLO checkpoint, which does not record "
            "body-part names. Pass keypoint_names in the model's training order."
        )
    keypoint_names = list(keypoint_names)
    n_kpts = len(keypoint_names)

    if fps is None:
        fps = _video_fps(video_path)
    if source is None:
        source = f"yolo_{model_path.stem}"

    from glider.vision.pose.device import gpu_info, resolve_device
    from glider.vision.video_source import video_resolution

    resolved_device = resolve_device(device, require_gpu=require_gpu)
    # Read once, before the loop: candidate re-ranking needs the frame size to
    # project keypoints into the arena, and the metadata block needs the same
    # value. A per-frame header read would cost a file open on every frame.
    resolution = video_resolution(video_path)

    if echo_device:
        if resolved_device.startswith("cuda"):
            info = gpu_info()
            idx = int(resolved_device.split(":", 1)[1]) if ":" in resolved_device else 0
            dev_name = "?"
            for d in info.get("cuda_devices", []):
                if d["index"] == idx:
                    dev_name = f"{d['name']} ({d['total_memory_gb']} GB)"
                    break
            print(f"[glider.pose] device = {resolved_device}  ({dev_name})")
        else:
            print(f"[glider.pose] device = {resolved_device}")

    model = YOLO(str(model_path))
    model.to(resolved_device)

    results_iter = model.predict(
        source=str(video_path),
        stream=True,
        conf=conf,
        device=resolved_device,
        verbose=verbose,
    )

    # Probe the frame count once; both the tqdm bar and progress_cb need it.
    # 0 means "unknown" — some containers don't carry a reliable count.
    total_frames = 0
    if progress or progress_cb is not None:
        try:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap.release()
        except Exception:
            total_frames = 0

    if progress:
        try:
            from tqdm import tqdm

            results_iter = tqdm(results_iter, total=total_frames or None, desc="YOLO inference")
        except Exception:
            pass

    xy_rows: list[np.ndarray] = []
    conf_rows: list[np.ndarray] = []

    for r in results_iter:
        # Checked before decoding so a cancel doesn't pay for one more frame.
        if cancel_cb is not None and cancel_cb():
            raise PoseCancelledError(f"inference cancelled after {len(xy_rows)} frames")

        if (
            getattr(r, "keypoints", None) is None
            or r.keypoints.xy is None
            or r.keypoints.xy.shape[0] == 0
        ):
            xy_rows.append(np.full((n_kpts, 2), np.nan))
            conf_rows.append(np.zeros(n_kpts))
        else:
            # Highest-confidence detection on this frame — preferring one
            # inside the arena, when there is an arena to prefer.
            if r.boxes is not None and r.boxes.conf is not None:
                confidences = r.boxes.conf.cpu().numpy()
                best = _pick_candidate(r, confidences, arena, gate_settings, resolution)
            else:
                best = 0

            xy = r.keypoints.xy[best].cpu().numpy()
            if r.keypoints.conf is not None:
                cf = r.keypoints.conf[best].cpu().numpy()
            else:
                cf = np.ones(n_kpts)

            if xy.shape[0] != n_kpts:
                raise ValueError(
                    f"model emitted {xy.shape[0]} keypoints but {n_kpts} names "
                    f"were supplied — check `keypoint_names` ordering against "
                    f"the training data.yaml"
                )

            xy_rows.append(np.asarray(xy, dtype=float))
            conf_rows.append(np.asarray(cf, dtype=float))

        # Single loop tail: undetected frames count toward progress too.
        if progress_cb is not None:
            progress_cb(len(xy_rows), total_frames)

    xy_arr = np.stack(xy_rows, axis=0) if xy_rows else np.zeros((0, n_kpts, 2))
    conf_arr = np.stack(conf_rows, axis=0) if conf_rows else np.zeros((0, n_kpts))

    return PoseData(
        xy=xy_arr,
        confidence=conf_arr,
        keypoint_names=keypoint_names,
        fps=fps,
        source=source,
        metadata={
            "model_path": str(model_path),
            "video_path": str(video_path),
            "conf_threshold": conf,
            "device": resolved_device,
            # Pose coordinates are pixels; a viewer without the video needs
            # the canvas they were measured on to place them.
            "resolution": resolution,
        },
    )
