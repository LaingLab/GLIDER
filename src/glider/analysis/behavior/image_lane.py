"""The image lane: an embedding of the animal crop, beside the keypoint features.

Keypoints say where the body parts are; they cannot say what the body looks
like. A rear and a hunched sit can put seven keypoints in nearly the same
place, and grooming forepaws are not tracked at all. The image lane adds what
the camera sees around the animal, in four steps per frame:

1. **Crop.** A ``crop_px`` square around the body centre, spanning
   ``crop_body_lengths`` body lengths (2.2 ≈ the 20 cm crop of the research
   pipeline for an adult mouse). Both the centre and the scale are *causal*:
   the centre carries forward over undetected frames and the scale is a
   running median of the body length over the frames seen so far, so the live
   path and an offline run crop every frame identically.
2. **Mask.** A YOLO segmentation model (``mouse_seg.pt``) masks the animal:
   the mask under the crop centre, else the largest one.
3. **Encode.** 19 mask-shape descriptors; DINOv2 ViT-S/14 on the masked crop
   (CLS + mean patch token, 768 values); and the neck of a YOLO pose model
   (``mmv2.pt`` layer 19, 128 channels) pooled inside the mask.
4. **Reduce and window.** The two learned embeddings are standardised and
   projected to ``dims`` principal components fitted on the *training*
   sessions only (:class:`ImageLaneArtifact`, stored in the model bundle).
   Each frame then gets its value, the mean and std over a ``window``-frame
   window, and its first difference.

Steps 1-3 are the expensive part and are cached per session
(:func:`load_or_compute_raw`), exactly as the motion lane caches its
features. Steps 4 are cheap, and the windows are computed by one function
(:func:`window_lane`) for training, offline scoring and the live stream, so
the three cannot drift apart.

Window direction. ``center=False`` (the default) is past-only: frame ``f``
uses frames ``f - window + 1 .. f``. That is the variant the manuscript calls
LIVE, and the only one the live pipeline accepts. ``center=True`` reproduces
the research's offline-best numbers but reads frames after ``f``.

Lane columns are named ``img_*`` and are *NaN-tolerant*: a frame whose mask
was missed still has its keypoint features, and LightGBM routes the missing
image values down its default branch, as the research pipeline did. Rows are
only dropped (or scored blank) for NaN in the keypoint features — see
:func:`required_columns`.

Reference: ``Data/behavior_embedding_data/scripts/pm_raster.py``
(``img_lane``/``win``) and ``pm_extract.py`` in the GLIDER manuscript folder.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Every image-lane column starts with this. Columns that do NOT are the
#: keypoint features, whose NaN still blanks a row.
IMAGE_LANE_PREFIX = "img_"

#: Number of mask-shape descriptors (:func:`shape_features`).
N_SHAPE = 19

#: Nominal width of the crop in cm, used only to put the shape descriptors in
#: the units the research pipeline used. The crop is scaled by body length,
#: so these are "cm for an adult mouse", not a calibration.
NOMINAL_CROP_CM = 20.0

#: Running-median history for the crop scale, in frames (30 s at 30 fps).
SCALE_HISTORY = 900

#: Stats appended to each per-frame lane value, in column order.
LANE_STATS: tuple[str, ...] = ("", "_m", "_s", "_d")

CACHE_DIR = Path("image_lane_cache")

#: Where :func:`default_config` looks for ``mouse_seg.pt`` and ``mmv2.pt``
#: unless ``GLIDER_IMAGE_LANE_MODELS`` names another folder.
DEFAULT_MODEL_DIR = Path.home() / ".glider" / "models" / "image_lane"
SEG_MODEL_NAME = "mouse_seg.pt"
NECK_MODEL_NAME = "mmv2.pt"

_BODY_KERNEL = (9, 9)  # opening that strips the tail before measuring body area
_TAIL_KERNEL = (17, 17)  # opening that strips the tail before the centre width
_GREY = (128, 128, 128)
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)
#: CLS + mean patch token width of each DINOv2 backbone.
_DINO_WIDTH = {
    "dinov2_vits14": 768,
    "dinov2_vitb14": 1536,
    "dinov2_vitl14": 2048,
    "dinov2_vitg14": 3072,
}


@dataclass(frozen=True)
class ImageLaneConfig:
    """Everything that decides what the image lane computes.

    ``seg_model`` and ``neck_model`` are paths to Ultralytics ``.pt`` files
    (the lab's ``mouse_seg.pt`` and ``mmv2.pt``). DINOv2 comes from
    ``torch.hub`` (downloaded once, then cached).
    """

    seg_model: str
    neck_model: str
    neck_layer: int = 19
    dino_repo: str = "facebookresearch/dinov2"
    dino_model: str = "dinov2_vits14"
    crop_px: int = 320
    crop_body_lengths: float = 2.2
    #: Keypoint the crop centres on; ``None`` = centroid of the visible ones.
    center_keypoint: int | None = None
    #: Principal components kept per learned embedding.
    dims: int = 32
    #: Frames in the lane's rolling window.
    window: int = 5
    #: Centred window (offline only) instead of past-only.
    center: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ImageLaneConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def cache_tag(self) -> str:
        """Short hash of everything that changes the raw (pre-PCA) embedding."""
        parts = [
            _file_ident(self.seg_model),
            _file_ident(self.neck_model),
            str(self.neck_layer),
            self.dino_repo,
            self.dino_model,
            str(self.crop_px),
            f"{self.crop_body_lengths:g}",
            str(self.center_keypoint),
        ]
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


def default_config(**overrides) -> ImageLaneConfig:
    """The manuscript's image lane, with the two YOLO models from the model folder.

    The folder is ``$GLIDER_IMAGE_LANE_MODELS`` or :data:`DEFAULT_MODEL_DIR`.
    Raises ``FileNotFoundError`` naming the folder when a model is missing, so
    a training run fails before it decodes anything.
    """
    folder = Path(os.environ.get("GLIDER_IMAGE_LANE_MODELS") or DEFAULT_MODEL_DIR)
    seg, neck = folder / SEG_MODEL_NAME, folder / NECK_MODEL_NAME
    missing = [p.name for p in (seg, neck) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"the image lane needs {' and '.join(missing)} in {folder} "
            "(or set GLIDER_IMAGE_LANE_MODELS to the folder that holds them)"
        )
    return ImageLaneConfig(seg_model=str(seg), neck_model=str(neck), **overrides)


def resolve_config(value) -> ImageLaneConfig | None:
    """``None``/``False`` -> no lane; ``True`` -> :func:`default_config`; a config or dict as is."""
    if value is None or value is False:
        return None
    if value is True:
        return default_config()
    if isinstance(value, ImageLaneConfig):
        return value
    if isinstance(value, dict):
        return ImageLaneConfig.from_dict(value)
    raise TypeError(f"image_lane must be a bool, dict or ImageLaneConfig, not {type(value)!r}")


def _file_ident(path: str) -> str:
    """Name + size of a model file, so a retrained model invalidates the cache."""
    p = Path(path)
    try:
        return f"{p.name}:{p.stat().st_size}"
    except OSError:
        return p.name


def required_columns(columns: Sequence[str]) -> list[str]:
    """The columns whose NaN makes a row unusable: everything but the image lane."""
    return [c for c in columns if not str(c).startswith(IMAGE_LANE_PREFIX)]


def has_lane_columns(columns: Sequence[str]) -> bool:
    return any(str(c).startswith(IMAGE_LANE_PREFIX) for c in columns)


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


class CropTracker:
    """Causal crop centre and scale, fed one frame's keypoints at a time.

    The same object drives the offline extraction and the live worker, which
    is what makes their crops identical. The centre is the chosen keypoint (or
    the centroid of the visible keypoints), carried forward when the frame has
    none. The scale is the running median body length over the last
    :data:`SCALE_HISTORY` frames that had one — a rear shortens the body, and a
    per-frame scale would zoom in on exactly the frames the lane exists for.
    """

    def __init__(self, config: ImageLaneConfig, body_axis: tuple[int, int]):
        self.config = config
        self.head, self.tail = int(body_axis[0]), int(body_axis[1])
        self._lengths: deque[float] = deque(maxlen=SCALE_HISTORY)
        self._center: np.ndarray | None = None

    def push(self, keypoints: np.ndarray) -> tuple[np.ndarray, float] | None:
        """``(centre_xy, px_per_crop_px)`` for this frame, or None before the first fix."""
        kp = np.asarray(keypoints, dtype=np.float64)
        ci = self.config.center_keypoint
        if ci is None:
            ok = np.isfinite(kp).all(axis=1)
            c = kp[ok].mean(axis=0) if ok.any() else None
        else:
            c = kp[ci] if np.isfinite(kp[ci]).all() else None
        if c is not None:
            self._center = c
        h, t = kp[self.head], kp[self.tail]
        if np.isfinite(h).all() and np.isfinite(t).all():
            length = float(np.hypot(*(t - h)))
            if length > 1e-6:
                self._lengths.append(length)
        if self._center is None or not self._lengths:
            return None
        body = float(np.median(self._lengths))
        return self._center, self.config.crop_px / (self.config.crop_body_lengths * body)


def crop_at(frame: np.ndarray, center: np.ndarray, scale: float, px: int) -> np.ndarray:
    """``px`` x ``px`` crop of ``frame`` centred on ``center``, magnified by ``scale``."""
    import cv2

    m = np.float32([[scale, 0, px / 2 - scale * center[0]], [0, scale, px / 2 - scale * center[1]]])
    return cv2.warpAffine(
        frame, m, (px, px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


# ---------------------------------------------------------------------------
# Mask shape
# ---------------------------------------------------------------------------


def _center_width(mask: np.ndarray, uv: tuple[float, float]) -> float | None:
    """Distance from the crop centre to the outline, tail removed (crop px)."""
    import cv2

    body = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _TAIL_KERNEL),
    )
    if body.sum() < 500:
        return None
    y = int(np.clip(int(uv[1]), 0, mask.shape[0] - 1))
    x = int(np.clip(int(uv[0]), 0, mask.shape[1] - 1))
    return float(cv2.distanceTransform(body, cv2.DIST_L2, 5)[y, x])


def shape_features(mask: np.ndarray) -> np.ndarray:
    """19 descriptors of the animal mask: size, elongation, solidity, Hu moments.

    All NaN when there is no usable mask (fewer than 50 px).
    """
    import cv2

    px = mask.shape[0]
    pc = px / NOMINAL_CROP_CM
    f = np.full(N_SHAPE, np.nan, np.float32)
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return f
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 50 or len(c) < 5:
        return f
    body = cv2.morphologyEx(
        m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _BODY_KERNEL)
    )
    (_, _), (a, b), _ = cv2.fitEllipse(c)
    hull = cv2.convexHull(c)
    x, y, w, h = cv2.boundingRect(c)
    hu = cv2.HuMoments(cv2.moments(c)).ravel()
    per = cv2.arcLength(c, True)
    width = _center_width(mask, (px / 2, px / 2))
    f[:12] = [
        area / pc**2,
        body.sum() / pc**2,
        max(a, b) / pc,
        min(a, b) / pc,
        min(a, b) / max(a, b, 1e-6),
        area / max(cv2.contourArea(hull), 1),
        area / max(w * h, 1),
        per**2 / max(area, 1),
        (width if width else np.nan) / pc,
        per / pc,
        w / pc,
        h / pc,
    ]
    f[12:] = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return f


# ---------------------------------------------------------------------------
# Encoder (the networks)
# ---------------------------------------------------------------------------


class ImageLaneEncoder:
    """Mask + shape + DINOv2 + pose-model neck, for a batch of crops.

    Returns one raw row per crop: ``[shape (19) | dino (768) | neck (C)]``.
    Loading takes a few seconds (three networks), so build one and reuse it.
    """

    def __init__(self, config: ImageLaneConfig, device: str | None = None):
        import torch
        from ultralytics import YOLO

        from glider.vision.pose.device import resolve_device

        self.config = config
        self.device = resolve_device(device)
        self._torch = torch
        self._half = self.device.startswith("cuda")
        self._seg = YOLO(config.seg_model)
        self._net = YOLO(config.neck_model).model.to(self.device).eval()
        self._feats: dict[str, object] = {}
        self._net.model[config.neck_layer].register_forward_hook(
            lambda _m, _i, out: self._feats.__setitem__("neck", out)
        )
        self._dino = (
            torch.hub.load(config.dino_repo, config.dino_model, verbose=False)
            .to(self.device)
            .eval()
        )
        self._mean = torch.tensor(_DINO_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_DINO_STD, device=self.device).view(1, 3, 1, 1)
        self.n_neck: int | None = None

    def masks(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        px = self.config.crop_px
        out = []
        results = self._seg(
            crops,
            imgsz=px,
            half=self._half,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )
        for res in results:
            if res.masks is None or len(res.masks) == 0:
                out.append(np.zeros((px, px), bool))
                continue
            ms = res.masks.data.cpu().numpy().astype(bool)
            hit = [j for j in range(len(ms)) if ms[j][px // 2, px // 2]]
            out.append(ms[hit[0]] if hit else ms[int(np.argmax(ms.sum((1, 2))))])
        return out

    def encode(self, crops: list[np.ndarray]) -> np.ndarray:
        """``(len(crops), D)`` float32 raw lane rows for BGR crops."""
        import cv2

        torch = self._torch
        if not crops:
            return np.empty((0, 0), np.float32)
        masks = self.masks(crops)
        with torch.no_grad():
            x = torch.from_numpy(np.stack(crops)).to(self.device)
            x = x.permute(0, 3, 1, 2).float()[:, [2, 1, 0]] / 255
            self._net(x)
            neck = self._feats["neck"].float()
            mk = torch.from_numpy(np.stack(masks)).to(self.device)[:, None].float()
            mk = torch.nn.functional.interpolate(mk, size=neck.shape[-2:], mode="area")
            pooled = (neck * mk).sum((2, 3)) / mk.sum((2, 3)).clamp(min=1e-3)

            masked = []
            for c, m in zip(crops, masks, strict=True):
                c2 = c.copy()
                c2[~m] = _GREY
                masked.append(cv2.resize(c2, (224, 224), interpolation=cv2.INTER_AREA))
            xd = torch.from_numpy(np.stack(masked)).to(self.device)
            xd = xd.permute(0, 3, 1, 2).float()[:, [2, 1, 0]] / 255
            o = self._dino.forward_features((xd - self._mean) / self._std)
            dino = torch.cat([o["x_norm_clstoken"], o["x_norm_patchtokens"].mean(1)], 1)
        self.n_neck = int(pooled.shape[1])
        shape = np.stack([shape_features(m) for m in masks])
        # Through float16, as the research cache stored them, so a fresh run
        # and a cached one feed the PCA the same numbers.
        learned = torch.cat([dino, pooled], 1).half().float().cpu().numpy()
        return np.concatenate([shape, learned], axis=1).astype(np.float32)


_ENCODERS: dict[tuple, ImageLaneEncoder] = {}


def get_encoder(config: ImageLaneConfig, device: str | None = None) -> ImageLaneEncoder:
    """A shared encoder per (config, device): loading three networks per session is slow."""
    key = (config.seg_model, config.neck_model, config.neck_layer, config.dino_model, device)
    if key not in _ENCODERS:
        _ENCODERS[key] = ImageLaneEncoder(config, device=device)
    return _ENCODERS[key]


# ---------------------------------------------------------------------------
# Per-session raw extraction + cache
# ---------------------------------------------------------------------------


def compute_raw_for_video(
    video_path: Path,
    xy: np.ndarray,
    body_axis: tuple[int, int],
    config: ImageLaneConfig,
    *,
    encoder: ImageLaneEncoder | None = None,
    batch: int = 64,
    progress=None,
) -> np.ndarray:
    """Raw lane rows for every pose frame of ``video_path`` (NaN where none).

    Video frame ``i`` is pose frame ``i``, as for the motion lane. Frames
    before the first keypoint fix, and past the end of the video, are NaN.
    """
    import cv2

    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    encoder = encoder or get_encoder(config)
    tracker = CropTracker(config, body_axis)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    out: np.ndarray | None = None
    idx: list[int] = []
    buf: list[np.ndarray] = []

    def flush():
        nonlocal out
        if not buf:
            return
        rows = encoder.encode(buf)
        if out is None:
            out = np.full((n, rows.shape[1]), np.nan, np.float32)
        out[idx] = rows
        idx.clear()
        buf.clear()

    try:
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break  # video shorter than pose; the rest stays NaN
            fix = tracker.push(xy[i])
            if fix is None:
                continue
            idx.append(i)
            buf.append(crop_at(frame, fix[0], fix[1], config.crop_px))
            if len(buf) >= batch:
                flush()
            if progress and i and i % 5000 == 0:
                progress(i, n)
        flush()
    finally:
        cap.release()
    if out is None:
        raise ValueError(f"no frame of {Path(video_path).name} had a keypoint fix to crop")
    return out


def load_or_compute_raw(
    pose_csv: Path,
    video_path: Path,
    xy: np.ndarray,
    body_axis: tuple[int, int],
    config: ImageLaneConfig,
    *,
    cache_dir: Path = CACHE_DIR,
    encoder: ImageLaneEncoder | None = None,
    progress=None,
) -> np.ndarray:
    """Cached :func:`compute_raw_for_video`; the key covers pose, video and config.

    The cache is a float16 ``.npy`` returned memory-mapped, so a cohort of long
    sessions can be fitted and windowed without holding every embedding in RAM.
    """
    pose_csv = Path(pose_csv)
    cache_dir = Path(cache_dir)
    ident = f"{pose_csv.resolve()}|{Path(video_path).resolve()}"
    digest = hashlib.sha1(ident.encode("utf-8")).hexdigest()[:12]
    head, tail = int(body_axis[0]), int(body_axis[1])
    cache_file = cache_dir / f"{pose_csv.stem}_{digest}_a{head}-{tail}_{config.cache_tag()}.npy"
    n = len(np.asarray(xy))
    if cache_file.exists():
        raw = np.load(cache_file, mmap_mode="r")
        if len(raw) == n:
            return raw
        del raw  # stale (pose length changed): recompute
    if not Path(video_path).exists():
        raise FileNotFoundError(
            f"the image lane needs the source video, but {video_path} is missing "
            f"(pose {pose_csv.name})."
        )
    raw = compute_raw_for_video(
        video_path, xy, body_axis, config, encoder=encoder, progress=progress
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, raw.astype(np.float16))
    return np.load(cache_file, mmap_mode="r")


# ---------------------------------------------------------------------------
# Reducer (fitted, stored in the model bundle)
# ---------------------------------------------------------------------------


@dataclass
class _Block:
    name: str
    start: int
    stop: int
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    components: np.ndarray | None = None  # (D_block, dims); None = pass through

    @property
    def width(self) -> int:
        return self.stop - self.start if self.components is None else self.components.shape[1]


@dataclass
class ImageLaneArtifact:
    """Fitted image-lane reducer: what a model bundle needs to rebuild the lane.

    Fitted on training sessions only — PCA fitted on a held-out session would
    leak it into the features that score it.
    """

    config: ImageLaneConfig
    blocks: list[_Block] = field(default_factory=list)

    @classmethod
    def fit(
        cls, raws: Sequence[np.ndarray], config: ImageLaneConfig, *, stride: int = 5
    ) -> ImageLaneArtifact:
        """Standardise + PCA each learned block over every ``stride``-th finite frame."""
        if not raws:
            raise ValueError("the image lane needs at least one training session to fit on")
        width = raws[0].shape[1]
        if width <= N_SHAPE:
            raise ValueError(f"raw image-lane rows have {width} values; expected > {N_SHAPE}")
        n_dino = _DINO_WIDTH.get(config.dino_model)
        if n_dino is None or N_SHAPE + n_dino >= width:
            raise ValueError(f"cannot split {width} raw values into shape | dino | neck")
        spans = [
            ("shape", 0, N_SHAPE),
            ("dino", N_SHAPE, N_SHAPE + n_dino),
            ("neck", N_SHAPE + n_dino, width),
        ]
        blocks = [_Block("shape", 0, N_SHAPE)]
        for name, a, b in spans[1:]:
            sample = np.concatenate([r[::stride, a:b] for r in raws]).astype(np.float64)
            sample = sample[np.isfinite(sample).all(axis=1)]
            if len(sample) < 2:
                raise ValueError("too few frames with an image embedding to fit the lane")
            mu = sample.mean(axis=0)
            sd = sample.std(axis=0) + 1e-6
            _u, _s, vt = np.linalg.svd((sample - mu) / sd, full_matrices=False)
            k = min(config.dims, vt.shape[0])
            blocks.append(_Block(name, a, b, mu, sd, vt[:k].T.astype(np.float64)))
        return cls(config=config, blocks=blocks)

    @property
    def frame_columns(self) -> list[str]:
        names = []
        for b in self.blocks:
            names += [f"{IMAGE_LANE_PREFIX}{b.name}{i:02d}" for i in range(b.width)]
        return names

    @property
    def columns(self) -> list[str]:
        base = self.frame_columns
        return [f"{c}{s}" for s in LANE_STATS for c in base]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Raw rows -> per-frame lane values ``(n, sum of block widths)``."""
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[None]
        parts = []
        for b in self.blocks:
            x = raw[:, b.start : b.stop]
            if b.components is not None:
                x = ((x - b.mean) / b.scale) @ b.components
            parts.append(x)
        return np.concatenate(parts, axis=1)

    def window(self, values: np.ndarray, frames: np.ndarray, query: np.ndarray) -> np.ndarray:
        return window_lane(
            values, frames, query, window=self.config.window, center=self.config.center
        )

    def frame_features(self, raw: np.ndarray, query: np.ndarray | None = None) -> pd.DataFrame:
        """Windowed lane columns for a whole session's raw rows, at ``query`` frames."""
        values = self.transform(raw)
        frames = np.arange(len(values))
        query = frames if query is None else np.asarray(query)
        return pd.DataFrame(self.window(values, frames, query), index=query, columns=self.columns)


def window_lane(
    values: np.ndarray,
    frames: np.ndarray,
    query: np.ndarray,
    *,
    window: int,
    center: bool = False,
    chunk: int = 20000,
) -> np.ndarray:
    """Value, window mean, window std and first difference at each query frame.

    ``values[i]`` belongs to source frame ``frames[i]`` (sorted, unique);
    frames that are absent or NaN are skipped, like pandas
    ``rolling(min_periods=1)``. Std is the sample std (NaN with one value);
    the difference is ``v[f] - v[f-1]``. Output columns follow
    :data:`LANE_STATS`: ``[value | mean | std | diff]``.
    """
    values = np.asarray(values, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    d = values.shape[1]
    offsets = np.arange(window) - (window // 2 if center else window - 1)
    out = np.full((len(query), 4 * d), np.nan)

    def lookup(targets: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(frames, targets).clip(0, max(len(frames) - 1, 0))
        got = np.full(targets.shape + (d,), np.nan)
        if len(frames):
            hit = frames[pos] == targets
            got[hit] = values[pos[hit]]
        return got

    for s in range(0, len(query), chunk):
        q = query[s : s + chunk]
        win = lookup(q[:, None] + offsets[None, :])  # (q, w, d)
        ok = np.isfinite(win)
        count = ok.sum(axis=1)
        total = np.where(ok, win, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / count
            dev = np.where(ok, win - mean[:, None, :], 0.0)
            std = np.sqrt((dev**2).sum(axis=1) / (count - 1))
        mean[count == 0] = np.nan
        std[count < 2] = np.nan
        cur = lookup(q)
        prev = lookup(q - 1)
        out[s : s + len(q)] = np.concatenate([cur, mean, std, cur - prev], axis=1)
    return out


# ---------------------------------------------------------------------------
# Streaming (live)
# ---------------------------------------------------------------------------


class StreamingImageLane:
    """Live counterpart of :meth:`ImageLaneArtifact.frame_features`.

    Push each frame's reduced lane values as they arrive; :meth:`row` gives
    the windowed columns for any recent frame using only frames pushed so far.
    It calls the same :func:`window_lane` the offline path does.
    """

    def __init__(self, artifact: ImageLaneArtifact, history: int = 32):
        if artifact.config.center:
            raise ValueError(
                "this model's image lane uses a centred window, which reads frames "
                "after the one it describes; retrain with a past-only lane for live use"
            )
        self.artifact = artifact
        self._frames: deque[int] = deque(maxlen=history)
        self._values: deque[np.ndarray] = deque(maxlen=history)
        self._width = len(artifact.frame_columns)

    def push(self, frame_idx: int, values: np.ndarray | None) -> None:
        if self._frames and frame_idx <= self._frames[-1]:
            self.reset()  # the stream restarted; a window must not span it
        v = np.full(self._width, np.nan) if values is None else np.asarray(values, float)
        self._frames.append(int(frame_idx))
        self._values.append(v.reshape(-1))

    def reset(self) -> None:
        self._frames.clear()
        self._values.clear()

    def row(self, frame_idx: int) -> np.ndarray:
        if not self._frames:
            return np.full(4 * self._width, np.nan)
        return self.artifact.window(
            np.stack(self._values), np.asarray(self._frames), np.array([frame_idx])
        )[0]
