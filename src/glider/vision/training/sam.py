"""Smart (auto) annotation: Segment Anything turns a few clicks into an outline.

The annotator's Auto tool works like Ultralytics Platform's smart annotation:
click on the animal (and, if the mask spills over, click the background to
push it back), or drag a box round it, and SAM proposes the animal's mask.
The mask becomes a polygon you accept and can then edit vertex by vertex.

The image is encoded once per frame -- the slow part, about a second on a CPU
-- and every click after that only runs SAM's small prompt decoder, so
refining a mask with more clicks is quick.

Qt-free: the window runs :class:`SamSegmenter` on a worker thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SAM_MODEL",
    "SAM_MODELS",
    "SamModel",
    "SamSegmenter",
    "mask_to_polygon",
    "sam_weights_dir",
]


@dataclass(frozen=True)
class SamModel:
    """One SAM checkpoint Ultralytics can download by name."""

    file: str
    label: str
    size_mb: int


#: Smallest and fastest first. Ultralytics downloads each on first use.
SAM_MODELS = (
    SamModel("mobile_sam.pt", "MobileSAM (fastest)", 40),
    SamModel("sam2.1_t.pt", "SAM 2.1 tiny", 39),
    SamModel("sam2.1_s.pt", "SAM 2.1 small", 46),
    SamModel("sam2.1_b.pt", "SAM 2.1 base (most accurate)", 162),
)
DEFAULT_SAM_MODEL = "sam2.1_t.pt"


def sam_weights_dir() -> Path:
    """Where SAM checkpoints are kept, so they download once per machine."""
    from glider.first_run import default_data_dir

    return default_data_dir() / "models" / "sam"


def mask_to_polygon(
    mask: np.ndarray, tolerance: float = 0.0015, min_vertices: int = 3
) -> list[tuple[float, float]]:
    """The outline of the largest region in a binary mask, in mask pixels.

    YOLO segmentation stores one polygon per animal with no holes, so only the
    outer boundary of the biggest connected region is kept; specks SAM leaves
    beside the animal are dropped. The outline is simplified (Douglas-Peucker)
    to within ``tolerance`` times its perimeter, at least half a pixel: a
    traced mask has a vertex on every boundary pixel, far more than needed.

    Returns:
        ``(x, y)`` vertices, or an empty list if the mask has no region large
        enough to make a polygon.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if not binary.any():
        return []
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(0.5, tolerance * cv2.arcLength(contour, closed=True))
    simple = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)
    if len(simple) < min_vertices:
        simple = contour.reshape(-1, 2)
        if len(simple) < min_vertices:
            return []
    return [(float(x), float(y)) for x, y in simple]


class SamSegmenter:
    """Point- and box-prompted SAM on one image at a time.

    Args:
        model: A file name from :data:`SAM_MODELS` (downloaded into
            :func:`sam_weights_dir` on first use), or a path to a checkpoint.
        device: Torch device string; None lets Ultralytics choose (GPU if any).
    """

    def __init__(self, model: str = DEFAULT_SAM_MODEL, device: str | None = None):
        self.model = model
        self.device = device
        self._predictor = None
        self._image_key: str | None = None
        self._image_shape: tuple[int, int] | None = None

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def _weights_path(self) -> str:
        path = Path(self.model)
        if path.is_absolute() or path.parent != Path("."):
            return str(path)
        folder = sam_weights_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder / path.name)

    def load(self) -> None:
        """Build the predictor, downloading the checkpoint if it is not on disk."""
        if self._predictor is not None:
            return
        from ultralytics.models.sam import Predictor, SAM2Predictor

        weights = self._weights_path()
        overrides = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": 1024,
            "model": weights,
            "save": False,
            "verbose": False,
        }
        if self.device:
            overrides["device"] = self.device
        cls = SAM2Predictor if "sam2" in Path(weights).stem else Predictor
        predictor = cls(overrides=overrides)
        predictor.setup_model(verbose=False)
        self._predictor = predictor

    def set_image(self, key: str, bgr: np.ndarray) -> None:
        """Encode ``bgr`` for prompting, unless it is already the image set under ``key``."""
        if key == self._image_key and self._predictor is not None:
            return
        self.load()
        self._predictor.reset_image()
        self._predictor.set_image(bgr)
        self._image_key = key
        self._image_shape = bgr.shape[:2]

    @property
    def image_key(self) -> str | None:
        return self._image_key

    def segment(
        self,
        points: list[tuple[float, float]] | None = None,
        labels: list[int] | None = None,
        box: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray | None:
        """The mask SAM proposes for one object, as an ``(H, W)`` bool array.

        Args:
            points: Clicks in image pixels.
            labels: Per point, 1 = on the object, 0 = background.
            box: ``(x1, y1, x2, y2)`` round the object, in image pixels.

        Returns:
            The mask, or None if SAM found nothing or there was no prompt.
        """
        if self._predictor is None or self._image_key is None:
            raise RuntimeError("set_image() first")
        if not points and box is None:
            return None
        kwargs = {}
        if points:
            kwargs["points"] = [[list(p) for p in points]]  # one object, N points
            kwargs["labels"] = [list(labels if labels is not None else [1] * len(points))]
        if box is not None:
            kwargs["bboxes"] = [list(box)]
        results = self._predictor(**kwargs)
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None
        mask = results[0].masks.data[0].cpu().numpy() > 0.5
        h, w = self._image_shape
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return mask if mask.any() else None
