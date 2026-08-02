"""Is this cohort at the scale the model was trained on?

Two quiet failures this catches, both of which produce a plausible-looking
ethogram rather than an error.

**Body length.** ``FeatureSpec`` divides nearly every feature by the animal's
per-frame body length, so the features are scale-invariant — with one
exception. ``body_length`` itself is emitted in absolute pixels, and in a
typical bundle it is among the most-used features. Apply the model to video
shot at a different resolution or camera height and that one feature stops
answering "how stretched is the animal" and starts answering "how far away is
the camera", with no warning anywhere.

The check is empirical rather than a rule of thumb: a gradient-boosted model
records the thresholds it learned to split on, so the honest question is where
this cohort's animal sits among them. Past either end, every tree takes the
same branch on every frame and the feature contributes nothing but a constant.
The comparison is against the thresholds themselves rather than the span
between the outermost two — splits cluster where the training data was
densest, so a span test is narrower than the data by construction and fires on
perfectly healthy cohorts.

**Calibration spread.** One rig, one camera height, one resolution should
give one pixel scale. A wide spread across a cohort is drawing error, and it
propagates straight into ``speed_cm_s`` and into per-video freeze/dart
thresholds — so two animals moving identically get scored differently.

Both return a message or ``None``. Neither blocks a run: a model whose splits
cannot be read, or a cohort that genuinely spans scales, must not stop work.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["body_length_splits", "calibration_spread_warning", "scale_warning"]

#: How far to the edge of the learned thresholds the cohort may sit. 0.05
#: means: warn once the typical animal is smaller than 95% of the thresholds
#: the model splits on (or larger than 95% of them), because at that point
#: almost every tree takes the same branch on almost every frame and the
#: feature has stopped carrying information.
#:
#: Deliberately not "what fraction of frames land between the outermost
#: splits": splits cluster where the training data was densest, so that test
#: is narrower than the data by construction and fires on healthy cohorts.
_EDGE_QUANTILE = 0.05

#: Relative spread (max/min - 1) above which one rig's calibrations disagree
#: more than measurement noise explains.
_MAX_SPREAD = 0.10


def body_length_splits(model) -> np.ndarray | None:
    """Every threshold the model splits a ``body_length`` statistic on.

    ``None`` when the classifier is not a gradient-boosted tree ensemble, or
    never split on the feature — in both cases there is nothing to compare
    against and no warning to give.
    """
    booster = getattr(getattr(model, "classifier", None), "booster_", None)
    if booster is None:
        return None
    try:
        dump = booster.dump_model()
        names = dump["feature_names"]
        # Only the statistics that are themselves a length. body_length__std
        # measures how much the length varied inside a window — a couple of
        # pixels — so pooling it with the means would drag the "expected
        # length" range down to near zero and the check would never fire.
        wanted = {
            i
            for i, n in enumerate(names)
            if n
            in ("body_length__mean", "body_length__max", "body_length__min", "body_length__median")
        }
        if not wanted:
            return None
        thresholds: list[float] = []
        stack = [tree["tree_structure"] for tree in dump["tree_info"]]
        while stack:
            node = stack.pop()
            if "split_feature" not in node:
                continue
            if node["split_feature"] in wanted:
                thresholds.append(float(node["threshold"]))
            stack.append(node["left_child"])
            stack.append(node["right_child"])
    except Exception:  # noqa: BLE001 - a diagnostic must never break a run
        logger.debug("could not read body_length splits from the model", exc_info=True)
        return None
    if not thresholds:
        return None
    return np.asarray(thresholds, dtype=float)


def scale_warning(model, pose, *, edge: float = _EDGE_QUANTILE) -> str | None:
    """Whether this session's animal sits at the edge of the model's range."""
    splits = body_length_splits(model)
    if splits is None:
        return None
    from glider.analysis.behavior.features import compute_features

    try:
        lengths = compute_features(pose, model.spec)["body_length"].to_numpy(dtype=float)
    except Exception:  # noqa: BLE001 - diagnostic only
        logger.debug("could not measure body length for the scale check", exc_info=True)
        return None
    lengths = lengths[np.isfinite(lengths)]
    if lengths.size == 0:
        return None

    observed = float(np.median(lengths))
    # Where this animal sits among the thresholds the model actually learned.
    # 0 means it is smaller than every one of them, 1 larger than every one —
    # either way each tree takes one branch always, so the feature is a
    # constant dressed up as a measurement.
    position = float((splits < observed).mean())
    if edge < position < 1.0 - edge:
        return None

    direction = "smaller" if position <= edge else "larger"
    return (
        f"This video's animal measures about {observed:.0f} px nose-to-tail, which is "
        f"{direction} than {max(position, 1.0 - position) * 100:.0f}% of the body-length "
        f"thresholds this model learned "
        f"({np.percentile(splits, 5):.0f}–{np.percentile(splits, 95):.0f} px).\n\n"
        "body_length is the one feature that is not scale-normalised, so at this "
        "apparent size it stops describing posture and starts describing camera "
        "distance — the model reads it as a constant. Labels will look plausible "
        "and be wrong.\n\n"
        "Re-record at the training resolution and camera height, or retrain "
        "with include_body_length=False."
    )


def calibration_spread_warning(
    calibration_master: Path | str, *, max_spread: float = _MAX_SPREAD
) -> str | None:
    """Whether one calibration file's videos disagree about the pixel scale.

    Only videos sharing a resolution are compared: a genuine mix of formats
    is a reason for different scales, whereas one rig recorded at one
    resolution has only one true answer.
    """
    from glider.vision.calibration_set import CalibrationSet

    try:
        cal_set = CalibrationSet.load(calibration_master)
    except Exception:  # noqa: BLE001 - diagnostic only
        logger.debug("could not read the calibration master for the spread check", exc_info=True)
        return None

    by_resolution: dict[tuple[int, int], list[float]] = {}
    for calibration in cal_set.entries.values():
        scale = float(getattr(calibration, "pixels_per_mm", 0.0) or 0.0)
        width = int(getattr(calibration, "calibration_width", 0) or 0)
        height = int(getattr(calibration, "calibration_height", 0) or 0)
        if scale <= 0 or width <= 0 or height <= 0:
            continue
        by_resolution.setdefault((width, height), []).append(scale)

    worst: tuple[float, tuple[int, int], list[float]] | None = None
    for key, scales in by_resolution.items():
        if len(scales) < 3:
            continue  # too few to tell drawing error from a real difference
        spread = max(scales) / min(scales) - 1.0
        if spread > max_spread and (worst is None or spread > worst[0]):
            worst = (spread, key, scales)
    if worst is None:
        return None

    spread, (width, height), scales = worst
    return (
        f"The {len(scales)} calibrated {width}×{height} videos disagree about scale by "
        f"{spread * 100:.0f}% ({min(scales):.3f}–{max(scales):.3f} px/mm).\n\n"
        "At one camera height there is only one right answer, so most of that "
        "is drawing error — and it lands directly in speed_cm_s and in the "
        "per-video freeze/dart thresholds, scoring identical movement "
        "differently between animals.\n\n"
        "Consider calibrating once and reusing that scale for the rig."
    )
