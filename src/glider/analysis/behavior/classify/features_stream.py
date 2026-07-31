"""Pure, non-threaded streaming feature extraction.

This module holds the per-frame feature math that used to live *inside*
:class:`~glider.analysis.behavior.classify.threads.FeatureEngine`. Extracting
it here — with no threading and no Qt — lets a future live classifier reuse the
EXACT same computation, so live features are bit-for-bit identical to the
offline ones :func:`~glider.analysis.behavior.features.compute_features`
produces.

Two pieces:

* :func:`derive_stream_columns` — recover the *base* per-frame feature names
  (and the kinematic bases carrying rolling spectral columns) from a trained
  model's windowed feature names, by stripping the ``__{stat}`` suffixes.
* :class:`StreamingFeatureExtractor` — a 5-frame keypoint ring that reproduces
  training's *centered* ``np.gradient`` kinematics. It emits the MIDDLE frame's
  features once the history is full, matching whole-session offline features.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence

import numpy as np

from glider.analysis.behavior.features import FeatureSpec, compute_features
from glider.vision.pose.core import PoseData


def expected_keypoint_order(model) -> list[str]:
    """The keypoint names, in order, that *model* was trained with.

    Recovered from the feature columns: :func:`compute_features` emits one
    ``speed_<name>`` per keypoint in keypoint order, so first appearance in
    ``feature_names`` reproduces the training order.

    Order matters, not just the name set. With ``FeatureSpec.auto_angles`` the
    angle features are generated from keypoint *triplets by index*, and the
    names sitting at those indices are baked into the resulting column names
    (``angle_body_center_at_left_ear_left_hip``). Enter the same names in a
    different order and those columns simply never appear — they arrive as
    NaN, every prediction comes back blank, and nothing raises. Returns ``[]``
    when the order cannot be recovered.
    """
    seen: set[str] = set()
    order: list[str] = []
    for column in getattr(model, "feature_names", []):
        match = re.match(r"speed_(.+?)__", column)
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            order.append(match.group(1))
    return order


def pose_csv_bodyparts(pose_csv) -> list[str]:
    """Bodypart names, in order, from a DeepLabCut CSV's header.

    Reads the three header rows only — a pose CSV is large and nothing here
    needs the data. Returns ``[]`` when the file cannot be parsed, because a
    cross-check that cannot run must not become an error of its own.
    """
    import csv as _csv

    try:
        with open(pose_csv, newline="", encoding="utf-8") as f:
            rows = [next(_csv.reader(f)) for _ in range(3)]
    except (OSError, StopIteration, UnicodeDecodeError):
        return []
    # Header shape: (scorer, bodyparts, coords); column 0 is the frame index.
    bodyparts = rows[1][1:]
    ordered: list[str] = []
    for name in bodyparts:
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def keypoint_order_problem(model, entered: Sequence[str]) -> str | None:
    """Why *entered* cannot produce this model's features, or None if it can.

    Returns a message naming the first mismatching position, because that is
    what the operator has to fix. None means the names are usable — it does
    NOT promise they are attached to the right body parts, which only a look
    at a labelled frame can confirm.
    """
    expected = expected_keypoint_order(model)
    if not expected:
        return None  # nothing to check against; don't invent a failure
    entered = list(entered)
    if len(entered) != len(expected):
        return (
            f"this model was trained with {len(expected)} keypoints but "
            f"{len(entered)} names were entered"
        )
    for i, (want, got) in enumerate(zip(expected, entered, strict=True)):
        if want != got:
            return f"position {i} should be {want!r}, not {got!r}"
    return None


def derive_stream_columns(model) -> tuple[list[str], list[str]]:
    """Derive ``(per_frame_feature_names, spectral_features)`` from a model.

    The model's windowed feature names carry suffixes like ``__mean`` /
    ``__std`` / ``__max``. To stream them, the live buffer needs the *base*
    (per-frame) feature names — recovered by stripping the suffix off the
    columns produced by the model's first stat.

    ``spectral_features`` are the kinematic base features that additionally
    carry rolling spectral columns (``__domfreq``); the presence of those
    columns is the signal (no bundle schema change needed). Empty for models
    trained without frequency features.

    Raises
    ------
    ValueError
        If no per-frame feature names can be derived (no column ends with the
        first stat's suffix) — mirroring the original pipeline guard.
    """
    first_stat = model.stats[0]
    suffix = f"__{first_stat}"
    per_frame_feature_names = [c[: -len(suffix)] for c in model.feature_names if c.endswith(suffix)]
    if not per_frame_feature_names:
        raise ValueError(
            f"could not derive per-frame feature names from the model; "
            f"first stat is {first_stat!r} but no columns end with {suffix!r}"
        )
    spectral_features = [
        c[: -len("__domfreq")] for c in model.feature_names if c.endswith("__domfreq")
    ]
    return per_frame_feature_names, spectral_features


class StreamingFeatureExtractor:
    """Per-frame features from a rolling keypoint history (centered gradients).

    Training computes velocity/acceleration with ``np.gradient`` over the whole
    session, so an interior frame uses *centered* differences. We reproduce
    that by keeping a full 5-frame keypoint history and returning the MIDDLE
    frame's features: centered velocity needs the ±1 neighbours and centered
    acceleration the ±2, so a 5-frame window centered on the target frame
    yields the exact training features (``np.gradient`` is a local stencil).

    :meth:`push` returns ``None`` until the history is full, so callers only
    ever receive centered rows. The extractor is pure — no threading, no Qt —
    and holds no queue state, so it is trivially reusable off any hot path.

    Note: ``fps`` is stored on the emitted :class:`PoseData` but is inert for
    feature values — :func:`compute_features` never reads it (gradients use
    unit frame spacing).
    """

    HISTORY = 5

    def __init__(
        self,
        spec: FeatureSpec,
        keypoint_names: list[str],
        fps: float = 30.0,
    ):
        self.spec = spec
        self.keypoint_names = list(keypoint_names)
        self.fps = float(fps)
        self._history: deque[np.ndarray] = deque(maxlen=self.HISTORY)

    @property
    def lag(self) -> int:
        """Frames the emitted MIDDLE row trails the current frame by."""
        return (self.HISTORY - 1) - (self.HISTORY // 2)

    def reset(self) -> None:
        """Clear the keypoint history (re-arm the warm-up)."""
        self._history.clear()

    def push(self, keypoints: np.ndarray) -> dict[str, float] | None:
        """Append one frame's keypoints; return the MIDDLE frame's features.

        Returns ``None`` until the 5-frame history is full. Once full, computes
        features over the window and returns the middle frame's row as a dict.
        Confidence isn't passed through — the caller (tracker) already NaN'd
        low-confidence keypoints; :func:`compute_features` propagates NaN
        through the math correctly.
        """
        self._history.append(np.asarray(keypoints).copy())
        if len(self._history) < self._history.maxlen:
            return None
        xy = np.stack(list(self._history), axis=0)
        conf = np.where(np.isnan(xy).any(axis=-1), 0.0, 1.0)
        pose = PoseData(
            xy=xy,
            confidence=conf,
            keypoint_names=self.keypoint_names,
            fps=self.fps,
        )
        df = compute_features(pose, spec=self.spec)
        return df.iloc[len(df) // 2].to_dict()
