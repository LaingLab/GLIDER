"""Per-frame geometric + kinematic feature extraction.

Inputs and outputs
------------------

* **Input**: :class:`glider.vision.pose.core.PoseData` (multi-keypoint pose, shape
  ``(n_frames, n_keypoints, 2)``).
* **Output**: ``pandas.DataFrame`` of shape ``(n_frames, n_features)``
  with named columns. The DataFrame is what the rolling-window step
  consumes; using pandas all the way means windowing is one ``.rolling``
  call.

What gets computed
------------------

Per the project plan, geometric + kinematic features only. The set is
intentionally compact (~K² distances + a few curated angles + per-
keypoint kinematics) so the trained model stays interpretable. If you
want the "kitchen sink" feature set, extend :class:`FeatureSpec` with
opt-in flags.

* **Body length** — ``dist(body_axis[0], body_axis[1])`` per frame. Used
  to normalize the pairwise distances so the resulting features are
  scale-invariant (i.e., the same behavior in a small mouse and a large
  mouse produces the same feature values).
* **Pairwise distances** — ``K*(K-1)/2`` columns. Each is the per-frame
  euclidean distance between two keypoints, divided by body length.
* **Curated angles** — ``body_curl`` (snout - neck - tail) and
  ``head_yaw`` (left_ear - snout - right_ear), each in radians. These
  are computed only when the named keypoints are present.
* **Per-keypoint speed** — magnitude of the per-frame velocity in
  body-lengths per frame. ``K`` columns.
* **Per-keypoint acceleration** — magnitude of the per-frame
  acceleration. ``K`` columns.
* **Body-axis angular velocity** — radians per frame of the body-axis
  vector. 1 column.

NaN handling
------------

Any frame where a required keypoint is NaN produces NaN feature values
for that frame. The training pipeline filters NaN rows after the
rolling step; per-feature NaN propagation through ``.rolling`` mean/std
respects ``min_periods`` so partial coverage doesn't silently fill in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from glider.vision.pose.core import PoseData


@dataclass
class FeatureSpec:
    """Knobs for feature extraction.

    The only field that's load-bearing is ``body_axis``: it picks the
    two keypoints whose distance defines "body length" (used to
    normalize pairwise distances + to scale velocities to body-lengths-
    per-frame). For a standard mouse skeleton this is typically the
    snout and tail-base indices.

    By default the spec produces a **comprehensive** feature set:

    * Every pairwise distance (``K * (K-1) / 2`` columns).
    * Every triplet angle (``K * (K-1) * (K-2) / 2`` columns —
      every vertex × every unordered pair of other keypoints).
      Controlled by ``auto_angles`` so users with very high K can opt
      out and supply a curated subset via ``angle_triplets`` instead.
    * Per-keypoint speed + acceleration.
    * Body length + body-axis angular velocity.
    """

    # Two keypoint indices that define the body axis: (head_idx, tail_idx).
    body_axis: tuple[int, int] = (0, -1)

    # If True (default), pairwise distances are divided by body length per
    # frame so the resulting features are scale-invariant. Disable only if
    # absolute pixel distances are what you actually want to compare.
    normalize_by_body_length: bool = True

    # If True (default), emit raw per-frame ``body_length`` as a feature.
    # It is the ONLY absolute-scale (non-normalized) feature, so it
    # encodes the animal's pixel size — which varies by recording (camera
    # height, mouse size) and can act as a cross-session leak. Set False
    # to drop it and make the entire feature set scale-invariant. Body
    # length is still computed internally as the distance normalizer
    # either way; this only controls whether it appears as a column.
    include_body_length: bool = True

    # Always-included angles supplied by the user. Each entry is
    # (name, (kp_a, kp_b, kp_c)) producing the angle at vertex kp_b
    # between vectors kp_b→kp_a and kp_b→kp_c. Combined with
    # ``auto_angles`` if that's True.
    angle_triplets: tuple[tuple[str, tuple[int, int, int]], ...] = ()

    # When True (default), compute ALL triplet angles automatically so
    # the model has full geometric coverage. Set False if you have many
    # keypoints (K=20+ produces 1000+ angles) and want to curate via
    # ``angle_triplets`` instead.
    auto_angles: bool = True

    # Minimum keypoint confidence for the frame's pose to count. Below
    # this, the keypoint's coordinates are treated as NaN so features
    # depending on it propagate NaN. (PoseData typically already has NaN
    # at low-confidence positions, in which case this knob is moot.)
    min_confidence: float = 0.0

    def with_resolved_body_axis(self, n_keypoints: int) -> FeatureSpec:
        """Replace negative body_axis indices with absolute ones."""
        head, tail = self.body_axis
        if head < 0:
            head += n_keypoints
        if tail < 0:
            tail += n_keypoints
        return FeatureSpec(
            body_axis=(head, tail),
            normalize_by_body_length=self.normalize_by_body_length,
            include_body_length=self.include_body_length,
            angle_triplets=self.angle_triplets,
            auto_angles=self.auto_angles,
            min_confidence=self.min_confidence,
        )

    def resolve_angle_triplets(
        self, keypoint_names: list[str]
    ) -> list[tuple[str, tuple[int, int, int]]]:
        """Return the full list of (name, indices) angle specs to compute.

        User-supplied ``angle_triplets`` always win — they keep their
        explicit names so existing model bundles continue to address
        them. Auto-generated angles get systematic names of the form
        ``"<kp_i>_at_<kp_j>_<kp_k>"`` so they're stable across runs.
        Auto-generated triplets that duplicate an explicit triplet
        (same vertex, same endpoint set) are suppressed so we don't
        produce two identical columns under different names.
        """
        explicit = list(self.angle_triplets)
        if not self.auto_angles:
            return explicit
        n = len(keypoint_names)
        # Track (vertex, sorted endpoints) pairs already covered by an
        # explicit triplet so we don't produce duplicate columns.
        explicit_keys: set[tuple[int, tuple[int, int]]] = set()
        for _name, (a, b, c) in explicit:
            explicit_keys.add((b, tuple(sorted((a, c)))))  # type: ignore[arg-type]

        auto: list[tuple[str, tuple[int, int, int]]] = []
        for vertex in range(n):
            for i in range(n):
                if i == vertex:
                    continue
                for k in range(i + 1, n):
                    if k == vertex:
                        continue
                    key = (vertex, (i, k))
                    if key in explicit_keys:
                        continue
                    name = (
                        f"{keypoint_names[i]}_at_" f"{keypoint_names[vertex]}_{keypoint_names[k]}"
                    )
                    auto.append((name, (i, vertex, k)))
        return explicit + auto

    def to_dict(self) -> dict:
        return {
            "body_axis": list(self.body_axis),
            "normalize_by_body_length": bool(self.normalize_by_body_length),
            "include_body_length": bool(self.include_body_length),
            "angle_triplets": [[name, list(triplet)] for name, triplet in self.angle_triplets],
            "auto_angles": bool(self.auto_angles),
            "min_confidence": float(self.min_confidence),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureSpec:
        return cls(
            body_axis=tuple(d.get("body_axis", (0, -1))),  # type: ignore[arg-type]
            normalize_by_body_length=bool(d.get("normalize_by_body_length", True)),
            # Default True so legacy bundles (saved before this knob) keep
            # emitting body_length and stay consistent with their training.
            include_body_length=bool(d.get("include_body_length", True)),
            angle_triplets=tuple(
                (str(name), tuple(t))  # type: ignore[misc]
                for name, t in d.get("angle_triplets", [])
            ),
            # Default True for new bundles; legacy bundles that don't
            # have this key get the old behaviour (no auto angles).
            auto_angles=bool(d.get("auto_angles", False)),
            min_confidence=float(d.get("min_confidence", 0.0)),
        )


def compute_features(
    pose: PoseData,
    spec: FeatureSpec | None = None,
) -> pd.DataFrame:
    """Extract per-frame features. Returns a DataFrame indexed 0..n_frames-1."""
    spec = (spec or FeatureSpec()).with_resolved_body_axis(pose.n_keypoints)
    xy = pose.xy.astype(np.float64, copy=True)  # (F, K, 2)

    # Mask low-confidence keypoints to NaN.
    if spec.min_confidence > 0 and pose.confidence is not None:
        mask = pose.confidence < spec.min_confidence
        # Broadcast (F, K) over the trailing xy dim.
        xy[mask] = np.nan

    n_frames, n_kpts, _ = xy.shape
    names = list(pose.keypoint_names)

    # ----- Body length (used as normalizer + a feature on its own) -----
    head_idx, tail_idx = spec.body_axis
    body_vec = xy[:, tail_idx, :] - xy[:, head_idx, :]
    body_length = np.linalg.norm(body_vec, axis=1)  # (F,)
    # Guard against zero/near-zero lengths (would explode normalized
    # distances). NaN-safe: if body length is NaN, downstream features
    # that divide by it become NaN — that's the correct outcome.
    safe_body_length = np.where(body_length < 1e-6, np.nan, body_length)

    columns: dict[str, np.ndarray] = {}
    # body_length is the only absolute-scale feature; emitting it is
    # opt-out because it can leak per-recording animal size. It's still
    # used as the distance/speed normalizer above regardless.
    if spec.include_body_length:
        columns["body_length"] = body_length

    # ----- Pairwise distances -----
    for i in range(n_kpts):
        for j in range(i + 1, n_kpts):
            d = np.linalg.norm(xy[:, i, :] - xy[:, j, :], axis=1)
            if spec.normalize_by_body_length:
                d = d / safe_body_length
            col = f"dist_{names[i]}_{names[j]}"
            columns[col] = d

    # ----- Triplet angles (auto + user-supplied) -----
    # The spec resolves to user-supplied entries first, then auto-
    # generated ones if spec.auto_angles is True. For 7 keypoints
    # that's 105 auto angles in addition to whatever the user added.
    for angle_name, (a, b, c) in spec.resolve_angle_triplets(names):
        v1 = xy[:, a, :] - xy[:, b, :]
        v2 = xy[:, c, :] - xy[:, b, :]
        # cosine of the angle at b
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = n1 * n2
        denom_safe = np.where(denom < 1e-6, np.nan, denom)
        cos_th = np.einsum("ij,ij->i", v1, v2) / denom_safe
        # Clip into [-1, 1] before arccos to dodge floating-point drift.
        cos_th = np.clip(cos_th, -1.0, 1.0)
        columns[f"angle_{angle_name}"] = np.arccos(cos_th)

    # ----- Per-keypoint kinematics (speed + acceleration) -----
    # np.gradient with a uniform step → centred differences in the
    # interior, one-sided at the ends. NaN-safe for individual missing
    # frames thanks to np.gradient's behaviour on NaN (propagates locally,
    # doesn't poison the whole column).
    # Axes: xy is (F, K, 2). We differentiate along axis 0 (frames).
    #
    # ``np.gradient`` requires at least 2 elements along the differentiation
    # axis (``edge_order + 1 = 2`` with the default edge_order=1). The
    # FeatureEngine in the live pipeline guards against single-frame
    # inputs upstream, but compute_features stays defensive so it can be
    # called with any-length PoseData without crashing. Streams shorter
    # than 2 frames simply get NaN kinematics — the windowed model
    # treats those as "unknown" and the rolling buffer pads them out.
    if n_frames >= 2:
        velocity = np.gradient(xy, axis=0)  # (F, K, 2)
        speed = np.linalg.norm(velocity, axis=2)  # (F, K)
        acceleration_v = np.gradient(velocity, axis=0)
        accel = np.linalg.norm(acceleration_v, axis=2)
    else:
        speed = np.full((n_frames, n_kpts), np.nan)
        accel = np.full((n_frames, n_kpts), np.nan)
    if spec.normalize_by_body_length:
        # Broadcast safe_body_length (F,) over the K dim.
        speed = speed / safe_body_length[:, None]
        accel = accel / safe_body_length[:, None]
    for k in range(n_kpts):
        columns[f"speed_{names[k]}"] = speed[:, k]
    for k in range(n_kpts):
        columns[f"accel_{names[k]}"] = accel[:, k]

    # ----- Body-axis angular velocity -----
    body_angle = np.arctan2(body_vec[:, 1], body_vec[:, 0])  # (F,)
    if n_frames >= 2:
        # Unwrap so 2π discontinuities don't show up as huge velocities. We
        # have to unwrap only over finite stretches so NaN doesn't poison
        # the whole stream — split on NaN, unwrap each finite run.
        body_angle_unwrapped = _safe_unwrap(body_angle)
        angvel = np.gradient(body_angle_unwrapped)
    else:
        angvel = np.full(n_frames, np.nan)
    columns["body_angular_velocity"] = angvel

    df = pd.DataFrame(columns)
    df.index.name = "frame"
    return df


def _safe_unwrap(angle: np.ndarray) -> np.ndarray:
    """Unwrap ``angle`` in radians while preserving NaN positions.

    ``numpy.unwrap`` propagates a single NaN across the rest of the
    stream, which would nuke our angular velocity feature on the first
    low-confidence frame. This variant unwraps each finite run
    independently and re-inserts NaN where the input was NaN.
    """
    out = np.copy(angle).astype(np.float64)
    finite = np.isfinite(angle)
    if not finite.any():
        return out
    # Identify contiguous finite runs.
    idx = 0
    n = len(angle)
    while idx < n:
        if not finite[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and finite[idx]:
            idx += 1
        # Run is [start, idx).
        if idx - start >= 2:
            out[start:idx] = np.unwrap(angle[start:idx])
        # Single-element runs need no unwrapping.
    return out
