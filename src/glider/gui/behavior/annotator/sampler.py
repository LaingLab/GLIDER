"""Diverse-clip sampler.

Given a pose CSV, produce N short clips that span the feature space —
the user labels these instead of scrubbing through the full video.

Diversity design
----------------

The original implementation used k-means++ over scale-invariant
behavioral features alone. That worked, but had two failure modes
on real lab video:

1. **Adjacent-in-time picks.** Two windows a few seconds apart in
   the same behavior are nearly identical in feature space, so
   k-means++ could pick both without realizing one was right after
   the other.
2. **Spatially blind.** The training features are scale-invariant by
   design (so behavior matching generalizes across mice and cage
   sizes), which means the sampler can't distinguish "rearing at the
   north wall" from "rearing at the south wall" — same behavior,
   different location.

This rewrite addresses both:

* **Spatial features**: per-window mean keypoint position (raw pixel
  xy, after windowing) is concatenated with the behavioral features.
  Each block is StandardScaler'd separately so we can tune the
  ``spatial_weight`` knob without it being eaten by scale differences.
* **Constrained k-means++**: after each pick, all windows within
  ``±min_frame_gap`` frames have their selection probability zeroed
  so the next pick can't be temporally adjacent. Auto-derived default
  ensures clips are spread across the recording without the user
  having to think about it.

Both knobs are exposed for power users but the defaults should "just
work" on a typical mouse video.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from glider.analysis.behavior.features import FeatureSpec, compute_features
from glider.analysis.behavior.windowing import apply_rolling
from glider.vision.pose.dlc import from_dlc_csv

# Default clip-playback durations the UI cycles through.
DEFAULT_CLIP_SECONDS: tuple[float, ...] = (0.4, 0.6, 0.8, 1.0)


@dataclass
class ProposedClip:
    """One clip the sampler picked + the metadata the UI needs to play it."""

    window_index: int  # row index in the windowed-feature matrix
    center_frame: int  # frame in the source video this clip centers on
    start_frame: int  # inclusive
    end_frame: int  # exclusive
    clip_seconds: float  # one of the configured durations
    video_path: str

    @property
    def duration_frames(self) -> int:
        return int(self.end_frame - self.start_frame)


def zones_to_clips(store, video_path: str | Path, fps: float) -> list[ProposedClip]:
    """Turn the zones of an :class:`AnnotationStore` into review clips.

    Used by ``--review`` mode: every saved behavior zone becomes a
    :class:`ProposedClip` the annotator can replay/re-trim/re-label,
    spanning exactly the zone's frames. Sorted by start frame. The pose
    CSV and sampler are not involved — this is a pure mapping.
    """
    video_path = str(video_path)
    fps = float(max(fps, 1e-3))
    clips: list[ProposedClip] = []
    zones = sorted(store, key=lambda z: (z.start_frame, z.end_frame))
    for i, z in enumerate(zones):
        center = (z.start_frame + z.end_frame) // 2
        clips.append(
            ProposedClip(
                window_index=i,
                center_frame=int(center),
                start_frame=int(z.start_frame),
                end_frame=int(z.end_frame),
                clip_seconds=float(z.end_frame - z.start_frame) / fps,
                video_path=video_path,
            )
        )
    return clips


def _excluded_window_mask(
    window_indices: np.ndarray,
    n_frames: int,
    exclude_zones: Iterable[tuple[int, int]],
    margin: int,
) -> np.ndarray:
    """Return a keep-mask (True = keep) over ``window_indices``.

    A window index is dropped if it falls within ``margin`` frames of any
    ``[start, end)`` zone in ``exclude_zones``. ``margin`` should cover the
    clip's half-length plus the window→centre offset so the resulting clip
    can't overlap an excluded zone.
    """
    forbidden = np.zeros(int(n_frames), dtype=bool)
    for s, e in exclude_zones:
        a = max(0, int(s) - margin)
        b = min(int(n_frames), int(e) + margin)
        if b > a:
            forbidden[a:b] = True
    idx = np.asarray(window_indices)
    return ~forbidden[idx]


def propose_clips(
    pose_csv: str | Path,
    video_path: str | Path,
    *,
    spec: FeatureSpec | None = None,
    window: int = 30,
    n_clips: int = 50,
    fps: float = 30.0,
    clip_lengths_seconds: Iterable[float] = DEFAULT_CLIP_SECONDS,
    random_state: int = 42,
    spatial_weight: float = 1.0,
    min_frame_gap: int | None = None,
    exclude_zones: Iterable[tuple[int, int]] | None = None,
    exclude_margin: int | None = None,
) -> list[ProposedClip]:
    """Propose ``n_clips`` diverse clips for the labelling UI.

    Parameters
    ----------
    spatial_weight
        How much weight to give the mouse's arena position when
        measuring "diversity". 0 = behavior-only (legacy); 1 (default)
        = spatial coordinates contribute equally to behavioral features
        after both blocks are independently z-scored.
    min_frame_gap
        Minimum number of frames between any two picked clips. ``None``
        (default) picks a sensible value: roughly ``n_frames /
        (n_clips * 4)``, clamped to at least one ``window``. Forces the
        sampler to spread picks across the full recording instead of
        clustering them.
    """
    pose_csv = Path(pose_csv)
    video_path = str(video_path)
    spec = spec or FeatureSpec()

    pose = from_dlc_csv(pose_csv, fps=fps)
    spec = spec.with_resolved_body_axis(pose.n_keypoints)
    # Materialise once so it can feed both the exclusion margin and the
    # per-clip length draw below without exhausting a generator.
    clip_lengths_seconds = tuple(clip_lengths_seconds)
    if not clip_lengths_seconds:
        raise ValueError("clip_lengths_seconds must be non-empty")

    # ---- 1. Behavioral features (the same set the trainer uses) ----
    features = compute_features(pose, spec=spec)
    windowed = apply_rolling(features, window=window)

    # ---- 2. Spatial features (sampler-only) ----
    # Per-frame mean of all keypoint positions. NaN-safe — frames with
    # any missing keypoint just contribute a NaN to the mean, which is
    # filtered out alongside the behavioral NaN rows below.
    xy = pose.xy.astype(np.float64, copy=False)
    # Shape (F, K, 2) → mean over keypoints → (F, 2). Frames where EVERY
    # keypoint is NaN (no pose that frame) make nanmean warn "Mean of
    # empty slice" and return NaN; that's expected — those rows are
    # dropped by the validity mask below — so silence the noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_xy = np.nanmean(xy, axis=1)
    spatial_df = pd.DataFrame(
        {"spatial_x": mean_xy[:, 0], "spatial_y": mean_xy[:, 1]},
        index=features.index,
    )
    # Window the spatial signal the same way behavioral features are
    # windowed so the two streams stay aligned. Just the rolling mean
    # is enough — we care about where the mouse is in this window, not
    # the variance of its location within the window.
    spatial_windowed = apply_rolling(spatial_df, window=window, stats=("mean",))

    # ---- 3. Drop NaN rows ----
    behavioral_ok = ~windowed.isna().any(axis=1)
    spatial_ok = ~spatial_windowed.isna().any(axis=1)
    valid_mask = (behavioral_ok & spatial_ok).to_numpy()
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError(
            "no valid windowed rows after dropping NaN; pose tracking may be "
            "too sparse, or `window` is larger than the video"
        )

    x_behavior = windowed.loc[valid_mask].to_numpy(dtype=np.float64)
    x_spatial = spatial_windowed.loc[valid_mask].to_numpy(dtype=np.float64)

    # ---- 3b. Drop windows near already-labelled zones (so new clips don't
    # re-cover frames the user already annotated) ----
    if exclude_zones:
        if exclude_margin is None:
            # Cover the window→centre offset plus the longest clip's half-span
            # so the picked clip can't bleed into an excluded zone.
            exclude_margin = window + int(round(max(clip_lengths_seconds) * fps))
        keep = _excluded_window_mask(valid_indices, pose.n_frames, exclude_zones, exclude_margin)
        valid_indices = valid_indices[keep]
        x_behavior = x_behavior[keep]
        x_spatial = x_spatial[keep]
        if len(valid_indices) == 0:
            raise ValueError(
                "no candidate windows left after excluding labelled zones; the "
                "session may be fully covered, or exclude_margin is too large"
            )

    # ---- 4. Z-score each block independently so weights are meaningful ----
    from sklearn.preprocessing import StandardScaler

    x_behavior = StandardScaler().fit_transform(x_behavior)
    if spatial_weight > 0:
        x_spatial = StandardScaler().fit_transform(x_spatial) * spatial_weight
        x_sampler = np.concatenate([x_behavior, x_spatial], axis=1)
    else:
        x_sampler = x_behavior

    # ---- 5. Constrained k-means++ ----
    n_request = int(min(n_clips, x_sampler.shape[0]))
    if n_request < 1:
        raise ValueError("n_clips must be >= 1")

    # Resolve the temporal-gap default. Goal: clips should be spread
    # across the full recording. Average gap = n_valid / n_clips;
    # require the minimum to be a healthy fraction of that.
    if min_frame_gap is None:
        avg_gap = len(valid_indices) / max(n_request, 1)
        min_frame_gap = int(max(window, avg_gap / 4))
    min_frame_gap = max(0, int(min_frame_gap))

    picked_in_valid = _constrained_kmeans_pp(
        x_sampler,
        n_clusters=n_request,
        valid_window_indices=valid_indices,
        min_frame_gap=min_frame_gap,
        random_state=random_state,
    )
    picked_window_indices = valid_indices[picked_in_valid]

    # ---- 6. Assign random clip lengths + build ProposedClip objects ----
    rng = np.random.default_rng(random_state)
    clip_lengths_seconds = tuple(clip_lengths_seconds)
    if not clip_lengths_seconds:
        raise ValueError("clip_lengths_seconds must be non-empty")

    clips: list[ProposedClip] = []
    n_frames = pose.n_frames
    for w_idx in picked_window_indices:
        # The rolling window covers [w_idx - window + 1, w_idx]; use
        # the window's centre as the clip centre.
        center = int(w_idx) - window // 2
        center = max(0, min(center, n_frames - 1))
        clip_seconds = float(rng.choice(clip_lengths_seconds))
        n_clip = int(round(clip_seconds * fps))
        half = n_clip // 2
        start = max(0, center - half)
        end = min(n_frames, center + (n_clip - half))
        if end <= start:
            end = min(n_frames, start + 1)
        clips.append(
            ProposedClip(
                window_index=int(w_idx),
                center_frame=int(center),
                start_frame=int(start),
                end_frame=int(end),
                clip_seconds=clip_seconds,
                video_path=video_path,
            )
        )

    clips.sort(key=lambda c: c.center_frame)
    return clips


def propose_clips_multi(
    sessions: list[tuple[Path, Path]],
    *,
    n_clips_total: int,
    spec: FeatureSpec | None = None,
    window: int = 30,
    fps: float = 30.0,
    clip_lengths_seconds: Iterable[float] = DEFAULT_CLIP_SECONDS,
    random_state: int = 42,
    spatial_weight: float = 1.0,
    min_frame_gap: int | None = None,
    exclude_zones_by_session: list[Iterable[tuple[int, int]]] | None = None,
) -> list[ProposedClip]:
    """Sample ``n_clips_total`` diverse clips across multiple (pose, video) sessions.

    Per-video quotas are allocated by ``divmod``; the first ``remainder``
    videos get one extra clip. Each video's ``propose_clips`` call uses
    ``random_state = random_state + i`` so picks across videos are
    decorrelated while remaining reproducible from the top-level seed.

    Output is a flat ``list[ProposedClip]`` shuffled cross-video (so a
    labeler doesn't sit on one video before switching). If a video has
    fewer valid windowed rows than its quota, that video contributes
    what it can and the unused quota is NOT redistributed — keeps
    semantics predictable.

    Parameters
    ----------
    sessions
        List of ``(pose_csv, video_path)`` tuples. Order is
        deterministic — the first ``remainder`` sessions get an extra
        clip.
    n_clips_total
        Total clips across all videos. Must be ``>= len(sessions)``.

    Returns
    -------
    list[ProposedClip]
        Flat, shuffled list. Each clip carries its source ``video_path``
        so the player knows which video to open.
    """
    if not sessions:
        raise ValueError("propose_clips_multi requires at least one session")
    n = len(sessions)
    if n_clips_total < n:
        raise ValueError(f"n_clips_total ({n_clips_total}) must be >= number of videos ({n})")

    base, remainder = divmod(int(n_clips_total), n)
    quotas = [base + (1 if i < remainder else 0) for i in range(n)]

    all_clips: list[ProposedClip] = []
    for i, ((pose_csv, video_path), quota) in enumerate(zip(sessions, quotas, strict=False)):
        clips_i = propose_clips(
            pose_csv=pose_csv,
            video_path=video_path,
            spec=spec,
            window=window,
            n_clips=quota,
            fps=fps,
            clip_lengths_seconds=clip_lengths_seconds,
            random_state=random_state + i,
            spatial_weight=spatial_weight,
            min_frame_gap=min_frame_gap,
            exclude_zones=(exclude_zones_by_session[i] if exclude_zones_by_session else None),
        )
        all_clips.extend(clips_i)

    rng = np.random.default_rng(random_state)
    rng.shuffle(all_clips)
    return all_clips


# ---------------------------------------------------------------------------
# Constrained k-means++ (private)
# ---------------------------------------------------------------------------


def _constrained_kmeans_pp(
    x: np.ndarray,
    n_clusters: int,
    valid_window_indices: np.ndarray,
    min_frame_gap: int,
    random_state: int,
) -> np.ndarray:
    """k-means++ initialization with a temporal exclusion zone.

    After each pick, all windows within ``±min_frame_gap`` frames of
    the just-picked window have their squared-distance set to zero so
    the next pick can't be temporally adjacent.

    Returns indices into ``x`` (i.e., into the valid-row matrix), not
    into the original window grid — caller maps back via
    ``valid_window_indices``.
    """
    n = x.shape[0]
    rng = np.random.default_rng(random_state)
    if n_clusters >= n:
        return np.arange(n)

    # First pick is uniform random. (Plain k-means++ picks the first
    # uniformly too.)
    first = int(rng.integers(0, n))
    picks: list[int] = [first]
    # Squared distance to nearest already-picked point.
    d2 = np.sum((x - x[first]) ** 2, axis=1)
    _apply_temporal_mask(d2, valid_window_indices, first, min_frame_gap)

    for _ in range(n_clusters - 1):
        total = float(d2.sum())
        if total <= 0:
            # No remaining candidates that satisfy the constraint.
            # This happens when min_frame_gap is too large for n_clusters;
            # fall back to picking the temporally-most-distant valid row
            # so we still return n_clusters when at all possible.
            remaining = _farthest_valid_in_time(picks, valid_window_indices)
            if remaining is None:
                break
            picks.append(remaining)
            new_d2 = np.sum((x - x[remaining]) ** 2, axis=1)
            d2 = np.minimum(d2, new_d2)
            _apply_temporal_mask(d2, valid_window_indices, remaining, min_frame_gap)
            continue
        probs = d2 / total
        nxt = int(rng.choice(n, p=probs))
        picks.append(nxt)
        new_d2 = np.sum((x - x[nxt]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
        _apply_temporal_mask(d2, valid_window_indices, nxt, min_frame_gap)

    return np.asarray(picks, dtype=np.int64)


def _apply_temporal_mask(
    d2: np.ndarray,
    valid_window_indices: np.ndarray,
    just_picked_idx: int,
    min_frame_gap: int,
) -> None:
    """Zero out d² for every row whose window index is within the gap.

    Operates on the d² array in place. ``just_picked_idx`` is an index
    into ``d2`` (i.e., a row of the valid matrix), so we look up its
    original window index, then build a mask over all rows whose
    window indices are within ``min_frame_gap``.
    """
    if min_frame_gap <= 0:
        # No constraint — just zero the picked row itself so it isn't
        # re-picked.
        d2[just_picked_idx] = 0.0
        return
    picked_window = int(valid_window_indices[just_picked_idx])
    diffs = np.abs(valid_window_indices - picked_window)
    too_close = diffs <= min_frame_gap
    d2[too_close] = 0.0


def _farthest_valid_in_time(picks: list[int], valid_window_indices: np.ndarray) -> int | None:
    """Fallback when the temporal constraint blocks every candidate.

    Picks the valid row whose temporal distance to its nearest already-
    picked row is largest. Doesn't honour the gap — we'd rather return
    a clip that's "as far as possible" than fail to deliver n_clusters.
    """
    if not picks:
        return None
    picked_windows = valid_window_indices[picks]
    # For each candidate row, distance to the nearest picked window.
    dists = np.min(
        np.abs(valid_window_indices[:, None] - picked_windows[None, :]),
        axis=1,
    )
    # Don't re-pick an already-picked row.
    dists[picks] = -1
    best = int(np.argmax(dists))
    return best if dists[best] > 0 else None
