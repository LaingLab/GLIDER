"""Score a whole session at once from poses already on disk.

The streaming pipeline in :mod:`.threads` exists to classify a *live* camera:
frames arrive one at a time and nothing downstream may block, so every stage
works on one frame. Applying a model to a recorded video with its poses
already tracked is the opposite problem — every frame is known up front — and
paying the per-frame price there is pure loss:

* ``model.predict_one`` spends most of its time building a one-row
  ``DataFrame`` so sklearn sees column names. Batched, the same LightGBM
  predicts ~300x faster per row.
* ``compute_features`` is a whole-session vectorised function. The streaming
  extractor calls it once per frame on a five-frame window to recover the
  centered gradients, re-paying the setup 45,000 times per video.
* LightGBM carries the thread count it was trained with (commonly one per
  core). On a one-row predict that pool has nothing to divide and spins,
  starving the decode and feature threads it shares a machine with.

So this module does the same arithmetic the streaming path does, in the order
training does it: features for every frame, rolling stats over the whole
column, one predict call. Output is row-for-row what
:class:`~glider.analysis.behavior.classify.threads.BehaviorClassifier` would
have written — same cadence, same frame indices, same blank-on-NaN rule, same
smoothing, same speed axis — because the ethogram is the analysis artifact and
a faster path that quietly scores differently would be worthless.

Parity notes, each of which is a test in
``tests/unit/analysis/behavior/classify/test_batch.py``:

* Emission cadence and frame tagging mirror ``FeatureEngine``: a row every
  ``predict_every`` frames, tagged ``frame - lag`` where ``lag`` is the
  streaming extractor's centered-gradient delay.
* Rolling stats use ``min_periods=1`` so the ramp-up matches the live
  buffer, which emits as soon as it holds one row rather than waiting for a
  full window.
* A row with any NaN scores blank, exactly as ``predict_one`` does.
* The speed axis is inherently sequential (``CausalSpeed`` smooths causally),
  so it is still a loop — but a cheap one, and it is the same object the
  live path uses rather than a re-derivation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["EthogramRows", "classify_pose_data"]

# The streaming extractor emits the middle row of a 5-frame history, so its
# output trails the current frame by this much. Imported rather than repeated
# would be circular at module load, so it is asserted against in the tests.
_STREAM_HISTORY = 5


@dataclass
class EthogramRows:
    """The rows an apply run writes, plus what the caller needs to time them."""

    frames: list[int]
    labels: list[str]
    speed_labels: list[str] = field(default_factory=list)
    speed_px: list[float] = field(default_factory=list)
    n_source_frames: int = 0

    def __len__(self) -> int:
        return len(self.frames)


def _stream_lag(history: int = _STREAM_HISTORY) -> int:
    """Frames the centered-gradient row trails the current frame by."""
    return (history - 1) - (history // 2)


def _speed_axis(
    pose_xy,
    freeze_threshold,
    dart_threshold,
    freeze_min_frames,
    dart_min_frames,
    dart_merge_gap=0,
):
    """Per-frame ``(speed_label, speed_px)`` for a recorded session.

    The speed itself is causal by construction — each frame's smoothed value
    depends on the frames before it — so that stays a loop over the very same
    ``CausalSpeed`` the live path uses, keeping one definition of the signal.

    The labelling does not stay online. ``FreezeDartDetector`` cannot name a
    bout until it has watched the whole minimum duration elapse, which is
    unavoidable live and wrong here: every frame is known before anything is
    written, so a run that qualifies is labelled from its first frame rather
    than from its thirtieth.
    """
    from glider.analysis.behavior.classify.speed_state import CausalSpeed, speed_axis_offline

    causal = CausalSpeed()
    values = [float(causal.push(frame)) for frame in pose_xy]
    labels = speed_axis_offline(
        values,
        freeze_threshold,
        dart_threshold,
        freeze_min_frames=freeze_min_frames,
        dart_min_frames=dart_min_frames,
        dart_merge_gap=dart_merge_gap,
    )
    return labels, values


def classify_pose_data(
    pose,
    model,
    *,
    predict_every: int = 3,
    confidence_threshold: float = 0.0,
    class_thresholds: dict[str, float] | None = None,
    smooth_window: int = 1,
    freeze_threshold: float | None = None,
    dart_threshold: float | None = None,
    freeze_min_frames: int = 30,
    dart_min_frames: int = 3,
    frame_range: tuple[int, int] | None = None,
) -> EthogramRows:
    """Score every frame of *pose* with *model*, as the streaming path would.

    Parameters mirror the corresponding :class:`LiveInferenceConfig` fields so
    a caller can hand the same settings to either path and get the same rows.

    ``frame_range`` limits the rows returned to an inclusive window of source
    frames. Everything upstream of the prediction still runs over the whole
    session, so a windowed run's labels are exactly the rows a whole-session
    run would have produced for those frames — only the classifier's work is
    skipped, which is where the time goes anyway.
    """
    from glider.analysis.behavior.classify.features_stream import derive_stream_columns
    from glider.analysis.behavior.classify.smoothing import MajorityVoteSmoother
    from glider.analysis.behavior.features import compute_features
    from glider.analysis.behavior.windowing import apply_rolling

    predict_every = max(1, int(predict_every))
    lag = _stream_lag()

    per_frame_names, spectral = derive_stream_columns(model)
    if spectral:
        # __domfreq columns come from a rolling FFT the offline windowing
        # helper does not produce. Rather than score a model on silently
        # missing columns, hand it back to the caller to run the streaming
        # path, which does compute them.
        raise NotImplementedError(
            "this model uses rolling spectral features (--freq-features), which "
            "the batch path does not compute; use the streaming pipeline"
        )

    features = compute_features(pose, model.spec)
    # Column order is fixed by the model, not by whatever compute_features
    # happened to emit, so a reordering upstream cannot silently shuffle
    # values into the wrong feature.
    features = features.reindex(columns=per_frame_names)
    # The streaming extractor emits nothing until its keypoint history fills,
    # so the live rolling buffer never contains the first `lag` rows. Dropping
    # them here keeps the warm-up windows identical instead of averaging in
    # rows the live path never saw.
    n_frames = len(features)
    features = features.iloc[lag:]
    # min_periods=1 matches the live SlidingFeatureBuffer, which emits from
    # its very first row rather than waiting for a full window.
    rolled = apply_rolling(features, window=model.window, stats=model.stats, min_periods=1)
    windowed = rolled.reindex(columns=model.feature_names)

    # Rows are tagged with the middle frame they describe, so row i of `windowed` is
    # frame i + lag. The FeatureEngine ticks once per tracked frame (tick =
    # frame + 1) and emits when tick % predict_every == 0.
    #
    # The last `lag` frames are excluded deliberately. A centered row needs
    # the frames on *both* sides, so the streaming extractor never emits them
    # — offline they could be filled from np.gradient's one-sided edge, but
    # then the fast path would stop being a drop-in for two frames in 45,000.
    # The cadence is counted on the frame the engine *ticked* on, which leads
    # the tagged frame by `lag` — not on the tagged frame itself.
    emit_frames = np.arange(lag, max(lag, n_frames - lag))
    keep = (emit_frames + lag + 1) % predict_every == 0
    if frame_range is not None:
        # Analysing a window restricts which rows are *emitted*, nothing else.
        # Features, rolling statistics and the causal speed are still computed
        # over the whole session, so the window's opening frames get the same
        # full-width rolling windows and the same warmed-up speed filter they
        # would have had in a whole-session run. Trimming the poses first would
        # instead give them partial windows and a cold filter, and the very
        # frames the operator chose to look at would be the least comparable
        # ones in the cohort.
        first, last = frame_range
        keep &= (emit_frames >= first) & (emit_frames <= last)
    emit_rows = np.nonzero(keep)[0]

    speed_labels_all: list[str] = []
    speed_values_all: list[float] = []
    use_speed = freeze_threshold is not None and dart_threshold is not None
    if use_speed:
        speed_labels_all, speed_values_all = _speed_axis(
            pose.xy, freeze_threshold, dart_threshold, freeze_min_frames, dart_min_frames
        )

    values = windowed.to_numpy(dtype=np.float64)
    scored = values[emit_rows]
    # Blank on any NaN, exactly as predict_one does — a partially-known row is
    # not a weak prediction, it is no prediction.
    usable = ~np.isnan(scored).any(axis=1)

    raw = np.full(len(scored), "", dtype=object)
    if usable.any():
        frame = pd.DataFrame(scored[usable], columns=model.feature_names)
        raw[usable] = _predict(model, frame, confidence_threshold, class_thresholds)

    smoother = MajorityVoteSmoother(window=smooth_window)
    frames: list[int] = []
    labels: list[str] = []
    speed_out: list[str] = []
    speed_px_out: list[float] = []
    for i, row_idx in enumerate(emit_rows):
        frame = int(row_idx) + lag  # row i of X describes frame i + lag
        labels.append(smoother.push(str(raw[i])))
        frames.append(frame)
        if use_speed:
            # The live path reads the speed for the *middle* frame — the same
            # frame this row is tagged with.
            if 0 <= frame < len(speed_labels_all):
                speed_out.append(speed_labels_all[frame])
                speed_px_out.append(speed_values_all[frame])
            else:
                speed_out.append("")
                speed_px_out.append(float("nan"))

    return EthogramRows(
        frames=frames,
        labels=labels,
        speed_labels=speed_out,
        speed_px=speed_px_out,
        n_source_frames=n_frames,
    )


def write_ethogram_csv(path, rows: EthogramRows, *, speed_axis: bool, cm_s_per_px_frame=None):
    """Write *rows* in the exact layout the streaming classifier writes."""
    import csv

    from glider.analysis.behavior.classify.threads import _fmt

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        if speed_axis:
            w.writerow(["frame", "behavior", "speed", "speed_px_frame", "speed_cm_s"])
            for i, frame in enumerate(rows.frames):
                px = rows.speed_px[i]
                cm = px * cm_s_per_px_frame if cm_s_per_px_frame else float("nan")
                w.writerow([frame, rows.labels[i], rows.speed_labels[i], _fmt(px), _fmt(cm)])
        else:
            w.writerow(["frame", "behavior"])
            for i, frame in enumerate(rows.frames):
                w.writerow([frame, rows.labels[i]])


def batch_apply(config, ethogram_csv, model, frame_range=None) -> bool:
    """Score ``config``'s pose CSV in one pass and write the ethogram.

    Returns False when this run needs the streaming pipeline after all — an
    annotated video, a CNN sequence model, or spectral features. Never
    silently degrades: the caller falls back to the streaming path, which
    produces the same artifacts more slowly.
    """
    from pathlib import Path

    from glider.vision.pose.dlc import from_dlc_csv

    if config.pose_csv_in is None or config.output_video is not None:
        return False
    # A CNN sequence model classifies raw keypoint windows, not tabular
    # rolling features, so none of this applies to it.
    if not all(hasattr(model, a) for a in ("spec", "window", "stats", "feature_names")):
        return False

    pose = from_dlc_csv(Path(config.pose_csv_in))
    if config.fps_override:
        pose.fps = float(config.fps_override)
    try:
        rows = classify_pose_data(
            pose,
            model,
            predict_every=config.predict_every,
            confidence_threshold=config.behavior_confidence_threshold,
            class_thresholds=config.behavior_class_thresholds,
            smooth_window=config.smooth_window,
            freeze_threshold=config.freeze_threshold,
            dart_threshold=config.dart_threshold,
            freeze_min_frames=config.freeze_min_frames,
            dart_min_frames=config.dart_min_frames,
            frame_range=frame_range,
        )
    except NotImplementedError as e:
        logger.info("batch apply not usable (%s); falling back to the streaming path", e)
        return False

    speed_axis = config.freeze_threshold is not None and config.dart_threshold is not None
    write_ethogram_csv(
        Path(ethogram_csv),
        rows,
        speed_axis=speed_axis,
        cm_s_per_px_frame=config.cm_s_per_px_frame,
    )
    return True


def _predict(model, frame: pd.DataFrame, confidence_threshold, class_thresholds):
    """One batched predict, honouring the same thresholds ``predict_one`` does."""
    if confidence_threshold > 0 or class_thresholds:
        from glider.analysis.behavior.model import _threshold_decision

        probs = model.classifier.predict_proba(frame)
        return np.asarray(
            _threshold_decision(
                probs, model.classifier.classes_, confidence_threshold, class_thresholds
            ),
            dtype=object,
        )
    return np.asarray(model.classifier.predict(frame), dtype=object)
