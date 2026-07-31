"""Qt-free recorded-video behavior classification (the apply path).

Runs a trained :class:`~glider.analysis.behavior.model.BehaviorModel` over a
recorded video to produce a per-frame behavior label series, an annotated
video, and an ethogram. Bouts / stats / transitions are derived by reusing
GLIDER's existing :mod:`glider.analysis.ethogram` primitives rather than a
duplicate implementation.

The engine (`LiveInferencePipeline`) is re-exported here. `ethogram_from_labels`
is thin glue over the existing ethogram primitives, and `classify()` is the
orchestration entry point that runs the pipeline headlessly over a recorded
video and writes bouts/stats/transitions next to the annotated video. The
processing engine is plain ``threading.Thread`` and all heavy deps (cv2 /
torch / ultralytics) are lazy-imported, so importing this package is cheap
and free of any Qt dependency.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from glider.analysis.behavior.classify.pipeline import (
    LiveInferenceConfig,
    LiveInferencePipeline,
)
from glider.analysis.ethogram import (
    compute_bouts,
    compute_intervals,
    compute_state_transitions,
)


def _video_fps(video: Path | str) -> float | None:
    """Frame rate read from the container header, or None if unreadable."""
    from glider.vision.video_source import VideoFileSource

    source = VideoFileSource()
    if not source.load(video):
        return None
    try:
        fps = source.fps
    finally:
        source.release()
    return fps if fps and fps > 0 else None


_MM_PER_CM = 10.0


def find_pose_csv(video: Path | str) -> Path | None:
    """The pose CSV sitting beside *video*, if any.

    Thin re-export of :func:`glider.vision.pose.batch.find_pose_csv`, which
    lives next to the code that chooses the output names. Kept here because
    this name is part of the classify surface.
    """
    from glider.vision.pose.batch import find_pose_csv as _find

    return _find(video)


def _percentile_thresholds(
    video: Path | str, pose_csv: Path | str | None, freeze_pct: float, dart_pct: float
) -> tuple[float, float]:
    """Thresholds from this video's own speed distribution, in px/frame.

    Reuses the same causal speed the live detector computes, so the percentiles
    describe exactly the signal being thresholded. Needs a pose CSV — deriving
    it would mean a second full inference pass.
    """
    from glider.analysis.behavior.classify.speed_state import calibrate_speed_thresholds
    from glider.vision.pose.dlc import from_dlc_csv

    path = Path(pose_csv) if pose_csv is not None else find_pose_csv(video)
    if path is None or not path.exists():
        raise ValueError(
            "percentile thresholds need this video's pose CSV, and none was found "
            f"beside {Path(video).name}. Run Batch Pose Tracking first, or pass "
            "pose_csv explicitly."
        )
    pose = from_dlc_csv(path)
    return calibrate_speed_thresholds(pose.xy, freeze_pct=freeze_pct, dart_pct=dart_pct)


def _min_frames(seconds: float | None, fps: float | None, *, default: int) -> int:
    """Bout minimum in frames. Seconds are the operator-facing unit because a
    frame count means something different at 30 vs 60 fps."""
    if seconds is None:
        return default
    if not fps or fps <= 0:
        raise ValueError("minimum bout durations are in seconds and need the frame rate")
    return max(1, int(round(float(seconds) * float(fps))))


def resolve_speed_thresholds(
    video: Path | str,
    *,
    freeze_cm_s: float | None = None,
    dart_cm_s: float | None = None,
    freeze_mm_s: float | None = None,
    dart_mm_s: float | None = None,
    freeze_pct: float | None = None,
    dart_pct: float | None = None,
    pose_csv: Path | str | None = None,
    freeze_min_s: float | None = None,
    dart_min_s: float | None = None,
    calibration_master: Path | str | None = None,
    px_per_mm: float | None = None,
    fps: float | None = None,
    freeze_threshold: float | None = None,
    dart_threshold: float | None = None,
) -> dict[str, float]:
    """Freeze/dart thresholds in px/frame, ready for :class:`LiveInferenceConfig`.

    Exactly one of three modes, never mixed:

    * **absolute** — ``freeze_cm_s`` / ``dart_cm_s`` (or ``_mm_s``). Comparable
      across sessions and rigs, and converted *exactly*: the live detector
      measures raw pixel displacement, so no body length is involved. Needs a
      scale, from ``px_per_mm`` or the Batch Pose Tracking master calibration
      file at ``calibration_master``, and a frame rate, from ``fps`` or the
      video itself.
    * **percentile** — ``freeze_pct`` / ``dart_pct`` of this video's own causal
      speed distribution, read from its pose CSV (found beside the video, or
      given as ``pose_csv``). Needs no calibration at all and self-adjusts per
      recording, but the thresholds then mean something different in each.
    * **native** — ``freeze_threshold`` / ``dart_threshold`` already in
      px/frame, passed straight through.

    ``freeze_min_s`` / ``dart_min_s`` set the minimum bout duration in seconds,
    converted to frames at the video's rate. Seconds rather than frames because
    a bout minimum is an ethological duration: 30 frames means one second at
    30 fps and half of one at 60.

    Returns ``{}`` when no thresholds were requested, leaving the speed axis
    off. Raises ValueError — rather than silently disabling the axis — when
    thresholds were asked for but cannot be honoured, because a silently
    missing freeze/dart column is worse than a failed run.
    """
    from glider.analysis.behavior.units import load_px_per_mm, mm_per_s_to_px_per_frame

    # cm/s is the operator-facing unit; mm/s stays accepted for callers that
    # already speak it. Normalise to mm/s once, here.
    if freeze_cm_s is not None or dart_cm_s is not None:
        if freeze_mm_s is not None or dart_mm_s is not None:
            raise ValueError("give the absolute thresholds in cm/s or mm/s, not both")
        freeze_mm_s = None if freeze_cm_s is None else float(freeze_cm_s) * _MM_PER_CM
        dart_mm_s = None if dart_cm_s is None else float(dart_cm_s) * _MM_PER_CM

    wants_abs = freeze_mm_s is not None or dart_mm_s is not None
    wants_pct = freeze_pct is not None or dart_pct is not None
    wants_px = freeze_threshold is not None or dart_threshold is not None

    chosen = [
        n
        for n, on in (("absolute", wants_abs), ("percentile", wants_pct), ("native", wants_px))
        if on
    ]
    if len(chosen) > 1:
        raise ValueError(
            "choose one threshold mode: absolute (cm/s or mm/s), percentile, or "
            f"native px/frame — got {' + '.join(chosen)}"
        )
    if not chosen:
        return {}

    rate = fps if fps is not None else _video_fps(video)

    if wants_px:
        if freeze_threshold is None or dart_threshold is None:
            raise ValueError(
                "the speed axis needs both freeze_threshold and dart_threshold; "
                "one alone would silently disable it"
            )
        freeze_px, dart_px = float(freeze_threshold), float(dart_threshold)

    elif wants_pct:
        if freeze_pct is None or dart_pct is None:
            raise ValueError(
                "the speed axis needs both freeze_pct and dart_pct; "
                "one alone would silently disable it"
            )
        if float(freeze_pct) >= float(dart_pct):
            raise ValueError(f"freeze_pct ({freeze_pct}) must be below dart_pct ({dart_pct})")
        # Percentiles describe this video's own distribution, so they need no
        # calibration and no frame rate.
        freeze_px, dart_px = _percentile_thresholds(
            video, pose_csv, float(freeze_pct), float(dart_pct)
        )

    else:
        if freeze_mm_s is None or dart_mm_s is None:
            raise ValueError(
                "the speed axis needs both freeze and dart thresholds; "
                "one alone would silently disable it"
            )
        if float(freeze_mm_s) >= float(dart_mm_s):
            raise ValueError(
                f"the freezing threshold ({freeze_mm_s / _MM_PER_CM:g} cm/s) must be "
                f"below the darting threshold ({dart_mm_s / _MM_PER_CM:g} cm/s)"
            )
        scale = px_per_mm if px_per_mm is not None else load_px_per_mm(calibration_master, video)
        if not scale or scale <= 0:
            raise ValueError(
                "absolute (cm/s) thresholds need a pixel scale: pass px_per_mm, or a "
                "calibration_master whose master file covers this video — or switch "
                "to percentile thresholds, which need no calibration"
            )
        if not rate or rate <= 0:
            raise ValueError(
                f"absolute thresholds need the frame rate, and it could not be read "
                f"from {video}; pass fps explicitly"
            )
        freeze_px = mm_per_s_to_px_per_frame(freeze_mm_s, px_per_mm=scale, fps=rate)
        dart_px = mm_per_s_to_px_per_frame(dart_mm_s, px_per_mm=scale, fps=rate)
        if freeze_px is None or dart_px is None:  # pragma: no cover - guarded above
            raise ValueError("could not convert the thresholds to pixels per frame")

    out: dict[str, float] = {"freeze_threshold": freeze_px, "dart_threshold": dart_px}
    if freeze_min_s is not None:
        out["freeze_min_frames"] = _min_frames(freeze_min_s, rate, default=30)
    if dart_min_s is not None:
        out["dart_min_frames"] = _min_frames(dart_min_s, rate, default=3)
    return out


@dataclass
class EthogramResult:
    """Output of :func:`ethogram_from_labels`."""

    ethogram: pd.DataFrame  # per-frame: frame, time_ms, name
    intervals: pd.DataFrame  # RLE runs: object_id, state, start_ms, end_ms, duration_ms
    bouts: dict[str, pd.Series]  # per-state bout-duration Series (compute_bouts return shape)
    transitions: pd.DataFrame  # from_state, to_state, count


def ethogram_from_labels(labels, fps: float) -> EthogramResult:
    """Per-frame behavior labels -> ethogram/intervals/bouts/transitions,
    reusing glider.analysis.ethogram (no duplicate bout/transition logic)."""
    frames = list(range(len(labels)))
    time_ms = [f / fps * 1000.0 for f in frames]
    etho = pd.DataFrame({"frame": frames, "time_ms": time_ms, "name": list(labels)})
    tracking = pd.DataFrame(
        {
            "object_id": 0,
            "behavioral_state": list(labels),
            "flow_elapsed_ms": time_ms,
        }
    )
    intervals = compute_intervals(tracking)
    bouts = compute_bouts(intervals)  # dict[state -> Series of durations]
    transitions = compute_state_transitions(intervals)
    return EthogramResult(ethogram=etho, intervals=intervals, bouts=bouts, transitions=transitions)


def classify(
    video,
    model_path,
    yolo_path,
    keypoint_names,
    output_dir,
    *,
    device=None,
    freeze_cm_s=None,
    dart_cm_s=None,
    freeze_mm_s=None,
    dart_mm_s=None,
    freeze_pct=None,
    dart_pct=None,
    pose_csv=None,
    freeze_min_s=None,
    dart_min_s=None,
    calibration_master=None,
    px_per_mm=None,
    write_annotated=False,
    **opts,
) -> EthogramResult:
    """Run the headless apply pipeline over a recorded video and write outputs.

    Drives :class:`LiveInferencePipeline` with ``display=False`` to produce
    an annotated mp4 and a per-frame ethogram CSV (written by
    ``BehaviorClassifier``/``SequenceClassifier`` as ``frame, behavior[,
    speed]`` rows, emitted at the configured ``predict_every`` cadence — not
    one row per raw video frame). We read that CSV back to recover the
    ordered label sequence (assumption: rows are appended in increasing
    frame order, which holds because the classifier thread processes frames
    sequentially), then hand the label list to :func:`ethogram_from_labels`
    using the *effective* per-label sample rate (``producer.fps /
    predict_every``) so the reconstructed interval timings line up with the
    real video clock.

    Writes ``bouts.csv``, ``stats.csv``, and ``transitions.csv`` into
    ``output_dir`` (created if missing), alongside the pipeline's own
    ``annotated.mp4`` / ``ethogram_raw.csv``. Returns the
    :class:`EthogramResult`.

    Extra ``LiveInferenceConfig`` knobs (e.g. ``conf_threshold``,
    ``smooth_window``, ``predict_every``, ``behavior_confidence_threshold``,
    ``fps_override``) can be passed through ``**opts``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Encoding an annotated MP4 costs more wall-clock than the inference on a
    # long recording, and it is a spot-checking aid rather than an analysis
    # artifact -- so it is opt-in.
    output_video = output_dir / "annotated.mp4" if write_annotated else None
    ethogram_csv = output_dir / "ethogram_raw.csv"

    # A pixel scale is worth having even when the thresholds did not need one:
    # percentile mode derives its cut-offs from the video's own distribution,
    # but the operator still wants the ethogram's speed in real units.
    from glider.analysis.behavior.units import load_px_per_mm

    scale = px_per_mm if px_per_mm is not None else load_px_per_mm(calibration_master, video)
    rate = opts.get("fps_override") or _video_fps(video)
    if scale and scale > 0 and rate and rate > 0:
        # px/frame -> px/s -> mm/s -> cm/s, folded into one factor.
        opts.setdefault("cm_s_per_px_frame", rate / scale / 10.0)

    # Resolved before the pipeline starts so a bad threshold or a missing
    # calibration fails immediately, not after a full pass of inference.
    opts.update(
        resolve_speed_thresholds(
            video,
            freeze_cm_s=freeze_cm_s,
            dart_cm_s=dart_cm_s,
            freeze_mm_s=freeze_mm_s,
            dart_mm_s=dart_mm_s,
            freeze_pct=freeze_pct,
            dart_pct=dart_pct,
            pose_csv=pose_csv,
            freeze_min_s=freeze_min_s,
            dart_min_s=dart_min_s,
            calibration_master=calibration_master,
            px_per_mm=px_per_mm,
            fps=opts.get("fps_override"),
            freeze_threshold=opts.pop("freeze_threshold", None),
            dart_threshold=opts.pop("dart_threshold", None),
        )
    )

    config = LiveInferenceConfig(
        source=str(video),
        keypoint_names=list(keypoint_names),
        yolo_model_path=yolo_path,
        behavior_model_path=model_path,
        display=False,
        output_video=output_video,
        ethogram_csv=ethogram_csv,
        device=device,
        **opts,
    )
    pipeline = LiveInferencePipeline(config)
    pipeline.run()

    # The classifier threads only write the ethogram CSV once at least one
    # prediction has been buffered. A too-short clip (never fills the feature
    # history) or a source that failed to open therefore leaves no file. Fail
    # loudly if the producer/tracker recorded an error; otherwise treat it as
    # an empty run so callers still get a valid (empty) EthogramResult instead
    # of an opaque FileNotFoundError.
    if ethogram_csv.exists():
        with ethogram_csv.open(newline="") as f:
            labels = [row["behavior"] for row in csv.DictReader(f)]
    else:
        err = getattr(getattr(pipeline, "producer", None), "error", None) or getattr(
            getattr(pipeline, "tracker", None), "error", None
        )
        if err:
            raise RuntimeError(f"classify(): pipeline produced no ethogram output: {err}")
        labels = []

    video_fps = getattr(pipeline.producer, "fps", None) or config.fps_override or 30.0
    predict_every = max(1, int(config.predict_every))
    effective_fps = video_fps / predict_every

    result = ethogram_from_labels(labels, fps=effective_fps)

    # bouts.csv: one row per bout (state, duration_s).
    bout_rows = [
        {"state": state, "duration_s": duration_ms / 1000.0}
        for state, series in result.bouts.items()
        for duration_ms in series
    ]
    pd.DataFrame(bout_rows, columns=["state", "duration_s"]).to_csv(
        output_dir / "bouts.csv", index=False
    )

    # stats.csv: per-state summary derived from the bouts dict.
    total_session_s = len(labels) / effective_fps if effective_fps else 0.0
    stats_rows = []
    for state, series in result.bouts.items():
        total_s = series.sum() / 1000.0
        stats_rows.append(
            {
                "state": state,
                "n_bouts": len(series),
                "total_s": total_s,
                "fraction": total_s / total_session_s if total_session_s else 0.0,
                "mean_s": series.mean() / 1000.0,
                "median_s": series.median() / 1000.0,
            }
        )
    pd.DataFrame(
        stats_rows, columns=["state", "n_bouts", "total_s", "fraction", "mean_s", "median_s"]
    ).to_csv(output_dir / "stats.csv", index=False)

    result.transitions.to_csv(output_dir / "transitions.csv", index=False)

    return result


__all__ = [
    "LiveInferenceConfig",
    "LiveInferencePipeline",
    "EthogramResult",
    "ethogram_from_labels",
    "classify",
]
