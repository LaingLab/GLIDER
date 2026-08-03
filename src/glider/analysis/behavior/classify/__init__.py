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
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from glider.analysis.behavior.classify.pipeline import (
    LiveInferenceConfig,
    LiveInferencePipeline,
)
from glider.analysis.ethogram import (
    UNSCORED,
    compute_bouts,
    compute_intervals,
    compute_state_transitions,
)

logger = logging.getLogger(__name__)

#: How unscored time is named in stats.csv. A blank cell reads as a bug and
#: sorts like a behavior; this says what it is.
UNSCORED_LABEL = "(unscored)"


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


def find_pose_csv(video: Path | str, search_dir: Path | str | None = None) -> Path | None:
    """The pose CSV belonging to *video*, if any.

    Thin re-export of :func:`glider.vision.pose.batch.find_pose_csv`, which
    lives next to the code that chooses the output names. Kept here because
    this name is part of the classify surface. ``search_dir`` defaults to the
    video's own directory.
    """
    from glider.vision.pose.batch import find_pose_csv as _find

    return _find(video, search_dir)


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
    cohort_thresholds: Path | str | None = None,
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
    wants_cohort = cohort_thresholds is not None

    chosen = [
        n
        for n, on in (
            ("absolute", wants_abs),
            ("percentile", wants_pct),
            ("cohort", wants_cohort),
            ("native", wants_px),
        )
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

    elif wants_cohort:
        # One physical cut-off derived from the whole cohort, converted here
        # through THIS video's scale and rate. Per-video percentiles would
        # judge each animal against only itself, which is circular in a
        # treatment study.
        from glider.analysis.behavior.cohort_speed import CohortSpeedThresholds

        cohort = CohortSpeedThresholds.load(cohort_thresholds)
        scale = px_per_mm if px_per_mm is not None else load_px_per_mm(calibration_master, video)
        freeze_px, dart_px = cohort.to_px_per_frame(px_per_mm=scale, fps=rate)

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
    cohort_thresholds=None,
    write_annotated=False,
    write_pose_csv=True,
    pose_csv_in=None,
    reuse_existing_poses=False,
    pose_dir=None,
    min_bout_s=None,
    start_s=None,
    end_s=None,
    model=None,
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

    ``model`` accepts an already-loaded bundle. Scoring many videos in one
    process otherwise re-reads it per video, which is both slow (seconds each)
    and, for bundles carrying an unrebuildable umap index, a way to corrupt
    native memory by unpickling the same broken object repeatedly.

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
    # Named the way Batch Pose Tracking names its output, so the same
    # discovery and cohort tooling recognises it. The poses are computed
    # regardless; keeping them saves the whole pass next time.
    pose_csv_out = (
        output_dir / f"{Path(video).stem}DLC_{Path(yolo_path).stem}.csv" if write_pose_csv else None
    )

    # Reuse poses rather than re-deriving them. Running the pose model again
    # to reproduce numbers Batch Pose Tracking already wrote is the single
    # biggest avoidable cost in an apply run. Resolved before anything
    # expensive starts, so a missing file fails immediately.
    # ``pose_dir`` lets one folder of pose CSVs serve a whole cohort. Batch
    # Pose Tracking writes them wherever the operator pointed it, which is
    # routinely not beside the videos, and copying 30 CSVs by hand to satisfy
    # a default is a chore that invites mistakes.
    if pose_csv_in is None and reuse_existing_poses:
        pose_csv_in = find_pose_csv(video, pose_dir)
    if pose_csv_in is not None:
        pose_csv_in = Path(pose_csv_in)
        if not pose_csv_in.exists():
            raise ValueError(f"pose CSV not found: {pose_csv_in}")
        # Nothing new was tracked, so there is nothing new to write.
        pose_csv_out = None

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
            cohort_thresholds=cohort_thresholds,
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
        pose_csv_out=pose_csv_out,
        pose_csv_in=pose_csv_in,
        device=device,
        **opts,
    )
    # With the poses already tracked and no annotated video to render, every
    # frame is known up front and there is nothing to stream. Scoring the
    # session in one vectorised pass is ~700x faster than pushing it through
    # the realtime threads, and it computes `body_angular_velocity` over the
    # whole session the way training does -- the five-frame streaming window
    # cannot, because angle unwrapping is cumulative, not a local stencil.
    # Analysing a stretch of each recording — a drug window, the minutes after
    # a stimulus — rather than the whole thing. Resolved against this video's
    # own rate, so one setting means the same clock time on a 30 fps and a
    # 60 fps recording.
    frame_range = _frame_range(start_s, end_s, rate, Path(video).name)

    pipeline = None
    used_batch = False
    if pose_csv_in is not None and output_video is None:
        from glider.analysis.behavior.classify import batch as _batch
        from glider.analysis.behavior.classify.pipeline import _load_behavior_model

        if model is None:
            model = _load_behavior_model(model_path)
        used_batch = _batch.batch_apply(config, ethogram_csv, model, frame_range=frame_range)
    if not used_batch:
        # Hand over the bundle only if we already loaded it: unpickling some
        # bundles twice in one process corrupts native memory.
        handover = {"model": model} if model is not None else {}
        pipeline = LiveInferencePipeline(config, **handover)
        pipeline.run()

    # The classifier threads only write the ethogram CSV once at least one
    # prediction has been buffered. A too-short clip (never fills the feature
    # history) or a source that failed to open therefore leaves no file. Fail
    # loudly if the producer/tracker recorded an error; otherwise treat it as
    # an empty run so callers still get a valid (empty) EthogramResult instead
    # of an opaque FileNotFoundError.
    if ethogram_csv.exists():
        with ethogram_csv.open(newline="") as f:
            rows_in = list(csv.DictReader(f))
        if frame_range is not None and not used_batch:
            # The streaming path scored the whole video — it has to, since the
            # annotated frames come off the same queue — so the window is
            # applied to what it wrote. Same rows either way; only the batch
            # path also saves the work.
            rows_in = _rows_in_range(rows_in, frame_range)
            _rewrite_ethogram(ethogram_csv, rows_in)
        labels = [row["behavior"] for row in rows_in]
    else:
        err = getattr(getattr(pipeline, "producer", None), "error", None) or getattr(
            getattr(pipeline, "tracker", None), "error", None
        )
        if err:
            raise RuntimeError(f"classify(): pipeline produced no ethogram output: {err}")
        labels = []

    video_fps = (
        getattr(getattr(pipeline, "producer", None), "fps", None) or config.fps_override or rate
    )
    predict_every = max(1, int(config.predict_every))
    effective_fps = video_fps / predict_every

    result = ethogram_from_labels(labels, fps=effective_fps)

    # bouts.csv: one row per bout (state, duration_s). Unscored frames are the
    # absence of a label, not a behavior, so they get no bouts — otherwise
    # "how long is a grooming bout" is answered partly by tracking dropout.
    #
    # min_bout_s drops bouts below an ethological floor. It filters the
    # summaries only: ethogram_raw.csv keeps every frame as classified, so
    # the decision stays visible and reversible rather than baked into the
    # one artifact everything else is derived from.
    floor_ms = float(min_bout_s) * 1000.0 if min_bout_s else 0.0
    bout_rows = [
        {"state": state, "duration_s": duration_ms / 1000.0}
        for state, series in result.bouts.items()
        if state != UNSCORED
        for duration_ms in series
        if duration_ms >= floor_ms
    ]
    pd.DataFrame(bout_rows, columns=["state", "duration_s"]).to_csv(
        output_dir / "bouts.csv", index=False
    )

    # stats.csv: per-state summary. Unscored time is still reported, under an
    # explicit name rather than a blank one, so the fractions account for the
    # whole session and coverage is visible instead of implied.
    total_session_s = len(labels) / effective_fps if effective_fps else 0.0
    stats_rows = []
    for state, series in result.bouts.items():
        if floor_ms and state != UNSCORED:
            series = series[series >= floor_ms]
        if series.empty:
            continue
        total_s = series.sum() / 1000.0
        stats_rows.append(
            {
                "state": UNSCORED_LABEL if state == UNSCORED else state,
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

    # No separate speed summary: freezing and darting are states of `behavior`
    # now, so bouts.csv and stats.csv already carry their bouts, totals and
    # fractions alongside every other behaviour. A second pair of files saying
    # the same thing is a second answer to a question that has one.

    # run.json: what produced these files. Written last, so it only appears
    # beside a complete set of outputs.
    #
    # The pose CSV is the load-bearing entry. Reusing existing poses means
    # nothing is copied into the output folder — right, since the CSV is tens
    # of megabytes — but it also left the session review window with nothing
    # to find, because it only ever looked beside the ethogram. Recording the
    # path is cheaper than a copy and says more.
    _write_run_manifest(
        output_dir,
        video=video,
        pose_csv=pose_csv_in,
        model_path=model_path,
        yolo_path=yolo_path,
        keypoint_names=keypoint_names,
        fps=video_fps,
        predict_every=predict_every,
        smooth_window=config.smooth_window,
        min_bout_s=min_bout_s,
        freeze_threshold=config.freeze_threshold,
        dart_threshold=config.dart_threshold,
        cm_s_per_px_frame=config.cm_s_per_px_frame,
        px_per_mm=scale,
        used_batch=used_batch,
    )

    return result


def _frame_range(
    start_s: float | None, end_s: float | None, fps: float | None, name: str
) -> tuple[int, int] | None:
    """``(first_frame, last_frame)`` inclusive for a time window, or None.

    Seconds rather than frames because a window is a fact about the
    experiment, not about the recording: "minutes two to seven" means the
    same stretch of an animal's session whether it was filmed at 30 or 60 fps.
    """
    if start_s is None and end_s is None:
        return None
    if not fps or fps <= 0:
        raise ValueError(
            f"a time range is in seconds and needs the frame rate, which could "
            f"not be read from {name}; pass fps_override"
        )
    first = int(round(float(start_s or 0.0) * fps))
    last = int(round(float(end_s) * fps)) - 1 if end_s is not None else 2**31
    if last < first:
        raise ValueError(
            f"the analysis window ends before it starts: "
            f"{float(start_s or 0.0):g} s to {float(end_s):g} s"
        )
    return max(0, first), last


def _rows_in_range(rows: list[dict], frame_range: tuple[int, int]) -> list[dict]:
    """Ethogram rows whose frame falls inside an inclusive window."""
    first, last = frame_range
    kept = []
    for row in rows:
        try:
            frame = int(row.get("frame", -1))
        except (TypeError, ValueError):
            continue
        if first <= frame <= last:
            kept.append(row)
    return kept


def _rewrite_ethogram(path: Path, rows: list[dict]) -> None:
    """Replace an ethogram with a subset of its own rows, header intact."""
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_manifest(output_dir: Path, **fields) -> Path | None:
    """Record how these outputs were produced, beside them.

    Best-effort: an unwritable manifest must not fail a finished run, and
    every reader treats it as optional.
    """
    import json

    payload = {"schema_version": 1}
    for key, value in fields.items():
        payload[key] = str(value) if isinstance(value, Path) else value
    path = output_dir / "run.json"
    try:
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    except OSError as e:  # pragma: no cover - depends on filesystem state
        logger.info("could not write %s: %s", path, e)
        return None
    return path


def read_run_manifest(output_dir: Path | str) -> dict | None:
    """The ``run.json`` beside a set of outputs, or None if absent/unusable."""
    import json

    try:
        data = json.loads((Path(output_dir) / "run.json").read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


__all__ = [
    "LiveInferenceConfig",
    "LiveInferencePipeline",
    "EthogramResult",
    "ethogram_from_labels",
    "classify",
]
