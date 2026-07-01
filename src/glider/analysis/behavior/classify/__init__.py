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
    output_video = output_dir / "annotated.mp4"
    ethogram_csv = output_dir / "ethogram_raw.csv"

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
