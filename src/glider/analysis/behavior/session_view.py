"""One analysed session, and statistics over any window of it.

An apply run leaves several files beside each other — the ethogram, the poses,
the pose sidecar, the rig's calibration. Separately each is a fragment; taken
together they describe a session well enough to scrub through, select a window
of, and ask questions about. This gathers them.

Qt-free on purpose: the loading, the slicing and every statistic are testable
without a display, and the window on top only draws.

Poses are the fallback that makes this useful at all. Annotated video is
expensive and usually not kept, but the pose CSV is small and almost always
survives — so a session can be replayed as moving keypoints long after the
pixels are gone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from glider.analysis.ethogram import UNSCORED

logger = logging.getLogger(__name__)

__all__ = ["SegmentStats", "SessionView", "SessionViewError"]


class SessionViewError(ValueError):
    """A session's files could not be loaded."""


@dataclass
class SegmentStats:
    """What a chosen window of a session contains."""

    start_frame: int
    end_frame: int
    fps: float
    duration_s: float
    bouts: pd.DataFrame  # state, n_bouts, total_s, fraction, mean_s, median_s
    distance_cm: float | None
    mean_speed_cm_s: float | None
    peak_speed_cm_s: float | None
    freeze_threshold: float | None  # this window's own percentile
    dart_threshold: float | None
    threshold_unit: str = ""


@dataclass
class SessionView:
    """Everything loadable about one analysed session."""

    labels: list[str]  # behavior per ethogram row
    frames: np.ndarray  # the frame index each row describes
    fps: float
    xy: np.ndarray | None = None  # (n_frames, k, 2) poses, if a CSV was found
    keypoint_names: list[str] = field(default_factory=list)
    resolution: tuple[int, int] | None = None
    px_per_mm: float | None = None
    body_axis: tuple[int, int] = (0, -1)
    source: Path | None = None
    pose_path: Path | None = None  # which CSV the poses came from

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        ethogram_csv: Path | str,
        *,
        pose_csv: Path | str | None = None,
        calibration_master: Path | str | None = None,
        video: Path | str | None = None,
    ) -> SessionView:
        """Load an ethogram and whatever else sits with it.

        Only the ethogram is required. A missing pose CSV costs the keypoint
        view but not the behaviour statistics; a missing calibration costs the
        real-world units but not the rest.
        """
        ethogram_csv = Path(ethogram_csv)
        try:
            etho = pd.read_csv(ethogram_csv)
        except (OSError, ValueError) as e:
            raise SessionViewError(f"cannot read {ethogram_csv.name}: {e}") from e
        if "behavior" not in etho.columns:
            raise SessionViewError(
                f"{ethogram_csv.name} has no 'behavior' column; is it an ethogram?"
            )

        labels = [("" if pd.isna(v) else str(v)) for v in etho["behavior"]]
        frames = (
            etho["frame"].to_numpy(dtype=int) if "frame" in etho.columns else np.arange(len(labels))
        )

        view = cls(labels=labels, frames=frames, fps=30.0, source=ethogram_csv)
        view._load_poses(ethogram_csv, pose_csv)
        view._load_scale(calibration_master, video, ethogram_csv)
        return view

    def _load_poses(self, ethogram_csv: Path, pose_csv: Path | str | None) -> None:
        from glider.vision.pose.dlc import from_dlc_csv, resolution_for_csv

        if pose_csv is None:
            # An apply run writes the poses beside the ethogram.
            candidates = sorted(ethogram_csv.parent.glob("*DLC_*.csv"))
            pose_csv = candidates[0] if candidates else None
        if pose_csv is None:
            return
        try:
            pose = from_dlc_csv(Path(pose_csv))
        except Exception as e:  # noqa: BLE001 - poses are optional
            logger.info("no usable poses beside %s: %s", ethogram_csv.name, e)
            return
        self.xy = pose.xy
        self.keypoint_names = list(pose.keypoint_names)
        self.fps = float(pose.fps) or self.fps
        self.pose_path = Path(pose_csv)
        self.resolution = resolution_for_csv(self.pose_path)
        if self.keypoint_names:
            self.body_axis = (0, len(self.keypoint_names) - 1)

    def _load_scale(self, calibration_master, video, ethogram_csv: Path) -> None:
        from glider.analysis.behavior.units import load_px_per_mm

        if calibration_master is None:
            beside = ethogram_csv.parent / "pose_calibration.json"
            calibration_master = beside if beside.exists() else None
        if calibration_master is None:
            return
        target = video
        if target is None:
            # The ethogram lives in <output>/<video stem>/, so the folder names
            # the session even when the video itself is elsewhere.
            target = ethogram_csv.parent / f"{ethogram_csv.parent.name}.mp4"
        self.px_per_mm = load_px_per_mm(calibration_master, target)

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self.labels)

    @property
    def duration_s(self) -> float:
        return self.n_rows / self.fps if self.fps else 0.0

    def centroid(self) -> np.ndarray | None:
        """Per-frame body centre, or None without poses.

        The mean of the tracked points rather than any single keypoint: a
        dropped nose should nudge the centre, not teleport it.
        """
        if self.xy is None:
            return None
        with np.errstate(invalid="ignore"):
            return np.nanmean(self.xy, axis=1)

    def label_at(self, frame: int) -> str:
        """The behaviour covering *frame*, or '' beyond the ethogram."""
        if self.n_rows == 0:
            return ""
        idx = int(np.searchsorted(self.frames, frame, side="right")) - 1
        return self.labels[idx] if 0 <= idx < self.n_rows else ""

    def trail(self, frame: int, seconds: float = 5.0) -> np.ndarray | None:
        """Centroid positions over the *seconds* leading up to *frame*."""
        centre = self.centroid()
        if centre is None:
            return None
        span = max(1, int(round(seconds * self.fps)))
        lo = max(0, frame - span + 1)
        segment = centre[lo : frame + 1]
        return segment[~np.isnan(segment).any(axis=1)]

    # ------------------------------------------------------------------
    # segments
    # ------------------------------------------------------------------

    def segment_stats(
        self, start_frame: int, end_frame: int, *, freeze_pct: float = 10.0, dart_pct: float = 99.5
    ) -> SegmentStats:
        """Behaviour and locomotion over ``[start_frame, end_frame]``.

        The thresholds returned are what this window *alone* would have
        produced. They are informational: the labels are whatever the run that
        wrote the ethogram decided, and are not recomputed here.
        """
        from glider.analysis.ethogram import compute_bouts, compute_intervals

        start_frame, end_frame = sorted((int(start_frame), int(end_frame)))
        rows = np.where((self.frames >= start_frame) & (self.frames <= end_frame))[0]
        labels = [self.labels[i] for i in rows]
        duration = (end_frame - start_frame + 1) / self.fps if self.fps else 0.0

        bouts = self._bout_table(labels, compute_intervals, compute_bouts)
        distance, mean_speed, peak_speed = self._locomotion(start_frame, end_frame)
        freeze, dart, unit = self._window_thresholds(start_frame, end_frame, freeze_pct, dart_pct)
        return SegmentStats(
            start_frame=start_frame,
            end_frame=end_frame,
            fps=self.fps,
            duration_s=duration,
            bouts=bouts,
            distance_cm=distance,
            mean_speed_cm_s=mean_speed,
            peak_speed_cm_s=peak_speed,
            freeze_threshold=freeze,
            dart_threshold=dart,
            threshold_unit=unit,
        )

    def _bout_table(self, labels, compute_intervals, compute_bouts) -> pd.DataFrame:
        """Per-state bout tally for a label run, reusing the ethogram primitives."""
        if not labels:
            return pd.DataFrame(
                columns=["state", "n_bouts", "total_s", "fraction", "mean_s", "median_s"]
            )
        # The rows are emitted at the classifier's cadence, not per frame, so
        # timestamps come from the frame indices rather than the row count.
        elapsed_ms = [i / self.fps * 1000.0 for i in range(len(labels))]
        tracking = pd.DataFrame(
            {"object_id": 0, "behavioral_state": labels, "flow_elapsed_ms": elapsed_ms}
        )
        intervals = compute_intervals(tracking)
        durations = compute_bouts(intervals)
        # The window, not the sum of scored bouts, so a session with dropout
        # reports fractions that still add up to its real duration.
        window_ms = sum(float(s.sum()) for s in durations.values()) or 1.0
        rows = []
        for state, series in durations.items():
            if state == UNSCORED:
                # Not a behavior: it is the classifier declining to answer.
                continue
            # compute_bouts speaks milliseconds; these columns are seconds.
            rows.append(
                {
                    "state": state,
                    "n_bouts": int(len(series)),
                    "total_s": float(series.sum()) / 1000.0,
                    "fraction": float(series.sum()) / window_ms,
                    "mean_s": float(series.mean()) / 1000.0 if len(series) else 0.0,
                    "median_s": float(series.median()) / 1000.0 if len(series) else 0.0,
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=["state", "n_bouts", "total_s", "fraction", "mean_s", "median_s"]
            )
        return pd.DataFrame(rows).sort_values("total_s", ascending=False, ignore_index=True)

    def _locomotion(self, start_frame: int, end_frame: int):
        """``(distance_cm, mean_cm_s, peak_cm_s)`` — None without a calibration."""
        centre = self.centroid()
        if centre is None or not self.px_per_mm or self.px_per_mm <= 0:
            return None, None, None
        track = centre[start_frame : end_frame + 1]
        track = track[~np.isnan(track).any(axis=1)]
        if len(track) < 2:
            return None, None, None
        steps_px = np.linalg.norm(np.diff(track, axis=0), axis=1)
        # px -> mm -> cm
        steps_cm = steps_px / self.px_per_mm / 10.0
        speeds = steps_cm * self.fps
        return float(steps_cm.sum()), float(speeds.mean()), float(speeds.max())

    def _window_thresholds(self, start_frame, end_frame, freeze_pct, dart_pct):
        """What this window alone would have chosen as freeze/dart cut-offs."""
        if self.xy is None:
            return None, None, ""
        from glider.analysis.behavior.classify.speed_state import CausalSpeed

        window = self.xy[start_frame : end_frame + 1]
        if len(window) < 3:
            return None, None, ""
        causal = CausalSpeed()
        speeds = np.asarray([causal.push(frame) for frame in window], dtype=np.float64)
        speeds = speeds[1:][np.isfinite(speeds[1:])]
        if speeds.size == 0:
            return None, None, ""
        if self.px_per_mm and self.px_per_mm > 0 and self.fps:
            speeds = speeds * self.fps / self.px_per_mm / 10.0
            unit = "cm/s"
        else:
            unit = "px/frame"
        return (
            float(np.percentile(speeds, freeze_pct)),
            float(np.percentile(speeds, dart_pct)),
            unit,
        )
