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

__all__ = [
    "SegmentStats",
    "SessionView",
    "SessionViewError",
    "find_session_poses",
    "find_session_video",
]


class SessionViewError(ValueError):
    """A session's files could not be loaded."""


#: How far up from the ethogram to look for the poses. An apply run writes
#: ``<output>/<video stem>/ethogram_raw.csv`` while the poses usually stay
#: with the videos, which is commonly the grandparent of that folder.
_SEARCH_LEVELS = 3

#: Frames by which a video may fall short of the session and still be treated
#: as frame-aligned. A couple of trailing frames go missing whenever a writer
#: closes before the last ones arrive; more than that is dropped frames, which
#: shift everything after them.
_ALIGNMENT_SLACK = 5


def _numeric_column(frame: pd.DataFrame, name: str, length: int) -> np.ndarray | None:
    """A float column from an ethogram, or None if it isn't there.

    Blank cells are NaN, not zero: the writer leaves a cell empty when the
    value was unknown (a dropout frame, or no pixel scale), and reading that
    as a real 0.0 would put a motionless animal in every distance total.
    """
    if name not in frame.columns:
        return None
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    if values.size != length:
        return None
    return None if np.isnan(values).all() else values


def _find_upward(start: Path, name: str) -> Path | None:
    """``name`` in *start* or a few folders above it, or None.

    Outputs are written into a subfolder per video, so the rig-level files an
    apply run used — the master calibration in particular — normally sit a
    level or two up rather than beside the ethogram.
    """
    folder = start
    for _ in range(_SEARCH_LEVELS + 1):
        candidate = folder / name
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            return None
        if folder == folder.parent:
            break
        folder = folder.parent
    return None


def find_session_video(ethogram_csv: Path | str, pose_path: Path | None = None) -> Path | None:
    """A video whose frame *n* is this session's frame *n*, if one exists.

    The source video is preferred over the run's own ``annotated.mp4``, which
    is not safe to scrub by index: the annotated file is written from the
    display queue, and that queue drops frames under back-pressure without
    padding the output. A run that dropped nine frames of forty-five thousand
    yields a video whose every later frame is nine early — a drift nothing on
    screen would reveal. The annotated file is still offered as a fallback,
    and :func:`video_is_aligned` is what decides whether to believe it.
    """
    ethogram_csv = Path(ethogram_csv)
    folder = ethogram_csv.parent

    from glider.analysis.behavior.classify import read_run_manifest

    recorded = (read_run_manifest(folder) or {}).get("video")
    if recorded:
        try:
            if Path(recorded).is_file():
                return Path(recorded)
        except OSError:
            logger.info("the video recorded in run.json is unreachable: %s", recorded)

    stem = folder.name
    roots = [pose_path.parent] if pose_path is not None else []
    roots += [folder, folder.parent, folder.parent.parent]
    for root in roots:
        for suffix in (".mp4", ".avi", ".mov", ".mkv"):
            candidate = root / f"{stem}{suffix}"
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue

    annotated = folder / "annotated.mp4"
    try:
        return annotated if annotated.is_file() else None
    except OSError:
        return None


def find_session_poses(ethogram_csv: Path | str) -> Path | None:
    """The pose CSV belonging to an apply run's ethogram, if it can be found.

    Resolution order, most trustworthy first:

    1. The path recorded in ``run.json`` beside the ethogram. Reusing already
       tracked poses copies nothing into the output folder — the CSV is tens
       of megabytes — so the recorded path is the only thing that knows where
       they went.
    2. A pose CSV sitting beside the ethogram, which is what a run that did
       its own tracking leaves.
    3. A pose CSV named for this session, in the folders above. The output
       folder is named after the video, so the search is anchored on that
       stem: it can find ``videos/t4.csv`` from ``videos/out/t4/`` but can
       never pick up a different animal's poses.
    """
    ethogram_csv = Path(ethogram_csv)
    folder = ethogram_csv.parent

    from glider.analysis.behavior.classify import read_run_manifest

    manifest = read_run_manifest(folder) or {}
    recorded = manifest.get("pose_csv")
    if recorded:
        path = Path(recorded)
        try:
            if path.is_file():
                return path
        except OSError:
            logger.info("the pose CSV recorded in run.json is unreachable: %s", recorded)

    beside = sorted(folder.glob("*DLC_*.csv"))
    if beside:
        return beside[0]

    # The output folder carries the video's name; so does the pose CSV.
    from glider.vision.pose.batch import find_pose_csv

    stem = folder.name
    parent = folder.parent
    for _ in range(_SEARCH_LEVELS):
        if not parent or parent == parent.parent:
            break
        try:
            found = find_pose_csv(parent / f"{stem}.mp4", parent)
        except OSError:
            found = None
        if found is not None:
            logger.info("found %s for %s by searching upward", found.name, ethogram_csv)
            return found
        parent = parent.parent
    return None


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
    video_path: Path | None = None  # a video to scrub alongside, if one exists
    video_frames: int = 0  # its length, for the alignment check
    # The measured speed behind the labels. Not a second scoring: `labels`
    # already has freezing and darting folded in, and these are what distance
    # and velocity are computed from.
    speed_px: np.ndarray | None = None
    speed_cm_s: np.ndarray | None = None

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

        speed_px = _numeric_column(etho, "speed_px_frame", len(labels))
        speed_cm_s = _numeric_column(etho, "speed_cm_s", len(labels))

        view = cls(
            labels=labels,
            frames=frames,
            fps=30.0,
            source=ethogram_csv,
            speed_px=speed_px,
            speed_cm_s=speed_cm_s,
        )
        view._load_poses(ethogram_csv, pose_csv)
        view._load_scale(calibration_master, video, ethogram_csv)
        view._load_video(ethogram_csv, video)
        return view

    def _load_video(self, ethogram_csv: Path, video: Path | str | None) -> None:
        """Find a video to scrub alongside the ethogram, and measure it."""
        from glider.vision.video_source import VideoFileSource

        path = (
            Path(video) if video is not None else find_session_video(ethogram_csv, self.pose_path)
        )
        if path is None:
            return
        source = VideoFileSource()
        if not source.load(path):
            logger.info("found %s but could not open it", path)
            return
        try:
            self.video_path = path
            self.video_frames = int(source.frame_count)
            # The video is the authority on its own size; a pose sidecar that
            # never recorded one no longer costs the arena.
            if self.resolution is None and all(source.resolution):
                self.resolution = source.resolution
        finally:
            source.release()

    @property
    def video_is_aligned(self) -> bool:
        """Whether video frame *n* is this session's frame *n*.

        A run's ``annotated.mp4`` is written from a queue that drops frames
        under load, so it can be shorter than the session by an unknown amount
        — and every frame after the first drop is then offset. Short by a hair
        is normal (the last frames arrive after the writer closes); short by
        more means the indices no longer line up and the viewer must say so
        rather than show a confidently wrong frame.
        """
        if not self.video_frames or self.n_rows == 0:
            return False
        expected = int(self.frames[-1]) + 1
        return abs(self.video_frames - expected) <= _ALIGNMENT_SLACK

    def _load_poses(self, ethogram_csv: Path, pose_csv: Path | str | None) -> None:
        from glider.vision.pose.dlc import from_dlc_csv, resolution_for_csv

        if pose_csv is None:
            pose_csv = find_session_poses(ethogram_csv)
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
        from glider.analysis.behavior.classify import read_run_manifest
        from glider.analysis.behavior.units import load_px_per_mm

        if calibration_master is None:
            # The run recorded the scale it actually used, which beats
            # re-deriving one and beats searching for the file.
            recorded = (read_run_manifest(ethogram_csv.parent) or {}).get("px_per_mm")
            if recorded:
                self.px_per_mm = float(recorded)
                return
            calibration_master = _find_upward(ethogram_csv.parent, "pose_calibration.json")
        if calibration_master is None:
            return
        # The ethogram lives in <output>/<video stem>/, so the folder names the
        # session — but the video itself is almost never in there. Look where
        # the session's other files actually turned up: beside its poses, then
        # beside the calibration that covers it. Handing the lookup a path
        # inside the output folder finds nothing and silently costs the
        # real-world units.
        stem = ethogram_csv.parent.name
        candidates = [video] if video is not None else []
        if self.pose_path is not None:
            candidates.append(self.pose_path.parent / f"{stem}.mp4")
        candidates.append(Path(calibration_master).parent / f"{stem}.mp4")
        candidates.append(ethogram_csv.parent / f"{stem}.mp4")

        for target in candidates:
            scale = load_px_per_mm(calibration_master, target)
            if scale:
                self.px_per_mm = float(scale)
                return

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
        import warnings

        with warnings.catch_warnings():
            # A fully dropped frame is an all-NaN slice; NaN is the right
            # answer for it, so the warning is noise on every session with
            # any dropout at all.
            warnings.filterwarnings("ignore", r"Mean of empty slice", RuntimeWarning)
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
        """``(distance_cm, mean_cm_s, peak_cm_s)`` — None without a calibration.

        Prefers the speed the run itself recorded. That column is the signal
        the freeze/dart thresholds were actually applied to — smoothed exactly
        as the detector smooths it — so deriving a second, subtly different
        speed here would let the reported mean disagree with the freezing the
        same window is showing. Falls back to the centroid track for ethograms
        written before the column existed.
        """
        recorded = self._recorded_speed(start_frame, end_frame)
        if recorded is not None:
            return recorded

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

    def _recorded_speed(self, start_frame: int, end_frame: int):
        """Locomotion from the ethogram's own ``speed_cm_s``, or None."""
        if self.speed_cm_s is None:
            return None
        rows = np.where((self.frames >= start_frame) & (self.frames <= end_frame))[0]
        if rows.size == 0:
            return None
        speeds = self.speed_cm_s[rows]
        speeds = speeds[np.isfinite(speeds)]
        if speeds.size == 0:
            return None
        # cm/s sampled per row; each row covers one row-period of the session.
        rows_per_second = self.fps / max(1, self._row_stride())
        if rows_per_second <= 0:
            return None
        distance = float(speeds.sum() / rows_per_second)
        return distance, float(speeds.mean()), float(speeds.max())

    def _row_stride(self) -> int:
        """Frames between consecutive ethogram rows (the classifier cadence)."""
        if self.n_rows < 2:
            return 1
        stride = int(np.median(np.diff(self.frames)))
        return max(1, stride)

    def _window_thresholds(self, start_frame, end_frame, freeze_pct, dart_pct):
        """What this window alone would have chosen as freeze/dart cut-offs.

        Prefers the speed the run recorded, for the same reason
        :meth:`_locomotion` does — it is the signal the real thresholds were
        applied to — and because re-deriving it is a per-frame Python loop
        that dominated everything else here: on a thirty-session cohort it
        was a second per session, so selecting a window took half a minute
        for a number the cohort table does not even display.
        """
        rows = np.where((self.frames >= start_frame) & (self.frames <= end_frame))[0]
        if self.speed_cm_s is not None and rows.size:
            recorded = self.speed_cm_s[rows]
            recorded = recorded[np.isfinite(recorded)]
            if recorded.size:
                return (
                    float(np.percentile(recorded, freeze_pct)),
                    float(np.percentile(recorded, dart_pct)),
                    "cm/s",
                )
        if self.speed_px is not None and rows.size:
            recorded = self.speed_px[rows]
            recorded = recorded[np.isfinite(recorded)]
            if recorded.size:
                scale = (
                    self.fps / self.px_per_mm / 10.0
                    if (self.px_per_mm and self.px_per_mm > 0 and self.fps)
                    else 1.0
                )
                unit = "cm/s" if scale != 1.0 else "px/frame"
                return (
                    float(np.percentile(recorded, freeze_pct) * scale),
                    float(np.percentile(recorded, dart_pct) * scale),
                    unit,
                )

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
