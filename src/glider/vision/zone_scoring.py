"""Zone occupancy scored from a pose track, without re-reading the video.

:class:`~glider.vision.video_tracking_runner.VideoTrackingRunner` produces
``zone_events.csv`` and ``zone_occupancy.csv`` as it decodes a video. That is
the right shape for live tracking, but wrong for a cohort already scored: the
question "was the animal inside this polygon" needs only the tracked point and
the polygon, both of which are already on disk once the pose batch has run.
Re-decoding hours of video to answer it again is pure cost.

So this module scores the same two files off a :class:`PoseData`, writing the
identical schema. Anything that reads the runner's output reads this.

Two decisions worth stating, because both are places where a scorer can quietly
produce a plausible wrong answer:

**A dropout is missing information, not movement.** A frame the model was not
confident about is skipped - it does not count toward occupancy, and it does
not end a bout. Treating it as "outside" would invent an exit and a re-entry
the animal never made, inflating crossing counts on exactly the animals whose
tracking is worst.

**A keypoint that is not there is an error, not a zero.** Ask for a name the
track does not carry and this raises. The alternative - scoring an absent
keypoint as never-in-zone - writes a clean-looking file of zeros, and a
vocabulary slip like ``centre_body`` for ``body_center`` then reads as an
animal that never entered the centre.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from glider.vision.pose.core import PoseData
from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)

__all__ = [
    "KeypointMissingError",
    "ZoneEvent",
    "ZoneScoring",
    "score_csv",
    "score_pose",
    "write_zone_csvs",
    "zone_output_dir",
]

#: Default tracked point. The body centre is the convention for open-field
#: centre-zone scoring: a nose-based score counts an animal that only peered in.
DEFAULT_KEYPOINT = "body_center"

#: Default confidence floor. Zero keeps every frame, matching
#: :class:`~glider.analysis.behavior.features.FeatureSpec`, so a caller opts
#: into dropout rejection rather than having a threshold chosen for them.
DEFAULT_MIN_CONFIDENCE = 0.0


class KeypointMissingError(KeyError):
    """The requested keypoint is not in this track."""


@dataclass(frozen=True)
class ZoneEvent:
    """A transition across one zone boundary."""

    frame: int
    elapsed_ms: float
    zone_id: str
    zone_name: str
    event: str  # "enter" or "exit"
    object_id: str = "subject"


@dataclass
class ZoneScoring:
    """What a track did in each zone."""

    frames_in_zone: dict[str, int]
    seconds_in_zone: dict[str, float]
    events: list[ZoneEvent]
    zone_names: dict[str, str]
    frames_total: int
    frames_scored: int
    fps: float
    keypoint: str = DEFAULT_KEYPOINT
    metadata: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of frames that carried a usable point."""
        return self.frames_scored / self.frames_total if self.frames_total else 0.0


def _keypoint_index(pose: PoseData, keypoint: str) -> int:
    try:
        return pose.keypoint_names.index(keypoint)
    except ValueError as exc:
        raise KeypointMissingError(
            f"{keypoint!r} is not in this track; it carries " f"{', '.join(pose.keypoint_names)}"
        ) from exc


def score_pose(
    pose: PoseData,
    zone_config: ZoneConfiguration,
    *,
    resolution: tuple[int, int] | None = None,
    keypoint: str = DEFAULT_KEYPOINT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ZoneScoring:
    """Score *pose* against *zone_config*.

    Args:
        pose: The track. Coordinates are in pixels.
        zone_config: Zones, whose vertices are normalized 0-1.
        resolution: ``(width, height)`` the pose was tracked at, needed to
            normalize its pixels against the zones. Falls back to the zone
            config's own recorded size.
        keypoint: Which tracked point decides occupancy.
        min_confidence: Frames below this are treated as dropouts.

    Raises:
        KeypointMissingError: *keypoint* is not in the track.
        ValueError: no resolution is available from either argument.
    """
    if resolution is None:
        if not zone_config.config_width or not zone_config.config_height:
            raise ValueError(
                "no resolution given and the zone configuration does not record one; "
                "pass resolution=(width, height)"
            )
        resolution = (zone_config.config_width, zone_config.config_height)
    width, height = resolution
    if not width or not height:
        raise ValueError(f"resolution {resolution!r} is not usable")

    index = _keypoint_index(pose, keypoint)
    xy = pose.xy[:, index, :]
    confidence = pose.confidence[:, index]
    fps = pose.fps or 30.0

    zones = list(zone_config.zones)
    frames_in_zone = {z.id: 0 for z in zones}
    zone_names = {z.id: z.name for z in zones}
    inside_now = {z.id: False for z in zones}
    events: list[ZoneEvent] = []
    scored = 0

    usable = np.isfinite(xy).all(axis=1) & (confidence >= min_confidence)
    for frame in range(len(xy)):
        if not usable[frame]:
            # Hold state: a dropout ends nothing.
            continue
        scored += 1
        x = float(xy[frame, 0]) / width
        y = float(xy[frame, 1]) / height
        for zone in zones:
            contains = zone.contains_point(x, y)
            if contains:
                frames_in_zone[zone.id] += 1
            if contains != inside_now[zone.id]:
                inside_now[zone.id] = contains
                events.append(
                    ZoneEvent(
                        frame=frame,
                        elapsed_ms=frame * 1000.0 / fps,
                        zone_id=zone.id,
                        zone_name=zone.name,
                        event="enter" if contains else "exit",
                    )
                )

    return ZoneScoring(
        frames_in_zone=frames_in_zone,
        seconds_in_zone={k: v / fps for k, v in frames_in_zone.items()},
        events=events,
        zone_names=zone_names,
        frames_total=len(xy),
        frames_scored=scored,
        fps=fps,
        keypoint=keypoint,
    )


def score_csv(
    csv_path: Path | str,
    zone_config: ZoneConfiguration,
    *,
    resolution: tuple[int, int] | None = None,
    keypoint: str = DEFAULT_KEYPOINT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ZoneScoring:
    """Score the pose CSV at *csv_path*.

    The resolution is taken from the sidecar the pose batch writes when it is
    not given. Without one the pixel track cannot be placed against normalized
    zone vertices at all, so a guess here would silently mis-score rather than
    fail - hence the refusal.
    """
    from glider.vision.pose.dlc import from_dlc_csv, resolution_for_csv

    csv_path = Path(csv_path)
    if resolution is None:
        resolution = resolution_for_csv(csv_path)
    if resolution is None:
        if zone_config.config_width and zone_config.config_height:
            resolution = (zone_config.config_width, zone_config.config_height)
        else:
            raise ValueError(
                f"{csv_path.name} has no recorded resolution and none was given; "
                "pass resolution=(width, height)"
            )

    pose = from_dlc_csv(csv_path)
    scoring = score_pose(
        pose,
        zone_config,
        resolution=resolution,
        keypoint=keypoint,
        min_confidence=min_confidence,
    )
    scoring.metadata["source_csv"] = str(csv_path)
    return scoring


def write_zone_csvs(scoring: ZoneScoring, output_dir: Path | str) -> list[Path]:
    """Write ``zone_events.csv`` and ``zone_occupancy.csv`` into *output_dir*.

    Schema matches :class:`~glider.vision.video_tracking_runner.VideoTrackingRunner`
    exactly, so existing readers do not need to know which produced a file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events_path = output_dir / "zone_events.csv"
    with open(events_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "elapsed_ms", "zone_id", "zone_name", "object_id", "event"])
        for event in scoring.events:
            writer.writerow(
                [
                    event.frame,
                    f"{event.elapsed_ms:.1f}",
                    event.zone_id,
                    event.zone_name,
                    event.object_id,
                    event.event,
                ]
            )

    occupancy_path = output_dir / "zone_occupancy.csv"
    with open(occupancy_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["zone_id", "zone_name", "frames_in_zone", "seconds"])
        for zone_id, frames in scoring.frames_in_zone.items():
            writer.writerow(
                [
                    zone_id,
                    scoring.zone_names.get(zone_id, ""),
                    frames,
                    f"{frames / scoring.fps:.3f}" if scoring.fps else "",
                ]
            )

    if not math.isclose(scoring.coverage, 1.0):
        logger.info(
            "zone scoring covered %.1f%% of frames (%d of %d had a usable %s)",
            scoring.coverage * 100,
            scoring.frames_scored,
            scoring.frames_total,
            scoring.keypoint,
        )
    return [events_path, occupancy_path]


def zone_output_dir(video: Path | str) -> Path:
    """Where zone CSVs for *video* belong: ``<stem>_zones/`` beside it.

    A directory rather than stem-prefixed files, so the two CSVs keep the
    names :class:`~glider.vision.video_tracking_runner.VideoTrackingRunner`
    gives them and a reader does not have to care which produced them.
    """
    video = Path(video)
    return video.parent / f"{video.stem}_zones"
