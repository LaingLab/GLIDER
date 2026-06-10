"""
VideoTrackingRunner — batch tracking over a recorded video.

Qt-free and synchronous so it is fully unit-testable. Reuses the live
recording machinery (CVProcessor, TrackingDataLogger) but drives it from a
VideoFileSource instead of a live camera, with timestamps taken from the
*video timeline* (frame / fps) rather than wall-clock. A Qt wrapper
(VideoTrackingWorker) adapts it to the GUI with signals.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from glider.vision.cv_processor import CVProcessor, CVSettings, TrackedObject
from glider.vision.tracking_logger import TrackingDataLogger
from glider.vision.video_recorder import open_video_writer
from glider.vision.video_source import VideoFileSource
from glider.vision.zones import Zone, ZoneConfiguration, draw_zones

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int], None]
CancelCb = Callable[[], bool]


@dataclass
class VideoTrackingConfig:
    """Configuration for a VideoTrackingRunner pass.

    write_zone_events / write_annotated / annotated_codec are consumed by the
    zone-event and annotated-video stages (added in later tasks); leaving them
    True is harmless until those stages exist.
    """

    source_path: Path
    output_dir: Path
    zone_config: ZoneConfiguration | None = None
    cv_settings: CVSettings = field(default_factory=CVSettings)
    write_tracking: bool = True
    write_zone_events: bool = True
    write_annotated: bool = True
    annotated_codec: str = "avc1"


class VideoTrackingRunner:
    """Run tracking over every frame of a video and write artifacts."""

    def __init__(
        self, config: VideoTrackingConfig, cv_processor: CVProcessor | None = None
    ) -> None:
        self._cfg = config
        # Injected for tests; otherwise build a fresh processor so object-track
        # IDs start clean and the live panel's processor is never disturbed.
        self._cv = cv_processor or CVProcessor(config.cv_settings)

    def run(
        self,
        progress_cb: ProgressCb | None = None,
        cancel_cb: CancelCb | None = None,
    ) -> Path:
        """Process the video and write artifacts; return the output directory.

        Timestamps derive from the video timeline (frame / fps), not wall clock.
        On cancel (cancel_cb returns True) the loop stops between frames and the
        output directory is still returned with metadata.json written. If
        process_frame/log_frame raises mid-loop, the logger is stopped and the
        source released (finally), then the exception propagates and
        metadata.json is NOT written (zone_events.csv / zone_occupancy.csv are
        still flushed with whatever frames were processed). Raises ValueError if
        the video cannot be opened. Must not be called from a thread with a
        running event loop (see the guard below).
        """
        # asyncio.run() below requires that no event loop is already running on
        # this thread. The intended caller is VideoTrackingWorker on a QThread
        # (no loop). Fail fast with a clear message if misused from the qasync
        # main thread.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop — good
        else:
            raise RuntimeError(
                "VideoTrackingRunner.run() must not be called from a thread with "
                "a running event loop; offload it to a QThread via "
                "VideoTrackingWorker."
            )

        cfg = self._cfg
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        source = VideoFileSource()
        if not source.load(cfg.source_path):
            raise ValueError(f"Cannot open video: {cfg.source_path}")

        if not self._cv.is_initialized:
            self._cv.initialize()

        fps = source.fps
        total = source.frame_count
        width, height = source.resolution
        # Anchor the video timeline to the file's mtime so timestamps are
        # plausible; elapsed_ms then equals frame/fps*1000 regardless of base.
        base = cfg.source_path.stat().st_mtime
        stem = cfg.source_path.stem

        tracker = None
        if cfg.write_tracking:
            tracker = TrackingDataLogger(output_dir=cfg.output_dir)
            tracker.set_session_epoch(base)
            if cfg.zone_config is not None:
                tracker.set_zone_configuration(cfg.zone_config)
                tracker.set_frame_size(width, height)
            asyncio.run(tracker.start(experiment_name=stem))

        zones = (
            self._cfg.zone_config.zones
            if (self._cfg.write_zone_events and self._cfg.zone_config)
            else []
        )
        zone_file, zone_writer = self._open_zone_writer()
        prev_members: dict[str, set[int]] = {z.id: set() for z in zones}
        frames_in_zone: dict[str, int] = {z.id: 0 for z in zones}

        annotated_writer = None
        if cfg.write_annotated:
            ann_path = cfg.output_dir / f"{stem}_annotated.mp4"
            annotated_writer, _codec = open_video_writer(
                ann_path, cfg.annotated_codec, fps, (width, height)
            )
            if annotated_writer is None:
                logger.warning("VideoTrackingRunner: no annotated-video codec available")

        try:
            for n, frame in source.frames():
                if cancel_cb is not None and cancel_cb():
                    logger.info("VideoTrackingRunner: cancelled at frame %d", n)
                    break
                ts = base + n / fps
                _detections, tracked, motion = self._cv.process_frame(frame, ts)
                if tracker is not None:
                    tracker.log_frame(ts, tracked, motion.motion_detected, motion.motion_area)
                if zones:
                    elapsed_ms = (n / fps) * 1000.0
                    for zone in zones:
                        current: set[int] = set()
                        for obj in tracked:
                            cx, cy = obj.centroid
                            if zone.contains_point_pixels(int(cx), int(cy), width, height):
                                current.add(obj.track_id)
                        if current:
                            frames_in_zone[zone.id] += 1
                        for tid in current - prev_members[zone.id]:
                            zone_writer.writerow(
                                [n + 1, f"{elapsed_ms:.1f}", zone.id, zone.name, tid, "enter"]
                            )
                        for tid in prev_members[zone.id] - current:
                            zone_writer.writerow(
                                [n + 1, f"{elapsed_ms:.1f}", zone.id, zone.name, tid, "exit"]
                            )
                        prev_members[zone.id] = current
                if annotated_writer is not None:
                    annotated_writer.write(self._annotate(frame, tracked, cfg.zone_config))
                if progress_cb is not None:
                    progress_cb(n + 1, total)
        finally:
            if tracker is not None:
                asyncio.run(tracker.stop())
            source.release()
            if annotated_writer is not None:
                annotated_writer.release()
            if zone_file is not None:
                zone_file.close()
                self._write_occupancy(zones, frames_in_zone, fps)

        self._write_metadata(fps, total, (width, height))
        return cfg.output_dir

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        tracked: list[TrackedObject],
        zone_config: ZoneConfiguration | None,
    ) -> np.ndarray:
        """Return a copy of *frame* with bbox/ID overlays and zones drawn."""
        out = frame.copy()
        for obj in tracked:
            x, y, w, h = obj.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                out,
                f"id:{obj.track_id}",
                (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )
        if zone_config and zone_config.zones:
            out = draw_zones(out, zone_config, alpha=0.3, show_labels=True)
        return out

    def _open_zone_writer(self) -> tuple[TextIO | None, Any]:
        """Return (file, csv_writer) for zone_events.csv, or (None, None)."""
        if not self._cfg.write_zone_events or not self._cfg.zone_config:
            return None, None
        f = open(self._cfg.output_dir / "zone_events.csv", "w", newline="", encoding="utf-8")
        w = csv.writer(f)
        w.writerow(["frame", "elapsed_ms", "zone_id", "zone_name", "object_id", "event"])
        return f, w

    def _write_occupancy(
        self, zones: list[Zone], frames_in_zone: dict[str, int], fps: float
    ) -> None:
        with open(
            self._cfg.output_dir / "zone_occupancy.csv", "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(["zone_id", "zone_name", "frames_in_zone", "seconds"])
            for zone in zones:
                fz = frames_in_zone[zone.id]
                w.writerow([zone.id, zone.name, fz, f"{fz / fps:.3f}"])

    def _write_metadata(self, fps: float, frame_count: int, resolution: tuple[int, int]) -> None:
        cfg = self._cfg
        meta = {
            "source_path": str(cfg.source_path),
            "fps": fps,
            "frame_count": frame_count,
            "resolution": list(resolution),
            "zone_config": cfg.zone_config.to_dict() if cfg.zone_config else None,
            "cv_settings": cfg.cv_settings.to_dict(),
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        (cfg.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
