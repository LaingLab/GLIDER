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
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from glider.vision.cv_processor import CVProcessor, CVSettings
from glider.vision.tracking_logger import TrackingDataLogger
from glider.vision.video_source import VideoFileSource
from glider.vision.zones import ZoneConfiguration

logger = logging.getLogger(__name__)

ProgressCb = Callable[[int, int], None]
CancelCb = Callable[[], bool]


@dataclass
class VideoTrackingConfig:
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
        """Process the video and return the output directory. Raises on
        unrecoverable setup errors (bad video, unwritable dir)."""
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
            asyncio.run(tracker.start(experiment_name=stem))

        try:
            for n, frame in source.frames():
                if cancel_cb is not None and cancel_cb():
                    logger.info("VideoTrackingRunner: cancelled at frame %d", n)
                    break
                ts = base + n / fps
                _detections, tracked, motion = self._cv.process_frame(frame, ts)
                if tracker is not None:
                    tracker.log_frame(ts, tracked, motion.motion_detected, motion.motion_area)
                if progress_cb is not None:
                    progress_cb(n + 1, total)
        finally:
            if tracker is not None:
                asyncio.run(tracker.stop())
            source.release()

        self._write_metadata(fps, total, (width, height))
        return cfg.output_dir

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
