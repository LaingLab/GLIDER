"""Feed a recorded video into the live pipeline, as if it were a camera.

Lets a whole closed loop be rehearsed off a recording: pose inference, behavior
classification, the nodes that trigger on it, and the real hardware they drive.
The point is to find out whether the rig fires correctly *before* an animal is
in the box, using footage where you already know what the animal did.

It hands frames to the same callback ``CameraManager`` streams to, with the
same ``(frame, timestamp)`` shape and the same wall-clock ``time.time()``
timestamps, so nothing downstream can tell the difference. Everything the live
path does -- classification, node triggers, BLE writes -- happens for real.

**Frames are never skipped.** That is the one hard rule. The live feature
extractor computes velocity and acceleration with ``np.gradient`` over *unit
frame spacing* (``compute_features`` never reads fps), so a dropped frame
doubles the apparent displacement across the gap and inflates the kinematics
that the behavior model was trained on. Falling behind is reported instead; a
rehearsal that quietly skipped frames would produce confident, wrong behavior.

**Speed does not change what gets classified.** Because features use unit frame
spacing, the same recording yields identical features at any playback rate.
Speed changes only two things, and they are the reasons both modes exist:

``speed=1.0``
    Real time. Answers "does inference keep up on this machine, and what is the
    end-to-end latency from behavior onset to stimulus?" -- the questions that
    decide whether the rig works.

``speed=0`` (as fast as the decoder and model allow)
    Answers "is any of this wired up correctly?" in a fraction of the runtime.
    Same classifications, same commands, wrong timing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from glider.vision.video_source import VideoFileSource

logger = logging.getLogger(__name__)

FrameCallback = Callable[[np.ndarray, float], None]


@dataclass(frozen=True)
class PumpStats:
    """A snapshot of how the rehearsal is going."""

    frames_delivered: int
    total_frames: int
    #: Seconds behind the real-time schedule right now. Always 0 when unpaced.
    lag_s: float
    #: Worst lag seen this run. This is the number that says whether inference
    #: keeps up: a rehearsal that ends seconds behind will miss stimulus timing
    #: on a live animal by the same margin.
    max_lag_s: float
    finished: bool

    @property
    def progress(self) -> float:
        """0.0-1.0, or 0.0 when the frame count is unknown."""
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, self.frames_delivered / self.total_frames)


class VideoPump:
    """Play a video file into ``on_frame``, in order, without skipping.

    Args:
        path: The video to play.
        on_frame: Called with ``(frame, timestamp)`` per frame, from the pump's
            own thread -- exactly like ``CameraManager``'s stream callback. A
            Qt consumer should marshal to the GUI thread, as the camera path
            already does.
        speed: Playback rate relative to the recording. 1.0 is real time; 0
            means as fast as possible. Values above 1 are allowed and are
            honest, because features do not depend on wall-clock rate.
        on_finished: Called once when the video ends or the pump is stopped,
            with the final :class:`PumpStats`. Also from the pump's thread.
    """

    def __init__(
        self,
        path: Path | str,
        on_frame: FrameCallback,
        *,
        speed: float = 1.0,
        on_finished: Callable[[PumpStats], None] | None = None,
    ) -> None:
        self._path = Path(path)
        self._on_frame = on_frame
        self._speed = max(0.0, float(speed))
        self._on_finished = on_finished

        self._source = VideoFileSource()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._lock = threading.Lock()
        self._delivered = 0
        self._lag = 0.0
        self._max_lag = 0.0
        self._finished = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._source.fps

    @property
    def total_frames(self) -> int:
        return self._source.frame_count

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Open the video and begin playing. False if it cannot be read."""
        if self.is_running:
            logger.warning("VideoPump: already running")
            return False
        if not self._source.load(self._path):
            logger.error("VideoPump: could not open %s", self._path)
            return False

        self._stop.clear()
        with self._lock:
            self._delivered = 0
            self._lag = 0.0
            self._max_lag = 0.0
            self._finished = False

        self._thread = threading.Thread(target=self._run, name="glider-video-pump", daemon=True)
        self._thread.start()
        logger.info(
            "VideoPump: playing %s (%d frames @ %.2f fps, speed=%s)",
            self._path.name,
            self.total_frames,
            self.fps,
            "max" if self._speed == 0 else f"{self._speed:g}x",
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Stop playing and wait for the thread to wind down.

        Safe to call when not running, and safe to call from the finished
        callback. The thread is a daemon, so a wedged decoder cannot keep the
        application alive.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        self._thread = None
        self._source.release()

    def stats(self) -> PumpStats:
        with self._lock:
            return PumpStats(
                frames_delivered=self._delivered,
                total_frames=self._source.frame_count,
                lag_s=self._lag,
                max_lag_s=self._max_lag,
                finished=self._finished,
            )

    # -- the pump ----------------------------------------------------------

    def _run(self) -> None:
        try:
            self._pump()
        except Exception:
            logger.exception("VideoPump: playback failed")
        finally:
            with self._lock:
                self._finished = True
            stats = self.stats()
            self._source.release()
            if self._on_finished is not None:
                try:
                    self._on_finished(stats)
                except Exception:
                    logger.exception("VideoPump: finished callback raised")

    def _pump(self) -> None:
        fps = self._source.fps or 30.0
        interval = 1.0 / (fps * self._speed) if self._speed > 0 else 0.0
        started = time.perf_counter()

        for index, frame in self._source.frames():
            if self._stop.is_set():
                logger.info("VideoPump: stopped after %d frames", index)
                return

            if interval:
                # Schedule against the start, not the previous frame, so a slow
                # frame does not push every later one back by the same amount.
                wait = (started + index * interval) - time.perf_counter()
                if wait > 0:
                    # Interruptible: a stop during a wait takes effect at once
                    # rather than after the frame's worth of sleep.
                    if self._stop.wait(wait):
                        return
                else:
                    self._record_lag(-wait)

            # Deliberately NOT wrapped in try/except. A callback that raises is
            # a broken consumer, and continuing would deliver a gap-riddled
            # stream that reads as an animal moving in jumps. _run logs it and
            # ends the rehearsal, which is the honest outcome.
            self._on_frame(frame, time.time())

            with self._lock:
                self._delivered = index + 1

        logger.info("VideoPump: reached the end of %s", self._path.name)

    def _record_lag(self, seconds: float) -> None:
        with self._lock:
            self._lag = seconds
            self._max_lag = max(self._max_lag, seconds)
