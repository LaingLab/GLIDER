"""Windows timer resolution, for the parts of GLIDER that time things.

Windows' default system timer tick is ~15.6 ms, and a blocking wait backed by
a condition variable -- which is what ``threading.Event.wait(timeout)`` is --
cannot resolve finer than one tick. So a Delay node asking for 0.5 s returns
somewhere in 0.5000-0.5625 s, quantised to that tick, with a measured median
overshoot of 31 ms and a worst case of 63 ms on an ordinary desktop.

That is invisible for a ten-minute habituation and material for anything
short: a 50 ms inter-stimulus gap is wrong by more than its own duration.

``timeBeginPeriod(1)`` asks Windows for a 1 ms tick, which takes the same
measurement to a median and worst case of 16 ms -- roughly a 2x improvement in
the typical case and 4x in the tail, and, more usefully, *deterministic*
rather than scattered across four tick boundaries.

macOS and Linux already resolve these waits to about a millisecond, so this is
a no-op there.

The request is process-global and reference-counted by Windows, and it has a
real cost: a faster tick means the CPU sleeps less deeply, which matters on a
laptop running on battery. So it is held only while something is actually
waiting on it rather than for the life of the process.
"""

from __future__ import annotations

import logging
import sys
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# The tick we ask Windows for, in milliseconds. 1 is the finest anything
# grants and what media and timing software conventionally requests.
_TARGET_PERIOD_MS = 1

_lock = threading.Lock()
_depth = 0
_winmm = None
_unavailable = False


def _load_winmm():
    """Return the winmm handle, or None if it cannot be used.

    Cached both ways. A machine where this fails is not broken -- it just
    keeps the coarse tick -- so the failure is logged once and never retried.
    """
    global _winmm, _unavailable
    if _winmm is not None or _unavailable:
        return _winmm
    try:
        import ctypes

        _winmm = ctypes.WinDLL("winmm")
    except Exception as e:  # noqa: BLE001 - any failure means "no fine tick"
        _unavailable = True
        logger.warning(
            "Could not load winmm; timed waits keep Windows' default ~15.6 ms " "resolution (%s)",
            e,
        )
    return _winmm


@contextmanager
def high_resolution_timers():
    """Hold a 1 ms system timer tick for the duration of the block.

    A no-op off Windows, and a no-op on Windows if the request is refused.
    Nested and concurrent uses share one request: the tick is raised on the
    first entry and released when the last holder leaves, so two overlapping
    delays do not release it out from under each other.

    Never raises. A timing improvement that could take down an experiment
    would be a bad trade.
    """
    if sys.platform != "win32":
        yield
        return

    raised = _acquire()
    try:
        yield
    finally:
        if raised:
            _release()


def _acquire() -> bool:
    """Raise the tick if we are the first holder. True if we hold a request."""
    global _depth
    winmm = _load_winmm()
    if winmm is None:
        return False
    with _lock:
        if _depth == 0:
            try:
                # 0 is TIMERR_NOERROR; anything else means the period was
                # refused and there is nothing to release later.
                if winmm.timeBeginPeriod(_TARGET_PERIOD_MS) != 0:
                    logger.debug("timeBeginPeriod(%d) refused", _TARGET_PERIOD_MS)
                    return False
            except Exception:  # noqa: BLE001 - never break the caller
                logger.debug("timeBeginPeriod failed", exc_info=True)
                return False
        _depth += 1
        return True


def _release() -> None:
    """Drop this holder's request, restoring the default tick at zero."""
    global _depth
    with _lock:
        _depth -= 1
        if _depth > 0:
            return
        _depth = 0
        winmm = _load_winmm()
        if winmm is None:
            return
        try:
            winmm.timeEndPeriod(_TARGET_PERIOD_MS)
        except Exception:  # noqa: BLE001 - never break the caller
            logger.debug("timeEndPeriod failed", exc_info=True)
