"""Pure elapsed-time formatting shared by the Dashboard header and run banner."""

from __future__ import annotations


def format_elapsed(elapsed: float) -> str:
    """Format seconds as MM:SS.cc, or HH:MM:SS.cc past one hour.

    Truncates toward zero (int()) rather than rounding nearest, so the display
    never jumps ahead of the wall clock and centiseconds never reads "60".
    """
    total_centiseconds = int(max(0.0, elapsed) * 100)
    hours, rem = divmod(total_centiseconds, 360_000)
    minutes, rem = divmod(rem, 6_000)
    seconds, centiseconds = divmod(rem, 100)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
