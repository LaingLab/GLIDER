"""Majority-vote smoothing for the live behavior-label stream.

The classifier emits one label per prediction tick; adjacent ticks are
independent, so the raw stream flickers between classes. This smoother
commits the most-common label over the last ``window`` predictions,
which stabilises the overlay, the output video, and the ethogram CSV
identically (it sits at the single point where labels are committed).

Pure and dependency-free — no Qt / cv2 / threads — so it's unit-tested
in isolation.
"""

from __future__ import annotations

from collections import Counter, deque


class MajorityVoteSmoother:
    """Sliding majority vote over the last ``window`` labels.

    ``window <= 1`` disables smoothing (pass-through). On a tie for the
    most-common label, the previously-committed label is kept if it's
    among the tied set (hysteresis, to avoid flip-flopping); otherwise
    the most recent of the tied labels wins.
    """

    def __init__(self, window: int):
        self.window = max(1, int(window))
        self._buf: deque[str] = deque(maxlen=self.window)
        self._committed: str | None = None

    def push(self, label: str) -> str:
        """Feed one raw prediction; return the label to commit."""
        self._buf.append(label)
        if self.window <= 1:
            self._committed = label
            return label
        counts = Counter(self._buf)
        top = max(counts.values())
        tied = [lab for lab, c in counts.items() if c == top]
        if len(tied) == 1:
            chosen = tied[0]
        elif self._committed in tied:
            chosen = self._committed  # hysteresis: stick with current label
        else:
            tied_set = set(tied)
            chosen = next(lab for lab in reversed(self._buf) if lab in tied_set)
        self._committed = chosen
        return chosen
