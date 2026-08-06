"""Majority-vote smoothing for the behavior-label stream.

The classifier emits one label per prediction tick; adjacent ticks are
independent, so the raw stream flickers between classes. Smoothing commits
the most-common label over a window, which stabilises the overlay, the
output video, and the ethogram CSV identically (it sits at the single point
where labels are committed).

Two smoothers, because live and offline scoring are not the same problem.
:class:`MajorityVoteSmoother` is causal -- it only ever sees labels already
emitted, which is all a live overlay can do. :func:`centered_majority_vote`
is for scoring a finished recording, where the frames after a given one are
already on disk and refusing to look at them buys nothing.

The difference is not cosmetic. Measured on eight held-out sessions against
a 4-class model, macro F1 went 0.780 raw -> 0.797 causal -> 0.823 centred,
and on transition frames causal scored *below* raw (0.743 vs 0.756): a
causal vote lags by half its window, so at every bout boundary it spends
that long still reporting the previous behavior. A centred vote is
symmetric about the frame and has no lag to pay.

Pure and dependency-free — no Qt / cv2 / threads — so it's unit-tested
in isolation.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence


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


#: Smoothing window that matches a typical bout, in frames. Bout lengths on
#: this rig run median 22 / p75 29 at 30 fps, and macro F1 sits on a broad
#: plateau from 21 to 31 -- flat enough that the exact value does not matter,
#: which is what makes "roughly one bout" a defensible default rather than a
#: number tuned against a test set. Past ~37 it falls away as real bouts start
#: being absorbed by their neighbours.
DEFAULT_OFFLINE_WINDOW = 25


def centered_majority_vote(
    labels: Sequence[str],
    window: int = DEFAULT_OFFLINE_WINDOW,
    *,
    blank: str = "",
) -> list[str]:
    """Majority vote over a window centred on each label. Offline only.

    Scoring a recording that already exists is not a streaming problem: the
    frames after frame *i* are on disk, and a causal vote that ignores them
    lags every bout boundary by half its window. This looks both ways, so a
    boundary is crossed once rather than smeared.

    ``blank`` labels -- what the model emits for a window it could not fill --
    are neither voted on nor voted with. They stay blank, and a frame's window
    is the non-blank labels within it, so a run of blanks does not silently
    become whatever surrounded it. ``window <= 1`` is pass-through.

    Erasing bouts shorter than the window is the obvious hazard, and on
    held-out data it did not materialise: predicted bout counts fell towards
    the true ones (117 investigate bouts -> 73, against 59 real) while
    recall at 50% overlap held, because what the vote removes is flicker
    rather than short bouts.
    """
    out = list(labels)
    size = max(1, int(window))
    if size <= 1 or not out:
        return out
    half = size // 2
    for i, raw in enumerate(out):
        if raw == blank:
            continue
        # Read from `labels`, never from `out`: voting on already-smoothed
        # neighbours would let one decision propagate along the whole
        # recording instead of each frame seeing the model's own output.
        seg = [lab for lab in labels[max(0, i - half) : i + half + 1] if lab != blank]
        if seg:
            out[i] = Counter(seg).most_common(1)[0][0]
    return out
