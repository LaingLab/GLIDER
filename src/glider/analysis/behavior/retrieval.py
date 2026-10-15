"""Pup retrieval: who was carried back to the nest, and how long it took.

The assay: pups are scattered outside the nest, the dam is returned, and the
measure is how fast she brings each one home. Everything here is derived from
three per-frame signals per pup, all in centimetres and already computed by
:mod:`glider.analysis.behavior.nest`:

* ``nest_cm``  -- signed distance of the pup's centroid to the nest edge
  (``<= 0`` is in the nest);
* ``nose_cm``  -- distance from the dam's nose to the nearest pixel of the pup;
* ``xy_cm``    -- the pup's centroid, for speed and displacement.

A missing pup (NaN) is a real state, not noise, and there are two kinds:

* **In her mouth.** A gap that starts while she is touching the pup is a
  carry. The pup is where her nose is, so -- given the nose's track -- it
  follows the nose until the nose reaches the nest (dropped there) or
  ``hidden_carry_max_s`` runs out. On real retrieval videos most retrievals
  are seen *only* this way: the pup goes into her mouth and comes out buried in
  the nest, never visible crossing the edge.
* **Buried or under her.** Any other gap holds the last known state rather
  than flipping to "out".

Pure numpy, no I/O. The thresholds are the lab's confirmed protocol defaults
and live on :class:`RetrievalSettings` so a different protocol is a setting,
not an edit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PupRetrieval", "RetrievalSettings", "litter_summary", "score_pup"]


@dataclass(frozen=True)
class RetrievalSettings:
    """Protocol thresholds. Defaults are the lab's (confirmed 2026-10-09)."""

    in_nest_s: float = 2.0  # must stay in the nest this long to count
    contact_cm: float = 1.0  # dam's nose this close to the pup = contact
    carry_speed_cm_s: float = 3.0  # pups this young do not crawl this fast
    # ponytail: two knobs the protocol never named, tuned on synthetic data
    # only. Revisit both against hand-scored videos in Phase 2.
    carry_merge_s: float = 0.5  # a carry that pauses this briefly is one carry
    min_carry_cm: float = 2.0  # sniffing over a pup moves it less than this
    speed_window_s: float = 0.2  # centred window for speed; damps mask jitter
    # ponytail: a carry longer than this while hidden is assumed to be the pup
    # buried, not carried. real carries take a few seconds; tune on more videos.
    hidden_carry_max_s: float = 10.0


@dataclass
class PupRetrieval:
    """One pup's trial. Latencies are seconds from trial start, NaN if never."""

    start_in_nest: bool
    contact_latency_s: float
    retrieval_latency_s: float
    carries: int
    drops: int
    carry_distance_cm: float
    in_nest: np.ndarray = field(repr=False)  # (F,) bool, settled-in-nest per frame
    carried: np.ndarray = field(repr=False)  # (F,) bool


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of every True run."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))


def _ffill(values: np.ndarray) -> np.ndarray:
    """Carry the last finite value forward; leading NaNs take the first one."""
    ok = np.isfinite(values)
    if not ok.any():
        return values.copy()
    idx = np.where(ok, np.arange(len(values)), 0)
    np.maximum.accumulate(idx, out=idx)
    idx[: np.argmax(ok)] = np.argmax(ok)
    return values[idx]


def _speed(xy: np.ndarray, fps: float, half: int) -> np.ndarray:
    """Centred-difference speed in cm/s; NaN where either end is missing."""
    n = len(xy)
    speed = np.full(n, np.nan)
    if n > 2 * half:
        step = np.linalg.norm(xy[2 * half :] - xy[: -2 * half], axis=1)
        speed[half : n - half] = step * fps / (2 * half)
    return speed


def score_pup(
    nest_cm: np.ndarray,
    nose_cm: np.ndarray,
    xy_cm: np.ndarray,
    *,
    fps: float,
    t0: int,
    nose_xy_cm: np.ndarray | None = None,
    nose_nest_cm: np.ndarray | None = None,
    settings: RetrievalSettings | None = None,
) -> PupRetrieval:
    """Score one pup from trial-start frame *t0* onward.

    ``nose_xy_cm`` and ``nose_nest_cm`` (the dam's nose position and its signed
    nest distance) let a pup hidden in her mouth be followed into the nest.
    Without them a hidden carry is still counted, but its end is never seen.
    """
    s = settings or RetrievalSettings()
    nest_cm = np.asarray(nest_cm, dtype=float).copy()
    nose_cm = np.asarray(nose_cm, dtype=float)
    xy_cm = np.asarray(xy_cm, dtype=float).copy()
    n = len(nest_cm)
    if not (len(nose_cm) == len(xy_cm) == n):
        raise ValueError("nest_cm, nose_cm and xy_cm must cover the same frames")
    if not 0 <= t0 < n:
        raise ValueError(f"trial start frame {t0} is outside the {n} frames scored")

    seen = np.isfinite(xy_cm).all(axis=1) & np.isfinite(nest_cm)
    contact = nose_cm <= s.contact_cm  # NaN compares False

    # Carried: touched and moving faster than a pup can crawl...
    half = max(1, round(s.speed_window_s * fps / 2))
    carried = contact & (_speed(xy_cm, fps, half) > s.carry_speed_cm_s)
    # ...or vanished while being touched: in her mouth, so at her nose.
    cap = max(1, round(s.hidden_carry_max_s * fps))
    for a, b in _runs(~seen):
        if a == 0 or not contact[a - 1]:
            continue
        end = min(b, a + cap)
        if nose_xy_cm is not None and nose_nest_cm is not None:
            home = np.flatnonzero(np.asarray(nose_nest_cm[a:end]) <= 0)
            if home.size:
                end = a + home[0] + 1  # dropped in the nest
            xy_cm[a:end] = nose_xy_cm[a:end]
            nest_cm[a:end] = nose_nest_cm[a:end]
        carried[a:end] = True
    seen = np.isfinite(xy_cm).all(axis=1) & np.isfinite(nest_cm)

    # One carry, even if the dam pauses mid-way.
    gap = round(s.carry_merge_s * fps)
    bouts: list[list[int]] = []
    for a, b in _runs(carried):
        if bouts and a - bouts[-1][1] <= gap:
            bouts[-1][1] = b
        else:
            bouts.append([a, b])

    # Settled in the nest: in it continuously for in_nest_s. Gaps hold state.
    raw_in = _ffill(np.where(seen, nest_cm, np.nan)) <= 0
    hold = max(1, round(s.in_nest_s * fps))
    in_nest = np.zeros(n, dtype=bool)
    for a, b in _runs(raw_in):
        if b - a >= hold:
            in_nest[a:b] = True

    # Last position seen at or before each frame, for displacement.
    pos = np.where(seen[:, None], xy_cm, np.nan)
    pos_ff = np.column_stack([_ffill(pos[:, i]) for i in range(2)])

    carries = drops = 0
    distance = 0.0
    keep = np.zeros(n, dtype=bool)
    for a, b in bouts:
        if b <= t0:
            continue
        start = pos_ff[max(a - 1, 0)]
        # Where it was put down: seen right after the carry, else where the
        # carry ended. Never a later sighting -- a buried pup's next one can be
        # minutes on and anywhere in the nest.
        end = pos[b] if b < n and seen[b] else pos_ff[b - 1]
        moved = float(np.linalg.norm(end - start))
        if not np.isfinite(moved) or moved < s.min_carry_cm:
            continue  # sniffing, not carrying
        keep[a:b] = True
        carries += 1
        distance += moved
        if not raw_in[min(b, n - 1)]:
            drops += 1

    start_in_nest = bool(raw_in[t0])
    first_contact = np.flatnonzero(contact[t0:])
    contact_latency = first_contact[0] / fps if first_contact.size else np.nan
    retrieval_latency = np.nan
    if not start_in_nest:
        home = np.flatnonzero(in_nest[t0:])
        if home.size:
            retrieval_latency = home[0] / fps

    return PupRetrieval(
        start_in_nest=start_in_nest,
        contact_latency_s=float(contact_latency) if not start_in_nest else np.nan,
        retrieval_latency_s=float(retrieval_latency),
        carries=carries,
        drops=drops,
        carry_distance_cm=distance,
        in_nest=in_nest,
        carried=keep,
    )


def litter_summary(pups: dict[str, PupRetrieval]) -> dict[str, object]:
    """Litter-level measures, plus each pup's retrieval order (1 = first)."""
    eligible = {k: p for k, p in pups.items() if not p.start_in_nest}
    done = sorted(
        (p.retrieval_latency_s, k)
        for k, p in eligible.items()
        if np.isfinite(p.retrieval_latency_s)
    )
    all_home = bool(eligible) and len(done) == len(eligible)
    return {
        "pups_scattered": len(eligible),
        "pups_retrieved": len(done),
        "time_all_retrieved_s": done[-1][0] if all_home else np.nan,
        "retrieval_order": {k: i + 1 for i, (_, k) in enumerate(done)},
    }
