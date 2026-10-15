"""Score one pup-retrieval session from its pups, the dam's nose and its setup.

Everything here is a re-run, not an inference: the pups come from the batch
pass's ``_pups.npz``, the nose from the pose CSV, and the rest from the two
JSON sidecars. So the Retrieval tab can re-score on every change.

The nest can move: each frame is measured against the outline whose period
it falls in. Frames from the trial start to the cutoff are scored; a pup not
retrieved by the cutoff has a blank latency, and the cutoff -- the video's end
if that comes first -- travels with the exports so stats can treat it as
censored.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from glider.analysis.behavior.nest import NestField, nest_field
from glider.analysis.behavior.retrieval import PupRetrieval, litter_summary, score_pup
from glider.analysis.behavior.retrieval_setup import RetrievalProtocol, RetrievalSetup
from glider.vision.pose.pups import PupTracks

__all__ = [
    "STATES",
    "NotScorable",
    "SessionRetrieval",
    "litter_row",
    "nest_distance_frame",
    "nose_to_pup_px",
    "pup_rows",
    "range_values",
    "score_session",
]

STATES = ("out", "contact", "carried", "in nest")


class NotScorable(ValueError):  # noqa: N818 -- a state to explain, not a fault
    """Why a session cannot be scored yet, worded for the Retrieval tab."""


@dataclass
class SessionRetrieval:
    pups: dict[str, PupRetrieval]
    litter: dict
    t0: int  # trial-start frame
    end: int  # first frame not scored: the cutoff or the video's end
    fps: float
    px_per_cm: float
    contact_cm: float
    fields: list[NestField]
    period: np.ndarray  # (F,) index into fields
    pup_xy_px: np.ndarray  # (N, F, 2)
    pup_nest_cm: np.ndarray  # (N, F) signed
    nose_pup_cm: np.ndarray  # (N, F)
    nose_xy_px: np.ndarray  # (F, 2)
    nose_nest_cm: np.ndarray  # (F,) signed
    hidden: np.ndarray  # (N, F) bool: carried while unseen (in her mouth)

    @property
    def n_frames(self) -> int:
        return len(self.period)

    @property
    def dam_in_nest_frac(self) -> float:
        window = self.nose_nest_cm[self.t0 : self.end]
        window = window[np.isfinite(window)]
        return float(np.mean(window <= 0)) if window.size else float("nan")

    def state_codes(self, k: int) -> np.ndarray:
        """Per frame: -1 outside the scored window, else an index into STATES."""
        r = list(self.pups.values())[k]
        codes = np.full(self.n_frames, -1, dtype=np.int8)
        window = slice(self.t0, self.end)
        codes[window] = 0
        contact = self.nose_pup_cm[k, : self.end] <= self.contact_cm
        codes[: self.end][contact & (codes[: self.end] == 0)] = 1
        codes[: self.end][r.in_nest] = 3
        codes[: self.end][r.carried] = 2
        codes[: self.t0] = -1
        return codes


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    """*a* cut or NaN-padded to *n* rows."""
    out = np.full((n, *a.shape[1:]), np.nan)
    m = min(n, len(a))
    out[:m] = a[:m]
    return out


def nose_to_pup_px(pups: PupTracks, nose_xy_px: np.ndarray) -> np.ndarray:
    """(N, F) pixels from the nose to the nearest pixel of each pup; 0 when on it.

    Each pup is measured to its outline as :meth:`PupTracks.outline_for` picks
    it -- the one whose vertex mean is nearest the pup -- but a frame's outline
    centres are found once and shared by its pups, not once per pup.

    ponytail: a Python loop over seen frames (~0.3-0.5 s for 15 min x 5 pups);
    callers cache it per session -- it does not depend on any setting.
    """
    nose = _fit(np.asarray(nose_xy_px, float), pups.n_frames)
    out = np.full((pups.n_pups, pups.n_frames), np.nan)
    seen = np.isfinite(pups.xy).all(axis=2)  # (N, F)
    for f in np.flatnonzero(np.isfinite(nose).all(axis=1) & seen.any(axis=0)):
        outlines = pups.outlines[f]
        if not outlines:
            continue
        centres = np.array([o.mean(axis=0) for o in outlines])  # (K, 2)
        here = np.flatnonzero(seen[:, f])
        offsets = centres[None] - pups.xy[here, f][:, None]  # (pups here, K, 2)
        nearest = np.hypot(offsets[..., 0], offsets[..., 1]).argmin(axis=1)
        point = (float(nose[f, 0]), float(nose[f, 1]))
        for k, j in zip(here, nearest, strict=True):
            contour = outlines[j].reshape(-1, 1, 2).astype(np.float32)
            out[k, f] = max(-cv2.pointPolygonTest(contour, point, True), 0.0)
    return out


def score_session(
    pups: PupTracks,
    nose_xy_px: np.ndarray | None,
    *,
    fps: float,
    size: tuple[int, int],
    setup: RetrievalSetup,
    protocol: RetrievalProtocol,
    nose_pup_px: np.ndarray | None = None,
) -> SessionRetrieval:
    if not setup.nests:
        raise NotScorable("Draw the nest to score")
    if setup.trial_start_s is None:
        raise NotScorable("Set trial start to score")
    if not setup.px_per_mm or setup.px_per_mm <= 0:
        raise NotScorable("Set scale to score")
    if nose_xy_px is None:
        raise NotScorable(
            f"No '{protocol.nose_keypoint}' keypoint in the pose; pick the nose in Thresholds…"
        )
    n_frames = pups.n_frames
    t0 = int(round(setup.trial_start_s * fps))
    if t0 >= n_frames:
        raise NotScorable("Trial start is after the end of the video")
    end = min(n_frames, t0 + max(1, int(round(protocol.cutoff_s * fps))))
    px_per_cm = float(setup.px_per_mm) * 10.0
    width, height = size

    nose = _fit(np.asarray(nose_xy_px, float), n_frames)
    starts = np.array([p.from_s for p in setup.nests])
    period = np.clip(np.searchsorted(starts, np.arange(n_frames) / fps, side="right") - 1, 0, None)
    fields = [nest_field(p.zone, width, height) for p in setup.nests]
    nose_nest = np.full(n_frames, np.nan)
    pup_nest = np.full((pups.n_pups, n_frames), np.nan)
    for i, nest in enumerate(fields):
        here = period == i
        nose_nest[here] = nest.at(nose[here])
        for k in range(pups.n_pups):
            pup_nest[k, here] = nest.at(pups.xy[k, here])
    nose_pup = nose_pup_px if nose_pup_px is not None else nose_to_pup_px(pups, nose)

    settings = protocol.settings()
    scored = slice(0, end)
    results: dict[str, PupRetrieval] = {}
    hidden = np.zeros((pups.n_pups, n_frames), dtype=bool)
    for k, name in enumerate(pups.names):
        r = score_pup(
            pup_nest[k, scored] / px_per_cm,
            nose_pup[k, scored] / px_per_cm,
            pups.xy[k, scored] / px_per_cm,
            fps=fps,
            t0=t0,
            nose_xy_cm=nose[scored] / px_per_cm,
            nose_nest_cm=nose_nest[scored] / px_per_cm,
            settings=settings,
        )
        results[name] = r
        seen = np.isfinite(pups.xy[k, scored]).all(axis=1)
        hidden[k, :end] = r.carried & ~seen
    return SessionRetrieval(
        pups=results,
        litter=litter_summary(results),
        t0=t0,
        end=end,
        fps=fps,
        px_per_cm=px_per_cm,
        contact_cm=settings.contact_cm,
        fields=fields,
        period=period,
        pup_xy_px=pups.xy,
        pup_nest_cm=pup_nest / px_per_cm,
        nose_pup_cm=nose_pup / px_per_cm,
        nose_xy_px=nose,
        nose_nest_cm=nose_nest / px_per_cm,
        hidden=hidden,
    )


def _cutoff_s(r: SessionRetrieval, protocol: RetrievalProtocol) -> float:
    """When scoring stopped after trial start: the cutoff, or the video's end if sooner.

    What a pup not retrieved is censored at -- 300 s for a pup watched for 230
    would claim 70 s nobody saw. A cutoff scored in full is reported as set,
    not as the whole frames it rounded to (19.99 s for 20 s at 29.97 fps).
    """
    scored = r.end - r.t0
    full = max(1, int(round(protocol.cutoff_s * r.fps)))  # as score_session cuts it
    return float(protocol.cutoff_s) if scored >= full else scored / r.fps


def pup_rows(
    r: SessionRetrieval,
    *,
    session_id: str,
    group: str,
    setup: RetrievalSetup,
    protocol: RetrievalProtocol,
) -> list[dict]:
    order = r.litter["retrieval_order"]
    return [
        {
            "session_id": session_id,
            "group": group,
            "pup": name,
            "start_in_nest": bool(p.start_in_nest),
            "contact_latency_s": p.contact_latency_s,
            "retrieval_latency_s": p.retrieval_latency_s,
            "retrieved": bool(np.isfinite(p.retrieval_latency_s)),
            "carries": p.carries,
            "drops": p.drops,
            "carry_distance_cm": p.carry_distance_cm,
            "retrieval_order": order.get(name, np.nan),
            "trial_start_s": setup.trial_start_s,
            "cutoff_s": _cutoff_s(r, protocol),
            "px_per_mm": setup.px_per_mm,
        }
        for name, p in r.pups.items()
    ]


def litter_row(
    r: SessionRetrieval,
    *,
    session_id: str,
    group: str,
    setup: RetrievalSetup,
    protocol: RetrievalProtocol,
) -> dict:
    order = r.litter["retrieval_order"]
    return {
        "session_id": session_id,
        "group": group,
        "pups_scattered": r.litter["pups_scattered"],
        "pups_retrieved": r.litter["pups_retrieved"],
        "time_all_retrieved_s": r.litter["time_all_retrieved_s"],
        "retrieval_order": " ".join(sorted(order, key=order.get)),
        "dam_in_nest_frac": r.dam_in_nest_frac,
        "trial_start_s": setup.trial_start_s,
        "cutoff_s": _cutoff_s(r, protocol),
    }


def _pad(flags: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[: len(flags)] = flags
    return out


def nest_distance_frame(r: SessionRetrieval) -> pd.DataFrame:
    """Every frame, long format: the dam's nose, then each pup."""
    n = r.n_frames
    frames = np.arange(n)
    parts = [
        pd.DataFrame(
            {
                "frame": frames,
                "time_s": frames / r.fps,
                "animal": "adult_nose",
                "x_px": r.nose_xy_px[:, 0],
                "y_px": r.nose_xy_px[:, 1],
                "dist_edge_cm": r.nose_nest_cm,
                "in_nest": r.nose_nest_cm <= 0,
                "nose_dist_cm": np.nan,
                "carried": False,
                "hidden": False,
            }
        )
    ]
    for k, (name, p) in enumerate(r.pups.items()):
        parts.append(
            pd.DataFrame(
                {
                    "frame": frames,
                    "time_s": frames / r.fps,
                    "animal": name,
                    "x_px": r.pup_xy_px[k, :, 0],
                    "y_px": r.pup_xy_px[k, :, 1],
                    "dist_edge_cm": r.pup_nest_cm[k],
                    "in_nest": _pad(p.in_nest, n),
                    "nose_dist_cm": r.nose_pup_cm[k],
                    "carried": _pad(p.carried, n),
                    "hidden": r.hidden[k],
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def range_values(r: SessionRetrieval, start: int, stop: int) -> dict:
    """Per-range numbers for frames ``[start, stop)``."""
    retrieved = sum(
        1
        for p in r.pups.values()
        if np.isfinite(p.retrieval_latency_s)
        and start <= r.t0 + round(p.retrieval_latency_s * r.fps) < stop
    )
    nose = r.nose_nest_cm[start:stop]
    nose = nose[np.isfinite(nose)]
    return {
        "pups_retrieved": retrieved,
        "dam_in_nest_frac": float(np.mean(nose <= 0)) if nose.size else float("nan"),
        "nose_nest_mean_cm": float(np.mean(nose)) if nose.size else float("nan"),
    }
