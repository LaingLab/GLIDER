"""Rejecting pose detections that left the arena.

``filtering.smooth()`` is per-keypoint and temporal: it masks by confidence,
fills gaps, and medians. None of that catches the detector finding something
that is not the animal, because the detector is confident when it does -- on
one cohort it sat on bench floor past the chamber wall at likelihood 0.58-0.87,
well clear of the 0.5 the batch tracker masks at.

Catching that needs geometry, which :class:`~glider.vision.arena.ArenaCalibration`
now supplies. Keypoints are mapped onto the floor in centimetres and judged
against a rectangle with a margin, rather than tested against a pixel polygon:
the margin is then a physical distance instead of a pixel count that means
something different at each wall, and the test is a comparison rather than a
point-in-polygon walk.

The margin exists because the arena quad is the *floor plane*. An animal
rearing against a wall projects above that plane and lands genuinely outside
the quad, so a bare containment test would delete real rearing -- invisibly,
which is worse than leaving a visible glitch.

Qt-free on purpose: ``run_batch`` and a GUI button both drive this, and a
notebook can too.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from glider.vision.arena import ArenaCalibration

logger = logging.getLogger(__name__)

__all__ = [
    "ArenaGateSettings",
    "GateReport",
    "gate_pose_csv",
    "gate_to_arena",
    "inside_fraction",
    "ungated_path",
]

#: Margin as a fraction of the shorter arena side when none is given. A quarter
#: of a 30 cm arena is 7.5 cm, which clears any plausible rear -- a 9 cm rear
#: under a 1 m camera projects about 3 cm past the far wall -- while still
#: catching the bench-floor detections this exists for.
_DEFAULT_MARGIN_FRACTION = 0.25


@dataclass(frozen=True)
class ArenaGateSettings:
    margin_cm: float | None = None
    min_inside_fraction: float = 0.5
    min_detected_fraction: float = 0.0

    def margin_for(self, arena: ArenaCalibration) -> float:
        if self.margin_cm is not None:
            return float(self.margin_cm)
        return _DEFAULT_MARGIN_FRACTION * min(arena.width_cm, arena.height_cm)


@dataclass(frozen=True)
class GateReport:
    frames_total: int
    frames_considered: int
    frames_blanked: int
    keypoints_masked: int
    masked_by_keypoint: dict[str, int] = field(default_factory=dict)
    settings: ArenaGateSettings = field(default_factory=ArenaGateSettings)
    arena_corners: list = field(default_factory=list)

    @property
    def blanked_fraction(self) -> float:
        """Share of *considered* frames the gate blanked.

        Considered, not total: a frame the tracker never saw is already blank,
        and counting it would dilute the number on precisely the heavy-dropout
        sessions where blanking concentrates -- so the warning would under-fire
        exactly where it is needed.
        """
        return self.frames_blanked / self.frames_considered if self.frames_considered else 0.0


def _resolve_resolution(pose, arena, explicit) -> tuple[int, int]:
    """Frame size the keypoints were measured on.

    Order matters. ``pose.metadata`` is what the video was actually tracked at;
    the explicit argument is what a caller read from the sidecar; and
    ``arena.frame_size`` is only where the corners happened to be clicked. The
    post-hoc path *must* pass the explicit one -- ``from_dlc_csv`` populates no
    metadata at all, so a CSV-loaded track would otherwise fall through to the
    arena's frame size and be gated against the wrong region, silently.
    """
    for candidate in ((pose.metadata or {}).get("resolution"), explicit, arena.frame_size):
        if candidate:
            width, height = (int(v) for v in candidate)
            if width > 0 and height > 0:
                return width, height
    raise ValueError(
        "cannot gate without a frame resolution: pass resolution=, or give the "
        "pose a metadata['resolution'], or draw the arena on a sized frame"
    )


def _detected(pose) -> np.ndarray:
    """Boolean ``(T, K)``: keypoints the detector actually localized.

    Finite coordinates are not enough. The Ultralytics branch of
    :func:`~glider.vision.pose.core.infer_video` does not NaN-mask below-
    threshold keypoints -- ``mask_low_confidence`` does that later, inside
    ``smooth()``, which runs *after* this gate -- so an unlocalized keypoint
    arrives as ``(0.0, 0.0)`` at confidence 0: a finite pixel at the frame's
    top-left corner, which is outside every arena.

    Testing NaN alone would therefore make ``min_detected_fraction`` inert on
    the inference path while live on the post-hoc path, which reads an already
    masked CSV -- identical settings meaning different things while the
    provenance block recorded them as the same.
    """
    return np.isfinite(pose.xy).all(axis=-1) & (pose.confidence > 0)


def _outside(arena, xy_px, resolution, margin_cm) -> np.ndarray:
    """Boolean ``(T, K)``: keypoints beyond the arena plus its margin.

    **The bounded rectangle test is sufficient; do not add a horizon guard.**
    It is tempting to also reject points with ``w <= 0``, on the reasoning that
    the divide by ``w`` wraps points past the vanishing line back into
    plausible coordinates. It does not, and the guard is actively harmful:

    * A homography is defined up to scale, so the sign of ``w`` is not
      intrinsic. On a steeply oblique rig ``w`` is negative across the *entire*
      arena -- -1.42 at the centre of the one in the tests -- so ``w <= 0``
      would blank every frame of every video from that camera.
    * The preimage of a bounded rectangle under a projective map is a bounded
      quadrilateral that cannot cross the vanishing line, so no point past the
      horizon can land inside arena-plus-margin. Verified numerically: a sweep
      of the frame found zero such points.
    * As ``w`` approaches 0 the coordinates go to ``±inf``, and ``inf`` compares
      correctly against the margin. A ``0/0`` NaN would read as inside, but it
      cannot arise: ``H @ v = 0`` has no non-trivial solution for an invertible
      ``H``, and ``_check_simple`` already rejects degenerate quads.

    The matmul is still written out rather than calling
    :meth:`ArenaCalibration.to_arena_cm`, which routes into
    ``cv2.perspectiveTransform``: this keeps the whole ``(T, K)`` sweep in one
    float64 numpy expression and does not depend on OpenCV's undocumented
    behaviour as ``w`` approaches zero.

    NaN maps to NaN and every comparison against NaN is False, so an absent
    keypoint is not "outside" -- it is simply not present, which the caller
    accounts for separately.
    """
    flat = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    width, height = resolution
    homogeneous = np.stack(
        [flat[:, 0] / width, flat[:, 1] / height, np.ones(len(flat))], axis=0
    )  # (3, N)
    projected = arena.homography() @ homogeneous
    w = projected[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        x, y = projected[0] / w, projected[1] / w
        beyond = (
            (x < -margin_cm)
            | (x > arena.width_cm + margin_cm)
            | (y < -margin_cm)
            | (y > arena.height_cm + margin_cm)
        )
    return beyond.reshape(np.asarray(xy_px).shape[:2])


def gate_to_arena(pose, arena, *, settings=None, resolution=None):
    """Blank detections that left the arena. Returns ``(gated pose, report)``."""
    settings = settings or ArenaGateSettings()
    out = pose.copy()
    corners = [list(c) for c in arena.corners]
    if pose.n_frames == 0:
        return out, GateReport(0, 0, 0, 0, {}, settings, corners)

    if np.all(pose.confidence == 1.0):
        # Not a hypothetical: core.py:428 substitutes np.ones when a model
        # emits no keypoint confidences, and (0,0) pads then read as real.
        logger.warning(
            "%s: confidences are uniformly 1.0, so unlocalized keypoints "
            "cannot be told from real ones and (0,0) padding may be gated as "
            "out-of-arena",
            getattr(pose, "source", "this track"),
        )

    resolution = _resolve_resolution(pose, arena, resolution)
    detected = _detected(pose)
    outside = _outside(arena, pose.xy, resolution, settings.margin_for(arena))

    # Strays. A keypoint the detector never localized is not a stray.
    stray = detected & outside
    out.xy[stray] = np.nan
    out.confidence[stray] = 0.0

    # The quorum, as an independent predicate rather than a second filter.
    # Sequencing a partial-skeleton test before this one would blank both of
    # the cases it exists to distinguish: a 3-of-7 occluded frame and a
    # 6-detected/5-outside relocation are both simply "partial".
    detected_count = detected.sum(axis=1)
    inside_count = (detected & ~outside).sum(axis=1)
    considered = detected_count > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        too_few_inside = considered & (
            inside_count / np.maximum(detected_count, 1) < settings.min_inside_fraction
        )
        too_few_detected = considered & (
            detected_count / pose.n_keypoints < settings.min_detected_fraction
        )
    blank = too_few_inside | too_few_detected
    out.xy[blank] = np.nan
    out.confidence[blank] = 0.0

    # Keypoints inside a blanked frame are not strays: they were discarded by
    # the frame verdict, not by their own position. blank is (T,), so the
    # trailing axis is added to broadcast against stray's (T, K).
    counted = stray & ~blank[:, None]
    names = list(pose.keypoint_names)
    return out, GateReport(
        frames_total=int(pose.n_frames),
        frames_considered=int(considered.sum()),
        frames_blanked=int(blank.sum()),
        keypoints_masked=int(counted.sum()),
        masked_by_keypoint={n: int(counted[:, i].sum()) for i, n in enumerate(names)},
        settings=settings,
        arena_corners=corners,
    )


#: GateReport fields reconstructible from a stored block. `gated` is added on
#: write and is not a field; `settings` round-trips as a plain dict and has to
#: be re-hydrated, or the returned report would compare unequal to a fresh one.
_REPORT_KEYS = (
    "frames_total",
    "frames_considered",
    "frames_blanked",
    "keypoints_masked",
    "masked_by_keypoint",
    "arena_corners",
)


def _report_from_block(block) -> GateReport:
    fields = {k: block[k] for k in _REPORT_KEYS if k in block}
    return GateReport(**fields, settings=ArenaGateSettings(**block.get("settings", {})))


def _same_gate(block, settings, arena) -> bool:
    """Whether *block* records this exact gate. Value comparison, not identity.

    ``arena_corners`` is declared ``list[tuple[float, float]]`` but comes back
    from JSON as a list of lists, so an identity comparison never matches and
    the idempotency skip would never fire -- making every re-run rewrite, and
    the ``_ungated`` guard the only thing standing between a re-run and data
    loss.
    """
    if not block:
        return False
    corners = [[float(x), float(y)] for x, y in arena.corners]
    stored = [[float(x), float(y)] for x, y in block.get("arena_corners", [])]
    return block.get("settings") == asdict(settings) and stored == corners


def ungated_path(csv_path) -> Path:
    """Where :func:`gate_pose_csv` keeps the pristine original of *csv_path*.

    Single-sourced because ``run_batch`` has to recognise the same file: it
    removes a stale one when a fresh inference run replaces the primary the
    companion was taken from.
    """
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}_ungated{csv_path.suffix}")


def gate_pose_csv(csv_path, arena, *, settings=None) -> GateReport:
    """Gate a tracked CSV in place, keeping the original as ``_ungated``.

    Always reads the *pristine* track. When ``<stem>_ungated.csv`` already
    exists it is the input and only the primary is overwritten; the original is
    never renamed over. Without that rule the documented workflow destroys it:
    run with defaults, read the report, escalate a known-bad cohort to
    ``min_detected_fraction=1.0`` -- and the second run would rename the
    already-gated primary over the true original, compound the second gate on
    the first, and record only the second settings as provenance.

    **That rule is only sound while ``_ungated`` is the original of the
    *current* primary, and two things enforce it.** ``run_batch`` deletes the
    companion whenever it writes a new primary over the one it was taken from,
    and this function refuses when the primary carries no ``arena_gate`` block
    of its own. Every primary this pass writes carries one, so its absence
    beside an ``_ungated`` means the primary came from somewhere else -- in
    practice a re-run of inference -- and gating the companion would write the
    *previous* run's coordinates over it, with a provenance block describing
    the gate and saying nothing about the substitution. Refusing rather than
    picking one of the two files matches the rename below: this pass never
    silently discards a track, in either direction.

    The residue that neither catches is a primary gated *at inference time*
    sitting beside a stale companion, which the block cannot distinguish from a
    legitimate re-gate. Only a build predating the ``run_batch`` cleanup can
    produce it.
    """
    from glider.vision.pose.dlc import from_dlc_csv, meta_path, read_pose_meta, to_dlc_csv

    csv_path = Path(csv_path)
    settings = settings or ArenaGateSettings()
    ungated = ungated_path(csv_path)
    source = ungated if ungated.exists() else csv_path

    existing = (read_pose_meta(csv_path) or {}).get("arena_gate")
    if _same_gate(existing, settings, arena):
        return _report_from_block(existing)

    if not ungated.exists() and existing:
        raise ValueError(
            f"{csv_path.name} was gated during inference and has no _ungated "
            f"companion, so the original cannot be preserved. Re-gate from its "
            f"_raw file, or re-run inference with the settings you want."
        )

    if ungated.exists() and not existing:
        raise ValueError(
            f"{csv_path.name} has an _ungated companion but records no arena "
            f"gate of its own, so it was not written by this pass -- most "
            f"likely inference has been re-run since, and {ungated.name} is the "
            f"original of the run before it. Gating that would silently restore "
            f"the older track. Delete {ungated.name} to gate what is there now, "
            f"or put it back over {csv_path.name} to keep the older run."
        )

    # Read before renaming: from_dlc_csv reads fps from the sidecar.
    meta = read_pose_meta(source) or {}
    pose = from_dlc_csv(source)
    gated, report = gate_to_arena(pose, arena, settings=settings, resolution=meta.get("resolution"))

    if source == csv_path:
        # rename, not os.replace: refusing an existing target is the point.
        csv_path.rename(ungated)
        # Best-effort, like write_pose_meta itself: a CSV predating sidecars is
        # a supported case, and one missing file must not end a batch re-gate.
        if meta_path(csv_path).exists():
            meta_path(csv_path).rename(meta_path(ungated))

    if meta.get("resolution"):
        gated.metadata["resolution"] = meta["resolution"]
    gated.metadata["arena_gate"] = {**asdict(report), "gated": True}
    to_dlc_csv(gated, csv_path)
    return report


def inside_fraction(arena, xy, confidence, resolution, settings=None) -> float:
    """Share of one detection's *localized* keypoints that are in the arena.

    Factored out so :func:`gate_to_arena` and the candidate re-ranking in
    :func:`~glider.vision.pose.core.infer_video` cannot drift into a state
    where inference keeps a candidate the gate then deletes.

    Takes ``confidence`` as well as ``xy`` for the reason :func:`_detected`
    explains: raw Ultralytics output pads unlocalized keypoints with ``(0, 0)``
    at confidence 0, and scoring those as out-of-arena would make a good
    detection with a few pads lose to a confident false one. Returns 0.0 when
    nothing was localized, so an empty detection never wins a comparison.
    """
    settings = settings or ArenaGateSettings()
    xy = np.asarray(xy, dtype=float).reshape(1, -1, 2)
    confidence = np.asarray(confidence, dtype=float).reshape(1, -1)
    detected = np.isfinite(xy).all(axis=-1) & (confidence > 0)
    if not detected.any():
        return 0.0
    outside = _outside(arena, xy, resolution, settings.margin_for(arena))
    return float((detected & ~outside).sum() / detected.sum())
