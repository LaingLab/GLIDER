# Arena-gated pose tracking

**Date:** 2026-09-04
**Branch:** `feat/arena-gate-drawn`
**Status:** design approved, not yet implemented

## Problem

YOLO-pose keypoints occasionally leave the arena — briefly, usually into a
corner, at speeds no mouse can produce. Two failures hide under that
description and they need different treatment:

* **The whole skeleton relocates.** The detector finds something that is not
  the animal — on the VMHAHA males, a patch of bench floor past the chamber
  wall, at likelihood 0.58–0.87. A confidence threshold cannot catch this
  because the detector is confident. It happened in 3–34% of male frames.
* **A single keypoint flicks out and back.** One body part jumps for a frame
  or two while the rest of the skeleton stays put.

Neither is caught today. `filtering.smooth()` is confidence masking, gap
interpolation, and a temporal median — all per-keypoint, all temporal, none
geometric. Nothing in the pipeline knows where the arena is.

The consequences are not cosmetic. Both smoothers in `CausalSpeed` will
*manufacture* a dart from one relocation: the 5-frame coordinate median holds
the old position for two frames then flips wholesale, emitting a single ~570 px
step, and the 3-frame speed mean spreads it across exactly `dart_min_frames`.
One detector glitch scores as one confirmed dart bout. This is how the VMHAHA
cohort arrived at a 345.7 cm/s dart threshold — 11.5 cm per frame, 21% of the
arena every frame.

`ArenaCalibration` (landed 2026-09-03) now knows exactly where the floor is.
This design uses it.

## Prior art

### The staged prototype

`.claude/worktrees/feat-arena-gate` holds a complete, tested, uncommitted
implementation from the August VMHAHA investigation, written before
`ArenaCalibration` existed. **Do not discard it.** Three of its ideas are
absorbed here:

* **Partial-skeleton rejection.** On that cohort 90–99% of out-of-arena
  detections were partial, making incompleteness the single strongest
  discriminator. Absorbed, with two changes — see section 1.
* **Blank the whole frame, not the offending keypoints.** NaN is already how a
  dropout is spelled everywhere downstream; `CausalSpeed` and
  `FreezeDartDetector` break their runs on it. A rejected frame becomes an
  honest gap rather than a fabricated speed.
* **The gate must travel with the thresholds.** From its `_gate_pose`: *"A
  cut-off calibrated against gated speed, applied to ungated speed, is the same
  class of mistake as thresholding one time window against another."* This is
  the constraint that shapes section 5.

What is replaced is its geometry. `ArenaBox` / `infer_arena_box` fit an
axis-aligned pixel box from pooled trusted centroids. That fits *occupancy*,
not the arena: a corner the animals rarely visit reads as out of bounds, an
axis-aligned box cannot follow a tilted view, it needs at least 100 trusted
detections, and it cannot help a session that is mostly broken. A drawn
perimeter has none of those limits and needs no data at all.

### What is already fixed

`CausalSpeed._forget()` (commit 9434483) resets the filter on a dropout. In
August, gating *without* that fix made sessions worse — Test 6 went 308.8 to
781.1 cm/s, because blanking a frame created a gap the stale reference
re-inflated. That coupling is resolved; the gate can now ship on its own.

### A pre-existing bug this design depends on fixing

`PoseBatchWindow._load_master` (`gui/pose_batch/window.py:905`) applies `loaded.entries` and
**never applies `loaded.arenas`**. Arenas are written to the master file by
`CalibrationSet.to_dict` but are silently discarded on read, so every arena
drawn today vanishes when the tool reopens. This is a bug in the arena feature
as shipped, independent of this design — but the mandate in section 6 cannot
work until it is fixed, and neither can the post-hoc pass, which resolves each
video's arena from the master file.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Gate scope | Inference **and** a post-hoc pass | 61 already-tracked sessions benefit without re-running inference |
| Rejection unit | Two-stage: per-keypoint strays, whole frame on quorum failure | Covers both failure modes; a per-keypoint-only gate leaves a 2-keypoint skeleton whose centroid is still garbage |
| Candidate selection | Prefer in-arena candidates at inference | Recovers frames instead of blanking them |
| Margin | 25% of the shorter arena side (7.5 cm on 30 cm) | Over-gating deletes real rearing invisibly; under-gating leaves visible glitches |
| Partial-skeleton test | Available, default off | See section 1 — default-on would blank every legitimately occluded frame and make the arena quorum unreachable |
| Enforcement | Hard block — no arena, no Run | A warning gets ignored, and the line is measurably worse |
| Arena copy | Copy as a starting point, confirm per video | A copied arena produces no residual warning if it does not fit |
| Post-hoc trigger | GUI button in Batch Pose Tracking | Operator-driven maintenance over a batch |

## 1. The gate

**New module `src/glider/vision/arena_gate.py`.** Qt-free like `arena.py`, so a
notebook can use it without a GUI import.

```python
@dataclass(frozen=True)
class ArenaGateSettings:
    margin_cm: float | None = None        # None -> 0.25 * min(width_cm, height_cm)
    min_inside_fraction: float = 0.5      # of DETECTED keypoints, must be in-arena
    min_detected_fraction: float = 0.0    # of ALL keypoints, must be detected; 0.0 = off

@dataclass(frozen=True)
class GateReport:
    frames_total: int
    frames_considered: int                # frames with >= 1 detected keypoint
    frames_blanked: int                   # whole-frame rejections
    keypoints_masked: int                 # per-keypoint only; excludes those inside
                                          # a frame that was subsequently blanked
    masked_by_keypoint: dict[str, int]
    settings: ArenaGateSettings
    arena_corners: list[tuple[float, float]]   # per-video fingerprint

    @property
    def blanked_fraction(self) -> float:
        """frames_blanked / frames_considered; 0.0 when nothing was considered."""

def gate_to_arena(pose, arena, *, settings=None, resolution=None)
        -> tuple[PoseData, GateReport]
```

**Work in arena centimetres, not pixel polygons.** Normalise `pose.xy` by frame
size, push through `arena.homography()`, and "outside" becomes a rectangle test
against `[-m, width_cm + m]` by `[-m, height_cm + m]`. Cheaper than
point-in-polygon, vectorised over all `T*K` points at once, and the margin is a
physical distance rather than a pixel count meaning something different at each
wall.

The gate does its **own** `(3,3) @ (3, T*K)` matmul rather than calling
`ArenaCalibration.to_arena_cm`, which routes through `_apply`
(`arena.py:174-188`) into `cv2.perspectiveTransform`. This keeps the whole
sweep in one float64 numpy expression and does not depend on OpenCV's
undocumented behaviour as the homogeneous divisor approaches zero — see the
horizon note below, which explains why no further guard is needed.

### Two independent fractions, not two sequential filters

The prototype's `reject_partial_frames` defaults `min_keypoints=None`, meaning
*every* keypoint must be present. Run before the arena test with that default,
it blanks both cases the arena quorum exists to distinguish — a 3-of-7 occluded
frame and a 6-detected/5-outside relocation are both simply "partial", and the
quorum never sees either. The two tests must therefore be independent
predicates on the same frame, not a pipeline:

1. `detected` = **finite `xy` AND `confidence > 0`**, per keypoint.
2. `inside` = per-keypoint arena test.
3. **Per-keypoint:** NaN every keypoint that is detected but outside.
4. **Per-frame blank** when `detected_count > 0` and either
   `inside_count / detected_count < min_inside_fraction`
   or `detected_count / n_keypoints < min_detected_fraction`.

**`detected` must test confidence, not just NaN.** The Ultralytics branch
appends `r.keypoints.xy[best]` unmodified (`vision/pose/core.py:424`) — it does *not*
NaN-mask below-threshold keypoints, despite the backend branch's comment at
`vision/pose/core.py:198` claiming parity. `mask_low_confidence` does that later, inside
`smooth()`, which the gate runs before. So at gate time an unlocalized keypoint
is `(0.0, 0.0)` at confidence 0 — a *finite* pixel in the frame's top-left
corner, which is outside every arena. Testing NaN alone would (a) make
`min_detected_fraction` inert on the inference path while live on the post-hoc
path, which reads an already-masked CSV, so identical `ArenaGateSettings` would
mean different things while the provenance block recorded them as the same;
(b) score a frame where YOLO localized 3 of 7 keypoints as 3/7 = 0.43 and blank
it under the default `min_inside_fraction`, which is exactly the
occluded-frame destruction that default-off `min_detected_fraction` exists to
prevent; and (c) inflate `keypoints_masked` with placeholders, so the quality
signal would not measure what it claims.

*Known limitation:* when a model emits no keypoint confidences, the YOLO branch
substitutes `np.ones(n_kpts)` (`vision/pose/core.py:428`), and `(0,0)` placeholders then read as detected.
The gate logs once when it sees a track whose confidences are uniformly 1.0.

**Zero-detection frames are excluded, not blanked.** A frame with
`detected_count == 0` is already all-NaN — both inference branches write it that
way (`vision/pose/core.py:415-416`). It is skipped before the quorum, so the
division is never attempted; it does not count toward `frames_blanked`, and
`blanked_fraction`'s denominator is `frames_considered`, the count of frames
with at least one detected keypoint. Left unguarded this is a
`ZeroDivisionError` in plain Python, or a silent `0/0 -> nan` in numpy where
`nan < 0.5` is False and the frame is quietly neither blanked nor counted.

`frames_considered` is a **declared field, not a derived one**: the block is
serialized with `dataclasses.asdict`, so a sidecar reader must be able to
reconstruct the fraction. An implementer reaching for `frames_blanked /
frames_total` instead would dilute the signal exactly on the sessions that
matter — heavy-dropout sessions are where blanking concentrates — so both the
10% warning and `_REJECT_WARN = 0.05` would under-fire. `frames_considered ==
0` (a wholly failed inference) yields `0.0`, not a second division by zero one
layer down from the empty-input guard.

They answer different questions — "how much of the skeleton did the detector
find?" versus "of what it found, how much is in the arena?" — and both are
fractions, so neither needs retuning per rig or model.

**`min_detected_fraction` defaults to 0.0 (off).** The partial test is a
cohort-specific remedy that worked on the VMHAHA males; default-on it would
blank legitimately occluded frames in every cohort. The arena quorum is the
general mechanism, the partial test the escalation for a known-bad cohort.
Setting it to `1.0` reproduces the prototype's behaviour exactly.

### One spelling for rejection

Gated keypoints get `xy = NaN` **and `confidence = 0`**, at both stages.
`mask_low_confidence` and the prototype's `reject_partial_frames` both blank
`xy` and leave confidence untouched, which writes contradictory rows a DLC
reader will trust. The absorbed code changes to match; two stages of one gate
must not spell rejection two different ways.

### Three traps

* **Resolution.** `arena.frame_size` is what the corners were *clicked* on; the
  video may have been tracked at another size. Prefer
  `pose.metadata["resolution"]` (as `_score_zones` already does), fall back to
  the explicit `resolution=` argument, then to `arena.frame_size`. Critically,
  `from_dlc_csv` does **not** populate `pose.metadata` — a CSV-loaded
  `PoseData` has no resolution of its own, so the post-hoc path must pass
  `resolution=` from `resolution_for_csv`. Without it the gate silently masks
  the wrong region. When none of the three is available, raise rather than
  guess.
* **The horizon — and why it needs no guard.** *(Corrected 2026-09-04 after
  numerical checking; an earlier revision of this spec required a `w > 0` test.
  It was wrong, and implementing it would have been a serious bug.)* It is
  tempting to reject points whose homogeneous divisor `w` is non-positive, on
  the reasoning that points past the vanishing line wrap back into plausible
  coordinates. Three things say otherwise. A homography is defined up to
  scale, so the sign of `w` is not intrinsic — on a steeply oblique rig it is
  negative across the **entire** arena (−1.42 at the centre of the test
  geometry), and `w <= 0` would blank every frame from that camera. The
  preimage of a bounded rectangle under a projective map is a bounded
  quadrilateral that cannot cross the vanishing line, so no point past the
  horizon can land inside arena-plus-margin; a sweep of the frame found zero
  such points. And as `w` approaches zero the coordinates go to `±inf`, which
  compares correctly against the margin, while a `0/0` NaN cannot arise because
  `H @ v = 0` has no non-trivial solution for an invertible `H` and
  `_check_simple` already rejects degenerate quads. **The bounded rectangle
  test in centimetres is sufficient on its own.** The gate still writes out its
  own matmul rather than calling `to_arena_cm`, but for a smaller reason: it
  keeps the whole `(T, K)` sweep in one float64 numpy expression instead of
  depending on OpenCV's undocumented behaviour as `w` approaches zero.
* **Empty input.** A zero-frame `PoseData` returns unchanged with a zeroed
  report rather than dividing by zero in `blanked_fraction`.

### Pipeline order

Gate, then mask, then interpolate, then median. Gating first means a 2-frame
glitch becomes a gap `interpolate_gaps` fills with a straight line through the
true path, while a 50-frame excursion exceeds `max_gap`, stays NaN, and
correctly trips `CausalSpeed._forget()`.

A `DegenerateArenaError` propagates; callers catch and skip, mirroring
`_score_zones`'s "never fails a video" discipline.

## 2. Arena-aware candidate selection

`core.py` currently keeps the highest box-confidence detection per frame
(`best = argmax(boxes.conf)`, `core.py:420`) and discards the rest. If YOLO
finds the real mouse at 0.85 and a blob on the bench floor at 0.90, GLIDER
takes the blob and the correct detection never reaches the CSV — unrecoverable
by any post-hoc pass.

`infer_video(..., arena=None, gate_settings=None)` — both, since the margin and
quorum live in `ArenaGateSettings` and re-ranking needs them. With `arena=None`
the path is byte-identical to today. With an arena: score each candidate by its
in-arena keypoint fraction,
prefer the highest box-confidence candidate clearing `min_inside_fraction`, and
**fall back to plain `argmax` when none clear it**.

That fallback makes this a re-ranking, not a filter: it can replace a bad pick
with a good one but can never turn a frame that had a usable detection into a
dropout. Blanking stays the gate's job.

Selection and blanking share one `ArenaGateSettings`, so they cannot drift into
a state where inference keeps a candidate the gate then deletes. Frame size is
known from the video here, so no resolution guessing.

**This applies to the Ultralytics branch only.** `infer_video` also routes
DLC/SLEAP models through `_infer_video_backend` (`vision/pose/core.py:141`), which yields
one detection per frame — there is nothing to re-rank. `arena=` is a documented
no-op there, logged once so it is not mistaken for a silent failure.

## 3. `run_batch` plumbing

New `arenas: Mapping[Path, ArenaCalibration]` and `gate: ArenaGateSettings | None`.

Pipeline: infer, write `_raw`, gate, smooth, write primary, score zones.

* **`_raw` must be pre-gate** — it is the "what did the model actually say"
  file you open when the gate looks wrong. Today it is written only when
  filtering is on; it now needs writing whenever *either* gating or filtering
  is active, or gating discards data with no companion.
  `raw_output_path`'s docstring (`vision/pose/batch.py:120`, "Written only when filtering is
  enabled") changes with it.
* **Gating before zone scoring is required, not incidental.** Centre-time
  computed from bench-floor detections is meaningless, and `_score_zones` runs
  on whatever `pose` it is handed.
* `run_batch` skips a video whose primary CSV already exists unless `overwrite`
  (`vision/pose/batch.py:339`). Enabling the gate therefore does **not** re-gate anything
  already tracked — that is the post-hoc pass's job, not a bug.
* The `GateReport` reaches the sidecar the same way section 4 describes: stashed
  on `pose.metadata` for `write_pose_meta` to emit. It survives the remaining
  filter stages because every step of `smooth()` goes through `PoseData.copy()`,
  which copies `metadata`.

## 4. The post-hoc pass

```python
def gate_pose_csv(csv_path, arena, *, settings=None) -> GateReport
```

`read_pose_meta` once, then `from_dlc_csv`, then `gate_to_arena` with
`resolution=` from that sidecar, then `to_dlc_csv`.

**The sidecar must survive the round trip.** `to_dlc_csv` calls
`write_pose_meta`, which rebuilds the sidecar from scratch off `pose.metadata`
— and `from_dlc_csv` populates no metadata at all. Writing naively therefore
destroys the existing `"resolution"` key, breaking the analysis viewer and
breaking idempotency (a second pass would find no resolution and, per the rule
above, refuse). So `gate_pose_csv` restores `resolution` onto
`pose.metadata` from the sidecar before writing, and stashes the `GateReport`
there for `write_pose_meta` to emit. (`fps` needs no such care —
`from_dlc_csv` already reads it via `fps_for_csv`.)

**Writes in place, preserving the original as `<stem>_ungated.csv` — and its
sidecar as `<stem>_ungated.meta.json`.** `meta_path` is stem-derived, so
renaming the CSV alone would strand the original without its fps and
resolution, making the "recoverable GPU hours" only partly recoverable. The
rename must happen *after* the read, since `from_dlc_csv` calls `fps_for_csv`
internally (`dlc.py:223-225`) and would otherwise lose the frame rate.

**Excluding the preserved file takes two changes, not one.**
`find_pose_csv`'s `_NOT_POSE_SUFFIXES` is not the only discovery path:
`_unique_pose_csvs` (`gui/behavior/window.py:2934`), the collector behind Build
Cohort that feeds `compute_cohort_thresholds`, globs `*.csv` and admits
anything whose stem contains `"DLC_"`, keyed on filename — it never calls
`find_pose_csv`. Left alone, every session would be pooled twice after the
post-hoc pass, once gated and once ungated, contaminating the very thresholds
section 5 exists to protect; its own docstring warns about exactly that
double-weighting. `_raw` has the same exposure today. So the exclusion list
moves to one shared constant consumed by both, which fixes the pre-existing
`_raw` leak in the same stroke.

The alternative (`<stem>_gated.csv` beside the original) leaves two pose CSVs
whose precedence depends on an mtime tie-break — implicit and fragile. In-place
means the 61 tracked sessions pick up the fix with **no change to any analysis
code**, while an inference run that cost GPU hours stays recoverable.

**Driven by a button beside Run/Cancel**, not in the calibration row: it acts
on the whole batch, off the same video list and calibration set. Its own worker
(a sibling to `PoseBatchWorker`, not an overload) on a QThread, reporting into
the existing log and overall progress bar. Enabled when every video has a
usable confirmed arena *and* at least one has a pose CSV on disk. Confirms
before running, since it rewrites in place.

**Always gate from the pristine original.** When `<stem>_ungated.csv` already
exists the pass reads *that* as its input and overwrites only the primary; it
never renames over an existing `_ungated`. Without this rule the documented
workflow destroys the original: run with defaults, read the report, escalate
the known-bad males to `min_detected_fraction=1.0` — and the second run renames
the already-gated primary over the true original, compounds the second gate on
the first, and records only the second settings in the `arena_gate` block. The
rename therefore refuses when the target exists rather than using
`os.replace`, which would clobber silently on both platforms. It refuses on a
second condition too: when the primary already carries an `arena_gate` block
and there is **no** `_ungated` companion. That is a primary gated at inference
time by section 3, which never writes an `_ungated`, so renaming it would make
the "the original survives" invariant false. The refusal points the operator at
`_raw`, which section 3 guarantees exists in exactly that case.

A video already carrying an `arena_gate` block with the same settings and arena
fingerprint is skipped. That comparison must be **value-normalised**:
`arena_corners` is declared `list[tuple[float, float]]` but returns from JSON as
a list of lists, so an identity comparison never matches and the skip never
fires. Settings compare field-wise, corners as nested floats.

Idempotency in the arithmetic holds for `min_detected_fraction` of 0.0 or 1.0.
At an intermediate value a second pass is *not* a no-op even on identical
settings: pass 1 sets `confidence = 0` on masked keypoints, which shrinks
`detected_count`, so a frame surviving at 5/7 can be blanked at 4/7. Gating
from `_ungated` makes this moot in practice, which is the other reason for the
rule above.

## 5. Gate provenance

The prototype's rule generalises: **whatever gating happened must be recorded
next to the data, and scoring must refuse thresholds derived under a different
gate.** This is what makes the in-place rewrite safe rather than dangerous.

* **`GateReport` goes into the `.meta.json` sidecar** as an optional
  `"arena_gate"` block, following the existing `"resolution"` pattern in
  `write_pose_meta` (which writes a whitelist, so this needs an explicit
  addition). No `META_SCHEMA_VERSION` bump: `read_pose_meta` is already
  tolerant and old readers ignore unknown keys.
* **`cohort_speed.py` records a *cohort-level subset* of that block** when
  deriving thresholds, replacing the prototype's `ArenaBox` / `n_rejected`
  fields while keeping its `SCHEMA_VERSION = 2` bump and its
  `_REJECT_WARN = 0.05` per-session callout. `from_dict`'s check must also
  relax from `!= SCHEMA_VERSION` (`cohort_speed.py:251`) to
  `not in (1, SCHEMA_VERSION)`, or every existing `cohort_speed.json` becomes
  unreadable rather than merely stale. **An absent block reads as ungated**, not
  as "skip the check" — gating did not exist when a v1 file was written, so
  absent-as-ungated is factually true, and the alternative would silently
  defeat the guard on precisely the stale files that motivate it.

  **`arena_corners` is deliberately not part of the cohort block.** It is
  per-video by construction — `compute_cohort_thresholds` (`cohort_speed.py:306`)
  pools 31 sessions with 31 different drawn perimeters, so no single
  fingerprint can match every sidecar and comparing corners would turn the hard
  error into an unconditional one. This is where the `ArenaBox` analogy breaks:
  the prototype's box was cohort-wide by construction, a drawn perimeter is
  not. The cohort block therefore carries **`ArenaGateSettings` plus a
  gated/ungated flag**, and nothing else. `arena_corners` stays in the per-video
  sidecar, where it serves the idempotency skip in section 4 and is well
  defined.

  **A mixed pool is refused.** The cohort block carries one gated/ungated
  boolean, so a pool combining gated and ungated sessions would hard-raise at
  scoring time for whichever half disagreed with it. `compute_cohort_thresholds`
  checks the sidecars up front and refuses a mixed pool with a message naming
  the odd sessions, rather than deriving a threshold that half the cohort
  cannot be scored against.

**Carrying it to the enforcement point.** `batch_apply` cannot see the cohort
file: `CohortSpeedThresholds` is loaded at `classify/__init__.py:260` and
reduced immediately to two floats, and `LiveInferenceConfig` carries only
`freeze_threshold` / `dart_threshold`. So `LiveInferenceConfig` gains
`gate_provenance: dict | None = None`, populated where the cohort object is
still in scope — structurally the same move the prototype made with
`config.arena`, carrying provenance instead of a box.

The comparison **raises on mismatch**, exactly as the prototype already refuses
a resolution mismatch: *"these thresholds were derived from ungated poses but
this CSV is gated; re-derive for this cohort."* A hard error, not a warning — a
silently mis-scored cohort is the failure this design exists to prevent.
(`gui/behavior/window.py:1900`'s `_cohort_window_mismatch` takes the softer
GUI-confirm route for time windows; a gate mismatch is not recoverable by
operator judgement in the same way, so it does not follow that precedent.)

**The check must sit before the batch/stream fork, not inside `batch_apply`.**
`batch_apply` declines and returns False on two paths that nonetheless go on to
read the CSV: an annotated output video (`classify/batch.py:428`) and a CNN
sequence model (`classify/batch.py:433`). The caller then falls through to
`LiveInferencePipeline`, whose `_make_tracker` builds a `PoseReplay` reading
`config.pose_csv_in` directly (`classify/pipeline.py:206-214`). Siting the check
inside `batch_apply` would let both of those score a gated CSV against ungated
thresholds with no error — precisely the failure being guarded against.

The anchor is **immediately after the `opts.update(resolve_speed_thresholds(...))`
block (`classify/__init__.py:508-529`)** — before `LiveInferenceConfig` is built
at 539 and well before the fork at 566. It cannot be earlier: `pose_csv_in` is
resolved at 479-484, but the cohort object is loaded *inside*
`resolve_speed_thresholds` (`CohortSpeedThresholds.load` at 260), so there is
nothing to compare against until that call returns. Returning
`gate_provenance` as a key in that function's dict lets it flow into
`LiveInferenceConfig(**opts)` for free.

**It runs only when a CSV is actually read.** The speed-only path passes an
already-loaded `pose=` with no CSV (`classify/__init__.py:573`); a direct
caller may also pass both, in which case `classify/batch.py:438` never touches the file.
With no sidecar there is no CSV-side provenance to compare, so the check is
skipped and logged — an in-memory `PoseData` handed in by a caller is that
caller's responsibility.

`resolve_speed_thresholds` is annotated `-> dict[str, float]`
(`classify/__init__.py:168`) and needs widening to carry the block.
`GateReport.settings` is a dataclass, serialized via `dataclasses.asdict`,
since `write_pose_meta` (`dlc.py:60-79`) builds a plain JSON payload.

**The report is the point, not a side effect.** 3–34% of frames out of arena is
the session quality signal that would have caught the VMHAHA males before they
reached a dart threshold. It also lands in the batch log per video via
`BatchEvent.message`, and a blanked fraction above 10% warns loudly.

## 6. Mandatory arena

**Fix the master-file round trip first** (see Prior art): `_load_master` must
apply `loaded.arenas` alongside `loaded.entries`. Nothing else in this section
works until it does.

New `CalibrationSet.missing_arenas(videos)`, parallel to `missing()` — a video
counts as missing when it has no arena, has one that raises
`DegenerateArenaError`, or has one that is copied-but-unconfirmed.

In `PoseBatchWindow._validate()` (`gui/pose_batch/window.py:1066`; the Run blocker is set at
line 1095) this **replaces** the line check rather than joining it: a usable
arena yields a scale by construction via `px_per_cm_centre`, so requiring both
would block a video whose arena is fine but which never had a line drawn. The
line becomes entirely optional.

**Backward compatibility is untouched.** `px_per_mm()` keeps its line fallback
and `missing()` keeps its meaning, because `load_px_per_mm` reads master files
written before 2026-09-03 that carry only lines. The mandate lives in the Run
gate, not the data model.

**Confirmed state lives on `CalibrationSet`, not `ArenaCalibration`** — the
latter is documented as pure geometry a notebook can build from four numbers,
and copy-provenance is a workflow concern. `set_arena(video, arena, *,
confirmed=True)`, an internal unconfirmed set, `is_arena_confirmed(video)`, and
`"arena_confirmed": false` written into the entry **only when unconfirmed**, so
normal files stay byte-identical and `schema_version` does not move — the
precedent the `"arena"` key itself set. `subset()` must carry the unconfirmed
set through, since `_write_master` saves through it; otherwise every save
re-emits the flag for every arena and the byte-identical goal is lost.

**Arena copy is net-new work.** No arena-copy path exists today —
`_copy_calibration_to_selected` (`gui/pose_batch/window.py:760`) operates purely on
`CameraCalibration` via `_retarget_calibration`. The new path stamps corners
onto the selected videos as *unconfirmed*, retargeting `frame_size` to each
target's own resolution the way `_retarget_calibration` does for lines. Copied
arenas do not satisfy the Run gate until the operator opens each and sees the
overlay sit on that video's floor: residuals are computed from the corners
alone, so a copied arena that does not fit shows *no warning at all*, and TRH
data shows camera height varied per animal — precisely the error the arena was
built to eliminate.

**`_clear_selected_calibrations`** (`gui/pose_batch/window.py:822`) calls only `discard`. With
arenas gating Run, Clear would appear to do nothing; it must also
`discard_arena`.

**UI:** calibration card badge becomes arenas-drawn (`12 / 31 arenas drawn`);
the table Status column gains "Needs arena" and marks copied-unconfirmed
distinctly; Arena becomes the primary action in that button row, Calibrate
demoted to secondary.

## Migration

* TRH's 30 sessions are already arena-calibrated (2026-09-03) and are ready —
  provided the `_load_master` fix lands, since their arenas are currently
  readable only from `arena_calibration.json`, not through the tool.
* VMHAHA's 31 need arenas drawn, but need re-tracking regardless.
* Any `cohort_speed.json` derived before gating goes stale once its CSVs are
  gated and must be re-derived. TRH should barely move (no out-of-arena
  detections reported); VMHAHA's are already known-bad and must not be reused.
* A rig whose floor perimeter genuinely cannot be drawn can no longer Run. The
  arena dialog accepts corners outside the frame, which covers close mountings;
  this is an accepted consequence of hard-block.

## Suggested landing order

Section 6 has no code dependency on sections 1–3 and touches a different
on-disk format. Landing it first is lower risk and immediately fixes the
`_load_master` data-loss bug:

1. **Master-file round trip + mandatory arena + copy/confirm** (section 6).
2. **The gate and its provenance record** (sections 1, 3, 5) — the arena is
   guaranteed present by then.
3. **Post-hoc pass and its button** (section 4).
4. **Candidate selection** (section 2), which is independent of all three and
   the easiest to measure in isolation.

## Testing

* `arena=None` produces byte-identical inference to today — the guard that this
  cannot silently move existing results.
* A keypoint near the vanishing line is masked by the bounded rectangle test
  alone, and a steeply oblique arena whose interior has `w < 0` throughout does
  **not** gate its own centre — the regression a `w > 0` guard would introduce.
* Pose tracked at a different resolution than the arena was drawn at uses the
  pose's resolution; a CSV-loaded `PoseData` with no resolution available from
  any of the three sources raises rather than gating the wrong region.
* Quorum with `min_detected_fraction=0.0`: 3-of-7 occluded but in-arena
  survives; 6-detected/5-outside is blanked.
* `min_detected_fraction=1.0` reproduces the prototype: any incomplete frame is
  blanked.
* Both stages set `confidence = 0` wherever they set `xy = NaN`.
* Margin: a keypoint 5 cm outside a 30 cm arena survives the default; 10 cm out
  does not.
* Raw Ultralytics output — `(0,0)` at confidence 0 for unlocalized keypoints —
  is not counted as detected, so a 3-of-7-localized frame is not blanked by the
  default `min_inside_fraction`, and `keypoints_masked` excludes the
  placeholders.
* Identical `ArenaGateSettings` produce the same blanking on a raw
  `(0,0)`-padded track and on the equivalent already-NaN-masked CSV.
* A frame with zero detected keypoints is excluded from the quorum, from
  `frames_blanked`, and from `blanked_fraction`'s denominator — no
  `ZeroDivisionError`, no silent `nan` comparison.
* Zero-frame `PoseData` returns unchanged with a zeroed report.
* In-arena candidate at 0.85 beats an out-of-arena one at 0.90; no in-arena
  candidate falls back to `argmax`; the backend (non-YOLO) path ignores `arena=`.
* `_raw` is written when gating is on and filtering is off.
* Gating is idempotent — a second `gate_pose_csv` blanks nothing further **and**
  the sidecar still carries `resolution` afterwards.
* A re-run with *different* settings gates from `<stem>_ungated.csv`, leaves it
  intact, and records only the new settings — the original survives and the
  gates do not compound.
* The idempotency skip fires across a JSON round trip, where `arena_corners`
  comes back as a list of lists rather than tuples.
* `<stem>_ungated.meta.json` is written beside `<stem>_ungated.csv`, and the
  rename happens after `from_dlc_csv` has read the fps.
* `find_pose_csv` never returns an `_ungated.csv`, and `_unique_pose_csvs`
  excludes both `_ungated` and `_raw`, so a gated folder pools each session once.
* A master file round-trips arenas: save, reload, arenas present.
* Run disabled for line-only and for copied-unconfirmed arenas; enabled once
  confirmed. Clear removes the arena as well as the line.
* A master file with no `arena_confirmed` key loads as confirmed; a line-only
  master file still yields `px_per_mm`; `subset()` preserves confirmed state.
* `cohort_speed.json` at `schema_version: 1` still loads.
* Gate provenance mismatch raises on **every** path that reads a pose CSV —
  including an annotated-output-video run and a CNN sequence model, both of
  which `batch_apply` declines before falling through to `LiveInferencePipeline`.
* The provenance check is skipped when no CSV is read (speed-only `pose=`).
* The cohort block carries `ArenaGateSettings` and a gated flag but **no**
  `arena_corners`, so a cohort pooled from 31 differently-drawn arenas still
  matches each of its videos.
* `compute_cohort_thresholds` refuses a pool mixing gated and ungated sessions,
  naming the odd ones.
* `blanked_fraction` is 0.0, not an error, when `frames_considered` is 0.

## Non-goals

**This does not fix the Test 6/7/8 failure class.** Those sessions have 9–20%
of their speed tail between two *complete*, 0.9-or-better-confidence skeletons
**both inside the arena** — e.g. Test 7 frame 10960 jumping (491,310) to
(339,113). No arena test touches that. Expect this design to move the male
sessions from unusable toward usable, not to clean. A displacement-plausibility
gate would be needed for the remainder, and is out of scope here.

**This does not change smoothing.** `speed_smooth=3` and the 5-frame coordinate
median stay as they are; both are load-bearing and removing either makes the
males worse, not better.

**The post-hoc pass cannot fully reproduce inference-time gating.** A tracked
CSV has already been through `smooth()`, whose 5-frame coordinate median smears
a relocation into its neighbours before the gate ever sees it. So the 61
existing sessions get a good approximation, not the same result they would get
from re-inference. Where a session matters enough to be exact, re-track it.
