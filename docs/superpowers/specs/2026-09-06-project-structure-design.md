# Projects: giving an experiment a structure

## The problem

GLIDER writes artifacts. It does not own an experiment.

Every artifact is addressed by filename convention, and the relationships
between them live only in those conventions. A working session on the TRH
cohort hit four failures, all of which trace to that one fact.

**A sidecar drifted from the file it describes.** The `exp-7` pose metadata sat
in a `_meta_parked/` folder while the `exp-7` CSVs sat elsewhere.
`resolution_for_csv` therefore returned `None`, so `classify` computed no speed
axis, so freezing and darting were never scored at all. Nothing raised. It
surfaced only because 21,576 speed cells came back empty in an unrelated check.

**One session had four spellings.** `Test 1.mp4`, `test1_summary.csv`,
`test1_zone.json`, `Test 1DLC_exp-7.csv`. Mapping them needed a regex per
artifact type.

**Artifacts were grouped by kind, not by session.** `heatmaps/`, `summaries/`,
`arena_zones/`, `TRH_final_outputs/`, plus loose pose CSVs at the top level.
Seeing everything about one animal meant visiting five folders. Cohort-level
files (`cohort_speed.json`) sat among per-session ones with nothing marking the
difference.

**Sessions had no subject.** Recovering which video was which mouse - and so
which was saline and which TRH - required correlating pose tracks against
AnyMaze trajectories. An earlier attempt to infer it from distance rank agreed
with the truth on 0 of 15 videos and would have mislabelled the entire cohort.

A `Project` class already exists in `glider.analysis.behavior.project`. Nothing
imports it. It is a training-only manifest (`videos_dir`, `poses_dir`, `vocab`,
`window`, `holdout`) and knows nothing about arenas, calibration, subjects or
analysis outputs.

## The decision

**A project owns its directory layout.** Not describes it - owns it. Writers
take a `Session`, not an output path, so "the sidecar ended up somewhere else"
stops being a thing that can happen rather than a thing we remember not to do.

The alternative - a manifest indexing files wherever they lie - was considered
and rejected. It works on existing folders immediately and moves nothing, but it
cannot prevent any of the four failures above. It records them.

## Model

### Session

One recording and everything derived from it. A canonical id, and properties
that resolve artifacts rather than callers guessing filenames:

```python
session.video          # the recording
session.pose_csv       # current pose track
session.pose_meta      # its sidecar - resolved together, never separately
session.arena          # drawn corners, scale, derived from them
session.zone           # centre zone, derived from the arena
session.ethogram       # per-frame labels
session.bouts          # summaries
session.stats
session.subject        # identity and group
```

The point of `pose_meta` being a property of the same object that owns
`pose_csv` is that the two cannot be moved apart by anything short of deliberate
effort. That single relationship is what silently erased freezing from eleven
recordings.

### Project

The directory, the session list, and everything cohort-scoped. Holds what the
selected manifest sections carry:

- **Subjects and groups** - `subject_id`, `group`, `sex`, `strain`, `solution`,
  `dose`, `route`. The existing `core.experiment_session.Subject` already has
  all of these and needs no change; it simply never reached analysis.
- **Arena and calibration** - `arena_calibration.json` and
  `pose_calibration.json` fold in, so scale and arena travel with the session
  instead of being loose files that a reorganisation can orphan.
- **Analysis provenance** - which pose model, classifier and thresholds produced
  the current outputs, so a stale re-run is detectable. Mid-session a
  `train-6` CSV was replaced by a `train-2-4` one, and nothing recorded that the
  ethograms beside it were derived under the old model.
- **Protocol and trials** - stimulation epochs, trial boundaries, treatment
  timing. See Phasing: this is the largest surface and lands last, but the
  manifest reserves a place so it is not bolted on.

## Layout

Per-session artifacts live in the session folder. Cohort artifacts live at the
root. The distinction is currently invisible and needs to be structural:

```
experiment/
  glider_project.json          # manifest: sessions, subjects, provenance
  cohort_speed.json            # cohort-scoped, derived from all sessions
  calibration/                 # masters, if kept separate from per-session
  sessions/
    Test 1/
      Test 1.mp4
      Test 1_arena.json        # corners, scale, residuals
      Test 1_zone.json         # derived from the arena
      Test 1DLC_exp-7.csv      # pose
      Test 1DLC_exp-7.meta.json  # beside its CSV, always
      analysis/
        ethogram_raw.csv
        bouts.csv
        stats.csv
        run.json
      zones/
        zone_events.csv
        zone_occupancy.csv
      heatmap/
        Test 1_heatmap_3600-12599.csv
        Test 1_heatmap_3600-12599.png
```

This is close to what the same session arrived at by hand when reorganising the
TRH folder, which is weak evidence it is the right shape - it fell out of
needing to actually use the data rather than from designing in the abstract.

**Naming derives from one canonical session id.** No `Test 1` / `test1` /
`Test_1` drift. The id is the session folder name and every artifact is prefixed
with it.

## Adopting an existing folder

Existing cohorts are flat, inconsistent, and are real data. `Project.adopt()`:

1. **Scan** - classify every file by type and guess its session from the various
   naming conventions in play.
2. **Propose** - show the complete source-to-destination mapping, plus anything
   it could not classify. Nothing has moved yet.
3. **Confirm** - the operator accepts, or edits the mapping.
4. **Move** - rename on one volume, writing a reversal manifest first.

Constraints learned the hard way when doing this by hand:

- **Refuse to start if the plan has any collision, missing source, or
  destination outside the root.** Check the whole plan before touching one file.
- **Use `os.rename`, never `shutil.move`.** On a locked file `shutil.move` falls
  back to copy-then-delete and leaves the data in both places with the operator
  guessing which is real. That happened; `os.rename` fails having changed
  nothing.
- **Be resumable.** A move whose source is gone and destination present is
  already done, not an error. A network share will interrupt this.
- **Never overwrite in place.** Superseded artifacts move to `superseded/`
  rather than being deleted, so a result computed against them stays
  reproducible.

### What the real cohorts corrected

Dry-running the planner against the two live cohorts changed three decisions
that unit tests could not have caught. All three were found by reading a plan,
not by a failing test.

**A cohort holds several parallel derivations, deliberately.** `output_2_to_7v3/`,
`rescored_filtered/`, `jump_zones/` each contain a full set of outputs for the
same sessions. The first design merged them into one `analysis/` folder, which
produced 156 and 227 collisions - and would, if resolved by picking one, have
destroyed results that a published number depends on. Batches now keep their
own namespace: `sessions/<id>/runs/<batch>/`. Refusals fell to 0 and 12.

**A media folder is not a batch.** `males/` and `females/` hold recordings, so
what is in them is the data, not an alternate derivation. The distinction is
made by looking: a top-level folder that directly contains recordings is media.

**Canonicalise names only where GLIDER resolves by name.** The first design
renamed every artifact onto the session id, which would have rewritten 500-odd
extracted frame images. A labelling project indexes those by relative path, so
a tidier name would have broken somebody's dataset. Renaming is now limited to
`_arena.json`, `_zone.json`, `_summary.csv` and the recording.

Two findings about the data itself came out of the same exercise, and are the
reason doctor exists rather than an argument for it: `Test 17`'s outputs name a
`train-6` pose track that no longer exists beside them, and four
`jump_only_sessions` pose CSVs have no sidecar anywhere.

## Doctor

Every failure this session was mechanically detectable. A `project doctor`
check should report:

- A pose CSV whose sidecar is missing or elsewhere.
- A session with no subject, or a subject with no group.
- Analysis outputs whose recorded model differs from the pose CSV beside them.
- Naming drift - artifacts that do not match their session id.
- Sessions whose calibration is absent, or whose scale is an outlier for the
  cohort. (`scale_guard` already does the last one; it has nowhere to report.)

Warnings, never errors. A cohort mid-analysis is legitimately inconsistent.

## Phasing

1. **`Session`.** Resolution only, behind the current flat layout. Nothing
   moves, nothing breaks; writers can adopt it incrementally. *(Done -
   `glider.core.session`.)*
2. **Writers take a `Session`.** `classify`, the pose batch, zone scoring, the
   heatmap tools. This is where the sidecar class of bug stops being possible.
   *(Partial - `classify_session` gives the apply path a `Session` entry point.
   `classify` itself still takes ~30 separate arguments and several call sites
   depend on that signature; changing it is its own piece of work, and doing it
   carelessly would put a subtle fault in the path that produces every
   ethogram. The pose batch, zone scoring and the heatmap tools are untouched.)*
3. **`Project` and the manifest.** Sessions, subjects, calibration, provenance.
   *(Done - `glider.core.project`. Subjects are shared across a subject's
   sessions; group and treatment belong to the session, so a crossover is
   representable.)*
4. **Adopt and doctor.** *(Done - `glider.core.adopt`, `glider.core.doctor`,
   driven by `tools/project.py`.)*
5. **Protocol and trials.** Largest surface, least certain shape, and the only
   part with no existing model to build on. Deliberately last. *(Not started.)*

Phases 1 and 2 deliver most of the value. Phase 3 makes the subject work fall
out almost free, because `session.subject` is one more resolved property rather
than a parallel system.

## What this does not do

It changes no result. Not one number in the TRH or VMHAHA analysis moves. This
is plumbing, justified by the hours it cost and the near-miss it caused, not by
better science.

It is also not AnyMaze parity. Multiple arenas per camera, and cohort statistics
as a feature rather than a script, are separate pieces of work. This is the
foundation both would sit on.

## Testing

`Session` and `Project` are pure path and manifest logic, so they test without
Qt or video:

- Artifact resolution for every type, including a pose CSV whose sidecar is
  absent.
- Round-trip of the manifest, including subjects and provenance.
- Adopt: collision refused, plan verified whole before moving, resumable after
  interruption, reversal manifest replays.
- Doctor: each detectable failure above, using a fixture folder shaped like the
  real one - including the `_meta_parked` case, since that is the bug that
  motivated this.
- Backward compatibility: a flat folder with no manifest still loads, and
  anonymous outputs are readable rather than refused.

## Open questions

- Does the session id come from the video filename, or is it assigned? Filenames
  carry real meaning in existing cohorts (`226rr.1Test 19`), and discarding that
  loses information; keeping it means the id is not a clean identifier.
- Do the calibration masters stay as one cohort-level file, or split per session?
  Per session is more consistent with everything else here, but the pose batch
  edits them as a set.
- How much of the existing `analysis.behavior.project.Project` survives? It is
  unused, so replacing it outright costs nothing today - but the CLI training
  flow was written against its shape.
