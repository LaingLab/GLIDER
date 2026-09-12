# Behavior Analysis

The Behavior Analysis tool learns to recognize behaviors from examples you label,
then scores new videos automatically. You annotate short clips, train a classifier
on those labels, and apply the trained model to produce an **ethogram** — a
frame-by-frame record of what the animal was doing.

!!! warning "Behavior Analysis needs the optional `behavior` extra"
    This tool relies on a machine-learning stack that isn't installed by default.
    Add it with:

    ```bash
    uv sync --extra behavior
    ```

    This installs UMAP, HDBSCAN, scikit-learn, matplotlib, LightGBM, and PyYAML.
    Until they're present, the menu item stays disabled.

## Opening the tool

Open the **Analyze** tab and choose **Behavior Analysis**. If the `behavior` extra isn't installed,
the item is greyed out and its tooltip tells you what's missing and how to fix it:

> Install the behavior extra: pip install glider[behavior] (missing: …)

The Behavior Analysis window opens with three tabs — **Annotate**, **Train**, and
**Apply** — which correspond to the three stages of the workflow.

!!! note "This is the supervised, machine-learning tool"
    Don't confuse this with the live **Enable Behavior Analysis** option in the
    camera settings, which is a simple movement-threshold classifier described in
    [Tracking](tracking.md). This window learns behaviors from *your* labeled
    examples.

## What you need first

The workflow is built on **pose data** — body keypoints tracked over time — not on
raw video alone. So before you start you should have, for each recording:

- the **video file**, and
- a matching **pose CSV** in DeepLabCut format, named after the video (for example
  `trial1.mp4` and `trial1.csv`).

You can produce pose CSVs with GLIDER's pose tracking (see [Tracking](tracking.md))
or bring them from another tool.

## Stage 1 — Annotate

On the **Annotate** tab:

1. Click **Choose videos folder…** and select the folder of recordings.
2. Optionally click **Choose pose CSV folder…** if the pose CSVs live elsewhere
   (by default GLIDER looks in the videos folder).
3. Click **Launch annotator**.

GLIDER samples a spread of short clips across your videos — chosen to be varied
rather than all alike, so your labels cover a representative range of behavior —
and opens the **Behavior Annotator** window.

In the annotator you build up a **vocabulary** of behaviors and label clips one at
a time:

- Use **Add behavior** to create a behavior with a name (for example `rearing`)
  and a single-key hotkey.
- For each clip, press the behavior's hotkey to label it. You can also mark a clip
  as `multi-behavior` (**M**), `unclear` (**U**), or skip it (**Space**).
- Move between clips with the arrow keys, and trim a clip's in/out points with the
  `[ ]` and `{ }` keys.
- **Merge behaviors** folds one behavior into another across all your videos if you
  decide two labels should be one.

Your labels are saved next to each pose CSV as an annotations file (for example
`trial1_annotations.csv`), and your behavior vocabulary is saved as a
`<video>_behaviors.yaml` file so you can reuse it.

!!! tip "Label consistently"
    The classifier can only be as good as your labels. Decide clear criteria for
    each behavior before you start, and use `unclear` rather than guessing on
    ambiguous clips — those are dropped from training instead of adding noise.

### Labelling a multi-animal session

If a video was tracked with more than one animal (see
[Multiple animals](tracking.md#multiple-animals) in Tracking), the annotator
labels **one animal per pass**. Clicking **Launch annotator** asks, once for
each such video, which animal (numbered from 0) you want to label this time:

> Label one animal per pass: every zone you create is stamped with this
> slot. All animals are drawn; the subject is highlighted.

Cancelling that prompt cancels the whole launch rather than guessing. Every
zone you create afterward is stamped with the animal you picked, and the
clip view draws **every** tracked animal in the frame — your subject in
bright green with a larger keypoint marker, the rest dimmed grey — because
two animals in a social assay usually look identical on camera, and the
highlight is the only thing telling you which one you're scoring.

Trimming a clip's in/out points never changes which animal the zone belongs
to, even when you're re-trimming a zone rather than creating one fresh —
only its bounds move. This matters most after **Resume**, which can bring
back a mix of animals into one sitting (it replays every zone already saved
for that video, not only the animal you most recently chose): re-trimming
any of those keeps each zone's own original animal.

Your annotations CSV gains an `individual` column recording which animal
each zone belongs to. Two animals can be labelled with the *same* behavior
at overlapping frames — two mice grooming at once is the assay, not a
conflict — so the check that stops you double-labelling a clip now also
checks *which* animal, and only rejects a repeat for the same one. An
annotations file saved before multi-animal support existed, or a
spreadsheet-edited one with a blank cell in that column, still loads —
those zones read as animal 0.

## Stage 2 — Train

On the **Train** tab you fit a classifier from your labeled sessions:

1. Under **Training sessions**, click **Add sessions…** and pick a matching pose
   CSV and its annotations CSV. Add as many sessions as you have.
2. Optionally add **Holdout sessions** — recordings kept aside to test how well
   the model generalizes to data it never trained on.
3. Choose a **Classifier**:
    - `rf` — a scikit-learn **Random Forest**.
    - `lightgbm` — a **LightGBM** gradient-boosted model. (If LightGBM isn't
      available, GLIDER falls back to Random Forest automatically.)
4. With `lightgbm` selected, **Advanced…** opens the per-knob hyperparameters —
   see [Advanced LightGBM settings](#advanced-lightgbm-settings).
5. Optionally enable **Include background class** (treat unlabeled frames as a
   catch-all "background" behavior) and **Mirror augment** (add left/right-mirrored
   copies of your data to make the model robust to direction).
6. Click **Choose output file…** to pick where to save the trained model. It's
   saved as a single **model bundle** (`.pkl`).
7. Click **Fit**.

When training finishes, a summary appears in the results box: how many rows were
kept versus dropped, the classes learned, training accuracy, the most important
features, and — if you provided a holdout set — test accuracy, per-class
precision/recall/F1, and a confusion matrix.

!!! note "Add a holdout set to see honest test scores"
    Without a holdout set, only training accuracy is reported, which always looks
    optimistic. Holding out one or more whole sessions gives you a realistic
    estimate of how the model will do on new recordings.

### Advanced LightGBM settings

Most users never need this. The defaults are already tuned to be *more*
regularized than stock LightGBM, which deliberately gives up some training
accuracy to generalize better to sessions the model has never seen.

Reach for **Advanced…** when the training summary and your holdout scores
disagree — near-perfect training accuracy alongside mediocre holdout accuracy
means the model has memorized your training sessions rather than learned the
behaviors. Every field has a tooltip explaining what it trades off, and
**Restore Defaults** puts all of them back.

| Setting | What it does | To reduce overfitting |
| --- | --- | --- |
| **Boosting rounds** | How many trees are boosted in sequence | Lower it, or raise it alongside a lower learning rate |
| **Learning rate** | How much each tree contributes | Lower it, and add boosting rounds to compensate |
| **Leaves per tree** | LightGBM's main capacity dial | Lower it |
| **Max tree depth** | Hard cap on tree depth (`No limit` by default) | Cap it |
| **Min samples per leaf** | Fewest training frames a leaf may cover | Raise it — especially with few labeled bouts |
| **Min split gain** | Improvement a split must buy to be kept | Raise it above 0 |
| **Feature fraction** | Fraction of features each tree samples | Lower it |
| **Row fraction** | Fraction of training rows each tree samples | Lower it |
| **L2 regularization** | Penalty on confident leaf weights | Raise it |

These settings apply to LightGBM only — the Random Forest backend ignores them,
which is why **Advanced…** is greyed out when `rf` is selected. Values you set
are used for the runs you fit afterward; they aren't saved between sessions, and
leaving the dialog untouched means the built-in defaults are used.

!!! tip "Change one knob at a time"
    Hyperparameter changes are only meaningful if you can measure them. Add a
    holdout set first, note the holdout accuracy, then change a single setting
    and re-fit. Without a holdout set you'll only see training accuracy, which
    gets *better* as the model overfits.

### What the model learns from

Under the hood, GLIDER turns each frame of pose data into a compact set of
**geometric and kinematic features** — things like the distances between
keypoints (scaled by body length so size doesn't matter), a few body angles, and
each keypoint's speed, acceleration, and turning rate. These per-frame features are
then summarized over a short rolling window (about one second) using their mean,
standard deviation, and maximum. The classifier is trained on those windowed
features paired with your labels.

For the full mechanical account — what each feature column is, which rows get
dropped before training, what LightGBM actually does with the table, and how the
same numbers are reproduced at apply time — see
[Behavior Classifier Internals](../reference/behavior-model.md).

### Training on a multi-animal session

A multi-animal recording gives you one pose CSV per animal (`animal0.csv`,
`animal1.csv`, ... — see [Multiple animals](tracking.md#multiple-animals) in
Tracking) but only one annotations file shared by the whole video. **Add
sessions…** lets you pick several pose CSVs in one dialog, so select every
animal's CSV from that session together — GLIDER resolves the annotations
file for each one against the video, not the `_animals` folder, so
`animal0.csv` and `animal1.csv` both find the same shared file
automatically.

Each per-animal CSV becomes its own training session, filtered to that
animal's own zones out of the shared annotations file — animal 0's session
trains only on the behaviors labelled for animal 0, animal 1's only on
animal 1's, even though both live in the same CSV. A **Holdout sessions**
entry built from a per-animal CSV is filtered the same way.

### Social features

Underneath, GLIDER's feature extractor can measure a subject against
whichever *other* animal is nearest it, recomputed every frame — so it
isn't tied to there being exactly two animals in the video. When turned on,
it adds five columns:

| Column | Meaning |
| --- | --- |
| `social_distance` | Distance to the nearest other animal, in body lengths |
| `social_approach` | How fast that distance is changing: negative while closing, positive while separating |
| `social_bearing` | Angle between the subject's own heading and the other animal, in radians — `0` facing them, `π` facing directly away |
| `social_nose_to_nose` | Nose-to-nose distance, in body lengths — present only when the skeleton names a `snout` keypoint |
| `social_nose_to_tail` | The subject's nose to the other animal's tail base, in body lengths — present only alongside `social_nose_to_nose`, when the skeleton also names a `tail_base` keypoint |

This is off by default, so it changes nothing about a model trained before
it existed, and there is no control for it anywhere in this window today —
the Train tab has nothing that turns it on, so nothing you do here produces
a model that uses it. It exists at the level GLIDER extracts features from
pose data, ahead of the Train tab growing a way to reach it.

Two things about it are worth knowing anyway, because they explain behavior
you may see elsewhere in GLIDER: a model that *does* use these columns can
only ever be scored offline, never on the live camera feed — the live path
tracks one animal and has nothing to measure a social column against, so
GLIDER refuses to start live inference with such a model rather than run it
and leave the overlay stuck on "(waiting...)". And it cannot be combined
with mirror-augmented training, because mirroring flips only the subject,
which would put a real partner animal on the wrong side of a mirrored
arena.

## Stage 3 — Apply

On the **Apply** tab you score new videos with a trained model:

1. Click **Choose model bundle…** and select the `.pkl` you trained.
2. Click **Choose YOLO weights…** and select the pose model (`.pt`) — the same
   kind used to produce your training pose data.
3. Under **Video(s) to classify**, click **Add video(s)…** to add one or more
   recordings.
4. In **Keypoint names**, enter the keypoint names in the model's training order,
   comma-separated (for example `nose, left_ear, right_ear, ...`).
5. Optionally set **Classify every** — see [Classifier cadence](#classifier-cadence).
6. Click **Choose output folder…**, then **Run**.

GLIDER runs pose inference over each video and classifies it at the chosen
cadence. Videos are processed one at a time, each writing into its own subfolder
(named after the video) inside your output folder.

### Classifier cadence

**Classify every** sets how many frames pass between behavior predictions. The
default of 3 asks the model for a label about 10 times per second on 30 fps
video — fine resolution for scoring bouts, at a third of the inference work of
labelling every frame.

Set it to **1** when you want a label on every single frame: frame-accurate
onset and offset times, an `ethogram_raw.csv` you can join row-for-row against
another per-frame signal, or short behaviors whose bouts are only a few frames
long. The cost is proportionally more classifier calls, so an Apply run takes
longer.

Two things stay true whichever value you pick:

- Pose tracking and feature extraction always run on **every** frame. Only the
  behavior prediction is sampled, so the model still sees a fully populated
  rolling feature window.
- Bout durations, totals, and transition counts are corrected for the cadence,
  so `bouts.csv` and `stats.csv` report real seconds either way.

### Apply outputs

For each video, GLIDER writes:

| File | Contents |
| --- | --- |
| `annotated.mp4` | The video with the predicted behavior drawn on each frame |
| `ethogram_raw.csv` | The raw ethogram: one row per *prediction* (`frame`, `behavior`), so one row per video frame only when **Classify every** is 1 |
| `bouts.csv` | Continuous runs of a behavior: `state`, `duration_s` |
| `stats.csv` | Per-behavior totals: number of bouts, total and mean/median duration, and fraction of time |
| `transitions.csv` | How often each behavior followed each other: `from_state`, `to_state`, `count` |

These files give you both the moment-to-moment ethogram and the summary statistics
most behavioral analyses report.

### Multiple animals

A trained model applies to a [multi-animal](tracking.md#multiple-animals)
session too — no retraining needed, since the per-animal geometric features it
was trained on don't change when a video holds more than one animal. GLIDER
scores every animal in the session with the same model and writes each its
own ethogram, `animal{slot}_ethogram.csv`, beside that animal's own pose CSV
under `<video>_animals/` — same three-row layout as a single-animal ethogram,
same `write_ethogram_csv`. There is deliberately no `individual` column: the
tools that read an ethogram (Session Review among them) read a flat list of
rows keyed only by position, and a shared multi-animal file would silently
double-count every frame rather than error.

To reach this path, check **Reuse already-tracked pose CSVs** (on by default)
and add a video that was already run through Batch Pose Tracking with
**Animals** above 1. This matters because it's the *only* way in: tracking a
video fresh from inside the Apply tab is always single-animal, so a
multi-animal video with `reuse_existing_poses` off, or with no per-animal
files on disk yet, tracks and scores one animal, not the ones you tracked
earlier. Track it with Batch Pose Tracking first.

The result box reports which animals were scored and where each ethogram
landed, rather than the usual frame count and file list — there is no
`annotated.mp4`, `bouts.csv`, `stats.csv`, or `transitions.csv` for a
multi-animal run. Per-animal bouts, stats, and transitions are deliberately
not produced yet: the reporting layer (`run_report`, and the review tab that
reads it) has no notion of per-animal identity today, and building that is
more than this feature needed to be useful. The per-animal ethograms
themselves are complete and readable — open one directly in **Session
Review** (its file picker isn't specific to single-animal sessions; point it
at `animal0_ethogram.csv` and, if you want the keypoint view too,
`animal0.csv` as the pose CSV) — bouts and stats just aren't computed for you
automatically per animal the way they are for a single-animal run.

A multi-animal apply refuses three things outright, rather than scoring
something misleading:

- **A speed-only run** (no model — freezing/darting from the speed trace
  alone). The per-animal scoring path always goes through a model; there is
  no per-animal equivalent of the speed-only mode yet.
- **An annotated video.** The renderer that draws predicted labels onto the
  video is part of the single-animal streaming pipeline the multi-animal path
  doesn't use.
- **A CNN sequence model.** Per-animal scoring only knows the tabular,
  feature-based scoring path; a sequence model has no per-animal equivalent
  to fall back to yet.

Each of these fails fast with an error naming what's missing, rather than
silently scoring one animal or skipping a step. If you need one of them for a
multi-animal session today, score one animal's pose CSV at a time by passing
it as `pose_csv_in` — that goes through the ordinary single-animal path,
which supports all three.

!!! warning "Percentile speed thresholds don't work here yet"
    The percentile resolver still looks for a single pose CSV beside the
    video (it calls the same `find_pose_csv` a single-animal tool would, and
    was never taught to consult the per-animal files instead). For a
    multi-animal session that search always comes back empty — the pose data
    exists, just split across `animal0.csv`, `animal1.csv`, and so on — so
    the run fails with *"no pose CSV found"* rather than a clear explanation.
    The pose data is not missing, and GLIDER does know by this point that the
    session is multi-animal; percentile mode's own threshold lookup just
    can't find it yet. Use an absolute threshold (cm/s or mm/s) for a
    multi-animal run instead — those don't depend on finding a single file.

!!! note "A slot nothing was ever tracked into scores as blank, not missing"
    If consolidation never filled one of the N slots for a stretch of video,
    that slot's pose CSV holds no position for those frames — `PoseTracks`
    still requires every slot to span the whole video, so there is no way to
    omit them — and the ethogram scores those frames with an empty
    `behavior`, the same way any frame with missing keypoints scores blank.
    A slot that is blank throughout usually means consolidation never found a
    long enough fragment to seed it; check that animal's row in the
    `_identity.csv` sidecar (described in [Tracking](tracking.md#the-identity-sidecar))
    — a slot like that carries the `gap` flag on every frame, which is the
    tell that nothing was ever tracked there, not that scoring failed.

## The workflow at a glance

```text
Annotate            Train                         Apply
--------            -----                         -----
label clips  ─────► fit classifier on pose  ─────► score new videos ─► ethogram
(vocabulary)        features + your labels         with the model      + bouts/stats
                    → model bundle (.pkl)          (.pkl + pose .pt)    + transitions
```

## Next steps

- Produce the pose data this tool needs: [Tracking](tracking.md).
- Review how recordings and pose files are organized: [Camera & Recording](camera.md).
