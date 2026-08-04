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

Choose **Tools ▸ Behavior Analysis…**. If the `behavior` extra isn't installed,
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

## Stage 2 — Train

On the **Train** tab you fit a classifier from your labeled sessions:

1. Under **Training sessions**, click **Add session…** and pick a matching pose
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
