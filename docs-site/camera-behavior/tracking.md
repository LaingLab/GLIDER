# Tracking

Tracking finds animals or objects in each video frame and follows them over time,
writing their positions to the tracking CSV. GLIDER uses YOLO for detection and
ByteTrack to keep a consistent ID on each object from frame to frame.

!!! warning "Tracking needs the optional `vision` extra"
    The detection models rely on [Ultralytics](https://docs.ultralytics.com/) YOLO,
    which is an optional dependency. Install it with:

    ```bash
    uv sync --extra vision
    ```

    This adds the `ultralytics` and `lap` packages. Without them, GLIDER still
    runs — it just falls back to the model-free detection modes described below.

## The detection backends

Tracking is configured in the camera settings dialog. Open **Settings** from the
camera panel, enable **CV Processing**, and choose a **Backend**:

| Backend | What it does | Needs `vision`? |
| --- | --- | --- |
| **Background Sub** | Detects moving foreground against a learned background | No |
| **Motion Only** | Reports motion without identifying objects | No |
| **YOLO v8** | YOLO object detection (bounding boxes per frame) | Yes |
| **YOLO+ByteTrack** | YOLO detection plus persistent multi-object track IDs | Yes |

For most animal-tracking work, **YOLO+ByteTrack** is the one you want — it gives
each animal a stable ID (`object_id`) so you can follow individuals across the
whole recording.

Alongside the backend you can set:

- **Confidence** — the minimum detection confidence to keep (default 0.5).
- **Min Area** — the smallest detection to accept, in pixels.
- **Skip Frames** — process every Nth frame to save CPU on slower machines.

The camera panel also has quick toggles for **Computer Vision** (on/off),
**Overlays** (draw boxes on the preview), and **Vision Cone**.

!!! note "If ByteTrack can't start"
    ByteTrack needs the `lap` package (installed with the `vision` extra). If it
    is missing, GLIDER logs a warning and quietly downgrades **YOLO+ByteTrack** to
    plain **YOLO v8** detection so you still get boxes, just without persistent IDs.

## Choosing a model

When you pick a YOLO backend, a model picker appears where you can browse to a
YOLO weights file (`.pt`). If you don't choose one, GLIDER uses the standard
lightweight `yolov8n.pt` model, which Ultralytics downloads automatically the
first time it's needed.

### NCNN models (faster on Raspberry Pi)

PyTorch inference on the Pi runs on the CPU and can starve the interface at
higher resolutions. For much faster live inference on ARM, export your model to
[NCNN](https://docs.ultralytics.com/integrations/ncnn/) and point GLIDER at it:

```python
from ultralytics import YOLO
YOLO("your_model.pt").export(format="ncnn")   # -> your_model_ncnn_model/
```

The export produces a `*_ncnn_model/` folder containing `model.ncnn.param`,
`model.ncnn.bin`, and `metadata.yaml`. In the model picker, browse into that
folder and select **`model.ncnn.param`** — GLIDER loads the containing folder
and reads the task (e.g. `pose`) from `metadata.yaml`. The Pi needs the `ncnn`
Python package installed (`pip install ncnn`).

!!! warning "Keep `metadata.yaml`"
    Always select a folder produced by Ultralytics' export, not a bare
    `model.ncnn.param`/`.bin` pair. Without `metadata.yaml`, Ultralytics can't
    tell a pose model from a detector and will misread keypoints.

For pose models, detected keypoints are drawn as dots on the preview when
**Overlays** are enabled (toggle with the *Show keypoints* CV setting).

### Installing Ultralytics on demand

Ultralytics is not bundled with GLIDER (it is licensed under AGPL-3.0). The first
time you select a YOLO backend without it installed, GLIDER offers to install it
for you:

- A dialog titled **Install YOLO (ultralytics)?** explains that roughly 700 MB
  will be downloaded (PyTorch plus model weights) and notes the AGPL-3.0 license.
- If you agree, GLIDER installs the `ultralytics` package into the active
  environment.

!!! warning "Packaged builds can't self-install"
    A frozen/packaged GLIDER build can't install packages into itself. In that
    case the dialog points you to install into a source checkout instead. For a
    normal `uv` install, running `uv sync --extra vision` up front avoids the
    prompt entirely.

## What tracking produces

While recording (or during an offline pass), the tracking results are written to
the `_tracking.csv` file. Its columns are:

```text
frame, timestamp, elapsed_ms, flow_elapsed_ms, object_id, class,
x, y, w, h, confidence, center_x, center_y,
distance_px, distance_mm, cumulative_mm,
zone_ids, behavioral_state, velocity_px_frame
```

Key columns:

| Column | Meaning |
| --- | --- |
| `frame` | Camera frame index — the join key to the video and other CSVs |
| `object_id` | The track ID assigned by ByteTrack (stable per object) |
| `class` | The detected class name (for example `mouse`, or `motion`) |
| `x, y, w, h` | Bounding box in pixels (top-left corner, width, height) |
| `center_x, center_y` | Bounding-box center, in pixels |
| `distance_px` | Distance moved since the previous frame, in pixels |
| `distance_mm`, `cumulative_mm` | The same distances in millimeters, when a [calibration](camera.md#calibration) is active (otherwise they mirror pixels) |
| `zone_ids` | Names of the [zones](camera.md#zones-regions-of-interest) containing the object |
| `behavioral_state` | Movement state, when the live classifier is on (see below) |
| `velocity_px_frame` | Speed in pixels per frame |

!!! note "Coordinates are in pixels"
    Positions and box sizes are always in pixels. The millimeter columns are only
    real-world distances when you have [calibrated](camera.md#calibration) the
    camera; otherwise they carry the pixel values unchanged.

A row with `object_id` of `-1` and class `motion` marks a frame where motion was
seen but nothing was identified; a periodic `heartbeat` row (also `-1`) marks
stretches with no activity so the file keeps ticking.

## Live vs. batch tracking

You can track in two ways:

- **Live** — with CV processing enabled, tracking runs during preview and
  recording, and results flow into the `_tracking.csv` of the current run.
- **Batch (offline)** — in the camera panel, set **Source** to **Video file**,
  load a recording, and click **Run tracking**. GLIDER asks for an output folder
  and processes the whole file.

A batch run writes several files into the chosen folder:

| File | Contents |
| --- | --- |
| `<video>_tracking.csv` | Per-frame tracking, same format as above |
| `zone_events.csv` | One row per zone enter/exit: `frame, elapsed_ms, zone_id, zone_name, object_id, event` |
| `zone_occupancy.csv` | Time per zone per animal: `object_id, zone_id, zone_name, frames_in_zone, seconds` |
| `<video>_annotated.mp4` | The video with boxes, track IDs, and zones drawn on |
| `metadata.json` | Source path, fps, frame count, resolution, zone and CV settings |

!!! tip "Batch timing follows the video, not the clock"
    In an offline run, timestamps are derived from the video's own timeline
    (`frame ÷ fps`), so the results are reproducible regardless of how fast the
    pass runs.

## Live movement-based behavioral state

The camera settings dialog has an **Enable Behavior Analysis** option (with
**Freeze**, **Immobile**, and **Darting** thresholds measured in pixels per
frame). This is a simple, rule-based classifier: it labels each tracked object's
movement each frame as freezing, immobile, moving, or darting, and writes that
into the `behavioral_state` column.

!!! note "This is not the machine-learning workflow"
    This live, threshold-based classifier is separate from the supervised
    [Behavior Analysis](behavior.md) tool, which learns behaviors from your own
    labeled examples. If you want to score complex behaviors (rearing, grooming,
    and so on), that is the tool to use.

## Pose and keypoint tracking

GLIDER can also run YOLO **pose** models, which return body **keypoints** (for
example nose, ears, tail base) per frame instead of just a bounding box. Pose data
is the input to the [Behavior Analysis](behavior.md) workflow.

Pose results can be exported in **DeepLabCut (DLC)** CSV/HDF5 format so they can be
used with downstream tools that expect that layout. (DLC here is an interchange
format — the pose inference itself is done by YOLO.)

## Multiple animals

YOLO pose models can track more than one animal through a video. In the
**Batch Pose Tracking** window (Analyze tab), set **Animals** to how many are
actually in the recording — it is not detected, it is the constraint you
supply, and it is what lets tracking stitch a tracker id that gets lost and
reassigned all night back into one lifelong animal. Too high and one real
animal is split into two that each vanish for half the video; too low and a
real animal is dropped entirely. DeepLabCut and SLEAP models cannot do this —
see [Pose models](pose-models.md#what-glider-supports) for why.

### Fragments into slots

ByteTrack answers "is this the same blob as last frame", and over a long
recording of animals crossing and occluding each other it answers that dozens
of times, handing out far more track ids than there are animals. Offline,
GLIDER has the whole video and the animal count up front, and uses both:
tracker fragments are stitched into exactly N lifelong slots — `animal0`,
`animal1`, and so on.

The longest fragments (above a minimum length) seed the N slots; slot numbers
are then assigned by which of those seeding fragments appears earliest in the
video, so `animal0` means the same thing on every re-run with the same
settings. Fragments below that minimum length are discarded outright, before
seeding or joining is considered — they never anchor a slot and are never
offered to one. Every fragment that cleared the length floor but was not
picked as a seed is offered to whichever slot it could plausibly continue —
"plausibly" meaning the speed implied by the gap and the jump between them is
under a threshold. A fragment nothing will take is dropped rather than
mis-joined; how many of *those* were dropped is recorded in the pose CSV's
`.meta.json` sidecar, under `consolidation`, and a video dropping a lot of
them is one whose tuning is wrong. Fragments discarded for falling under the
length floor are counted too, in the same block, as `consolidation.below_floor`
— a count, not the spans, so it says how many, not which. Without it, a
re-run that only changed `min_fragment_frames` produced metadata identical in
every other field; now the two runs are distinguishable after the fact.

### One pose CSV per animal

A run with more than one animal writes one plain, three-row DeepLabCut CSV
**per animal** — the same format a single-animal run has always written —
into a `<video's pose stem>_animals/` directory beside the video:

```text
session01DLC_exp-6.csv                  # n_animals == 1. Exactly as before.

session01DLC_exp-6_animals/             # n_animals > 1
    animal0.csv                         # plain three-row DLC CSV
    animal1.csv
    animal0_ethogram.csv                # see Behavior Analysis, below
    animal1_ethogram.csv

session01DLC_exp-6_identity.csv         # one shared sidecar -- see below
```

`n_animals=1` is unaffected in the strongest sense: one three-row CSV at
today's path, no subdirectory, byte-identical to a run from before this
existed. There is no in-between shape — tracking a video that already has a
single-animal CSV as multi-animal removes the old file (and its `_raw`
companion) once the new `_animals/` directory is written, and going back the
other way removes the stale `_animals/` directory once the fresh single file
lands. Re-running a multi-animal video with fewer animals than last time also
removes the extra slots' pose CSV and ethogram, so a directory listing never
shows an animal this run didn't produce.

Because each animal's file is the ordinary three-row format, any tool you
point at one directly opens it unmodified — behavior classification, the
annotator, cohort speed, training, zone scoring, and the arena gate all read
`animal0.csv` exactly as they read a single-animal session's CSV. There is no
second, four-row format any of them have to learn.

That is the format, not discovery. Every one of those tools normally finds
its input by asking `find_pose_csv(video)` for "the" pose CSV, and that
returns `None` for a multi-animal session (see below) — so *automatic*
discovery finds nothing, even though the file it would have opened is
perfectly readable. **Re-gate tracked CSVs** goes through exactly that path
and so skips multi-animal sessions outright: it has no per-animal mode, the
button greys out for a multi-animal video, and nothing is re-gated. The other
tools degrade more quietly — an empty state, a missing-file error, or a video
invisible to pooling — described session by session below.

Zones are still scored per animal, into the same `zone_events.csv` /
`zone_occupancy.csv` pair described above, with `object_id` reading `animal0`,
`animal1`, and so on.

### Finding a multi-animal session's pose data

`find_pose_csv(video)` — what every single-animal tool asks to get "the" pose
CSV for a video — has no single file to hand back for a multi-animal session,
so it returns `None`, exactly as if the video had never been tracked. It
logs, at info level, the video, how many animals it found, and which
directory — that log line is the only place the reason surfaces. This is
deliberate rather than an oversight: raising instead would abort every scan
built over a folder of videos (several exist) for the sake of one
multi-animal session among many.

**What this means in practice:** if you point a single-animal-only tool — or
`reuse_existing_poses` in a workflow that expects one file — at a video that
was tracked multi-animal, the video looks untracked. There's no error, no
dialog; the session is just missing from whatever list or scan you're
looking at. If a video you know you tracked isn't showing up somewhere,
check whether it was tracked with **Animals** set above 1, and check the log
for the "is a multi-animal session" line naming it.

Code that wants the whole set uses `find_pose_csvs(video)` instead, which
returns every animal's CSV in slot order (`animal0.csv`, `animal1.csv`, ...).
It's empty for a single-animal or untracked video, since those have no
`_animals/` directory to list.

### The four-row DLC CSV is an opt-in export

DLC's own convention for holding several animals in one file adds a fourth
header row, `individuals`, between `scorer` and `bodyparts`:

```text
scorer,my_yolo,my_yolo,my_yolo,...,my_yolo,my_yolo,my_yolo,...
individuals,animal0,animal0,animal0,...,animal1,animal1,animal1,...
bodyparts,snout,snout,snout,...,snout,snout,snout,...
coords,x,y,likelihood,...,x,y,likelihood,...
0,412.3,288.1,0.98,...,55.0,301.2,0.95,...
```

GLIDER no longer writes this as part of a batch run — the per-animal files
above are the only artifact `run_batch` produces for a multi-animal session.
The four-row file is now something you ask for: the **Export multi-animal
DLC CSV** button in the Batch Pose Tracking window (beside **Re-gate tracked
CSVs**) reads a session's per-animal files and writes the combined file
fresh, every time you click it. Because it's rebuilt from the per-animal CSVs
rather than kept in sync with them by hand, it cannot drift from what those
files actually hold — there's exactly one place the coordinates live, and
this is a view onto it, not a second copy.

Use the export when you need to hand a session to a tool outside GLIDER that
expects DeepLabCut's own multi-animal layout. Nothing inside GLIDER reads it
back except as an interchange format.

Reading a four-row file back requires saying which animal you want.
`from_dlc_csv` inspects the second row to tell a three-row single-animal file
from a four-row multi-animal one automatically, but for a multi-animal file
it **raises** unless you pass `individual=` — by name (`"animal1"`) or by
slot (`1`). Handing back the first animal by default would be
indistinguishable from a single-animal read, and silently analysing one
arbitrary mouse out of a social recording is the exact failure this feature
exists to end. A pre-existing three-row pose CSV — including each per-animal
file above — reads exactly as it always has; the format is unchanged, and
there is nothing to name.

!!! note "Already have four-row files from before this changed?"
    If you ran a multi-animal batch before per-animal files existed, the
    four-row CSV it wrote is still there and still reads back with
    `from_dlc_csv(individual=...)`. It just isn't what a fresh batch run
    produces anymore — that writes the per-animal layout above. This only
    holds until you re-track that video: the export lands at the same path a
    plain batch run would have used, so re-tracking it (single- or
    multi-animal) overwrites or deletes that file as part of the same
    reconciliation that replaces any other stale output. There's no undo. If
    you want a four-row file for a session that's already been re-tracked
    into per-animal CSVs, use **Export multi-animal DLC CSV** rather than
    keeping the old one, since only the export is guaranteed to match the
    current coordinates.

### The identity sidecar

Consolidation does not only observe identity, it *infers* it — a fragment is
joined across a gap on the strength of a plausible speed, not a certainty.
Every run of more than one animal writes one shared `_identity.csv`, recording
exactly where that happened for every animal in the session. It sits beside
the `_animals/` directory rather than inside it — `<stem>_identity.csv`, using
the same stem the directory's own name is built from — so an analyst does not
have to take the stitching on faith. Its `individual` column (`animal0`,
`animal1`, ...) is what tells rows for different animals apart within the one
file.

The file is **sparse**: a row exists only where an animal's identity at that
frame is in some doubt. A frame with no row for an animal is unambiguous —
measured, not inferred — so join it to the pose data with a left join on
`(frame, individual)` and read a missing `identity_flag` as empty, not as an
error:

```text
frame,individual,identity_flag
118,animal0,close
118,animal1,close
340,animal0,stitched
502,animal1,gap
```

| Flag | Means |
| --- | --- |
| `close` | Another animal's centroid was within 60 px (configurable) of this one on this frame — a real near-contact, or the moment two ids could have swapped. |
| `stitched` | This frame's data came from a fragment consolidation *joined* onto the slot, not the fragment that originally seeded it — an inferred link, not an observation. |
| `gap` | No detection at all for this animal on this frame — there is no position, so there is nothing for the id to be standing on either. |

`close` and `stitched` can both apply to one frame (written as `close+stitched`,
in that order); `gap` never combines with either, because a frame with no
position cannot also be measured near something.

### Tuning it

**Animals**, in the Batch Pose Tracking window, is the only one of these
exposed in the GUI. Three more knobs shape consolidation and the identity
sidecar; they take their defaults below unless you are driving `run_batch`
yourself from a script:

| Knob | Default | What it does |
| --- | --- | --- |
| `max_travel_px_per_frame` | 40.0 px/frame | The speed cap: a fragment joins a slot only if the implied speed to close the gap is at or under this. Raise it for a fast animal in a big arena; lower it to stop distant fragments from getting joined together. |
| `min_fragment_frames` | 5 frames | The length floor: a fragment shorter than this is discarded entirely, before seeding or joining is even considered — it never anchors a slot and is never offered to join one. Raising it does not just make seeding stricter; it silently removes more short tracker output from the result. |
| `identity_min_separation_px` | 60.0 px | The distance below which two animals' centroids mark each other `close` in the identity sidecar. |

### How much to trust it

!!! warning "Identity is a guess in places, and the guess is greedy"
    Consolidation is a greedy, longest-first heuristic — not a globally optimal
    assignment. It processes fragments one at a time, longest first, and joins
    each to whichever slot it could plausibly continue; it never goes back to
    reconsider a join once made. That is a real improvement over dropping
    every fragment shorter than the whole video, and it is also exactly where
    a swap gets baked in: two animals that cross, occlude, and separate can
    resume as fragments a plausible speed apart in the *wrong* order, and
    nothing after that point would look wrong. The CSV would still show two
    complete, sensible-looking tracks — just with the labels swapped from the
    crossing on.

    The identity sidecar tells you where to look, not that it is safe
    everywhere else. `stitched` frames are where a link was inferred rather
    than observed; `close` frames are where two animals were near enough that
    a mix-up was possible whether or not one happened. Treat conclusions that
    hinge on *which* animal did something — not just how many did — with
    suspicion through any `close` or `stitched` stretch, and check the
    annotated video at those frames before trusting the labels past them.

## Checking your GPU

Tracking and pose models run much faster on a GPU. To see what GLIDER will use,
open the **Analyze** tab and choose **GPU / Device Check**, or run it headless:

```bash
glider --gpu-check
```

The report lists your accelerator status and ends with the device inference will
use. GLIDER prefers, in order, **CUDA** (NVIDIA GPU), then **MPS** (Apple Silicon),
then **CPU**. The check works even when no GPU or PyTorch is present — it simply
reports what's missing, which makes it a good first stop when tracking is slow.

## Next steps

- Learn what the tracking data lines up with: [Camera & Recording](camera.md).
- Turn pose data into scored behavior: [Behavior Analysis](behavior.md).
- See which pose model formats support more than one animal: [Pose Models](pose-models.md).
