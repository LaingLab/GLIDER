# Scoring pup retrieval

In a pup retrieval assay the pups are scattered outside the nest, the dam is
returned, and the measure is how fast she brings each one home. GLIDER scores it
from a video: a pose model follows the dam, a segmentation model follows every
pup, and **Session Review** turns the two into a latency per pup and a few
numbers per litter — with the video, the nest and the timeline on screen so you
can check every one of them.

!!! warning "Needs the optional `vision` extra, and two models you bring"
    Both passes run YOLO models, so install the extra:

    ```bash
    uv sync --extra vision
    ```

    GLIDER doesn't ship either model. You need an **adult pose model** and a
    **semantic pup segmentation model** (an Ultralytics `.pt` file).

## What it measures

For each pup, counted from the **trial start**:

| Measure | Meaning |
| --- | --- |
| **Contact latency** | Seconds until the dam's nose first comes within 1 cm of the pup |
| **Retrieval latency** | Seconds until the pup is in the nest and stays there for 2 s |
| **Carries** | How many times she picked it up and moved it |
| **Drops** | Carries that ended outside the nest |
| **Carry distance** | How far the carries moved it, in cm |

And for the litter: how many pups were scattered, how many were retrieved, the
time until the last one was home, the order they came home in, and the fraction
of the trial the dam's nose spent inside the nest. A pup that starts in the nest
isn't scattered, so it has no latency and isn't counted.

A pup that was carried out of sight — in her mouth — is followed by her nose
until the nose reaches the nest. Often that is the only way a retrieval is seen:
the pup disappears into her mouth and comes out buried in the nest.

## Pick the Pup Retrieval protocol first

Everything on this page appears only when the open experiment's **Protocol** is
**Pup Retrieval**. Every other experiment sees none of it. The protocol is one
value shared across GLIDER, so set it once, anywhere:

- the **Protocol** dropdown on the **Experiment** tab (also in **Experiment →
  Experiment Settings…**), or
- the **Run** tab's title dropdown: pick the built-in **Pup Retrieval** preset,
  or one of your own presets named for it (`Pup Retrieval (P4, 5 min)` counts),
  or
- the **Protocol ▾** selector in Session Review's top bar.

Change it in one place and the others follow. The Batch Pose window's
**Pup segmentation** card and Session Review's **Retrieval** tab, **Pups**
toggle and retrieval exports all come and go with it. (**Open video…** is
available for every protocol.)

The built-in **Pup Retrieval** preset sets the protocol on whatever experiment
is open and asks for three things before each run: **Dam ID** (required),
**Postnatal day** and **Litter size**. It leaves the flow, devices and camera as
they are, and its runs are saved under a `pup-retrieval` folder. (There is a
built-in **Open Field** preset too, asking for **Animal ID** and **Group**.)
If one of your own presets is open when you pick a built-in, the experiment is
detached from that preset's file, so your preset is never overwritten: the next
Save asks where to save.

## Step 1 — track the dam and the pups

Open **Tools → Batch Pose Tracking**. With the protocol set to Pup Retrieval, a
**Pup segmentation** card appears.

1. Choose your **adult pose model**, and set its bodyparts in this order:
   `nose, ear_l, ear_r, spine_1, spine_2, spine_3, tail_base`. A YOLO `.pt`
   usually records only how many keypoints it has, not their names, so the
   **Bodyparts** field starts as `kp0` to `kp6`: rename them, `nose` first. The
   retrieval score reads the dam's nose, so one keypoint has to be named `nose`
   (or you name another one later, in **Thresholds…**). Left as `kp0`…`kp6`, there
   is no nose to find.
2. In **Pup segmentation**, choose the **Weights** of your pup model and set the
   **Litter size**, the number of pups in the video.
3. Add your videos and run the batch.

Each video gets its usual pose CSV and, beside it, a `<video>_pups.npz` holding
every pup's outline and position through the video. Pups are named `pup_0`,
`pup_1`, … in the order the tracker numbers them; the number is a slot, not an
identity that carries between videos.

A batch you resume adds pups to videos whose pose is already done. An existing
`_pups.npz` is kept (the log says `pups: kept existing …`) unless you tick
**Overwrite existing outputs**. A pup failure
is reported on the video but never costs you its pose.

## Step 2 — set up each video in Session Review

Open **Tools → Session Review**, make sure its **Protocol** selector reads
Pup Retrieval, and open your footage:

- **Open → Open video…** for one video, or
- **Open → Open folder as cohort…** for a folder of them, grouped by treatment
  like any other cohort.

Session Review finds each video's pose CSV and `_pups.npz` beside it, so keep
them together. A pup video inside a GLIDER recording folder (one recorded from a
Run-tab preset, say) *is* that recording's session: it goes by the recording's
name, brings its hardware lanes, and the recording isn't listed a second time.

The **Retrieval** tab on the right holds the setup. Nothing is scored until the
tab has all three of a nest, a trial start and a scale; it tells you which is
missing.

**Nest.** Scrub to a frame where the nest is clear and press **Draw nest**, then
outline it as a rectangle, circle or polygon. If the dam moves the nest, scrub to
the moment it moved and press **Redraw from here**: the new outline applies from
the playhead on, and the old one stays for the time before. (Pressing OK without
drawing anything changes nothing.) The list under the button shows each period;
click one to jump to it, or select it and press **Remove**.

**Trial start.** Scrub to the moment the dam is released and press **Set at
playhead**. Everything is timed from here.

**Scale.** **Set scale…** opens the [arena](camera.md#arena-calibration) tool on a
frame of the video: outline the floor and give its real size. The scale lives in that video's setup. If
more than one pup video is open it asks whether to use it for every session in
the cohort, and afterwards tells you about any session whose file couldn't be
read and so kept its old scale.

**Thresholds…** holds the rules every video in the cohort is scored by: the
cutoff, which keypoint is the nose, and the three thresholds below. It is saved
once, for the whole folder.

The tab's **Pups** table fills as soon as the video can be scored: one row per
pup, with the numbers above. Click a row to jump to that pup's retrieval (or to
its first contact, if it never came home). The line under the table gives the
litter.

!!! note "Your files are never overwritten if GLIDER can't read them"
    The setup is saved with the video as `<video>.retrieval.json`, and the
    thresholds at the cohort's root as `retrieval_protocol.json`. If one of them
    exists but isn't valid — or holds a value GLIDER would never write, like a
    negative trial start or a scale of zero — the tab says it is read-only and
    says why, and GLIDER leaves the file exactly as it found it. Fix or delete
    it by hand.

## What's drawn

With the **Pups** toggle on, the video shows:

- the **nest** in force at that moment, as a dashed outline;
- every pup's **outline**, numbered with its slot;
- a **dashed ring** at the dam's nose when a pup is hidden in her mouth, labelled
  with its number;
- a **line from her nose to the nearest point of the nest edge**.

The HUD names any pup being carried (`pup_2 carried`) and the nose's distance
(`nose 7.8 cm from nest`, or `nose in nest`).

On the timeline a **Retrieval** group shows:

- one **state lane per pup**: out of the nest, in contact, carried, in the nest —
  with a ▼ where it was retrieved;
- a **nose → nest (cm)** lane, with a dashed line at 0, the nest's edge;
- read-only flags for the **trial start**, each **nest** period and the
  **cutoff** (when the video runs past it).

The **Epoch table**'s **Metrics** menu gains **Pups retrieved**, **Dam in nest
(fraction)** and **Nose to nest (cm)** for any range you select or mark, and they
come out in the range CSV. A pup counts for a range if its retrieval falls inside
it.

## Exports

Both are in Session Review's **Export** menu, under Pup Retrieval only. Each asks
for a folder and writes the sessions that can be scored; any that can't are
named in the status line.

**Retrieval tables (CSV)…** writes two files:

- `retrieval.csv` has one row per pup per session: `session_id`, `group`, `pup`,
  `start_in_nest`, `contact_latency_s`, `retrieval_latency_s`, `retrieved`,
  `carries`, `drops`, `carry_distance_cm`, `retrieval_order`, `trial_start_s`,
  `cutoff_s`, `px_per_mm`.
- `retrieval_litter.csv` has one row per session: `session_id`, `group`,
  `pups_scattered`, `pups_retrieved`, `time_all_retrieved_s`, `retrieval_order`,
  `dam_in_nest_frac`, `trial_start_s`, `cutoff_s`.

**Nest distance per frame (CSV)…** writes `<video>_nest_distance.csv` for each
scored session — every frame, the dam's nose first (`adult_nose`) and then each
pup, one row per animal per frame: `frame`, `time_s`, `animal`, `x_px`, `y_px`,
`dist_edge_cm`, `in_nest`, `nose_dist_cm`, `carried`, `hidden`. `dist_edge_cm` is
the signed distance to the nest edge, so zero or below is inside. Videos that
share a name get a ` (2)`, ` (3)` suffix rather than overwriting one another.

## The defaults

| Setting | Default | Meaning |
| --- | --- | --- |
| Cutoff after trial start | 300 s | Scoring stops here; later retrievals don't count |
| Nose keypoint | `nose` | Which pose keypoint is the dam's nose |
| In nest for | 2 s | A pup has to stay in the nest this long to count as retrieved |
| Contact within | 1 cm | The nose is touching a pup if it's this close to its outline |
| Carried faster than | 3 cm/s | A touched pup moving faster than this is being carried; pups this young don't crawl that fast |

A pup that was **not retrieved** by the cutoff has a blank `retrieval_latency_s`,
not a made-up number. Every row carries `cutoff_s`, the time scoring stopped
after trial start — the cutoff in **Thresholds…**, or the end of the video if that comes
first — so you can treat those pups as censored at that time in a survival
analysis instead of dropping them.

All distances are in centimetres, from the scale you set. Three more knobs are
fixed in the code, not in **Thresholds…**: a carry that pauses for under half a
second is one carry, a touch that moves a pup less than 2 cm is sniffing and not
a carry, and a pup hidden for more than 10 s is taken to be buried, not carried.

## Limits

- **Pup numbers aren't reliable once the pups are in the nest.** They pile up
  there and the segmentation can't tell them apart, so a number can swap or merge
  in the pile. Trust each pup's retrieval and the litter totals; don't follow a
  single pup's identity after it is home.
- **The thresholds were checked on one video.** The defaults reproduce a hand
  count, frame by frame, on one lab video. Score a few of your own videos by
  hand and compare before you trust a cohort, and change the thresholds in
  **Thresholds…** if your pups or camera need it.
- **A missing nose is a missing score.** Where the pose model loses the dam's
  nose, nothing about contact or the nose line is known for those frames.
- **An ethogram and a pup video of one recording are two sessions.** If you have
  also run behaviour classification on a recording's video, its ethogram and its
  pup video both appear in the cohort, under different names. Open them as
  separate cohorts if that would double-count an animal.
- **One scale per video.** A camera that moves between videos needs each one
  measured; **Set scale…** can apply one measurement to the whole cohort when it
  didn't.
