# Camera & Recording

The camera panel is where you connect a webcam, watch a live preview, and record
video together with a set of data files that stay aligned to the video frame by
frame. This page covers everything up to (but not including) tracking, which has
[its own page](tracking.md).

## Connecting a camera

Open the **Camera** panel. Near the top you'll find a **Camera** dropdown, a
**Refresh** button, and a **Settings** button.

1. Click **Refresh** to scan for connected cameras. GLIDER probes the first few
   device indices and lists whatever it finds (for example `Camera 0`,
   `Camera 1`). If none are found, the dropdown shows `No cameras found` and the
   preview button is disabled.
2. Pick your camera from the **Camera** dropdown.
3. Click **Start Preview**. The button changes to **Stop Preview** while the
   camera is streaming.

Above the selectors, a small status strip shows the live frame rate (for example
`30 FPS`) and the current resolution (for example `640x480`). A **REC** indicator
appears there while a recording is in progress.

!!! note "Camera names"
    On Windows, GLIDER reads the real device names. On macOS and Linux the
    cameras are listed generically as `Camera 0`, `Camera 1`, and so on, in
    device-index order. If you have several cameras, use **Start Preview** to
    confirm which is which.

## Camera settings

Click **Settings** to open the camera settings dialog. On its **Camera** tab you
can choose:

- **Resolution** — a dropdown with common sizes: `320x240`, `608x608 (Miniscope)`,
  `640x480`, `720x540`, `800x600`, `1280x720 (HD)`, and `1920x1080 (Full HD)`.
- **Frame Rate** — a value from 1 to 120 frames per second (fps). The default is
  30 fps.

The default camera setup is 640×480 at 30 fps with automatic exposure.

!!! tip "Match resolution to your needs"
    Higher resolutions and frame rates produce larger video files and more work
    for the computer, which matters if you also enable [tracking](tracking.md).
    Pick the lowest settings that still resolve the detail you care about.

## Multiple cameras

If your GLIDER build was started with multi-camera support, a **Multi-Camera**
checkbox is enabled in the camera panel. Turning it on switches the single
preview to a grid showing every detected camera at once.

The first camera added becomes the **primary** camera; you can change which one
is primary from the grid. Only the primary camera feeds the vision and tracking
pipeline — the others are recorded but not analyzed.

## Recording video and data

Recording is tied to the experiment lifecycle: when an experiment runs, GLIDER
starts the camera recording and the data logs together, and stops them when the
experiment ends. The camera panel reflects this with the **REC** indicator — you
do not arm recording from the panel itself.

Every recording run produces one video file and up to three companion data files.
They all share the same base name — your experiment name followed by a timestamp
(`YYYYMMDD_HHMMSS`) — so a single run's files sort together:

| File | Example name | Contents |
| --- | --- | --- |
| Video | `experiment_20260711_143022.mp4` | The recorded footage (MP4) |
| Device data | `experiment_20260711_143022.csv` | Device/sensor state, one row per frame |
| Tracking data | `experiment_20260711_143022_tracking.csv` | Per-frame tracking results (see [Tracking](tracking.md)) |
| Event log | `experiment_20260711_143022_events.csv` | Every device state change, with timing |

!!! note "Video format"
    GLIDER writes `.mp4` video encoded with H.264 (`avc1`). If your system's
    OpenCV build can't open the H.264 encoder, it automatically falls back to a
    widely-compatible MPEG-4 codec (`mp4v`) so recording still succeeds. Video
    recording requires FFmpeg — see [Installation](../getting-started/installation.md).

### The `frame` column: how everything lines up

The key idea that makes the recorded files useful together is a shared **`frame`**
column. Each processed camera frame gets a running index, starting at the first
frame of the session. That same index is written into:

- the tracking CSV (`frame` column),
- the device-data CSV (`frame` column), and
- the event log (`frame` column),

and it matches the frame number in the MP4. Because every processed frame ticks
all of the loggers once, you can join any of these files to any other — or to a
specific video frame — on the `frame` column alone.

!!! tip "Aligning by time instead of frame"
    Two extra time columns help when frame indices aren't convenient.
    `elapsed_ms` counts milliseconds from a shared session start, so it's
    comparable across all three CSVs. `flow_elapsed_ms` counts milliseconds from
    the moment your experiment flow starts (it's blank for rows recorded before
    then), which is handy for aligning traces to `t = 0` at the start of a trial.

### Device data CSV

The device-data CSV captures the state of your hardware, sampled once per camera
frame while recording. Its columns are:

```text
frame, timestamp, elapsed_ms, flow_elapsed_ms, <one column per device>, <one column per zone>
```

- Each device gets a column named `device_id:device_type` (for example
  `led1:LED`).
- Each [zone](#zones-regions-of-interest) gets a column named `zone:<name>` that
  holds `1` when the zone is occupied and `0` when it isn't.
- Boolean values are written as `1`/`0`, numbers to four decimal places, and
  anything unknown is left blank.

The file also begins with `#`-prefixed metadata rows (experiment name, start
time, board and device list, and subject details when present) and ends with the
end time and duration.

!!! note
    When you record without a camera, the device CSV is written on a timer
    instead (about every 0.1 s), and the `frame` column is left blank because
    there are no video frames to align to.

### Event log CSV

The device data CSV samples state once per frame, so a switch that flips and flips
back between two frames could be missed. The event log solves that by recording
*every* device state change the moment it happens, with sub-frame timing. Its
columns are:

```text
frame, timestamp, elapsed_ms, source, board_id, device_id, device_type, pin, pin_type, value
```

The `source` column tells you what kind of event each row is:

- `input_change` — an input pin changed (a sensor or button transition).
- `output_write` — GLIDER wrote to an output (digital write, analog, PWM, or servo).
- `flow_marker` — an experiment-flow boundary, with `value` of `start` or `end`.

The `frame` column holds the most recent camera frame seen when the event fired,
so events line up with the video too.

## Calibration

Calibration teaches GLIDER how many pixels correspond to a real-world distance, so
tracking can report movement in millimeters instead of pixels.

Open the camera settings dialog, go to the **Tools** tab, and click **Open
Calibration…**. In the calibration window:

1. Click **Capture Frame** to grab a still from the camera (an object of known
   size should be visible — for example the width of your arena).
2. Click **Draw Line**, then click two points to draw a measurement line across
   that known distance. Clicking near an existing endpoint snaps to it.
3. Enter a **Name** (for example `Apparatus Width`), the real **Length**, and the
   **Unit**. Supported units are millimeters (`mm`), centimeters (`cm`), meters
   (`m`), inches (`in`), and feet (`ft`).
4. Click **Add Line**. You can add several lines; GLIDER averages them to compute
   a **pixels/mm** ratio, shown in the info panel.

Use **Save…** / **Load…** to store a calibration as a JSON file and reuse it later.

Once a calibration is active, the tracking CSV's `distance_mm` and `cumulative_mm`
columns are filled with real distances. Without calibration, those columns simply
mirror the pixel values.

!!! note "Calibration is tied to resolution, not to a specific camera"
    A calibration is stored against the frame size it was drawn at and scales if
    the resolution changes. There is a single active calibration per session
    rather than one per camera, so in multi-camera setups it applies to the view
    you drew it on.

## Zones (regions of interest)

A **zone** is a region you draw on the camera view — an arena corner, a feeder, a
shelter — so GLIDER can report when a tracked object is inside it.

Open the camera settings dialog, go to the **Tools** tab, and click **Open Zone
Editor…**. In the zone window:

1. Click **Capture Frame** to grab a still.
2. Choose a shape: **Rectangle**, **Circle**, or **Polygon**.
3. Click **Draw Zone** and place the shape:
    - **Rectangle** — click two opposite corners.
    - **Circle** — click the center, then a point on the edge.
    - **Polygon** — click each vertex, then close it by clicking near the first
      vertex or pressing **Finish Polygon** (at least three vertices).
4. Give the zone a **Name** (for example `Feeding Area`) and optionally a
   **Color**, then click **Add Zone**.

Save and load zone sets as JSON with **Save…** / **Load…**.

!!! tip "Drawing zones on a recorded video"
    In the camera panel you can switch **Source** from **Live** to **Video file**,
    load a recording, and use **Draw Zones…** to define zones directly on a
    scrubbed video frame — useful for scoring footage after the fact.

### What zones give you

Once zones are defined, they show up in several places:

- **In the data** — the tracking CSV lists the zones an object is inside in its
  `zone_ids` column, and the device CSV gets a `1`/`0` occupancy column per zone.
- **In the node graph** — each zone can drive a **Zone Input** node that outputs
  `Occupied` (true/false), `Object Count` (how many objects are inside), and
  `On Enter` / `On Exit` triggers. Wire these into your experiment to react when
  an animal enters or leaves a region. See [The Node Graph](../building/node-graph.md).
- **In batch tracking** — an offline tracking run writes per-zone enter/exit
  events and total occupancy time (see [Tracking](tracking.md)).

## Working with recorded video

The **Source** control in the camera panel toggles between **Live** and
**Video file**. Choose **Video file** and **Browse…** to load an existing
recording; a seek slider and frame counter appear so you can scrub through it.
From there you can draw zones on the footage or run a batch tracking pass with
**Run tracking**, which is covered in [Tracking](tracking.md).

## Next steps

- Add automatic detection and animal tracking: [Tracking](tracking.md).
- Score behavior from pose data: [Behavior Analysis](behavior.md).
- Trigger hardware from zones and other inputs: [The Node Graph](../building/node-graph.md).
