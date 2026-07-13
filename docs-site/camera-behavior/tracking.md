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
| `zone_occupancy.csv` | Time per zone: `zone_id, zone_name, frames_in_zone, seconds` |
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

## Checking your GPU

Tracking and pose models run much faster on a GPU. To see what GLIDER will use,
open **Tools ▸ GPU / Device Check…**, or run it headless:

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
