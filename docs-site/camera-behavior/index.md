# Camera & Behavior

GLIDER can watch your experiment through a camera, record synchronized video and
data, track animals or objects on screen, and score their behavior. This section
walks through each of those capabilities in the order you would normally use them.

## What's in this section

<div class="grid cards" markdown>

- **[Camera & Recording](camera.md)**

    Connect a camera, watch the live preview, and record video alongside a set of
    frame-aligned CSV files. Also covers calibration (pixels to millimeters) and
    zones (regions of interest).

- **[Tracking](tracking.md)**

    Detect and follow animals or objects with YOLO and ByteTrack, either live or
    as a batch pass over a recorded video. Requires the optional `vision` extra.

- **[Behavior Analysis](behavior.md)**

    Label short video clips, train a classifier from your labels, and apply it to
    new videos to produce an ethogram. Requires the optional `behavior` extra.

</div>

## Before you start

The camera and recording features work with a standard desktop install. Two of
the more advanced capabilities live behind optional dependency groups, so you
only install them if you need them:

| Capability | Extra | Install |
| --- | --- | --- |
| Live preview + video/data recording | *(included with `pc`)* | see [Installation](../getting-started/installation.md) |
| YOLO / pose tracking | `vision` | `uv sync --extra vision` |
| Behavior classification | `behavior` | `uv sync --extra behavior` |

!!! note "Video recording needs FFmpeg"
    Recording video requires FFmpeg to be installed on your system. If you
    followed the [Installation](../getting-started/installation.md) guide you
    already have it. Preview, tracking, and data logging work without it.

!!! tip "Two different things are called 'Behavior Analysis'"
    GLIDER has a **live, movement-based** behavior classifier (built into the
    camera settings, described in [Tracking](tracking.md)) and a **supervised
    machine-learning** workflow (a separate window, described in
    [Behavior Analysis](behavior.md)). They are unrelated tools. The pages below
    point out which is which so you don't confuse them.
