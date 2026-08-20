# Pose models: YOLO, DeepLabCut, and SLEAP

GLIDER runs three kinds of pose model. Ultralytics YOLO weights (`.pt`) load
directly. DeepLabCut and SLEAP models run through ONNX, after a one-time export
you perform in your own DeepLabCut or SLEAP environment.

Once a model is prepared, **drag its folder onto the camera panel**. The panel
routes a dropped item by type, so a pose model, a behavior model (`.pkl`), and a
video can all be dropped the same way — together or separately.

## Why the export step exists

Converting a native DeepLabCut or SLEAP checkpoint requires the framework that
produced it: DeepLabCut's model classes to rebuild a snapshot, TensorFlow plus
`tf2onnx` for a SLEAP SavedModel. GLIDER does not depend on either.

That is not squeamishness about dependencies. Full `sleap` is TensorFlow-pinned
and has historically capped out around Python 3.10, while GLIDER targets
3.11–3.13 across Windows, macOS, and Linux — the two genuinely cannot share an
environment. Exporting once, in the environment that already has those tools,
keeps GLIDER installable and lets the same exported folder run on a Raspberry Pi
where neither framework will install at all.

## What GLIDER supports

| Format | Architecture | Status |
| --- | --- | --- |
| Ultralytics YOLO | pose `.pt` | Loads directly |
| DeepLabCut | single-animal (heatmaps + location refinement) | Via ONNX export |
| SLEAP | single-instance (confidence maps) | Via ONNX export |
| SLEAP | top-down, bottom-up | **Not supported** |
| Any | multi-animal | **Not supported** |

GLIDER's pose container is single-animal throughout, so multi-animal
architectures have nowhere to put a second subject. A SLEAP model with a
`centered_instance` or bottom-up head is rejected by name at load time rather
than silently producing one arbitrary animal's keypoints.

## Exporting

Install nothing new in GLIDER. Copy `tools/export_pose_onnx.py` into the
environment where DeepLabCut or SLEAP already works, and run it there — it
imports nothing from `glider`.

### DeepLabCut

Export the network to ONNX from your DLC environment, then run the helper to
write the sidecar beside it:

```bash
python export_pose_onnx.py dlc /path/to/dlc-project -o exported-model
```

### SLEAP

SLEAP takes two steps — its own exporter produces a TensorFlow SavedModel, which
`tf2onnx` converts:

```bash
sleap-export -m /path/to/sleap-model -o exported
```

```bash
python -m tf2onnx.convert --saved-model exported --output exported/model.onnx
```

```bash
python export_pose_onnx.py sleap /path/to/sleap-model -o exported-model
```

### What you get

```
exported-model/
    model.onnx
    glider_pose.json
```

`glider_pose.json` is what GLIDER actually reads. It records the ordered body-part
names, the output stride, the location-refinement scale, and the exact
preprocessing the network expects.

The helper **refuses to write a sidecar it cannot fully populate**. A guessed
stride or normalisation shifts every decoded keypoint by a constant — which
yields a plausible-looking skeleton rather than an error, and quietly corrupts
everything computed from it. A hard stop at export time is the cheapest place to
catch that.

## Using the model

Drag `exported-model/` onto the camera panel. GLIDER detects the format, shows it
(`Pose model: exported-model (DeepLabCut, 7 kp)`), and **fills in the keypoint
names from the model's own config, in training order**, locking the field.

That last part matters more than it looks. Keypoint order is not cosmetic: with
`auto_angles`, angle features are generated from keypoint *triplets by index*,
and the names at those indices are baked into the resulting column names. Enter
the same names in a different order and those columns never appear — they arrive
as NaN, every prediction comes back blank, and nothing raises. Reading the order
from the model removes the chance to get it wrong.

YOLO checkpoints record class names and a keypoint count, never body-part names,
so a dropped `.pt` leaves the names field editable and you must still type them
in training order.

Exported folders also work anywhere a pose model is selected — batch inference
over directories of video, and the offline behavior-classification path — through
the existing pickers.

## Installing the runtime

Running an exported model needs `onnxruntime`:

```bash
pip install 'glider[pose-onnx]'
```

CPU-only by design. It covers Python 3.11–3.13 on Windows, macOS, Linux, **and**
ARM, which is what makes exported models viable on the Raspberry Pi runner.

## Verifying a model before you trust it

GLIDER's own tests prove its decoding is internally consistent. They do not
prove it matches what DeepLabCut or SLEAP would have produced for the same
video — only a comparison against real output does that.

If you rely on absolute keypoint coordinates, run the parity check once per
model with a clip you have already analysed in the source tool:

```bash
GLIDER_POSE_FIXTURES=/path/to/fixtures pytest -m pose_parity
```

The fixture layout is documented in `tests/unit/vision/pose/test_parity.py`. A
roughly constant offset across every keypoint points at a stride convention; a
consistent scale factor points at the input scaling.

## Troubleshooting

**"no GLIDER pose sidecar or recognisable config found"** — the folder has an
`.onnx` but no `glider_pose.json`. Re-run the export helper against it.

**"GLIDER cannot tell which to run"** — more than one `.onnx` in the folder. Keep
one per folder, or name the right one in the sidecar's `onnx` field.

**"the pose model emits N keypoints but the behavior model expects M"** — the
pose model and the behavior classifier were trained against different keypoint
sets. They must match.

**"keypoint names are in a different order than the behavior model was trained
on"** — the names are right but the order is not. The message prints both
orderings.
