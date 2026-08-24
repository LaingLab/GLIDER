# glider-sleap

Run [SLEAP](https://sleap.ai) pose models in GLIDER.

Install this, point GLIDER at the folder SLEAP wrote, and it converts once and
runs. There is nothing to export beforehand and no command to remember.

```bash
uv pip install glider-sleap
```

## What it does

GLIDER runs pose models through onnxruntime, described by a `glider_pose.json`
sidecar next to the model. SLEAP writes a Keras checkpoint and a
`training_config.json`. This plugin is the bridge: it recognises the second
shape and produces the first.

Pick a model folder in the Camera panel. If it is a SLEAP folder that has not
been converted, GLIDER asks once, converts, and writes `model.onnx` and
`glider_pose.json` beside the checkpoint. Every later run loads the ONNX
directly. Retrain and drop in a new checkpoint and it notices the stamp is
stale and offers to convert again — a conversion that quietly answered with the
network you just replaced would be the worst outcome available.

Conversion runs in a child process, so TensorFlow does not stay resident in
GLIDER for the rest of the session. A few seconds to a couple of minutes,
depending on the model.

## It does not need SLEAP

That is worth stating plainly, because the opposite is the usual assumption.

A SLEAP model is saved as an ordinary Keras checkpoint. TensorFlow opens it on
its own; nothing in `sleap` is involved in reading one. So this plugin depends
on `tensorflow-cpu` and `tf2onnx` and not on `sleap` — which matters, because
`sleap` pins older Pythons than GLIDER runs on and could not share the
environment even if it were wanted. That ceiling applies to *running* SLEAP,
not to reading what it wrote.

Verified against SLEAP's own `minimal_robot.UNet.single_instance` test fixture:
loads with plain Keras, converts in one call, and the ONNX matches the original
network to within 7e-7 (Python 3.11, tensorflow-cpu 2.21, tf2onnx 1.17).

## Why it is a separate package

TensorFlow is a large thing to put in the dependency tree of a lab that tracks
with YOLO and has never opened SLEAP. Installing this plugin is how you say you
have SLEAP models. GLIDER itself carries onnxruntime and nothing else.

The same reasoning splits DeepLabCut into `glider-dlc`.

## Scope

Single animal. A converted model is run as one decode per frame, producing one
set of keypoints, which is what the behaviour classifier downstream expects.
Multi-animal SLEAP models (the ones with a PAF head) convert but are not
usefully runnable yet.

Python 3.11 to 3.13, the same range GLIDER supports. Conversion is verified on
tensorflow-cpu 2.21 with tf2onnx 1.17, and the dependency floors are set so a
resolver cannot land you below that: earlier TensorFlow has no 3.13 wheels, and
tf2onnx before 1.17 cannot convert a Keras 3 model, which is what any
TensorFlow new enough to be worth installing loads a SLEAP checkpoint into.

## Converting from the command line

The converter is a plain script and imports nothing from GLIDER, so it also
runs on its own — useful on a machine that trains models but does not run
experiments:

```bash
python -m glider_sleap.convert /path/to/sleap/model/folder
```

## Development

```bash
uv pip install -e ./plugins/glider-sleap
PYTHONPATH="src;plugins/glider-sleap/src" QT_QPA_PLATFORM=offscreen pytest plugins/glider-sleap/tests/
```

The suite runs without TensorFlow. The one test that performs a real conversion
skips when TF is absent, which is how CI runs it; everything else — detection,
staleness, and the error paths a researcher actually hits — runs everywhere.
