# A pose model as a tracking backend

**Status:** approved 2026-08-25
**Reported as:** "when I load a SLEAP model and run tracking, it defaults to background subtraction."

---

## 1. The problem

It is not defaulting wrongly. There is nothing for it to default to.

`DetectionBackend` ([cv_processor.py:170](../../../src/glider/vision/cv_processor.py)) has four members —
`BACKGROUND_SUBTRACTION`, `YOLO_V8`, `YOLO_BYTETRACK`, `MOTION_ONLY` — and no pose
entry at all. A SLEAP or DeepLabCut model cannot drive tracking by construction,
so tracking falls to its default and the researcher's model is never consulted.

Underneath that, the Camera panel's pose picker and the tracking engine are two
subsystems that never touch:

| | written by | read by |
|---|---|---|
| `CameraPanel._pose_model_path` | the panel's **Pose model** picker | live behaviour classification, only |
| `CVSettings.backend` / `model_path` | the Camera **Settings** dialog, only | `CVProcessor` |

Nothing copies one into the other, and the Settings dialog's model field is
labelled for YOLO weights. So the model goes to the behaviour classifier and
tracking never hears about it.

**The expectation is reasonable and the UI invites it.** The panel reads
`Pose model: <name> (sleap, 2 kp)` beside a Run-tracking button, and nothing says
that model drives something else.

## 2. What already exists

Most of this feature is built. The keypoint pipeline is complete and is simply
never fed by anything but Ultralytics:

- `TrackedObject.keypoints` ([cv_processor.py:246](../../../src/glider/vision/cv_processor.py)), carried through tracking and serialised at :259
- `CVSettings.show_keypoints`, `keypoint_min_confidence`, and the bodypart-name list
- keypoint drawing at :1217, shared by live annotation and video export
- `load_pose_backend` ([pose/backend.py:234](../../../src/glider/vision/pose/backend.py)), which already returns a uniform
  `predict(bgr) -> ((K,2) xy, (K,) conf)` for SLEAP and DLC ONNX models and
  resolves a folder through `identify_pose_model`

The YOLO paths attach `det._keypoints` and it flows downstream untouched. What is
missing is a backend that produces those keypoints from a pose model.

## 3. What this is

A `POSE_MODEL` detection backend that runs the loaded SLEAP/DLC model per frame
and emits one detection carrying its keypoints, so everything already built for
YOLO-pose applies unchanged.

## 4. Non-goals

- **Multi-animal.** GLIDER runs single-instance pose models only, which the spec
  layer already enforces. One detection per frame, and the tracker's existing
  identity handling covers it.
- **GPU.** The ONNX backend is CPU-only by design; unchanged here.
- **Replacing YOLO-pose.** `YOLO_V8` keeps producing keypoints exactly as now.
- **Live behaviour classification.** It keeps reading `_pose_model_path` and is
  not rerouted through `CVSettings`.

## 5. The backend

```python
POSE_MODEL = auto()  # SLEAP / DeepLabCut single-instance keypoints
```

added to `DetectionBackend`, and to `_MODEL_BACKED_BACKENDS` so a changed
`model_path` forces a reload the way it does for YOLO.

`CVProcessor.initialize()` gains a branch that calls `load_pose_backend(
self._settings.model_path, conf_threshold=self._settings.keypoint_min_confidence)`.
Failure raises the same way a missing YOLO checkpoint does — the operator gets
the model's own error, not a silent fallback to background subtraction. **A pose
backend that cannot load must not degrade quietly into a different algorithm;**
that is the whole complaint this spec answers.

`_detect_objects` gains the matching branch:

1. `xy, conf = backend.predict(frame)`
2. Keep the keypoints whose confidence clears `keypoint_min_confidence`.
3. **Fewer than two clear it → no detection for this frame.** The same answer
   background subtraction gives when it finds nothing, so the tracker's existing
   disappeared-frame handling covers a brief occlusion. Two is the floor because
   one point has no extent to make a box from.
4. Otherwise emit exactly one `Detection`:
   - `bbox` = the extent of the confident keypoints, padded by `POSE_BBOX_MARGIN`
     (10% of the larger side, at least 4 px), clipped to the frame. A box that
     hugs the keypoints exactly would have zero area on an animal seen end-on.
   - `confidence` = mean confidence of the kept keypoints.
   - `class_name` = `"animal"`.
   - `_keypoints` = the full `(K, 3)` xy-plus-confidence array — **every**
     keypoint, not only the confident ones, so the CSV keeps the model's own
     output and a downstream consumer can apply its own threshold. Low-confidence
     points are already skipped at draw time by `keypoint_min_confidence`.

The centroid falls out of `Detection.__post_init__` as it does for every other
backend, so zones, velocity and behaviour states keep working unchanged.

## 6. Selecting the model

Choosing a pose model in the Camera panel now also points tracking at it:
`_apply_pose_model` sets `CVSettings.backend = POSE_MODEL` and
`CVSettings.model_path` to the **resolved** `spec.model_path`, then calls
`cv_processor.update_settings(...)`.

Two details that matter:

- **The resolved path, not the folder.** `_apply_pose_model` currently stores the
  original argument and discards `spec.model_path`, so `_pose_model_path` is a
  folder for a SLEAP export and a file for a `.pt`. Tracking gets the resolved
  `model.onnx`, which is what `_MODEL_BACKED_BACKENDS`' reload check compares.
- **The keypoint names come with it.** `spec.keypoint_names` is written into
  `CVSettings`, so the tracking CSV carries the model's own bodypart names in the
  model's own order rather than `kp0, kp1, …`.

The Camera Settings dialog gains **Pose Model** in its backend combo, and its
model field stops claiming to be YOLO-only when that backend is selected. Picking
background subtraction or YOLO there still overrides, so the panel picker is a
convenience rather than a lock.

## 7. What the operator sees

Loading a SLEAP model and pressing Run tracking now tracks with that model. The
Camera panel's label already names the model, its kind and its keypoint count;
the tracking controls stop silently disagreeing with it.

A pose model that fails to load reports why and tracking does not start. That is
a change in behaviour and the intended one — the alternative is what was
reported, where a quiet fallback produced plausible-looking results from a
different algorithm entirely.

## 8. Testing

- `POSE_MODEL` is in `_MODEL_BACKED_BACKENDS`, so changing `model_path` reloads.
- `initialize()` builds a pose backend for `POSE_MODEL` and raises, rather than
  falling back, when the model will not load.
- `_detect_objects` with a fake backend returning known keypoints emits exactly
  one detection, with the expected padded bbox and the mean confidence.
- Fewer than two confident keypoints → **no** detection.
- `_keypoints` carries every keypoint including low-confidence ones, not only the
  kept ones.
- The bbox is clipped to the frame when keypoints sit near an edge.
- A single confident keypoint does not produce a zero-area box (it produces
  nothing, per the floor of two).
- Selecting a pose model in the Camera panel sets the backend, the resolved
  `model.onnx` path, and the model's keypoint names on `CVSettings`, and calls
  `update_settings`.
- Selecting a YOLO `.pt` in the panel does **not** switch the backend to
  `POSE_MODEL` — `load_pose_backend` needs caller-supplied names for YOLO, and
  that path stays as it is.
- End to end on SLEAP's own `minimal_robot.UNet.single_instance`: convert, select,
  track a synthetic clip, and assert the result carries two named keypoints.

## 9. Files

| File | Change |
|---|---|
| `src/glider/vision/cv_processor.py` | `POSE_MODEL` member, `_MODEL_BACKED_BACKENDS`, load branch in `initialize`, detect branch, `POSE_BBOX_MARGIN` |
| `src/glider/gui/panels/camera_panel.py` | `_apply_pose_model` points tracking at the resolved model |
| `src/glider/gui/dialogs/camera_settings_dialog.py` | Pose Model in the combo; model field not YOLO-only |
| `tests/unit/vision/test_cv_processor_pose_backend.py` | new |
| `tests/unit/gui/test_camera_panel_pose_tracking.py` | new |
