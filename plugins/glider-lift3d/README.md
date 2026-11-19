# glider-lift3d

Mouse 3-D posture from one top-down camera, for GLIDER. **Experimental: estimates, not
measurements.**

* **Lift**: 2-D keypoints + the frame → a height surface of the mouse above the floor,
  five spine anchor heights, and a rear weight. Mask model (`mouse_seg.pt`) and depth
  model (`yolo26x-depth.pt`) are both optional.
* **Back-arch curvature** after Mittal et al. 2016 (*Blood Cells Mol Dis* 57:58-66): the
  side-view profile along neck → withers → centre → lumbar → tail base, fitted with an
  ellipse; eccentricity `e = √(a² − b²)/a`. Lower `e` = more hunched. Needs a model with
  a `neck` keypoint (the 10-keypoint model).
* **Session Review panels**: the 3-D mouse beside the video (drag to turn) with the
  back's height profile along the spine, and a *Posture (3-D)* timeline tab with arch
  eccentricity, body length and rear weight (click to seek).
* **Live**: a frame consumer for the live pipeline (`glider_lift3d.live.frame_consumer`).

## Install

From the GLIDER checkout, into GLIDER's own environment:

```
uv pip install -e plugins/glider-lift3d
```

Models live in `~/.glider/models/lift3d` (or the folder in `GLIDER_LIFT3D_MODELS`):

* `yolo26x-depth.pt`, Ultralytics' depth model (110 MB), downloads there on the first
  compute. It needs `ultralytics>=8.4.192`, which GLIDER now requires. Offline, the lift
  runs without depth and the depth-only arch stays blank.
* `mouse_seg.pt`, the mouse mask, is copied there by hand from the lab share. Without
  it the lift uses the keypoints alone.

## Use in Session Review

The panels start hidden. Turn them on in Session Review's **View** menu (*3-D mouse*,
and *Posture (3-D)* for the timeline tab); the choice is remembered.

Open a session whose pose CSV and video are found, with a scale (px/mm) or a drawn
arena. The *3-D mouse* panel offers **Compute 3-D for this session**: it lifts every
frame once (roughly 60 fps on an RTX 4090 without depth) and saves the results
into a folder `<pose>.lift3d/` beside the pose CSV:

| file | contents |
|---|---|
| `frames.csv` | per frame: rear weight, body ratio/length, spine length, `ecc_surface`, `ecc_depth`, arch peak |
| `surface.npy` | `(frames, 48, 48)` float16 height map of the 20 cm crop (~160 MB per 20 min at 30 fps) |
| `spine.npy` | `(frames, 5, 2)` neck … tail base in grid coordinates |
| `meta.json` | scale, models and keypoints used |

Once a session is computed, its range numbers join GLIDER's own in the Epoch table,
**Export → Range stats** and **Export → Time bins** (e.g. 1-minute bins): arch `e`
(surface and depth-only), body and spine length, and rearing fraction, as means over
the frames that have a value. `Arch frames (3-D)` is how many frames each arch mean
stands on; a bin spent mostly rearing has few. A session without a computed lift
leaves the columns blank.

Offline, without the GUI: `glider-lift3d VIDEO --pose-csv POSES.csv --ppc 11.7`.

## Reading the arch numbers

* `ecc_surface` comes from the lift's surface, which for a mouse on four paws is 75% an
  outline dome sized by how short the body looks. On Blast session 4 it correlates 0.54
  with body length: it largely re-states length.
* `ecc_depth` comes from the depth model alone, the only part that sees the back. Compare
  the two before trusting either.
* Both are blank while rearing (rear weight ≥ 0.5): a rear's profile is a ramp, not an
  arch.
* Not yet validated against side-view ground truth or grimace scores.
