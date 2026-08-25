# Multi-animal pose, part A: top-down inference

**Status:** approved 2026-08-25
**First of four.** See §9 for the decomposition and what the other three carry.

---

## 1. Why now

GLIDER runs single-instance pose models only. The lab's assays are social —
a labelled dataset to hand holds two to five mice per frame in 97 of 100 frames —
so the current path cannot serve them at all.

The constraint is not one check. It goes three layers deep:

| Layer | Shape today |
|---|---|
| `PoseBackend.predict()` | `(K, 2)` xy, `(K,)` confidence — no instance axis |
| `decode.py` `_peaks` | per-channel **global argmax**: one peak per keypoint by construction |
| `_from_sleap_config` | rejects every head but `single_instance` |

## 2. What is already ready

Most of the tracking stack was built collection-shaped and has simply never been
fed more than one detection:

| | |
|---|---|
| `ObjectTracker` (`cv_processor.py:417`) | a real N-object centroid tracker with persistent ids |
| `TrackingDataLogger` (`tracking_logger.py:690`, `:438`) | both CSVs loop objects, keyed by `track_id` |
| `ZoneTracker` / `ZoneState` (`zones.py:194`, `:366`) | `set[int]` of member track_ids per zone |
| `BehaviorAnalyzer` (`behavior_analyzer.py:75`) | `dict[int, ObjectBehaviorState]` |
| `VideoTrackingRunner` (`video_tracking_runner.py:174`, `:217`) | annotates every object, writes `object_id` per zone event |

So A is narrower than it looks: produce N instances, and five subsystems start
doing what they were written to do.

## 3. The contract gains an instance axis

`PoseBackend.predict()` becomes:

```python
def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(N, K, 2)`` xy in source-frame pixels, ``(N, K)`` confidence in [0, 1].

    N is the number of instances found; a single-instance model returns N == 1.
    """
```

One shape everywhere, rather than a second protocol and a branch at every
consumer. It also repairs something already broken: `pose_extract.py:20` does
`kp.xy[0]` with the comment *"the first instance is the main subject in a
single-animal setup"* — Ultralytics YOLO-pose already detects multiple animals
and GLIDER discards them. The new contract picks those up nearly for free.

`OnnxPoseBackend` and `UltralyticsBackend` return `N == 1` and `N == whatever
YOLO found` respectively. Neither needs new decode logic.

## 4. `TopDownPoseBackend`

New class in `src/glider/vision/pose/topdown.py`, holding two sessions.

```
predict(bgr)
  1. centroid model      -> anchor confidence map
  2. multi-peak decode   -> N anchor points          (§5, the only new decode)
  3. per anchor: crop    -> crop_size box around it
  4. centered-instance model on each crop
  5. decode_sleap_confmaps on each crop -- UNCHANGED
  6. map local xy -> source pixels: crop offset + the crop's own scale
  7. return (N, K, 2), (N, K)
```

Step 5 is why top-down was chosen first: the decoder that took a year of
coordinate-convention care is reused verbatim, per crop. Bottom-up would have
required replacing it.

**Crops are clipped to the frame and padded, never shifted.** An animal against
the arena wall would otherwise have its crop slid inward, and every keypoint in
it would come back displaced by an amount nothing downstream can recover.

**Empty result is empty, not an error.** Zero anchors above threshold returns
`(0, K, 2)`. That is the same answer background subtraction gives on an empty
arena, and `_detect_pose` already treats "no detection" as a normal frame.

## 5. Multi-peak decode

The one genuinely new piece. `_peaks` (`decode.py:30`) takes a global argmax per
channel — exactly one peak, by construction. Multi-animal needs local maxima:

```python
def find_peaks(confmap, *, threshold: float, min_distance: int) -> np.ndarray:
    """Local maxima above ``threshold``, thinned so none is within
    ``min_distance`` of a stronger one. Returns ``(N, 2)`` row/col."""
```

Both become `CVSettings` fields exposed in the Camera Settings dialog:

| setting | default | why |
|---|---|---|
| `pose_peak_threshold` | 0.2 | below SLEAP's own typical peak values, so a faint but real animal is not dropped |
| `pose_peak_min_distance_px` | 40 | in **source** pixels, converted to confmap cells using the model's stride, so the number means the same thing at any resolution |

The right values depend on arena scale and animal size, and a wrong
`min_distance` is the difference between "two mice" and "one mouse counted
twice". That is a mistake an operator must be able to correct without
retraining, so neither can be a constant. Both defaults are starting points to
tune against a real recording, not values with evidence behind them yet.

Sub-pixel refinement reuses the existing intensity-weighted-centroid window
around each peak rather than a second implementation.

## 6. Detections, tracking, and honest identity

`_detect_pose` emits **one `Detection` per instance** instead of one per frame.
The padded-bbox and mean-confidence rules already specified apply per animal,
unchanged. `ObjectTracker` then assigns persistent ids, and §2's five
subsystems light up.

**Identity is greedy and says so.** `ObjectTracker` is nearest-centroid with no
re-identification, so two mice that touch and separate can swap ids. With one
animal that was invisible; with two it silently corrupts per-animal data, and
social assays are made of animals interacting.

Rather than hide that, the tracking CSV gains an **`identity_flag`** column — a
short string, not a score, because the three causes need different responses and
a number would blur them:

| value | meaning |
|---|---|
| `""` | assignment unambiguous |
| `close` | another track within `identity_min_separation_px` (default 60) — a swap is physically possible |
| `reacquired` | this track was re-acquired after disappearing; the id is a guess, not a continuation |
| `tie` | the best and second-best candidate for this detection were within 20% of each other in distance |

Multiple causes join with `+` (`close+tie`), so a filter on "any flag set" is
`identity_flag != ""` and a filter on one cause is a substring test.

A researcher can then filter or hand-check those stretches. **This is the point
of the column: a number you can distrust on purpose beats one you cannot.**
Appearance-based re-identification and SLEAP's identity-predicting multi-class
models are both out of scope (§8), and this column is what makes shipping
without them defensible rather than negligent.

## 7. Finding the model pair

Top-down is two trained models. SLEAP names them
`<prefix>.<backbone>.centroid` and `<prefix>.<backbone>.centered_instance`.

Selecting either one in the Camera panel makes GLIDER strip the head suffix,
look for the sibling beside it, and **show what it found before using it**.

- No sibling → ask for the second folder. Never guess.
- More than one candidate → list them and ask. Same refuse-to-guess rule
  `BLEDevice._find_by_service` applies to six identical stimulators, for the
  same reason: picking one silently is worse than stopping.

`_from_sleap_config` stops rejecting `centroid` and `centered_instance` heads.
`PoseModelSpec` gains **`head_kind: Literal["single_instance", "centroid",
"centered_instance"]`**, so the pairing logic matches on the model's own
declaration rather than on its folder name. Folder naming is a convention SLEAP
happens to follow; the config is what it promises.

A `centroid` or `centered_instance` spec on its own is not runnable — selecting
one alone must produce the pairing prompt, never a half-configured backend.

`multi_instance` (bottom-up) stays rejected, with a message that now says
*not yet* rather than *not supported*.

## 8. Non-goals

- **Bottom-up (PAF).** Needs multi-peak plus part-affinity grouping — a second
  decode subsystem. Deferred deliberately, not forgotten.
- **Re-identification.** A real computer-vision subsystem, likely larger than
  the rest of A combined. §6's column is the honest interim.
- **SLEAP multi-class identity models.** Strongest science where animals are
  distinguishable, but a separate training workflow and a different head type.
- **Identity binding, per-animal nodes, multi-animal offline analysis.** These
  are B, C and D.
- **DeepLabCut multi-animal.** The same shape, but maDLC's `paf`/`identity`
  heads are a different conversion problem. A is SLEAP-only.

## 9. The decomposition, and what A leaves undone

| | | Depends on |
|---|---|---|
| **A** | Multi-instance top-down inference | — |
| **B** | Identity binding: `track_id` → `Subject`, so the CSV says `M001` | A |
| **C** | Per-animal experiment control: identity-addressable zone and behaviour nodes | A, B |
| **D** | Multi-animal offline analysis: `PoseData` gains an instance axis | A |

C and D are independent of each other.

**What A explicitly narrows, with a comment naming its fix:**

- `LiveBehaviorClassifier` (`live_behavior.py:156`) takes instance 0. It
  classifies one label per frame and `BehaviorEvent` has no subject field. D.
- `PoseData` keeps `(n_frames, n_keypoints, 2)`. Its own docstring already
  anticipates the instance axis. D.
- `ZoneInputNode` fires `On Enter` with no argument saying which animal. C.

**One thing C will have to build regardless of animal count:**
`CVProcessor.on_zone_update` and `ZoneInputNode.update_zone_state` are **never
wired together anywhere** — `live_signals.py:4` documents it. Zones do not reach
the flow graph today, for any number of animals. C builds that hop, and should
build it identity-aware rather than building it twice.

## 10. Testing

**Decode, against synthetic maps — no model, no onnxruntime:**
- `find_peaks` returns N peaks for N separated blobs, and **one** for two blobs
  closer than `min_distance`
- a peak below `threshold` is not returned
- an empty map returns `(0, 2)`, not an error
- peaks on the array border are found and not clipped away
- sub-pixel refinement matches the existing single-instance result for a
  single-blob map, which pins the reuse

**The backend, with stub sessions:**
- two anchors produce two instances, each with the full keypoint set
- local keypoints map back to source pixels — an anchor at a known offset puts
  its keypoints at the expected absolute position
- a crop clipped at the frame edge is **padded, not shifted**: an animal against
  the wall keeps its true coordinates
- zero anchors returns `(0, K, 2)` and no exception
- a single-instance model still returns `N == 1` through the new contract

**Tracking and identity:**
- two instances become two `Detection`s and two persistent track ids
- both CSVs get one row per animal per frame, with distinct `object_id`s
- `identity_confidence` flags a frame where two tracks are within the threshold
- it flags a re-acquisition after a disappearance
- it does **not** flag two animals that stay well apart — a column that always
  fires says nothing

**Pairing:**
- selecting a centroid folder finds its centered-instance sibling and reports it
- a missing sibling asks rather than proceeding
- two candidate siblings ask rather than picking
- `multi_instance` is still rejected, with a message saying *not yet*

**End to end**, once a top-down pair exists: two animals through
`CVProcessor`, both in the CSV with per-animal keypoints under the model's own
bodypart names.
