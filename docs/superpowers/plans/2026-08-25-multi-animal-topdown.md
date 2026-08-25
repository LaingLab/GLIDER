# Multi-Animal Top-Down Pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run SLEAP top-down models so GLIDER tracks N animals per frame, with honest flagging of the frames where identity cannot be trusted.

**Architecture:** `PoseBackend.predict()` gains an instance axis — `(N,K,2)` — with single-instance becoming N=1. A new `TopDownPoseBackend` runs a centroid model, finds N peaks, crops around each, runs a centered-instance model per crop, and reuses the existing `decode_sleap_confmaps` verbatim. `_detect_pose` then emits one `Detection` per instance, which lights up the tracker, both CSV writers, `ZoneTracker` and `BehaviorAnalyzer` — all already collection-shaped.

**Tech Stack:** Python 3.11+, numpy, OpenCV, onnxruntime, PyQt6, pytest.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-08-25-multi-animal-topdown-design.md`. Read it before Task 1.
- **Test command:** `QT_QPA_PLATFORM=offscreen uv run --no-sync pytest <paths> -q`
- **Never run `uv sync`** — it prunes the four in-repo editable plugins and breaks collection. If you hit `ModuleNotFoundError` for `glider_harp` / `glider_maimu` / `glider_sleap` / `glider_dlc`, reinstall: `uv pip install -e ./plugins/<name>`.
- **Baseline:** `pytest -q -m "not slow"` is **4475 passed / 3 skipped / 0 failed**. Never finish a task below it.
- **Test output must be pristine** — no new warnings.
- **Lint gates:** `uv run ruff check src tests plugins` and `uv run black --check src tests plugins`.
- **mypy must not regress:** `uv run mypy src 2>&1 | tail -1` — record the count before Task 1 and do not exceed it.
- **Line length:** 100.
- **Commits:** Conventional Commits. **No `Co-Authored-By` trailer, no "Generated with Claude Code" footer.**
- **Never write** "TODO", "TBD", or "for now" in any docstring or comment.
- **Exact names later tasks depend on:** `find_peaks`, `TopDownPoseBackend`, `PoseModelSpec.head_kind`, `CVSettings.pose_peak_threshold`, `CVSettings.pose_peak_min_distance_px`, `CVSettings.identity_min_separation_px`, `identity_flag`.
- **A trap that has bitten three times in this repo:** when inserting a test near an existing one, check whether the anchor function carries a decorator (`@pytest.mark.slow`) on the line above — inserting before the `def` steals it. Verify with `grep -n "pytest.mark.slow" -A1 <file>` after editing.

---

### Task 1: `find_peaks` — multi-peak decode

Spec §5. Pure numpy, no model, no onnxruntime. Nothing else depends on a running model, so this is first and fully testable in isolation.

**Files:**
- Modify: `src/glider/vision/pose/decode.py`
- Test: `tests/unit/vision/pose/test_decode.py`

**Interfaces produced:**
```python
def find_peaks(confmap: np.ndarray, *, threshold: float, min_distance: int) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 2)`` row/col of local maxima, and ``(N,)`` their values.

    Sorted strongest first. Local maxima above ``threshold``, thinned so no
    returned peak is within ``min_distance`` cells of a stronger one.
    """
```

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/vision/pose/test_decode.py`:

```python
# --- multi-peak detection ------------------------------------------------------


def _blob(arr, r, c, peak=1.0, sigma=1.5):
    """Add a small Gaussian blob centred on (r, c)."""
    h, w = arr.shape
    rr, cc = np.ogrid[:h, :w]
    arr += peak * np.exp(-((rr - r) ** 2 + (cc - c) ** 2) / (2 * sigma**2))
    return arr


def test_two_separated_blobs_give_two_peaks():
    from glider.vision.pose.decode import find_peaks

    m = np.zeros((40, 40))
    _blob(m, 10, 10)
    _blob(m, 30, 30)

    rc, vals = find_peaks(m, threshold=0.2, min_distance=5)

    assert rc.shape == (2, 2)
    assert vals.shape == (2,)
    found = {tuple(p) for p in rc.tolist()}
    assert found == {(10, 10), (30, 30)}


def test_blobs_closer_than_min_distance_collapse_to_one():
    """The setting that decides 'two mice' from 'one mouse counted twice'."""
    from glider.vision.pose.decode import find_peaks

    m = np.zeros((40, 40))
    _blob(m, 20, 20, peak=1.0)
    _blob(m, 20, 24, peak=0.8)

    rc, vals = find_peaks(m, threshold=0.2, min_distance=10)

    assert rc.shape == (1, 2)
    assert tuple(rc[0]) == (20, 20)  # the stronger one survives


def test_a_peak_below_threshold_is_not_returned():
    from glider.vision.pose.decode import find_peaks

    m = np.zeros((40, 40))
    _blob(m, 10, 10, peak=1.0)
    _blob(m, 30, 30, peak=0.05)

    rc, _vals = find_peaks(m, threshold=0.2, min_distance=5)

    assert rc.shape == (1, 2)
    assert tuple(rc[0]) == (10, 10)


def test_an_empty_map_returns_no_peaks_not_an_error():
    """An empty arena is a normal frame, not a failure."""
    from glider.vision.pose.decode import find_peaks

    rc, vals = find_peaks(np.zeros((20, 20)), threshold=0.2, min_distance=5)

    assert rc.shape == (0, 2)
    assert vals.shape == (0,)


def test_a_peak_on_the_border_is_found():
    """An animal against the arena wall is exactly the case that must not be lost."""
    from glider.vision.pose.decode import find_peaks

    m = np.zeros((40, 40))
    m[0, 0] = 1.0

    rc, _vals = find_peaks(m, threshold=0.2, min_distance=5)

    assert tuple(rc[0]) == (0, 0)


def test_peaks_come_back_strongest_first():
    from glider.vision.pose.decode import find_peaks

    m = np.zeros((40, 40))
    _blob(m, 10, 10, peak=0.5)
    _blob(m, 30, 30, peak=0.9)

    _rc, vals = find_peaks(m, threshold=0.2, min_distance=5)

    assert list(vals) == sorted(vals, reverse=True)
```

- [ ] **Step 2: Run them, watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision/pose/test_decode.py -q -k "peak or blob"
```
Expected: `ImportError: cannot import name 'find_peaks'`.

- [ ] **Step 3: Implement**

Add to `decode.py`, after `_peaks`:

```python
def find_peaks(
    confmap: np.ndarray, *, threshold: float, min_distance: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 2)`` row/col of local maxima, and ``(N,)`` their values.

    Sorted strongest first, and thinned so no returned peak is within
    ``min_distance`` cells of a stronger one -- one animal produces one broad
    response, and without the thinning its shoulders come back as extra
    animals.

    This is what :func:`_peaks` cannot do: that takes a global argmax per
    channel, which is exactly one instance by construction.
    """
    confmap = np.asarray(confmap, dtype=float)
    if confmap.ndim != 2:
        raise ValueError(f"confmap must be (H, W); got {confmap.shape}")

    rows, cols = np.nonzero(confmap >= threshold)
    if rows.size == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    vals = confmap[rows, cols]
    order = np.argsort(vals)[::-1]
    rows, cols, vals = rows[order], cols[order], vals[order]

    keep_r: list[int] = []
    keep_c: list[int] = []
    keep_v: list[float] = []
    for r, c, v in zip(rows, cols, vals, strict=True):
        # Squared distance: the comparison is the same and the sqrt is not.
        if any(
            (r - kr) ** 2 + (c - kc) ** 2 < min_distance**2
            for kr, kc in zip(keep_r, keep_c, strict=True)
        ):
            continue
        keep_r.append(int(r))
        keep_c.append(int(c))
        keep_v.append(float(v))

    return (
        np.array(list(zip(keep_r, keep_c, strict=True)), dtype=int).reshape(-1, 2),
        np.array(keep_v, dtype=float),
    )
```

- [ ] **Step 4: Run them, watch them pass**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision/pose/test_decode.py -q
```
Expected: all pass.

- [ ] **Step 5: Full suite, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```
```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins
```
```bash
git add src/glider/vision/pose/decode.py tests/unit/vision/pose/test_decode.py
git commit -m "feat(pose): find every peak in a confidence map, not just the strongest

_peaks takes a global argmax per channel, which is exactly one instance by
construction -- the floor under GLIDER's single-animal limit. find_peaks
returns local maxima above a threshold, thinned by a minimum separation so
one animal's broad response does not come back as several."
```

---

### Task 2: The contract gains an instance axis

Spec §3. The breaking change, done before the top-down backend so that backend can return its natural shape.

**Files:**
- Modify: `src/glider/vision/pose/backend.py` (protocol, `OnnxPoseBackend.predict`, `UltralyticsBackend.predict`)
- Modify: `src/glider/analysis/behavior/classify/pose_extract.py`
- Modify: `src/glider/vision/cv_processor.py` (`_detect_pose` — reads instance 0 until Task 5 emits N)
- Modify: `src/glider/gui/panels/live_behavior.py` (`classify_frame` takes instance 0)
- Test: `tests/unit/vision/pose/test_backend.py`

**Interfaces produced:** `PoseBackend.predict(bgr) -> tuple[np.ndarray, np.ndarray]` returning `(N,K,2)` and `(N,K)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/vision/pose/test_backend.py`:

```python
# --- the instance axis ---------------------------------------------------------


def test_onnx_backend_returns_an_instance_axis(tmp_path):
    """Single-instance is N == 1, not a bare (K, 2)."""
    from glider.vision.pose.backend import OnnxPoseBackend

    spec = _spec_for_fit(tmp_path)  # 2 keypoints, NHWC, stride 1

    class _Session:
        def get_inputs(self):
            return [type("I", (), {"name": "input", "shape": [1, 8, 8, 3]})()]

        def run(self, _out, _feed):
            maps = np.zeros((1, 8, 8, 2))
            maps[0, 2, 3, 0] = 1.0
            maps[0, 5, 6, 1] = 1.0
            return [maps]

    b = OnnxPoseBackend(_Session(), spec)
    xy, conf = b.predict(np.zeros((8, 8, 3), np.uint8))

    assert xy.ndim == 3 and xy.shape[0] == 1 and xy.shape[2] == 2
    assert conf.ndim == 2 and conf.shape[0] == 1
    assert xy.shape[1] == conf.shape[1] == 2  # K


def test_ultralytics_backend_keeps_every_instance():
    """YOLO-pose already finds several animals; GLIDER used to keep only the
    first, with a comment saying so. The instance axis is where they go."""
    from glider.vision.pose.backend import UltralyticsBackend

    class _KP:
        xy = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
        conf = np.array([[0.9, 0.8], [0.7, 0.6]])

    class _Result:
        keypoints = _KP()

    class _Yolo:
        def predict(self, *_a, **_k):
            return [_Result()]

    b = UltralyticsBackend(_Yolo(), ["a", "b"])
    xy, conf = b.predict(np.zeros((8, 8, 3), np.uint8))

    assert xy.shape == (2, 2, 2)
    assert conf.shape == (2, 2)


def test_no_instances_is_an_empty_axis_not_an_error():
    from glider.vision.pose.backend import UltralyticsBackend

    class _Yolo:
        def predict(self, *_a, **_k):
            return []

    xy, conf = UltralyticsBackend(_Yolo(), ["a", "b"]).predict(np.zeros((8, 8, 3), np.uint8))

    assert xy.shape == (0, 2, 2)
    assert conf.shape == (0, 2)
```

- [ ] **Step 2: Run them, watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision/pose/test_backend.py -q -k "instance"
```
Expected: shape assertions fail — `xy.ndim` is 2.

- [ ] **Step 3: Update the protocol docstring**

In `backend.py`, the `PoseBackend` protocol and the module docstring:

```python
    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(N, K, 2)`` xy in source-frame pixels, ``(N, K)`` confidence in [0, 1].

        N is the number of instances found. A single-instance model returns
        ``N == 1``; a frame with nothing in it returns ``N == 0``, which is a
        normal answer and not an error. Undetected keypoints within an
        instance are NaN xy with zero confidence, as before.
        """
```

Update the module docstring's "Every backend turns one BGR frame into `(K, 2)`" sentence to match.

- [ ] **Step 4: Add the axis in `OnnxPoseBackend.predict`**

At the end of `predict`, where it currently returns `xy, conf`, wrap:

```python
        # One instance, shaped like many. The whole-frame decode finds exactly
        # one set of peaks by construction; TopDownPoseBackend is what produces
        # a real N.
        return xy[None, ...], conf[None, ...]
```

- [ ] **Step 5: Keep every instance in `UltralyticsBackend.predict`**

Replace the `extract_keypoints` call with a loop over instances. In
`pose_extract.py`, `extract_keypoints` keeps its single-instance contract but
gains an `index` parameter:

```python
def extract_keypoints(result, conf_threshold: float, n_keypoints: int, index: int = 0):
    """Keypoints for one detected instance. ``index`` selects which."""
```
and its internal `kp.xy[0]` becomes `kp.xy[index]` (same for `kp.conf`). Delete
the comment claiming the first instance is the main subject — that is the
assumption being removed.

Then in `UltralyticsBackend.predict`:

```python
        results = self.yolo.predict(bgr, **kwargs)
        result = results[0] if results else None
        k = len(self.keypoint_names)
        n = 0
        if result is not None and getattr(result, "keypoints", None) is not None:
            n = len(result.keypoints.xy)
        if n == 0:
            return np.empty((0, k, 2)), np.empty((0, k))
        pairs = [extract_keypoints(result, self.conf_threshold, k, i) for i in range(n)]
        return (
            np.stack([xy for xy, _ in pairs]),
            np.stack([c for _, c in pairs]),
        )
```

- [ ] **Step 6: Narrow the two consumers explicitly**

`cv_processor.py` `_detect_pose`, immediately after `predict`:

```python
        xy, conf = self._pose_backend.predict(frame)
        if len(xy) == 0:
            return []
        # Task 5 emits one Detection per instance; until then the first one
        # keeps single-animal behaviour identical.
        xy, conf = xy[0], conf[0]
```

`live_behavior.py` `classify_frame`, after its `predict` call:

```python
        xy, conf = self.backend.predict(bgr)
        if len(xy) == 0:
            return None
        # One label per frame, and BehaviorEvent carries no subject field, so
        # this classifier is single-animal by shape. Sub-project D is what
        # gives it an instance axis; see the multi-animal spec, section 9.
        xy, conf = xy[0], conf[0]
```

- [ ] **Step 7: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Existing pose and behaviour tests exercise `predict` heavily. **Any that fail should fail on shape, and the fix is the caller or the test's fake backend — never weakening an assertion.** If one fails for another reason, stop and report it.

- [ ] **Step 8: Lint, mypy, commit**

```bash
uv run ruff check src tests plugins && uv run black --check src tests plugins && uv run mypy src 2>&1 | tail -1
```
```bash
git commit -m "feat(pose): give the backend contract an instance axis

predict() returned (K, 2) with no room for a second animal. It now returns
(N, K, 2) and (N, K), with single-instance as N == 1 and an empty frame as
N == 0 -- one shape everywhere rather than a second protocol and a branch at
every consumer.

It also picks up something already lost: Ultralytics YOLO-pose detects
several animals per frame and pose_extract hard-indexed the first, with a
comment calling it the main subject. Those instances now survive.

CVProcessor and the live behaviour classifier take instance 0 with a comment
naming the sub-project that widens them."
```

---

### Task 3: `head_kind`, and stop rejecting top-down heads

Spec §7. Small and independent; the pairing logic in Task 7 matches on it.

**Files:**
- Modify: `src/glider/vision/pose/spec.py`
- Test: `tests/unit/vision/pose/test_spec.py`

**Interfaces produced:** `PoseModelSpec.head_kind: Literal["single_instance", "centroid", "centered_instance"]`, default `"single_instance"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_centroid_model_is_read_and_labelled(tmp_path):
    """Top-down is two models; the centroid half must be readable on its own
    so the pairing logic has something to match."""
    _write_sleap(
        tmp_path,
        {
            "model": {
                "heads": {
                    "single_instance": None,
                    "centroid": {"anchor_part": "thorax", "output_stride": 4},
                    "centered_instance": None,
                    "multi_instance": None,
                }
            }
        },
    )
    spec = identify_pose_model(tmp_path)
    assert spec.head_kind == "centroid"


def test_a_centered_instance_model_is_read_and_labelled(tmp_path):
    _write_sleap(
        tmp_path,
        {
            "model": {
                "heads": {
                    "centered_instance": {"part_names": ["a", "b"], "output_stride": 4},
                }
            }
        },
    )
    spec = identify_pose_model(tmp_path)
    assert spec.head_kind == "centered_instance"
    assert spec.keypoint_names == ["a", "b"]


def test_a_single_instance_model_still_says_so(tmp_path):
    _write_sleap(
        tmp_path,
        {"model": {"heads": {"single_instance": {"part_names": ["a"], "output_stride": 2}}}},
    )
    assert identify_pose_model(tmp_path).head_kind == "single_instance"


def test_bottom_up_is_still_refused_but_says_not_yet(tmp_path):
    """multi_instance needs part-affinity grouping that does not exist. The
    message should not imply it never will."""
    _write_sleap(
        tmp_path,
        {"model": {"heads": {"multi_instance": {"part_names": ["a", "b"]}}}},
    )
    with pytest.raises(PoseModelError) as excinfo:
        identify_pose_model(tmp_path)
    assert "not yet" in str(excinfo.value).lower()
    assert "found: multi_instance" in str(excinfo.value)
```

- [ ] **Step 2: Run them, watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision/pose/test_spec.py -q -k "centroid or centered or not_yet or still_says"
```

- [ ] **Step 3: Implement**

Add the field to `PoseModelSpec`, beside `kind`:

```python
    #: Which SLEAP head this model carries. Top-down is two models, and this is
    #: how they are told apart -- from the config, not from the folder name.
    #: SLEAP's naming convention is a convention; the config is a promise.
    head_kind: Literal["single_instance", "centroid", "centered_instance"] = "single_instance"
```

Rewrite the head selection in `_from_sleap_config`:

```python
    configured = {k: v for k, v in heads.items() if v is not None}
    if "multi_instance" in configured:
        raise PoseModelError(
            f"{cfg_path} is a bottom-up model (found: multi_instance). GLIDER "
            "does not run bottom-up models yet -- they need part-affinity "
            "grouping, which is not built. Train a single-instance or top-down "
            "model instead."
        )
    for candidate in ("single_instance", "centered_instance", "centroid"):
        if candidate in configured:
            head_kind, head = candidate, configured[candidate]
            break
    else:
        found = ", ".join(sorted(configured)) or "none"
        raise PoseModelError(
            f"{cfg_path} has no pose head GLIDER can run (found: {found}). "
            "GLIDER runs SLEAP single-instance and top-down models."
        )
```

A `centroid` head has no `part_names` — it names an `anchor_part`. Guard the
`part_names` check so it applies only where names are expected:

```python
    names = [str(n) for n in head.get("part_names", [])]
    if head_kind != "centroid" and not names:
        raise PoseModelError(f"{cfg_path} lists no part_names.")
```

and pass `head_kind=head_kind` into the returned `PoseModelSpec`.

`PoseModelSpec` also gains `crop_size: int | None = None`, read in
`_from_sleap_config` from `data.instance_cropping.crop_size`. It is optional
because only a centered-instance model records one; Task 7 fails loudly rather
than guessing when one is needed and missing.

Add a test for it:

```python
def test_a_centered_instance_model_records_its_crop_size(tmp_path):
    """Top-down needs it to size the crop; guessing would misplace every
    keypoint by however far the guess was wrong."""
    _write_sleap(
        tmp_path,
        {
            "model": {"heads": {"centered_instance": {"part_names": ["a"], "output_stride": 4}}},
            "data": {"instance_cropping": {"crop_size": 96}},
        },
    )
    assert identify_pose_model(tmp_path).crop_size == 96
```

- [ ] **Step 4: Run them, then the full suite**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision/pose -q && QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

- [ ] **Step 5: Lint, commit**

```bash
git commit -m "feat(pose): read top-down heads, and label which one a model carries

_from_sleap_config accepted single_instance and rejected everything else, so
neither half of a top-down pair could be read at all. Both are now read and
PoseModelSpec records which -- from the config rather than the folder name,
because SLEAP's naming is a convention and the config is a promise.

Bottom-up is still refused, with a message that says not yet rather than not
supported."
```

---

### Task 4: `TopDownPoseBackend`

Spec §4. The centrepiece. Depends on Tasks 1-3.

**Files:**
- Create: `src/glider/vision/pose/topdown.py`
- Modify: `src/glider/vision/cv_processor.py` (`CVSettings`: two new fields)
- Test: `tests/unit/vision/pose/test_topdown.py` (create)

**Interfaces consumed:** `find_peaks` (T1), the `(N,K,2)` contract (T2), `PoseModelSpec.head_kind` (T3), and the existing `preprocess_frame`, `decode_sleap_confmaps`.

**Interfaces produced:**
```python
class TopDownPoseBackend:
    def __init__(self, centroid_session, centroid_spec, instance_session, instance_spec,
                 *, crop_size: int, peak_threshold: float = 0.2,
                 peak_min_distance_px: int = 40): ...
    keypoint_names: list[str]          # from instance_spec
    def predict(self, bgr) -> tuple[np.ndarray, np.ndarray]: ...   # (N,K,2), (N,K)
    def close(self) -> None: ...
```

New `CVSettings` fields: `pose_peak_threshold: float = 0.2`, `pose_peak_min_distance_px: int = 40`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/vision/pose/test_topdown.py`:

```python
"""Top-down multi-animal pose: centroid model finds them, instance model poses them.

Sessions are stubs returning canned tensors, so the whole class is testable
with no onnxruntime and no model file -- the same seam OnnxPoseBackend uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.pose.spec import PoseModelSpec

CROP = 64


def _spec(names, tmp_path, head_kind="centered_instance"):
    onnx = tmp_path / f"{head_kind}.onnx"
    onnx.write_bytes(b"stub")
    return PoseModelSpec(
        kind="sleap",
        model_path=onnx,
        root=tmp_path,
        keypoint_names=list(names),
        source_label="t",
        head_kind=head_kind,
        input_layout="NHWC",
        output_stride=1.0,
        scale=1.0,
        pad_to_stride=1,
        divide_by_255=True,
    )


class _Session:
    """Returns a canned (1, H, W, C) map, and records what it was fed."""

    def __init__(self, maps):
        self._maps = maps
        self.fed: list[np.ndarray] = []

    def get_inputs(self):
        return [type("I", (), {"name": "input", "shape": [1, None, None, None]})()]

    def run(self, _outputs, feed):
        self.fed.append(next(iter(feed.values())))
        return [self._maps]


def _centroid_map(h, w, anchors):
    m = np.zeros((1, h, w, 1))
    for r, c in anchors:
        m[0, r, c, 0] = 1.0
    return m


def _instance_map(k, size=CROP):
    """One keypoint peak per channel, at a known offset inside the crop."""
    m = np.zeros((1, size, size, k))
    for i in range(k):
        m[0, 10 + i, 20 + i, i] = 1.0
    return m


def _backend(tmp_path, anchors, k=2, frame_hw=(200, 200)):
    from glider.vision.pose.topdown import TopDownPoseBackend

    h, w = frame_hw
    return TopDownPoseBackend(
        centroid_session=_Session(_centroid_map(h, w, anchors)),
        centroid_spec=_spec(["anchor"], tmp_path, head_kind="centroid"),
        instance_session=_Session(_instance_map(k)),
        instance_spec=_spec([f"kp{i}" for i in range(k)], tmp_path),
        crop_size=CROP,
        peak_threshold=0.2,
        peak_min_distance_px=10,
    )


def test_two_anchors_give_two_instances(tmp_path):
    b = _backend(tmp_path, anchors=[(50, 50), (150, 150)])
    xy, conf = b.predict(np.zeros((200, 200, 3), np.uint8))
    assert xy.shape == (2, 2, 2)
    assert conf.shape == (2, 2)


def test_keypoints_land_in_source_coordinates(tmp_path):
    """The instance model sees a crop; its keypoints must come back in frame
    pixels, or every downstream consumer is measuring the wrong arena."""
    b = _backend(tmp_path, anchors=[(100, 100)])
    xy, _conf = b.predict(np.zeros((200, 200, 3), np.uint8))

    # Crop is CROP wide, centred on (100, 100) -> origin at (68, 68).
    # The stub puts keypoint 0 at row 10, col 20 inside the crop.
    assert xy[0, 0, 0] == pytest.approx(68 + 20, abs=1.0)
    assert xy[0, 0, 1] == pytest.approx(68 + 10, abs=1.0)


def test_a_crop_at_the_frame_edge_is_padded_not_shifted(tmp_path):
    """An animal against the wall keeps its true coordinates. A shifted crop
    displaces every keypoint in it by an amount nothing downstream can undo."""
    b = _backend(tmp_path, anchors=[(5, 5)])
    xy, _conf = b.predict(np.zeros((200, 200, 3), np.uint8))

    # Unclipped origin is (5 - 32) = -27, and stays -27 for the mapping.
    assert xy[0, 0, 0] == pytest.approx(-27 + 20, abs=1.0)
    assert xy[0, 0, 1] == pytest.approx(-27 + 10, abs=1.0)


def test_no_anchors_returns_an_empty_instance_axis(tmp_path):
    b = _backend(tmp_path, anchors=[])
    xy, conf = b.predict(np.zeros((200, 200, 3), np.uint8))
    assert xy.shape == (0, 2, 2)
    assert conf.shape == (0, 2)


def test_the_instance_model_is_run_once_per_animal(tmp_path):
    b = _backend(tmp_path, anchors=[(50, 50), (150, 150), (50, 150)])
    b.predict(np.zeros((200, 200, 3), np.uint8))
    assert len(b._instance_session.fed) == 3


def test_keypoint_names_come_from_the_instance_model(tmp_path):
    """The centroid model names an anchor part; the pose names are the other
    model's, and they are what reaches the CSV."""
    b = _backend(tmp_path, anchors=[(50, 50)], k=3)
    assert b.keypoint_names == ["kp0", "kp1", "kp2"]
```

- [ ] **Step 2: Run them, watch them fail**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision/pose/test_topdown.py -q
```
Expected: `ModuleNotFoundError: glider.vision.pose.topdown`.

- [ ] **Step 3: Implement**

Create `src/glider/vision/pose/topdown.py`:

```python
"""Top-down multi-animal pose: find each animal, then pose each one.

Two models, run in sequence. A centroid model produces one confidence map
whose peaks are the animals; a centered-instance model is then run on a crop
around each peak. The per-crop decode is
:func:`~glider.vision.pose.decode.decode_sleap_confmaps`, unchanged -- which is
the whole reason top-down came before bottom-up. Bottom-up would have to
replace that decoder; this reuses it N times.
"""

from __future__ import annotations

import numpy as np

from glider.vision.pose.backend import preprocess_frame
from glider.vision.pose.decode import decode_sleap_confmaps, find_peaks
from glider.vision.pose.spec import PoseModelSpec


class TopDownPoseBackend:
    """A centroid model and a centered-instance model, behind one predict()."""

    def __init__(
        self,
        centroid_session,
        centroid_spec: PoseModelSpec,
        instance_session,
        instance_spec: PoseModelSpec,
        *,
        crop_size: int,
        peak_threshold: float = 0.2,
        peak_min_distance_px: int = 40,
    ) -> None:
        self._centroid_session = centroid_session
        self._centroid_spec = centroid_spec
        self._instance_session = instance_session
        self._instance_spec = instance_spec
        self._crop_size = int(crop_size)
        self._peak_threshold = float(peak_threshold)
        self._peak_min_distance_px = int(peak_min_distance_px)
        # The pose names, not the centroid model's anchor part. These reach the
        # CSV, so taking them from the wrong model mislabels every column.
        self.keypoint_names = list(instance_spec.keypoint_names)

    @property
    def native_keypoint_count(self) -> int:
        return len(self.keypoint_names)

    def _anchors(self, bgr: np.ndarray) -> np.ndarray:
        """``(N, 2)`` xy anchor points in source pixels."""
        tensor, scale = preprocess_frame(bgr, self._centroid_spec)
        name = self._centroid_session.get_inputs()[0].name
        out = np.asarray(self._centroid_session.run(None, {name: tensor})[0])
        if out.ndim == 4:
            out = out[0]
        if self._centroid_spec.input_layout == "NHWC":
            out = np.transpose(out, (2, 0, 1))
        confmap = out[0]  # centroid models emit exactly one channel

        stride = float(self._centroid_spec.output_stride)
        # min_distance is given in source pixels so the number means the same
        # thing at any resolution; the map is in cells.
        min_cells = max(1, int(round(self._peak_min_distance_px * scale / stride)))
        # Integer cells are enough here: an anchor only decides where to put a
        # crop, and the crop is far larger than a cell. Sub-pixel refinement is
        # applied where it changes an answer -- to the keypoints themselves, by
        # decode_sleap_confmaps in _pose_one.
        rc, _vals = find_peaks(
            confmap, threshold=self._peak_threshold, min_distance=min_cells
        )
        if len(rc) == 0:
            return np.empty((0, 2), dtype=float)
        xy = rc[:, ::-1].astype(float) * stride  # (row, col) -> (x, y)
        return xy / scale

    def _crop(self, bgr: np.ndarray, cx: float, cy: float) -> tuple[np.ndarray, int, int]:
        """A ``crop_size`` square around (cx, cy), padded at the frame edge.

        Returns the crop and its **unclipped** origin. The origin may be
        negative, and that is the point: padding rather than sliding the box
        inward keeps ``local + origin == source`` true for every keypoint. A
        shifted crop would displace an animal against the arena wall by an
        amount nothing downstream could undo.
        """
        size = self._crop_size
        x0 = int(round(cx)) - size // 2
        y0 = int(round(cy)) - size // 2
        h, w = bgr.shape[:2]

        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
        crop = np.zeros((size, size, bgr.shape[2]), dtype=bgr.dtype)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = bgr[sy0:sy1, sx0:sx1]
        return crop, x0, y0

    def _pose_one(self, crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor, scale = preprocess_frame(crop, self._instance_spec)
        name = self._instance_session.get_inputs()[0].name
        out = np.asarray(self._instance_session.run(None, {name: tensor})[0])
        if out.ndim == 4:
            out = out[0]
        if self._instance_spec.input_layout == "NHWC":
            out = np.transpose(out, (2, 0, 1))
        xy, conf = decode_sleap_confmaps(
            out,
            stride=float(self._instance_spec.output_stride),
            window=int(self._instance_spec.refine_window),
            apply_sigmoid=bool(self._instance_spec.apply_sigmoid),
        )
        return xy / scale, np.clip(conf, 0.0, 1.0)

    def predict(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = len(self.keypoint_names)
        anchors = self._anchors(bgr)
        if len(anchors) == 0:
            # An empty arena is a normal frame, not a failure.
            return np.empty((0, k, 2)), np.empty((0, k))

        all_xy, all_conf = [], []
        for cx, cy in anchors:
            crop, x0, y0 = self._crop(bgr, cx, cy)
            xy, conf = self._pose_one(crop)
            all_xy.append(xy + np.array([x0, y0], dtype=float))
            all_conf.append(conf)
        return np.stack(all_xy), np.stack(all_conf)

    def close(self) -> None:
        self._centroid_session = None
        self._instance_session = None
```

- [ ] **Step 4: Add the two settings**

In `cv_processor.py`, beside `keypoint_min_confidence` in `CVSettings`:

```python
    # Multi-animal peak finding. Starting points to tune against a real
    # recording, not values with evidence behind them: min_distance especially
    # is the difference between "two mice" and "one mouse counted twice", and
    # the right number depends on arena scale and animal size.
    pose_peak_threshold: float = 0.2
    pose_peak_min_distance_px: int = 40
```

Add both to `to_dict`/`from_dict` beside the existing keypoint settings, matching their style exactly.

- [ ] **Step 5: Run the tests, then the full suite**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision/pose/test_topdown.py -q
```
Expected: 7 passed.

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

- [ ] **Step 6: Lint, commit**

```bash
git commit -m "feat(pose): run SLEAP top-down models, N animals per frame

A centroid model finds the animals, a centered-instance model poses each one
on a crop around it. The per-crop decode is decode_sleap_confmaps unchanged,
which is why top-down came first -- bottom-up would have to replace that
decoder rather than reuse it N times.

Crops are padded at the frame edge, never slid inward: the unclipped origin
is what maps local keypoints back, so an animal against the arena wall keeps
its true coordinates."
```

---

### Task 5: One `Detection` per animal

Spec §6, first half. This is where the five already-collection-shaped subsystems light up.

**Files:**
- Modify: `src/glider/vision/cv_processor.py` (`_detect_pose`)
- Test: `tests/unit/vision/test_cv_processor_pose_backend.py`

**Interfaces consumed:** the `(N,K,2)` contract (T2).

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/vision/test_cv_processor_pose_backend.py`:

```python
def test_two_instances_become_two_detections():
    """The moment multi-animal starts working: ObjectTracker, both CSV
    writers, ZoneTracker and BehaviorAnalyzer are all already N-capable and
    have simply never been given more than one detection."""
    p, backend = _processor_with_backend()
    backend.xy = np.array(
        [
            [[100.0, 100.0], [140.0, 160.0]],
            [[400.0, 300.0], [440.0, 360.0]],
        ]
    )
    backend.conf = np.array([[0.9, 0.9], [0.8, 0.8]])

    dets = p._detect(np.zeros((480, 640, 3), np.uint8))

    assert len(dets) == 2
    assert dets[0].centroid != dets[1].centroid
    assert all(d._keypoints.shape == (2, 3) for d in dets)


def test_each_detection_gets_its_own_bbox():
    p, backend = _processor_with_backend()
    backend.xy = np.array(
        [[[10.0, 10.0], [30.0, 30.0]], [[500.0, 400.0], [520.0, 420.0]]]
    )
    backend.conf = np.ones((2, 2)) * 0.9

    dets = p._detect(np.zeros((480, 640, 3), np.uint8))

    assert dets[0].bbox[0] < dets[1].bbox[0]
    assert dets[0].bbox[1] < dets[1].bbox[1]


def test_an_instance_below_the_keypoint_floor_is_dropped_individually():
    """One animal being poorly seen must not discard the other."""
    p, backend = _processor_with_backend()
    backend.xy = np.array(
        [[[100.0, 100.0], [140.0, 160.0]], [[400.0, 300.0], [440.0, 360.0]]]
    )
    backend.conf = np.array([[0.9, 0.9], [0.05, 0.05]])

    dets = p._detect(np.zeros((480, 640, 3), np.uint8))

    assert len(dets) == 1
    assert dets[0].centroid[0] < 300


def test_no_instances_is_no_detections():
    p, backend = _processor_with_backend()
    backend.xy = np.empty((0, 2, 2))
    backend.conf = np.empty((0, 2))

    assert p._detect(np.zeros((480, 640, 3), np.uint8)) == []


def test_two_detections_become_two_tracked_objects_with_distinct_ids():
    p, backend = _processor_with_backend()
    backend.xy = np.array(
        [[[100.0, 100.0], [140.0, 160.0]], [[400.0, 300.0], [440.0, 360.0]]]
    )
    backend.conf = np.ones((2, 2)) * 0.9

    dets = p._detect(np.zeros((480, 640, 3), np.uint8))
    tracked = p._tracker.update(dets)

    assert len({o.track_id for o in tracked}) == 2
    assert all(o.keypoints is not None for o in tracked)
```

Add the helper at the top of the file if one does not already exist:

```python
def _processor_with_backend():
    """A CVProcessor wired to a fake pose backend returning canned instances."""
    from glider.vision.cv_processor import CVProcessor, CVSettings, DetectionBackend

    class _Fake:
        keypoint_names = ["a", "b"]
        xy = np.empty((0, 2, 2))
        conf = np.empty((0, 2))

        def predict(self, _bgr):
            return self.xy, self.conf

        def close(self):
            pass

    settings = CVSettings(
        backend=DetectionBackend.POSE_MODEL,
        keypoint_names=["a", "b"],
        keypoint_min_confidence=0.3,
    )
    p = CVProcessor(settings)
    p._active_backend = DetectionBackend.POSE_MODEL
    backend = _Fake()
    p._pose_backend = backend
    return p, backend
```

- [ ] **Step 2: Run them, watch them fail**

Expected: `len(dets) == 1` where 2 was asserted — Task 2 left `_detect_pose` reading instance 0.

- [ ] **Step 3: Implement**

Replace `_detect_pose`'s body. The per-instance rules are exactly what the
single-instance version already did; the change is that they run in a loop:

```python
    def _detect_pose(self, frame: np.ndarray) -> list[Detection]:
        """One Detection per instance the pose model found.

        The bbox, confidence and keypoint rules are unchanged -- they now
        apply per animal. Everything downstream (ObjectTracker, both CSV
        writers, ZoneTracker, BehaviorAnalyzer) is already collection-shaped
        and has simply never been handed more than one.
        """
        if self._pose_backend is None:
            return []

        xy_all, conf_all = self._pose_backend.predict(frame)
        xy_all = np.asarray(xy_all, dtype=np.float64)
        conf_all = np.asarray(conf_all, dtype=np.float64)

        detections: list[Detection] = []
        frame_h, frame_w = frame.shape[:2]
        for xy, conf in zip(xy_all, conf_all, strict=True):
            confident = conf >= self._settings.keypoint_min_confidence
            # One poorly-seen animal must not discard the others, so this
            # skips the instance rather than returning early.
            if int(np.count_nonzero(confident)) < 2:
                continue

            kept = xy[confident]
            min_xy, max_xy = kept.min(axis=0), kept.max(axis=0)
            margin = max(
                float(np.max(max_xy - min_xy)) * POSE_BBOX_MARGIN, POSE_BBOX_MIN_MARGIN_PX
            )
            x1 = max(0.0, float(min_xy[0]) - margin)
            y1 = max(0.0, float(min_xy[1]) - margin)
            x2 = min(float(frame_w), float(max_xy[0]) + margin)
            y2 = min(float(frame_h), float(max_xy[1]) + margin)

            det = Detection(
                class_id=0,
                class_name="animal",
                confidence=float(conf[confident].mean()),
                bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
            )
            det._keypoints = np.concatenate([xy, conf[:, None]], axis=1)
            detections.append(det)

        return detections
```

- [ ] **Step 4: Run them, then the full suite**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision -q && QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

- [ ] **Step 5: Lint, commit**

```bash
git commit -m "feat(vision): emit one detection per animal, not per frame

_detect_pose collapsed a whole frame into a single Detection. The bbox,
confidence and keypoint rules are unchanged -- they now run per instance.

That is the whole switch: ObjectTracker, both CSV writers, ZoneTracker and
BehaviorAnalyzer were all built collection-shaped and have simply never been
handed more than one detection.

An instance below the keypoint floor is skipped rather than returning early,
so one poorly-seen animal does not discard the others."
```

---

### Task 6: `identity_flag` — say when identity is a guess

Spec §6, second half. The honest half of the greedy-tracking decision.

**Files:**
- Modify: `src/glider/vision/cv_processor.py` (`TrackedObject`, `ObjectTracker.update`, `CVSettings`)
- Modify: `src/glider/vision/tracking_logger.py` (column)
- Test: `tests/unit/vision/test_identity_flag.py` (create)

**Interfaces produced:** `TrackedObject.identity_flag: str = ""`, `CVSettings.identity_min_separation_px: int = 60`, `IDENTITY_TIE_RATIO = 1.2`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/vision/test_identity_flag.py`:

```python
"""Saying when a track id is a guess.

ObjectTracker is greedy nearest-centroid with no re-identification, so two
animals that touch and separate can swap ids. With one animal that was
invisible; with two it silently corrupts per-animal data, and social assays
are made of animals interacting. This column is what makes shipping the
greedy tracker defensible rather than negligent.
"""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.cv_processor import Detection, ObjectTracker


def _det(x, y, size=20):
    return Detection(
        class_id=0, class_name="animal", confidence=0.9, bbox=(x, y, size, size)
    )


def test_animals_far_apart_are_not_flagged():
    """A column that always fires says nothing."""
    t = ObjectTracker()
    t.update([_det(10, 10), _det(400, 400)])
    tracked = t.update([_det(12, 12), _det(402, 402)])

    assert all(o.identity_flag == "" for o in tracked)


def test_two_tracks_within_the_separation_are_flagged_close():
    t = ObjectTracker(identity_min_separation_px=60)
    t.update([_det(100, 100), _det(400, 400)])
    tracked = t.update([_det(100, 100), _det(130, 100)])

    assert all("close" in o.identity_flag for o in tracked)


def test_a_reacquired_track_is_flagged():
    """After a disappearance the id is a guess, not a continuation."""
    t = ObjectTracker(max_disappeared=5)
    t.update([_det(100, 100)])
    t.update([])  # gone
    tracked = t.update([_det(105, 105)])

    assert any("reacquired" in o.identity_flag for o in tracked)


def test_a_near_tie_in_the_assignment_is_flagged():
    """Two candidates almost equidistant means the assignment was near
    arbitrary, which is exactly when a swap happens."""
    t = ObjectTracker()
    t.update([_det(100, 100)])
    tracked = t.update([_det(120, 100), _det(80, 100)])

    assert any("tie" in o.identity_flag for o in tracked)


def test_several_causes_join_with_a_plus():
    t = ObjectTracker(identity_min_separation_px=100)
    t.update([_det(100, 100)])
    tracked = t.update([_det(120, 100), _det(80, 100)])

    flagged = [o.identity_flag for o in tracked if o.identity_flag]
    assert any("+" in f for f in flagged)


def test_the_flag_reaches_the_tracking_csv(tmp_path):
    from glider.vision.tracking_logger import TrackingDataLogger

    logger = TrackingDataLogger(output_dir=tmp_path)
    logger.start_session("s")
    t = ObjectTracker(identity_min_separation_px=60)
    t.update([_det(100, 100), _det(400, 400)])
    tracked = t.update([_det(100, 100), _det(130, 100)])
    logger.log_frame(1, tracked)
    path = logger.stop_session()

    text = path.read_text()
    assert "identity_flag" in text.splitlines()[0] or any(
        "identity_flag" in line for line in text.splitlines()[:20]
    )
    assert "close" in text
```

- [ ] **Step 2: Run them, watch them fail**

Expected: `TypeError: ObjectTracker.__init__() got an unexpected keyword argument 'identity_min_separation_px'`.

- [ ] **Step 3: Implement**

`TrackedObject` gains the field:

```python
    #: Why this frame's id assignment might be wrong: "", "close",
    #: "reacquired", "tie", or several joined with "+". Empty means the
    #: assignment was unambiguous.
    identity_flag: str = ""
```

Include it in `to_dict` beside the other fields.

`ObjectTracker.__init__` gains `identity_min_separation_px: int = 60`, stored on
`self._identity_min_separation`. Add the module constant:

```python
#: Best and second-best candidate within this ratio counts as a tie -- an
#: assignment that near-arbitrary is exactly when a swap happens.
IDENTITY_TIE_RATIO = 1.2

#: Ordered so a multi-cause flag is deterministic and therefore testable.
_IDENTITY_CAUSES = ("close", "reacquired", "tie")
```

Add a helper beside `ObjectTracker`:

```python
def _join_causes(causes: set[str]) -> str:
    """Causes as a stable string: 'close+tie', never 'tie+close'."""
    return "+".join(c for c in _IDENTITY_CAUSES if c in causes)
```

In `update`, collect causes per track_id and apply them at the end. The three
insertions:

**(a)** At the top of the matching path, before the `for row, col` loop:

```python
        causes: dict[int, set[str]] = {}
        for obj in self._objects.values():
            obj.identity_flag = ""
```

**(b)** Inside that loop, right after `track_id = object_ids[row]`, flag a
near-arbitrary assignment:

```python
            # Two candidates almost equidistant means this pairing was close to
            # a coin flip, which is exactly the situation a swap comes out of.
            row_distances = np.sort(distance_matrix[row])
            if (
                len(row_distances) > 1
                and row_distances[0] > 0
                and row_distances[1] <= row_distances[0] * IDENTITY_TIE_RATIO
            ):
                causes.setdefault(track_id, set()).add("tie")
```

**(c)** Replace the `unused_cols` registration loop so a track that appears in
the same frame another was lost is marked a guess, and add the proximity pass:

```python
        # A brand-new animal walking into frame is not a guess; an id handed out
        # in the same frame another disappeared may well be the same animal.
        lost_this_frame = len(unused_rows) > 0
        for col in unused_cols:
            new_id = self._register(detections[col])
            if lost_this_frame and new_id is not None:
                causes.setdefault(new_id, set()).add("reacquired")

        # Close enough that a swap is physically possible.
        survivors = list(self._objects.values())
        for i, a in enumerate(survivors):
            for b in survivors[i + 1 :]:
                dx = a.centroid[0] - b.centroid[0]
                dy = a.centroid[1] - b.centroid[1]
                if dx * dx + dy * dy < self._identity_min_separation**2:
                    causes.setdefault(a.track_id, set()).add("close")
                    causes.setdefault(b.track_id, set()).add("close")

        for track_id, found in causes.items():
            if track_id in self._objects:
                self._objects[track_id].identity_flag = _join_causes(found)
```

`_register` currently returns `None`; change it to return the new `track_id` so
(c) can mark it. Check no other caller depends on the old return.

`CVSettings` gains `identity_min_separation_px: int = 60` (with `to_dict` /
`from_dict`), and `CVProcessor` passes it when constructing `ObjectTracker`.

`tracking_logger.py`: add `identity_flag` to the tracking CSV's column list and
write `obj.identity_flag` in `log_frame`. Update the class docstring's column
listing to match — it enumerates them explicitly.

- [ ] **Step 4: Run them, then the full suite**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest tests/unit/vision -q && QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

Existing tracking-CSV tests may assert an exact header. **Update them to include
the new column — do not remove the assertion.**

- [ ] **Step 5: Lint, commit**

```bash
git commit -m "feat(vision): flag the frames where a track id is a guess

ObjectTracker is greedy nearest-centroid with no re-identification, so two
animals that touch and separate can swap ids. With one animal that was
invisible. With two it silently corrupts per-animal data, and social assays
are made of animals interacting.

The tracking CSV now carries identity_flag: close (another track within the
separation), reacquired (the id is a continuation only by assumption), tie
(the assignment was near arbitrary), joined with + when several apply.

A number you can distrust on purpose beats one you cannot."
```

---

### Task 7: Finding and selecting the model pair

Spec §7. The user-facing half.

**Files:**
- Create: `src/glider/vision/pose/pairing.py`
- Modify: `src/glider/vision/pose/backend.py` (`load_pose_backend` builds a top-down backend)
- Modify: `src/glider/gui/panels/camera_panel.py` (`_apply_pose_model`)
- Modify: `src/glider/gui/dialogs/camera_settings_dialog.py` (expose the three new settings)
- Test: `tests/unit/vision/pose/test_pairing.py` (create), `tests/unit/gui/test_camera_panel_pose_tracking.py`

**Interfaces produced:**
```python
def find_partner(spec: PoseModelSpec) -> tuple[Path | None, list[Path]]:
    """The sibling folder completing a top-down pair, and all candidates seen.

    Returns ``(None, candidates)`` when there is no single answer -- zero
    candidates or several. The caller asks; this never guesses.
    """
```

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/vision/pose/test_pairing.py`:

```python
"""Finding the other half of a top-down pair.

SLEAP writes the two models as sibling folders from two training runs. Given
one, GLIDER looks for the other -- and refuses to guess when the answer is not
unique, the same rule BLEDevice._find_by_service applies to six identical
stimulators, for the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glider.vision.pose.pairing import find_partner
from glider.vision.pose.spec import identify_pose_model


def _model(root: Path, name: str, head: str, names=("a", "b")):
    d = root / name
    d.mkdir(parents=True)
    heads = {"single_instance": None, "centroid": None, "centered_instance": None}
    heads[head] = (
        {"anchor_part": "thorax", "output_stride": 4}
        if head == "centroid"
        else {"part_names": list(names), "output_stride": 4}
    )
    (d / "training_config.json").write_text(
        json.dumps({"model": {"heads": heads, "backbone": {}}, "data": {"preprocessing": {}}})
    )
    (d / "model.onnx").write_bytes(b"stub")
    return d


def test_a_centroid_finds_its_centered_instance_sibling(tmp_path):
    _model(tmp_path, "exp.UNet.centroid", "centroid")
    want = _model(tmp_path, "exp.UNet.centered_instance", "centered_instance")

    partner, candidates = find_partner(identify_pose_model(tmp_path / "exp.UNet.centroid"))

    assert partner == want
    assert candidates == [want]


def test_a_centered_instance_finds_its_centroid_sibling(tmp_path):
    want = _model(tmp_path, "exp.UNet.centroid", "centroid")
    _model(tmp_path, "exp.UNet.centered_instance", "centered_instance")

    partner, _ = find_partner(identify_pose_model(tmp_path / "exp.UNet.centered_instance"))

    assert partner == want


def test_no_sibling_returns_none_rather_than_guessing(tmp_path):
    _model(tmp_path, "exp.UNet.centroid", "centroid")

    partner, candidates = find_partner(identify_pose_model(tmp_path / "exp.UNet.centroid"))

    assert partner is None
    assert candidates == []


def test_two_candidates_refuse_to_pick(tmp_path):
    """Two training runs in one folder. Choosing silently would pair the wrong
    halves and produce plausible, wrong keypoints."""
    _model(tmp_path, "exp.UNet.centroid", "centroid")
    a = _model(tmp_path, "expA.UNet.centered_instance", "centered_instance")
    b = _model(tmp_path, "expB.UNet.centered_instance", "centered_instance")

    partner, candidates = find_partner(identify_pose_model(tmp_path / "exp.UNet.centroid"))

    assert partner is None
    assert sorted(candidates) == sorted([a, b])


def test_a_single_instance_model_has_no_partner_to_find(tmp_path):
    _model(tmp_path, "exp.UNet.single_instance", "single_instance")

    partner, candidates = find_partner(
        identify_pose_model(tmp_path / "exp.UNet.single_instance")
    )

    assert partner is None
    assert candidates == []
```

- [ ] **Step 2: Run them, watch them fail**

Expected: `ModuleNotFoundError: glider.vision.pose.pairing`.

- [ ] **Step 3: Implement `pairing.py`**

```python
"""Pairing the two halves of a SLEAP top-down model.

Top-down is two trained models from two training runs, which SLEAP writes as
sibling folders. Given one, this finds the other -- by reading each candidate's
``head_kind`` rather than by parsing folder names, because SLEAP's naming is a
convention it happens to follow and the config is what it promises.

It refuses to answer when the answer is not unique. Two training runs in one
directory would otherwise pair the wrong halves silently and produce keypoints
that look entirely plausible.
"""

from __future__ import annotations

from pathlib import Path

from glider.vision.pose.spec import PoseModelError, PoseModelSpec, identify_pose_model

_PARTNER_OF = {"centroid": "centered_instance", "centered_instance": "centroid"}


def find_partner(spec: PoseModelSpec) -> tuple[Path | None, list[Path]]:
    """The sibling folder completing *spec*'s top-down pair, and all candidates.

    ``(None, [])`` when *spec* is single-instance (nothing to pair) or nothing
    matched; ``(None, candidates)`` when several matched. The caller asks.
    """
    wanted = _PARTNER_OF.get(spec.head_kind)
    if wanted is None:
        return None, []

    parent = Path(spec.root).parent
    candidates: list[Path] = []
    for sibling in sorted(p for p in parent.iterdir() if p.is_dir()):
        if sibling == Path(spec.root):
            continue
        try:
            if identify_pose_model(sibling).head_kind == wanted:
                candidates.append(sibling)
        except PoseModelError:
            continue  # not a pose model; not our business

    return (candidates[0] if len(candidates) == 1 else None), candidates
```

- [ ] **Step 4: Teach `load_pose_backend` to build a top-down backend**

In `backend.py`, after the spec is resolved:

```python
    if spec.head_kind in ("centroid", "centered_instance"):
        raise PoseModelError(
            f"{Path(spec.root).name} is one half of a top-down model "
            f"({spec.head_kind}). Both halves are needed; select the model in "
            "the Camera panel, which pairs them."
        )
```

and add a sibling loader the panel calls once it has both:

```python
def load_topdown_backend(
    centroid_spec: PoseModelSpec,
    instance_spec: PoseModelSpec,
    *,
    peak_threshold: float = 0.2,
    peak_min_distance_px: int = 40,
):
    """Build a TopDownPoseBackend from an already-paired centroid + instance."""
    from glider.vision.pose.topdown import TopDownPoseBackend

    ensure_onnxruntime()
    crop = int(instance_spec.crop_size or 0)
    if crop <= 0:
        raise PoseModelError(
            f"{Path(instance_spec.root).name} does not record a crop size, which "
            "a centered-instance model needs. Re-export it from SLEAP."
        )
    return TopDownPoseBackend(
        centroid_session=_make_session(centroid_spec),
        centroid_spec=centroid_spec,
        instance_session=_make_session(instance_spec),
        instance_spec=instance_spec,
        crop_size=crop,
        peak_threshold=peak_threshold,
        peak_min_distance_px=peak_min_distance_px,
    )
```

- [ ] **Step 5: Wire the Camera panel**

In `_apply_pose_model`, after `identify_pose_model` succeeds and before the
tracking wiring: when `spec.head_kind` is `centroid` or `centered_instance`,
call `find_partner`. Then:

- **one partner** — show a `QMessageBox.information` naming both folders and
  what each is, then proceed with the pair;
- **no partner** — open a folder picker for the other half, titled with which
  head is missing. Cancel means no model applied;
- **several candidates** — `QMessageBox.warning` listing them, then the same
  folder picker. Never auto-pick.

Store the pair on the panel and pass both to `load_topdown_backend`. Point
`CVSettings.model_path` at the **centered-instance** folder (the one carrying
the keypoint names) and set `keypoint_names` from its spec.

- [ ] **Step 6: Expose the three settings in the dialog**

Add `pose_peak_threshold`, `pose_peak_min_distance_px` and
`identity_min_separation_px` to `camera_settings_dialog.py`, in the same group
as `keypoint_min_confidence` and following its widget style. Give each a
tooltip saying what a wrong value looks like — for `min_distance`, "too small
counts one animal twice; too large merges two into one."

- [ ] **Step 7: Run everything**

```bash
QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"
```

- [ ] **Step 8: Lint, mypy, commit**

```bash
git commit -m "feat(gui): pair the two halves of a top-down model, or ask

Top-down is two trained models. Selecting either half now looks for the other
beside it and shows what it found before using it -- matching on the config's
head_kind rather than on SLEAP's folder-naming convention.

It refuses to guess. No sibling asks for the folder; several candidates lists
them and asks. Two training runs in one directory would otherwise pair the
wrong halves silently and produce keypoints that look entirely plausible."
```

---

### Task 8: Document it

**Files:** `docs-site/reference/`, `CHANGELOG.md`

- [ ] **Step 1:** Write against what shipped — verify each claim in the code first. Match the surrounding style; read the neighbouring sections before writing.

Cover: that GLIDER now runs SLEAP top-down models and tracks several animals; that selecting either half finds the other; the two peak settings and what a wrong value looks like; the `identity_flag` column, its four values, and **why it exists** — an operator who does not know identity can swap will trust the CSV further than they should.

State plainly what A does not do: no `M001` identity binding, no per-animal experiment triggering, no multi-animal offline analysis, no bottom-up.

- [ ] **Step 2:** `uv run mkdocs build --strict`, then commit.

---

## Verification checklist

- [ ] `QT_QPA_PLATFORM=offscreen uv run --no-sync pytest -q -m "not slow"` — 0 failed, total above 4475.
- [ ] `uv run ruff check src tests plugins` and `uv run black --check src tests plugins` clean.
- [ ] `uv run mypy src 2>&1 | tail -1` — not above the count recorded before Task 1.
- [ ] `git log origin/main..HEAD --format='%B' | grep -c "Co-Authored-By\|Generated with Claude"` prints `0`.
- [ ] No test file lost an assertion: `git diff origin/main..HEAD -- tests/ | grep "^-.*assert"` shows only lines that moved or were genuinely superseded.
- [ ] Manual, once a top-down pair exists: two animals tracked through a real clip, both in the tracking CSV with distinct `object_id`s, both in the keypoints CSV under the model's own bodypart names, and `identity_flag` non-empty during a crossing and empty when they are apart.
