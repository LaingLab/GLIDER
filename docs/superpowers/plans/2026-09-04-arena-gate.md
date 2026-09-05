# Arena-Gated Pose Tracking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the drawn arena perimeter to reject pose detections that left the arena, make drawing one mandatory before tracking, and record what was gated so scoring can never mix gated data with ungated thresholds.

**Architecture:** A new Qt-free `vision/arena_gate.py` maps keypoints into arena centimetres through the existing homography and applies two independent fraction tests per frame. It is called from `run_batch` during inference, from a new post-hoc pass over already-tracked CSVs, and its report is written into the `.meta.json` sidecar as provenance that `classify/__init__.py` checks against the cohort threshold file before scoring.

**Tech Stack:** Python 3.11–3.13, numpy, OpenCV, PyQt6, pytest / pytest-qt.

**Spec:** `docs/superpowers/specs/2026-09-04-arena-gate-design.md` — read it before starting. It carries the *why* for every non-obvious choice here, and several of them look wrong without it.

---

## Before you start

**Working directory:** `C:/Users/bradh/glider/.worktrees/arena-gate` (branch `feat/arena-gate-drawn`).

**Running tests.** This is a git worktree, so `PYTHONPATH=src` is mandatory — without it pytest imports `glider` from the *main* checkout and you will test code you did not write. There is no venv in the worktree; use the main checkout's:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src "C:/Users/bradh/glider/.venv/Scripts/python.exe" -m pytest tests/unit/vision -q
```

Every `Run:` line below is shorthand for that full invocation. Baseline before you touch anything, verified: **1354 passed, 3 skipped** across `tests/unit/vision` and `tests/unit/analysis/behavior`.

**Commits.** Commit steps follow TDD convention, but this repo's owner prefers to authorize commits explicitly. Ask before the first one, then follow whatever they say. Never add `Co-Authored-By` or any attribution trailer.

**Two tasks fix pre-existing bugs**, not new features (Task 1 and Task 14). Their tests must fail on `main` for the reason stated — that is how you know you reproduced the bug rather than encoded current behaviour.

---

## Test fixtures: use what exists

**Do not invent helpers.** These already exist and every GUI task below uses them:

| Where | Provides |
|---|---|
| `tests/unit/gui/pose_batch/test_window_arena.py` | `window` fixture (a built `PoseBatchWindow`), `_video(tmp_path, name=...)`, `_arena(corners=TRAPEZOID)`, `TRAPEZOID` |
| `tests/unit/gui/pose_batch/test_calibration_table_arena.py` | Calibration-table cases; extend this rather than starting a file |
| `tests/unit/vision/pose/conftest.py` | `synthetic_pose`, `gappy_pose`, `kpt_names` (5 keypoints) |
| `tests/unit/vision/test_arena.py` | `SQUARE` (a real 400×400 px square in a 640×480 frame), `TRAPEZOID` |

Chunk 1's GUI tests go in the **existing** `tests/unit/gui/pose_batch/test_window_arena.py`. Do not create `tests/unit/gui/test_pose_batch_arena.py`.

**Two helpers you must add**, both to `test_window_arena.py`, because the tests below need setup the existing fixture does not do:

```python
def _ready_window(window, tmp_path, *, count=1, with_csv=False):
    """A window whose only remaining Run blocker is calibration.

    `_validate` (window.py:1074-1092) checks model path, keypoint names, the
    model's keypoint count and the video list *before* it reaches calibration,
    so a test that only sets videos never exercises the branch it means to.
    """
    videos = [_video(tmp_path, f"t{i}_d1.mp4") for i in range(count)]
    window._videos = [v.resolve() for v in videos]
    window._model_path = tmp_path / "exp-7.pt"
    window._names_field.setText(",".join(f"kp{i}" for i in range(7)))
    window._meta = SimpleNamespace(n_keypoints=7)
    window._cal_table.set_videos(window._videos)   # else selected_videos() == []
    if with_csv:
        for v in window._videos:
            (v.parent / f"{v.stem}DLC_exp-7.csv").write_text("x")
    return videos[0] if count == 1 else videos


def _line_calibration():
    """A CameraCalibration with a drawn line, i.e. what satisfies Run today."""
    from glider.vision.calibration import CameraCalibration

    return CameraCalibration(
        point1=(0.2, 0.5), point2=(0.8, 0.5), known_distance_mm=300.0,
        calibration_width=640, calibration_height=480,
    )
```

Check `CameraCalibration`'s real field names before using `_line_calibration` — mirror whatever `tests/unit/vision/test_calibration.py` constructs.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/glider/vision/arena_gate.py` | **New.** `ArenaGateSettings`, `GateReport`, `inside_fraction`, `gate_to_arena`, `gate_pose_csv`. All geometry and all blanking. Qt-free. |
| `src/glider/vision/calibration_set.py` | Confirmed-arena state and `missing_arenas`. |
| `src/glider/vision/pose/dlc.py` | `write_pose_meta` emits the `arena_gate` block; `NOT_POSE_SUFFIXES` moves here as the shared exclusion list. |
| `src/glider/vision/pose/batch.py` | `run_batch` gains `arenas` / `gate`. |
| `src/glider/vision/pose/core.py` | Arena-aware candidate re-ranking (YOLO branch only). |
| `src/glider/gui/pose_batch/arena_actions.py` | **New.** Arena copy/confirm/re-gate actions, lifted out of `window.py`. |
| `src/glider/gui/pose_batch/regate_worker.py` | **New.** QThread worker for the post-hoc pass. |
| `src/glider/gui/pose_batch/window.py` | Master round trip, Run gate, buttons wired to the two new modules. |
| `src/glider/gui/behavior/window.py` | `_unique_pose_csvs` consumes the shared exclusion list. |
| `src/glider/analysis/behavior/cohort_speed.py` | Cohort-level gate block; refuses a mixed pool. |
| `src/glider/analysis/behavior/classify/__init__.py` | Carries `gate_provenance` and enforces the match before the batch/stream fork. |

`window.py` is already 1233 lines, and `calibration_table.py`'s docstring says it was split out precisely to stop it growing unwieldy. This plan adds three actions and two buttons, so **`arena_actions.py` and `regate_worker.py` are new files rather than more methods on the window** — same reasoning, applied again.

`arena_gate.py` deliberately does **not** live in `vision/pose/filtering.py`. Filtering is per-keypoint and temporal; this is per-frame and geometric, it needs an `ArenaCalibration`, and folding it in would make `filtering.py` depend on a calibration stack nothing else in it touches.

---

## Chunk 1: Master-file round trip and the arena mandate

Spec section 6. No dependency on any other chunk, and it fixes a live data-loss bug, so it lands first.

### Task 1: Load arenas back from the master file

`_load_master` applies `loaded.entries` and silently drops `loaded.arenas`. Arenas are written but never read, so every arena drawn today vanishes when the tool reopens. Everything else in this chunk depends on the round trip closing.

**Files:**
- Modify: `src/glider/gui/pose_batch/window.py:905`
- Test: `tests/unit/gui/pose_batch/test_window_arena.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_master_applies_arenas(window, tmp_path):
    """Regression: _load_master applied entries and dropped arenas."""
    from glider.vision.calibration_set import CalibrationSet

    video = _video(tmp_path)
    master = tmp_path / "master.json"
    seed = CalibrationSet()
    seed.set_arena(video, _arena())
    seed.save(master)

    window._videos = [video.resolve()]
    window._load_master(master)

    assert window._calibrations.get_arena(video) is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/unit/gui/pose_batch/test_window_arena.py::test_load_master_applies_arenas -v`
Expected: FAIL — `assert None is not None`.

- [ ] **Step 3: Fix it**

In `_load_master`, alongside the existing `entries` update:

```python
        self._calibrations.entries.update(loaded.entries)
        self._calibrations.arenas.update(loaded.arenas)
        self._log.appendPlainText(
            f"Loaded calibration for {len(loaded.entries)} video(s) "
            f"and {len(loaded.arenas)} arena(s) from {path.name}."
        )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/unit/gui/pose_batch/test_window_arena.py -v`
Expected: PASS.

- [ ] **Step 5: Commit** (ask first — see *Before you start*)

```bash
git add src/glider/gui/pose_batch/window.py tests/unit/gui/pose_batch/test_window_arena.py
git commit -m "fix(pose-batch): load arenas back from the master calibration file"
```

---

### Task 2: Confirmed-arena state on CalibrationSet

Copy-provenance is a workflow concern, so it lives on `CalibrationSet` and not on `ArenaCalibration`, which stays pure geometry a notebook can build from four numbers.

**Files:**
- Modify: `src/glider/vision/calibration_set.py`
- Test: `tests/unit/vision/test_calibration_set_arena.py` (existing; it defines `TRAPEZOID`)

- [ ] **Step 1: Write the failing tests**

```python
def _set_with_arena(tmp_path, *, confirmed=True):
    video = tmp_path / "s1.mp4"
    video.write_bytes(b"")
    cal_set = CalibrationSet()
    cal_set.set_arena(video, ArenaCalibration(corners=TRAPEZOID), confirmed=confirmed)
    return cal_set, video


class TestConfirmedState:
    def test_a_drawn_arena_is_confirmed(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path)
        assert cal_set.is_arena_confirmed(video)

    def test_an_unconfirmed_arena_counts_as_missing(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        assert cal_set.get_arena(video) is not None
        assert cal_set.missing_arenas([video]) == [video]

    def test_confirming_clears_it(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        cal_set.set_arena(video, cal_set.get_arena(video), confirmed=True)
        assert cal_set.missing_arenas([video]) == []

    def test_a_degenerate_arena_counts_as_missing(self, tmp_path):
        video = tmp_path / "s1.mp4"
        cal_set = CalibrationSet()
        cal_set.set_arena(video, ArenaCalibration(corners=[(0.5, 0.5)] * 4))
        assert cal_set.missing_arenas([video]) == [video]

    def test_discarding_clears_the_unconfirmed_flag(self, tmp_path):
        """A stale flag must not outlive the arena it described.

        Written against a direct ``arenas`` write rather than ``set_arena``,
        because that is the path that actually breaks: ``_load_master`` does
        ``arenas.update(loaded.arenas)``, bypassing ``set_arena`` and its
        flag-clearing, so a leftover flag would mark a freshly loaded arena
        unconfirmed and block Run forever. Going through ``set_arena`` here
        would clear the flag on its own and the test would pass even with the
        cleanup removed.
        """
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        cal_set.discard_arena(video)
        cal_set.arenas.update({cal_set._key(video): ArenaCalibration(corners=TRAPEZOID)})
        assert cal_set.is_arena_confirmed(video)

    def test_subset_carries_confirmed_state(self, tmp_path):
        cal_set, video = _set_with_arena(tmp_path, confirmed=False)
        assert not cal_set.subset([video]).is_arena_confirmed(video)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/vision/test_calibration_set_arena.py::TestConfirmedState -v`
Expected: FAIL — `TypeError: set_arena() got an unexpected keyword argument 'confirmed'`.

- [ ] **Step 3: Implement**

```python
    #: Arenas stamped by a copy and not yet checked against their own video.
    #: A copied arena that does not fit shows no residual warning -- residuals
    #: are computed from the corners alone -- so it must not satisfy the Run
    #: gate until an operator has seen the overlay on that video's floor.
    _unconfirmed: set[Path] = field(default_factory=set)

    def set_arena(self, video, arena, *, confirmed: bool = True) -> None:
        key = self._key(video)
        self.arenas[key] = arena
        if confirmed:
            self._unconfirmed.discard(key)
        else:
            self._unconfirmed.add(key)

    def is_arena_confirmed(self, video) -> bool:
        return self._key(video) not in self._unconfirmed

    def missing_arenas(self, videos: Sequence[Path]) -> list[Path]:
        """Videos still needing a usable, confirmed arena, in the order given.

        Parallel to :meth:`missing`, which asks the weaker question "is there a
        scale". An arena that will not fit a homography is ignored the same way
        ``px_per_mm`` ignores it, but here that makes the video missing rather
        than merely falling back to a line.
        """
        out = []
        for video in videos:
            arena = self.get_arena(video)
            if arena is None or not self.is_arena_confirmed(video):
                out.append(video)
                continue
            try:
                arena.homography()
            except DegenerateArenaError:
                out.append(video)
        return out
```

Add `self._unconfirmed.discard(key)` to `discard_arena`, and carry the flag in `subset()`:

```python
            arena = self.arenas.get(key)
            if arena is not None:
                picked.arenas[key] = arena
                if key in self._unconfirmed:
                    picked._unconfirmed.add(key)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/vision/test_calibration_set_arena.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/glider/vision/calibration_set.py tests/unit/vision/test_calibration_set_arena.py
git commit -m "feat(calibration): track whether an arena has been confirmed for its video"
```

---

### Task 3: Serialize `arena_confirmed` without moving the schema

Written **only when unconfirmed**, so a normal master file stays byte-identical to what today's build produces and `schema_version` does not move — the same precedent the `arena` key itself set.

**Files:**
- Modify: `src/glider/vision/calibration_set.py` (`to_dict`, `load`)
- Test: `tests/unit/vision/test_calibration_set_arena.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_confirmed_arenas_write_no_extra_key(tmp_path):
    """A normal file must stay byte-identical to what earlier builds wrote."""
    cal_set, _ = _set_with_arena(tmp_path)
    assert "arena_confirmed" not in cal_set.to_dict()["videos"][0]


def test_unconfirmed_arenas_round_trip(tmp_path):
    cal_set, video = _set_with_arena(tmp_path, confirmed=False)
    master = tmp_path / "m.json"
    cal_set.save(master)
    assert not CalibrationSet.load(master, known_videos=[video]).is_arena_confirmed(video)


def test_a_file_without_the_key_loads_as_confirmed(tmp_path):
    """Every master file written before this change. Absent means drawn."""
    cal_set, video = _set_with_arena(tmp_path)
    master = tmp_path / "m.json"
    cal_set.save(master)
    assert CalibrationSet.load(master, known_videos=[video]).is_arena_confirmed(video)
```

- [ ] **Step 2: Run to verify the round trip fails**

Run: `pytest tests/unit/vision/test_calibration_set_arena.py -k confirmed -v`
Expected: `test_unconfirmed_arenas_round_trip` FAILS (it loads as confirmed). The other two pass already; they are regression guards.

- [ ] **Step 3: Implement**

In `to_dict`, inside the `if arena is not None:` branch:

```python
                entry["arena"] = arena.to_dict()
                # Only when unconfirmed: absent means drawn-and-checked, so
                # files written by earlier builds keep their meaning and files
                # written by this one stay diffable against them.
                if video in self._unconfirmed:
                    entry["arena_confirmed"] = False
```

In `load`, replacing the existing arena assignment:

```python
            if arena is not None:
                cal_set.arenas[key] = arena
                if entry.get("arena_confirmed") is False:
                    cal_set._unconfirmed.add(key)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/vision/test_calibration_set_arena.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/glider/vision/calibration_set.py tests/unit/vision/test_calibration_set_arena.py
git commit -m "feat(calibration): persist unconfirmed arenas without moving the schema"
```

---

### Task 4: Require an arena to Run

Replaces the line check rather than joining it. A usable arena yields a scale by construction via `px_per_cm_centre`, so requiring both would block a video whose arena is fine but which never had a line drawn.

**Files:**
- Modify: `src/glider/gui/pose_batch/window.py:1090-1105` (inside `_validate`)
- Test: `tests/unit/gui/pose_batch/test_window_arena.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_is_blocked_by_a_line_only_calibration(window, tmp_path):
    video = _ready_window(window, tmp_path)
    window._calibrations.set(video, _line_calibration())
    window._validate()
    assert not window._run_button.isEnabled()
    assert "arena" in window._run_button.toolTip().lower()


def test_run_is_blocked_by_an_unconfirmed_arena(window, tmp_path):
    video = _ready_window(window, tmp_path)
    window._calibrations.set_arena(video, _arena(), confirmed=False)
    window._validate()
    assert not window._run_button.isEnabled()


def test_a_confirmed_arena_alone_enables_run(window, tmp_path):
    """No line drawn at all. The arena carries the scale."""
    video = _ready_window(window, tmp_path)
    window._calibrations.set_arena(video, _arena())
    window._validate()
    assert window._run_button.isEnabled()


def test_the_badge_counts_arenas(window, tmp_path):
    videos = _ready_window(window, tmp_path, count=2)
    window._calibrations.set_arena(videos[0], _arena())
    window._validate()
    assert "1 / 2 arenas drawn" in window._calibration_card.badge_text()
```

Check the real accessor for the badge before writing that last assertion; `Card.set_badge` is the setter, so read back whatever it stores.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/gui/pose_batch/test_window_arena.py -k run_ -v`
Expected: the first two FAIL — a drawn line enables Run today.

- [ ] **Step 3: Implement**

```python
        else:
            # An arena, not just a line: only the perimeter can place a zone or
            # gate a detection, and on this cohort the line ran systematically
            # 2.34% high. A usable arena carries the scale by construction, so
            # this replaces the old line check rather than adding to it.
            no_arena = self._calibrations.missing_arenas(self._videos)
            if no_arena:
                problem = f"{len(no_arena)} video(s) still need an arena drawn."
```

And the badge:

```python
        if self._videos:
            missing = len(self._calibrations.missing_arenas(self._videos))
            done = len(self._videos) - missing
            self._calibration_card.set_badge(f"{done} / {len(self._videos)} arenas drawn")
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/unit/gui/pose_batch/test_window_arena.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/pose_batch/window.py tests/unit/gui/pose_batch/test_window_arena.py
git commit -m "feat(pose-batch): require a drawn arena before a batch can run"
```

---

### Task 5: Clear removes the arena too

`_clear_selected_calibrations` calls only `discard`. With arenas gating Run, Clear would appear to do nothing.

**Files:**
- Modify: `src/glider/gui/pose_batch/window.py:822`
- Test: `tests/unit/gui/pose_batch/test_window_arena.py`

- [ ] **Step 1: Write the failing test**

```python
def test_clear_removes_the_arena_as_well_as_the_line(window, tmp_path):
    video = _ready_window(window, tmp_path)
    window._calibrations.set_arena(video, _arena())
    window._cal_table.selectRow(0)
    assert window._cal_table.selected_videos()  # guard: the row really is selected
    window._clear_selected_calibrations()
    assert window._calibrations.get_arena(video) is None
```

The `selected_videos()` guard matters: without `set_videos` (which `_ready_window` does) the selection is empty and the test would pass for the wrong reason.

- [ ] **Step 2: Run to verify it fails.** Expected: FAIL — the arena survives.

- [ ] **Step 3: Implement** — add `self._calibrations.discard_arena(video)` beside the existing `discard` in the loop.

- [ ] **Step 4: Run to verify it passes.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "fix(pose-batch): clear the arena along with the line"
```

---

### Task 6: Copy an arena as a starting point

Stamps corners onto the selected videos as **unconfirmed**, retargeting `frame_size` to each target's own resolution the way `_retarget_calibration` does for lines. Net-new: no arena-copy path exists.

**Files:**
- Create: `src/glider/gui/pose_batch/arena_actions.py`
- Modify: `src/glider/gui/pose_batch/window.py` (button + delegation; `_open_arena` confirms on accept)
- Test: `tests/unit/gui/pose_batch/test_window_arena.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_copying_an_arena_lands_unconfirmed(window, tmp_path, monkeypatch):
    """A copied arena that does not fit shows no residual warning, so it must
    not satisfy the Run gate until someone has seen the overlay."""
    from glider.gui.pose_batch import arena_actions

    monkeypatch.setattr(arena_actions, "resolution_of", lambda v: (640, 480))
    videos = _ready_window(window, tmp_path, count=2)
    window._calibrations.set_arena(videos[0], _arena())

    arena_actions.copy_arena_to(window._calibrations, videos[0], videos[1:])

    assert window._calibrations.get_arena(videos[1]) is not None
    assert not window._calibrations.is_arena_confirmed(videos[1])
    assert window._calibrations.missing_arenas(videos) == [videos[1]]


def test_a_copied_arena_takes_the_target_resolution(window, tmp_path, monkeypatch):
    from glider.gui.pose_batch import arena_actions

    monkeypatch.setattr(arena_actions, "resolution_of", lambda v: (1280, 720))
    videos = _ready_window(window, tmp_path, count=2)
    window._calibrations.set_arena(videos[0], _arena())
    arena_actions.copy_arena_to(window._calibrations, videos[0], videos[1:])
    assert window._calibrations.get_arena(videos[1]).frame_size == (1280, 720)


def test_an_unreadable_target_is_skipped_not_guessed(window, tmp_path, monkeypatch):
    from glider.gui.pose_batch import arena_actions

    monkeypatch.setattr(arena_actions, "resolution_of", lambda v: None)
    videos = _ready_window(window, tmp_path, count=2)
    window._calibrations.set_arena(videos[0], _arena())
    skipped = arena_actions.copy_arena_to(window._calibrations, videos[0], videos[1:])
    assert window._calibrations.get_arena(videos[1]) is None
    assert skipped == [videos[1]]


def test_accepting_the_arena_dialog_confirms_a_copy(window, tmp_path, monkeypatch):
    """Opening a copied arena and pressing OK is what confirms it."""
    videos = _ready_window(window, tmp_path, count=2)
    window._calibrations.set_arena(videos[1], _arena(), confirmed=False)
    _accept_arena_dialog(monkeypatch, returning=_arena())
    window._open_arena(videos[1])
    assert window._calibrations.is_arena_confirmed(videos[1])
```

`_accept_arena_dialog` patches `ArenaDialog` so `exec()` returns Accepted and `calibration()` returns the given arena — model it on however `test_window_arena.py` already stubs that dialog.

- [ ] **Step 2: Run to verify they fail.** Expected: `ModuleNotFoundError: glider.gui.pose_batch.arena_actions`.

- [ ] **Step 3: Implement `arena_actions.py`**

```python
"""Arena actions for the Batch Pose Tracking window.

Split out of ``window.py`` for the same reason ``calibration_table.py`` was:
that file is over 1200 lines and these are self-contained operations on a
:class:`CalibrationSet` that need no window state. Keeping them here also
makes them testable without building a window.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from glider.vision.arena import ArenaCalibration
from glider.vision.calibration_set import CalibrationSet

logger = logging.getLogger(__name__)


def resolution_of(video: Path) -> tuple[int, int] | None:
    """``(width, height)`` of *video*, or None when it will not open.

    Mirrors what ``_retarget_calibration`` does inline for lines. Returning
    None rather than a default is deliberate: guessing a resolution is exactly
    the error this retargeting exists to prevent.
    """
    from glider.vision.video_source import VideoFileSource

    try:
        with VideoFileSource(video) as source:
            return (source.width, source.height)
    except Exception as e:
        logger.info("cannot read the size of %s: %s", video, e)
        return None


def copy_arena_to(
    calibrations: CalibrationSet, source: Path, targets: Iterable[Path]
) -> list[Path]:
    """Stamp *source*'s corners onto *targets*, unconfirmed. Returns skips.

    Corners are normalized, so they carry across resolutions -- but
    ``frame_size`` must follow the target or ``px_per_cm_at`` reports the
    source's scale for a video that does not have it, which is the same error
    ``_retarget_calibration`` exists to prevent for lines.

    Unconfirmed on purpose. ``residuals()`` is computed from the corners alone,
    so a copy that does not fit this video's floor produces no warning at all;
    on the TRH cohort the camera height varied per animal, which is precisely
    the error the arena was drawn to eliminate.
    """
    arena = calibrations.get_arena(source)
    if arena is None:
        return []
    skipped: list[Path] = []
    for target in targets:
        resolution = resolution_of(target)
        if resolution is None:
            skipped.append(target)
            continue
        calibrations.set_arena(
            target,
            ArenaCalibration(
                corners=list(arena.corners),
                width_cm=arena.width_cm,
                height_cm=arena.height_cm,
                frame_size=resolution,
            ),
            confirmed=False,
        )
    return skipped
```

Check `VideoFileSource`'s real context-manager and attribute names against `_retarget_calibration` (`window.py:729-758`) and copy whatever it does.

- [ ] **Step 4: Wire the window**

Add `_copy_arena_to_selected`, modelled on `_copy_calibration_to_selected` (`window.py:760-807`) — take the source from `_cal_table.selected_videos()[0]`, pass the rest as targets, log each skip by name, then `refresh()` and `_validate()`. Add a "Copy arena…" button beside the existing Copy. In `_open_arena`, pass `confirmed=True` on accept.

- [ ] **Step 5: Run to verify they pass.** Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(pose-batch): copy an arena as an unconfirmed starting point"
```

---

### Task 7: Show arena state in the calibration table

The status cell is written **inline in `refresh()`** at `calibration_table.py:101-105`; there is no `_status_text` method. Extract one as part of this task.

**Files:**
- Modify: `src/glider/gui/pose_batch/calibration_table.py:101-105`
- Test: `tests/unit/gui/pose_batch/test_calibration_table_arena.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize(
    "state, expected",
    [
        (None, "Needs arena"),
        ("unconfirmed", "Confirm it"),
    ],
)
def test_status_column_reports_arena_state(window, tmp_path, state, expected):
    video = _ready_window(window, tmp_path)
    if state:
        window._calibrations.set_arena(video, _arena(), confirmed=False)
    window._cal_table.refresh()
    assert expected in window._cal_table.item(0, 4).text()


def test_a_confirmed_arena_still_reads_as_calibrated(window, tmp_path):
    video = _ready_window(window, tmp_path)
    window._calibrations.set_arena(video, _arena())
    window._cal_table.refresh()
    assert "Calibrated" in window._cal_table.item(0, 4).text()
```

The third case is split out because it already passes — an arena supplies `px_per_mm`, so today's text is already "✓ Calibrated". Only the first two fail first.

- [ ] **Step 2: Run to verify the first two fail.**

- [ ] **Step 3: Implement** — extract the inline status write into `_status_text(video)` and branch:

```python
    def _status_text(self, video: Path) -> str:
        """Run-readiness, not merely whether a scale exists.

        The Run gate now asks for a confirmed arena, so a row that shows only
        "Calibrated" while Run stays disabled sends the operator hunting.
        """
        if self._calibrations.get_arena(video) is None:
            return "⚠ Needs arena"
        if not self._calibrations.is_arena_confirmed(video):
            return "⚠ Copied — confirm it"
        return "✓ Calibrated" if self._calibrations.px_per_mm(video) else "⚠ Needs calibration"
```

- [ ] **Step 4: Run to verify they pass.**

- [ ] **Step 5: Promote the Arena button** — spec section 6 asks for Arena as the primary action in that row with Calibrate demoted. Reorder the button row and update the tooltips to say the arena is required and the line is optional.

- [ ] **Step 6: Run the GUI and vision suites**

Run: `pytest tests/unit/gui tests/unit/vision -q`
Expected: no regressions against the 1354/3 baseline.

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(pose-batch): report arena state in the calibration table"
```

---
## Chunk 2: The gate

Spec section 1. Pure geometry, no GUI, no I/O — the whole chunk is testable with synthetic `PoseData`.

**Task order matters here.** Task 8 delivers a *working* `gate_to_arena` — settings, report, resolution and per-keypoint masking — so that every later task's "Expected: PASS" is a real checkpoint. Building the geometry first and the entry point last would leave three tasks whose tests cannot even be collected.

All Chunk 2 tests go in a new `tests/unit/vision/test_arena_gate.py` with this preamble:

```python
"""Arena gating: rejecting detections that left the floor."""

from __future__ import annotations

import numpy as np
import pytest

from glider.vision.arena import ArenaCalibration
from glider.vision.arena_gate import ArenaGateSettings, gate_to_arena

# The fronto-parallel square from test_arena.py: a real 400x400 px square in a
# 640x480 frame, which is close to what these rigs produce (30 cm at 13.3 px/cm).
_L, _R, _T, _B = 120 / 640, 520 / 640, 40 / 480, 440 / 480
SQUARE = [(_L, _T), (_R, _T), (_R, _B), (_L, _B)]

# A steeply oblique view whose vanishing line crosses the frame at y ~ 90.
# Its homography carries a NEGATIVE scale: w is -1.42 at the arena centre, so
# the whole interior sits on the negative-w side. That is the geometry behind
# `test_a_steeply_oblique_arena_does_not_gate_its_own_interior`.
HORIZON_CORNERS = [(0.40, 0.35), (0.60, 0.35), (0.99, 0.98), (0.01, 0.98)]
#: Pixel centre of the oblique arena, for padding frames that must stay inside
#: it. (320, 240) is the centre of the SQUARE arena and lands outside this one.
OBLIQUE_CENTRE_PX = [320.0, 219.3]


def _pose(xy, confidence=None, names=("a", "b", "c", "d")):
    from glider.vision.pose import PoseData

    xy = np.asarray(xy, dtype=float)
    if confidence is None:
        confidence = np.where(np.isfinite(xy[:, :, 0]), 0.9, 0.0)
    return PoseData(xy=xy, confidence=confidence, keypoint_names=list(names), fps=30.0)


def _arena(corners=SQUARE, **kw):
    kw.setdefault("width_cm", 30.0)
    kw.setdefault("height_cm", 30.0)
    kw.setdefault("frame_size", (640, 480))
    return ArenaCalibration(corners=corners, **kw)


def _one_frame(*points, pad=(320.0, 240.0)):
    """One frame of four keypoints, padded to length with an in-arena point.

    The pad matters: it must be inside whichever arena the test uses, or the
    quorum blanks the frame and per-keypoint assertions read 0 for the wrong
    reason. (320, 240) is the centre of SQUARE; use OBLIQUE_CENTRE_PX for
    HORIZON_CORNERS.
    """
    pts = list(points) + [list(pad)] * (4 - len(points))
    return np.array([pts], dtype=float)
```

### Task 8: A working gate — geometry, resolution, per-keypoint masking

**Files:**
- Create: `src/glider/vision/arena_gate.py`
- Test: `tests/unit/vision/test_arena_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestGeometry:
    def test_a_centred_keypoint_is_inside(self):
        out, report = gate_to_arena(_pose(_one_frame()), _arena())
        assert report.keypoints_masked == 0
        assert np.isfinite(out.xy).all()

    def test_a_keypoint_just_outside_survives_the_default_margin(self):
        """Default margin is 7.5 cm on a 30 cm arena; 13.33 px/cm. 5 cm past
        the left wall is x = 120 - 66.7, which is a rear, not a glitch."""
        _, report = gate_to_arena(_pose(_one_frame([53.3, 240.0])), _arena())
        assert report.keypoints_masked == 0

    def test_a_keypoint_well_outside_is_masked(self):
        """10 cm out exceeds the 7.5 cm margin. Only ONE keypoint moves, so the
        frame survives the quorum and the mask is per-keypoint -- putting all
        four outside would blank the frame and report keypoints_masked == 0."""
        _, report = gate_to_arena(_pose(_one_frame([120 - 133.3, 240.0])), _arena())
        assert report.keypoints_masked == 1
        assert report.frames_blanked == 0

    def test_a_masked_stray_loses_its_confidence_too(self):
        """Spec: both stages spell rejection the same way. A NaN position with
        a 0.9 likelihood is a contradictory row a DLC reader will trust."""
        out, _ = gate_to_arena(_pose(_one_frame([120 - 133.3, 240.0])), _arena())
        assert np.isnan(out.xy[0, 0]).all()
        assert out.confidence[0, 0] == 0.0

    def test_a_point_near_the_vanishing_line_is_masked(self):
        """As w approaches 0 the projected centimetres blow up. inf compares
        correctly against the margin, so the rectangle test catches this on its
        own -- no separate horizon guard is needed or wanted (see below)."""
        pose = _pose(_one_frame([320.0, 95.0], pad=OBLIQUE_CENTRE_PX))
        _, report = gate_to_arena(pose, _arena(corners=HORIZON_CORNERS))
        assert report.keypoints_masked == 1

    def test_a_steeply_oblique_arena_does_not_gate_its_own_interior(self):
        """Regression guard against a tempting bug. A homography is defined up
        to scale, so `w` can be negative across the WHOLE arena -- it is -1.42
        at the centre of this one. Rejecting points on `w <= 0` would therefore
        blank every frame of every video on a rig like this."""
        pose = _pose(_one_frame(pad=OBLIQUE_CENTRE_PX))
        _, report = gate_to_arena(pose, _arena(corners=HORIZON_CORNERS))
        assert report.keypoints_masked == 0
        assert report.frames_blanked == 0


class TestResolution:
    def test_the_pose_resolution_wins_over_the_arena(self):
        """The arena records where the corners were clicked; the pose records
        what the video was tracked at. Using the wrong one skews every point."""
        pose = _pose(_one_frame())
        pose.xy[:] = [640.0, 480.0]
        pose.metadata["resolution"] = [1280, 960]
        _, report = gate_to_arena(pose, _arena())
        assert report.keypoints_masked == 0

    def test_an_explicit_resolution_beats_the_arena(self):
        pose = _pose(np.full((1, 4, 2), [640.0, 480.0]))
        _, report = gate_to_arena(pose, _arena(), resolution=(1280, 960))
        assert report.keypoints_masked == 0

    def test_it_refuses_rather_than_guessing(self):
        """from_dlc_csv populates no metadata, so a CSV-loaded track has no
        resolution of its own. Falling through silently would gate the wrong
        region."""
        with pytest.raises(ValueError, match="resolution"):
            gate_to_arena(_pose(_one_frame()), _arena(frame_size=(0, 0)))


class TestReport:
    def test_it_names_the_keypoints_it_masked(self):
        _, report = gate_to_arena(_pose(_one_frame([-900.0, -900.0])), _arena())
        assert report.masked_by_keypoint["a"] == 1
        assert report.masked_by_keypoint["b"] == 0

    def test_it_records_the_settings_and_the_arena(self):
        settings = ArenaGateSettings(margin_cm=2.0)
        _, report = gate_to_arena(_pose(_one_frame()), _arena(), settings=settings)
        assert report.settings == settings
        assert len(report.arena_corners) == 4

    def test_an_explicit_margin_overrides_the_default(self):
        """5 cm out survives the 7.5 cm default but not a 2 cm margin."""
        pose = _pose(_one_frame([53.3, 240.0]))
        _, report = gate_to_arena(pose, _arena(), settings=ArenaGateSettings(margin_cm=2.0))
        assert report.keypoints_masked == 1

    def test_an_empty_pose_returns_a_zeroed_report(self):
        _, report = gate_to_arena(_pose(np.zeros((0, 4, 2))), _arena())
        assert report.frames_total == 0
        assert report.blanked_fraction == 0.0

    def test_a_degenerate_arena_propagates(self):
        """Spec: callers catch and skip, mirroring _score_zones. The gate does
        not silently pass the pose through."""
        from glider.vision.arena import DegenerateArenaError

        with pytest.raises(DegenerateArenaError):
            gate_to_arena(_pose(_one_frame()), _arena(corners=[(0.5, 0.5)] * 4))
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/unit/vision/test_arena_gate.py -v`
Expected: collection error — `ModuleNotFoundError: glider.vision.arena_gate`.

- [ ] **Step 3: Write the module**

```python
"""Rejecting pose detections that left the arena.

``filtering.smooth()`` is per-keypoint and temporal: it masks by confidence,
fills gaps, and medians. None of that catches the detector finding something
that is not the animal, because the detector is confident when it does -- on
one cohort it sat on bench floor past the chamber wall at likelihood 0.58-0.87,
well clear of the 0.5 the batch tracker masks at.

Catching that needs geometry, which :class:`~glider.vision.arena.ArenaCalibration`
now supplies. Keypoints are mapped onto the floor in centimetres and judged
against a rectangle with a margin, rather than tested against a pixel polygon:
the margin is then a physical distance instead of a pixel count that means
something different at each wall, and the test is a comparison rather than a
point-in-polygon walk.

The margin exists because the arena quad is the *floor plane*. An animal
rearing against a wall projects above that plane and lands genuinely outside
the quad, so a bare containment test would delete real rearing -- invisibly,
which is worse than leaving a visible glitch.

Qt-free on purpose: ``run_batch`` and a GUI button both drive this, and a
notebook can too.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from glider.vision.arena import ArenaCalibration

logger = logging.getLogger(__name__)

# NOTE: this is the FINAL form. `inside_fraction` does not exist until Task 11,
# and listing it here early makes ruff fail the Task 8 and Task 9 commits with
# F822 (undefined name in __all__). Omit that entry until Task 11 adds it.
__all__ = ["ArenaGateSettings", "GateReport", "gate_to_arena", "inside_fraction"]

#: Margin as a fraction of the shorter arena side when none is given. A quarter
#: of a 30 cm arena is 7.5 cm, which clears any plausible rear -- a 9 cm rear
#: under a 1 m camera projects about 3 cm past the far wall -- while still
#: catching the bench-floor detections this exists for.
_DEFAULT_MARGIN_FRACTION = 0.25


@dataclass(frozen=True)
class ArenaGateSettings:
    margin_cm: float | None = None
    min_inside_fraction: float = 0.5
    min_detected_fraction: float = 0.0

    def margin_for(self, arena: ArenaCalibration) -> float:
        if self.margin_cm is not None:
            return float(self.margin_cm)
        return _DEFAULT_MARGIN_FRACTION * min(arena.width_cm, arena.height_cm)


@dataclass(frozen=True)
class GateReport:
    frames_total: int
    frames_considered: int
    frames_blanked: int
    keypoints_masked: int
    masked_by_keypoint: dict[str, int] = field(default_factory=dict)
    settings: ArenaGateSettings = field(default_factory=ArenaGateSettings)
    arena_corners: list = field(default_factory=list)

    @property
    def blanked_fraction(self) -> float:
        """Share of *considered* frames the gate blanked.

        Considered, not total: a frame the tracker never saw is already blank,
        and counting it would dilute the number on precisely the heavy-dropout
        sessions where blanking concentrates -- so the warning would under-fire
        exactly where it is needed.
        """
        return self.frames_blanked / self.frames_considered if self.frames_considered else 0.0


def _resolve_resolution(pose, arena, explicit) -> tuple[int, int]:
    """Frame size the keypoints were measured on.

    Order matters. ``pose.metadata`` is what the video was actually tracked at;
    the explicit argument is what a caller read from the sidecar; and
    ``arena.frame_size`` is only where the corners happened to be clicked. The
    post-hoc path *must* pass the explicit one -- ``from_dlc_csv`` populates no
    metadata at all, so a CSV-loaded track would otherwise fall through to the
    arena's frame size and be gated against the wrong region, silently.
    """
    for candidate in ((pose.metadata or {}).get("resolution"), explicit, arena.frame_size):
        if candidate:
            width, height = (int(v) for v in candidate)
            if width > 0 and height > 0:
                return width, height
    raise ValueError(
        "cannot gate without a frame resolution: pass resolution=, or give the "
        "pose a metadata['resolution'], or draw the arena on a sized frame"
    )


def _outside(arena, xy_px, resolution, margin_cm) -> np.ndarray:
    """Boolean ``(T, K)``: keypoints beyond the arena plus its margin.

    **The bounded rectangle test is sufficient; do not add a horizon guard.**
    It is tempting to also reject points with ``w <= 0``, on the reasoning that
    the divide by ``w`` wraps points past the vanishing line back into
    plausible coordinates. It does not, and the guard is actively harmful:

    * A homography is defined up to scale, so the sign of ``w`` is not
      intrinsic. On a steeply oblique rig ``w`` is negative across the *entire*
      arena -- -1.42 at the centre of the one in the tests -- so ``w <= 0``
      would blank every frame of every video from that camera.
    * The preimage of a bounded rectangle under a projective map is a bounded
      quadrilateral that cannot cross the vanishing line, so no point past the
      horizon can land inside arena-plus-margin. Verified numerically: a sweep
      of the frame found zero such points.
    * As ``w`` approaches 0 the coordinates go to ``±inf``, and ``inf`` compares
      correctly against the margin. A ``0/0`` NaN would read as inside, but it
      cannot arise: ``H @ v = 0`` has no non-trivial solution for an invertible
      ``H``, and ``_check_simple`` already rejects degenerate quads.

    The matmul is still written out rather than calling
    :meth:`ArenaCalibration.to_arena_cm`, which routes into
    ``cv2.perspectiveTransform``: this keeps the whole ``(T, K)`` sweep in one
    float64 numpy expression and does not depend on OpenCV's undocumented
    behaviour as ``w`` approaches zero.

    NaN maps to NaN and every comparison against NaN is False, so an absent
    keypoint is not "outside" -- it is simply not present, which the caller
    accounts for separately.
    """
    flat = np.asarray(xy_px, dtype=np.float64).reshape(-1, 2)
    width, height = resolution
    homogeneous = np.stack(
        [flat[:, 0] / width, flat[:, 1] / height, np.ones(len(flat))], axis=0
    )  # (3, N)
    projected = arena.homography() @ homogeneous
    w = projected[2]
    with np.errstate(invalid="ignore", divide="ignore"):
        x, y = projected[0] / w, projected[1] / w
        beyond = (
            (x < -margin_cm)
            | (x > arena.width_cm + margin_cm)
            | (y < -margin_cm)
            | (y > arena.height_cm + margin_cm)
        )
    return beyond.reshape(np.asarray(xy_px).shape[:2])
```

- [ ] **Step 4: Write `gate_to_arena` with per-keypoint masking only**

The quorum arrives in Task 10; this version reports `frames_blanked=0` and is enough to make every test above pass.

```python
def gate_to_arena(pose, arena, *, settings=None, resolution=None):
    """Blank detections that left the arena. Returns ``(gated pose, report)``."""
    settings = settings or ArenaGateSettings()
    out = pose.copy()
    corners = [list(c) for c in arena.corners]
    if pose.n_frames == 0:
        return out, GateReport(0, 0, 0, 0, {}, settings, corners)

    resolution = _resolve_resolution(pose, arena, resolution)
    detected = np.isfinite(pose.xy).all(axis=-1)
    outside = _outside(arena, pose.xy, resolution, settings.margin_for(arena))

    # Strays. A keypoint the detector never localized is not a stray.
    stray = detected & outside
    out.xy[stray] = np.nan
    out.confidence[stray] = 0.0

    names = list(pose.keypoint_names)
    return out, GateReport(
        frames_total=int(pose.n_frames),
        frames_considered=int((detected.sum(axis=1) > 0).sum()),
        frames_blanked=0,
        keypoints_masked=int(stray.sum()),
        masked_by_keypoint={n: int(stray[:, i].sum()) for i, n in enumerate(names)},
        settings=settings,
        arena_corners=corners,
    )
```

- [ ] **Step 5: Run to verify they pass**

Run: `pytest tests/unit/vision/test_arena_gate.py -v`
Expected: PASS — all of `TestGeometry`, `TestResolution`, `TestReport`.

- [ ] **Step 6: Confirm the oblique-arena regression test bites**

Temporarily add `beyond |= np.isfinite(flat).all(axis=1) & ~(w > 0)` before the
`return` — the horizon guard this design deliberately omits — and re-run.
Expected: `test_a_steeply_oblique_arena_does_not_gate_its_own_interior` FAILS,
because that arena's interior has `w < 0` throughout. Remove the line again. If
the test still passes, it is not pinning the invariant it claims to.

- [ ] **Step 7: Commit**

```bash
git add src/glider/vision/arena_gate.py tests/unit/vision/test_arena_gate.py
git commit -m "feat(vision): mask keypoints that left the arena floor"
```

---

### Task 9: `detected` must test confidence, not just NaN

The trap that would make this gate destroy good data. The Ultralytics branch appends `r.keypoints.xy[best]` unmodified (`vision/pose/core.py:424`) — it does **not** NaN-mask, despite the backend branch's parity comment at `core.py:198`. At gate time an unlocalized keypoint is `(0.0, 0.0)` at confidence 0: a *finite* pixel in the top-left corner, outside every arena.

**Files:**
- Modify: `src/glider/vision/arena_gate.py`
- Test: `tests/unit/vision/test_arena_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestDetected:
    def test_ultralytics_zero_padding_is_not_detected(self):
        """Raw YOLO pads unlocalized keypoints with (0,0) at confidence 0.
        Counting those as detected would mask them as out-of-arena and, once
        the quorum lands, blank any frame under half localized."""
        xy = _one_frame([0.0, 0.0])
        conf = np.array([[0.0, 0.9, 0.9, 0.9]])
        _, report = gate_to_arena(_pose(xy, conf), _arena())
        assert report.keypoints_masked == 0

    def test_a_raw_track_and_its_masked_equivalent_gate_identically(self):
        """Same settings must mean the same thing on the inference path (raw,
        zero-padded) and the post-hoc path (an already NaN-masked CSV)."""
        conf = np.array([[0.0, 0.9, 0.9, 0.9]])
        raw = _one_frame([0.0, 0.0])
        masked = raw.copy()
        masked[0, 0] = np.nan

        _, from_raw = gate_to_arena(_pose(raw, conf), _arena())
        _, from_masked = gate_to_arena(_pose(masked, conf), _arena())
        assert from_raw.keypoints_masked == from_masked.keypoints_masked
        assert from_raw.frames_considered == from_masked.frames_considered

    def test_a_uniform_confidence_track_warns_once(self, caplog):
        """A model with no keypoint confidences gets np.ones (core.py:428), so
        (0,0) pads read as detected. Say so rather than gating silently."""
        pose = _pose(_one_frame(), confidence=np.ones((1, 4)))
        with caplog.at_level("WARNING"):
            gate_to_arena(pose, _arena())
        assert caplog.text.count("uniformly 1.0") == 1
```

- [ ] **Step 2: Run to verify they fail.** Expected: the first two FAIL — the `(0,0)` point is masked.

- [ ] **Step 3: Implement**

```python
def _detected(pose) -> np.ndarray:
    """Boolean ``(T, K)``: keypoints the detector actually localized.

    Finite coordinates are not enough. The Ultralytics branch of
    :func:`~glider.vision.pose.core.infer_video` does not NaN-mask below-
    threshold keypoints -- ``mask_low_confidence`` does that later, inside
    ``smooth()``, which runs *after* this gate -- so an unlocalized keypoint
    arrives as ``(0.0, 0.0)`` at confidence 0: a finite pixel at the frame's
    top-left corner, which is outside every arena.

    Testing NaN alone would therefore make ``min_detected_fraction`` inert on
    the inference path while live on the post-hoc path, which reads an already
    masked CSV -- identical settings meaning different things while the
    provenance block recorded them as the same.
    """
    return np.isfinite(pose.xy).all(axis=-1) & (pose.confidence > 0)
```

Replace the inline `detected = ...` in `gate_to_arena` with `_detected(pose)`, and warn once, immediately after the empty-pose early return:

```python
    if np.all(pose.confidence == 1.0):
        # Not a hypothetical: core.py:428 substitutes np.ones when a model
        # emits no keypoint confidences, and (0,0) pads then read as real.
        logger.warning(
            "%s: confidences are uniformly 1.0, so unlocalized keypoints "
            "cannot be told from real ones and (0,0) padding may be gated as "
            "out-of-arena", getattr(pose, "source", "this track"),
        )
```

- [ ] **Step 4: Run to verify they pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(vision): treat unlocalized keypoints as absent, not as out of arena"
```

---

### Task 10: The quorum, and the zero-detection guard

**Files:**
- Modify: `src/glider/vision/arena_gate.py`
- Test: `tests/unit/vision/test_arena_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestQuorum:
    def test_an_occluded_but_in_arena_frame_survives(self):
        """3 of 4 localized, all inside. Legitimate occlusion, not a glitch."""
        xy = np.full((1, 4, 2), np.nan)
        xy[0, :3] = [320.0, 240.0]
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_blanked == 0

    def test_a_relocated_skeleton_is_blanked_whole(self):
        """3 of 4 detected keypoints outside: below min_inside_fraction=0.5."""
        xy = _one_frame([-900.0, -900.0], [-900.0, -900.0], [-900.0, -900.0])
        out, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_blanked == 1
        assert np.isnan(out.xy[0]).all()
        assert (out.confidence[0] == 0).all()

    def test_keypoints_in_a_blanked_frame_are_not_counted_as_strays(self):
        """They were discarded by the frame verdict, not by their position;
        counting both would double-report the same rejection."""
        xy = _one_frame([-900.0, -900.0], [-900.0, -900.0], [-900.0, -900.0])
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.keypoints_masked == 0

    def test_a_zero_detection_frame_is_excluded_not_blanked(self):
        xy = np.full((2, 4, 2), np.nan)
        xy[0] = [320.0, 240.0]
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_total == 2
        assert report.frames_considered == 1
        assert report.frames_blanked == 0
        assert report.blanked_fraction == 0.0

    def test_blanked_fraction_is_zero_when_nothing_was_considered(self):
        _, report = gate_to_arena(_pose(np.full((5, 4, 2), np.nan)), _arena())
        assert report.frames_considered == 0
        assert report.blanked_fraction == 0.0
```

- [ ] **Step 2: Run to verify they fail.** Expected: `test_a_relocated_skeleton_is_blanked_whole` FAILS with `frames_blanked == 0`, and `test_keypoints_in_a_blanked_frame_are_not_counted_as_strays` FAILS with `3 == 0`.

- [ ] **Step 3: Implement**

Replace the tail of `gate_to_arena`, after the stray masking:

```python
    # The quorum, as an independent predicate rather than a second filter.
    # Sequencing a partial-skeleton test before this one would blank both of
    # the cases it exists to distinguish: a 3-of-7 occluded frame and a
    # 6-detected/5-outside relocation are both simply "partial".
    detected_count = detected.sum(axis=1)
    inside_count = (detected & ~outside).sum(axis=1)
    considered = detected_count > 0

    with np.errstate(invalid="ignore", divide="ignore"):
        too_few_inside = considered & (
            inside_count / np.maximum(detected_count, 1) < settings.min_inside_fraction
        )
    blank = too_few_inside
    out.xy[blank] = np.nan
    out.confidence[blank] = 0.0

    # Keypoints inside a blanked frame are not strays: they were discarded by
    # the frame verdict, not by their own position. blank is (T,), so the
    # trailing axis is added to broadcast against stray's (T, K).
    counted = stray & ~blank[:, None]
    names = list(pose.keypoint_names)
    return out, GateReport(
        frames_total=int(pose.n_frames),
        frames_considered=int(considered.sum()),
        frames_blanked=int(blank.sum()),
        keypoints_masked=int(counted.sum()),
        masked_by_keypoint={n: int(counted[:, i].sum()) for i, n in enumerate(names)},
        settings=settings,
        arena_corners=corners,
    )
```

`np.maximum(detected_count, 1)` guards the division; `considered &` then discards the frames where that guard was doing the work, so a zero-detection frame is excluded rather than silently compared against a `nan`.

- [ ] **Step 4: Run the whole gate suite**

Run: `pytest tests/unit/vision/test_arena_gate.py -v`
Expected: PASS. Task 8's `test_a_keypoint_well_outside_is_masked` must still report `keypoints_masked == 1` — one stray in a surviving frame.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(vision): blank frames whose skeleton left the arena"
```

---

### Task 11: `min_detected_fraction`, and `inside_fraction` for reuse

The partial-skeleton test, defaulted **off**. Task 19 needs the same in-arena predicate for candidate re-ranking, so it is factored out here rather than duplicated there — selection and blanking must not be able to disagree.

**Files:**
- Modify: `src/glider/vision/arena_gate.py`
- Test: `tests/unit/vision/test_arena_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestDetectedFraction:
    def test_it_is_off_by_default(self):
        """A partial-but-in-arena frame survives. Default-on would blank every
        legitimately occluded frame in every cohort."""
        xy = np.full((1, 4, 2), np.nan)
        xy[0, :1] = [320.0, 240.0]
        _, report = gate_to_arena(_pose(xy), _arena())
        assert report.frames_blanked == 0

    def test_at_one_it_reproduces_the_prototype(self):
        """min_detected_fraction=1.0 blanks any incomplete frame, which is what
        reject_partial_frames(min_keypoints=None) did."""
        xy = np.full((1, 4, 2), np.nan)
        xy[0, :3] = [320.0, 240.0]
        _, report = gate_to_arena(
            _pose(xy), _arena(), settings=ArenaGateSettings(min_detected_fraction=1.0)
        )
        assert report.frames_blanked == 1

    def test_a_complete_in_arena_frame_survives_at_one(self):
        _, report = gate_to_arena(
            _pose(_one_frame()), _arena(),
            settings=ArenaGateSettings(min_detected_fraction=1.0),
        )
        assert report.frames_blanked == 0


class TestInsideFraction:
    def test_it_scores_a_fully_in_arena_detection_as_one(self):
        from glider.vision.arena_gate import inside_fraction

        xy = np.full((4, 2), [320.0, 240.0])
        conf = np.full(4, 0.9)
        assert inside_fraction(_arena(), xy, conf, (640, 480)) == 1.0

    def test_padding_does_not_drag_the_score_down(self):
        """The same (0,0) trap, at candidate-selection time: a good detection
        with one pad must not score 3/4 and lose to a bench-floor blob."""
        from glider.vision.arena_gate import inside_fraction

        xy = np.array([[0.0, 0.0], [320.0, 240.0], [320.0, 240.0], [320.0, 240.0]])
        conf = np.array([0.0, 0.9, 0.9, 0.9])
        assert inside_fraction(_arena(), xy, conf, (640, 480)) == 1.0

    def test_a_detection_with_nothing_detected_scores_zero(self):
        from glider.vision.arena_gate import inside_fraction

        assert inside_fraction(_arena(), np.zeros((4, 2)), np.zeros(4), (640, 480)) == 0.0
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement**

```python
def inside_fraction(arena, xy, confidence, resolution, settings=None) -> float:
    """Share of one detection's *localized* keypoints that are in the arena.

    Factored out so :func:`gate_to_arena` and the candidate re-ranking in
    :func:`~glider.vision.pose.core.infer_video` cannot drift into a state
    where inference keeps a candidate the gate then deletes.

    Takes ``confidence`` as well as ``xy`` for the reason :func:`_detected`
    explains: raw Ultralytics output pads unlocalized keypoints with ``(0, 0)``
    at confidence 0, and scoring those as out-of-arena would make a good
    detection with a few pads lose to a confident false one. Returns 0.0 when
    nothing was localized, so an empty detection never wins a comparison.
    """
    settings = settings or ArenaGateSettings()
    xy = np.asarray(xy, dtype=float).reshape(1, -1, 2)
    confidence = np.asarray(confidence, dtype=float).reshape(1, -1)
    detected = np.isfinite(xy).all(axis=-1) & (confidence > 0)
    if not detected.any():
        return 0.0
    outside = _outside(arena, xy, resolution, settings.margin_for(arena))
    return float((detected & ~outside).sum() / detected.sum())
```

Add the second predicate to the quorum in `gate_to_arena`:

```python
        too_few_detected = considered & (
            detected_count / pose.n_keypoints < settings.min_detected_fraction
        )
    blank = too_few_inside | too_few_detected
```

- [ ] **Step 4: Run the whole gate suite.** Expected: PASS.

- [ ] **Step 5: Run the full vision suite**

Run: `pytest tests/unit/vision -q`
Expected: no regressions against baseline.

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(vision): add the partial-skeleton test and share the in-arena predicate"
```

---
## Chunk 3: Wiring the gate into inference and into existing tracks

Spec sections 3 and 4, plus the sidecar half of section 5.

### Task 12: Emit the gate report into the sidecar

`write_pose_meta` writes a **whitelist**, so the report will not ride along in `pose.metadata` by itself. No `META_SCHEMA_VERSION` bump: `read_pose_meta` is already tolerant and old readers ignore unknown keys.

**Files:**
- Modify: `src/glider/vision/pose/dlc.py:52-88`
- Test: `tests/unit/vision/pose/test_dlc.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_sidecar_carries_the_gate_block(tmp_path, synthetic_pose):
    synthetic_pose.metadata["arena_gate"] = {
        "frames_total": 100, "frames_considered": 90, "frames_blanked": 9,
        "keypoints_masked": 4, "gated": True,
    }
    csv = tmp_path / "s.csv"
    to_dlc_csv(synthetic_pose, csv)
    assert read_pose_meta(csv)["arena_gate"]["frames_blanked"] == 9


def test_an_ungated_pose_writes_no_gate_block(tmp_path, synthetic_pose):
    """Absent means ungated. Files from before this change keep their meaning."""
    csv = tmp_path / "s.csv"
    to_dlc_csv(synthetic_pose, csv)
    assert "arena_gate" not in read_pose_meta(csv)
```

- [ ] **Step 2: Run to verify the first fails.** Expected: `KeyError: 'arena_gate'`.

- [ ] **Step 3: Implement** — in `write_pose_meta`, beside the existing `resolution` block:

```python
    # Provenance, not decoration: scoring refuses thresholds derived under a
    # different gate, so this block is what makes that check possible. Absent
    # means ungated, which is true of every file written before the gate
    # existed. Optional and additive, so META_SCHEMA_VERSION does not move.
    gate = pose.metadata.get("arena_gate") if pose.metadata else None
    if gate:
        payload["arena_gate"] = gate
```

- [ ] **Step 4: Run to verify they pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/glider/vision/pose/dlc.py tests/unit/vision/pose/test_dlc.py
git commit -m "feat(pose): record arena-gate provenance in the pose sidecar"
```

---

### Task 13: Gate inside `run_batch`

**Files:**
- Modify: `src/glider/vision/pose/batch.py` (`run_batch` signature and body; `raw_output_path` docstring at line 120)
- Test: `tests/unit/vision/pose/test_batch.py`

- [ ] **Step 1: Write the failing tests**

`test_batch.py` already drives `run_batch` with an injected `infer=`; reuse that harness. Spell out every argument — do not elide them.

```python
def _run(tmp_path, video, *, pose, **kw):
    return run_batch(
        [video],
        tmp_path / "exp-7.pt",
        ["a", "b", "c", "d"],
        infer=lambda **_: pose.copy(),
        **kw,
    )


def test_gating_runs_before_zone_scoring(tmp_path, monkeypatch):
    """Centre-time computed from bench-floor detections is meaningless."""
    seen = {}

    def fake_score(video, pose, zones, keypoint):
        seen["blanked"] = int(np.isnan(pose.xy[:, 0, 0]).sum())
        return ""

    monkeypatch.setattr(batch, "_score_zones", fake_score)
    _run(tmp_path, video, pose=_relocated_pose(), arenas={video: _arena()},
         gate=ArenaGateSettings())
    assert seen["blanked"] == 1


def test_raw_is_written_when_gating_without_filtering(tmp_path):
    """_raw is 'what the model actually said'. Gating without it would discard
    data with no companion."""
    _run(tmp_path, video, pose=_relocated_pose(), filtering=None,
         arenas={video: _arena()}, gate=ArenaGateSettings())
    assert raw_output_path(video, tmp_path / "exp-7.pt").exists()


def test_the_primary_carries_the_gate_block(tmp_path):
    _run(tmp_path, video, pose=_relocated_pose(), arenas={video: _arena()},
         gate=ArenaGateSettings())
    meta = read_pose_meta(dlc_output_path(video, tmp_path / "exp-7.pt"))
    assert meta["arena_gate"]["gated"] is True


def test_no_arena_means_no_gating(tmp_path):
    """arenas={} must leave the pipeline byte-identical to today."""
    _run(tmp_path, video, pose=_relocated_pose(), arenas={}, gate=ArenaGateSettings())
    meta = read_pose_meta(dlc_output_path(video, tmp_path / "exp-7.pt"))
    assert "arena_gate" not in meta


def test_a_clean_video_reports_no_warning(tmp_path):
    """gate_warning must be initialized per video, not only assigned inside the
    over-threshold branch -- otherwise the common case raises NameError."""
    result = _run(tmp_path, video, pose=_clean_pose(), arenas={video: _arena()},
                  gate=ArenaGateSettings())
    assert result.completed == [video.resolve()]
```

- [ ] **Step 2: Run to verify they fail.** Expected: `TypeError: run_batch() got an unexpected keyword argument 'arenas'`.

- [ ] **Step 3: Implement**

Add to the signature: `arenas: Mapping[Path, ArenaCalibration] | None = None`, `gate: ArenaGateSettings | None = None`. Add `from dataclasses import asdict` at module level (numpy-free, cheap) and import `ArenaCalibration` / `ArenaGateSettings` under `TYPE_CHECKING` only — `batch.py`'s docstring commits to staying cheap to import because the GUI imports it while building menus.

Replace the `if filtering is not None:` block with:

```python
            arena = (arenas or {}).get(video)
            gating = gate is not None and arena is not None
            # Per video, not before the loop: assigning only inside the
            # over-threshold branch below would raise NameError on every clean
            # video. (zone_warning has this bug today -- it leaks across
            # videos. Fix it here while you are in the file.)
            gate_warning = ""

            # _raw is the "what did the model actually say" file, so it must be
            # pre-gate as well as pre-filter -- and it must exist whenever
            # either is active, or gating discards data with no companion.
            if gating or filtering is not None:
                to_dlc_csv(pose, raw_output_path(video, model_path))

            if gating:
                from glider.vision.arena_gate import gate_to_arena

                try:
                    pose, report = gate_to_arena(pose, arena, settings=gate)
                except ValueError as e:
                    # DegenerateArenaError subclasses ValueError (arena.py:67),
                    # so this covers both. Mirrors _score_zones: by here the
                    # inference is done and valid, and that is what matters.
                    logger.warning("could not gate %s: %s", video.name, e)
                else:
                    pose.metadata["arena_gate"] = {**asdict(report), "gated": True}
                    if report.blanked_fraction > _GATE_WARN:
                        gate_warning = (
                            f"gate blanked {report.blanked_fraction:.1%} of "
                            f"{report.frames_considered} tracked frames"
                        )

            if filtering is not None:
                from glider.vision.pose.filtering import smooth

                pose = smooth(
                    pose,
                    confidence_threshold=filtering.confidence_threshold,
                    max_gap=filtering.max_gap,
                    median_window=filtering.median_window,
                )
```

At module level:

```python
#: Blanked share above which a video is called out rather than merely logged.
#: On the cohort this was built for the bad sessions ran 3-34%, so a session
#: past this is a tracking problem to look at, not a result to analyse.
_GATE_WARN = 0.10
```

Join both warnings in the `WROTE` emit: `message="; ".join(w for w in (zone_warning, gate_warning) if w)`. Update `raw_output_path`'s docstring, which says "Written only when filtering is enabled".

The report reaches the sidecar because every stage of `smooth()` goes through `PoseData.copy()`, which copies `metadata`.

- [ ] **Step 4: Run to verify they pass.** Expected: PASS.

- [ ] **Step 5: Run the pose suite**

Run: `pytest tests/unit/vision/pose -q`
Expected: no regressions.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(pose): gate detections against the arena during batch inference"
```

---

### Task 14: One shared pose-CSV exclusion list

Second pre-existing bug. `_unique_pose_csvs` (`gui/behavior/window.py:2934`) globs `*.csv` and admits anything with `DLC_` in the stem, keyed on filename — it never calls `find_pose_csv`. So `<stem>DLC_<model>_raw.csv` is pooled alongside the primary and **every session is weighted twice**. This is the observed `n_sessions: 122` for 61 videos. Task 15 adds `_ungated`, which would make it worse.

**Files:**
- Modify: `src/glider/vision/pose/dlc.py` (new shared constant), `src/glider/vision/pose/batch.py`, `src/glider/gui/behavior/window.py:2934`
- Test: `tests/unit/gui/test_cohort_collection.py` (new — this tests a GUI symbol, so it belongs under `tests/unit/gui/`, not in the analysis suite)

- [ ] **Step 1: Write the failing test**

```python
def test_cohort_collection_pools_each_session_once(tmp_path):
    """Regression: a folder with primaries and _raw companions pooled every
    session twice, silently halving each animal's weight in the percentiles."""
    from glider.gui.behavior.window import _unique_pose_csvs

    for stem in ("t1_d1", "t1_d2"):
        (tmp_path / f"{stem}DLC_exp-7.csv").write_text("x")
        (tmp_path / f"{stem}DLC_exp-7_raw.csv").write_text("x")
        (tmp_path / f"{stem}DLC_exp-7_ungated.csv").write_text("x")

    found = _unique_pose_csvs(tmp_path)
    assert len(found) == 2
    assert all("_raw" not in p.stem and "_ungated" not in p.stem for p in found)
```

- [ ] **Step 2: Run to verify it fails.** Expected: `assert 6 == 2`.

- [ ] **Step 3: Implement**

Put the constant in `dlc.py`, beside the writers:

```python
#: Stem suffixes that share a pose CSV's "<stem>DLC_<model>" prefix but are not
#: pose data to analyse. Lives here because two separate discovery paths need
#: it and they drifted apart once already: find_pose_csv excluded _raw while
#: the cohort collector did not, so every session was pooled twice and each
#: animal's weight in the percentiles was silently halved.
NOT_POSE_SUFFIXES = ("_raw", "_annotations", "_ungated")
```

**Import it inside function bodies, not at module scope.** `dlc.py` imports pandas at module level; `batch.py`'s docstring commits to staying cheap to import because `gui/behavior/window.py:66` imports it while building menus, and `window.py` already imports `dlc` only lazily (lines 807, 866, 1879). A module-level import here would drag pandas into GUI startup.

In `batch.py`'s `find_pose_csv`, replace the module-level `_NOT_POSE_SUFFIXES` with a local import. In `_unique_pose_csvs`:

```python
    from glider.vision.pose.dlc import NOT_POSE_SUFFIXES

    for path in sorted(...):
        if "DLC_" in path.stem and not path.stem.endswith(NOT_POSE_SUFFIXES):
            if path.name not in seen:
                seen[path.name] = path
```

Extend that docstring to say `_raw` and `_ungated` are excluded and why.

- [ ] **Step 4: Run**

Run: `pytest tests/unit/gui/test_cohort_collection.py tests/unit/vision/pose/test_batch.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "fix(behavior): stop pooling _raw companions into cohort thresholds"
```

---

### Task 15: The post-hoc pass

**Files:**
- Modify: `src/glider/vision/arena_gate.py`
- Test: `tests/unit/vision/test_arena_gate.py`

- [ ] **Step 1: Write the failing tests**

```python
def _write_track(tmp_path, *, name="t1_d1DLC_exp-7.csv", resolution=(640, 480),
                 fps=30.0, gate_block=None, outside=False):
    """A real pose CSV plus its sidecar, so gate_pose_csv has something to read."""
    from glider.vision.pose.dlc import to_dlc_csv

    xy = _one_frame([-900.0, -900.0]) if outside else _one_frame()
    pose = _pose(np.repeat(xy, 10, axis=0))
    pose.fps = fps
    pose.metadata["resolution"] = list(resolution)
    if gate_block:
        pose.metadata["arena_gate"] = gate_block
    csv = tmp_path / name
    to_dlc_csv(pose, csv)
    return csv


class TestPostHoc:
    def test_it_preserves_the_original_and_its_sidecar(self, tmp_path):
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        assert (tmp_path / f"{csv.stem}_ungated.csv").exists()
        assert (tmp_path / f"{csv.stem}_ungated.meta.json").exists()

    def test_the_sidecar_keeps_its_resolution(self, tmp_path):
        """to_dlc_csv rebuilds the sidecar from pose.metadata and from_dlc_csv
        populates none, so a naive round trip destroys resolution -- breaking
        the viewer and making a second pass refuse."""
        csv = _write_track(tmp_path, resolution=(640, 480))
        gate_pose_csv(csv, _arena())
        assert read_pose_meta(csv)["resolution"] == [640, 480]

    def test_it_keeps_the_fps(self, tmp_path):
        csv = _write_track(tmp_path, fps=25.0)
        gate_pose_csv(csv, _arena())
        assert read_pose_meta(csv)["fps"] == 25.0

    def test_a_second_pass_with_the_same_settings_is_skipped(self, tmp_path):
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        before = csv.read_bytes()
        gate_pose_csv(csv, _arena())
        assert csv.read_bytes() == before

    def test_the_skip_does_not_re_read_the_track(self, tmp_path, monkeypatch):
        """arena_corners is declared list[tuple] but returns as list[list], so
        an identity comparison would never match and the skip never fire.

        This must be pinned on BEHAVIOUR, not on file bytes. Because a re-gate
        reads from `_ungated`, an unskipped repeat is byte-identical to a skip
        — so every file-content assertion passes with the comparison broken,
        while the real cost (rewriting every CSV in a cohort on every pass) is
        invisible. Making the read itself explode is what actually bites.
        """
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())

        def _boom(*a, **k):
            raise AssertionError("skipped pass must not re-read the track")

        monkeypatch.setattr("glider.vision.pose.dlc.from_dlc_csv", _boom)
        gate_pose_csv(csv, _arena())

    def test_a_skipped_pass_returns_the_recorded_report(self, tmp_path):
        """Exercises _report_from_block, which nothing else covers."""
        csv = _write_track(tmp_path, outside=True)
        first = gate_pose_csv(csv, _arena())
        assert gate_pose_csv(csv, _arena()).frames_blanked == first.frames_blanked

    def test_changed_settings_regate_from_the_original(self, tmp_path):
        """The documented workflow: run with defaults, read the report,
        escalate. The second run must not eat the true original."""
        csv = _write_track(tmp_path, outside=True)
        gate_pose_csv(csv, _arena())
        original = (tmp_path / f"{csv.stem}_ungated.csv").read_bytes()
        gate_pose_csv(csv, _arena(),
                      settings=ArenaGateSettings(min_detected_fraction=1.0))
        assert (tmp_path / f"{csv.stem}_ungated.csv").read_bytes() == original

    def test_it_refuses_an_inference_gated_primary(self, tmp_path):
        """A primary gated by run_batch has a gate block and no _ungated twin.
        Renaming it would make 'the original survives' false."""
        csv = _write_track(tmp_path, gate_block={"gated": True, "settings": {}})
        with pytest.raises(ValueError, match="_raw"):
            gate_pose_csv(csv, _arena(),
                          settings=ArenaGateSettings(min_detected_fraction=1.0))

    def test_a_sidecar_less_csv_still_gates(self, tmp_path):
        """write_pose_meta is best-effort and DEFAULT_FPS exists for exactly
        these files, so a missing sidecar must not end the pass."""
        csv = _write_track(tmp_path)
        meta_path(csv).unlink()
        gate_pose_csv(csv, _arena(), settings=ArenaGateSettings(margin_cm=7.5))
        assert (tmp_path / f"{csv.stem}_ungated.csv").exists()
```

The last test needs the arena's own `frame_size` to supply the resolution, since the sidecar is gone — `_resolve_resolution`'s third fallback.

- [ ] **Step 2: Run to verify they fail.** Expected: `ImportError: cannot import name 'gate_pose_csv'`.

- [ ] **Step 3: Implement**

```python
#: GateReport fields reconstructible from a stored block. `gated` is added on
#: write and is not a field; `settings` round-trips as a plain dict and has to
#: be re-hydrated, or the returned report would compare unequal to a fresh one.
_REPORT_KEYS = (
    "frames_total", "frames_considered", "frames_blanked",
    "keypoints_masked", "masked_by_keypoint", "arena_corners",
)


def _report_from_block(block) -> GateReport:
    fields = {k: block[k] for k in _REPORT_KEYS if k in block}
    return GateReport(**fields, settings=ArenaGateSettings(**block.get("settings", {})))


def gate_pose_csv(csv_path, arena, *, settings=None) -> GateReport:
    """Gate a tracked CSV in place, keeping the original as ``_ungated``.

    Always reads the *pristine* track. When ``<stem>_ungated.csv`` already
    exists it is the input and only the primary is overwritten; the original is
    never renamed over. Without that rule the documented workflow destroys it:
    run with defaults, read the report, escalate a known-bad cohort to
    ``min_detected_fraction=1.0`` -- and the second run would rename the
    already-gated primary over the true original, compound the second gate on
    the first, and record only the second settings as provenance.
    """
    from glider.vision.pose.dlc import (
        from_dlc_csv, meta_path, read_pose_meta, to_dlc_csv,
    )

    csv_path = Path(csv_path)
    settings = settings or ArenaGateSettings()
    ungated = csv_path.with_name(f"{csv_path.stem}_ungated{csv_path.suffix}")
    source = ungated if ungated.exists() else csv_path

    existing = (read_pose_meta(csv_path) or {}).get("arena_gate")
    if _same_gate(existing, settings, arena):
        return _report_from_block(existing)

    if not ungated.exists() and existing:
        raise ValueError(
            f"{csv_path.name} was gated during inference and has no _ungated "
            f"companion, so the original cannot be preserved. Re-gate from its "
            f"_raw file, or re-run inference with the settings you want."
        )

    # Read before renaming: from_dlc_csv reads fps from the sidecar.
    meta = read_pose_meta(source) or {}
    pose = from_dlc_csv(source)
    gated, report = gate_to_arena(
        pose, arena, settings=settings, resolution=meta.get("resolution")
    )

    if source == csv_path:
        # rename, not os.replace: refusing an existing target is the point.
        csv_path.rename(ungated)
        # Best-effort, like write_pose_meta itself: a CSV predating sidecars is
        # a supported case, and one missing file must not end a batch re-gate.
        if meta_path(csv_path).exists():
            meta_path(csv_path).rename(meta_path(ungated))

    if meta.get("resolution"):
        gated.metadata["resolution"] = meta["resolution"]
    gated.metadata["arena_gate"] = {**asdict(report), "gated": True}
    to_dlc_csv(gated, csv_path)
    return report


def _same_gate(block, settings, arena) -> bool:
    """Whether *block* records this exact gate. Value comparison, not identity.

    ``arena_corners`` is declared ``list[tuple[float, float]]`` but comes back
    from JSON as a list of lists, so an identity comparison never matches and
    the idempotency skip would never fire -- making every re-run rewrite, and
    the ``_ungated`` guard the only thing standing between a re-run and data
    loss.
    """
    if not block:
        return False
    corners = [[float(x), float(y)] for x, y in arena.corners]
    stored = [[float(x), float(y)] for x, y in block.get("arena_corners", [])]
    return block.get("settings") == asdict(settings) and stored == corners
```

- [ ] **Step 4: Run to verify they pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(vision): re-gate tracked pose CSVs without losing the original"
```

---

### Task 16: The re-gate button

**Files:**
- Create: `src/glider/gui/pose_batch/regate_worker.py`
- Modify: `src/glider/gui/pose_batch/window.py` (button), `arena_actions.py` (the walk)
- Test: `tests/unit/gui/pose_batch/test_window_arena.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_regate_is_disabled_without_pose_csvs(window, tmp_path):
    video = _ready_window(window, tmp_path)
    window._calibrations.set_arena(video, _arena())
    window._validate()
    assert not window._regate_button.isEnabled()


def test_regate_is_disabled_by_an_unconfirmed_arena(window, tmp_path):
    video = _ready_window(window, tmp_path, with_csv=True)
    window._calibrations.set_arena(video, _arena(), confirmed=False)
    window._validate()
    assert not window._regate_button.isEnabled()


def test_regate_is_enabled_with_a_confirmed_arena_and_a_csv(window, tmp_path):
    video = _ready_window(window, tmp_path, with_csv=True)
    window._calibrations.set_arena(video, _arena())
    window._validate()
    assert window._regate_button.isEnabled()


def test_one_bad_video_does_not_end_the_pass(tmp_path, monkeypatch):
    """A refusal or an unreadable file is a skip, not a stop -- the whole point
    of a batch operation is that it finishes."""
    from glider.gui.pose_batch.arena_actions import regate_videos

    calls = []

    def flaky(csv, arena, settings=None):
        calls.append(csv)
        if len(calls) == 1:
            raise ValueError("refused")
        return None

    monkeypatch.setattr("glider.vision.arena_gate.gate_pose_csv", flaky)
    gated, skipped = regate_videos([v1, v2], calibrations, on_log=lambda m: None)
    assert len(calls) == 2
    assert gated == 1 and skipped == 1
```

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement the walk in `arena_actions.py`**

```python
def regate_videos(videos, calibrations, *, settings=None, on_log=None, on_progress=None):
    """Re-gate each video's pose CSV in place. Returns ``(gated, skipped)``.

    Never raises for one video. A refusal (``ValueError``) and an unreadable
    file (``OSError``) are both skips with a logged reason: this is a batch
    maintenance pass over a whole cohort, and stopping at the first awkward
    session would leave the folder half-converted with no record of where.
    """
    from glider.vision.arena_gate import gate_pose_csv
    from glider.vision.pose.batch import find_pose_csv

    gated = skipped = 0
    for index, video in enumerate(videos):
        arena = calibrations.get_arena(video)
        csv = find_pose_csv(video)
        if arena is None or csv is None:
            skipped += 1
        else:
            try:
                report = gate_pose_csv(csv, arena, settings=settings)
            except (ValueError, OSError) as e:
                skipped += 1
                if on_log:
                    on_log(f"{video.name}: skipped ({e})")
            else:
                gated += 1
                if on_log:
                    on_log(f"{video.name}: blanked {report.blanked_fraction:.1%}")
        if on_progress:
            on_progress(index + 1, len(videos))
    return gated, skipped
```

- [ ] **Step 4: Implement `regate_worker.py`**

```python
class RegateWorker(QObject):
    """Re-gate tracked CSVs off the GUI thread.

    A sibling to :class:`PoseBatchWorker` rather than a mode of it: that one
    owns a GPU-bound inference run with per-frame progress and a cancel that
    has to land between frames, this one is I/O over a handful of CSVs.
    Folding them together would put two cancellation stories in one object.
    """

    progress = pyqtSignal(int, int)
    log = pyqtSignal(str)
    finished = pyqtSignal(int, int)  # gated, skipped
```

Its `run()` calls `regate_videos(..., on_log=self.log.emit, on_progress=self.progress.emit)` and emits `finished`. All the logic lives in `arena_actions` so it stays testable without a thread.

- [ ] **Step 5: Wire the button** beside Run/Cancel. Enabled when `missing_arenas(self._videos)` is empty **and** at least one video has a `find_pose_csv` hit. Confirm before starting:

> Rewrite N pose CSVs in place? Originals are kept as `_ungated.csv`.

- [ ] **Step 6: Run to verify they pass.** Expected: PASS.

- [ ] **Step 7: Run the GUI suite**

Run: `pytest tests/unit/gui -q`
Expected: no regressions.

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat(pose-batch): add a button to re-gate already-tracked videos"
```

---

## Chunk 4: Provenance enforcement and candidate selection

Spec section 5's enforcement half, and section 2.

### Task 17: Cohort-level gate block

**Files:**
- Modify: `src/glider/analysis/behavior/cohort_speed.py`
- Test: `tests/unit/analysis/behavior/test_cohort_speed.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_cohort_block_omits_arena_corners(tmp_path):
    """Corners are per-video: 31 sessions have 31 perimeters, so no single
    fingerprint could match them all and comparing would raise every time."""
    thresholds = _thresholds(gated=True)
    block = thresholds.to_dict()["arena_gate"]
    assert "arena_corners" not in block
    assert block["gated"] is True
    assert "settings" in block


def test_a_v1_file_still_loads(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "freeze_mm_s": 5.0, "dart_mm_s": 300.0, "n_sessions": 4,
    }))
    assert CohortSpeedThresholds.load(path) is not None


def test_a_v1_file_reads_as_ungated(tmp_path):
    """Gating did not exist when v1 was written, so absent means ungated --
    the only reading that does not silently defeat the guard on stale files."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "freeze_mm_s": 5.0, "dart_mm_s": 300.0, "n_sessions": 4,
    }))
    assert CohortSpeedThresholds.load(path).gate_provenance["gated"] is False


def test_a_mixed_pool_is_refused(tmp_path):
    """One boolean cannot describe a half-gated cohort: whichever value it
    took, the other half would hard-raise at scoring time."""
    gated = _session_csv(tmp_path, "a", gated=True)
    ungated = _session_csv(tmp_path, "b", gated=False)
    with pytest.raises(CohortSpeedError, match="mix of gated and ungated"):
        compute_cohort_thresholds([gated, ungated], px_per_mm=1.3, fps=30.0)
```

Match `compute_cohort_thresholds`' real signature before writing that last call. `_thresholds` and `_session_csv` are small local helpers — build them from whatever `test_cohort_speed.py` already uses to construct thresholds and session files.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement**

- Bump `SCHEMA_VERSION` to 2.
- Relax `from_dict`'s check (`cohort_speed.py:251`) from `!= SCHEMA_VERSION` to `not in (1, SCHEMA_VERSION)`.
- Add `gate_provenance: dict = field(default_factory=lambda: {"gated": False})`, defaulting a missing block to `{"gated": False}` on load and writing it under `"arena_gate"` in `to_dict`.
- Add the up-front mixed-pool check in `compute_cohort_thresholds`, reading each session's sidecar via `read_pose_meta`.
- Add `_REJECT_WARN = 0.05` (this is **new** — it exists only in the prototype worktree, not in this tree) and the per-session callout.

**`blanked_fraction` is not in the sidecar** — it is a `@property`, so `asdict(report)` omits it. Compute it from the stored fields with a zero guard:

```python
def _blanked_fraction(block) -> float:
    considered = block.get("frames_considered", 0)
    return block.get("frames_blanked", 0) / considered if considered else 0.0
```

- [ ] **Step 4: Run to verify they pass.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(behavior): record which gate a cohort's thresholds were derived under"
```

---

### Task 18: Refuse to score gated poses against ungated thresholds

The check goes **before the batch/stream fork**, not inside `batch_apply`. `batch_apply` declines and returns False for an annotated output video (`classify/batch.py:428`) and for a CNN sequence model (`classify/batch.py:433`), and the caller then falls through to `LiveInferencePipeline`, whose `_make_tracker` reads `config.pose_csv_in` directly. Siting it inside `batch_apply` would let both score gated data against ungated thresholds silently.

**Files:**
- Modify: `src/glider/analysis/behavior/classify/__init__.py` (`resolve_speed_thresholds` at 148, annotation at 168, insertion after the `opts.update(...)` at 508-529), `src/glider/analysis/behavior/classify/pipeline.py` (`LiveInferenceConfig`)
- Test: `tests/unit/analysis/behavior/classify/test_gate_provenance.py` (new)

- [ ] **Step 1: Write the failing tests**

```python
def test_gated_csv_against_ungated_thresholds_raises(tmp_path):
    with pytest.raises(ValueError, match="re-derive"):
        run_classification(_video(tmp_path), pose_csv_in=_gated_csv(tmp_path),
                           cohort_thresholds=_cohort(tmp_path, gated=False))


def test_it_raises_on_the_annotated_video_path_too(tmp_path):
    """batch_apply declines this one and falls through to the streaming
    pipeline, which reads the same CSV."""
    with pytest.raises(ValueError, match="re-derive"):
        run_classification(_video(tmp_path), pose_csv_in=_gated_csv(tmp_path),
                           cohort_thresholds=_cohort(tmp_path, gated=False),
                           output_video=tmp_path / "out.mp4")


def test_it_raises_for_a_cnn_sequence_model(tmp_path):
    """The other batch_apply decline path. Same fall-through, same exposure."""
    with pytest.raises(ValueError, match="re-derive"):
        run_classification(_video(tmp_path), pose_csv_in=_gated_csv(tmp_path),
                           cohort_thresholds=_cohort(tmp_path, gated=False),
                           model=_cnn_sequence_model())


def test_matching_provenance_scores_normally(tmp_path):
    run_classification(_video(tmp_path), pose_csv_in=_gated_csv(tmp_path),
                       cohort_thresholds=_cohort(tmp_path, gated=True))


def test_absolute_thresholds_are_not_checked(tmp_path):
    """cm/s cut-offs are not 'derived from ungated poses' at all. Since the
    post-hoc pass rewrites every tracked session, checking here would break
    every non-cohort scoring run."""
    run_classification(_video(tmp_path), pose_csv_in=_gated_csv(tmp_path),
                       freeze_cm_s=0.5, dart_cm_s=30.0)


def test_percentile_thresholds_are_not_checked(tmp_path):
    """They are derived from the very CSV being scored, so they cannot
    disagree with it."""
    run_classification(_video(tmp_path), pose_csv_in=_gated_csv(tmp_path),
                       freeze_pct=1.0, dart_pct=99.5)


def test_a_speed_only_run_with_no_csv_is_not_checked(tmp_path):
    """No CSV, no sidecar, no CSV-side provenance to compare."""
    run_classification(_video(tmp_path), pose_csv_in=None, speed_only=True,
                       cohort_thresholds=_cohort(tmp_path, gated=False))
```

`run_classification` has **no `pose` parameter** — the spec's "already-loaded `pose=`" refers to the internal `batch_apply(..., pose=pose)` at line 575. The last test drives the same path from the public API.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Produce `gate_provenance`**

Nothing emits it today: the cohort branch reduces the loaded object to two floats at `classify/__init__.py:260-261`, and the tail builds `out: dict[str, float] = {"freeze_threshold": ..., "dart_threshold": ...}` at 329-333. Widening the annotation alone would leave `opts.get("gate_provenance")` permanently `None`, so every gated CSV would raise.

In `resolve_speed_thresholds`, capture it in the cohort branch:

```python
        cohort = CohortSpeedThresholds.load(cohort_thresholds)
        gate_provenance = cohort.gate_provenance
```

default it to `None` on the other branches, widen the return annotation from `dict[str, float]` to `dict[str, object]` (line 168), and add it to `out`:

```python
    out: dict[str, object] = {"freeze_threshold": freeze_px, "dart_threshold": dart_px}
    if gate_provenance is not None:
        out["gate_provenance"] = gate_provenance
```

Add `gate_provenance: dict | None = None` to `LiveInferenceConfig` so `LiveInferenceConfig(**opts)` accepts it.

- [ ] **Step 4: Add the check**

Immediately after the `opts.update(...)` block (508-529), before `LiveInferenceConfig` at 539 and the fork at 566:

```python
    # Before the batch/stream fork, because batch_apply declines two paths that
    # still read this CSV. A cut-off calibrated against gated speed and applied
    # to ungated speed is the same class of mistake as thresholding one time
    # window against another -- and it is silent, which is why this raises.
    #
    # Only against cohort thresholds. Absolute cm/s cut-offs are not derived
    # from poses at all, and percentile ones come from the very CSV being
    # scored, so neither can disagree with it.
    if pose_csv_in is not None and cohort_thresholds is not None:
        _refuse_gate_mismatch(pose_csv_in, opts.get("gate_provenance"))
```

```python
def _refuse_gate_mismatch(pose_csv, cohort_gate) -> None:
    from glider.vision.pose.dlc import read_pose_meta

    csv_gated = bool((read_pose_meta(pose_csv) or {}).get("arena_gate", {}).get("gated"))
    cohort_gated = bool((cohort_gate or {}).get("gated"))
    if csv_gated == cohort_gated:
        return
    raise ValueError(
        f"{Path(pose_csv).name} is {'gated' if csv_gated else 'ungated'} but "
        f"these thresholds were derived from "
        f"{'gated' if cohort_gated else 'ungated'} poses. Re-derive the cohort "
        f"thresholds from the same data you are scoring."
    )
```

- [ ] **Step 5: Run to verify they pass.** Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(behavior): refuse to score a gated track against ungated thresholds"
```

---

### Task 19: Arena-aware candidate selection

`core.py:420` keeps the highest box-confidence detection and discards the rest. If YOLO finds the mouse at 0.85 and a blob on the bench floor at 0.90, GLIDER takes the blob and the correct detection never reaches the CSV. A re-ranking, not a filter: it can only replace a bad pick with a good one.

**Files:**
- Modify: `src/glider/vision/pose/core.py:318-330` (backend delegation) and `405-440` (the loop), `src/glider/vision/pose/batch.py` (pass the arena into `infer`)
- Test: `tests/unit/vision/pose/test_core.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_an_in_arena_candidate_beats_a_more_confident_outsider():
    result = _fake_result(
        boxes_conf=[0.85, 0.90],
        keypoints=[_inside_arena(), _on_the_bench_floor()],
        keypoint_conf=[[0.9] * 4, [0.9] * 4],
    )
    pose = _infer_with(result, arena=_arena())
    assert _is_inside(pose.xy[0])


def test_it_falls_back_to_argmax_when_none_are_inside():
    """Never turns a frame that had a usable detection into a dropout."""
    result = _fake_result(
        boxes_conf=[0.4, 0.9],
        keypoints=[_far_outside(), _further_outside()],
        keypoint_conf=[[0.9] * 4, [0.9] * 4],
    )
    pose = _infer_with(result, arena=_arena())
    np.testing.assert_allclose(pose.xy[0], _further_outside())


def test_a_padded_but_correct_detection_still_wins():
    """The (0,0) trap at selection time: judging on xy alone would score a good
    detection with two pads at 2/4 and hand the frame to the blob."""
    result = _fake_result(
        boxes_conf=[0.85, 0.90],
        keypoints=[_inside_with_two_pads(), _on_the_bench_floor()],
        keypoint_conf=[[0.9, 0.9, 0.0, 0.0], [0.9] * 4],
    )
    pose = _infer_with(result, arena=_arena())
    assert _is_inside(pose.xy[0, 0])


def test_without_an_arena_the_result_is_unchanged():
    """The guard that this cannot silently move existing results."""
    result = _fake_result(boxes_conf=[0.85, 0.90],
                          keypoints=[_inside_arena(), _on_the_bench_floor()],
                          keypoint_conf=[[0.9] * 4, [0.9] * 4])
    np.testing.assert_array_equal(
        _infer_with(result, arena=None).xy, _infer_with(result).xy
    )


def test_an_arena_without_settings_does_not_crash():
    """run_batch's `gating = gate is not None and arena is not None` means the
    arena can arrive without settings."""
    result = _fake_result(boxes_conf=[0.9], keypoints=[_inside_arena()],
                          keypoint_conf=[[0.9] * 4])
    _infer_with(result, arena=_arena(), gate_settings=None)


def test_the_backend_path_logs_the_no_op(caplog, tmp_path):
    """_infer_video_backend yields one detection per frame -- nothing to
    re-rank. A documented no-op, logged so it is not read as a silent failure."""
    with caplog.at_level("INFO"):
        _infer_backend_with(arena=_arena())
    assert "no candidates to re-rank" in caplog.text
```

`_fake_result` stands in for an Ultralytics `Results`: an object with `.boxes.conf`, `.keypoints.xy` and `.keypoints.conf`, each `.cpu().numpy()`-able. `test_core.py` already builds one — reuse it and add `keypoint_conf`.

- [ ] **Step 2: Run to verify they fail.**

- [ ] **Step 3: Implement**

```python
def _pick_candidate(result, confidences, arena, settings, resolution) -> int:
    """Index of the detection to keep.

    Plain ``argmax`` when there is no arena, so this path stays byte-identical
    to what it was. With one, the highest-confidence candidate whose keypoints
    clear ``min_inside_fraction`` wins -- and if none do, ``argmax`` again.

    That fallback is what makes this a re-ranking rather than a filter: it can
    replace a bad pick with a good one but can never turn a frame that had a
    usable detection into a dropout. Blanking is the gate's job, downstream.
    """
    if arena is None:
        return int(confidences.argmax())

    from glider.vision.arena_gate import ArenaGateSettings, inside_fraction

    # run_batch gates on `gate is not None and arena is not None`, so an arena
    # can arrive here with no settings.
    settings = settings or ArenaGateSettings()
    keypoint_conf = result.keypoints.conf
    for index in np.argsort(confidences)[::-1]:
        xy = result.keypoints.xy[index].cpu().numpy()
        conf = (
            keypoint_conf[index].cpu().numpy()
            if keypoint_conf is not None
            else np.ones(len(xy))
        )
        if inside_fraction(arena, xy, conf, resolution, settings) >= settings.min_inside_fraction:
            return int(index)
    return int(confidences.argmax())
```

**`frame_size` does not exist in `core.py`** — the string appears zero times, and `video_resolution(video_path)` is only called at line 460, after the loop. Hoist it before the loop:

```python
    from glider.vision.video_source import video_resolution

    resolution = video_resolution(video_path)
```

and reuse that single value for both the metadata dict at 460 and `_pick_candidate`. Then in the loop:

```python
            if r.boxes is not None and r.boxes.conf is not None:
                confidences = r.boxes.conf.cpu().numpy()
                best = _pick_candidate(r, confidences, arena, gate_settings, resolution)
            else:
                best = 0
```

Add `arena=None, gate_settings=None` to `infer_video`. Log the no-op **before delegating** at `core.py:318-330`, since `_infer_video_backend` is never given them:

```python
    if spec.kind != "yolo":
        if arena is not None:
            logger.info(
                "%s yields one detection per frame, so there are no candidates "
                "to re-rank; the arena still gates downstream", spec.kind,
            )
        return _infer_video_backend(...)
```

In `run_batch`, resolve `arena` **before** the `infer(...)` call — Task 13 currently resolves it after — and pass `arena=arena, gate_settings=gate`.

- [ ] **Step 4: Run to verify they pass.** Expected: PASS.

- [ ] **Step 5: Run everything**

Run: `pytest tests/unit -q`
Expected: the 1354/3 baseline plus the new tests, 0 failures.

- [ ] **Step 6: Lint and format**

```bash
ruff check . && black --check .
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat(pose): prefer in-arena detections when picking a candidate"
```

---

## Done when

- `pytest tests/unit -q` passes with no regressions against the 1354/3 baseline.
- `ruff check .` and `black --check .` are clean.
- An arena drawn in Batch Pose Tracking survives closing and reopening the tool.
- Run is disabled until every video has a confirmed arena.
- A folder of primaries plus `_raw` and `_ungated` companions pools one CSV per session.
- A tracked video's `.meta.json` carries an `arena_gate` block, and scoring it against ungated cohort thresholds raises rather than producing a number.
