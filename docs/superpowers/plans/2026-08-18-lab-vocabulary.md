# Lab Vocabulary Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a lab define its groups, strains, solutions and routes once, so every subject form offers them as choices instead of asking for free text — and make that discoverable at first launch.

**Architecture:** A headless JSON store beside the existing device library, read by `SubjectDialog` to populate editable combo boxes, plus a setup dialog shown once after the onboarding tour and reachable afterwards from the Experiment menu.

**Tech Stack:** Python 3.11–3.13, PyQt6, pytest + pytest-qt, ruff + black at line length 100.

**Spec:** `docs/superpowers/specs/2026-08-18-lab-vocabulary-and-setup-design.md`

---

## Conventions for every task

```bash
cd C:\Users\bradh\glider\.worktrees\lab-vocabulary
set PYTHONPATH=src
C:/Users/bradh/glider/.venv/Scripts/python.exe -m pytest tests/unit/core/test_vocabulary.py -q
```

Windows: the `PYTHONPATH` separator is `;`, not `:` — Python splits on `os.pathsep` regardless of shell, and getting it wrong silently tests the main checkout instead of this worktree. Verify once with `python -c "import glider; print(glider.__file__)"`.

GUI tasks additionally need `QT_QPA_PLATFORM=offscreen`. **Task 1 must not** — if its tests need it, something imported Qt that should not have.

After each task: `ruff check src tests` and `black --check src tests` clean.

Known pre-existing flake: `test_delay_node_is_accurate_under_realistic_loop_pressure` (Windows, ~1 in 5, marked slow). If it is the only failure, re-run it alone rather than chasing it.

---

## File structure

| File | Responsibility | Imports Qt |
|---|---|---|
| `src/glider/core/vocabulary.py` | *New.* Load/save the five lists; case-insensitive dedup; forgiving failure. | No |
| `src/glider/gui/dialogs/subject_dialog.py` | *Modified.* Four fields become vocabulary-backed combos; learn novel values on save. | Yes |
| `src/glider/gui/dialogs/lab_setup_dialog.py` | *New.* The setup form: five editable lists, Skip and Done. | Yes |
| `src/glider/gui/main_window.py` | *Modified.* First-run hook + Experiment → Lab Setup… | Yes |
| `docs-site/building/devices.md` or a new page | *Modified.* Document it. | — |

---

## Chunk 1: The store (Task 1)

### Task 1: `core/vocabulary.py`

Mirror `src/glider/core/device_library.py` — same shape, same forgiving posture. Read it first; it is the precedent and it already works.

**Files:**
- Create: `src/glider/core/vocabulary.py`
- Test: `tests/unit/core/test_vocabulary.py`

**Verified facts you will need:**
- `config.py:105` — `library_dir: Path = Path.home()/".glider"/"library"`. Devices live in `library_dir/"devices"`, functions in `library_dir/"functions"`. Vocabulary is a **single file**, `library_dir/"vocabulary.json"`, not a directory.
- `device_library.load_definitions` catches `(json.JSONDecodeError, OSError, UnicodeDecodeError)`, logs a warning and skips. Match that: a malformed vocabulary must never stop the app starting.
- `subject_dialog.py:32-33` defines module-level `SEX_OPTIONS` and `ROUTE_OPTIONS`. The defaults must match them, or the dialog will offer two different sets. **Read those constants** rather than copying values from this plan.

  **But both lists lead with `""`** — an empty first entry so the combo can show "nothing chosen". That is a UI affordance, not a vocabulary value. Copying it in would put a blank row in the setup dialog that a user can neither name nor sensibly remove. Strip the empty string when deriving the defaults, and keep it when populating the combo. The Task 2 test that pins defaults against these constants must compare against the stripped form, or it will encode the bug.

- [ ] **Step 1: Write the failing tests**

`tests/unit/core/test_vocabulary.py`:

```python
"""The lab's vocabulary: what it stores, and what it refuses to lose.

Case-insensitive de-duplication is the whole point of this module, not a
nicety -- "Control" and "control" splitting one treatment group into two is
the failure that makes a cohort comparison silently wrong, and no downstream
code can tell the two apart afterwards.
"""

import json

from glider.core.vocabulary import LISTS, Vocabulary, load, save


def test_a_missing_file_yields_defaults(tmp_path):
    vocab = load(tmp_path)

    assert vocab.groups == []
    assert vocab.strains == []
    assert "IP" in vocab.routes
    assert "Male" in vocab.sexes


def test_round_trip(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Drug A")
    save(vocab, tmp_path)

    assert load(tmp_path).groups == ["Drug A"]


def test_a_malformed_file_yields_defaults_rather_than_raising(tmp_path):
    """A broken vocabulary must never stop the app starting."""
    (tmp_path / "vocabulary.json").write_text("{ not json", encoding="utf-8")

    vocab = load(tmp_path)

    assert vocab.groups == []
    assert "IP" in vocab.routes


def test_a_file_that_is_not_an_object_yields_defaults(tmp_path):
    (tmp_path / "vocabulary.json").write_text("[1, 2, 3]", encoding="utf-8")

    assert load(tmp_path).groups == []


def test_case_insensitive_dedup(tmp_path):
    """The failure this module exists to prevent."""
    vocab = load(tmp_path)
    vocab.add("groups", "Control")
    vocab.add("groups", "control")
    vocab.add("groups", "  CONTROL  ")

    assert vocab.groups == ["Control"]


def test_first_spelling_wins(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Drug A")
    vocab.add("groups", "drug a")

    assert vocab.groups == ["Drug A"]


def test_entry_order_is_preserved(tmp_path):
    vocab = load(tmp_path)
    for name in ("Vehicle", "Drug A", "Control"):
        vocab.add("groups", name)

    assert vocab.groups == ["Vehicle", "Drug A", "Control"]


def test_whitespace_only_is_rejected(tmp_path):
    vocab = load(tmp_path)

    assert vocab.add("groups", "   ") is False
    assert vocab.groups == []


def test_values_are_stored_trimmed(tmp_path):
    vocab = load(tmp_path)
    vocab.add("strains", "  C57BL/6J  ")

    assert vocab.strains == ["C57BL/6J"]


def test_add_reports_whether_it_changed_anything(tmp_path):
    """The subject dialog only re-saves when something was learned."""
    vocab = load(tmp_path)

    assert vocab.add("groups", "Control") is True
    assert vocab.add("groups", "Control") is False


def test_remove(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Control")

    assert vocab.remove("groups", "control") is True
    assert vocab.groups == []


def test_unknown_list_name_is_refused(tmp_path):
    vocab = load(tmp_path)

    try:
        vocab.add("colours", "red")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown list")


def test_an_unwritable_directory_reports_rather_than_raising(tmp_path, monkeypatch):
    """A read-only home must not take the app down; the caller shows the failure."""
    vocab = load(tmp_path)

    def boom(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr("pathlib.Path.write_text", boom)

    assert save(vocab, tmp_path) is False


def test_saved_file_is_readable_json(tmp_path):
    vocab = load(tmp_path)
    vocab.add("groups", "Control")
    save(vocab, tmp_path)

    data = json.loads((tmp_path / "vocabulary.json").read_text(encoding="utf-8"))
    assert data["groups"] == ["Control"]
    assert data["schema_version"] == "1.0"


def test_every_declared_list_round_trips(tmp_path):
    """Guards against a list being added to LISTS but forgotten in save/load."""
    vocab = load(tmp_path)
    for name in LISTS:
        vocab.add(name, f"value-for-{name}")
    save(vocab, tmp_path)

    reloaded = load(tmp_path)
    for name in LISTS:
        assert f"value-for-{name}" in reloaded.get(name)
```

- [ ] **Step 2: Run to verify they fail**

Expected: `ModuleNotFoundError: No module named 'glider.core.vocabulary'`.

- [ ] **Step 3: Implement**

Required public surface, because the tests pin it: `LISTS` (the five names), `Vocabulary` with `.groups/.strains/.solutions/.routes/.sexes`, `.get(name)`, `.add(name, value) -> bool`, `.remove(name, value) -> bool`; module functions `load(library_dir) -> Vocabulary` and `save(vocab, library_dir) -> bool`.

Points the tests will not tell you but the design depends on:

- Import **no Qt**, and do not import `subject_dialog` for the defaults — that would drag Qt in. Instead define the defaults here and add a test in Task 2 asserting they match `ROUTE_OPTIONS`/`SEX_OPTIONS`, so the duplication is pinned rather than assumed.
- `add` compares on `value.strip().casefold()`. Store the **first** spelling seen.
- `save` returns `False` on `OSError` and logs; it never raises. The setup dialog surfaces that.
- Unknown list name raises `KeyError` — that is a programming error, not user data.

- [ ] **Step 4: Run tests — all pass, with `QT_QPA_PLATFORM` unset**

Verify the Qt-free property explicitly:

```bash
python -c "import sys, glider.core.vocabulary; print([m for m in sys.modules if m.startswith('PyQt')] or 'no Qt')"
```

- [ ] **Step 5: Commit**

```bash
git add src/glider/core/vocabulary.py tests/unit/core/test_vocabulary.py
git commit -m "feat(core): a lab vocabulary store beside the device library"
```

---

## Chunk 2: The dialogs (Tasks 2–4)

### Task 2: `SubjectDialog` reads the vocabulary

**Files:**
- Modify: `src/glider/gui/dialogs/subject_dialog.py`
- Test: `tests/unit/gui/test_subject_dialog_vocabulary.py`

**Read this before editing — it is where this task goes wrong.**

Three fields convert from `QLineEdit` to editable `QComboBox`:

| Field | Line | Currently |
|---|---|---|
| group | 136 | `QLineEdit` |
| strain | 194 | `QLineEdit` |
| solution | 221 | `QLineEdit` |

**`route` (236) is already an editable `QComboBox`** — the spec said otherwise and the spec was wrong. It needs only its items sourced from the vocabulary. Same for `sex` (176), already a combo.

**Each converted field has FOUR call sites, not one:**

1. construction (~136/194/221)
2. `_populate` — `setText(...)` → `setCurrentText(...)` (~274, 301, 302)
3. `get_subject`, **update-existing branch** — `.text()` → `.currentText()` (~352, 356, 357)
4. `get_subject`, **create-new branch** — `.text()` → `.currentText()` (~368, 372, 373)

Miss branch 3 or 4 and half the save paths silently stop persisting. A test that only ever creates a new subject will pass while editing is broken, so **the tests below cover both branches deliberately.**

- [ ] **Step 1: Write the failing tests**

Cover: combos populate from vocabulary; a value not in the vocabulary can still be typed and is returned; a novel value is added to the vocabulary on save; **editing an existing subject persists a changed group** (the update branch); creating a new subject persists it (the create branch); an empty vocabulary leaves the dialog fully usable; `ROUTE_OPTIONS`/`SEX_OPTIONS` match `vocabulary`'s defaults.

- [ ] **Step 2: Run — verify they fail for the right reason**

The both-branches tests should fail on the *values*, not on import errors. If they error rather than fail, the fixture is wrong.

- [ ] **Step 3: Implement**

The dialog takes an optional `vocabulary` argument, defaulting to `load(get_config().paths.library_dir)`. Learn-on-save: after building the Subject, `add()` each non-empty value and `save()` **only if any `add` returned True** — that is what `add`'s bool return is for. A failed save is logged, never raised: losing a vocabulary entry must not lose the subject.

- [ ] **Step 4: Run tests + the existing subject dialog tests**

Existing tests must pass unchanged. If one fails on `.text()`, that is a call site you missed.

- [ ] **Step 5: Commit**

### Task 3: `LabSetupDialog`

**Files:**
- Create: `src/glider/gui/dialogs/lab_setup_dialog.py`
- Test: `tests/unit/gui/test_lab_setup_dialog.py`

Five editable lists, one per `LISTS` entry. Add by typing and pressing Enter; remove per row. **Skip and Done both close and both mark seen** — Skip must not feel like a trap, and a user who cannot answer should be able to leave in two seconds.

Skip does **not** write. Done writes, and if `save()` returns False shows the failure inline and stays open rather than silently discarding what was typed.

Tests: adding and removing; Skip leaves the file absent; Done writes; a save failure keeps the dialog open with a visible message; duplicate entry folds silently.

- [ ] Steps 1–5 as above.

### Task 4: First-run hook and menu entry

**Files:**
- Modify: `src/glider/gui/main_window.py`
- Test: `tests/unit/gui/test_lab_setup_entry.py`

**Verified facts:**
- `tour.py:29` — `TOUR_COMPLETE_KEY = "first_run/tour_complete"`, and `tour_complete(settings, key)` returns True once finished **or skipped**. Follow that pattern exactly: new key `first_run/setup_complete`.
- `main_window.py:881` — the Experiment menu, where `&Add Subject...` is added. Put `Lab Setup...` beside it.

Show the setup dialog once, after the golden-path tour resolves either way, and only if `first_run/setup_complete` is unset. The menu entry opens the same dialog unconditionally.

Tests: shown when unset and the tour is complete; **not** shown when already set; not shown before the tour resolves; the menu action exists and opens it; opening from the menu when already complete still works.

- [ ] Steps 1–5 as above.

---

## Chunk 3: Docs (Task 5)

### Task 5: Document it

**Files:** a new `docs-site/building/subjects.md`, linked from `mkdocs.yml` nav.

Cover: what the vocabulary is for; that setup appears once and can be reopened from Experiment → Lab Setup…; that free text always works and novel values are learned; where the file lives.

**Accuracy constraint.** This repo has had seven documented cases of docs asserting things the code does not do. Verify every claim against the code you just wrote. Run `mkdocs build --strict` if mkdocs is available (`uvx mkdocs-material`); skip and say so if not.

- [ ] Write, verify, `mkdocs build --strict`, commit.

---

## Acceptance

- [ ] `test_vocabulary.py` passes with **no** `QT_QPA_PLATFORM` set, and the module imports no Qt
- [ ] `Control` and `control` cannot both exist
- [ ] Editing an existing subject persists a changed group (the update branch)
- [ ] A typed novel strain is offered as a choice for the next subject
- [ ] Skipping setup leaves the dialog fully usable with an empty vocabulary
- [ ] Experiment → Lab Setup… reopens it after first run
- [ ] An existing `.glider` file with free-text groups loads and saves unchanged
- [ ] Full suite green; ruff and black clean

## Not in this plan

Cohort analysis — grouping recordings by treatment and the tidy export — is designed in spec §9 and deliberately deferred until group names are trustworthy. Nothing here reads recordings back.
