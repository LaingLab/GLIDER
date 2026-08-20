# Lab vocabulary and the setup step

**Status:** approved 2026-08-18
**First of two.** The second, cohort analysis, is designed but deferred until this
lands — it depends on group names being trustworthy.

---

## 1. The problem

GLIDER already records everything needed to compare treatments. `Subject` carries
twelve fields — subject ID, name, group, age, sex, weight, strain, solution,
concentration, dose, route, notes — and `DataRecorder` writes every one of them
into the header of each recording (`data_recorder.py:337-362`). `parse_csv`
already reads them back as parsed metadata; verified, `Group` and `Dose` come out
as ordinary dict entries.

Three things break the loop between that and a usable experiment:

1. **Nobody finds the fields.** The entire surface is one menu item, Experiment →
   Add Subject…. New lab members do not know it exists and use spreadsheets.
2. **`group` is a free-text string.** `Control`, `control` and `Ctrl` are three
   groups. Nothing defines a group, so nothing can validate one.
3. **Nothing downstream consumes any of it.** Every reference to `subject.group`
   outside the session model is a regex `match.group(1)` or an unrelated
   cross-validation grouping. The analysis pipeline does not know treatments
   exist.

This spec addresses 1 and 2. Fixing 3 without them means comparing groups that a
typo can silently split, which is worse than not comparing at all.

## 2. The idea, and the reframe it needed

The proposal was a setup tab at the end of the onboarding tour, walking a new user
through the fields they may need.

The reframe: **at first launch, per-animal facts are unanswerable.** Nobody knows
this cohort's doses on day one. But one question up the ladder is answerable and
stable — *what strains does this lab work with? what treatment groups does it run?
what routes does it use?* That is knowledge the person already has, and it
produces something reusable.

So the setup step collects the **lab's vocabulary**, not an experiment's data.
Every later subject form becomes a dropdown over that vocabulary, with free text
still allowed.

## 3. Non-goals

- **A subject/animal library.** Persisting individual animals across experiments
  is real and wanted, but it is a larger change to the session model and file
  format. Vocabulary is per-lab and small; animals are per-study and structured.
  Separate work.
- **Blocking free text.** A lab that meets a new strain mid-study types it. The
  vocabulary seeds choices; it never constrains them.
- **A mandatory wizard.** Skip is a first-class exit.
- **Cohort analysis.** Designed, deferred (§9).
- **Migrating existing `.glider` files.** Vocabulary is additive; old files load
  unchanged and their free-text values keep working.

## 4. What a vocabulary is

Five lists, each a set of named values a lab reuses:

| List | Example values | Feeds |
|---|---|---|
| `groups` | Control, Drug A, Vehicle | `Subject.group` |
| `strains` | C57BL/6J, BALB/c | `Subject.strain` |
| `solutions` | Saline, Ketamine | `Subject.solution` |
| `routes` | IP, IV, PO, SC | `Subject.route` |
| `sexes` | Male, Female, Unknown | `Subject.sex` (already a combo) |

Deliberately **flat lists of strings**, not structured objects. A group is not yet
a treatment with a dose and a schedule — modelling that is the next spec's job,
and guessing at it now would bake in a shape the cohort work then has to unpick.
What this fixes is that the same *name* is used every time.

`routes` and `sexes` ship with sensible defaults; the rest start empty.

## 5. Storage

Mirrors the device library, which already works and which users already have on
disk:

- One JSON file, `~/.glider/library/vocabulary.json`, beside `devices/` and
  `functions/` under the existing `library_dir` (`config.py:105`).
- Module `src/glider/core/vocabulary.py` with `load(library_dir)` and
  `save(vocab, library_dir)`, mirroring `device_library.save_definition` /
  `load_definitions` — including its failure posture: a malformed or unreadable
  file logs and yields empty lists rather than raising, because a broken
  vocabulary must never stop the app starting.
- No Qt import. Testable headless.

Schema:

```json
{
  "schema_version": "1.0",
  "groups": ["Control", "Drug A"],
  "strains": ["C57BL/6J"],
  "solutions": ["Saline"],
  "routes": ["IP", "IV", "PO", "SC"],
  "sexes": ["Male", "Female", "Unknown"]
}
```

Values are stored trimmed, de-duplicated case-insensitively, and ordered as
entered. Case-insensitive de-duplication is the point of the whole feature: it is
what stops `Control` and `control` both existing.

## 6. The setup step

A form, not a spotlight. The existing tour is a scrim-and-cutout overlay that
teaches *where things are* (`onboarding/overlay.py`); this collects input, so it
is a normal dialog shown when the golden-path tour completes or is skipped.

- One page per list, or one page with five labelled groups — implementation choice
  at plan time; the content is five short lists either way.
- Each list is an editable set: type a value, Enter adds it, a remove control per
  row.
- **Skip** and **Done** both close it and both mark it seen.
- Text at the top says what this is for in one sentence and that it can be changed
  later. A user who cannot answer should be able to tell in two seconds that
  skipping is fine.

**Gating.** A new QSettings key `first_run/setup_complete`, alongside the existing
`first_run/tour_complete` (`tour.py:29`). Shown once, after the golden-path tour
resolves either way.

**Re-entry is required, not optional.** The person doing first launch may not be
the person who knows the answers. Reachable afterwards from **Experiment → Lab
Setup…**, which opens the same dialog. Without this the knowledge arrives after
the only chance to enter it.

## 7. How the vocabulary reaches the subject form

`SubjectDialog` (`gui/dialogs/subject_dialog.py`) currently uses `QLineEdit` for
group, strain, solution and route. Each becomes an **editable `QComboBox`** —
`setEditable(True)` — populated from the vocabulary:

- Existing values are offered as choices.
- Typing a new value is allowed and accepted.
- A value typed that is not in the vocabulary is **added to it on save**, so the
  second animal offers what the first one introduced. This is what keeps the
  vocabulary alive without anyone maintaining it.
- An empty vocabulary yields an empty dropdown that still accepts typing — the
  dialog behaves exactly as today for a user who skipped setup.

The `sex` combo already exists and gains nothing but its values from the same
source.

## 8. Error handling

| Failure | Response |
|---|---|
| `vocabulary.json` missing | Treated as empty; defaults for `routes`/`sexes` |
| Malformed or unreadable | Logged, treated as empty; app starts normally |
| Unwritable library dir | Setup reports it inline and stays open; nothing else blocked |
| Duplicate value entered | Silently folded into the existing one, case-insensitively |
| Value that is only whitespace | Rejected at entry |

## 9. What this unlocks next

With group names trustworthy, the deferred cohort spec becomes worth building:
`analysis/cohort.py` loading many recording directories, grouping sessions by any
metadata key (`Group` by default, but `Strain`, `Dose` and `Sex` are equally valid
axes), a tidy per-session export with metadata columns beside metric columns, and
per-group comparisons using the existing plot functions.

That work deliberately excludes significance testing. Choosing a test, checking
its assumptions and correcting for multiple comparisons are the experimenter's
decisions; a tool that prints an unstated *p* invites exactly the error that gets
papers retracted. Show distributions, group means and n; export data that goes
straight into a tool built for statistics.

## 10. Testing

**Headless** (`core/vocabulary.py`, no Qt): round-trip save/load; missing file
yields defaults; malformed file yields empty rather than raising; case-insensitive
de-duplication (`Control` + `control` → one); whitespace-only rejected; ordering
preserved.

**GUI** (`pytest-qt`): setup dialog adds and removes values; Skip marks complete
without writing; Done writes; the dialog is reachable from the menu after
first-run; `SubjectDialog` combos populate from vocabulary; a typed novel value is
accepted and lands in the vocabulary on save; an empty vocabulary leaves the
dialog fully usable.

**Regression:** an existing `.glider` file with free-text group values loads and
saves unchanged.

## 11. Build order

1. `core/vocabulary.py` + tests. Headless, no UI.
2. `SubjectDialog` combos over the vocabulary, including learn-on-save.
3. The setup dialog itself.
4. Onboarding hook + `Experiment → Lab Setup…` menu entry.
5. Docs.

Steps 1–2 deliver the whole value for anyone who already knows the fields exist.
Steps 3–4 are what make it findable, which was the original complaint.
