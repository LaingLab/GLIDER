# Subjects & Lab Vocabulary

Every experiment you record can carry information about the animals in it: an
ID, a name, a treatment group, a strain, a solution and dose, a route of
administration. GLIDER keeps that information on **subjects**, and it keeps the
*terms* your lab uses for it in a **lab vocabulary** — one shared list of the
groups, strains, solutions, routes and sexes that this lab actually works with.

## Why a vocabulary

Left as free text, these fields drift. One person types `Control`, another
types `control`, a third types `  CONTROL  `, and now there are three treatment
groups where the lab only ever had one. Nothing downstream can tell that they
were meant to be the same, and putting them back together is manual work
somebody does months later at analysis time.

The vocabulary is the fix. You define your terms once, and from then on every
subject form offers them as choices — so the same group is spelled the same way
every time, by everyone.

!!! info "Case, spacing and accents are all treated as the same term"
    The vocabulary compares terms after trimming spaces, ignoring case, and
    normalising Unicode. `Control`, `control` and `  CONTROL  ` are one entry,
    and so are the composed and decomposed spellings of an accented strain name
    — which matters when one bench machine is a Mac and another is a PC, since
    they favour different spellings that look identical on screen. The **first**
    spelling your lab used is the one kept, so your own capitalisation survives.

## Setting up your vocabulary

GLIDER offers the setup form once, on its own, shortly after launch — after the
first-run welcome dialog and the guided tour are out of the way. It is a plain
list editor:

- Type a term into the box at the top of a list and press ++enter++ to add it.
- Press **✕** next to a term to remove it.
- Duplicates and blank entries are simply ignored — if you type `Control`
  twice, you still get one row.

There are five lists:

| List | What goes in it | Starts as |
|---|---|---|
| **Treatment groups** | `Control`, `Vehicle`, `Drug A`… | Empty |
| **Strains** | `C57BL/6J`, `Long-Evans`… | Empty |
| **Solutions and drugs** | `Saline`, `Drug X`… | Empty |
| **Routes of administration** | How a solution is given | `IP`, `IV`, `PO`, `SC`, `IM`, `Topical`, `Inhalation`, `Other` |
| **Sexes** | `Male`, `Female`, `Unknown` | Those three |

The three lists that are specific to your lab start empty; routes and sexes
come pre-filled with the usual options, which you are free to edit.

### Skip is a real option

The person who happens to do the first launch is often not the person who knows
the lab's strains. **Skip** closes the form and writes nothing at all — no file,
no half-filled vocabulary — and GLIDER does not ask again. Only **Done** saves.

Either way, the offer is one-time. Whichever button you press (or if you close
the form with ++esc++ or the window button), GLIDER records that you have seen
it and stops offering.

### Reopening it later

**Experiment → Lab Setup...** opens the same form at any time, whether or not
you skipped it. That is where the lab's strains get filled in later, and where
you tidy up terms that crept in.

!!! note "Runner mode"
    The setup form is a desktop feature. It is never offered on the Raspberry Pi
    runner screen — a 480-pixel touch surface with no menu bar is the wrong
    shape for five editable lists. A runner uses whatever vocabulary was defined
    on the desktop side.

!!! warning "If saving fails"
    If GLIDER cannot write the file — a read-only home directory, a missing
    folder — **Done** does not close the form. It shows you the reason and
    stays open with everything you typed still on screen, so you can fix the
    problem and press **Done** again. Nothing is lost silently.

## Filling in a subject

Open **Experiment → Add Subject...**, or use the **Subjects** table in
**Experiment → Experiment Settings...** to add and edit them. The subject form
has four tabs — Basic, Biological, Solution and Notes — and the vocabulary feeds
five of the fields across them:

| Field | Tab | Behaves like |
|---|---|---|
| **Group** | Basic | Dropdown you can also type into |
| **Strain** | Biological | Dropdown you can also type into |
| **Solution** | Solution | Dropdown you can also type into |
| **Route** | Solution | Dropdown you can also type into |
| **Sex** | Biological | Dropdown only — pick from the list |

Every one of them opens on a blank row, so nothing is ever pre-selected for you:
an animal never ends up quietly labelled with whichever group happens to be
first in the list.

**Sex is the exception to free text.** It is a plain dropdown, so the options
you put in the Sexes list in Lab Setup are the complete set of choices on the
subject form. The other four stay typeable.

### New terms are learned as you go

You never have to stop and go define a term first. Type a group the lab has not
used before, press **OK**, and it is added to the vocabulary there and then —
so the second animal you enter is offered what the first one introduced.

This is what makes skipping the setup form genuinely free: a lab that never
opens Lab Setup still ends up with a vocabulary, built out of what people
actually typed. The setup form is the shortcut, not the requirement.

If the vocabulary file cannot be written at that moment, your subject is still
saved normally — only the new term goes unremembered.

!!! tip "Old experiment files keep working"
    A `.glider` file written before any of this existed opens unchanged. When
    you edit one of its subjects, values that are not in your vocabulary are
    still shown in the Group, Strain, Solution and Route boxes, because those
    boxes accept free text. Sex is the exception: if a saved subject's sex is
    not one of the terms in your Sexes list, that box opens blank, so check it
    before pressing **OK**.

## Where the vocabulary is stored

One JSON file, next to the device library:

```
~/.glider/library/vocabulary.json
```

On Windows that is `C:\Users\<you>\.glider\library\vocabulary.json`.

The file only appears once something writes it — pressing **Done** in Lab Setup,
or saving a subject that introduced a new term. Until then GLIDER just uses the
built-in defaults. Writes are done to a temporary file and then moved into
place, so an interrupted save leaves your previous vocabulary intact.

Because it is a single small file, it is easy to copy: setting up a second rig
with the same vocabulary is a matter of copying `vocabulary.json` across.

### Editing it by hand

You can edit the file in any text editor. It looks like this:

```json
{
  "schema_version": "1.0",
  "groups": ["Control", "Vehicle", "Drug A"],
  "strains": ["C57BL/6J"],
  "solutions": ["Saline", "Drug X"],
  "routes": ["IP", "SC"],
  "sexes": ["Male", "Female", "Unknown"]
}
```

A few things worth knowing before you do:

- **Duplicates are folded when the file is read.** If a hand-edit (or a merge
  between two rigs) leaves both `Control` and `control` in the list, GLIDER
  keeps the first and drops the rest. You do not have to de-duplicate by hand.
- **Leaving a list out restores its defaults.** Delete the `routes` key
  entirely and you get the standard routes back. To genuinely empty a list,
  give it an empty array: `"routes": []`.
- **A broken file will not stop GLIDER starting.** If the JSON is malformed or
  unreadable, GLIDER logs a warning, falls back to the defaults, and carries on.
  So if your terms suddenly vanish from the subject form, suspect a typo in the
  file.
- **Close the Lab Setup form before editing the file.** That form holds its own
  copy of the vocabulary from the moment it opens, and **Done** writes that copy
  over the whole file — so a hand-edit made while it is open would be
  overwritten. Changes you make on disk are picked up the next time you open a
  subject form or Lab Setup.
