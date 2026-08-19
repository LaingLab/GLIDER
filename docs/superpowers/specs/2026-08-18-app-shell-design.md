# The Builder shell

**Status:** approved 2026-08-18
**Mockup:** https://claude.ai/code/artifact/f3aa57bc-8788-4743-a711-fdbca94e0e68
**First of two.** The spacing and type scale follows, applied to this frame rather than to chrome this replaces.

---

## 1. The problem

GLIDER reads as an engineering tool rather than a product, and the cause is not
the palette or the spacing. It is that the frame is louder than the content.

Measured on `main`:

| | Today |
|---|---|
| Docks | 7 `QDockWidget`s, each with a Qt title bar, drag handle, float and close buttons |
| Toolbars | 2 |
| Menus | 8 top-level, 30 `QAction`s |
| Finding an action | Read the menus |
| Layout persistence | **None.** `saveState`/`restoreState` are never called |

Before reaching the node graph you pass a menu bar, two toolbars and seven dock
headers. None of the applications this is measured against — VS Code, Obsidian,
Zotero, Claude, Codex — shows a title bar on a panel.

The layout finding is a plain bug independent of any restyle: arrange the
Builder for a rig, quit, and it is gone.

## 2. What this is

One content surface with a collapsible panel on each side and a single permanent
strip above.

- **Left** — what you can add: Nodes, Hardware, Files.
- **Right** — what is selected or live: Properties, Camera.
- **Collapsed** — each side becomes a 34 px icon rail, not nothing (§6).
- **Strip** — experiment name, run state, one dot per device. The only chrome
  that cannot be summoned.
- **Palette** — `Ctrl+K`, fuzzy over every action.

## 3. Non-goals

- **Runner mode.** The Pi surface is a separate 480 px shell (`RunnerShell`) and
  stays exactly as it is. Nothing here touches it.
- **The node graph's own rendering.** `QGraphicsScene`, node cards, ports and
  wires are unchanged. This is the frame around them.
- **Spacing and type scale.** Deliberately the next project: tuning padding on
  chrome about to be deleted is wasted work.
- **Palette colour.** Deep Navy throughout, unchanged.
- **Analysis as a panel.** It does not fit either side (§6) and wants its own
  surface. Out of scope; it keeps its current dock until that is designed.

## 4. Why a new module

`main_window.py` is 2 639 lines. Adding a shell to it makes the largest file in
the GUI substantially larger and harder to reason about.

New package `src/glider/gui/shell/`:

| File | Responsibility |
|---|---|
| `app_shell.py` | The frame: strip, two `SidePanel`s, centre widget, layout persistence |
| `side_panel.py` | One collapsible tabbed panel; knows nothing about what it hosts |
| `status_strip.py` | Experiment name, run state, device dots |
| `command_palette.py` | Fuzzy search over a supplied action list |

**The panel contents do not move.** Every dock already wraps a standalone widget
— `_node_library_panel`, `_hardware_panel`, `_device_control_panel`,
`_camera_panel` — so the shell re-hosts existing widgets. This is a change of
container, not of contents, which is what keeps it tractable.

`MainWindow` keeps ownership of the panels and the core wiring; it hands them to
the shell.

## 5. The strip

Fixed 40 px. Left to right: panel toggles, experiment name with a dirty marker,
run state pill, then device dots and the `Ctrl+K` hint.

**Why this is the exception.** Everything else can be summoned on demand. Live
hardware and recording state cannot: a board dropping 40 minutes into an
unattended run must be visible without being asked for. That single requirement
is what the design is judged on — see mockup state 3, where a device is red
while recording.

Device dots reuse the semantic colours added for the plugin card
(`STATE_OK`/`STATE_WARN`/`STATE_ERR`), which are deliberately distinct from the
cyan accent so "needs attention" never competes with a primary action.

## 6. Side panels

Each side is a `SidePanel`: a tab strip and a content area, collapsible to a
rail. Tabs are supplied by `MainWindow`, so the panel has no knowledge of node
libraries or cameras.

**Collapsed is a rail, not hidden.** 34 px of icons. Pure Claude/Codex would
hide it entirely, but GLIDER has no existing users — for the foreseeable future
*every* user is a first-time user, and a rail keeps the areas visible and gives
back a click target that does not require knowing a shortcut exists. This is the
main deviation from the reference apps and it is deliberate.

**Widths are draggable and remembered.** A splitter each side.

**Analysis belongs to neither side.** It is a full working surface, not an
inspector. It keeps its existing dock for now rather than being forced into a
tab that fits badly.

## 7. Command palette

`Ctrl+K` opens a centred overlay; typing filters; Enter runs; Escape closes.

Sourced from the existing `QAction`s rather than a parallel registry — there are
30, they already carry text, shortcuts and enabled state, and a second list
would drift from the first. Disabled actions are shown greyed rather than
hidden, so the palette answers "can I do this yet?" as well as "where is it?".

Matching is subsequence-based (`sbj` finds "Add Subject…"), ranked by match
position then action text. No fuzzy-matching dependency; the corpus is 30 items.

## 8. Layout persistence

`QSettings` under a `shell/` prefix: which side panels are expanded, which tab
is active on each, both widths, and window geometry. Restored on construction,
saved on close.

A malformed or partial value falls back to the default layout rather than
raising — same posture as the vocabulary store, and for the same reason: a
settings file must never stop the app starting.

Tests inject a `QSettings`; none may touch a developer's real settings.

## 9. The menu bar stays, slimmed

**Decision, flagged for veto.** Obsidian and Claude have no menu bar; VS Code
keeps a full one. GLIDER keeps a slim one: **File, Edit, View, Help**.

Two reasons. On Windows and Linux a desktop application without File → Save is
surprising in a way that reads as unfinished rather than clean. And the menu bar
is free inventory for a first-time user, which matters more here than in an app
with a trained userbase.

Everything domain-specific — Experiment, Hardware, Run, Tools — moves into the
panels and the palette. That is 4 of 8 menus removed and the great majority of
the 30 actions relocated.

## 10. Error handling

| Failure | Response |
|---|---|
| Saved layout malformed or partial | Default layout; logged |
| A saved tab no longer exists | Falls back to the first tab on that side |
| Palette opened with no actions | Empty state naming the condition, not a blank box |
| A panel widget fails to construct | Shell renders without that tab; the rest of the app starts |
| Window geometry off-screen (monitor removed) | Geometry ignored, window centred |

That last one is a real case on a rig whose second monitor is not always
present.

## 11. Testing

**Headless-ish** — `command_palette` matching and ranking, and the layout
serialisation, are plain logic and testable without a shown window.

**GUI** (`pytest-qt`), and every one of these against a **shown** widget:
collapse and expand each side; tab switching; width drag; palette open, filter,
run, dismiss; layout round-trips through `QSettings`; a missing saved tab falls
back; off-screen geometry is ignored.

The shown-widget requirement is not incidental. On the previous branch, 36
dialog tests passed while exercising a widget in a state no user occupies,
because the fixture never called `show()` — Qt installs default-button
behaviour, focus and final geometry on the show event. Two of those tests were
passing while asserting the opposite of reality. Any test of this shell that
does not show the widget is testing a different object.

**Regression:** the Builder still opens, the node graph still edits, hardware
still connects, a recording still runs end to end.

## 12. Build order

1. `SidePanel` — collapsible tabbed panel, standalone, with tests.
2. `StatusStrip` — including device dots driven by real hardware state.
3. `AppShell` — composes them around a centre widget; layout persistence.
4. `MainWindow` re-hosts the existing panels into the shell; docks removed.
5. `CommandPalette`, sourced from the existing actions.
6. Menu bar slimmed; the removed menus' actions verified reachable elsewhere.
7. Docs.

Steps 1–3 build the shell without touching `MainWindow`. Step 4 is the one that
changes what a user sees, and it is reversible on its own.

## 13. Risks

**Concurrency.** Claude and Codex are single-surface because they do one thing.
GLIDER edits a graph while hardware is live and a camera runs. Camera is a
right-panel tab here; if watching video *while* editing is a real workflow, a
tab cannot do it and the right side needs to split vertically. Unresolved, and
the first thing to revisit after step 4.

**Discoverability.** Menus are poor navigation but excellent inventory. Removing
four of them shifts that load onto the panels and the palette. The lab that
prompted this work could not find the subject fields — this must not make that
worse, which is why the rail, the slim menu bar and the greyed-not-hidden
palette entries all exist.

**Scale.** `MainWindow` is 2 639 lines and step 4 touches its construction. The
shell living in its own package limits the blast radius, but this is the
riskiest step and should land on its own.
