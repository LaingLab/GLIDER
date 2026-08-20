# The Builder Window

The Builder is where you design an experiment. This page is about the frame
around the canvas — the panels, the status strip along the top, the command
palette, and the fact that GLIDER remembers how you left it.

## What's on screen

From top to bottom:

| Part | What it is |
|---|---|
| **Menu bar** | Four menus: **File**, **Edit**, **View**, **Help**. |
| **Status strip** | A single 40-pixel row: panel toggles, the experiment name, the run-state pill, one dot per board, and the ++ctrl+k++ hint. |
| **Left panel** | Tabs: **Nodes**, **Hardware**, **Control**, **Files**. |
| **The canvas** | The node graph itself. Everything else gets out of its way. |
| **Right panel** | Tabs: **Properties**, **Camera**. |
| **Status bar** | The thin line along the bottom: connection indicator, session state, and transient messages. |

Four menus, not eight; two panels, not a window full of separate dockable ones.
Nothing became unreachable in the process:
[the four menus that came off the bar](#where-the-other-menus-went) are still
built, their shortcuts still work, and ++ctrl+k++ reaches every command in all
of them.

## The side panels

Everything that used to be its own dockable panel is now a tab in one of two
panels, one on each side of the canvas.

| Panel | Tabs |
|---|---|
| Left | **Nodes** (the node library), **Hardware** (boards and devices), **Control** (drive a device by hand), **Files** (New, Open, Save, Save As) |
| Right | **Properties** (settings for the selected node), **Camera** (the live camera panel) |

Click a tab at the top of a panel to switch to it. Only one tab per side is
visible at a time; the panels are independent, so you can have **Hardware** open
on the left and **Camera** on the right.

### Collapsing and expanding

Each panel can be collapsed to give the canvas its width. There are three ways
to do it, and they all stay in step with each other:

- The two small toggle buttons at the far left of the status strip — the left
  one controls the left panel, the right one the right panel.
- **View → Left Panel** and **View → Right Panel**, which are tick boxes.
- Clicking an icon on a collapsed panel's rail (see below) expands it again.

### Collapsed is a rail, not nothing

A collapsed panel does not disappear. It becomes a **34-pixel rail of icon
buttons**, one per tab, down the outside edge of the window:

| Panel | Rail buttons |
|---|---|
| Left | **N** Nodes, **H** Hardware, **D** Control, **F** Files |
| Right | **P** Properties, **C** Camera |

Hovering a rail button names the tab. Clicking one does two things at once: it
selects that tab **and** expands the panel onto it. So collapsing a panel never
hides where anything lives, and you never have to know a shortcut to get a panel
back.

### Panel widths

Drag the divider between a panel and the canvas to resize it. Three rules apply:

- An expanded panel opens at 260 pixels until you drag it, and then keeps the
  width you chose.
- Collapsing and re-expanding returns a panel to *your* width, not the default.
- An expanded panel will not go narrower than 160 pixels, and the canvas will
  not be squeezed below 240 pixels.

## The status strip

The strip is the one piece of the Builder that cannot be collapsed, hidden or
summoned. That is deliberate: a board that drops out 40 minutes into an
unattended run has to be visible to whoever walks past the rig.

Left to right, it carries:

- **The two panel toggles.**
- **The experiment name**, followed by **— edited** while there are unsaved
  changes. A brand-new session carries its default name, *Untitled
  Experiment*, until you name it in **Experiment Settings…**.
- **The run-state pill.**
- **One dot per board**, each labelled with the board's name.
- **The `Ctrl K` hint**, which is a button — clicking it opens the command
  palette, exactly as the shortcut does.

### The run-state pill

The pill has four words. Some session states ride on a second word after the
first rather than getting a colour of their own, because the failure this pill
exists to prevent is reading "Idle" while hardware is being driven.

| Session state | Pill reads | Colour |
|---|---|---|
| Idle | **Idle** | Grey |
| Ready | **Idle Ready** | Grey |
| Initializing | **Running Starting** | Green |
| Running | **Running** | Green |
| Running, and the recorder is writing data | **Recording** | Red |
| Paused | **Running Paused** | Green |
| Stopping | **Running Stopping** | Green |
| Error | **Error** | Solid red |

A paused or stopping run therefore still reads **Running**. That is the point:
the pill says *idle* only when nothing is live.

### The device dots

There is one dot per **board** registered in this session — an Arduino, a
Raspberry Pi, a mock board — not one per device attached to a board. Each dot
carries the board's name beside it, because a colour alone cannot tell you
*which* board to go and look at. Hovering a dot shows `name — state`.

| Dot | Board's connection state | What it means |
|---|---|---|
| :material-circle:{ style="color:#4ade80" } **Green** | Connected | The board is up and usable. This is the only state that is ever green. |
| :material-circle:{ style="color:#fbbf24" } **Amber** | Connecting **or** Reconnecting | The board is **not** usable right now. *Connecting* is a handshake still in flight. *Reconnecting* is a board that has **already dropped at least once** and is trying to come back. |
| :material-circle:{ style="color:#f87171" } **Red** | Disconnected **or** Error | The board is down. The whole chip tints red and the board's name turns red with it, so it stands out from the healthy boards beside it and not only from how it looked a moment ago. |
| :material-circle:{ style="color:#718096" } **Grey** | Anything GLIDER does not recognise | A state reported by a driver that this version of the strip has no colour for. Hover the dot to read the raw value. Never assume grey is healthy. |

!!! warning "Amber is not 'fine'"
    An amber dot is a board you cannot trust. If it is amber because it is
    *reconnecting*, the board has already dropped at least once during this
    session, and whatever it was doing at that moment was interrupted. Treat
    amber the way you would treat red, and check the **Hardware** tab for what
    happened.

The dots follow each board's real connection state, so the strip describes the
rig as it is now rather than replaying the last transition it happened to hear
about.

## The command palette { #command-palette }

Press ++ctrl+k++ anywhere in the window — or click the **`Ctrl K`** button on
the status strip — and a search box opens over the middle of the Builder.

**It lists every action in every menu GLIDER builds**, including the four menus
that are not on the menu bar. Each row shows three things: the command's name,
the menu it belongs to, and its keyboard shortcut if it has one.

- **Type to narrow it.** Matching is by subsequence, so the letters you type
  have to appear in order but need not be next to each other: `sbj` finds
  **Add Subject…**. Commands whose match starts earlier in the name are ranked
  first, so typing `dsc` puts **Disconnect All** above **Add Subject…**.
- **With nothing typed** the list is in menu order: File, Edit, Experiment,
  View, Hardware, Run, Tools, Help. That is the grouping the menus had.
- ++up++ / ++down++ move, ++enter++ runs the selected command, ++esc++ closes
  it, and clicking a row runs it. Clicking the dimmed area outside the card also
  closes it, without running anything.
- **Unavailable commands are greyed out, not hidden.** If **Undo** is greyed,
  there is nothing to undo — which is a different and more useful answer than
  the command not being there at all. Greyed rows cannot be selected, so
  ++enter++ and the arrow keys skip them and cannot run one by accident.
- The list is re-read **every time you open it**, so what is greyed is what is
  genuinely unavailable at that moment. The search box is cleared each time too,
  so you never reopen onto the last search's leftovers.
- **The palette does not list itself.** There is no "Command Palette" entry
  inside the command palette.

If you type something with no matches, the box says so — *No commands match
'…'* — rather than going blank.

!!! note "Runner mode has no palette contents"
    The palette follows the menus, and [Runner mode](../runner/index.md) builds
    none — so on a Pi kiosk ++ctrl+k++ opens on *No commands available*.
    Switching to the operator view from the desktop with ++f11++ is different:
    the menus are still there, so the palette is still full.

## Where the other menus went

The menu bar keeps **File**, **Edit**, **View** and **Help**. Four menus came
off it. They still exist, they are still built with exactly the same commands,
and **their keyboard shortcuts still work** — they are simply not on the bar.

| Menu | Its commands | How to reach them now |
|---|---|---|
| **Experiment** | Experiment Settings…, Add Subject…, Lab Setup… | ++ctrl+k++ |
| **Hardware** | Add Board…, Add Device…, New Custom Device Type…, Connect All, Disconnect All | ++ctrl+k++ — and **Add Board** and **Add Device** are also buttons in the **Hardware** tab, which is where you were probably already looking |
| **Run** | Start (++f5++), Stop (++shift+f5++), Emergency Stop (++ctrl+shift+escape++) | The shortcuts, or ++ctrl+k++ |
| **Tools** | Behavior Analysis…, Batch Pose Tracking…, Session Review…, GPU / Device Check…, Plugins… | ++ctrl+k++ |

Every row in the palette names the menu it came from, so if you remember that
something "was under Tools", the palette still tells you so.

The **View** menu changed too. The six separate panel toggles it used to carry
are now two — **Left Panel** and **Right Panel** — because every panel that
needed its own menu entry is now a tab, and a collapsed panel leaves its rail
behind rather than vanishing.

For the full list of shortcuts, see
[Keyboard Shortcuts](../reference/shortcuts.md).

## GLIDER remembers your layout

When you close the window, the Builder writes down:

- whether each panel was open or collapsed,
- which tab each panel was showing,
- how wide each panel was, and
- the window's position and size.

Next time you start GLIDER, you get that back. The panel layout is applied after
the tabs have been built, so a saved tab is restored onto the real thing.

Nothing here can stop GLIDER starting. Every saved value is checked on its own
and falls back independently, so a half-written or hand-edited settings file
restores the parts that made sense and uses defaults for the rest:

| Situation | What happens |
|---|---|
| A saved tab no longer exists (a panel changed between versions) | That panel opens on its **first** tab — **Nodes** on the left, **Properties** on the right — and a warning is written to the log. |
| The window was last on a second monitor that is now unplugged | The saved position overlaps no attached screen, so it is ignored and the window is **centred on your primary screen** at its own size. |
| A value is missing, malformed or absurd | That one value falls back to its default (and a width below the 160-pixel minimum is raised to it). The rest of the layout still restores. |
| Nothing has ever been saved — a first launch | Both panels open at 260 pixels, on **Nodes** and **Properties**. |

!!! tip "Getting back to a known layout"
    **View → Default Layout** puts both panels back open on their first tabs and
    resizes the window to 1400 × 900. **View → Compact (1024x768)** shrinks the
    window for a small screen.

## Two status readouts, not one

The strip at the top and the status bar at the bottom are different things and
do not say the same thing:

- The **status strip** describes the *session*: its name, whether it has unsaved
  changes, whether a run is live, and the state of each board.
- The **status bar** carries the connection indicator, the raw session state
  (`State: RUNNING`), and the short messages GLIDER shows when you save a file,
  resize the window or add a device.

## The Analysis panel

One panel is not a tab. When a video tracking run finishes and you choose **Open
in Analysis panel**, the **Analysis** panel opens as a separate dockable panel on
the right of the window, alongside the Builder rather than inside it. You can
move it, float it or close it like any dockable panel; closing it does not
disturb the two side panels. See
[Behavior Analysis](../camera-behavior/behavior.md).
