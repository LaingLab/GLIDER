# The Node Graph

The node graph is where you design your experiment: a canvas of **nodes** (each
does one job) joined by **wires** (which decide the order things happen in and
what data flows where). This page walks through building and running a graph.

## The editor at a glance

When GLIDER opens in its design layout you'll see three main areas:

- **Node Library** (a dockable panel, usually on the left) — the palette of
  nodes you can add, grouped into sections.
- **The canvas** (the center) — the graph itself, where nodes live and get
  wired together.
- **Properties** (a dockable panel) — shows the settings for whichever node you
  have selected, such as a delay's duration or a node's bound device.

You can show or hide these panels from the **View** menu.

!!! note "Nodes are color-coded by category"
    Every node belongs to one of three categories, shown by its color:

    | Category | Color | What it's for |
    |---|---|---|
    | Hardware | Green | Talking to real devices (`Output`, `Input`, `MotorGovernor`) |
    | Logic | Blue | Flow, timing, and math (`StartExperiment`, `Delay`, `Loop`) |
    | Interface | Orange | On-screen controls and displays used in the Runner |

## Adding nodes

The Node Library groups the nodes you'll reach for most often:

| Section | Nodes |
|---|---|
| **Flow** | Start Experiment, End Experiment, Delay |
| **Functions** | Start Function, End Function |
| **Control** | Loop, Wait For Input |
| **I/O** | Output, Input |
| **Audio** | Audio Playback |
| **Video** | Video Playback |
| **Graph Functions** | Call buttons for functions you've defined (see [Functions](functions.md)) |
| **Zones** | Call buttons for camera zones you've drawn (see [Tracking](../camera-behavior/tracking.md)) |

To place a node, you can either:

- **Drag** it from the library onto the canvas where you want it, or
- **Click** it in the library to drop it in the center of the current view.

Move a node afterward by dragging it. Delete a selected node with your keyboard's
delete key. Every add, move, wire, and delete can be undone with **Edit → Undo**
(`Cmd/Ctrl+Z`) and redone with **Edit → Redo**.

!!! tip "The library is the common set, not the whole catalog"
    The library holds the nodes most experiments need. GLIDER knows about many
    more node types — math (`Add`, `Multiply`, `Clamp`, `Map Range`), comparison
    (`Threshold`, `In Range`), a `PID Controller`, interface widgets (`Button`,
    `Slider`, `Gauge`, `Chart`), and more. See the full
    [Node Catalog](../reference/nodes.md) for everything that exists and what
    each port does.

## Ports: execution vs. data

Every wire in GLIDER is one of two kinds, and each node's ports tell you which
is which by their shape:

- **Execution ports are triangles.** They carry the *turn to act* signal. When a
  node finishes, it fires its execution output, which triggers the next node.
  This is what makes your experiment run step by step. Execution wires are drawn
  as thicker lines with an arrowhead pointing the way the flow travels.
- **Data ports are circles.** They carry *values* — a number, a reading, a
  true/false. A sensor node reads a value and offers it on a data output; another
  node takes it in on a data input. Data wires are thinner and have no arrow.

!!! note "The distinction matters"
    An execution wire says *"do this next."* A data wire says *"here's a value to
    use."* A `Delay` node, for example, has an execution input (start waiting),
    an execution output that fires once the wait is over, and a data input for the
    duration. Wiring the duration into the execution input by mistake won't do
    what you want — keep triangles with triangles and circles with circles.

## Wiring nodes together

To make a connection, click and drag from an **output** port on one node to an
**input** port on another. GLIDER decides the wire's kind automatically from the
port you started on (execution or data).

A few rules the editor enforces:

- You always connect an **output to an input** (you can't join two outputs or two
  inputs).
- A node can't connect to itself.

To remove a wire, select it and delete it. Reconnecting and deleting are undoable
like everything else.

## The Start / End Experiment flow

Every runnable experiment has a beginning and an end:

- **`StartExperiment`** is the entry point. It has a single execution output
  named `next`. When you press **Start**, this node fires, which triggers
  whatever is wired to it — and the chain propagates from there.
- **`EndExperiment`** is the exit point. It has one execution input. When the
  flow reaches it, that branch of the experiment is done.

A minimal graph is just:

```text
StartExperiment ──▶ (your nodes) ──▶ EndExperiment
```

Because execution flows along the triangle wires, the order you connect nodes in
*is* the order they run. Here's the shape of the `dispense` example experiment,
which repeats an action in a loop:

```text
StartExperiment ──▶ Output (set direction) ──▶ Loop
                                                 │  body ──▶ Output (step) ──▶ Delay ──▶ …
                                                 │  done ──▶ EndExperiment
```

The `Loop` node has two execution outputs: `body` (fired once per iteration) and
`done` (fired after the last iteration). Wiring `done` into `EndExperiment` ends
the experiment cleanly once the loop finishes.

## Configuring a node

Select a node and its settings appear in the **Properties** panel. What you see
depends on the node:

- **`Delay`** — a duration and a unit (seconds or milliseconds).
- **`Output` / `Input` / `WaitForInput` / `MotorGovernor` / `Device Action`** —
  a **Device** dropdown to bind the node to a real device (covered in
  [Devices & Hardware](devices.md)).
- **`Loop`** — a repeat count (0 means loop forever until you stop) and a delay
  between iterations.
- **`Timer`** — an interval and unit for how often it ticks.

Changes you make in Properties are saved with the node.

## Useful flow-control nodes

Beyond Start/End, a few nodes shape how your experiment runs:

| Node | What it does |
|---|---|
| **Loop** | Repeats its `body` output N times (0 = forever), then fires `done`. |
| **Sequence** | Fires several execution outputs one after another, in order. |
| **Timer** | Fires a `Tick` at a fixed interval and counts ticks. |
| **Delay** | Waits a set duration, then continues. |
| **Wait For Input** | Pauses until a bound device meets a condition (a digital edge, an analog threshold, or a rotation target), then fires `triggered` — or `timeout` if it waits too long. |

!!! tip "Wait For Input is device-driven"
    `WaitForInput` reads whatever device you bind to it. Depending on the device
    and mode, it can wait for a button press, for a sensor to cross a threshold,
    or for an encoder to turn a set number of revolutions. Some custom devices
    add their own named wait behaviors — see
    [Custom Devices & Plugins](plugins.md).

## Running your graph

Once your flow goes from `StartExperiment` to `EndExperiment`, run it from the
**Run** menu:

| Action | Menu | Shortcut |
|---|---|---|
| Start | Run → Start | `F5` |
| Stop | Run → Stop | `Shift+F5` |
| Emergency Stop | Run → Emergency Stop | `Ctrl+Shift+Esc` |

**Start** fires `StartExperiment` and lets the flow run. **Stop** halts it and
asks nodes to clean up (timers cancel, delays return early, loops end).
**Emergency Stop** is the panic button — it drives devices to a safe state right
away.

!!! warning "You can run without hardware"
    A graph will run even with no boards connected. Hardware nodes simply report
    that nothing is bound and move on, so you can test your logic offline. When
    you're ready for real hardware, connect it (see
    [Devices & Hardware](devices.md)) and bind your nodes.

For running an experiment full-screen on a touchscreen or kiosk — the way you'd
run it day to day in the lab — see [Runner Mode](../runner/index.md).

## Saving your work

Save the whole experiment — graph, hardware setup, dashboard, and camera config
— to a `.glider` file with **File → Save** (`Cmd/Ctrl+S`). Reopen it later with
**File → Open**, or share the file with a colleague. A `.glider` file is plain
JSON under the hood, so it's easy to version-control and diff.

## Where to go next

- [Devices & Hardware](devices.md) — make your hardware nodes actually do
  something.
- [Functions](functions.md) — reuse a chunk of graph without copying it.
- [Node Catalog](../reference/nodes.md) — every node type and its ports.
