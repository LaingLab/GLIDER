# Your First Experiment

Let's build a tiny but complete experiment: **blink an LED once a second**, then
run it. It touches every part of the workflow — adding hardware, building a
graph, and pressing Start — in about ten minutes.

!!! note "No hardware handy?"
    You can still follow along. Build the graph exactly as described; when you
    press Start, the `Output` nodes will simply report that no device is
    attached. Swap in a real LED whenever you like.

## What you'll build

```
StartExperiment → Loop ─▶ Output (ON) → Delay (0.5s) → Output (OFF) → Delay (0.5s)
                                                                          ↓
                                                                    (loops back)
```

When the loop finishes its repetitions, control returns and the experiment ends.

## 1. Open GLIDER

Launch the app (see [Installation](installation.md)). You'll land in **desktop
mode**: a canvas in the middle, a **Node Library** to add nodes, a **Properties**
panel for the selected node, and the **File / Edit / Experiment / View / Hardware
/ Run / Tools / Help** menu bar.

Start fresh with **File → New**.

## 2. Connect a board and add an LED

Open the **Hardware** panel and add your controller — an **Arduino** over USB, a
**Raspberry Pi**, or a **Bluetooth** device — then connect to it. (Full details:
[Devices & Hardware](../building/devices.md).)

Add a **Digital Output** device for your LED and note the pin it's on (say, pin
13, the Arduino's built-in LED). Give it a friendly name like `led`.

!!! tip "Connecting later"
    You don't have to connect hardware before building the graph. Design first,
    connect when you're ready — GLIDER will bind your nodes to devices when you
    press Start.

## 3. Lay down Start and End

From the **Node Library**, add a **StartExperiment** node and an
**EndExperiment** node to the canvas. Every experiment runs from Start to End
along **execution wires** — if you're fuzzy on exec vs. data wires, see
[How GLIDER Works](concepts.md).

## 4. Add a loop and the blink steps

Add these nodes from the library:

| Node | Role |
|---|---|
| **Loop** | Repeats the blink a set number of times |
| **Output** ×2 | One to turn the LED on, one to turn it off |
| **Delay** ×2 | Hold the LED on, then off, for half a second each |

Select each node and configure it in the **Properties** panel:

- **Loop** — set the number of repetitions (e.g. `10`).
- First **Output** — bind it to your `led` device and set its value to **ON**
  (HIGH).
- First **Delay** — set the duration to `0.5` seconds.
- Second **Output** — bind it to `led`, value **OFF** (LOW).
- Second **Delay** — `0.5` seconds.

!!! note "Binding a node to a device"
    Hardware nodes like `Output` have a **device** selector in their Properties.
    Pick your `led` there so the node knows what to drive. A digital output
    offers **HIGH/LOW**; a PWM output would give you a slider instead — see
    [Devices & Hardware](../building/devices.md).

## 5. Wire it up

Drag from one node's **exec output** port to the next node's **exec input** port,
in order:

1. `StartExperiment` → `Loop`
2. `Loop`'s body → `Output (ON)` → `Delay (0.5s)` → `Output (OFF)` → `Delay (0.5s)`
3. `Loop`'s "done" → `EndExperiment`

The loop drives its body once per repetition; when it's out of repetitions it
follows its "done" exec output to `EndExperiment`.

!!! warning "Exec wires make it run"
    Connect the **exec** ports (the "go" pulse), not just data ports. If Start
    does nothing, an exec wire is missing somewhere between Start and End.

## 6. Press Start

Open the **Run** menu and choose **Start** (or use the Start control in the
toolbar). The LED should blink ten times, half a second on and half a second
off. Use **Run → Stop** to end early.

!!! tip "Emergency stop"
    There's an emergency stop bound to ++ctrl+shift+escape++ that immediately
    drives devices to a safe state — handy while testing new hardware.

## 7. Save your work

**File → Save** writes everything — the graph, your `led` device, and the
experiment's metadata — into a single `.glider` file. Reopen it later, share it,
or run it on a touchscreen [Runner](../runner/index.md).

## Where to go next

<div class="grid cards" markdown>

-   :material-graph-outline: **[The Node Graph](../building/node-graph.md)**

    Loops, sequences, timers, logic, and math — the full toolkit.

-   :material-chip: **[Devices & Hardware](../building/devices.md)**

    PWM, servos, analog inputs, I²C, and Bluetooth devices.

-   :material-camera: **[Camera & Recording](../camera-behavior/camera.md)**

    Record synchronized video and data while your experiment runs.

-   :material-function: **[Functions](../building/functions.md)**

    Package a routine (like this blink) so you can reuse and one-tap it.

</div>
