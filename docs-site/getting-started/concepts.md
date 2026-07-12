# How GLIDER Works

A handful of ideas explain almost everything in GLIDER. Once these click, the
rest of the app is just detail.

## Experiments are graphs

An experiment in GLIDER is a **node graph**: boxes (nodes) connected by wires.
Each node does one small thing — wait a second, write a pin HIGH, read a sensor,
check whether a value crossed a threshold. Wiring them together describes what
your experiment does, and in what order.

You build this graph on a canvas in **desktop mode**, using:

- the **Node Library** to find and add nodes,
- the **canvas** to place and wire them,
- the **Properties** panel to configure the selected node.

## Two kinds of wires: exec and data

Every connection is one of two types, and telling them apart is the single most
important thing to understand:

- **Execution (exec) wires** decide *order*. They carry a "go now" pulse from one
  node to the next. Your experiment runs by following exec wires.
- **Data wires** carry *values* — a number, a reading, an on/off state — from a
  node's output into another node's input.

A node typically has an exec input, an exec output, and some data ports. Think of
exec wires as the railway the train runs on, and data wires as the cargo it
carries.

!!! tip "The golden rule"
    If nothing happens when you press Start, check that your nodes are joined by
    **exec** wires all the way from `StartExperiment` to `EndExperiment`. Data
    wires alone won't make anything run.

## Start and End

Every experiment begins at a **StartExperiment** node and finishes at an
**EndExperiment** node. Pressing **Run → Start** fires the exec output of
`StartExperiment`, which flows through your graph until it reaches
`EndExperiment`.

In between, you might [loop](../reference/nodes.md), delay, read sensors, and
write to devices — in whatever structure your experiment needs.

## Devices and boards

GLIDER controls the physical world through **devices** attached to a **board**:

- A **board** is the controller GLIDER talks to — an **Arduino** (over USB), a
  **Raspberry Pi**'s GPIO, or a **Bluetooth Low Energy** peripheral.
- A **device** is a specific thing on that board — a digital output (an LED, a
  valve), a PWM output (LED brightness, motor speed), a servo, an analog input
  (a potentiometer, a light sensor), an I²C sensor, and so on.

You add and configure devices in the **Hardware** panel, then **bind** a device
to a hardware node (like `Output` or `Input`) so that node knows what to drive or
read. See [Devices & Hardware](../building/devices.md).

!!! note "Each device knows its own limits"
    A device declares the valid range for each of its actions — for example, a
    PWM output on an 8-bit board accepts `0–255`, while a 12-bit board accepts
    `0–4095`. GLIDER uses that range everywhere: the sliders you get, the values
    a node will write, and the clamping that keeps an out-of-range value from
    reaching the hardware.

## Sessions and `.glider` files

Your whole experiment — the graph, the devices, the dashboard, camera and zone
settings, and metadata (protocol, experimenter, subjects, notes) — lives in a
**session**. Saving writes it to a single `.glider` file (it's JSON under the
hood). Open that file later to pick up exactly where you left off, share it with
a collaborator, or run it on a touchscreen [Runner](../runner/index.md).

## Recording

While an experiment runs, GLIDER can record:

- **Video** from one or more cameras (MP4),
- a **frame-aligned data log** — a CSV of device state that shares a `frame`
  column with the tracking data, so it lines up with the video frame for frame,
- an **event log** of the moments that matter (a pin going HIGH, a zone entry).

See [Camera & Recording](../camera-behavior/camera.md).

## Desktop mode vs Runner mode

- **Desktop mode** is where you *design* — the full graph editor, docks, and
  menus.
- **Runner mode** is where you *operate* — a touch-first, four-tab kiosk
  (**Setup, Run, Manual, Camera**) meant for a bench touchscreen or a Raspberry
  Pi. It runs the same `.glider` experiments, adds big manual device controls,
  and lets you trigger saved [functions](../building/functions.md) with one tap.

Design once, run anywhere. Ready to build something? Continue to
[Your First Experiment](first-experiment.md).
