# Node Catalog

This page is the complete catalog of node **types** GLIDER supports, grouped by
what they do.

!!! note "Library palette vs. the full catalog"
    The **Node Library** palette you drag from is a focused set for building
    common experiments — **Start/End Experiment**, **Delay**, and **Timer** (Flow),
    **Start/End Function** (Functions), **Loop** and **Wait For Input**
    (Control), **Output** and **Input** (I/O), plus **Audio**, **Video**, and
    dynamic sections for your custom devices, functions, and camera zones. The
    remaining node types below (math, comparison, PID, and the interface
    widgets) are part of experiments loaded from `.glider` files. Hardware is
    driven through the **Output**/**Input** nodes and the generic **Device
    Action**/**Device Read** nodes rather than a node per device.

!!! tip "Ports at a glance"
    Most nodes have an **exec** input and output (the "go" pulse that decides
    order) plus some **data** ports (values in and out). See
    [How GLIDER Works](../getting-started/concepts.md) for the exec-vs-data
    distinction.

## Flow control

The backbone of every experiment — where it starts, ends, repeats, and waits.

| Node | What it does |
|---|---|
| **StartExperiment** | Entry point — begins the experiment flow. |
| **EndExperiment** | Exit point — ends the experiment. |
| **Loop** | Repeats its body a set number of times, then continues. |
| **Sequence** | Fires its outputs one after another, in order. |
| **Delay** | Waits for a set duration, then continues. |
| **Timer** | Triggers execution at a regular interval. |
| **WaitForInput** | Pauses until an input condition is met — a digital signal, a threshold, or a target number of encoder revolutions/counts. |
| **StartFunction** | Entry point for a reusable [function](../building/functions.md). |
| **EndFunction** | Exit point for a function. |
| **FunctionCall** | Runs a defined function from elsewhere in the graph. |

## Hardware

Drive and read physical [devices](devices.md). Most bind to a device in their
Properties panel.

| Node | What it does |
|---|---|
| **Output** | Writes to a bound device — digital HIGH/LOW or a PWM value. |
| **Input** | Reads from a bound device (digital or analog). |
| **Digital Write** | Writes HIGH or LOW to a digital output pin. |
| **Digital Read** | Reads HIGH or LOW from a digital input pin. |
| **Analog Read** | Reads an analog value from a device. |
| **PWM Write** | Writes a PWM value (range set by the device). |
| **Device Action** | Executes a named action on a device (e.g. a Bluetooth `write`). |
| **Device Read** | Reads a value from a device. |
| **MotorGovernor** | Controls a motor governor — up, down, or stop. |

## Logic & math

Compute, compare, and shape values that flow along data wires.

| Node | What it does |
|---|---|
| **Add** | `A + B` |
| **Subtract** | `A − B` |
| **Multiply** | `A × B` |
| **Divide** | `A ÷ B` |
| **Clamp** | Constrains a value between a min and max. |
| **Map Range** | Rescales a value from an input range to an output range. |
| **In Range** | Tests whether a value falls within a min/max range. |
| **Threshold** | Tests whether a value crosses a threshold, with optional hysteresis. |
| **PID Controller** | Proportional–Integral–Derivative control loop. |
| **Toggle** | Flips its state each time it's triggered. |

## Interface

On-screen controls and readouts for the dashboard.

| Node | What it does |
|---|---|
| **Button** | A clickable button that triggers execution. |
| **Slider** | A slider for continuous value input. |
| **Toggle Switch** | An on/off switch. |
| **Numeric Input** | A numeric field with virtual-keypad support (touch-friendly). |
| **Chart** | A real-time chart of values. |
| **Gauge** | Shows a value as a gauge/meter. |
| **Label** | Displays a value as text. |
| **LED Indicator** | An on/off indicator light. |
| **Zone Input** | Monitors camera-zone occupancy and triggers on enter/exit. |
| **VideoPlayback** | Plays a video clip. |
| **AudioPlayback** | Plays an audio clip. |

!!! note "New device types add nodes automatically"
    Custom and [plugin devices](../building/plugins.md) expose their actions
    through the generic **Device Action** and **Device Read** nodes — you don't
    need a bespoke node per device.
