# Building Experiments

This section is the heart of GLIDER: how you turn an idea for an experiment
into a working, runnable design. You do this by wiring together a **node
graph** — a visual flowchart where each box (a *node*) does one small job, and
the wires between them decide what happens, and in what order.

You don't write any code. You drag nodes onto a canvas, connect them, bind the
hardware ones to real devices, and press **Start**.

## What's in this section

<div class="grid cards" markdown>

-   :material-graph: **[The Node Graph](node-graph.md)**

    Add nodes, wire execution and data ports, and build a flow that runs from
    `StartExperiment` to `EndExperiment`.

-   :material-connection: **[Devices & Hardware](devices.md)**

    Add boards and devices, bind a device to a node, and understand how a
    value's range is decided.

-   :material-rabbit: **[Subjects & Lab Vocabulary](subjects.md)**

    Record the animals in an experiment, and define the treatment groups,
    strains, solutions and routes your lab uses so they are spelled the same
    way every time.

-   :material-function-variant: **[Functions](functions.md)**

    Package a reusable chunk of your graph as a named function you can call
    again and again.

-   :material-puzzle: **[Custom Devices & Plugins](plugins.md)**

    Teach GLIDER about hardware it doesn't ship with — either with the no-code
    builder or a Python plugin.

</div>

## How the pieces fit together

A GLIDER experiment has two halves that you build side by side:

1. **The graph** — the logic of *what happens*. Nodes fire one after another
   along **execution** wires, and pass numbers and readings to each other along
   **data** wires.
2. **The hardware** — the boards and devices that the graph acts on. A node
   like `Output` doesn't know anything about pins on its own; it becomes
   meaningful only when you **bind** it to a device.

!!! tip "New to GLIDER?"
    If you haven't built anything yet, start with
    [Your First Experiment](../getting-started/first-experiment.md) for a
    guided end-to-end walkthrough, then come back here to go deeper. The
    concepts behind it all are collected in
    [How GLIDER Works](../getting-started/concepts.md).

When you want the full, exhaustive list of every node or device type — beyond
the ones covered here — see the [Node Catalog](../reference/nodes.md) and the
[Device Catalog](../reference/devices.md).
