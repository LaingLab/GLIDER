# Functions

A **function** is a reusable piece of your graph — a named sequence of nodes you
build once and then call from anywhere in your experiment. If you find yourself
copying the same run of nodes over and over (dispense a reward, flash a cue, home
a motor), turn it into a function instead.

## The three pieces

Functions are built from nodes you already know from the library's **Functions**
section, plus a call button:

| Node | Role |
|---|---|
| **`StartFunction`** | Marks the beginning of a function. You give it a name in Properties. |
| **`EndFunction`** | Marks the end. When the flow reaches it, the function is done and control returns to whoever called it. |
| **`FunctionCall`** | Runs a defined function. You don't drag this from the main library — GLIDER creates a call button for each function you define (see below). |

## Defining a function

1. Drag a **`StartFunction`** node onto the canvas.
2. Select it, and in **Properties** set its **name** (for example, `Dispense`).
   This is the name you'll see and call it by.
3. Build the body: wire the `StartFunction`'s execution output through whatever
   nodes the function should perform — outputs, delays, waits, and so on.
4. End the chain with an **`EndFunction`** node.

```text
StartFunction "Dispense" ──▶ Output (open valve) ──▶ Delay (500 ms) ──▶ Output (close valve) ──▶ EndFunction
```

That's a complete function. It lives right in your graph alongside the main flow.

!!! note "A function is defined by its wiring"
    GLIDER recognizes a function when a `StartFunction` reaches an `EndFunction`
    along the execution wires. It follows execution flow to find the end — a plain
    data wire doesn't count as the function's finish line.

## Calling a function

Once a `StartFunction → EndFunction` pair is complete, GLIDER detects it and adds
a **call button** for it in the **Graph Functions** section of the Node Library,
labeled with the function's name. (Until you've defined one, that section shows a
reminder to define functions with `StartFunction → EndFunction`.)

To call the function, add its button to the graph like any other node — it drops
in a **`FunctionCall`** node. Wire that node into your main flow wherever you want
the function to run:

```text
StartExperiment ──▶ FunctionCall "Dispense" ──▶ FunctionCall "Dispense" ──▶ EndExperiment
```

When a `FunctionCall` fires, it runs the whole function body — every node between
`StartFunction` and `EndFunction` — and **waits until the `EndFunction` is
reached** before continuing to whatever comes after the call. This means the
nodes downstream of your call don't start until the function has actually
finished.

You can drop as many calls to the same function as you like; they all run the one
definition, so editing the body updates every call at once.

## Good to know

- **One run at a time.** A given function runs to completion before another call
  to the *same* function starts, so two callers can't tangle each other's steps.
  Different functions can still run independently.
- **It won't hang forever.** If a function's body gets stuck — a node that stops
  the chain, or hardware that disappears mid-run — the call gives up after a
  timeout and hands control back rather than freezing your experiment.
- **Names are how you tell them apart.** Give each function a distinct, memorable
  name in the `StartFunction` properties; that's the label on its call button.

!!! tip "When to use a function vs. a loop"
    Use a [`Loop`](node-graph.md#useful-flow-control-nodes) when you want to
    *repeat* the same steps in place. Use a **function** when you want to *reuse*
    the same steps in several different places in your flow — and keep them
    defined in exactly one spot.

## Where to go next

- [The Node Graph](node-graph.md) — the flow, ports, and wiring functions plug
  into.
- [Devices & Hardware](devices.md) — bind the hardware nodes inside your
  function.
- [Node Catalog](../reference/nodes.md) — reference for every node type.
