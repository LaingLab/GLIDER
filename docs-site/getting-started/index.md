# Getting Started

Everything you need to go from nothing to a running experiment.

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    Install GLIDER on macOS, Windows, Linux, or a Raspberry Pi — either from a
    prebuilt app or from source.

-   :material-flask: **[Your First Experiment](first-experiment.md)**

    Build a simple blink-and-record experiment end to end, then run it.

-   :material-lightbulb-on: **[How GLIDER Works](concepts.md)**

    The handful of concepts — nodes, the exec flow, devices, sessions — that
    make everything else click.

</div>

## The short version

1. **Install** GLIDER and, if you plan to record video, FFmpeg.
2. **Connect hardware** (an Arduino, a Pi, or a Bluetooth device) — or skip it
   and design offline.
3. **Build a graph**: drop a `StartExperiment` node, wire it through some
   actions, and finish at `EndExperiment`.
4. **Press Start.** Watch it run, and record video and data as it goes.
5. **Save** your work as a `.glider` file to reopen, share, or run on a
   touchscreen [Runner](../runner/index.md).

!!! note "How much hardware do I need?"
    None, to start. You can build and even run a graph with no hardware
    connected — nodes that would drive a device simply report that nothing is
    attached. Add real hardware when you're ready.
