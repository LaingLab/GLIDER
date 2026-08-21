# Rehearsing a closed loop from a recording

A closed-loop experiment — stimulate *because* the animal did something — has a
lot of links: pose inference, the behavior model, the node that watches for a
behavior, the wiring on the canvas, the device binding, and the hardware itself.
Any one of them can be wrong in a way that looks fine until an animal is in the
box.

**Rehearsal mode plays a recording through the live path instead of the camera.**
The classifier runs, the nodes fire, and the hardware is driven for real — on
footage where you already know what the animal did.

!!! warning "The hardware really fires"
    This is the point: a rehearsal that faked the output would not tell you the
    stimulator works. Don't rehearse with an animal connected.

## Running one

1. Load your pose model and behavior model in **Camera → Live behavior**, set the
   keypoint names, and press **Start**. The classifier has to be running — a
   rehearsal with nothing listening does nothing, and the panel says so.
2. Build the flow you'd run for real: a **Behavior Input** node watching the
   behavior you care about, wired to whatever it triggers.
3. Press **Rehearse from video…** and pick a clip.

The status line reports progress, and at the end it tells you the worst lag.

## Real time or as fast as possible

Both classify the recording **identically**. Feature values come from
`compute_features`, which uses unit frame spacing and never reads fps — so the
playback rate cannot change what the model sees. The two modes exist because
they answer different questions:

| Mode | Answers |
| --- | --- |
| **Real time** | Does inference keep up on this machine, and how long does a stimulus take to arrive after a behavior starts? |
| **As fast as possible** | Is any of this wired up correctly? Same answer, sooner. |

Use *as fast as possible* while you're still fixing wiring, and *real time* for
the run you actually believe.

## Reading the result

**"kept up with real time"** — inference is fast enough on this machine for this
footage. The rig should behave the same live.

**"worst lag N ms — inference did not keep up"** — the model is slower than the
frame rate. A stimulus will arrive late on a live animal by roughly that margin,
on top of the confirmation delay from the Behavior Input node's `min_frames`
setting (at 30 fps, 5 frames is about 167 ms). Either is fine if it is small
against the behavior you are studying, and neither is fine if it isn't.

Frames are **never skipped**, whatever happens. The live feature extractor
computes velocity and acceleration over unit frame spacing, so a dropped frame
doubles the apparent displacement across the gap and inflates the kinematics the
model keys on. A rehearsal that dropped frames to keep up would report confident,
wrong behavior — so it reports lag instead.

## What it does and doesn't prove

**Does:** the model loads and classifies; `min_frames` is tuned for your actual
footage; node wiring; the device binds; the peripheral connects; the commands
sent are the ones you meant; end-to-end latency.

**Doesn't:** camera setup, lighting, or whether inference keeps up on your
*camera's* resolution and frame rate — unless the clip was recorded on the same
camera at the same settings. It's worth using footage from the rig you're about
to run.

## Getting a clip

Any recording the model was trained to handle. A session recorded by GLIDER's own
video recorder is ideal, because it matches the camera and settings you'll use.
Pick one where the behavior you're triggering on actually occurs — a rehearsal on
footage with no freezing tells you nothing about a freeze trigger.
