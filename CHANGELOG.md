# Changelog

All notable changes to GLIDER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **BLE peripherals report their own connection state**, separately from the
  host adapter's. A peripheral that goes out of range, loses power, or is
  claimed by another central shows **Disconnected** in the Hardware panel and
  gets its own dot on the status strip, beside the boards. GLIDER reconnects it
  on its own with bounded backoff — 5, 10, 20, 40 and 60 seconds, up to twelve
  attempts — before giving up into **Error**. A run is not paused while this
  happens; a warning notification says so and names whether GLIDER is still
  retrying or has stopped.
- **A Maimu writes `off` when a dropped link comes back.** Its firmware runs a
  pulse train on its own, so a link that died mid-train left it stimulating
  with nothing attached to stop it; the reconnect now puts it in a known state
  before anything else touches the device.
- **Devices can declare an action's arguments** (`ACTION_ARGS_SCHEMA`),
  rendered as labelled number fields beside the action's button in both the
  Builder's Device Control panel and the Runner's manual controls. This is
  what makes Maimu's `pulse` — a period and a duration — pressable from
  either.

### Fixed

- A BLE device with notifications enabled lost its subscription on the first
  reconnect and stayed silent for the rest of the session — `get_state()`
  returning `None` — with nothing logged.
- The Runner's manual controls called an argument-taking action with no
  arguments, raising `TypeError` on every press.
- The Hardware panel and the Device Control panel reported whether a device
  had ever been initialized, rather than its current link, so a peripheral
  that walked out of range kept reading "Ready".
- `plugins/glider-maimu/tests` was missing from `testpaths`, so a bare
  `pytest` never collected that plugin's suite.

### Added

- **Rehearse a closed loop from a recording.** **Camera → Live behavior → Rehearse from video…** plays a clip through the live path instead of the camera: the classifier runs, the nodes fire, and the hardware is driven for real, on footage where you already know what the animal did. Frames are never skipped — the live feature extractor uses unit frame spacing, so a dropped frame inflates exactly the kinematics the model keys on — so a run that cannot keep up reports its worst lag instead, which is also the number that says whether the rig will hold up live. Real-time and as-fast-as-possible modes classify identically (`compute_features` never reads fps); they differ only in whether they can answer "does inference keep up?".
- **Maimu moves out of core into an installable plugin, `glider-maimu`.** The device and node are unchanged; what changed is that they are no longer bundled. One lab's stimulator does not belong in every install, and moving it made it the first real consumer of the plugin node extension points — which is the only way to find out whether they are good enough. Two gaps surfaced doing it and are fixed here: a plugin node could not declare its canvas ports (the editor now reads them off the node's own `NodeDefinition` instead of falling back to a generic one-in-one-out, which also fixes every built-in node missing from the hand-written table), and a plugin node could not be bound to a device (a node now declares `REQUIRES_DEVICE`, which `HardwareNode` sets, instead of the editor matching a hardcoded list of type names).
- **Maimu BLE stimulator** — a `Maimu` device type with the peripheral's GATT layout built in (**Add Device → Maimu → Scan**, no UUIDs to paste), and a **Maimu** node in the library's I/O section offering Mode (On / Off / Pulse) with a period and duration, instead of a Device Action node writing `"500,10"` by hand. Unlike the generic BLE devices, its shutdown writes `off` before disconnecting — the firmware runs a pulse autonomously, so an emergency stop that only dropped the link would leave the device stimulating.
- **DeepLabCut and SLEAP pose models** — a `PoseBackend` seam behind the three places GLIDER loaded a pose net lets DeepLabCut single-animal and SLEAP single-instance models drive live camera inference, offline classification, and batch runs. They run through `onnxruntime` (new `glider[pose-onnx]` extra), after a one-time export performed in your own DeepLabCut or SLEAP environment with `tools/export_pose_onnx.py` — GLIDER never imports `deeplabcut` or `sleap`, which is what keeps it installable on Python 3.11-3.13 and on the Pi. Ultralytics YOLO `.pt` weights are unchanged: that path is a pass-through over the same code as before. Multi-animal architectures, including SLEAP top-down and bottom-up, are rejected by name rather than silently scoring one arbitrary animal.
- **Drag-and-drop on the camera panel** — a pose model (folder, `.pt`, or `.onnx`), a behavior model (`.pkl`), or a video can be dropped anywhere on the panel and is routed to the right slot by type. One drop can fill several slots; a drop while live behavior is running is refused rather than half-applied.
- **Keypoint names are read from the model** — a dropped DeepLabCut or SLEAP folder fills the keypoint-names field from its own config, in training order, and locks it. Order is not cosmetic: with `auto_angles`, angle columns are named after the keypoints at given indices, so a re-ordering yields all-NaN columns and uniformly blank predictions with nothing raising. The live classifier now also rejects names whose *order* disagrees with the behavior model, not just their count.
- **A decode parity harness** (`pytest -m pose_parity`) comparing our DeepLabCut/SLEAP decoding against output from the source tools. It cannot run in CI — it needs real weights plus the reference CSV those tools produced — and skips unless `GLIDER_POSE_FIXTURES` names a fixture directory. The rest of the pose suite proves the decode maths is self-consistent, not that it agrees with DeepLabCut and SLEAP; only this closes that gap.
- Documentation: "Pose models: YOLO, DeepLabCut, and SLEAP", covering the export step, the folder layout, and why the export happens in your environment rather than GLIDER's.

### Fixed

- `gui/pose_batch/window.py` imported `glider.vision.pose.model_meta` inside a `try`/`except ImportError`, but that module was never committed on any branch — the `except` was the only branch that ever ran, so the batch window's keypoint-name reader always returned `None`. The function now exists and the seam is gone.

- **Harp device support, as an installable plugin** — `glider-harp`, a separate plugin distribution living in `plugins/glider-harp`, adds a Harp transport board and a schema-driven Harp device: binary frame decoding with resync after line noise, a register cache, and a background reader draining the port. Harp stays a plugin; GLIDER keeps the master clock.
- Core support the Harp work needed, usable by any driver: a device can declare **multiple state columns** and the `DataRecorder` expands them into CSV headers and per-row values; a device that degrades mid-recording writes a **warning row** into the CSV instead of failing silently; a pulled cable is survived and noted in the recording rather than ending it.
- **PyPI release workflow for glider-harp** — `.github/workflows/release-glider-harp.yml` publishes the plugin to PyPI via Trusted Publishing (OIDC — no API token stored anywhere), driven by namespaced tags (`glider-harp-v*`) with a guard that refuses a tag disagreeing with the package version. `glider-harp` 0.1.0 is published on PyPI.
- **Plugin catalogue and installer** — **Tools → Plugins…** opens a non-modal window listing a curated catalogue of published plugins, with search and All / Installed / Available filters. Install runs `pip` — or `uv pip install --python …` when the environment has no pip — as a subprocess targeting GLIDER's own interpreter, and streams its output onto the row; failures render inline on the row that caused them, with pip's message verbatim, never in a modal. A newly installed plugin loads without a restart; upgrading one already in use does not, and the window says so. Rows offer Disable / Enable / Reload — there is no uninstall.
- A plugin that installs but then fails to import reads as **Not loaded**, with the import error on its own row, rather than as a green "Enabled". pip succeeding and the plugin working are different things, and the load error was previously recorded and then discarded.
- A row whose catalogue entry has an unreadable `glider_requires` says so instead of taking the window down. The index arrives over the network and nothing validates its fields, so `"1.0"` where `">=1.0"` was meant is data, not a crash.
- A failure that stops the Plugins window opening at all now reaches the user as a message box, instead of a garbage-collection-time warning behind a menu item that appeared to do nothing. Opening the window twice reuses the one that is already up.
- Plugin catalogue resolution with three sources in order — the live index, the last cached copy, then the copy bundled in the release. The window's footer permanently names which one won and how old it is: the catalogue is curated, installing runs arbitrary code with GLIDER's privileges, and that footer is where the trust question is answered.
- Compatibility gate: a plugin declaring a `glider_requires` specifier the running version does not satisfy is shown greyed out with both versions named, rather than offering an Install that pip would then decline.
- Documentation: "Installing plugins from the catalogue" in the Custom Devices & Plugins guide.

### Changed

- The Maimu node's Mode / Period / Duration are now rendered from a declared `PROPERTIES_SCHEMA` rather than a hardcoded branch in the editor. One deliberate loss: the schema form has no notion of one field depending on another, so Period and Duration no longer grey out outside Pulse mode — that cue moved into their help text.
- Entry points of the `module:Class` shape now register the class, instead of calling it and discarding the result.
- Desktop theme: styling for the Plugins window — card rows, the seven state pills, filter chips, the pip transcript and the indeterminate progress bar — added to `desktop.qss`. The plugin card and dialog set no colours from Python at all, so the whole surface stays restylable from the stylesheet.
- The Plugins window shows the version pip actually installed, not the one the catalogue advertised, and no longer hides a row at the moment its install completes when a filter is active.
- **Reload** says when a package registers no entry points, rather than reporting "Reloaded." for having reloaded nothing.
- Documentation now describes what **Disable** really does: it skips the plugin at the *next* load, does not unregister anything already registered, and is not saved across restarts. The previous text claimed the opposite. Disable's behaviour is unchanged; only the claims about it are.

### Fixed

- **One `.glider` format.** GLIDER had two readers/writers that could not read each other's files: `ExperimentSession` (File > Save / File > Open, and every file in `examples/`) and `ExperimentSerializer` (`save_experiment`/`load_experiment`, and therefore `glider --file`). Passing `glider --file` any file the GUI had saved — which is every file that exists — raised `SchemaValidationError` uncaught during startup. The session format is now the one format the app reads and writes; files in the old serializer format are detected and converted on load, and rewritten in the current format when next saved.
- **Camera, zone and manual-control settings survive `save_experiment`/`load_experiment`.** The serializer schema had no `camera`, `zones` or `manual_controls` block, so a round trip through it silently dropped all three.
- **CV settings actually persist now.** The vision block (detection backend, model path, keypoint names) was only ever written by `save_experiment`, which nothing in the GUI calls — so File > Save never recorded the operator's model choice and reopening lost it. The session format carries a `vision` block, and File > Save writes it. A file without one leaves the live CV configuration alone rather than stomping it with defaults, and a backend that degraded at runtime for missing weights still saves the configured choice, not the degradation.
- `save_experiment` adopts flow-engine nodes and connections the session model does not have, so a graph built directly on the engine rather than through the editor still saves. The adoption is additive: it can never empty the flow of a session whose engine has not been populated yet.
- Opening an experiment no longer stacks its boards and devices on top of the previously open one's, and no longer leaves the session in INITIALIZING — a state in which it would then refuse to start.
- **Hardware-node device bindings survive a `.glider` round-trip through the serialization layer.** `ExperimentSerializer` never recorded which device a node was bound to — `NodeSchema` had no field for it, the save path only wrote `get_state()` (which does not include the device), and the load path passed no `device_id` to `create_node`. So a file written by `save_experiment`, or opened with `glider --file …`, came back with every Output / Input / Device Action node unbound. Schema 1.2.0 adds an optional `device_id` on a node, resolved to the same key the device is saved under; it is omitted when there is no binding, and older readers ignore it. (The GUI's File > Save / File > Open path goes through `ExperimentSession`, which already persisted bindings — this brings the two into agreement.)
- Clearing a node's device in the properties panel now unbinds the runtime node instead of leaving the previous device attached. Combined with the above, a stale binding would otherwise have been written straight back into the file on the next save.
- A flow engine asked to restore a node's device binding without a hardware manager says so in the log, rather than leaving the node silently unbound.
- A device whose settings include a key named `name` — every BLE device, which uses it for the advertised local name — could not be added or loaded at all: `add_device`/`add_device_multi_pin` took the display name as `name=` and splatted the rest, so the two collided and raised `TypeError: got multiple values for keyword argument 'name'`. Both now take a `settings=` dict; `**kwargs` still works and still wins on a conflict. This affected the existing `BLE` (read/notify/write) device type as well as the new Maimu.
- The plugin installer now works on the documented setup. `uv venv` + `uv sync` creates an environment **without pip**, and the installer ran `sys.executable -m pip`, which failed immediately. It now uses pip when the interpreter has it and falls back to `uv pip install --python <GLIDER's interpreter>` otherwise — the `--python` is what keeps the plugin in GLIDER's environment. Neither being available is reported on the row, not raised.
- A catalogue entry can carry **direct `requirements`** — PEP 508 strings appended to the install command. uv honours a pre-release marker only on a *direct* requirement, so this is what lets `glider-harp` (whose dependency pins `harp-protocol>=0.5.0rc1`, pre-release-only upstream) install through the catalogue in uv-built environments; under pip the field is a no-op.

## [1.0.0] — 2026-08-07

Initial public release. This is the version described in Bradham et al.
(2026) and the release that manuscript's results were produced from.

### Major features

- Visual flow programming for experimental protocols.
- Direct GPIO / serial / I²C / camera control across Arduino, Raspberry Pi, and PC.
- Multi-camera capture with YOLO + ByteTrack object tracking and zone enter/exit events.
- UCLA Miniscope V4 integration (LED + electrowetting-lens focus control).
- Touchscreen-optimised "runner mode" for Pi kiosk deployment.
- One-file `.glider` JSON experiment storage with embedded metadata.
- Live behavior-state classification (resting / walking / darting / freezing).
- Synchronous CSV logging of every trial event and per-frame tracking position.
- Audio playback and recording nodes.
- Plugin system for user-supplied node and driver extensions.
- In-app updater (GitHub Releases).
- Atomic `.glider` save via `tempfile.mkstemp` + `os.replace`.
- Pi-gen image build pipeline (`packaging/pi/`) for turnkey kiosk SD-card images.
- Inno Setup installer pipeline (`packaging/windows/`) for Windows distribution.
- **Live behavior classifier** wired into the camera panel: `LiveBehaviorClassifier` + `BehaviorInferenceWorker`, with a live == offline feature-parity test guard.
- **Interactive onboarding tour** (`gui/onboarding/`): spotlight overlay + callout card walking new users through the Node Library, panel tabs, Hardware, canvas, Properties, Camera, and Run. Offered on first launch (Take the Tour / Skip) in Builder mode; replayable anytime from Help → Replay Tutorial.
- **GPU/accelerator diagnostics**: `glider --gpu-check` CLI flag and Tools → GPU / Device Check menu action, reporting CUDA/MPS/CPU availability and the device inference will use.
- **macOS packaging** (`packaging/macos/`): PyInstaller spec producing `GLIDER.app` (with camera/microphone usage descriptions) and a one-command local `build.sh` that packages `GLIDER-<version>-<arch>.dmg`. Unsigned for now; per-arch (arm64/x86_64).
- Local Windows installer build script (`packaging/windows/build.ps1`) and local Pi image build script (`packaging/pi/build.sh`) — one-command equivalents of the CI workflows for building on your own machines.

### Added

- `LICENSE` file (MIT).
- `CITATION.cff` for academic citation.
- `CHANGELOG.md` (this file).
- `CONTRIBUTING.md` covering dev setup, code style, hardware contribution, and `.glider` file-format change procedure.
- `SECURITY.md` for responsible disclosure of safety-relevant bugs (e.g., Pi sudoers rule, hardware-safety regressions).
- Comprehensive `README.md` covering what GLIDER does, supported hardware, install across all four platforms, CLI options, citation, troubleshooting, and known limitations.
- `.github/workflows/ci.yml` — test + lint + type-check + PyInstaller smoke build on every push and PR, across Python 3.11/3.12/3.13 on Ubuntu/Windows/macOS. *(correction: at release the test matrix ran Python 3.11 only across the three OSes, the PyInstaller smoke build had been removed, and mypy is advisory — it runs but does not fail CI)*
- Parametric test asserting every registered node class round-trips its `get_state()` through `set_state()` without loss (catches `property_names`-style silent state drops in CI).
- Parametric test asserting every exported node class registers with the `FlowEngine` (catches the "node in the library but engine can't instantiate it" class of bug).
- Round-trip test `tests/unit/serialization/test_serializer.py` covering save → load → save with at least one board, one device, one node, and one connection (catches the API-mismatch class of bug in the apply path).

### Changed

- **Apple Silicon (MPS) acceleration**: behavior-classifier inference and live YOLO detection now resolve CUDA → MPS → CPU instead of silently falling back to CPU on Macs. Training keeps its deterministic CUDA-or-CPU path.
- HiDPI scale-factor rounding changed from `PassThrough` to `RoundPreferFloor` so 1px borders stay crisp at scaled macOS resolutions and on external displays.
- Desktop theme: dock separators blend into the panel background, the node-editor canvas no longer draws a contrasting border, and tabbed-dock tabs stretch the full width of the tab bar.
- `diagnose()`/`format_gpu_info()` no longer advise reinstalling a CUDA wheel on macOS (where CUDA is not applicable); they report MPS as the platform accelerator instead.
- Python version policy unified across `pyproject.toml`, classifiers, `black` target, `ruff` target, `mypy` config, CI matrix, and README — all now **3.11 / 3.12 / 3.13**.
- `requires-python` tightened from `>=3.10,<3.14` to `>=3.11,<3.14` (the codebase uses `match` statements and `X | None` unions, which require 3.10+; we standardise on 3.11 as the floor since 3.10 is not CI-tested).
- `[tool.ruff] target-version = "py311"` added.
- `pyproject.toml` `[dev]` extras group consolidated: now uses `GLIDER[pc,vision,i2c]` self-reference rather than duplicating individual dependencies (no more drift when bumping a `[vision]` pin).
- `uv.lock` is now tracked (was gitignored). Reproducible installs for the application.
- `__main__.spec` (the auto-generated PyInstaller stub at repo root) deleted. `packaging/windows/glider.spec` is the canonical spec; new contributors building locally are no longer routed to the broken stub.
- Removed install one-liner curl-pipe-bash references from the README; the `install.sh` / `install.ps1` scripts they pointed at do not exist and the pattern is a supply-chain risk.
- **Runner UI replaced by a customizable Dashboard.** The four-tab Runner view was replaced by a customizable 2×2 quadrant Dashboard (`gui/dashboard/`) shared by both the desktop and Pi surfaces, with bidirectional Builder↔Dashboard switching. Each quadrant hosts a user-pickable panel (camera, hardware, run control, experiment info, etc.). *(correction: before release the Pi surface was returned to the four-tab RunnerShell, which fits the 480px touchscreen; the quadrant Dashboard is the desktop-only operator view)*
- ~2,500 LOC of dead `gui/` code purged (`runner/dashboard.py`, `runner/widget_factory.py`, `widgets/touch_widgets.py`, `widgets/device_card.py`, `controllers/hardware_controller.py`, `controllers/device_control_controller.py`, `panels/experiment_panel.py`). Active runner UI is the quadrant Dashboard in `gui/dashboard/`; active hardware UI is `panels/hardware_panel.py`; etc.
- CI now runs unit + integration tests with coverage, type-check (mypy), lint (ruff), format check (black), and a PyInstaller smoke build that asserts `dist/glider/glider --version` succeeds. *(correction: the PyInstaller smoke build was removed before release, and mypy is advisory — it does not fail CI)*

### Fixed

- **Serializer apply path was structurally broken.** `_apply_hardware_config` called `add_board(board_type=…)` (the real kwarg is `driver_type`), `_apply_flow_config` passed a class as the first positional to `create_node` (it expects a `node_id: str`), called the non-existent `flow_engine.connect(…)` (the real methods are `create_connection` and `connect_nodes`), and the save path iterated `flow_engine.connections.items()` (the underlying attribute is `_connections: list[dict]`, accessed via `get_connections()`). Every `.glider` save and every load crashed; `glider --file path.glider` was non-functional and the experiments in `examples/` could not be loaded. All four API mismatches are fixed and covered by a round-trip test.
- **Half the node library was unreachable from the editor.** Hardware nodes (DigitalWrite, DigitalRead, AnalogRead, PWMWrite, ServoWrite, DeviceAction, DeviceRead), interface widgets (Button, Toggle, Slider, NumericInput, Label, Gauge, Chart, LEDIndicator), math nodes (Add, Subtract, Multiply, Divide, MapRange, Clamp), comparison nodes (Threshold, InRange), Toggle (logic), and PID were defined and exported but never registered with the `FlowEngine` — the Builder library showed them but `create_node(type_str, …)` returned `None`. Each module now exposes a `register_*_nodes` function that is invoked from `GliderCore._register_builtin_nodes`, and a parametric test asserts the registry is complete.
- **`_extract_node_properties` always returned an empty dict.** It iterated `getattr(node, "property_names", [])`, but no node class defined `property_names`. Every node-local parameter (camera index, GPIO pin, threshold, ITI duration, PWM value) was silently dropped on every save. Now extracts via the per-node `get_state()` API.
- **`HardwareManager.emergency_stop` could hang on one stuck device.** Per-device `await device.shutdown()` and per-board `await board.emergency_stop()` are now wrapped in `asyncio.wait_for(..., timeout=DEVICE_IO_TIMEOUT_S)` and run via `asyncio.gather(..., return_exceptions=True)` so a single hung device cannot block the rest of the e-stop sequence.
- **`_handle_flow_complete` raced `stop_experiment` without taking `_experiment_lock`.** When the flow reached an `EndExperiment` node concurrently with the operator clicking STOP, two shutdown sequences ran against the same recorders and devices. `_handle_flow_complete` now acquires the same lock and short-circuits if the session is already in a stop state.
- **ERROR state was invisible to the operator.** `_on_core_state_change` only mutated a QSS `statusState` property and a status-bar string; `_on_core_error` emitted an `error_occurred` signal that had zero subscribers. After a partial-failure stop, operators could not distinguish "transient error" from "hardware output still driving." `error_occurred` is now wired through the existing `_notify_user` helper, and `_on_core_state_change` opens an explicit critical modal when entering ERROR.
- **`_run_async` swallowed coroutine exceptions silently.** The done-callback only discarded the task from the pending set; `task.exception()` was never inspected. Emergency-stop or hardware-action failures vanished into Python's "Task exception was never retrieved" message at GC time. Now uses `glider.core.async_utils.log_task_exception` so exceptions surface in the log and through `_notify_user`.
- **`closeEvent` busy-spun without sleep and raced shutdown with task cancellation.** Replaced the spin loop with `asyncio.gather(..., return_exceptions=True)` to drain cancelled tasks before invoking `_core.shutdown()`.
- **Miniscope LED and EWL public-API methods accepted unbounded values.** `set_led_power` and `set_ewl_focus` stored the raw user value into `self._settings.*` before any clamping; internal helpers clamped before sending to hardware, so the GUI slider and lab notebook value diverged from the actual hardware value. Public methods now validate and raise `ValueError` on out-of-range input; the GUI surfaces the error inline; `CameraSettings.from_dict` validates loaded values.
- **`tracking_logger` CSV writes were unprotected.** A disk-full or USB unplug during recording would terminate the log silently, leak the file handle, and leave `is_recording` lying to the UI. All `writerow`/`flush`/`close` calls are now wrapped in try/except with a `_failed` flag, a registered `writer_error` callback (mirroring `VideoRecorder`'s pattern), and the file is rotated to a `.csv.partial` suffix until a clean `stop()` renames it.
- **Auto-reconnect made exactly one attempt then died.** The reconnect task exited because `connect()` failure transitioned state to `ERROR`, breaking the `while _state == RECONNECTING` predicate. Re-firing `start_reconnect()` short-circuited because the now-finished task reference was still non-None. Fixed: the task clears its handle in `finally`, the loop predicate now uses `_auto_reconnect and not is_connected`, and exponential backoff (5 → 10 → 30 → 60 s capped) replaces fixed-interval retries. Error callbacks fire on each failure for UI visibility.
- **`CustomDeviceRunner` bypassed `PinManager`.** Pin conflicts between custom devices and standard devices were undetected — two devices could claim the same pin without raising. Now allocates pins through the board's `PinManager` and refuses binding on `PinConflictError`.
- **`_initialized` was never cleared on shutdown for 6 of 7 device classes.** After `shutdown → initialize(fails) → shutdown`, the second shutdown thought it was still initialized and could write to a board that had been disconnected and reconnected. Every concrete `shutdown()` now clears the flag in a `try/finally`.
- **`TelemetrixBoard.set_pin_mode` and `disconnect` still blocked the qasync loop up to 10 seconds.** Other writers had been wrapped in `asyncio.to_thread`; these two were missed. Now all `_call_telemetrix` invocations are async, so a wedged USB cable cannot freeze the GUI during pin configuration or graceful shutdown.

### Removed

- `_exec_callbacks` channel on `ExecNode` and `ZoneInputNode`. The registrar `on_exec()` was never called from anywhere, so every node firing `exec_output(index)` via the inherited dispatch produced nothing. Node `exec_output` now routes through `_update_callbacks` (the channel the FlowEngine subscribes to) with output-name resolution.

### Known limitations

- **Engine data-flow propagation** between data nodes is not yet wired — math/comparison/display nodes show real values for hardware reads they're directly subscribed to, but chained `Add(A,B) → Threshold → LED` reactive flows are limited. Tracked as a 1.1 architectural item.
- **`ZoneInputNode` live wiring** from the CV processor's frame loop to per-node `update_zone_state` calls is staged but uses an interim direct-call pathway pending a `ZoneOrchestrator` event-bus refactor.
- **GUI test surface is shallow.** `pytest-qt` infrastructure is configured but only a smoke test asserts main-window construction.
- **macOS first-class platform support is community-supported** for 1.0.0. Mac CI matrix and `.dmg` build with notarization are planned for 1.1.
- **Windows installer is unsigned.** Azure Trusted Signing planned for 1.1.

