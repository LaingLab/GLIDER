# Changelog

All notable changes to GLIDER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-07-11

### Added

- **Live behavior classifier** wired into the camera panel: `LiveBehaviorClassifier` + `BehaviorInferenceWorker`, with a live == offline feature-parity test guard.
- **Interactive onboarding tour** (`gui/onboarding/`): spotlight overlay + callout card walking new users through the Node Library, panel tabs, Hardware, canvas, Properties, Camera, and Run. Offered on first launch (Take the Tour / Skip) in Builder mode; replayable anytime from Help → Replay Tutorial.
- **GPU/accelerator diagnostics**: `glider --gpu-check` CLI flag and Tools → GPU / Device Check menu action, reporting CUDA/MPS/CPU availability and the device inference will use.
- **macOS packaging** (`packaging/macos/`): PyInstaller spec producing `GLIDER.app` (with camera/microphone usage descriptions) and a one-command local `build.sh` that packages `GLIDER-<version>-<arch>.dmg`. Unsigned for now; per-arch (arm64/x86_64).
- Local Windows installer build script (`packaging/windows/build.ps1`) and local Pi image build script (`packaging/pi/build.sh`) — one-command equivalents of the CI workflows for building on your own machines.

### Changed

- **Apple Silicon (MPS) acceleration**: behavior-classifier inference and live YOLO detection now resolve CUDA → MPS → CPU instead of silently falling back to CPU on Macs. Training keeps its deterministic CUDA-or-CPU path.
- HiDPI scale-factor rounding changed from `PassThrough` to `RoundPreferFloor` so 1px borders stay crisp at scaled macOS resolutions and on external displays.
- Desktop theme: dock separators blend into the panel background, the node-editor canvas no longer draws a contrasting border, and tabbed-dock tabs stretch the full width of the tab bar.
- `diagnose()`/`format_gpu_info()` no longer advise reinstalling a CUDA wheel on macOS (where CUDA is not applicable); they report MPS as the platform accelerator instead.

## [Unreleased] — release-prep-1.0 branch

Release-readiness work for the 1.0.0 cut. See [`code-review-laing.md`](code-review-laing.md) for the engineering audit driving these changes.

### Added

- `LICENSE` file (MIT).
- `CITATION.cff` for academic citation.
- `CHANGELOG.md` (this file).
- `CONTRIBUTING.md` covering dev setup, code style, hardware contribution, and `.glider` file-format change procedure.
- `SECURITY.md` for responsible disclosure of safety-relevant bugs (e.g., Pi sudoers rule, hardware-safety regressions).
- Comprehensive `README.md` covering what GLIDER does, supported hardware, install across all four platforms, CLI options, citation, troubleshooting, and known limitations.
- `.github/workflows/ci.yml` — test + lint + type-check + PyInstaller smoke build on every push and PR, across Python 3.11/3.12/3.13 on Ubuntu/Windows/macOS.
- Parametric test asserting every registered node class round-trips its `get_state()` through `set_state()` without loss (catches `property_names`-style silent state drops in CI).
- Parametric test asserting every exported node class registers with the `FlowEngine` (catches the "node in the library but engine can't instantiate it" class of bug).
- Round-trip test `tests/unit/serialization/test_serializer.py` covering save → load → save with at least one board, one device, one node, and one connection (catches the API-mismatch class of bug in the apply path).

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

### Changed

- Python version policy unified across `pyproject.toml`, classifiers, `black` target, `ruff` target, `mypy` config, CI matrix, and README — all now **3.11 / 3.12 / 3.13**.
- `requires-python` tightened from `>=3.10,<3.14` to `>=3.11,<3.14` (the codebase uses `match` statements and `X | None` unions, which require 3.10+; we standardise on 3.11 as the floor since 3.10 is not CI-tested).
- `[tool.ruff] target-version = "py311"` added.
- `pyproject.toml` `[dev]` extras group consolidated: now uses `GLIDER[pc,vision,i2c]` self-reference rather than duplicating individual dependencies (no more drift when bumping a `[vision]` pin).
- `uv.lock` is now tracked (was gitignored). Reproducible installs for the application.
- `__main__.spec` (the auto-generated PyInstaller stub at repo root) deleted. `packaging/windows/glider.spec` is the canonical spec; new contributors building locally are no longer routed to the broken stub.
- Removed install one-liner curl-pipe-bash references from the README; the `install.sh` / `install.ps1` scripts they pointed at do not exist and the pattern is a supply-chain risk.
- **Runner UI replaced by a customizable Dashboard.** The four-tab Runner view was replaced by a customizable 2×2 quadrant Dashboard (`gui/dashboard/`) shared by both the desktop and Pi surfaces, with bidirectional Builder↔Dashboard switching. Each quadrant hosts a user-pickable panel (camera, hardware, run control, experiment info, etc.).
- ~2,500 LOC of dead `gui/` code purged (`runner/dashboard.py`, `runner/widget_factory.py`, `widgets/touch_widgets.py`, `widgets/device_card.py`, `controllers/hardware_controller.py`, `controllers/device_control_controller.py`, `panels/experiment_panel.py`). Active runner UI is the quadrant Dashboard in `gui/dashboard/`; active hardware UI is `panels/hardware_panel.py`; etc.
- CI now runs unit + integration tests with coverage, type-check (mypy), lint (ruff), format check (black), and a PyInstaller smoke build that asserts `dist/glider/glider --version` succeeds.

### Removed

- `_exec_callbacks` channel on `ExecNode` and `ZoneInputNode`. The registrar `on_exec()` was never called from anywhere, so every node firing `exec_output(index)` via the inherited dispatch produced nothing. Node `exec_output` now routes through `_update_callbacks` (the channel the FlowEngine subscribes to) with output-name resolution.

### Known limitations carried into 1.0.0

- **Engine data-flow propagation** between data nodes is not yet wired — math/comparison/display nodes show real values for hardware reads they're directly subscribed to, but chained `Add(A,B) → Threshold → LED` reactive flows are limited. Tracked as a 1.1 architectural item.
- **`ZoneInputNode` live wiring** from the CV processor's frame loop to per-node `update_zone_state` calls is staged but uses an interim direct-call pathway pending a `ZoneOrchestrator` event-bus refactor.
- **GUI test surface is shallow.** `pytest-qt` infrastructure is configured but only a smoke test asserts main-window construction.
- **macOS first-class platform support is community-supported** for 1.0.0. Mac CI matrix and `.dmg` build with notarization are planned for 1.1.
- **Windows installer is unsigned.** Azure Trusted Signing planned for 1.1.

## [1.0.0] — 2026-XX-XX (planned)

Initial public release.

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
