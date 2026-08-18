# Contributing to GLIDER

Thanks for your interest. GLIDER is a scientific instrument that researchers rely on for reproducible experimental data, so the contribution bar is *"would I trust this to run unattended overnight?"*

This document covers the practical workflow. For architectural context, read the [GLIDER Ecosystem Reference](docs-site/reference/ecosystem-reference.md) — it explains what each subsystem is responsible for and which claims should be made with appropriate scope.

---

## Setting up

```bash
git clone https://github.com/LaingLab/glider
cd glider

# Install uv if you don't have it (recommended for fast, reproducible installs)
# See https://docs.astral.sh/uv/getting-started/installation/

uv venv
uv sync --extra dev        # pulls pc, vision, i2c, behavior via [dev] self-reference
```

Verify your environment (`uv run` executes commands inside the `.venv` that
`uv venv` created — no activation needed):

```bash
uv run ruff check src tests
uv run black --check src tests
uv run mypy src
uv run pytest tests/ -v
uv run glider --help
```

If any of those fail on a clean checkout, open an issue.

---

## Before you start work

**Open an issue first.** GLIDER is a small project with a focused scope. A PR that arrives without prior discussion is much more likely to be declined or reworked. Issues are also where compatibility constraints (Pi hardware, supported Python versions, manuscript-linked APIs) get surfaced before you've invested time.

For trivial fixes (typos, one-line bug fixes, obvious omissions): an issue is not required.

For new features, hardware drivers, or anything that changes saved-file format: an issue is required, and may also need a brief design note in the PR description.

---

## Code style

- **Formatter:** `black` with line length 100.
- **Linter:** `ruff` with the configuration in `pyproject.toml`. Don't disable rules to suppress your own warnings — fix the warning or open a discussion if the rule is wrong for this codebase.
- **Type hints:** required on every public function and method. Run `mypy src` before sending a PR and keep it clean; CI runs mypy and surfaces its findings, but does not yet fail the build on them.
- **Logging, not print:** every module has `logger = logging.getLogger(__name__)` at the top. Use it. The codebase is `print()`-free.
- **No `eval`, `exec`, `pickle`, `shell=True`, `yaml.load` without `SafeLoader`.** If you genuinely need one of these (you almost certainly don't), open an issue first explaining why.
- **Async patterns:** every coroutine that does hardware I/O must have a timeout. The pattern is `await asyncio.wait_for(device.write(...), timeout=DEVICE_IO_TIMEOUT_S)`. Fire-and-forget `asyncio.create_task` without storing the task handle is a recipe for "STOP doesn't actually stop" bugs — an unreferenced task can be garbage-collected mid-flight, and nothing can cancel a task nobody holds.

---

## Testing

Every PR must include tests for the change. Specifically:

- **New node type:** at minimum, a round-trip test (set every property, save state, load state, assert equality) and an exec-output dispatch test (register a callback, fire each exec output, assert the callback ran). The `tests/unit/nodes/test_node_registration.py` parametric tests will automatically cover new nodes if you follow the `register_*_nodes` pattern.
- **New hardware driver:** unit tests against a fake board (see `tests/conftest.py` for `MockBoard` and async fixtures). At minimum: `connect/disconnect` lifecycle, `set_pin_mode` validation, `write_digital`/`read_digital` happy path, `emergency_stop` semantics.
- **Bug fix:** a regression test that fails before your fix and passes after.

Run the suite locally before pushing:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src pytest tests/ -v
```

GUI tests use `pytest-qt`'s `qtbot` fixture. Don't add tests that require a real display — set `QT_QPA_PLATFORM=offscreen` (CI does this already).

---

## Hardware contributions

Adding support for a new board or device:

1. **Board:** subclass `BaseBoard` in `src/glider/hal/boards/`. Implement `connect`, `disconnect`, `set_pin_mode`, `write_digital`, `write_analog`, `read_digital`, `read_analog`, **and `emergency_stop`** (now `@abstractmethod`). Every I/O method must be `async` and have an internal timeout — a wedged cable or board must not be able to hang the GUI, a running experiment, or an emergency stop.
2. **Device class:** subclass `BaseDevice` in `src/glider/hal/base_device.py`. Implement `initialize` and `shutdown`. Always clear `self._initialized = False` in `shutdown` (don't leave the device in a half-state if reinit later fails).
3. **Entry point:** register the board in `pyproject.toml` under `[project.entry-points."glider.driver"]`. Document the wiring in `docs-site/`.
4. **Tests:** add a test file in `tests/unit/hal/`. Faithful fake boards beat MagicMocks every time.

---

## Adding flow nodes

1. Subclass the appropriate base (`DataNode`, `ExecNode`, `HardwareNode`, `InterfaceNode`, `LogicNode`) in the right subpackage (`nodes/hardware/`, `nodes/interface/`, `nodes/logic/`, `nodes/vision/`).
2. Define a `NodeDefinition` with typed `PortDefinition` inputs and outputs.
3. Implement `execute()` (for ExecNode descendants) or `process()` (for DataNode descendants).
4. Use `self.exec_output(index)` to fire downstream — it now routes through `_update_callbacks` (the channel the FlowEngine subscribes to). Do **not** add a separate `_exec_callbacks` channel.
5. Register the node in the appropriate `register_*_nodes` function (e.g., `nodes/hardware/__init__.py`'s `register_hardware_nodes`). The parametric registration test will fail in CI if a class is exported but not registered.
6. Add a round-trip test if the node has any state worth preserving (delay duration, threshold value, pin number, etc.).

---

## File-format changes

Anything that touches `.glider` JSON schema, the per-node `to_dict` format, or the `ExperimentSchema` dataclasses is a **breaking change** unless explicitly migration-tested. The procedure:

1. Bump `SCHEMA_VERSION` in `src/glider/serialization/schema.py`.
2. Add a migration function that loads the old version and produces the new format.
3. Add a test fixture: an `.glider` file in the old format, plus a test that asserts it loads correctly in the new GLIDER.
4. Update `CHANGELOG.md` under `### Changed` with the migration note.

---

## Commit messages

Use imperative present tense ("Add", not "Added"). One-line summary on the first line, blank line, then body if needed.

Reference issues (`Fixes #123`, `Closes #456`) where applicable.

**Do not attribute commits to AI tools.** GLIDER is published scientific software; authorship matters for academic integrity. If you used an AI assistant during development that's fine, but the commit author and any acknowledgements should be human.

---

## Pull requests

- One logical change per PR. "Refactor X" + "Add feature Y" should be two PRs.
- Update `CHANGELOG.md` under `## [Unreleased]` in the same PR.
- If you touched anything in the "Known limitations" list of `CHANGELOG.md`, update or remove that bullet.
- The PR description should include: what the change does, why it's needed, how you tested it, and any backwards-compatibility implications.
- Don't merge with red CI. If a test that fails is unrelated to your change, fix it in a separate PR first.

---

## Reporting bugs

When opening a bug report, please include:

- **Platform** (OS, Python version, GLIDER version from `glider --version`)
- **Hardware** (board type, devices connected, camera model if relevant)
- **Steps to reproduce**
- **Expected vs. actual behavior**
- **Logs** — GLIDER logs to the terminal it was launched from (there are no log files); include the last ~100 lines of that output. On a Pi kiosk, use `journalctl -u glider`.
- **`.glider` file** if relevant (sanitise any subject IDs first)

Bugs that involve potential data corruption (CSV truncation, video file unplayable, wrong sensor values) or hardware safety (outputs driving after STOP, hung shutdown) are highest priority and should be tagged `safety`.

For security-sensitive disclosures (e.g., the Pi `sudoers-glider` rule), use the private channel in [SECURITY.md](SECURITY.md) instead of opening a public issue.

---

## License

By contributing, you agree your contributions will be licensed under the MIT License (see [LICENSE](LICENSE)).
