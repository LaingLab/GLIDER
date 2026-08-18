# Plugin Manager GUI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give GLIDER a Plugins window that lists a curated catalogue, installs plugins with pip, and shows every failure on the row it belongs to.

**Architecture:** Two headless modules with no Qt import (`registry.py` resolves the index fetch → cache → bundled; `installer.py` gates on version and drives pip as an async subprocess) plus two Qt modules that render them (`plugin_card.py`, `plugin_manager_dialog.py`). A separate fix makes `module:Class` entry points actually register.

**Tech Stack:** Python 3.11–3.13, PyQt6, qasync, pytest + pytest-asyncio + pytest-qt, ruff + black at line length 100.

**Spec:** `docs/superpowers/specs/2026-08-18-plugin-manager-gui-design.md`

---

## Conventions for every task

Run tests from the worktree root with `PYTHONPATH=src`, or they import the main
checkout instead of this one:

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/ -q
```

GUI tests additionally need `QT_QPA_PLATFORM=offscreen`. **Tasks 1–3 must not
need it** — if a test in those tasks fails without it, something imported Qt
that should not have, and that is a bug in the task, not in the test.

After each task: `ruff check src tests` and `black --check src tests` must pass.

---

## File structure

| File | Responsibility | Imports Qt |
|---|---|---|
| `src/glider/plugins/plugin_manager.py` | *Modified.* Entry-point registration fix + idempotent component registration. | No |
| `src/glider/plugins/registry.py` | *New.* `PluginRegistry`: resolve index fetch → cache → bundled, recording which won. | No |
| `src/glider/plugins/index.json` | *New.* Bundled catalogue, shipped as package data. | No |
| `src/glider/plugins/installer.py` | *New.* `install()`: version gate, async pip, re-discovery. | No |
| `src/glider/gui/widgets/plugin_card.py` | *New.* One row widget. | Yes |
| `src/glider/gui/dialogs/plugin_manager_dialog.py` | *New.* The window. | Yes |
| `src/glider/gui/main_window.py` | *Modified.* Tools → Plugins… menu action. | Yes |

---

## Chunk 1: Headless core (Tasks 1–3)

### Task 1: Make `module:Class` entry points register

Today `load_plugin` calls the attribute after the colon as a setup function.
For `harp = "glider_harp.board:HarpBoard"` that constructs `HarpBoard()` with no
arguments, throws the result away, then looks for `BOARD_DRIVERS` on
`glider_harp.board` — which does not define it. Nothing registers and nothing
raises that the user sees.

**Files:**
- Modify: `src/glider/plugins/plugin_manager.py:317-345` (entry-point branch), `:396-420` (`_register_plugin_components`)
- Test: `tests/unit/plugins/test_entry_point_registration.py` (create)

Registry facts you will need — all three are plain dicts:

| Group / `plugin_type` | Registry | Write API |
|---|---|---|
| `glider.driver` / `"driver"` | `HardwareManager._driver_registry` | `HardwareManager.register_driver(name, cls)` |
| `glider.device` / `"device"` | `glider.hal.base_device.DEVICE_REGISTRY` | `DEVICE_REGISTRY[name] = cls` |
| `glider.node` / `"node"` | `FlowEngine._node_registry` | `FlowEngine.register_node(name, cls)` |

`_discover_from_entry_points` already sets `info.plugin_type` from the group
(`group.split(".")[-1]`), so the group is available without re-reading metadata.
The registry **key** is the entry-point *name* (`info.name`), which is what
`harp = ...` means.

- [ ] **Step 1: Write the failing tests**

`tests/unit/plugins/test_entry_point_registration.py`:

```python
"""What a `module:Class` entry point must do.

These tests own the contract that a plugin author gets what the entry-point
syntax appears to promise. They are deliberately free of Qt and of the real
plugin discovery machinery: a PluginInfo is constructed directly, because the
behaviour under test is what `load_plugin` does with one, not how one is found.
"""

import sys
import types

import pytest

from glider.core.hardware_manager import HardwareManager
from glider.hal.base_device import DEVICE_REGISTRY
from glider.plugins.plugin_manager import PluginInfo, PluginManager


class _Board:
    """Stands in for a board class. Deliberately requires no constructor args,
    so that a test failure means the class was *called*, not that calling it
    happened to raise."""


class _OtherBoard:
    pass


@pytest.fixture
def fake_module(monkeypatch):
    """Install a throwaway module that tests can point entry points at."""
    module = types.ModuleType("fake_plugin_mod")
    module.Board = _Board
    module.OtherBoard = _OtherBoard
    calls = []
    module.setup = lambda: calls.append("setup")
    module.calls = calls
    monkeypatch.setitem(sys.modules, "fake_plugin_mod", module)
    return module


@pytest.fixture(autouse=True)
def clean_registries():
    """Registries are class-level and leak between tests otherwise."""
    drivers = dict(HardwareManager._driver_registry)
    devices = dict(DEVICE_REGISTRY)
    yield
    HardwareManager._driver_registry.clear()
    HardwareManager._driver_registry.update(drivers)
    DEVICE_REGISTRY.clear()
    DEVICE_REGISTRY.update(devices)


async def _load(manager, info):
    manager._plugins[info.name] = info
    return await manager.load_plugin(info.name)


async def test_a_class_entry_point_registers_into_its_group(fake_module):
    manager = PluginManager()
    info = PluginInfo(
        name="fakeboard", entry_point="fake_plugin_mod:Board", plugin_type="driver"
    )

    assert await _load(manager, info) is True
    assert HardwareManager._driver_registry["fakeboard"] is _Board


async def test_a_device_class_lands_in_the_device_registry(fake_module):
    manager = PluginManager()
    info = PluginInfo(
        name="fakedev", entry_point="fake_plugin_mod:Board", plugin_type="device"
    )

    assert await _load(manager, info) is True
    assert DEVICE_REGISTRY["fakedev"] is _Board


async def test_the_class_is_registered_not_instantiated(fake_module):
    """The old behaviour called the attribute. Registering the *class* is the
    whole point, so assert on identity rather than on truthiness."""
    manager = PluginManager()
    info = PluginInfo(
        name="fakeboard", entry_point="fake_plugin_mod:Board", plugin_type="driver"
    )

    await _load(manager, info)
    registered = HardwareManager._driver_registry["fakeboard"]
    assert registered is _Board
    assert not isinstance(registered, _Board)


async def test_a_function_entry_point_is_still_called(fake_module):
    """Regression guard: the existing contract must not change."""
    manager = PluginManager()
    info = PluginInfo(
        name="fakesetup", entry_point="fake_plugin_mod:setup", plugin_type="driver"
    )

    assert await _load(manager, info) is True
    assert fake_module.calls == ["setup"]
    assert "fakesetup" not in HardwareManager._driver_registry


async def test_a_missing_attribute_records_an_error(fake_module):
    manager = PluginManager()
    info = PluginInfo(
        name="ghost", entry_point="fake_plugin_mod:NoSuchThing", plugin_type="driver"
    )

    assert await _load(manager, info) is False
    assert info.error is not None
    assert "NoSuchThing" in info.error


async def test_registering_the_same_class_twice_is_a_no_op(fake_module, caplog):
    """glider-harp declares both a module:Class and a module-only entry point,
    so the same class arrives twice. That must be silent."""
    manager = PluginManager()
    for _ in range(2):
        info = PluginInfo(
            name="dup", entry_point="fake_plugin_mod:Board", plugin_type="driver"
        )
        info.loaded = False
        manager._plugins["dup"] = info
        await manager.load_plugin("dup")

    assert HardwareManager._driver_registry["dup"] is _Board
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


async def test_a_conflicting_class_is_logged_and_the_first_wins(fake_module, caplog):
    manager = PluginManager()
    first = PluginInfo(name="clash", entry_point="fake_plugin_mod:Board", plugin_type="driver")
    await _load(manager, first)

    second = PluginInfo(
        name="clash", entry_point="fake_plugin_mod:OtherBoard", plugin_type="driver"
    )
    await _load(manager, second)

    assert HardwareManager._driver_registry["clash"] is _Board
    assert any("clash" in r.message for r in caplog.records if r.levelname == "WARNING")
```

- [ ] **Step 2: Run to verify they fail**

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/test_entry_point_registration.py -q
```

Expected: the class-registration tests FAIL (nothing lands in the registry);
`test_a_function_entry_point_is_still_called` PASSES already.

**If every test fails, stop** — that usually means the fixture is wrong, not
that the whole feature is missing.

- [ ] **Step 3: Add the idempotent registrar**

In `src/glider/plugins/plugin_manager.py`, above `_register_plugin_components`:

```python
def _registry_for(kind: str) -> dict[str, type] | None:
    """Return the mutable registry dict a plugin component of ``kind`` belongs in.

    Returned by reference on purpose: the caller needs to *read* the current
    occupant to decide whether a write is a no-op, a conflict, or new, and the
    three registries expose no such query in common.
    """
    if kind == "driver":
        from glider.core.hardware_manager import HardwareManager

        return HardwareManager._driver_registry
    if kind == "device":
        from glider.hal.base_device import DEVICE_REGISTRY

        return DEVICE_REGISTRY
    if kind == "node":
        from glider.core.flow_engine import FlowEngine

        return FlowEngine._node_registry
    return None


def _register_component(kind: str, name: str, component: type, plugin: str) -> None:
    """Register one component, tolerating the same thing arriving twice.

    Duplicates are normal rather than exceptional: a package may declare both a
    ``module:Class`` entry point and a ``BOARD_DRIVERS`` table naming the same
    class, and both paths now register. Same class under the same name is a
    no-op. A *different* class under the same name is a real collision between
    two plugins, so it is logged and the first registration is kept -- silently
    overwriting would mean load order decides which hardware driver the lab gets.
    """
    registry = _registry_for(kind)
    if registry is None:
        logger.warning("Plugin %s: unknown component kind %r for %r", plugin, kind, name)
        return

    existing = registry.get(name)
    if existing is component:
        logger.debug("Plugin %s: %s %r already registered", plugin, kind, name)
        return
    if existing is not None:
        logger.warning(
            "Plugin %s: %s %r is already registered to %s; keeping the first",
            plugin,
            kind,
            name,
            getattr(existing, "__name__", existing),
        )
        return

    registry[name] = component
    logger.debug("Plugin %s: registered %s %r", plugin, kind, name)
```

- [ ] **Step 4: Branch on what the attribute is**

Replace the `# Call setup function if it exists` block (around line 340) with:

```python
                # What follows the colon may be a setup function *or* a
                # component class. Both shapes appear in the wild and the
                # syntax does not distinguish them, so branch on the object.
                if hasattr(module, attr_name):
                    attr = getattr(module, attr_name)
                    if inspect.isclass(attr):
                        _register_component(info.plugin_type, info.name, attr, info.name)
                    elif callable(attr):
                        if asyncio.iscoroutinefunction(attr):
                            await attr()
                        else:
                            attr()
                    else:
                        raise TypeError(
                            f"entry point {info.entry_point!r} names {attr!r}, "
                            "which is neither a class nor a callable"
                        )
                elif ":" in info.entry_point:
                    # An explicit attribute that does not exist is an error. A
                    # bare module entry point is not: `setup` is optional.
                    raise AttributeError(
                        f"{module_name} has no attribute {attr_name!r} "
                        f"(from entry point {info.entry_point!r})"
                    )
```

Add `import inspect` to the imports at the top of the file.

- [ ] **Step 5: Route `_register_plugin_components` through the same registrar**

Replace the three loop bodies so duplicates and conflicts behave identically no
matter which path found the component:

```python
    async def _register_plugin_components(self, info: PluginInfo, module: Any) -> None:
        """Register components a plugin exposes as module-level tables."""
        for attr_name, kind in (
            ("BOARD_DRIVERS", "driver"),
            ("DEVICE_TYPES", "device"),
            ("NODE_TYPES", "node"),
        ):
            table = getattr(module, attr_name, None)
            if not table:
                continue
            for name, component in table.items():
                _register_component(kind, name, component, info.name)
```

- [ ] **Step 6: Run the new tests, then the whole plugin suite**

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/ -q
```

Expected: all PASS.

- [ ] **Step 7: Prove the Harp packaging test still passes**

This is the regression that matters most — `test_packaging.py:197` pins that
registration works through the module-only entry point.

```bash
PYTHONPATH="src;plugins/glider-harp/src" python -m pytest plugins/glider-harp/tests/test_packaging.py -q
```

Expected: PASS (some tests may skip if the plugin is not installed; none may fail).

- [ ] **Step 8: Update the now-stale comments**

Two places assert the old behaviour in prose and are now wrong:

1. `plugins/glider-harp/pyproject.toml` — the comment above `[project.entry-points."glider.device"]` says loading the `module:Class` shape "would call `HarpDevice()` as a setup function and fail". After this change it registers instead. Rewrite it to say the shape now registers the class, and that the `glider.device` duplicate remains shadowed by name.
2. `plugins/glider-harp/tests/test_packaging.py:197-205` — the docstring says "the `module:Class` shape registers nothing". Update to describe what it now does. **Do not change the assertions**; only the prose.

- [ ] **Step 9: Commit**

```bash
git add src/glider/plugins/plugin_manager.py tests/unit/plugins/test_entry_point_registration.py plugins/glider-harp/pyproject.toml plugins/glider-harp/tests/test_packaging.py
git commit -m "fix(plugins): register the class a module:Class entry point names"
```

---

### Task 2: `PluginRegistry`

**Files:**
- Create: `src/glider/plugins/registry.py`, `src/glider/plugins/index.json`
- Modify: `pyproject.toml` (package data)
- Test: `tests/unit/plugins/test_registry.py`

- [ ] **Step 1: Write `index.json`**

`src/glider/plugins/index.json`:

```json
{
  "schema_version": "1.0",
  "updated": "2026-08-18",
  "plugins": [
    {
      "name": "glider-harp",
      "display_name": "Harp Devices",
      "version": "0.1.0",
      "pypi": "glider-harp",
      "description": "Harp-protocol instruments (lickometers, behavior boards, olfactometers).",
      "author": "Laing Lab",
      "homepage": "https://github.com/LaingLab/glider-harp",
      "glider_requires": ">=1.0,<2.0",
      "provides": ["driver", "device"]
    }
  ]
}
```

`glider_requires` is `>=1.0`, not `>=1.1` — `src/glider/_version.py` is `1.0.0`,
and `>=1.1` would make the one shipped plugin refuse to install on the version
it ships with.

- [ ] **Step 2: Write the failing tests**

`tests/unit/plugins/test_registry.py`:

```python
"""How the catalogue is resolved, and how it says so.

The resolution order exists because a lab machine may be offline, may have been
offline for a month, or may never have been online. Which source won is not a
debugging detail -- the window shows it, because "why isn't the new plugin
listed" is otherwise unanswerable.
"""

import json

import pytest

from glider.plugins.registry import PluginRegistry

GOOD = {
    "schema_version": "1.0",
    "updated": "2026-08-01",
    "plugins": [{"name": "a", "pypi": "a", "version": "1.0.0", "glider_requires": ">=1.0"}],
}


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


async def test_a_successful_fetch_wins_and_is_cached(cache_dir):
    async def fetch(url, timeout):
        return json.dumps(GOOD)

    reg = PluginRegistry(cache_dir=cache_dir, fetcher=fetch)
    result = await reg.resolve()

    assert result.source == "network"
    assert result.updated == "2026-08-01"
    assert [p["name"] for p in result.plugins] == ["a"]
    assert json.loads((cache_dir / "plugin_index.json").read_text())["updated"] == "2026-08-01"


async def test_cache_wins_when_the_fetch_fails(cache_dir):
    cache_dir.mkdir(parents=True)
    (cache_dir / "plugin_index.json").write_text(json.dumps(GOOD))

    async def fetch(url, timeout):
        raise TimeoutError("no network")

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "cache"
    assert result.updated == "2026-08-01"


async def test_bundled_wins_when_fetch_and_cache_both_fail(cache_dir):
    async def fetch(url, timeout):
        raise TimeoutError("no network")

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "bundled"
    assert any(p["name"] == "glider-harp" for p in result.plugins)


async def test_a_malformed_network_index_falls_through_rather_than_raising(cache_dir):
    """Garbage from the network must not take the app down, and must not be
    cached -- caching it would poison every later run."""
    async def fetch(url, timeout):
        return "{ this is not json"

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "bundled"
    assert not (cache_dir / "plugin_index.json").exists()


async def test_a_malformed_cache_falls_through_to_bundled(cache_dir):
    cache_dir.mkdir(parents=True)
    (cache_dir / "plugin_index.json").write_text("{ not json")

    async def fetch(url, timeout):
        raise TimeoutError("no network")

    result = await PluginRegistry(cache_dir=cache_dir, fetcher=fetch).resolve()

    assert result.source == "bundled"


def test_the_bundled_index_is_valid_json_and_lists_harp():
    """A packaging guard: the shipped file is the last line of defence, so a
    typo in it is not something to discover at runtime on a lab machine."""
    result = PluginRegistry.load_bundled()

    assert result.schema_version == "1.0"
    assert any(p["name"] == "glider-harp" for p in result.plugins)
```

- [ ] **Step 3: Run to verify they fail**

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/test_registry.py -q
```

Expected: FAIL, `ModuleNotFoundError: glider.plugins.registry`.

- [ ] **Step 4: Implement `registry.py`**

```python
"""Resolve the plugin catalogue from the best source available.

Order is network, then cache, then the copy shipped inside the wheel. Each step
down is a degradation the user should be able to see, so the resolved result
carries which source won and how old it is; the Plugins window prints both.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INDEX_URL = "https://raw.githubusercontent.com/LaingLab/glider-plugins/main/index.json"
CACHE_FILENAME = "plugin_index.json"
FETCH_TIMEOUT_SECONDS = 3.0

Fetcher = Callable[[str, float], Awaitable[str]]


@dataclass(frozen=True)
class ResolvedIndex:
    """A catalogue plus the provenance the window has to display."""

    plugins: list[dict[str, Any]] = field(default_factory=list)
    updated: str = ""
    schema_version: str = ""
    source: str = "bundled"  # "network" | "cache" | "bundled"


def _parse(text: str) -> ResolvedIndex | None:
    """Parse an index, returning None rather than raising on anything malformed.

    Callers use None to mean "try the next source". A bad index is a reason to
    fall back, not a reason to fail: the alternative is that one broken file on
    a web server bricks the Plugins window for everyone.
    """
    try:
        data = json.loads(text)
        return ResolvedIndex(
            plugins=list(data["plugins"]),
            updated=str(data.get("updated", "")),
            schema_version=str(data.get("schema_version", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Ignoring malformed plugin index: %s", exc)
        return None


async def _default_fetcher(url: str, timeout: float) -> str:
    """Fetch over HTTP in a worker thread.

    urllib is blocking, and this runs on the Qt event loop via qasync -- calling
    it directly would freeze the UI for the whole timeout.
    """
    import asyncio
    import urllib.request

    def _get() -> str:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")

    return await asyncio.to_thread(_get)


class PluginRegistry:
    def __init__(
        self,
        cache_dir: Path,
        url: str = DEFAULT_INDEX_URL,
        fetcher: Fetcher | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._url = url
        self._fetch = fetcher or _default_fetcher

    @property
    def _cache_path(self) -> Path:
        return self._cache_dir / CACHE_FILENAME

    @staticmethod
    def load_bundled() -> ResolvedIndex:
        """Read the copy shipped in the wheel.

        Unlike the other two sources this one raises. A malformed bundled index
        is a packaging defect that shipped, not a runtime condition to absorb.
        """
        path = Path(__file__).with_name("index.json")
        parsed = _parse(path.read_text(encoding="utf-8"))
        if parsed is None:
            raise ValueError(f"bundled plugin index is malformed: {path}")
        return parsed

    async def resolve(self) -> ResolvedIndex:
        try:
            text = await self._fetch(self._url, FETCH_TIMEOUT_SECONDS)
            parsed = _parse(text)
            if parsed is not None:
                self._write_cache(text)
                return ResolvedIndex(
                    plugins=parsed.plugins,
                    updated=parsed.updated,
                    schema_version=parsed.schema_version,
                    source="network",
                )
        except Exception as exc:
            logger.info("Plugin index fetch failed, falling back: %s", exc)

        if self._cache_path.exists():
            parsed = _parse(self._cache_path.read_text(encoding="utf-8"))
            if parsed is not None:
                return ResolvedIndex(
                    plugins=parsed.plugins,
                    updated=parsed.updated,
                    schema_version=parsed.schema_version,
                    source="cache",
                )

        return self.load_bundled()

    def _write_cache(self, text: str) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            # A read-only home directory must not break resolution.
            logger.warning("Could not cache the plugin index: %s", exc)
```

- [ ] **Step 5: Ship `index.json` as package data**

`[tool.setuptools.package-data]` at `pyproject.toml:141` declares one `glider`
list whose entries are paths **relative to the package**, so this is a new entry
in that list, not a new `glider.plugins` key:

```toml
glider = [
    "schema/*.json",
    "gui/styles/*.qss",
    "gui/styles/icons/*.svg",
    "assets/*.png",
    "plugins/index.json",
]
```

Without this the file is present in a source checkout and absent from an
installed wheel — the same trap `plugins/glider-harp/tests/test_packaging.py`
documents for the Harp profiles. Verify:

```bash
PYTHONPATH=src python -c "from glider.plugins.registry import PluginRegistry; print(PluginRegistry.load_bundled().source, len(PluginRegistry.load_bundled().plugins))"
```

Expected: `bundled 1`

- [ ] **Step 6: Run the tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/test_registry.py -q
```

Expected: all PASS, with no `QT_QPA_PLATFORM` set.

- [ ] **Step 7: Commit**

```bash
git add src/glider/plugins/registry.py src/glider/plugins/index.json tests/unit/plugins/test_registry.py pyproject.toml
git commit -m "feat(plugins): resolve the catalogue from network, cache or bundle"
```

---

### Task 3: `install()`

**Files:**
- Create: `src/glider/plugins/installer.py`
- Test: `tests/unit/plugins/test_installer.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/plugins/test_installer.py`:

```python
"""Installing a plugin, and refusing to.

pip is driven as a subprocess rather than imported, so every test here fakes the
subprocess. That is the point: the failures worth testing are pip's, and pip's
failures are exit codes and text, not exceptions.
"""

import pytest

from glider.plugins.installer import InstallResult, install

ENTRY = {
    "name": "glider-harp",
    "pypi": "glider-harp",
    "version": "0.1.0",
    "glider_requires": ">=1.0,<2.0",
}


def _runner(returncode: int, output: str):
    async def run(args, on_output=None):
        if on_output:
            for line in output.splitlines():
                on_output(line)
        return returncode, output

    return run


async def test_a_compatible_plugin_installs():
    result = await install(
        ENTRY, glider_version="1.0.0", runner=_runner(0, "Successfully installed glider-harp")
    )

    assert result.ok is True
    assert "Successfully installed" in result.output


async def test_the_pip_command_targets_this_interpreter():
    """Installing with the wrong python puts the plugin somewhere GLIDER will
    never import from -- and it would look like it worked."""
    seen = {}

    async def run(args, on_output=None):
        seen["args"] = args
        return 0, ""

    await install(ENTRY, glider_version="1.0.0", runner=run)

    import sys

    assert seen["args"][:4] == [sys.executable, "-m", "pip", "install"]
    assert seen["args"][-1] == "glider-harp"


async def test_too_old_a_glider_is_refused_naming_both_versions():
    entry = {**ENTRY, "glider_requires": ">=2.0"}

    result = await install(entry, glider_version="1.0.0", runner=_runner(0, ""))

    assert result.ok is False
    assert "2.0" in result.message
    assert "1.0.0" in result.message


async def test_a_refusal_never_runs_pip():
    entry = {**ENTRY, "glider_requires": ">=2.0"}
    ran = False

    async def run(args, on_output=None):
        nonlocal ran
        ran = True
        return 0, ""

    await install(entry, glider_version="1.0.0", runner=run)

    assert ran is False


async def test_pip_failure_surfaces_its_own_output():
    output = "ERROR: Could not find a version that satisfies the requirement zmq>=26"
    result = await install(ENTRY, glider_version="1.0.0", runner=_runner(1, output))

    assert result.ok is False
    assert "zmq>=26" in result.output
    assert "1" in result.message


async def test_progress_lines_reach_the_caller():
    """The window streams pip's output onto the row, so the callback has to fire
    as lines arrive rather than once at the end."""
    lines = []
    await install(
        ENTRY,
        glider_version="1.0.0",
        runner=_runner(0, "Collecting glider-harp\nInstalling collected packages"),
        on_output=lines.append,
    )

    assert lines == ["Collecting glider-harp", "Installing collected packages"]


@pytest.mark.parametrize(
    "spec,version,ok",
    [
        (">=1.0,<2.0", "1.0.0", True),
        (">=1.0,<2.0", "1.9.9", True),
        (">=1.0,<2.0", "2.0.0", False),
        (">=1.0,<2.0", "0.9.0", False),
        ("", "1.0.0", True),
    ],
)
async def test_the_version_gate(spec, version, ok):
    result = await install({**ENTRY, "glider_requires": spec}, glider_version=version,
                           runner=_runner(0, ""))
    assert result.ok is ok
```

- [ ] **Step 2: Run to verify they fail**

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/test_installer.py -q
```

Expected: FAIL, `ModuleNotFoundError: glider.plugins.installer`.

- [ ] **Step 3: Implement `installer.py`**

```python
"""Install a catalogue entry with pip.

pip runs as a subprocess of *this* interpreter rather than being imported: pip's
API is explicitly not public, and installing into a different environment than
the one GLIDER is running from would look like success and import like failure.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstallResult:
    ok: bool
    message: str
    output: str = ""


async def _default_runner(args: list[str], on_output: Callable[[str], None] | None = None):
    """Run a command, streaming stdout line by line as it arrives."""
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    chunks: list[str] = []
    assert process.stdout is not None
    async for raw in process.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        chunks.append(line)
        if on_output:
            on_output(line)
    await process.wait()
    return process.returncode or 0, "\n".join(chunks)


async def install(
    entry: dict[str, Any],
    glider_version: str,
    runner=None,
    on_output: Callable[[str], None] | None = None,
) -> InstallResult:
    """Install one catalogue entry, refusing before pip runs if it cannot fit."""
    run = runner or _default_runner
    requires = entry.get("glider_requires", "") or ""

    if requires and Version(glider_version) not in SpecifierSet(requires):
        # Name both versions: "incompatible" alone sends people to the issue
        # tracker to ask which half is wrong.
        return InstallResult(
            ok=False,
            message=(
                f"{entry['name']} needs GLIDER {requires}. You are running {glider_version}."
            ),
        )

    args = [sys.executable, "-m", "pip", "install", entry["pypi"]]
    returncode, output = await run(args, on_output)

    if returncode != 0:
        return InstallResult(
            ok=False, message=f"pip exited with code {returncode}.", output=output
        )

    importlib.invalidate_caches()
    return InstallResult(ok=True, message=f"Installed {entry['name']}.", output=output)
```

- [ ] **Step 4: Confirm `packaging` resolves**

Already declared at `pyproject.toml:38` (`packaging>=21.0`, used by the in-app
updater), so no dependency change is needed. Confirm it imports:

```bash
PYTHONPATH=src python -c "import packaging.specifiers; print('ok')"
```

Note the hazard recorded at `pyproject.toml:199`: this repository has a
top-level `packaging/` directory that tooling can mistake for the distribution.
If this import resolves to the repo directory instead of the library, that is
the cause — do not work around it by hand-rolling version comparison.

- [ ] **Step 5: Run the tests**

```bash
PYTHONPATH=src python -m pytest tests/unit/plugins/ -q
```

Expected: all PASS, no `QT_QPA_PLATFORM`.

- [ ] **Step 6: Commit**

```bash
git add src/glider/plugins/installer.py tests/unit/plugins/test_installer.py
git commit -m "feat(plugins): install a catalogue entry, gating on GLIDER version"
```

---

## Chunk 2: The window (Tasks 4–6)

### Task 4: `PluginCard`

**Files:**
- Create: `src/glider/gui/widgets/plugin_card.py`
- Test: `tests/unit/gui/test_plugin_card.py`

The card owns one plugin's row: identity, description, state pill, actions, and
inline failure text. It holds no registry or installer reference — the dialog
wires its signals. That keeps the card renderable from a plain dict in tests.

States and their controls, from spec §8:

| `state` | Pill text | Buttons |
|---|---|---|
| `enabled` | Enabled | Disable, Reload |
| `disabled` | Disabled | Enable |
| `available` | Available | Install |
| `installing` | Installing | Cancel (disabled), progress bar, output |
| `incompatible` | Not compatible | Install (disabled) |
| `failed` | Install failed | Retry |

- [ ] **Step 1: Write the failing tests**

`tests/unit/gui/test_plugin_card.py`:

```python
"""What a plugin row shows, per state.

These assert on *which controls exist*, not on pixels: the spec fixes the
control set per state, and that is the part a user depends on.
"""

import pytest

from glider.gui.widgets.plugin_card import PluginCard

ENTRY = {
    "name": "glider-harp",
    "display_name": "Harp Devices",
    "version": "0.1.0",
    "pypi": "glider-harp",
    "description": "Harp-protocol instruments.",
    "author": "Laing Lab",
    "provides": ["driver", "device"],
}


@pytest.mark.parametrize(
    "state,expected",
    [
        ("enabled", ["Disable", "Reload"]),
        ("disabled", ["Enable"]),
        ("available", ["Install"]),
        ("incompatible", ["Install"]),
        ("failed", ["Retry"]),
    ],
)
def test_each_state_offers_its_controls(qtbot, state, expected):
    card = PluginCard(ENTRY, state=state)
    qtbot.addWidget(card)

    assert [b.text() for b in card.buttons()] == expected


def test_an_incompatible_plugin_cannot_be_installed(qtbot):
    card = PluginCard(ENTRY, state="incompatible", message="Needs GLIDER >=2.0. Running 1.0.0.")
    qtbot.addWidget(card)

    assert card.buttons()[0].isEnabled() is False
    assert "2.0" in card.message_text()


def test_the_package_name_and_version_are_shown_verbatim(qtbot):
    """These are what you type into pip and what a bug report needs."""
    card = PluginCard(ENTRY, state="available")
    qtbot.addWidget(card)

    assert "glider-harp" in card.identity_text()
    assert "0.1.0" in card.identity_text()


def test_pip_output_is_shown_when_an_install_fails(qtbot):
    card = PluginCard(
        ENTRY,
        state="failed",
        message="pip exited with code 1.",
        output="ERROR: Could not find a version that satisfies the requirement zmq>=26",
    )
    qtbot.addWidget(card)

    assert "zmq>=26" in card.output_text()


def test_clicking_install_emits_the_plugin_name(qtbot):
    card = PluginCard(ENTRY, state="available")
    qtbot.addWidget(card)

    with qtbot.waitSignal(card.install_requested) as blocker:
        card.buttons()[0].click()

    assert blocker.args == ["glider-harp"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests/unit/gui/test_plugin_card.py -q
```

Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement `plugin_card.py`**

Build with `QFrame` + `QVBoxLayout`. Requirements the tests pin:

- signals `install_requested`, `enable_requested`, `disable_requested`, `reload_requested`, each emitting `str` (the plugin name)
- `buttons() -> list[QPushButton]` in the order given by the table above
- `identity_text()`, `message_text()`, `output_text()` accessors
- pill is a `QLabel` with `objectName` `pluginStatePill` and a `state` Qt property, so `desktop.qss` can style it per state without the widget knowing colours
- colours come from `glider.gui.styles.colors`; semantic state colours must be
  distinct from `ACCENT` (spec §8). Add `STATE_OK`, `STATE_WARN`, `STATE_ERR`
  to `colors.py` if not present — do not inline hex in the widget.
- the output area is a read-only `QPlainTextEdit` with a monospace font, hidden
  unless `output` is non-empty

- [ ] **Step 4: Run the tests**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests/unit/gui/test_plugin_card.py -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/widgets/plugin_card.py tests/unit/gui/test_plugin_card.py src/glider/gui/styles/colors.py
git commit -m "feat(gui): one plugin row, with its state and its failure"
```

---

### Task 5: `PluginManagerDialog` and menu wiring

**Files:**
- Create: `src/glider/gui/dialogs/plugin_manager_dialog.py`
- Modify: `src/glider/gui/main_window.py` (Tools menu, around line 983)
- Test: `tests/unit/gui/test_plugin_manager_dialog.py`

- [ ] **Step 1: Write the failing tests**

`tests/unit/gui/test_plugin_manager_dialog.py`:

```python
"""The window: what it lists, how it filters, and where failures land."""

import pytest

from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog
from glider.plugins.registry import ResolvedIndex

INDEX = ResolvedIndex(
    plugins=[
        {"name": "glider-harp", "display_name": "Harp Devices", "version": "0.1.0",
         "pypi": "glider-harp", "description": "Harp instruments.", "author": "Laing Lab",
         "glider_requires": ">=1.0,<2.0"},
        {"name": "glider-bpod", "display_name": "Bpod", "version": "0.2.0",
         "pypi": "glider-bpod", "description": "Bpod state machine.", "author": "Laing Lab",
         "glider_requires": ">=1.0,<2.0"},
    ],
    updated="2026-08-18",
    source="cache",
)


def test_every_catalogue_entry_gets_a_row(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    assert len(dialog.cards()) == 2


def test_an_installed_plugin_reads_as_enabled(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={"glider-harp": "0.1.0"})
    qtbot.addWidget(dialog)

    by_name = {c.plugin_name: c for c in dialog.cards()}
    assert by_name["glider-harp"].state == "enabled"
    assert by_name["glider-bpod"].state == "available"


def test_the_installed_filter_hides_the_rest(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={"glider-harp": "0.1.0"})
    qtbot.addWidget(dialog)

    dialog.set_filter("installed")

    assert [c.plugin_name for c in dialog.visible_cards()] == ["glider-harp"]


def test_search_matches_description_as_well_as_name(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    dialog.set_search("state machine")

    assert [c.plugin_name for c in dialog.visible_cards()] == ["glider-bpod"]


def test_the_footer_names_the_index_source_and_date(qtbot):
    """Spec section 9 makes this the whole security model, so it is not optional
    furniture."""
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    footer = dialog.footer_text()
    assert "cache" in footer.lower()
    assert "2026-08-18" in footer


def test_a_failure_lands_on_its_own_row_only(qtbot):
    dialog = PluginManagerDialog(index=INDEX, installed={})
    qtbot.addWidget(dialog)

    dialog.show_install_failure("glider-bpod", "pip exited with code 1.", "ERROR: no match")

    by_name = {c.plugin_name: c for c in dialog.cards()}
    assert by_name["glider-bpod"].state == "failed"
    assert by_name["glider-harp"].state == "available"
    assert by_name["glider-harp"].message_text() == ""
```

- [ ] **Step 2: Run to verify they fail**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests/unit/gui/test_plugin_manager_dialog.py -q
```

- [ ] **Step 3: Implement the dialog**

`QDialog`, non-modal (`setModal(False)`) — installs take minutes and must not
lock the app. Layout: search `QLineEdit` and three filter buttons in a
`QHBoxLayout`; a `QScrollArea` holding a `QVBoxLayout` of `PluginCard`s; a
footer `QLabel`.

The constructor takes `index: ResolvedIndex` and `installed: dict[str, str]` so
tests construct it with no network and no `PluginManager`. A separate classmethod
`async def open_for(parent, plugin_manager)` does the real resolution and
constructs the dialog — that is the only place network happens.

State per row: in `installed` → `enabled` or `disabled` per `PluginManager`;
otherwise version-gate the entry and use `available` or `incompatible`.

Install runs through `glider.plugins.installer.install`, with `on_output`
appending to the originating card. On success re-run discovery and recompute
that row only.

- [ ] **Step 4: Wire the menu**

In `src/glider/gui/main_window.py`, in the Tools menu block near line 983:

```python
        plugins_action = QAction("&Plugins...", self)
        plugins_action.triggered.connect(self._on_open_plugins)
        tools_menu.addAction(plugins_action)
```

And a handler alongside `_on_new_custom_device` (line 2069), following the same
lazy-import pattern:

```python
    def _on_open_plugins(self) -> None:
        """Open the plugin browser. Non-modal: installs take minutes."""
        from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog

        asyncio.ensure_future(
            PluginManagerDialog.open_for(parent=self, plugin_manager=self._plugin_manager())
        )
```

`GliderCore` stores the manager as `self._plugin_manager` (`glider_core.py:52`)
and exposes **no** public accessor — verified, not assumed. It is also
`None` until `discover_plugins()` has run (`glider_core.py:528-536`), so two
things are required rather than reaching through the private attribute from the
GUI:

1. Add a `plugin_manager` property to `GliderCore` returning `self._plugin_manager`.
2. Have the handler show a plain message if it is `None` rather than opening a
   window that lists nothing — that state is real on a cold start and would
   otherwise look like an empty catalogue.

Confirm `asyncio` is already imported in `main_window.py` before using
`ensure_future`; add the import if not.

- [ ] **Step 5: Run the GUI tests**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -m pytest tests/unit/gui/ -q
```

- [ ] **Step 6: Commit**

```bash
git add src/glider/gui/dialogs/plugin_manager_dialog.py tests/unit/gui/test_plugin_manager_dialog.py src/glider/gui/main_window.py
git commit -m "feat(gui): browse and install plugins from a window"
```

---

### Task 6: Styling and docs

**Files:**
- Modify: `src/glider/gui/styles/desktop.qss`, `docs-site/building/plugins.md`, `CHANGELOG.md`

- [ ] **Step 1: Style the card in `desktop.qss`**

Add rules for `PluginCard`, `QLabel#pluginStatePill[state="..."]` for each of the
six states, and the output area. Use the Deep Navy surfaces already defined in
the file; semantic pill colours come from the `STATE_*` tokens added in Task 4.

- [ ] **Step 2: Verify by eye**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src python -c "
from PyQt6.QtWidgets import QApplication
from glider.gui.dialogs.plugin_manager_dialog import PluginManagerDialog
from glider.plugins.registry import PluginRegistry
app = QApplication([])
d = PluginManagerDialog(index=PluginRegistry.load_bundled(), installed={})
d.grab().save('/tmp/plugins.png')
print('saved')
"
```

Compare against the approved mockup:
https://claude.ai/code/artifact/06c755c8-3bfd-47e8-addc-03c39521941e

- [ ] **Step 3: Document it**

Add a "Installing plugins from the catalogue" section to
`docs-site/building/plugins.md`, above the existing directory-plugin section.
Cover: Tools → Plugins…, what the footer means, that upgrading a loaded plugin
needs a restart, and that there is no uninstall (disable instead).

State plainly that the catalogue is curated and installing runs arbitrary code —
this matches spec §9 and the warning admonition already on that page.

- [ ] **Step 4: Changelog and full suite**

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH="src;plugins/glider-harp/src" python -m pytest tests/ plugins/glider-harp/tests/ -q
ruff check src tests plugins
black --check src tests plugins
```

Note: `test_delay_node_is_accurate_under_realistic_loop_pressure` is a known
pre-existing flake on Windows (~1 run in 5) and is unrelated to this work. If it
is the only failure, re-run it alone to confirm rather than chasing it.

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/styles/desktop.qss docs-site/building/plugins.md CHANGELOG.md
git commit -m "docs(plugins): document installing from the catalogue"
```

---

## Acceptance

- [ ] `module:Class` entry points register; `test_packaging.py` still passes unchanged in its assertions
- [ ] Registry tests pass with **no** `QT_QPA_PLATFORM` set
- [ ] Installer tests pass with **no** `QT_QPA_PLATFORM` set
- [ ] Tools → Plugins… opens a window listing Harp from the bundled index
- [ ] The footer names the index source and its date
- [ ] An install failure renders on its own row and nowhere else
- [ ] Version gate refuses with both versions named

**Not verifiable on this branch:** a real end-to-end install. `glider-harp` is
not published to PyPI, so pressing Install produces pip's "no matching
distribution" error. That is correct behaviour and the failure path is tested,
but the success path is only exercised against a fake runner.
