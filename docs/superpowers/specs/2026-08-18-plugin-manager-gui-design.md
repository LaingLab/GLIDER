# Plugin manager GUI and marketplace

**Status:** approved 2026-08-18
**Supersedes:** §7 of `2026-08-17-harp-integration-and-plugin-marketplace-design.md` (Project C), which is carried forward here with the GUI and the entry-point fix added.
**Mockup:** https://claude.ai/code/artifact/06c755c8-3bfd-47e8-addc-03c39521941e

---

## 1. Context and goals

GLIDER can load plugins but has no way to discover or install them. `PluginManager`
discovers from entry points and `~/.glider/plugins/`, and `install_requirements`
installs a plugin's dependencies — but nothing installs a *plugin*, and nothing
lists what exists. `glider-harp` currently has to be installed by hand.

This builds the missing half: a curated index, an installer, and a window to
browse both.

Goals:

- A researcher can open **Plugins**, see what is available, and install Harp
  without touching a terminal.
- A researcher can see what is already installed, its version, and whether it is
  enabled.
- Every failure — no network, wrong GLIDER version, pip refusing — is legible on
  the row it belongs to.
- A third-party plugin author using the natural entry-point shape gets a working
  plugin instead of silence.

## 2. Non-goals

- **Uninstall.** Removing a package whose modules are already imported cannot
  fully take effect until restart, and half-removing a plugin that owns live
  hardware drivers is worse than leaving it disabled. Disable is offered instead.
- **Automatic update checks.** No background polling; the index is fetched when
  the window opens.
- **Signing, sandboxing, or permission prompts.** See §9.
- **Retheming the rest of GLIDER.** This window adopts the existing Deep Navy
  palette.
- **Publishing `glider-harp` to PyPI.** Tracked separately; the index entry can
  point at a package that is not yet published, and the install will fail with
  pip's own message, which is the correct behaviour.

## 3. Decisions and rationale

| Decision | Choice | Why |
|---|---|---|
| Scope | Full marketplace: registry + install + GUI | The panel is hollow without something to browse. |
| Placement | Dedicated window from a **Plugins** menu item | Occasional, focused task; stays out of the Builder dock layout. |
| Visual direction | Borrow Claude Code's *structure*, keep Deep Navy | Structure is what makes it feel good; a warm palette here would be a foreign island inside a cool-navy app. |
| Entry-point defect | Fix it in this work | A marketplace inviting third-party plugins must not silently drop the most natural entry-point shape. |
| Concurrency | Async-native, headless core | qasync already drives the loop; keeps registry and installer testable with no Qt. |

### Why not a QThread worker

GLIDER already runs an asyncio loop under Qt via qasync. Introducing QThread
workers would add a second concurrency model and couple the registry to
`QObject`, meaning tests need Qt to exercise index fallback. The parts most
likely to break — network fallback, pip failure — are then GUI tests instead of
unit tests.

### Why not synchronous with a progress dialog

A 3 s index timeout freezes the UI for 3 s; a pip install freezes it for a
minute or more. On a machine driving live hardware that is the wrong trade.

## 4. Architecture

Each unit has one job, and the two that carry the real logic import no Qt.

| File | Responsibility | Imports Qt |
|---|---|---|
| `src/glider/plugins/registry.py` | Resolve the index: fetch → cache → bundled. Records which source won and its `updated` date. | No |
| `src/glider/plugins/installer.py` | `install(name)`: version gate, pip subprocess, cache invalidation, re-discovery. | No |
| `src/glider/plugins/index.json` | Bundled catalogue, shipped as package data. | No |
| `src/glider/gui/dialogs/plugin_manager_dialog.py` | The window: search, filters, list, footer. | Yes |
| `src/glider/gui/widgets/plugin_card.py` | One row: identity, description, state, actions, inline error. | Yes |
| `src/glider/plugins/plugin_manager.py` | *Modified* — entry-point registration fix. | No |

Data flows one way: the dialog asks the registry for entries, merges them with
`PluginManager`'s discovered plugins to compute per-row state, and renders. User
actions call the installer, which reports progress back through an async
callback the dialog renders onto the originating row.

## 5. `PluginRegistry`

Index schema is unchanged from the Harp spec:

```json
{
  "schema_version": "1.0",
  "updated": "2026-08-17",
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

Resolution order: **fetch URL (3 s timeout) → cached copy in `~/.glider/` → bundled copy.**
A successful fetch is written to the cache. The resolved result records which
source won and the index's `updated` date; the window shows both, so *"why isn't
the new plugin listed"* is answerable without a debugger.

A malformed index from the network is treated as a failed fetch — fall through to
cache — rather than propagating a parse error. A malformed *bundled* index is a
packaging bug and raises.

## 6. `PluginManager.install(name)`

1. Look up the entry in the resolved index; unknown name is an error naming the resolved source.
2. Check `glider_requires` against the running GLIDER version. Refuse on mismatch, naming **both** versions.
3. Run `sys.executable -m pip install <pypi>` as an async subprocess, streaming output.
4. On success: `importlib.invalidate_caches()`, re-run discovery, recompute row state.
5. On failure: surface pip's exit code and last output lines; leave state untouched.

Fresh installs load without restart. **Upgrading an already-imported plugin
requires a restart**, and the window says so permanently rather than pretending
otherwise.

## 7. Entry-point registration fix

### The defect

`load_plugin` splits the entry point on `:`, imports the module half, and *calls*
the attribute half as a setup function. `_register_plugin_components` then reads
`BOARD_DRIVERS` / `DEVICE_TYPES` / `NODE_TYPES` off that module. So a
`module:Class` entry point constructs the class with no arguments, discards it,
finds no dictionaries, and registers nothing — raising nothing the user sees.

This is verified, not theoretical: `plugins/glider-harp/tests/test_packaging.py:197`
pins that registration works through the *module-only* entry point and that the
`module:Class` shape registers nothing.

### The fix

Branch on what the attribute actually is:

- **A class** → register it into the registry implied by its entry-point group:
  `glider.driver` → boards, `glider.device` → devices, `glider.node` → nodes.
- **A function** → call it as setup, exactly as today.
- **Neither, or the attribute is missing** → record a load error on the
  `PluginInfo`; the window shows it on the row.

Module-only entry points (`glider_harp = "glider_harp"`, attribute defaulting to
`setup`) are unaffected.

### Idempotence

`glider-harp` declares **both** shapes, so after this fix `HarpBoard` is
registered twice — once directly by class, once via the package's lazy
`BOARD_DRIVERS`. Registration must be explicitly idempotent: registering the same
name with the same class is a no-op; registering the same name with a *different*
class is a conflict and is logged. Relying on dict-overwrite happening to be
harmless is not good enough, because the second case is a real bug worth naming.

### Compatibility

`test_packaging.py:197` must stay green unchanged. Its comment describing the
`module:Class` shape as registering nothing becomes stale and is updated in the
same commit.

## 8. The window

Opened from **Plugins…** in the menu bar. Not modal — installs take minutes and
should not lock the app.

Layout, top to bottom: title bar; toolbar with a search field and All / Installed /
Available filter chips; a scrolling list of cards; a footer.

Each card carries the human name, the pypi package and version in **monospace**
(that is what you type into pip and what a bug report needs), the description, the
author and what it provides, a state pill, and its actions. Failures render
inline on the card, including pip's own output verbatim.

State pills use semantic colour held deliberately separate from the cyan accent,
so "needs attention" reads at a glance without competing with the primary action:

| State | Actions |
|---|---|
| Enabled | Disable, Reload |
| Disabled | Enable |
| Available | Install |
| Installing | Cancel, with streamed pip output |
| Not compatible | Install (disabled), with both versions named |
| Install failed | Retry, with pip output |

The footer permanently shows which index source won, its `updated` date, and the
restart caveat. Per §9 that footer *is* the security model, so it is furniture,
not a tooltip.

## 9. Trust

The index is curated by the maintainers. Anything listed has been vetted by them.
Installation runs arbitrary code with full privileges on a machine driving lab
hardware. **The window showing the index source is the entire security model.**
This is a deliberate position, not an oversight — the same position taken in the
Harp spec and unchanged here.

## 10. Error handling

| Failure | Response |
|---|---|
| Index fetch fails or times out | Silent fallback to cache, then bundled; winning source named in footer |
| Index from network is malformed | Treated as a failed fetch; falls through |
| Bundled index is malformed | Raises — this is a packaging bug |
| Unknown plugin name | Error naming the resolved index source |
| `glider_requires` mismatch | Install refused, both versions named on the row |
| pip fails | Exit code and last output lines inline on the row; state unchanged |
| Entry point attribute missing or not class/function | Load error recorded, shown on the row |
| Same name registered with a different class | Conflict logged; first registration wins |

## 11. Testing

The point of the headless split is that most of this needs no Qt.

**Registry** — fetch succeeds and is cached; fetch times out and cache wins; both
fail and bundled wins; malformed network index falls through; the recorded source
and `updated` date are correct in each case.

**Installer** — version gate accepts and refuses; refusal message names both
versions; pip success triggers re-discovery; pip failure surfaces output and
leaves state unchanged.

**Entry-point fix** — a class registers into the group's registry; a function is
still called as setup; a missing attribute records a load error; double
declaration is idempotent; same-name-different-class is a logged conflict;
`test_packaging.py:197` still passes.

**GUI** (`pytest-qt`) — filter chips and search narrow the list; each state
renders the controls in §8's table; an install failure renders on the
originating row and not elsewhere.

## 12. Risks and open questions

- **The catalogue has one entry at launch.** A browsable marketplace containing
  exactly one plugin is odd but is the right first step. Open question: whether
  the index should live in its own repository from day one, so adding an entry
  does not require a GLIDER release. Recommended, not yet decided.
- **`glider-harp` is not on PyPI yet.** Until it is, Install fails with pip's
  "no matching distribution" message. That is correct behaviour, but it means the
  end-to-end path cannot be demonstrated until publication.
- **No uninstall** may frustrate users who expect symmetry with Install. Revisit
  if it comes up; the reasoning is in §2.
- **Restart-to-upgrade** is stated but not enforced. A user can install an upgrade
  and keep working with the old code loaded. The footer says so; we do not block it.

## 13. Build order

1. Entry-point registration fix, with tests. Independent, and everything else
   assumes it.
2. `PluginRegistry` + bundled `index.json`.
3. `PluginManager.install()`.
4. `plugin_card.py`.
5. `plugin_manager_dialog.py` + menu wiring.
6. GUI tests, docs page.

Steps 1–3 are headless and deliver a working install path from a Python REPL.
Steps 4–6 put a window on it.
