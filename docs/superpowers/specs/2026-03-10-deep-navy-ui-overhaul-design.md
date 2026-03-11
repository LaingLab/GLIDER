# UI Overhaul Design: "Deep Navy"

## Overview

Complete visual overhaul of the GLIDER desktop and touch interfaces. Replaces the current purple-tinted dark theme with a deep navy-blue palette inspired by Figma/GitHub dark. Centralizes all styling into QSS files with a colors module, eliminating scattered inline styles. Fixes camera panel checkbox overflow and device control panel sizing.

## Color System

### Surfaces (layered depth, blue-black tones)

| Token | Hex | Usage |
|-------|-----|-------|
| CANVAS | `#0a0e13` | Node graph background (deepest) |
| CHROME | `#0b0f14` | Menu bar, toolbar, status bar |
| BASE | `#0f1419` | Window background, input fields |
| SURFACE_1 | `#111820` | Panels, docks |
| SURFACE_2 | `#151c25` | Cards, node bodies, raised elements |
| BORDER | `#1e2530` | Borders, separators, dividers |

### Text

| Token | Hex | Usage |
|-------|-----|-------|
| TEXT_PRIMARY | `#e2e8f0` | Headings, active text |
| TEXT_SECONDARY | `#c0c8d4` | Port labels, descriptions |
| TEXT_TERTIARY | `#94a3b8` | Labels, inactive tabs |
| TEXT_MUTED | `#718096` | Section headers, placeholders |
| TEXT_DISABLED | `#475569` | Disabled text |

### Accent (cyan)

| Token | Hex | Usage |
|-------|-----|-------|
| ACCENT | `#38bdf8` | Selected tabs, buttons, selection borders |
| ACCENT_HOVER | `#0ea5e9` | Hover state |
| ACCENT_PRESSED | `#0284c7` | Pressed state |

### Status (semantic)

| Token | Hex | Usage |
|-------|-----|-------|
| SUCCESS | `#34d399` | Ready, running, connected |
| WARNING | `#fbbf24` | Paused, stopping |
| ERROR | `#f87171` | Error, emergency, recording |
| INFO | `#60a5fa` | Info, data flow |

### Node Categories (header gradients)

| Category | From | To |
|----------|------|-----|
| Hardware/I/O | `#1a4a2e` | `#163d26` |
| Logic/Flow | `#1e3a5f` | `#1a2d47` |
| Interface/Control | `#4a3a1a` | `#3d3018` |
| Script | `#3d1a5f` | `#301447` |

### Ports & Connections

| Token | Hex | Usage |
|-------|-----|-------|
| PORT_DATA | `#38bdf8` | Data port circles |
| PORT_EXEC | `#e2e8f0` | Execution port circles |
| CONN_ACTIVE | `#34d399` | Active data flow |
| CONN_SELECTED | `#38bdf8` | Selected connection |

## Architecture: QSS-First

### Colors Module (`styles/colors.py`)

Single source of truth for every color. Flat module of constants:
- String constants (e.g., `ACCENT = "#38bdf8"`) for QSS template strings
- `QColor` objects (e.g., `Q_ACCENT = QColor(ACCENT)`) for QPainter code
- Node category gradients as tuples: `CAT_HARDWARE_GRADIENT = ("#1a4a2e", "#163d26")`
- Helper: `def with_alpha(hex_color: str, alpha: float) -> str` for `rgba()` values in QSS/QPainter (e.g., unselected node borders)

### QSS Stylesheets

Both `desktop.qss` and `touch.qss` are fully rewritten with the Deep Navy palette. They use Qt dynamic property selectors to style specific components:

```css
/* Example: text roles targeted by QSS */
QLabel[textRole="muted"] { color: #718096; font-size: 11px; }
QLabel[textRole="section"] { color: #718096; font-size: 10px; font-weight: bold; }
```

### Inline Style Elimination

Replace `setStyleSheet(...)` calls with Qt dynamic properties:

```python
# Before
self._fps_label.setStyleSheet("color: #888; font-size: 11px;")

# After
self._fps_label.setProperty("textRole", "muted")
```

Node graph files (QPainter-based, can't use QSS) import `QColor` constants from `colors.py` directly.

Only truly dynamic styles (e.g., toggling recording indicator) stay as code, but reference `colors.py` constants.

## Layout Changes

### Camera Panel

- CV checkboxes: `QHBoxLayout` → `QVBoxLayout` (fixes overflow)
- Remove `setFixedWidth(150)` from CV checkbox
- Add section labels ("Vision", "Camera") as small uppercase `QLabel`s with `textRole="section"`
- Add 1px dividers between sections
- Button hierarchy: Start Preview = accent color, Settings = muted surface color

### Device Control Panel

- `setMinimumWidth`: 200 → 240
- `setWordWrap(True)` on status and value labels
- Increase GroupBox title padding

### Dock Widgets (Hybrid Tab/Dock)

- Default layout groups related docks via `tabifyDockWidget()`:
  - Left: Node Library + Hardware + Device Control
  - Right: Properties + Camera
  - Files dock remains in left area but not tabified (sits below)
- Panels remain `QDockWidget` — users can still undock, rearrange, float
- Dock title bars: flat, `CHROME` background, 1px bottom border
- Active tab: 2px cyan bottom line; inactive: muted text

### Node Graph

- Grid: `CANVAS` background with dot-style grid using `BORDER` color
- Node body: `SURFACE_2` (universal across categories)
- Node headers: category-specific linear gradient
- Selected: 2px `ACCENT` border; unselected: 1px border using `with_alpha(category_color, 0.2)`
- Header text: `TEXT_PRIMARY`; port labels: `TEXT_SECONDARY`
- Content text (PRIMARY, SECONDARY) ≥ 7:1 contrast (WCAG AAA); decorative text (MUTED, DISABLED) ≥ 4.5:1 (WCAG AA)

### Toolbar

- 36px height
- Play/pause/stop with semantic colors (green/gray/gray)
- Emergency stop: subtle red bg + red border, separated by divider
- File name right-aligned in muted text

### Status Bar

- 24px height, `CHROME` background
- Connection dot (green/red) + board name
- Session state
- Node/connection count right-aligned

## Typography

- Body: `-apple-system, "Segoe UI", "Roboto", sans-serif` at 12px (reduced from current 13px for tighter visual density)
- Small/labels: 11px
- Section headers: 10px semibold, uppercase, 0.5px letter-spacing
- Titles: 14px semibold
- Mono: `"SF Mono", "JetBrains Mono", "Cascadia Code", "Consolas", monospace`

## Spacing

- Input fields: 6px padding, 28px min height, 6px border-radius
- Buttons: 7px vertical padding, 30px min height, 6px border-radius
- Panel internal padding: 10px
- Section dividers: 1px `BORDER` with 10px margin
- Scrollbars: 8px wide, rounded handle, transparent track
- Border radius: 6px (controls), 8px (cards/panels/nodes), fully rounded (pills/badges)

## Touch Mode

Same Deep Navy palette. Differences:
- All font sizes +2px
- Dashboard control button min height: 80px (generic buttons remain 44px)
- Scrollbar width: 20px
- Touch target minimums preserved from existing touch.qss

## Files Changed

### New (1)
- `src/glider/gui/styles/colors.py`

### Full Rewrites (2)
- `src/glider/gui/styles/desktop.qss`
- `src/glider/gui/styles/touch.qss`

### Node Graph (4 — QPainter colors)
- `src/glider/gui/node_graph/node_item.py`
- `src/glider/gui/node_graph/graph_view.py`
- `src/glider/gui/node_graph/connection_item.py`
- `src/glider/gui/node_graph/port_item.py`

### Panels (8 — layout fixes + inline style removal)
- `src/glider/gui/panels/camera_panel.py`
- `src/glider/gui/panels/device_control_panel.py`
- `src/glider/gui/panels/hardware_panel.py`
- `src/glider/gui/panels/runner_panel.py`
- `src/glider/gui/panels/node_library_panel.py`
- `src/glider/gui/panels/node_editor_controller.py`
- `src/glider/gui/panels/agent_panel.py`
- `src/glider/gui/panels/experiment_panel.py`

### Dialogs (up to 10 — inline style removal)
- All files under `src/glider/gui/dialogs/`

### Widgets (3 — inline style removal + QPainter colors)
- `src/glider/gui/widgets/touch_widgets.py`
- `src/glider/gui/widgets/device_card.py`
- `src/glider/gui/widgets/multi_camera_preview.py`

### Runner & Controllers (2 — inline style removal)
- `src/glider/gui/runner/dashboard.py`
- `src/glider/gui/controllers/device_control_controller.py`

### Main Window & View Manager (2)
- `src/glider/gui/main_window.py`
- `src/glider/gui/view_manager.py`

### Styles Init (1)
- `src/glider/gui/styles/__init__.py` — add `from glider.gui.styles import colors` re-export

**Total: ~33 files**

### Cleanup
- Delete stale `src/glider/gui/styles/__pycache__/colors.cpython-*.pyc` from previous failed attempt

## Risk Mitigation

The previous "Lab Dark" attempt failed due to:
1. **Wrong palette** — colors didn't feel right. Mitigated: palette validated via interactive mockups.
2. **Invisible node text** — header/port text colors clashed with new backgrounds. Mitigated: universal node body color (`SURFACE_2`) with guaranteed 7:1+ contrast for all text. Colors defined once in `colors.py`.
3. **Camera panel checkbox overflow** — horizontal layout in narrow dock. Mitigated: switch to vertical `QVBoxLayout`.

Additional safeguards:
- All colors in one file (`colors.py`) — single place to audit contrast
- QSS-first approach means fewer inline styles to miss during updates
- Node graph colors are QPainter constants imported from `colors.py` — no magic hex strings
