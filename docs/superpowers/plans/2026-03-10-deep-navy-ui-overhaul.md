# Deep Navy UI Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the purple-tinted dark theme with a Deep Navy palette across all ~33 GUI files, centralizing colors into a constants module and QSS stylesheets, fixing camera panel checkbox overflow and device control panel sizing.

**Architecture:** Create `styles/colors.py` as the single source of truth for all colors. Rewrite both QSS files with the new palette and Qt dynamic property selectors. Migrate all inline `setStyleSheet()` calls to use either QSS properties or `colors.py` constants. Update QPainter code in node graph files to use `QColor` constants.

**Tech Stack:** Python 3.13, PyQt6, QSS (Qt Style Sheets), pytest

**Design Spec:** `docs/superpowers/specs/2026-03-10-deep-navy-ui-overhaul-design.md`

**Note:** `view_manager.py` is listed in the spec but requires no changes — it loads stylesheets dynamically and has no hardcoded colors.

---

## Chunk 1: Foundation (colors module + QSS rewrites)

### Task 1: Cleanup stale artifacts and create colors module

**Files:**
- Create: `src/glider/gui/styles/colors.py`
- Modify: `src/glider/gui/styles/__init__.py`

- [ ] **Step 1: Delete stale pycache from previous attempt**

```bash
find src/glider/gui/styles/__pycache__/ -name "colors*" -delete 2>/dev/null; echo "done"
```

- [ ] **Step 2: Create `colors.py`**

Write `src/glider/gui/styles/colors.py` with all color constants from the design spec. The file should contain:

```python
"""
Centralized color palette for the GLIDER "Deep Navy" theme.

All colors across the GUI reference these constants.
String constants for QSS/f-strings, QColor objects for QPainter.
"""

from PyQt6.QtGui import QColor


def with_alpha(hex_color: str, alpha: float) -> str:
    """Convert a hex color to rgba() string for use in QSS stylesheets.

    Args:
        hex_color: Color in "#RRGGBB" format.
        alpha: Alpha value from 0.0 (transparent) to 1.0 (opaque).

    Returns:
        String like "rgba(R, G, B, A)" suitable for QSS.
        NOTE: This is for QSS only. For QPainter, use qcolor_with_alpha() instead.
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def qcolor_with_alpha(color: QColor, alpha: float) -> QColor:
    """Return a copy of a QColor with the given alpha. For QPainter usage.

    Args:
        color: Source QColor.
        alpha: Alpha from 0.0 to 1.0.

    Returns:
        New QColor with alpha applied.
    """
    c = QColor(color)
    c.setAlphaF(alpha)
    return c


# === Surfaces (layered depth, blue-black tones) ===
CANVAS = "#0a0e13"
CHROME = "#0b0f14"
BASE = "#0f1419"
SURFACE_1 = "#111820"
SURFACE_2 = "#151c25"
BORDER = "#1e2530"

# === Text ===
TEXT_PRIMARY = "#e2e8f0"
TEXT_SECONDARY = "#c0c8d4"
TEXT_TERTIARY = "#94a3b8"
TEXT_MUTED = "#718096"
TEXT_DISABLED = "#475569"

# === Accent (cyan) ===
ACCENT = "#38bdf8"
ACCENT_HOVER = "#0ea5e9"
ACCENT_PRESSED = "#0284c7"

# === Status (semantic) ===
SUCCESS = "#34d399"
WARNING = "#fbbf24"
ERROR = "#f87171"
INFO = "#60a5fa"

# === Node Categories ===
CAT_HARDWARE = "#1a4a2e"
CAT_LOGIC = "#1e3a5f"
CAT_INTERFACE = "#4a3a1a"
CAT_SCRIPT = "#3d1a5f"
CAT_DEFAULT = "#2a2a35"

# Category gradients: (from, to) for QLinearGradient
CAT_HARDWARE_GRADIENT = ("#1a4a2e", "#163d26")
CAT_LOGIC_GRADIENT = ("#1e3a5f", "#1a2d47")
CAT_INTERFACE_GRADIENT = ("#4a3a1a", "#3d3018")
CAT_SCRIPT_GRADIENT = ("#3d1a5f", "#301447")
CAT_DEFAULT_GRADIENT = ("#2a2a35", "#22222c")

# === Node Graph ===
NODE_BODY = "#151c25"
NODE_HEADER_TEXT = "#e2e8f0"
NODE_PORT_LABEL = "#c0c8d4"
NODE_SELECTED = "#38bdf8"

# === Ports & Connections ===
PORT_DATA = "#38bdf8"
PORT_EXEC = "#e2e8f0"
CONN_ACTIVE = "#34d399"
CONN_SELECTED = "#38bdf8"

# === Special ===
RECORDING = "#f87171"
LED_ON = "#34d399"
LED_OFF = "#718096"
INPUT_VALUE = "#34d399"

# === Node Library Category Colors ===
LIB_FLOW = "#1e3a5f"
LIB_FUNCTIONS = "#1a4a4a"
LIB_CONTROL = "#4a3a1a"
LIB_IO = "#1a4a2e"
LIB_AUDIO = "#3d1a5f"
LIB_VIDEO = "#1e3a5f"
LIB_CUSTOM_DEVICES = "#3d1a5f"
LIB_FLOW_FUNCTIONS = "#1e3a5f"
LIB_ZONES = "#4a3a1a"

# === Behavior State Colors (CV-specific, not theme colors) ===
BEHAVIOR_FREEZE = "#0000FF"
BEHAVIOR_IMMOBILE = "#FFFF00"
BEHAVIOR_MOVING = "#00FF00"
BEHAVIOR_DARTING = "#FF0000"

# === QColor versions for QPainter usage ===
Q_CANVAS = QColor(CANVAS)
Q_CHROME = QColor(CHROME)
Q_BASE = QColor(BASE)
Q_SURFACE_1 = QColor(SURFACE_1)
Q_SURFACE_2 = QColor(SURFACE_2)
Q_BORDER = QColor(BORDER)
Q_TEXT_PRIMARY = QColor(TEXT_PRIMARY)
Q_TEXT_SECONDARY = QColor(TEXT_SECONDARY)
Q_TEXT_TERTIARY = QColor(TEXT_TERTIARY)
Q_TEXT_MUTED = QColor(TEXT_MUTED)
Q_TEXT_DISABLED = QColor(TEXT_DISABLED)
Q_ACCENT = QColor(ACCENT)
Q_ACCENT_HOVER = QColor(ACCENT_HOVER)
Q_ACCENT_PRESSED = QColor(ACCENT_PRESSED)
Q_SUCCESS = QColor(SUCCESS)
Q_WARNING = QColor(WARNING)
Q_ERROR = QColor(ERROR)
Q_INFO = QColor(INFO)
Q_CAT_HARDWARE = QColor(CAT_HARDWARE)
Q_CAT_LOGIC = QColor(CAT_LOGIC)
Q_CAT_INTERFACE = QColor(CAT_INTERFACE)
Q_CAT_SCRIPT = QColor(CAT_SCRIPT)
Q_CAT_DEFAULT = QColor(CAT_DEFAULT)
Q_NODE_BODY = QColor(NODE_BODY)
Q_NODE_HEADER_TEXT = QColor(NODE_HEADER_TEXT)
Q_NODE_PORT_LABEL = QColor(NODE_PORT_LABEL)
Q_NODE_SELECTED = QColor(NODE_SELECTED)
Q_PORT_DATA = QColor(PORT_DATA)
Q_PORT_EXEC = QColor(PORT_EXEC)
Q_CONN_ACTIVE = QColor(CONN_ACTIVE)
Q_CONN_SELECTED = QColor(CONN_SELECTED)
Q_RECORDING = QColor(RECORDING)
Q_LED_ON = QColor(LED_ON)
Q_LED_OFF = QColor(LED_OFF)
Q_INPUT_VALUE = QColor(INPUT_VALUE)
```

- [ ] **Step 3: Update `styles/__init__.py`**

Add the colors import to `src/glider/gui/styles/__init__.py`. Add this line after the existing imports:

```python
from glider.gui.styles import colors  # noqa: F401
```

- [ ] **Step 4: Verify import works**

Run: `python -c "from glider.gui.styles.colors import ACCENT, Q_ACCENT, with_alpha; print(with_alpha(ACCENT, 0.2))"`
Expected: `rgba(56, 189, 248, 0.2)`

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/styles/colors.py src/glider/gui/styles/__init__.py
git commit -m "feat: add centralized Deep Navy color constants module"
```

---

### Task 2: Rewrite desktop stylesheet

**Files:**
- Rewrite: `src/glider/gui/styles/desktop.qss`

- [ ] **Step 1: Write the new desktop.qss**

Full rewrite of `src/glider/gui/styles/desktop.qss` using the Deep Navy palette. Every selector from the current file must be present with updated colors. Key mapping:

| Old | New | Notes |
|-----|-----|-------|
| `#1a1a2e` | `#0f1419` | BASE |
| `#16162a` | `#0b0f14` | CHROME |
| `#0d0d1a` | `#0a0e13` | CANVAS |
| `#2d2d44` | `#1e2530` | BORDER |
| `#3498db` | `#38bdf8` | ACCENT |
| `#2980b9` | `#0ea5e9` | ACCENT_HOVER |
| `#1f6aa5` | `#0284c7` | ACCENT_PRESSED |
| `#3d3d5c` | `#1e2530` | scrollbar handle → BORDER |
| `#4d4d6c` | `#2a3440` | scrollbar hover (slightly lighter than BORDER) |
| `#888` | `#718096` | TEXT_MUTED |
| `#666` | `#475569` | TEXT_DISABLED |
| `#e0e0e0` | `#e2e8f0` | TEXT_PRIMARY |
| `#fff` | `#e2e8f0` | TEXT_PRIMARY |
| `#444` | `#1e2530` | category header → BORDER |

Additional changes:
- Font size: `13px` → `12px`
- Add `border-radius: 6px` to inputs, buttons
- GroupBox: replace border box with section-header style (top border only, no box)
- Scrollbars: 8px wide, transparent track (`background: transparent`), rounded handle
- Add dynamic property selectors:
  ```css
  /* Text roles */
  QLabel[textRole="muted"] { color: #718096; font-size: 11px; }
  QLabel[textRole="section"] { color: #718096; font-size: 10px; font-weight: bold; text-transform: uppercase; }
  QLabel[textRole="secondary"] { color: #94a3b8; }

  /* Button roles */
  QPushButton[buttonRole="secondary"] { background-color: #1e2530; color: #94a3b8; }
  QPushButton[buttonRole="secondary"]:hover { background-color: #2a3440; }
  QPushButton[buttonRole="danger"] { background-color: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }

  /* Recording indicator */
  QLabel[recording="true"] { background-color: rgba(248, 113, 113, 0.2); color: #f87171; }

  /* Input value display */
  QLabel[inputValue="true"] { background-color: #151c25; border-radius: 4px; color: #34d399; }

  /* Status states */
  QLabel[statusState="IDLE"] { background-color: #1e2530; }
  QLabel[statusState="READY"] { background-color: rgba(52, 211, 153, 0.15); color: #34d399; }
  QLabel[statusState="RUNNING"] { background-color: rgba(56, 189, 248, 0.15); color: #38bdf8; }
  QLabel[statusState="PAUSED"], QLabel[statusState="STOPPING"] { background-color: rgba(251, 191, 36, 0.15); color: #fbbf24; }
  QLabel[statusState="ERROR"] { background-color: rgba(248, 113, 113, 0.15); color: #f87171; }
  ```

- Node library buttons: update category colors to match new `LIB_*` constants
- Runner action buttons: update green/red to `SUCCESS`/`ERROR`

- [ ] **Step 2: Verify no load errors**

Run: `python -c "from glider.gui.styles import get_desktop_stylesheet; s = get_desktop_stylesheet(); print(f'{len(s)} chars loaded')"`
Expected: Prints char count (should be 3000-6000+ chars for a complete stylesheet)

- [ ] **Step 3: Commit**

```bash
git add src/glider/gui/styles/desktop.qss
git commit -m "feat: rewrite desktop stylesheet with Deep Navy palette"
```

---

### Task 3: Rewrite touch stylesheet

**Files:**
- Rewrite: `src/glider/gui/styles/touch.qss`

- [ ] **Step 1: Write the new touch.qss**

Same Deep Navy palette as desktop. Apply the same color mapping table from Task 2. Keep all touch-specific sizing:
- Font sizes: all +2px from desktop (body 14px, small 13px, section 12px, title 16px)
- `QPushButton` generic: `min-height: 44px`
- `QPushButton[runnerAction]` variants: `min-height: 80px`
- Scrollbar width: `20px` (not 8px)
- Checkbox/radio indicators: keep existing touch sizes
- Include the same dynamic property selectors as desktop.qss

- [ ] **Step 2: Verify no load errors**

Run: `python -c "from glider.gui.styles import get_touch_stylesheet; s = get_touch_stylesheet(); print(f'{len(s)} chars loaded')"`
Expected: Prints char count

- [ ] **Step 3: Commit**

```bash
git add src/glider/gui/styles/touch.qss
git commit -m "feat: rewrite touch stylesheet with Deep Navy palette"
```

---

**Note:** Tasks 2 and 3 are independent of each other and can be parallelized.

---

## Chunk 2: Node Graph (QPainter updates)

**Note:** Tasks 4-7 are independent of each other (they all depend on Task 1 only) and can be parallelized.

### Task 4: Update node_item.py colors

**Files:**
- Modify: `src/glider/gui/node_graph/node_item.py`

- [ ] **Step 1: Add colors import**

At the top of the file, after the existing PyQt6 imports, add:

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Replace CATEGORY_COLORS dict (line 53-59)**

Replace the `CATEGORY_COLORS` class variable:

```python
CATEGORY_COLORS = {
    "hardware": colors.Q_CAT_HARDWARE,
    "logic": colors.Q_CAT_LOGIC,
    "interface": colors.Q_CAT_INTERFACE,
    "script": colors.Q_CAT_SCRIPT,
    "default": colors.Q_CAT_DEFAULT,
}

CATEGORY_GRADIENTS = {
    "hardware": colors.CAT_HARDWARE_GRADIENT,
    "logic": colors.CAT_LOGIC_GRADIENT,
    "interface": colors.CAT_INTERFACE_GRADIENT,
    "script": colors.CAT_SCRIPT_GRADIENT,
    "default": colors.CAT_DEFAULT_GRADIENT,
}
```

- [ ] **Step 3: Replace body and selection colors (line 91-92)**

```python
self._body_color = colors.Q_NODE_BODY
self._selected_border_color = colors.Q_NODE_SELECTED
```

- [ ] **Step 4: Replace header text color (line 112)**

```python
self._header_text.setDefaultTextColor(colors.Q_NODE_HEADER_TEXT)
```

- [ ] **Step 5: Update paint() method**

In `paint()`, replace lines 230-232 (the `gradient = QLinearGradient(...)`, `gradient.setColorAt(0, self._header_color.lighter(120))`, `gradient.setColorAt(1, self._header_color)`) with:

```python
# Header gradient — use category gradient tuple
gradient = QLinearGradient(0, 0, 0, self.HEADER_HEIGHT)
grad_tuple = self.CATEGORY_GRADIENTS.get(
    self._category, self.CATEGORY_GRADIENTS["default"]
)
gradient.setColorAt(0, QColor(grad_tuple[0]))
gradient.setColorAt(1, QColor(grad_tuple[1]))
```

Replace the port label color (line 250):

```python
painter.setPen(QPen(colors.Q_NODE_PORT_LABEL))
```

Update the unselected border to use `qcolor_with_alpha` (NOT `with_alpha` which is QSS-only):

```python
if self.isSelected():
    pen = QPen(self._selected_border_color, self._border_width + 1)
else:
    alpha_color = colors.qcolor_with_alpha(self._header_color, 0.3)
    pen = QPen(alpha_color, 1)
```

- [ ] **Step 6: Verify import**

Run: `python -c "from glider.gui.node_graph.node_item import NodeItem; print('OK')"`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add src/glider/gui/node_graph/node_item.py
git commit -m "feat: update node_item colors to Deep Navy palette"
```

---

### Task 5: Update graph_view.py colors

**Files:**
- Modify: `src/glider/gui/node_graph/graph_view.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Replace grid colors (around line 45-47)**

Find the grid/background color definitions in `NodeGraphScene.__init__` and replace:

```python
self._grid_color = colors.Q_BORDER       # was QColor(50, 50, 50)
self._grid_color_major = colors.Q_BORDER  # was QColor(70, 70, 70)
self._background_color = colors.Q_CANVAS  # was QColor(30, 30, 30)
```

Note: Both minor and major grid use `Q_BORDER` since the Deep Navy theme uses a dot-grid pattern where subtle uniformity looks better than two grid levels.

- [ ] **Step 3: Verify import**

Run: `python -c "from glider.gui.node_graph.graph_view import NodeGraphView; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/glider/gui/node_graph/graph_view.py
git commit -m "feat: update graph_view colors to Deep Navy palette"
```

---

### Task 6: Update connection_item.py colors

**Files:**
- Modify: `src/glider/gui/node_graph/connection_item.py`

- [ ] **Step 1: Add colors import and replace constants (around lines 24-26, 121)**

```python
from glider.gui.styles import colors
```

Replace the color constants:
- `QColor(100, 180, 255)` → `colors.Q_PORT_DATA` (line 24)
- `QColor(255, 255, 255)` → `colors.Q_PORT_EXEC` (line 25)
- `QColor(100, 255, 100)` → `colors.Q_CONN_ACTIVE` (line 26)
- `QColor(255, 180, 0)` → `colors.Q_CONN_SELECTED` (line 121, in paint method)

- [ ] **Step 2: Verify import**

Run: `python -c "from glider.gui.node_graph.connection_item import ConnectionItem; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/glider/gui/node_graph/connection_item.py
git commit -m "feat: update connection_item colors to Deep Navy palette"
```

---

### Task 7: Update port_item.py colors

**Files:**
- Modify: `src/glider/gui/node_graph/port_item.py`

- [ ] **Step 1: Add colors import and replace port colors (around lines 38-39)**

```python
from glider.gui.styles import colors
```

Replace:
- `QColor(100, 180, 255)` → `colors.Q_PORT_DATA`
- `QColor(255, 255, 255)` → `colors.Q_PORT_EXEC`

- [ ] **Step 2: Verify import**

Run: `python -c "from glider.gui.node_graph.port_item import PortItem; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/glider/gui/node_graph/port_item.py
git commit -m "feat: update port_item colors to Deep Navy palette"
```

---

## Chunk 3: Panel layout fixes + inline style removal

**Note:** Tasks 8-15 are independent of each other and can be parallelized.

### Task 8: Fix camera panel layout and remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/camera_panel.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Fix CV checkbox layout (lines 366-384)**

Replace the horizontal CV layout section:

```python
# Old (lines 366-384):
cv_layout = QHBoxLayout()
self._cv_enabled_cb = QCheckBox("Computer Vision")
self._cv_enabled_cb.setFixedWidth(150)  # DELETE THIS
self._cv_enabled_cb.toggled.connect(self._on_cv_toggle)
cv_layout.addWidget(self._cv_enabled_cb)
# ... etc

# New:
cv_section_label = QLabel("Vision")
cv_section_label.setProperty("textRole", "section")
layout.addWidget(cv_section_label)

cv_layout = QVBoxLayout()
cv_layout.setSpacing(4)

self._cv_enabled_cb = QCheckBox("Computer Vision")
self._cv_enabled_cb.toggled.connect(self._on_cv_toggle)
cv_layout.addWidget(self._cv_enabled_cb)

self._overlay_cb = QCheckBox("Overlays")
self._overlay_cb.toggled.connect(self._on_overlay_toggle)
cv_layout.addWidget(self._overlay_cb)

self._vision_cone_cb = QCheckBox("Vision Cone")
self._vision_cone_cb.toggled.connect(self._on_vision_cone_toggle)
cv_layout.addWidget(self._vision_cone_cb)

layout.addLayout(cv_layout)
```

- [ ] **Step 3: Replace all inline `setStyleSheet()` calls with Qt properties**

For each inline style in camera_panel.py, replace with a Qt dynamic property:

| Line | Old | New |
|------|-----|-----|
| 94-100 | `setStyleSheet("background: #0d0d1a; border: ...")` | Remove — QSS handles `CameraPreviewWidget` via `QLabel` styling |
| 186-194 | Placeholder stylesheet | Remove — use `setProperty("textRole", "muted")` on text |
| 299-305 | Status frame stylesheet | Remove — `status_frame.setProperty("role", "statusBar")` and add QSS selector |
| 310-319 | Recording indicator stylesheet | `self._recording_indicator.setProperty("recording", True)` — QSS handles |
| 324 | `"color: #888; font-size: 11px;"` | `self._fps_label.setProperty("textRole", "muted")` |
| 330 | `"color: #888; font-size: 11px;"` | `self._resolution_label.setProperty("textRole", "muted")` |

For the status frame, add a section divider QFrame before it:

```python
divider = QFrame()
divider.setFrameShape(QFrame.Shape.HLine)
layout.addWidget(divider)
```

- [ ] **Step 4: Add button hierarchy**

Set the Settings button as secondary:

```python
self._settings_btn.setProperty("buttonRole", "secondary")
```

- [ ] **Step 5: Verify import**

Run: `python -c "from glider.gui.panels.camera_panel import CameraPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/glider/gui/panels/camera_panel.py
git commit -m "fix: camera panel vertical checkboxes, remove inline styles"
```

---

### Task 9: Fix device control panel layout and remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/device_control_panel.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Fix layout issues**

- Change `setMinimumWidth(200)` → `setMinimumWidth(240)` (find the line with `setMinimumWidth`)
- Add `self._device_status_label.setWordWrap(True)` after status label creation
- Add `self._input_value_label.setWordWrap(True)` after value label creation

- [ ] **Step 3: Replace inline styles (lines 151, 194)**

| Line | Old | New |
|------|-----|-----|
| 151 | `"background-color: #2d2d2d; border-radius: 4px; color: #00ff00;"` | `self._input_value_label.setProperty("inputValue", True)` |
| 194 | `"font-size: 11px; color: #888; padding: 2px;"` | `label.setProperty("textRole", "muted")` |

- [ ] **Step 4: Verify import**

Run: `python -c "from glider.gui.panels.device_control_panel import DeviceControlPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/panels/device_control_panel.py
git commit -m "fix: device control panel min width, word wrap, remove inline styles"
```

---

### Task 10: Update hardware_panel.py — remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/hardware_panel.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Remove the large `setStyleSheet()` block (lines 139-193)**

This block styles the entire tree widget and buttons. Remove it entirely — the QSS file now handles all `QTreeWidget`, `QPushButton`, and related selectors.

- [ ] **Step 3: Replace remaining inline styles**

For each remaining `setStyleSheet` call:

| Line | Old | New |
|------|-----|-----|
| 220 | `"color: #888; font-style: italic;"` | `label.setProperty("textRole", "muted")` |
| 251-253 | Dynamic status color `#2ecc71`/`#e74c3c` | Use f-string with colors: `f"color: {colors.SUCCESS if connected else colors.ERROR};"` |
| 260, 264 | `"color: #888;"` / `"color: #e74c3c;"` | `setProperty("textRole", "muted")` / `f"color: {colors.ERROR};"` |
| 467+ | Various `"color: #888;"` | `setProperty("textRole", "muted")` |

- [ ] **Step 4: Verify import**

Run: `python -c "from glider.gui.panels.hardware_panel import HardwarePanel; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/panels/hardware_panel.py
git commit -m "feat: remove hardware panel inline styles, use QSS + colors module"
```

---

### Task 11: Update runner_panel.py — remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/runner_panel.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Replace stylesheet blocks**

The runner panel has many large stylesheet blocks (lines 76-95, 103-113, 122-133, 167-173, 371-396, 469-486). Strategy:

- **Experiment name input** (lines 76-95): Remove — QSS `QLineEdit` selector handles this
- **Timer** (lines 103-113): Keep but use colors: `f"color: {colors.SUCCESS};"`
- **Menu button** (lines 122-133): Remove — use `setProperty("buttonRole", "secondary")`
- **No devices label** (lines 167-173): `setProperty("textRole", "muted")`
- **Device cards** (lines 371-396): Remove large block — use `setProperty("deviceCard", True)` and QSS selector
- **Menu** (lines 469-486): Remove — QSS `QMenu` selector handles this

- [ ] **Step 3: Replace dynamic state colors**

Replace all state color variables (lines 328-345, 419-434) with `colors.*`:

```python
# Old: state_color = "#3498db"
# New:
state_color = colors.ACCENT  # or colors.SUCCESS, colors.ERROR, etc.
```

Mapping:
- `"#3498db"` → `colors.ACCENT`
- `"#444"` → `colors.BORDER`
- `"#27ae60"` → `colors.SUCCESS`
- `"#7f8c8d"` → `colors.TEXT_MUTED`
- `"#666"` → `colors.TEXT_DISABLED`
- `"#fff"` → `colors.TEXT_PRIMARY`
- `"#888"` → `colors.TEXT_MUTED`

- [ ] **Step 4: Verify import**

Run: `python -c "from glider.gui.panels.runner_panel import RunnerPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/panels/runner_panel.py
git commit -m "feat: remove runner panel inline styles, use QSS + colors module"
```

---

### Task 12: Update node_library_panel.py — remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/node_library_panel.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Replace category colors dict (lines 183-189)**

```python
CATEGORY_COLORS = {
    "Flow": colors.LIB_FLOW,
    "Functions": colors.LIB_FUNCTIONS,
    "Control": colors.LIB_CONTROL,
    "I/O": colors.LIB_IO,
    "Audio": colors.LIB_AUDIO,
    "Video": colors.LIB_VIDEO,
    "default": colors.BORDER,
}
```

- [ ] **Step 3: Replace inline styles**

| Lines | Old | New |
|-------|-----|-----|
| 203 | Category header stylesheet with hardcoded color | Use `colors.LIB_*` in f-string |
| 225 | Node button with `{color}40` alpha | Use `colors.with_alpha(color, 0.25)` |
| 266 | `#6a4a8a` | `colors.LIB_CUSTOM_DEVICES` |
| 280 | `#4a6a8a` | `colors.LIB_FLOW_FUNCTIONS` |
| 294 | `#5a4a2d` | `colors.LIB_ZONES` |
| 368, 394, 403, 487 | `"color: #888;"` | `setProperty("textRole", "muted")` |

- [ ] **Step 4: Verify import**

Run: `python -c "from glider.gui.panels.node_library_panel import NodeLibraryPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/panels/node_library_panel.py
git commit -m "feat: remove node library panel inline styles, use colors module"
```

---

### Task 13: Update node_editor_controller.py — remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/node_editor_controller.py`

- [ ] **Step 1: Replace inline styles (lines 479, 638, 709)**

All three are `"color: #888; font-size: 10px;"`. Replace with:

```python
label.setProperty("textRole", "muted")
```

No colors import needed since we're only using Qt properties.

- [ ] **Step 2: Verify import**

Run: `python -c "from glider.gui.panels.node_editor_controller import NodeEditorController; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/glider/gui/panels/node_editor_controller.py
git commit -m "feat: remove node editor controller inline styles"
```

---

### Task 14: Update agent_panel.py — remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/agent_panel.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Replace all inline styles**

This file has many stylesheet blocks. Strategy:

| Lines | Component | Action |
|-------|-----------|--------|
| 47 | Header label | `setProperty("textRole", "muted")` |
| 65-85 | MessageBubble styles | Keep as inline but use colors: user bg=`colors.ACCENT_PRESSED`, assistant bg=`colors.SURFACE_2`, text=`colors.TEXT_PRIMARY` |
| 112 | Action title | Keep inline, use `colors.WARNING` |
| 118 | Action description | `setProperty("textRole", "secondary")` |
| 126-151 | Confirm/reject buttons | Keep inline, use `colors.SUCCESS`/`colors.ERROR` |
| 160-166 | Action frame border | Keep inline, use `colors.WARNING` for border |
| 229 | Chat header | Remove — QSS handles |
| 241 | Model label | `setProperty("textRole", "muted")` |
| 259 | Scroll area | Remove — QSS handles |
| 274 | Message container | Remove — QSS handles |
| 286-297 | Quick prompt buttons | Remove — use `setProperty("buttonRole", "secondary")` |
| 306 | Input area | Remove — QSS handles |
| 311-322 | Input field | Remove — QSS `QLineEdit:focus` handles |
| 328-342 | Send button | Remove — QSS `QPushButton` handles primary button |

- [ ] **Step 3: Verify import**

Run: `python -c "from glider.gui.panels.agent_panel import AgentPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/glider/gui/panels/agent_panel.py
git commit -m "feat: remove agent panel inline styles, use QSS + colors module"
```

---

### Task 15: Update experiment_panel.py — remove inline styles

**Files:**
- Modify: `src/glider/gui/panels/experiment_panel.py`

- [ ] **Step 1: Replace inline styles**

| Line | Old | New |
|------|-----|-----|
| 101 | `"QScrollArea { border: none; }"` | Remove — QSS handles |
| 226-232 | Active subject frame with `#1a3a1a` | Keep inline, use `colors.with_alpha(colors.SUCCESS, 0.15)` for bg and `colors.with_alpha(colors.SUCCESS, 0.3)` for border |
| 238 | `"font-weight: bold; color: #8f8;"` | `f"font-weight: bold; color: {colors.SUCCESS};"` |
| 242 | `"color: #afa;"` | `f"color: {colors.SUCCESS};"` |

- [ ] **Step 2: Verify import**

Run: `python -c "from glider.gui.panels.experiment_panel import ExperimentPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/glider/gui/panels/experiment_panel.py
git commit -m "feat: remove experiment panel inline styles, use colors module"
```

---

## Chunk 4: Dialogs, Widgets, Controllers, Main Window

**Note:** Tasks 16-20 are independent of each other and can be parallelized.

### Task 16: Update all dialogs — remove inline styles

**Files:**
- Modify: `src/glider/gui/dialogs/help_dialog.py`
- Modify: `src/glider/gui/dialogs/analysis_dialog.py`
- Modify: `src/glider/gui/dialogs/camera_settings_dialog.py`
- Modify: `src/glider/gui/dialogs/experiment_dialog.py`
- Modify: `src/glider/gui/dialogs/zone_dialog.py`
- Modify: `src/glider/gui/dialogs/calibration_dialog.py`
- Modify: `src/glider/gui/dialogs/flow_function_dialog.py`
- Modify: `src/glider/gui/dialogs/subject_dialog.py`
- Skip: `src/glider/gui/dialogs/agent_settings_dialog.py` (no inline styles, no changes needed)
- Skip: `src/glider/gui/dialogs/custom_device_dialog.py` (no inline styles)

- [ ] **Step 1: Add colors import to each dialog that needs it**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Update help_dialog.py**

Replace the large `STYLE` and `CONTENT_STYLE` string constants (lines 19-102). These use many hardcoded colors. Rewrite using f-strings referencing `colors.*`:

```python
STYLE = f"""
    QDialog {{
        background-color: {colors.BASE};
    }}
    QTabWidget::pane {{
        border: 1px solid {colors.BORDER};
        background-color: {colors.BASE};
    }}
    ...
"""
```

- [ ] **Step 3: Update analysis_dialog.py**

Same pattern as agent_panel (they share MessageBubble code). Replace all hardcoded colors with `colors.*` references. Key replacements:
- `#2d2d2d` → `colors.SURFACE_2`
- `#444` → `colors.BORDER`
- `#888` → `colors.TEXT_MUTED`
- `#1e1e1e` → `colors.SURFACE_1`
- `#3c3c3c` → `colors.SURFACE_2`
- `#4caf50` → `colors.SUCCESS`
- `#f44336` → `colors.ERROR`
- `#2196f3` → `colors.ACCENT`

- [ ] **Step 4: Update camera_settings_dialog.py**

Remove the large stylesheet blocks (lines 92-249) — QSS handles tab widgets, group boxes, scrollbars, etc. Keep behavior state colors as they are domain-specific (FREEZE/IMMOBILE/MOVING/DARTING), but reference `colors.BEHAVIOR_*` constants. Replace `#888`/`#aaa` with `setProperty("textRole", "muted")`.

- [ ] **Step 5: Update experiment_dialog.py**

Replace active subject frame colors (lines 240-256) same as experiment_panel.py (Task 15).

- [ ] **Step 6: Update zone_dialog.py and calibration_dialog.py**

Replace preview label stylesheets with `colors.CANVAS` and `colors.BORDER`. Replace `#666` border with `colors.BORDER`.

- [ ] **Step 7: Update flow_function_dialog.py**

Replace QColor constants (lines 220-240):
- `QColor("#3a3a3a")` → `colors.Q_SURFACE_2`
- `QColor("#5a9bd4")` → `colors.Q_ACCENT`
- `QColor("#ffffff")` → `colors.Q_TEXT_PRIMARY`

- [ ] **Step 7b: Update subject_dialog.py**

Replace inline style at line 105 (`"font-size: 14px; padding: 8px 16px;"`) — remove it, QSS handles `QPushButton` sizing.

- [ ] **Step 8: Verify all dialog imports**

Run: `python -c "from glider.gui.dialogs.help_dialog import HelpDialog; from glider.gui.dialogs.analysis_dialog import AnalysisDialog; print('OK')"`
Expected: `OK`

- [ ] **Step 9: Lint check**

Run: `ruff check src/glider/gui/dialogs/`
Expected: All checks passed

- [ ] **Step 10: Commit**

```bash
git add src/glider/gui/dialogs/
git commit -m "feat: remove all dialog inline styles, use QSS + colors module"
```

---

### Task 17: Update touch_widgets.py — remove inline styles and QPainter colors

**Files:**
- Modify: `src/glider/gui/widgets/touch_widgets.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Replace all setStyleSheet calls**

This file has ~30 inline stylesheet calls. Replace each with either:
- `setProperty("textRole", "muted")` for text-only styles
- `setProperty("buttonRole", "secondary")` for secondary buttons
- Colors module references for any remaining dynamic styles

Key mappings:
- `#3498db` → `colors.ACCENT`
- `#2980b9` → `colors.ACCENT_HOVER`
- `#2ecc71` → `colors.SUCCESS`
- `#7f8c8d` → `colors.TEXT_MUTED`
- `#e74c3c` → `colors.ERROR`
- `#c0392b` → `colors.ERROR` (pressed)
- `#34495e` → `colors.SURFACE_2`
- `#27ae60` → `colors.SUCCESS` (pressed)
- `#888` → `colors.TEXT_MUTED`
- `#fff` → `colors.TEXT_PRIMARY`
- `#666` → `colors.TEXT_DISABLED`

- [ ] **Step 3: Replace all QColor() calls in paintEvent methods**

| Line | Old | New |
|------|-----|-----|
| 437 | `QColor("#34495e")` | `colors.Q_SURFACE_2` |
| 445 | `QColor("#3498db")` | `colors.Q_ACCENT` |
| 450 | `QColor("#fff")` | `colors.Q_TEXT_PRIMARY` |
| 485 | `QColor("#34495e")` (gauge) | `colors.Q_SURFACE_2` |
| 545 | `QColor("#1a1a2e")` (chart bg) | `colors.Q_SURFACE_1` |
| 548 | `QColor("#2d2d44")` (chart border) | `colors.Q_BORDER` |
| 572 | `QColor("#3498db")` (chart line) | `colors.Q_ACCENT` |
| 587-588 | LED colors | `colors.Q_LED_ON` / `colors.Q_LED_OFF` |
| 614-615 | LED string colors | `colors.LED_ON` / `colors.LED_OFF` |
| 636-637 | LED QColor | `colors.Q_LED_ON` / `colors.Q_LED_OFF` |

- [ ] **Step 4: Verify import**

Run: `python -c "from glider.gui.widgets.touch_widgets import TouchLabel; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/widgets/touch_widgets.py
git commit -m "feat: remove touch widget inline styles, use colors module"
```

---

### Task 18: Update device_card.py and multi_camera_preview.py

**Files:**
- Modify: `src/glider/gui/widgets/device_card.py`
- Modify: `src/glider/gui/widgets/multi_camera_preview.py`

- [ ] **Step 1: Update device_card.py**

Add `from glider.gui.styles import colors` and replace all hardcoded colors:

| Lines | Old | New |
|-------|-----|-----|
| 54-80 | State color variables | `colors.ACCENT`, `colors.BORDER`, `colors.SUCCESS`, `colors.TEXT_MUTED` |
| 123 | `"#27ae60"` / `"#666"` | `colors.SUCCESS` / `colors.TEXT_DISABLED` |
| 154-160 | Widget stylesheet | Remove — use `setProperty("deviceCard", True)` |
| 172-174 | Name label | `setProperty("textRole", "primary")` — remove stylesheet |
| 180-181 | Status label | `setProperty("textRole", "muted")` — remove stylesheet |

- [ ] **Step 2: Update multi_camera_preview.py**

Add `from glider.gui.styles import colors` and replace:

| Lines | Old | New |
|-------|-----|-----|
| 67-76 | Primary indicator `#3498db` | Use `colors.ACCENT` in stylesheet f-string |
| 82-91 | Recording indicator `#c0392b` | Use `colors.ERROR` |
| 103-108 | Preview bg `#0d0d1a` | Use `colors.CANVAS` |
| 114 | `"color: #888;"` | `setProperty("textRole", "muted")` |
| 124-141 | Tile borders `#3498db`/`#2d2d44` | `colors.ACCENT`/`colors.BORDER` |
| 236 | `"color: #666;"` | `setProperty("textRole", "disabled")` |

- [ ] **Step 3: Verify imports**

Run: `python -c "from glider.gui.widgets.device_card import DeviceCard; from glider.gui.widgets.multi_camera_preview import MultiCameraPreviewWidget; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/glider/gui/widgets/
git commit -m "feat: remove device card and camera preview inline styles"
```

---

### Task 19: Update runner/dashboard.py and controllers/device_control_controller.py

**Files:**
- Modify: `src/glider/gui/runner/dashboard.py`
- Modify: `src/glider/gui/controllers/device_control_controller.py`

- [ ] **Step 1: Update dashboard.py (line 111)**

Replace `setStyleSheet("color: #888; font-size: 16px;")` with:

```python
label.setProperty("textRole", "muted")
```

- [ ] **Step 2: Update device_control_controller.py (lines 168, 208)**

Add `from glider.gui.styles import colors` and replace:
- Line 168: `"background-color: #2d2d2d; border-radius: 4px; color: #00ff00;"` → `label.setProperty("inputValue", True)`
- Line 208: `"font-size: 11px; color: #888; padding: 2px;"` → `label.setProperty("textRole", "muted")`

- [ ] **Step 3: Verify imports**

Run: `python -c "from glider.gui.runner.dashboard import RunnerDashboard; from glider.gui.controllers.device_control_controller import DeviceControlController; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/glider/gui/runner/dashboard.py src/glider/gui/controllers/device_control_controller.py
git commit -m "feat: remove dashboard and controller inline styles"
```

---

### Task 20: Update main_window.py — inline styles + dock tab grouping

**Files:**
- Modify: `src/glider/gui/main_window.py`

- [ ] **Step 1: Add colors import**

```python
from glider.gui.styles import colors
```

- [ ] **Step 2: Replace inline styles (lines 703, 715)**

- Line 703 (`font-size: 24px; font-weight: bold; color: #f39c12;`): Replace `#f39c12` with `colors.WARNING`
- Line 715 (`color: #888;`): `setProperty("textRole", "muted")`

- [ ] **Step 3: Update dock tab grouping in `_setup_dock_widgets()`**

After all docks are added, update the `tabifyDockWidget` calls to group:
- Left: Node Library + Hardware + Device Control (tabified)
- Right: Properties + Camera (tabified)
- Node Library raised by default on left, Properties raised on right

The current code already tabifies hardware + device control (line 296). Add:

```python
# Tab group: left sidebar
self.tabifyDockWidget(self._node_library_dock, self._hardware_dock)
self.tabifyDockWidget(self._hardware_dock, self._control_dock)
self._node_library_dock.raise_()

# Tab group: right sidebar
self.tabifyDockWidget(self._properties_dock, self._camera_dock)
self._properties_dock.raise_()
```

Remove the existing `tabifyDockWidget` and `raise_()` calls (lines 296-297) to avoid duplication.

- [ ] **Step 4: Verify import**

Run: `python -c "from glider.gui.main_window import MainWindow; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/glider/gui/main_window.py
git commit -m "feat: update main window dock grouping and remove inline styles"
```

---

## Chunk 5: Verification

### Task 21: Full verification and cleanup

- [ ] **Step 1: Lint check**

Run: `ruff check src/glider/gui/`
Expected: All checks passed. Fix any issues.

- [ ] **Step 2: Format check**

Run: `ruff format --check src/glider/gui/`
Expected: All files clean. Run `ruff format src/glider/gui/` if needed.

- [ ] **Step 3: Import verification**

Run: `python -c "from glider.gui.main_window import MainWindow; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: All existing tests pass.

- [ ] **Step 5: Grep for remaining hardcoded colors**

Run: `rg '#[0-9a-fA-F]{3,8}' src/glider/gui/ --glob '*.py' --glob '!colors.py' --glob '!*.qss' -n`

Review output. Remaining hex colors should only be:
- In `colors.py` (expected)
- Behavior state colors that reference `colors.BEHAVIOR_*`
- Dynamic f-strings that reference `colors.*` constants
- Comments

Any raw hex color not referencing `colors.*` needs to be fixed.

- [ ] **Step 6: Visual smoke test**

Run: `glider --builder --debug`

Check:
- [ ] Window opens without crash
- [ ] Menu bar renders with Deep Navy colors
- [ ] Toolbar buttons visible and properly colored
- [ ] Node library panel shows categories with correct colors
- [ ] Drag a node onto canvas — node renders with gradient header
- [ ] Select a node — cyan border appears
- [ ] Ports and connections render with correct colors
- [ ] Camera panel checkboxes stack vertically
- [ ] Status bar shows correctly
- [ ] All text is readable (no invisible text)

- [ ] **Step 7: Commit any fixes**

```bash
git add -A
git commit -m "chore: final cleanup for Deep Navy theme"
```
