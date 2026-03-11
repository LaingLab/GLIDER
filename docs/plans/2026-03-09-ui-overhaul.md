# UI Overhaul: "Lab Dark" Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the generic dark theme with a polished JetBrains/VS Code-inspired "Lab Dark" theme across all GUI components, fix text overflow issues in camera and device control panels.

**Architecture:** Create a centralized color constants module (`styles/colors.py`) so all inline styles and QColor references use named constants. Rewrite both QSS files with the new palette. Update all inline `setStyleSheet()` and `QColor()` calls across ~22 files to reference the constants.

**Tech Stack:** PyQt6, QSS (Qt Style Sheets), Python

---

### Task 1: Create Color Constants Module

**Files:**
- Create: `src/glider/gui/styles/colors.py`
- Modify: `src/glider/gui/styles/__init__.py`

**Step 1: Create the color constants module**

Create `src/glider/gui/styles/colors.py`:

```python
"""
Centralized color palette for the GLIDER "Lab Dark" theme.

All hardcoded colors across the GUI should reference these constants.
"""

from PyQt6.QtGui import QColor


# === Surfaces (layered depth, neutral grays) ===
BASE = "#1b1b1f"
SURFACE_1 = "#232328"
SURFACE_2 = "#2b2b31"
SURFACE_3 = "#333339"
BORDER = "#3c3c44"

# === Text ===
TEXT_PRIMARY = "#e8e8ed"
TEXT_SECONDARY = "#a0a0ab"
TEXT_MUTED = "#6b6b76"

# === Accent (teal) ===
ACCENT = "#2ba6a6"
ACCENT_HOVER = "#239090"
ACCENT_PRESSED = "#1d7a7a"

# === Status (semantic) ===
SUCCESS = "#3dab5a"
WARNING = "#d4a03c"
ERROR = "#d44040"
INFO = "#5b8ad4"

# === Node Categories ===
CAT_HARDWARE = "#2d6b3d"
CAT_LOGIC = "#2d5470"
CAT_INTERFACE = "#705a2d"
CAT_SCRIPT = "#5a2d70"
CAT_DEFAULT = "#444450"

# === Node Graph ===
NODE_BODY = "#2b2b31"
NODE_HEADER_TEXT = "#dcdce2"
NODE_PORT_LABEL = "#b0b0ba"
NODE_SELECTED = "#2ba6a6"
GRID_BG = "#1b1b1f"
GRID_MINOR = "#232328"
GRID_MAJOR = "#2b2b31"
PORT_DATA = "#6ab4e8"
PORT_EXEC = "#e8e8ed"
CONN_ACTIVE = "#3dab5a"
CONN_SELECTED = "#2ba6a6"

# === Special ===
RECORDING = "#d44040"
LED_ON = "#3dab5a"
LED_OFF = "#6b6b76"
INPUT_VALUE = "#3dab5a"  # Green readout text

# === QColor versions for QPainter usage ===
Q_BASE = QColor(BASE)
Q_SURFACE_1 = QColor(SURFACE_1)
Q_SURFACE_2 = QColor(SURFACE_2)
Q_SURFACE_3 = QColor(SURFACE_3)
Q_BORDER = QColor(BORDER)
Q_TEXT_PRIMARY = QColor(TEXT_PRIMARY)
Q_TEXT_SECONDARY = QColor(TEXT_SECONDARY)
Q_TEXT_MUTED = QColor(TEXT_MUTED)
Q_ACCENT = QColor(ACCENT)
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
Q_GRID_BG = QColor(GRID_BG)
Q_GRID_MINOR = QColor(GRID_MINOR)
Q_GRID_MAJOR = QColor(GRID_MAJOR)
Q_PORT_DATA = QColor(PORT_DATA)
Q_PORT_EXEC = QColor(PORT_EXEC)
Q_CONN_ACTIVE = QColor(CONN_ACTIVE)
Q_CONN_SELECTED = QColor(CONN_SELECTED)
Q_RECORDING = QColor(RECORDING)
Q_LED_ON = QColor(LED_ON)
Q_LED_OFF = QColor(LED_OFF)
Q_INPUT_VALUE = QColor(INPUT_VALUE)
```

**Step 2: Update styles `__init__.py` to export colors**

Add to `src/glider/gui/styles/__init__.py`:

```python
from glider.gui.styles import colors  # noqa: F401
```

**Step 3: Verify import**

Run: `python -c "from glider.gui.styles.colors import ACCENT, Q_ACCENT; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/glider/gui/styles/colors.py src/glider/gui/styles/__init__.py
git commit -m "feat: add centralized color constants module for Lab Dark theme"
```

---

### Task 2: Rewrite Desktop Stylesheet

**Files:**
- Rewrite: `src/glider/gui/styles/desktop.qss`

**Step 1: Replace desktop.qss with new theme**

The new stylesheet uses the Lab Dark palette. Key changes from the old stylesheet:
- Purple-tinted backgrounds (`#1a1a2e`, `#16162a`) → neutral grays (`#1b1b1f`, `#232328`)
- `#3498db` accent → `#2ba6a6` teal accent
- `#2d2d44` borders → `#3c3c44` borders
- GroupBox gets section-header style (no border box)
- Scrollbars: 8px wide, minimal
- Buttons: 30px min height, 4px radius
- Status indicators use semantic colors

Write the full QSS file (see design doc for all color mappings). Every selector from the old file must be present in the new file with updated colors.

**Step 2: Verify no syntax errors**

Run: `python -c "from glider.gui.styles import get_desktop_stylesheet; s = get_desktop_stylesheet(); print(f'{len(s)} chars loaded')"`
Expected: Prints char count

**Step 3: Commit**

```bash
git add src/glider/gui/styles/desktop.qss
git commit -m "feat: rewrite desktop stylesheet with Lab Dark theme"
```

---

### Task 3: Rewrite Touch Stylesheet

**Files:**
- Rewrite: `src/glider/gui/styles/touch.qss`

**Step 1: Replace touch.qss with new theme**

Same color palette as desktop. Keep all touch-specific sizing (80px min height, 48px handles, 20px scrollbars). Key changes:
- Same background/border/accent color updates as desktop
- Typography +2px across the board
- Touch-specific selectors preserved

**Step 2: Verify no syntax errors**

Run: `python -c "from glider.gui.styles import get_touch_stylesheet; s = get_touch_stylesheet(); print(f'{len(s)} chars loaded')"`
Expected: Prints char count

**Step 3: Commit**

```bash
git add src/glider/gui/styles/touch.qss
git commit -m "feat: rewrite touch stylesheet with Lab Dark theme"
```

---

### Task 4: Update Node Graph Colors

**Files:**
- Modify: `src/glider/gui/node_graph/node_item.py`
- Modify: `src/glider/gui/node_graph/graph_view.py`
- Modify: `src/glider/gui/node_graph/connection_item.py`
- Modify: `src/glider/gui/node_graph/port_item.py`

**Step 1: Update node_item.py**

Replace hardcoded QColor values with imports from `glider.gui.styles.colors`:
- `CATEGORY_COLORS`: Use `Q_CAT_HARDWARE`, `Q_CAT_LOGIC`, `Q_CAT_INTERFACE`, `Q_CAT_SCRIPT`, `Q_CAT_DEFAULT`
- `self._body_color`: Use `Q_NODE_BODY`
- `self._selected_border_color`: Use `Q_NODE_SELECTED`
- Header text color: Use `Q_NODE_HEADER_TEXT`
- Port label color: Use `Q_NODE_PORT_LABEL`

**Step 2: Update graph_view.py (NodeGraphScene)**

Replace grid colors:
- `self._grid_color`: Use `Q_GRID_MINOR`
- `self._grid_color_major`: Use `Q_GRID_MAJOR`
- `self._background_color`: Use `Q_GRID_BG`

**Step 3: Update connection_item.py**

Replace connection colors:
- `DATA_COLOR`: Use `Q_PORT_DATA`
- `EXEC_COLOR`: Use `Q_PORT_EXEC`
- `ACTIVE_COLOR`: Use `Q_CONN_ACTIVE`
- Selected color in `paint()`: Use `Q_CONN_SELECTED`

**Step 4: Update port_item.py**

Replace port colors:
- `PORT_COLORS[PortType.DATA]`: Use `Q_PORT_DATA`
- `PORT_COLORS[PortType.EXEC]`: Use `Q_PORT_EXEC`

**Step 5: Verify import**

Run: `python -c "from glider.gui.node_graph.graph_view import NodeGraphView; print('OK')"`
Expected: `OK`

**Step 6: Lint check**

Run: `ruff check src/glider/gui/node_graph/`
Expected: All checks passed

**Step 7: Commit**

```bash
git add src/glider/gui/node_graph/
git commit -m "feat: update node graph colors to Lab Dark palette"
```

---

### Task 5: Fix Camera Panel Layout

**Files:**
- Modify: `src/glider/gui/panels/camera_panel.py`

**Step 1: Fix CV checkbox overflow**

In `_setup_ui()`, replace the horizontal `cv_layout` section (lines ~366-384) with vertical stacking:

Change from:
```python
cv_layout = QHBoxLayout()
self._cv_enabled_cb = QCheckBox("Computer Vision")
self._cv_enabled_cb.setFixedWidth(150)
...
cv_layout.addWidget(self._cv_enabled_cb)
cv_layout.addWidget(self._overlay_cb)
cv_layout.addWidget(self._vision_cone_cb)
cv_layout.addStretch()
layout.addLayout(cv_layout)
```

To:
```python
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

Remove `setFixedWidth(150)` from the CV checkbox — no longer needed with vertical layout.

**Step 2: Update inline styles to use color constants**

Replace all hardcoded hex colors in `setStyleSheet()` calls with references to `glider.gui.styles.colors`:

- `#0d0d1a` → `colors.BASE`
- `#2d2d44` → `colors.BORDER`
- `#1a1a2e` → `colors.SURFACE_1`
- `#c0392b` → `colors.ERROR`
- `#888` → `colors.TEXT_SECONDARY`
- `#666` → `colors.TEXT_MUTED`

Use f-strings in setStyleSheet calls: `self._fps_label.setStyleSheet(f"color: {colors.TEXT_SECONDARY}; font-size: 11px;")`

**Step 3: Verify import**

Run: `python -c "from glider.gui.panels.camera_panel import CameraPanel; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/glider/gui/panels/camera_panel.py
git commit -m "fix: stack camera panel checkboxes vertically, update colors"
```

---

### Task 6: Fix Device Control Panel Layout

**Files:**
- Modify: `src/glider/gui/panels/device_control_panel.py`

**Step 1: Fix layout issues**

In `_setup_ui()`:
- Change `control_widget.setMinimumWidth(200)` → `control_widget.setMinimumWidth(240)`
- Add `self._device_status_label.setWordWrap(True)` after creating the status label
- Add `self._input_value_label.setWordWrap(True)` after creating the input value label

**Step 2: Update inline styles to use color constants**

Replace hardcoded colors:
- Input value label: `#2d2d2d` → `colors.SURFACE_2`, `#00ff00` → `colors.INPUT_VALUE`
- Status label: `#888` → `colors.TEXT_SECONDARY`

**Step 3: Verify import**

Run: `python -c "from glider.gui.panels.device_control_panel import DeviceControlPanel; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/glider/gui/panels/device_control_panel.py
git commit -m "fix: device control panel min width, word wrap, update colors"
```

---

### Task 7: Update Remaining Panel Inline Styles

**Files:**
- Modify: `src/glider/gui/panels/hardware_panel.py`
- Modify: `src/glider/gui/panels/runner_panel.py`
- Modify: `src/glider/gui/panels/node_library_panel.py`
- Modify: `src/glider/gui/panels/node_editor_controller.py`

**Step 1: Update hardware_panel.py inline styles**

Replace all hardcoded hex colors in `setStyleSheet()` calls:
- `#1a1a2e` → `colors.SURFACE_1`
- `#2d2d44` → `colors.BORDER`
- `#3498db` → `colors.ACCENT`
- `#2980b9` → `colors.ACCENT_HOVER`
- `#34495e` → `colors.SURFACE_3`
- `#888` → `colors.TEXT_SECONDARY`
- `#e74c3c` → `colors.ERROR`
- `#2ecc71` → `colors.SUCCESS`

**Step 2: Update runner_panel.py inline styles**

Replace all hardcoded hex colors:
- `#2d2d44` → `colors.SURFACE_2`
- `#3d3d5c` → `colors.SURFACE_3`
- `#4CAF50` → `colors.SUCCESS`
- `#27ae60` → `colors.SUCCESS`
- `#1a1a2e` → `colors.SURFACE_1`
- `#3498db` → `colors.ACCENT`

**Step 3: Update node_library_panel.py inline styles (if any)**

Replace any hardcoded colors with constants.

**Step 4: Update node_editor_controller.py inline styles (if any)**

Replace any hardcoded colors with constants.

**Step 5: Verify imports**

Run: `python -c "from glider.gui.panels import HardwarePanel, RunnerPanel, NodeLibraryPanel, NodeEditorController; print('OK')"`
Expected: `OK`

**Step 6: Lint check**

Run: `ruff check src/glider/gui/panels/`
Expected: All checks passed

**Step 7: Commit**

```bash
git add src/glider/gui/panels/
git commit -m "feat: update all panel inline styles to Lab Dark palette"
```

---

### Task 8: Update Dialog Inline Styles

**Files:**
- Modify: `src/glider/gui/dialogs/analysis_dialog.py`
- Modify: `src/glider/gui/dialogs/camera_settings_dialog.py`
- Modify: `src/glider/gui/dialogs/experiment_dialog.py`
- Modify: `src/glider/gui/dialogs/flow_function_dialog.py`
- Modify: `src/glider/gui/dialogs/agent_settings_dialog.py`
- Modify: `src/glider/gui/dialogs/help_dialog.py`

**Step 1: Update all dialog inline styles**

For each dialog file, replace hardcoded hex colors in `setStyleSheet()` and `QColor()` calls with imports from `glider.gui.styles.colors`.

Common replacements across dialogs:
- `#888`, `#aaa` → `colors.TEXT_SECONDARY`
- `#666` → `colors.TEXT_MUTED`
- `#1e1e1e`, `#1a1a2e` → `colors.BASE` or `colors.SURFACE_1`
- `#2d2d2d`, `#2d2d44` → `colors.SURFACE_2` or `colors.BORDER`
- `#444` → `colors.SURFACE_3`
- `#3498db`, `#5a9bd4` → `colors.ACCENT`
- `#4caf50`, `#2ecc71`, `#8f8`, `#afa` → `colors.SUCCESS`
- `#f44336`, `#e74c3c` → `colors.ERROR`
- `#00ff00` → `colors.INPUT_VALUE`
- `#ffa726`, `#f39c12` → `colors.WARNING`
- `#e0e0e0` → `colors.TEXT_PRIMARY`
- `#3a3a3a` → `colors.SURFACE_2`
- `#ffffff`, `#fff` → `colors.TEXT_PRIMARY`

**Step 2: Verify imports**

Run: `python -c "from glider.gui.dialogs.experiment_dialog import ExperimentDialog; print('OK')"`
Expected: `OK`

**Step 3: Lint check**

Run: `ruff check src/glider/gui/dialogs/`
Expected: All checks passed

**Step 4: Commit**

```bash
git add src/glider/gui/dialogs/
git commit -m "feat: update all dialog inline styles to Lab Dark palette"
```

---

### Task 9: Update Widget Inline Styles

**Files:**
- Modify: `src/glider/gui/widgets/touch_widgets.py`
- Modify: `src/glider/gui/widgets/device_card.py`
- Modify: `src/glider/gui/widgets/multi_camera_preview.py`

**Step 1: Update touch_widgets.py**

This file has the most inline styles. Replace all hardcoded colors in both `setStyleSheet()` and `QColor()` (in `paintEvent` methods):

Stylesheet replacements:
- `#888` → `colors.TEXT_SECONDARY`
- `#fff` → `colors.TEXT_PRIMARY`
- `#666` → `colors.TEXT_MUTED`
- `#3498db` → `colors.ACCENT`
- `#2980b9` → `colors.ACCENT_HOVER`
- `#2ecc71` → `colors.SUCCESS`
- `#27ae60` → `colors.SUCCESS` (pressed variant can darken)
- `#7f8c8d` → `colors.TEXT_MUTED`
- `#e74c3c` → `colors.ERROR`
- `#c0392b` → `colors.ERROR` (pressed variant)
- `#34495e` → `colors.SURFACE_3`

QPainter QColor replacements:
- `QColor("#34495e")` → `colors.Q_SURFACE_3`
- `QColor("#3498db")` → `colors.Q_ACCENT`
- `QColor("#fff")` → `colors.Q_TEXT_PRIMARY`
- `QColor("#1a1a2e")` → `colors.Q_SURFACE_1`
- `QColor("#2d2d44")` → `colors.Q_BORDER`
- `QColor("#2ecc71")` → `colors.Q_LED_ON`
- `QColor("#7f8c8d")` → `colors.Q_LED_OFF`

**Step 2: Update device_card.py**

Replace hardcoded colors with constants.

**Step 3: Update multi_camera_preview.py**

Replace `#888` and any other hardcoded colors.

**Step 4: Verify imports**

Run: `python -c "from glider.gui.widgets.touch_widgets import TouchLabel; print('OK')"`
Expected: `OK`

**Step 5: Lint check**

Run: `ruff check src/glider/gui/widgets/`
Expected: All checks passed

**Step 6: Commit**

```bash
git add src/glider/gui/widgets/
git commit -m "feat: update all widget inline styles to Lab Dark palette"
```

---

### Task 10: Update Main Window Inline Styles

**Files:**
- Modify: `src/glider/gui/main_window.py`

**Step 1: Update all inline styles**

Replace hardcoded colors:
- `#f39c12` → `colors.WARNING`
- `#888` → `colors.TEXT_SECONDARY`
- Any other hardcoded colors found in the main window

**Step 2: Verify import**

Run: `python -c "from glider.gui.main_window import MainWindow; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/glider/gui/main_window.py
git commit -m "feat: update main window inline styles to Lab Dark palette"
```

---

### Task 11: Final Verification

**Step 1: Full lint check**

Run: `ruff check src/glider/gui/`
Expected: All checks passed

**Step 2: Format check**

Run: `black --check src/glider/gui/`
Expected: All files clean (run `black src/glider/gui/` if needed)

**Step 3: Import check**

Run: `python -c "from glider.gui.main_window import MainWindow; print('OK')"`
Expected: `OK`

**Step 4: Run tests**

Run: `pytest tests/`
Expected: All tests pass

**Step 5: Grep for remaining hardcoded colors**

Run: `grep -rn '#[0-9a-fA-F]\{3,6\}' src/glider/gui/ --include='*.py' | grep -v colors.py | grep -v '__pycache__'`

Review output — any remaining hardcoded hex colors should be either:
- In comments
- Dynamic/user-specified colors (e.g., zone colors from camera)
- Intentionally not part of the theme

**Step 6: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final cleanup for Lab Dark theme"
```
