# UI Overhaul Design: "Lab Dark"

## Design Language

Polished dark IDE theme inspired by JetBrains/VS Code, tailored for scientific instrumentation. Clean hierarchy, comfortable spacing, readable text, strategic accent colors for states and categories.

## Color System

### Surfaces (layered depth, neutral grays)
- `#1b1b1f` — Base (window background)
- `#232328` — Surface 1 (panels, docks)
- `#2b2b31` — Surface 2 (cards, group boxes, inputs)
- `#333339` — Surface 3 (hover states, raised elements)
- `#3c3c44` — Borders

### Text
- `#e8e8ed` — Primary
- `#a0a0ab` — Secondary (labels, descriptions)
- `#6b6b76` — Muted (placeholders, disabled)

### Accent (teal)
- `#2ba6a6` — Primary
- `#239090` — Hover
- `#1d7a7a` — Pressed

### Status (semantic)
- `#3dab5a` — Success/Ready/Running
- `#d4a03c` — Warning/Paused
- `#d44040` — Error/Emergency
- `#5b8ad4` — Info/Data

### Node Categories
- Hardware: `#2d6b3d`
- Logic/Flow: `#2d5470`
- Interface/Control: `#705a2d`
- Script: `#5a2d70`

## Typography

- Body: 12px regular
- Small: 11px regular
- Section: 12px semibold
- Title: 14px semibold
- Font: `-apple-system, "Segoe UI", "Roboto", sans-serif`
- Mono: `"SF Mono", "JetBrains Mono", "Cascadia Code", "Consolas", monospace`

## Layout Fixes

### Camera Panel
- Stack CV checkboxes vertically (currently overflow horizontally)

### Device Control Panel
- Increase min width from 200px to 240px
- Word wrap on status labels
- Increase GroupBox title padding

## Component Styles

### Dock Widgets
- Flat title bar, Surface 1 bg, 1px bottom border
- Active tab: 2px teal bottom line

### Inputs
- Surface 2 bg, 1px border, 7px padding, 28px min height
- Focus: accent border

### Buttons
- Primary: Accent bg, white text
- Secondary: Surface 3 bg, primary text
- Danger: Red bg (emergency only)
- Ghost: Transparent, accent text
- All: 4px radius, 30px min height

### Group Boxes
- Replace border box with section header + 1px rule
- Content indented 8px

### Scrollbars
- 8px wide, rounded handle, invisible track

### Status Bar
- Surface 1 bg, 1px top border, monospace state values

## Node Graph (Minimal Changes)
- Grid colors match new palette
- Node body: `#2b2b31`
- Selection border: teal `#2ba6a6` (was orange)
- Connections unchanged

## Touch Stylesheet
- Same palette as desktop
- Keep touch target sizes (80px min)
- Typography +2px across the board

## Files Changed
- `styles/desktop.qss` — Full rewrite
- `styles/touch.qss` — Full rewrite
- `panels/camera_panel.py` — Vertical CV checkboxes
- `panels/device_control_panel.py` — Min width, word wrap, spacing
- `node_graph/node_item.py` — Body + selection colors
- `node_graph/graph_view.py` — Grid colors
- `node_graph/connection_item.py` — Selection color
