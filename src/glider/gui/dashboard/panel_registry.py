"""Single source of truth for the dashboard's interchangeable panels.

`PANEL_KEYS` fixes the stable identifiers persisted in the layout file and
referenced by the quadrant picker. `PANEL_NAMES` maps each to its human label.
Widget construction lives in the main window (it needs `core`/slots); this
module holds only the identity data so it stays UI-free and testable.
"""

from __future__ import annotations

PANEL_KEYS: tuple[str, ...] = (
    "run_control",
    "device_states",
    "camera",
    "manual_controls",
    "experiment_info",
)

PANEL_NAMES: dict[str, str] = {
    "run_control": "Run Control",
    "device_states": "Device States",
    "camera": "Camera",
    "manual_controls": "Manual Controls",
    "experiment_info": "Experiment Info",
}
