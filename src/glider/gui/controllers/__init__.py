"""
GUI controllers subpackage.

Originally intended to host UI logic extracted from ``MainWindow``. The two
prototype controllers (``HardwareTreeController``, ``DeviceControlController``)
were never wired up; the active equivalents live in
``glider.gui.panels.hardware_panel.HardwarePanel`` and
``glider.gui.panels.device_control_panel.DeviceControlPanel``. Both were
removed in the 1.0 release-prep pass.

This module is kept as a package boundary for the planned ``MainWindow``
decomposition (see ``code-review-laing.md`` Section 6 architecture notes).
"""

__all__: list[str] = []
