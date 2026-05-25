"""
Runner subpackage.

The user-facing runner UI lives in ``glider.gui.panels.runner_panel.RunnerPanel``
(constructed inline by ``MainWindow``). Earlier exploratory `RunnerDashboard`
and `WidgetFactory` classes were removed in the 1.0 release-prep pass — the
inline panel proved to be the canonical implementation. This module is kept as
a package boundary for future refactors that may want to extract the touch UI
back out of ``MainWindow``.
"""

__all__: list[str] = []
