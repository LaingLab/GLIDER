"""
GLIDER custom Qt widgets.

This subpackage hosted a touch-optimised widget kit (``TouchButton``,
``TouchSlider``, etc.) that was never wired into the active runner UI; the
widgets built directly into the dashboard panels (``glider.gui.dashboard.panels``)
won out in practice. The kit was removed in the 1.0 release-prep pass to
reduce drift between two parallel runner implementations.

Add new shared widgets here only when at least two panels need them.
"""

__all__: list[str] = []
