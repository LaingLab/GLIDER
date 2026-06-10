"""
In-app Analysis panel — embeds glider.analysis library plots in a Qt
widget for inspecting finished recordings without leaving the app.
"""

from glider.gui.panels.analysis.analysis_panel import AnalysisPanel
from glider.gui.panels.analysis.plot_widgets import MplCanvas

__all__ = ["AnalysisPanel", "MplCanvas"]
