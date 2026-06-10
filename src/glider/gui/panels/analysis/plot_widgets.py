"""
Reusable matplotlib-in-Qt widget for embedding plots in the GUI.

``MplCanvas`` wraps ``FigureCanvasQTAgg`` and the standard matplotlib
navigation toolbar so any plot from ``glider.analysis.plots`` drops
into a panel as a normal QWidget. Each canvas owns its own Figure;
``redraw`` after calling a plot helper to push the new content.

Matplotlib's Qt backend chooses Qt5 vs Qt6 automatically via qtpy,
which honors the same PyQt6 we already depend on. No explicit
backend selection is needed in this module.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QVBoxLayout, QWidget

# These imports must come from backend_qtagg specifically — that
# backend talks to qtpy, which respects whichever Qt binding is already
# imported (PyQt6 in our case). Importing matplotlib.pyplot first
# would risk picking up an unrelated default backend.
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402


class MplCanvas(QWidget):
    """A self-contained matplotlib plotting area.

    Use:

        canvas = MplCanvas(parent=self)
        ax = canvas.fresh_axes()
        plot_ethogram(intervals, ax=ax)
        canvas.redraw()

    ``fresh_axes`` clears the existing figure and adds a single subplot,
    which is what every plot helper expects.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        figsize: tuple[float, float] = (8.0, 4.0),
        dpi: int = 100,
        show_toolbar: bool = True,
    ):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, dpi=dpi, layout="tight")
        self.canvas = FigureCanvasQTAgg(self.fig)
        # Some compositing systems will only redraw if the canvas has a
        # non-zero minimum size; the toolbar provides one implicitly,
        # but we set a fallback for canvases instantiated without it.
        self.canvas.setMinimumSize(200, 100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if show_toolbar:
            self.toolbar: NavigationToolbar2QT | None = NavigationToolbar2QT(self.canvas, self)
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas)

    def fresh_axes(self):
        """Clear any existing axes and return a single new Axes ready
        for the next plot. Use this instead of clearing manually so
        plot helpers always render onto a clean figure."""
        self.fig.clear()
        return self.fig.add_subplot(111)

    def redraw(self) -> None:
        """Push current figure content to the on-screen widget."""
        self.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the figure and redraw — used to blank the canvas
        when no session is loaded."""
        self.fig.clear()
        self.redraw()
